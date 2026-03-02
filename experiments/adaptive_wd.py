"""
Adaptive weight decay experiment for faster grokking.

Hypothesis (from evindor/AdderBoard): Carry circuits in addition require large
weights for step-function-like behavior. Constant weight decay fights this,
slowing grokking. Decreasing WD as training progresses should help.

Tests multiple WD schedules across seeds and configs:
  - constant WD = 0.01 (baseline, AdamW default)
  - constant WD = 0.001
  - constant WD = 0.0
  - cosine decay: WD from 0.01 → 0.0 over training
  - step decay: WD halves every 10K steps
  - adaptive: WD = 0.01 * (1 - eval_acc) — drops as model improves

Usage:
    python experiments/adaptive_wd.py --config 89p --seeds 0,1,2,3,4
    python experiments/adaptive_wd.py --config 122p --seeds 0,1,2 --steps 50000
"""

import argparse
import csv
import json
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from minimal10digittransformer.model.qwen3 import (
    Qwen3AdditionModel, VOCAB_SIZE, INPUT_LEN, OUTPUT_LEN, TOTAL_LEN,
    MAX_ADDEND, SUM_DIGITS,
)
from minimal10digittransformer.data.addition import (
    encode, expected_output, generate_batch, load_test_set,
)
from minimal10digittransformer.evaluation.metrics import evaluate


CONFIGS = {
    "89p": {
        "d_model": 3, "ff": 2, "n_heads": 1, "n_kv_heads": 1,
        "head_dim": 4, "rope_theta": 3.0,
        "tie_kv": True, "tie_qo": True,
    },
    "122p": {
        "d_model": 3, "ff": 3, "n_heads": 1, "n_kv_heads": 1,
        "head_dim": 4, "rope_theta": 3.0,
    },
    "101p": {
        "d_model": 3, "ff": 2, "n_heads": 1, "n_kv_heads": 1,
        "head_dim": 4, "rope_theta": 3.0,
        "tie_qo": True,
    },
    "86p": {
        "d_model": 3, "ff": 2, "n_heads": 1, "n_kv_heads": 1,
        "head_dim": 4, "rope_theta": 3.0,
        "tie_kv": True, "tie_qo": True,
        "share_block_norms": True,
    },
}

WD_SCHEDULES = {
    "const_0.01": {"type": "constant", "value": 0.01},
    "const_0.001": {"type": "constant", "value": 0.001},
    "const_0.0": {"type": "constant", "value": 0.0},
    "cosine": {"type": "cosine", "start": 0.01, "end": 0.0},
    "step_half": {"type": "step", "start": 0.01, "factor": 0.5, "interval": 10000},
    "adaptive": {"type": "adaptive", "base": 0.01},
    # Evindor's metric-triggered approach (the one that actually works)
    "metric_triggered": {
        "type": "metric_triggered", "base": 0.01, "factor": 0.1,
        "exact_thresh1": 0.01, "exact_thresh2": 0.05, "tok_thresh": 0.70,
    },
}


def get_wd(schedule: dict, step: int, total_steps: int,
           eval_acc: float = 0.0, digit_acc: float = 0.0) -> float:
    """Compute weight decay value for given schedule and state."""
    stype = schedule["type"]
    if stype == "constant":
        return schedule["value"]
    elif stype == "cosine":
        progress = step / max(total_steps, 1)
        return schedule["end"] + 0.5 * (schedule["start"] - schedule["end"]) * (1 + math.cos(math.pi * progress))
    elif stype == "step":
        n_drops = step // schedule["interval"]
        return schedule["start"] * (schedule["factor"] ** n_drops)
    elif stype == "adaptive":
        # WD proportional to (1 - accuracy): drops as model improves
        return schedule["base"] * max(0.0, 1.0 - eval_acc)
    elif stype == "metric_triggered":
        # Evindor's 2-stage approach: drop WD when grokking starts
        base = schedule["base"]
        factor = schedule["factor"]
        if eval_acc >= schedule["exact_thresh2"] and digit_acc >= schedule["tok_thresh"]:
            return base * factor * factor  # 0.01 * 0.01 = 0.0001
        if eval_acc >= schedule["exact_thresh1"] and digit_acc >= schedule["tok_thresh"]:
            return base * factor  # 0.01 * 0.1 = 0.001
        return base
    return 0.01


def train_step(model, full_seq, labels):
    logits = model(full_seq)
    shift_logits = logits[:, :-1, :].reshape(-1, VOCAB_SIZE)
    shift_labels = labels[:, 1:].reshape(-1)
    return F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)


def run_one(config_name: str, seed: int, wd_name: str, wd_schedule: dict,
            steps: int, batch_size: int, lr: float, eval_interval: int,
            test_pairs: list, device: torch.device) -> dict:
    """Run a single training experiment with specified WD schedule."""
    torch.manual_seed(seed)

    cfg = CONFIGS[config_name]
    model = Qwen3AdditionModel(
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_kv_heads=cfg["n_kv_heads"],
        head_dim=cfg["head_dim"],
        ff=cfg["ff"],
        rope_theta=cfg["rope_theta"],
        tie_kv=cfg.get("tie_kv", False),
        tie_qo=cfg.get("tie_qo", False),
        share_block_norms=cfg.get("share_block_norms", False),
        share_norms=cfg.get("share_norms", False),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())

    # Start with initial WD
    initial_wd = get_wd(wd_schedule, 0, steps, 0.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=initial_wd)

    out_dir = f"experiments/wd_{config_name}_{wd_name}_s{seed}"
    os.makedirs(out_dir, exist_ok=True)

    metrics_file = open(f"{out_dir}/metrics.csv", "w", newline="")
    metrics_writer = csv.writer(metrics_file)
    metrics_writer.writerow(["step", "loss", "lr", "wd", "exact_acc", "digit_acc", "elapsed"])

    best_acc = 0.0
    current_eval_acc = 0.0
    current_digit_acc = 0.0
    t0 = time.time()
    grok_step = None  # Step where acc first exceeds 90%

    for step in range(1, steps + 1):
        # Cosine LR schedule
        progress = step / max(steps, 1)
        current_lr = lr / 10 + 0.5 * (lr - lr / 10) * (1 + math.cos(math.pi * progress))

        # Adaptive WD
        current_wd = get_wd(wd_schedule, step, steps, current_eval_acc, current_digit_acc)

        for pg in optimizer.param_groups:
            pg["lr"] = current_lr
            pg["weight_decay"] = current_wd

        model.train()
        full_seq, labels = generate_batch(batch_size, device, 10)
        optimizer.zero_grad()
        loss = train_step(model, full_seq, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % eval_interval == 0:
            eval_pairs = test_pairs[:200] if test_pairs else None
            seq_acc, dig_acc = evaluate(model, device, n_samples=200, test_pairs=eval_pairs)
            current_eval_acc = seq_acc
            current_digit_acc = dig_acc
            elapsed = time.time() - t0

            if seq_acc > best_acc:
                best_acc = seq_acc

            if grok_step is None and seq_acc > 0.90:
                grok_step = step

            metrics_writer.writerow([
                step, f"{loss.item():.6f}", f"{current_lr:.6f}",
                f"{current_wd:.6f}", f"{seq_acc:.4f}", f"{dig_acc:.4f}",
                f"{elapsed:.1f}"
            ])
            metrics_file.flush()

            if step % (eval_interval * 5) == 0:
                print(f"  {config_name} {wd_name} s{seed}: step {step:6d} | "
                      f"acc {seq_acc:.4f} | wd {current_wd:.6f} | {elapsed:.0f}s")
                sys.stdout.flush()

    metrics_file.close()

    result = {
        "config": config_name,
        "wd_schedule": wd_name,
        "seed": seed,
        "n_params": n_params,
        "best_acc": best_acc,
        "grok_step": grok_step,
        "elapsed": time.time() - t0,
        "steps": steps,
    }

    with open(f"{out_dir}/results.json", "w") as f:
        json.dump(result, f, indent=2)

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="89p",
                        choices=list(CONFIGS.keys()))
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4",
                        help="Comma-separated seed list")
    parser.add_argument("--wd-schedules", type=str, default="all",
                        help="Comma-separated WD schedules, or 'all'")
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--eval-interval", type=int, default=1000)
    parser.add_argument("--test-set", type=str, default="data/test_10k.json")
    parser.add_argument("--max-parallel", type=int, default=4)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    device = torch.device(args.device)

    if args.wd_schedules == "all":
        wd_names = list(WD_SCHEDULES.keys())
    else:
        wd_names = [s.strip() for s in args.wd_schedules.split(",")]

    test_pairs = load_test_set(args.test_set) if args.test_set else None

    print(f"{'='*70}")
    print(f"ADAPTIVE WEIGHT DECAY EXPERIMENT")
    print(f"Config: {args.config}, Seeds: {seeds}, Steps: {args.steps}")
    print(f"WD schedules: {wd_names}")
    print(f"{'='*70}\n")
    sys.stdout.flush()

    # Run all combinations
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing

    n_cores = multiprocessing.cpu_count()
    max_par = args.max_parallel

    os.environ["OMP_NUM_THREADS"] = str(max(1, n_cores // max(max_par, 1)))

    jobs = []
    for wd_name in wd_names:
        for seed in seeds:
            jobs.append((args.config, seed, wd_name, WD_SCHEDULES[wd_name],
                         args.steps, args.batch_size, args.lr, args.eval_interval,
                         test_pairs, device))

    print(f"Total jobs: {len(jobs)}, max_parallel: {max_par}")
    sys.stdout.flush()

    all_results = []
    completed = 0
    total = len(jobs)

    # Run sequentially (ProcessPoolExecutor has issues with torch in subprocesses)
    # Instead, use subprocess approach for parallel execution
    for job_args in jobs:
        config_name, seed, wd_name, wd_sched, steps, bs, lr, ei, tp, dev = job_args
        print(f"\n  Starting: {config_name} {wd_name} s{seed}")
        sys.stdout.flush()

        r = run_one(config_name, seed, wd_name, wd_sched, steps, bs, lr, ei, tp, dev)
        all_results.append(r)
        completed += 1

        status = f"grok@{r['grok_step']}" if r['grok_step'] else f"best={r['best_acc']:.4f}"
        print(f"  [{completed}/{total}] {config_name} {wd_name} s{seed}: "
              f"{status} ({r['elapsed']:.0f}s)")
        sys.stdout.flush()

    # Summary
    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}")

    # Group by WD schedule
    by_wd = {}
    for r in all_results:
        by_wd.setdefault(r["wd_schedule"], []).append(r)

    for wd_name in wd_names:
        runs = by_wd.get(wd_name, [])
        accs = [r["best_acc"] for r in runs]
        grok_steps = [r["grok_step"] for r in runs if r["grok_step"] is not None]
        grok_rate = len(grok_steps) / len(runs) if runs else 0
        mean_grok = sum(grok_steps) / len(grok_steps) if grok_steps else float("inf")
        best = max(accs) if accs else 0
        mean = sum(accs) / len(accs) if accs else 0

        print(f"  {wd_name:>15s}: grok_rate={grok_rate:.0%} ({len(grok_steps)}/{len(runs)}), "
              f"mean_grok_step={mean_grok:,.0f}, best={best:.4f}, mean_acc={mean:.4f}")

    # Save all results
    out_path = f"experiments/adaptive_wd_{args.config}_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
