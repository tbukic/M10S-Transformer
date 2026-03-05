"""Search d_model=2 circular arc configs.

Tests all tying/norm-sharing combinations with d_model=2, ff=2.
Saves checkpoints for promising runs (>10% accuracy).

Usage:
    python experiments/d2_search.py --seeds 5 --max-steps 100000
"""

import argparse
import csv
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import torch.nn as nn
import torch.nn.functional as F

from minimal10digittransformer.model.circular_arc import CircularArcQwen3
from minimal10digittransformer.data.addition import generate_batch
from minimal10digittransformer.evaluation.metrics import evaluate


def generate_configs():
    configs = []
    for tie_kv in [False, True]:
        for tie_qo in [False, True]:
            for sn, sbn in [(False, False), (False, True), (True, False)]:
                name_parts = ["arc_d2_ff2"]
                if tie_kv: name_parts.append("tieKV")
                if tie_qo: name_parts.append("tieQO")
                if sn: name_parts.append("shnorm")
                if sbn: name_parts.append("shbnorm")
                configs.append({
                    "name": "_".join(name_parts),
                    "d_model": 2, "ff": 2,
                    "n_heads": 1, "n_kv_heads": 1,
                    "head_dim": 4, "rope_theta": 3.0,
                    "tie_kv": tie_kv, "tie_qo": tie_qo,
                    "share_norms": sn, "share_block_norms": sbn,
                })
    return configs


def count_params(model):
    seen = set()
    total = 0
    for p in model.parameters():
        if p.data_ptr() not in seen:
            seen.add(p.data_ptr())
            total += p.numel()
    return total


def train_one(cfg, seed, max_steps, eval_interval, device, ckpt_dir):
    torch.manual_seed(seed)
    random.seed(seed)

    model = CircularArcQwen3(**{k: v for k, v in cfg.items() if k != "name"})
    model = model.to(device)
    n_params = count_params(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps, eta_min=0.001)

    best_acc = 0.0
    best_step = 0
    start_time = time.time()

    for step in range(1, max_steps + 1):
        model.train()
        batch, labels = generate_batch(128, device)
        logits = model(batch)
        shift_logits = logits[:, :-1, :].reshape(-1, logits.size(-1))
        shift_labels = labels[:, 1:].reshape(-1)
        loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % eval_interval == 0 or step == max_steps:
            model.eval()
            with torch.no_grad():
                acc, _ = evaluate(model, device, n_samples=500)

            elapsed = time.time() - start_time
            print(f"  [{cfg['name']} s{seed}] step {step}/{max_steps} "
                  f"loss={loss.item():.4f} acc={acc:.1%} best={best_acc:.1%} "
                  f"[{elapsed:.0f}s]", flush=True)

            if acc > best_acc:
                best_acc = acc
                best_step = step
                # Save checkpoint if promising
                if acc > 0.10 and ckpt_dir:
                    path = os.path.join(ckpt_dir, f"{cfg['name']}_s{seed}")
                    os.makedirs(path, exist_ok=True)
                    torch.save({
                        "state_dict": model.state_dict(),
                        "config": cfg,
                        "step": step,
                        "acc": acc,
                        "seed": seed,
                    }, os.path.join(path, "best.pt"))

    elapsed = time.time() - start_time
    final_loss = loss.item()

    return {
        "config_name": cfg["name"],
        "n_params": n_params,
        "seed": seed,
        "best_exact_acc": best_acc,
        "best_step": best_step,
        "final_loss": final_loss,
        "final_step": max_steps,
        "elapsed_s": elapsed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=100000)
    parser.add_argument("--eval-interval", type=int, default=5000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="experiments/d2_search_results.csv")
    parser.add_argument("--ckpt-dir", default="checkpoints/d2_search")
    parser.add_argument("--resume", default=None, help="Resume from existing CSV")
    args = parser.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", "4")

    configs = generate_configs()
    seed_list = [1, 8, 15, 22, 29][:args.seeds]

    # Load completed runs
    completed = set()
    if args.resume and os.path.exists(args.resume):
        with open(args.resume) as f:
            for row in csv.DictReader(f):
                completed.add((row["config_name"], int(row["seed"])))
    elif os.path.exists(args.output):
        with open(args.output) as f:
            for row in csv.DictReader(f):
                completed.add((row["config_name"], int(row["seed"])))

    # Sort configs by param count (smallest first — most interesting)
    configs_with_params = []
    for cfg in configs:
        m = CircularArcQwen3(**{k: v for k, v in cfg.items() if k != "name"})
        n = count_params(m)
        configs_with_params.append((n, cfg))
    configs_with_params.sort(key=lambda x: (x[0], x[1]["name"]))

    total_runs = sum(1 for _, cfg in configs_with_params for s in seed_list
                     if (cfg["name"], s) not in completed)
    print(f"d_model=2 search: {len(configs_with_params)} configs × {len(seed_list)} seeds")
    print(f"Remaining: {total_runs} runs, {len(completed)} already done")
    print()

    fieldnames = ["config_name", "n_params", "seed", "best_exact_acc",
                  "best_step", "final_loss", "final_step", "elapsed_s"]

    write_header = not os.path.exists(args.output) or os.path.getsize(args.output) == 0
    done = 0

    for n_params, cfg in configs_with_params:
        for seed in seed_list:
            if (cfg["name"], seed) in completed:
                continue

            done += 1
            print(f"\n[{done}/{total_runs}] {cfg['name']} ({n_params}p) seed={seed}")
            result = train_one(cfg, seed, args.max_steps, args.eval_interval,
                              args.device, args.ckpt_dir)

            with open(args.output, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if write_header:
                    writer.writeheader()
                    write_header = False
                writer.writerow(result)

            print(f"  → {result['best_exact_acc']:.1%} (step {result['best_step']})")

    print(f"\nDone. Results in {args.output}")


if __name__ == "__main__":
    main()
