"""Windowed attention + shared-weight layers search.

Tests whether narrow attention windows (2-8 positions) combined with
multiple shared-weight layers (repeats=2-8) can solve 10-digit addition.

Intuition: addition is local (carry propagates one position at a time),
so a deep stack of local-attention layers might learn digit-by-digit
carry propagation.

Usage:
    # Full search (~192 runs)
    python experiments/windowed_search.py --max-steps 100000

    # Quick smoke test
    python experiments/windowed_search.py --max-steps 500 --seeds 1

    # Resume from previous run
    python experiments/windowed_search.py --max-steps 100000
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

from minimal10digittransformer.model.qwen3 import (
    Qwen3AdditionModel,
    VOCAB_SIZE,
)
from minimal10digittransformer.model.circular_arc import CircularArcQwen3
from minimal10digittransformer.data.addition import generate_batch, generate_test_set
from minimal10digittransformer.evaluation.metrics import evaluate


# ============================================================================
# Configuration space
# ============================================================================

def generate_configs():
    """Generate search configs: 4 architectures × 4 windows × 4 repeats."""
    configs = []

    # Base architectures (known to grok at 1 layer)
    bases = [
        ("arc_ff3", "arc", dict(
            d_model=3, n_heads=1, n_kv_heads=1, head_dim=4, ff=3, rope_theta=3.0,
        )),
        ("arc_ff3_tieQO_shnorm", "arc", dict(
            d_model=3, n_heads=1, n_kv_heads=1, head_dim=4, ff=3, rope_theta=3.0,
            tie_qo=True, share_norms=True,
        )),
        ("qwen3_ff3", "qwen3", dict(
            d_model=3, n_heads=1, n_kv_heads=1, head_dim=4, ff=3, rope_theta=3.0,
        )),
        ("arc_ff2_tieQO", "arc", dict(
            d_model=3, n_heads=1, n_kv_heads=1, head_dim=4, ff=2, rope_theta=3.0,
            tie_qo=True,
        )),
    ]

    window_sizes = [8, 0]  # 0 = full causal (baseline); skip 2,4 (RF too small)
    repeat_counts = [1, 2, 5, 8]

    for base_name, arch, base_kwargs in bases:
        for window in window_sizes:
            for rep in repeat_counts:
                name_parts = [base_name]
                if window > 0:
                    name_parts.append(f"w{window}")
                else:
                    name_parts.append("wfull")
                if rep > 1:
                    name_parts.append(f"rep{rep}")

                # Receptive field = 1 + rep * (window - 1); skip if < 35
                rf = 1 + rep * (window - 1) if window > 0 else 9999
                if rf < 35:
                    continue

                cfg = {
                    "name": "_".join(name_parts),
                    "arch": arch,
                    "window_size": window,
                    "repeats": rep,
                    **base_kwargs,
                }
                configs.append(cfg)

    return configs


def build_model(cfg):
    """Build model from config."""
    arch = cfg["arch"]
    kwargs = {k: v for k, v in cfg.items() if k not in ("name", "arch")}

    if arch == "qwen3":
        return Qwen3AdditionModel(**kwargs)
    elif arch == "arc":
        return CircularArcQwen3(**kwargs)
    raise ValueError(f"Unknown arch: {arch}")


def count_params(model):
    """Count unique parameters (handles tied weights)."""
    seen = set()
    total = 0
    for p in model.parameters():
        if p.data_ptr() not in seen:
            seen.add(p.data_ptr())
            total += p.numel()
    return total


# ============================================================================
# Training
# ============================================================================

def train_one(cfg, seed, max_steps, eval_interval, device, eval_pairs, ckpt_dir):
    """Train a single config/seed and return metrics."""
    random.seed(seed)
    torch.manual_seed(seed)

    model = build_model(cfg)
    model = model.to(device)
    n_params = count_params(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.01)

    best_acc = 0.0
    best_step = 0
    grok_step = None
    start_time = time.time()

    for step in range(1, max_steps + 1):
        # Cosine LR: 0.01 -> 0.001
        progress = step / max(max_steps, 1)
        cur_lr = 0.001 + 0.5 * (0.01 - 0.001) * (1 + math.cos(math.pi * progress))
        for pg in optimizer.param_groups:
            pg["lr"] = cur_lr

        model.train()
        batch, labels = generate_batch(128, device)
        logits = model(batch)
        # Shifted loss (next-token prediction)
        shift_logits = logits[:, :-1, :].reshape(-1, VOCAB_SIZE)
        shift_labels = labels[:, 1:].reshape(-1)
        loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % eval_interval == 0 or step == max_steps:
            model.eval()
            with torch.no_grad():
                acc, _ = evaluate(model, device, test_pairs=eval_pairs[:500])

            elapsed = time.time() - start_time
            print(f"  [{cfg['name']} s{seed}] step {step}/{max_steps} "
                  f"loss={loss.item():.4f} acc={acc:.1%} best={best_acc:.1%} "
                  f"[{elapsed:.0f}s]", flush=True)

            if acc > best_acc:
                best_acc = acc
                best_step = step
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

            if grok_step is None and acc > 0.5:
                grok_step = step

            # Early stop if perfect
            if acc >= 0.999:
                break

    elapsed = time.time() - start_time
    return {
        "config_name": cfg["name"],
        "arch": cfg["arch"],
        "window_size": cfg["window_size"],
        "repeats": cfg["repeats"],
        "n_params": n_params,
        "seed": seed,
        "best_exact_acc": best_acc,
        "best_step": best_step,
        "grok_step": grok_step or "",
        "final_loss": loss.item(),
        "final_step": step,
        "elapsed_s": elapsed,
    }


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Windowed attention search")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=100000)
    parser.add_argument("--eval-interval", type=int, default=5000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="experiments/windowed_search_results.csv")
    parser.add_argument("--ckpt-dir", default="checkpoints/windowed_search")
    args = parser.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", "4")

    configs = generate_configs()
    seed_list = [1, 8, 15][:args.seeds]
    eval_pairs = generate_test_set(2000, seed=12345)

    # Load completed runs
    completed = set()
    if os.path.exists(args.output):
        with open(args.output) as f:
            for row in csv.DictReader(f):
                completed.add((row["config_name"], int(row["seed"])))

    # Sort by param count (ascending), then by name
    configs_with_params = []
    for cfg in configs:
        m = build_model(cfg)
        n = count_params(m)
        configs_with_params.append((n, cfg))
        del m
    configs_with_params.sort(key=lambda x: (x[0], x[1]["name"]))

    total_runs = sum(1 for _, cfg in configs_with_params for s in seed_list
                     if (cfg["name"], s) not in completed)
    print(f"Windowed attention search: {len(configs_with_params)} configs × {len(seed_list)} seeds")
    print(f"Remaining: {total_runs} runs, {len(completed)} already done")
    print()

    # Print config summary
    for n, cfg in configs_with_params:
        w = f"w={cfg['window_size']}" if cfg['window_size'] > 0 else "w=full"
        r = f"rep={cfg['repeats']}"
        print(f"  {n:3d}p  {cfg['name']:45s}  {w:6s}  {r}")
    print()

    fieldnames = ["config_name", "arch", "window_size", "repeats", "n_params",
                  "seed", "best_exact_acc", "best_step", "grok_step",
                  "final_loss", "final_step", "elapsed_s"]

    write_header = not os.path.exists(args.output) or os.path.getsize(args.output) == 0
    done = 0

    for n_params, cfg in configs_with_params:
        for seed in seed_list:
            if (cfg["name"], seed) in completed:
                continue

            done += 1
            w = f"w={cfg['window_size']}" if cfg['window_size'] > 0 else "w=full"
            print(f"\n[{done}/{total_runs}] {cfg['name']} ({n_params}p) "
                  f"{w} rep={cfg['repeats']} seed={seed}")
            result = train_one(cfg, seed, args.max_steps, args.eval_interval,
                              args.device, eval_pairs, args.ckpt_dir)

            with open(args.output, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if write_header:
                    writer.writeheader()
                    write_header = False
                writer.writerow(result)

            grok = f"grok@{result['grok_step']}" if result['grok_step'] else "no grok"
            print(f"  -> {result['best_exact_acc']:.1%} (step {result['best_step']}) {grok}")

    print(f"\nDone. Results in {args.output}")


if __name__ == "__main__":
    main()
