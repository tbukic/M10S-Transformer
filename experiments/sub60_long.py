"""Long training for sub-60p configs — 800K steps phase 1.

These configs show loss signal (1.5-1.9) but didn't grok in 100K steps.
Give them 8x longer to see if they cross the grokking threshold.

Usage:
    python experiments/sub60_long.py --device cuda
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
from minimal10digittransformer.data.addition import generate_batch, generate_test_set
from minimal10digittransformer.evaluation.metrics import evaluate
from minimal10digittransformer.model.qwen3 import VOCAB_SIZE


CONFIGS = [
    {"name": "47p_ff1_tieKV_tieQO_shnorm",
     "d_model": 3, "ff": 1, "n_heads": 1, "n_kv_heads": 1, "head_dim": 4,
     "rope_theta": 3.0, "tie_kv": True, "tie_qo": True, "share_norms": True},
    {"name": "50p_ff1_tieKV_tieQO_shbnorm",
     "d_model": 3, "ff": 1, "n_heads": 1, "n_kv_heads": 1, "head_dim": 4,
     "rope_theta": 3.0, "tie_kv": True, "tie_qo": True, "share_block_norms": True},
    {"name": "53p_ff1_tieKV_tieQO",
     "d_model": 3, "ff": 1, "n_heads": 1, "n_kv_heads": 1, "head_dim": 4,
     "rope_theta": 3.0, "tie_kv": True, "tie_qo": True},
    {"name": "56p_ff2_tieKV_tieQO_shnorm",
     "d_model": 3, "ff": 2, "n_heads": 1, "n_kv_heads": 1, "head_dim": 4,
     "rope_theta": 3.0, "tie_kv": True, "tie_qo": True, "share_norms": True},
    {"name": "59p_ff1_tieQO_shnorm",
     "d_model": 3, "ff": 1, "n_heads": 1, "n_kv_heads": 1, "head_dim": 4,
     "rope_theta": 3.0, "tie_qo": True, "share_norms": True},
    {"name": "59p_ff1_tieKV_shnorm",
     "d_model": 3, "ff": 1, "n_heads": 1, "n_kv_heads": 1, "head_dim": 4,
     "rope_theta": 3.0, "tie_kv": True, "share_norms": True},
    {"name": "59p_ff2_tieKV_tieQO_shbnorm",
     "d_model": 3, "ff": 2, "n_heads": 1, "n_kv_heads": 1, "head_dim": 4,
     "rope_theta": 3.0, "tie_kv": True, "tie_qo": True, "share_block_norms": True},
]

SEEDS = [1, 8, 15, 22, 29]


def count_params(model):
    seen = set()
    total = 0
    for p in model.parameters():
        if p.data_ptr() not in seen:
            seen.add(p.data_ptr())
            total += p.numel()
    return total


def train_one(cfg, seed, max_steps, eval_interval, device, eval_pairs, ckpt_dir,
              metrics_dir=None):
    random.seed(seed)
    torch.manual_seed(seed)

    model_kwargs = {k: v for k, v in cfg.items() if k != "name"}
    model = CircularArcQwen3(**model_kwargs).to(device)
    n_params = count_params(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.01)

    best_acc = 0.0
    best_step = 0
    grok_step = None
    start_time = time.time()

    # Dense metrics log (loss every 1K steps, acc at eval_interval)
    metrics_path = None
    metrics_file = None
    if metrics_dir:
        os.makedirs(metrics_dir, exist_ok=True)
        metrics_path = os.path.join(metrics_dir, f"{cfg['name']}_s{seed}.csv")
        metrics_file = open(metrics_path, "w", newline="")
        mw = csv.writer(metrics_file)
        mw.writerow(["step", "loss", "lr", "exact_acc", "elapsed"])

    for step in range(1, max_steps + 1):
        # Cosine LR: 0.01 -> 0.001
        progress = step / max(max_steps, 1)
        cur_lr = 0.001 + 0.5 * (0.01 - 0.001) * (1 + math.cos(math.pi * progress))
        for pg in optimizer.param_groups:
            pg["lr"] = cur_lr

        model.train()
        batch, labels = generate_batch(128, device)
        logits = model(batch)
        shift_logits = logits[:, :-1, :].reshape(-1, VOCAB_SIZE)
        shift_labels = labels[:, 1:].reshape(-1)
        loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Log loss every 1K steps
        if step % 1000 == 0 and step % eval_interval != 0:
            elapsed = time.time() - start_time
            if metrics_file:
                mw.writerow([step, f"{loss.item():.6f}", f"{cur_lr:.6f}", "", f"{elapsed:.1f}"])
                metrics_file.flush()

        if step % eval_interval == 0 or step == max_steps:
            model.eval()
            with torch.no_grad():
                acc, _ = evaluate(model, device, test_pairs=eval_pairs[:500])

            elapsed = time.time() - start_time
            print(f"  [{cfg['name']} s{seed}] step {step}/{max_steps} "
                  f"loss={loss.item():.4f} acc={acc:.1%} best={best_acc:.1%} "
                  f"lr={cur_lr:.5f} [{elapsed:.0f}s]", flush=True)

            if metrics_file:
                mw.writerow([step, f"{loss.item():.6f}", f"{cur_lr:.6f}",
                             f"{acc:.6f}", f"{elapsed:.1f}"])
                metrics_file.flush()

            if acc > best_acc:
                best_acc = acc
                best_step = step

            # Save checkpoint every eval (rolling + best)
            if ckpt_dir:
                path = os.path.join(ckpt_dir, f"{cfg['name']}_s{seed}")
                os.makedirs(path, exist_ok=True)
                ckpt_data = {
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": cfg,
                    "step": step,
                    "acc": acc,
                    "loss": loss.item(),
                    "seed": seed,
                }
                # Rolling latest
                torch.save(ckpt_data, os.path.join(path, "latest.pt"))
                # Best
                if acc >= best_acc and acc > 0:
                    torch.save(ckpt_data, os.path.join(path, "best.pt"))

            if grok_step is None and acc > 0.5:
                grok_step = step

            if acc >= 0.999:
                break

    if metrics_file:
        metrics_file.close()

    elapsed = time.time() - start_time
    return {
        "config_name": cfg["name"],
        "n_params": n_params,
        "seed": seed,
        "best_exact_acc": best_acc,
        "best_step": best_step,
        "grok_step": grok_step or "",
        "final_loss": loss.item(),
        "final_step": step,
        "elapsed_s": elapsed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=800000)
    parser.add_argument("--eval-interval", type=int, default=10000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="experiments/sub60_long_results.csv")
    parser.add_argument("--ckpt-dir", default="checkpoints/sub60_long")
    parser.add_argument("--config", default=None, help="Run only this config name")
    parser.add_argument("--seed", type=int, default=None, help="Run only this seed")
    args = parser.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", "2")

    eval_pairs = generate_test_set(2000, seed=12345)

    # Filter configs/seeds if specified
    configs = CONFIGS
    seeds = SEEDS
    if args.config:
        configs = [c for c in CONFIGS if c["name"] == args.config]
        if not configs:
            print(f"Unknown config: {args.config}")
            print(f"Available: {[c['name'] for c in CONFIGS]}")
            return
    if args.seed is not None:
        seeds = [args.seed]

    # Load completed
    completed = set()
    if os.path.exists(args.output):
        with open(args.output) as f:
            for row in csv.DictReader(f):
                completed.add((row["config_name"], int(row["seed"])))

    total = sum(1 for c in configs for s in seeds if (c["name"], s) not in completed)
    print(f"Sub-60p long training: {len(configs)} configs × {len(seeds)} seeds")
    print(f"Remaining: {total}, {len(completed)} done")
    print(f"Steps: {args.max_steps}, device: {args.device}")
    print()

    fieldnames = ["config_name", "n_params", "seed", "best_exact_acc",
                  "best_step", "grok_step", "final_loss", "final_step", "elapsed_s"]

    write_header = not os.path.exists(args.output) or os.path.getsize(args.output) == 0
    done = 0

    for cfg in configs:
        for seed in seeds:
            if (cfg["name"], seed) in completed:
                continue

            done += 1
            print(f"\n[{done}/{total}] {cfg['name']} seed={seed}")
            result = train_one(cfg, seed, args.max_steps, args.eval_interval,
                              args.device, eval_pairs, args.ckpt_dir,
                              metrics_dir="experiments/sub60_long_metrics")

            with open(args.output, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                if write_header:
                    w.writeheader()
                    write_header = False
                w.writerow(result)

            grok = f"grok@{result['grok_step']}" if result['grok_step'] else "no grok"
            print(f"  -> {result['best_exact_acc']:.1%} (step {result['best_step']}) {grok}")

    print(f"\nDone. Results in {args.output}")


if __name__ == "__main__":
    main()
