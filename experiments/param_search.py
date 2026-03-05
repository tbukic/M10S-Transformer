"""Systematic parameter search for sub-62p models.

Sweeps all promising configurations below 62 parameters, testing multiple seeds
per config. Designed to run for 24+ hours unattended.

Key search dimensions:
- Architecture: Qwen3, CircularArc, CircularArc+Rank1Out
- ff: 1, 2, 3
- Weight tying: tieKV, tieQO, shnorm, shbnorm
- Seeds: 3-5 per config

Output: CSV with config, seed, param_count, best_acc, grok_step, etc.

Usage:
    # Full search (runs for ~24 hours)
    python experiments/param_search.py --device cpu --max-steps 100000

    # Quick test (runs ~1 hour)
    python experiments/param_search.py --device cpu --max-steps 20000 --seeds 2

    # Resume from a previous run (skip completed configs)
    python experiments/param_search.py --device cpu --resume results.csv
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
    INPUT_LEN,
    OUTPUT_LEN,
    TOTAL_LEN,
)
from minimal10digittransformer.model.circular_arc import CircularArcQwen3
from minimal10digittransformer.model.rank1_out import Rank1OutModel
from minimal10digittransformer.data.addition import generate_batch, generate_test_set
from minimal10digittransformer.evaluation.metrics import evaluate


# ============================================================================
# Configuration space
# ============================================================================

def generate_configs():
    """Generate all configurations to search."""
    configs = []

    # --- CircularArc models (most promising for low params) ---
    for ff in [1, 2, 3]:
        for tie_kv in [False, True]:
            for tie_qo in [False, True]:
                for norm_mode in ["none", "shbnorm", "shnorm"]:
                    share_norms = norm_mode == "shnorm"
                    share_block_norms = norm_mode == "shbnorm"

                    cfg = {
                        "arch": "circular_arc",
                        "d_model": 3, "ff": ff,
                        "n_heads": 1, "n_kv_heads": 1,
                        "head_dim": 4, "rope_theta": 3.0,
                        "tie_kv": tie_kv, "tie_qo": tie_qo,
                        "share_norms": share_norms,
                        "share_block_norms": share_block_norms,
                    }
                    configs.append(cfg)

    # --- Rank1Out models (only with tieKV, NOT tieQO) ---
    for ff in [1, 2, 3]:
        for tie_kv in [False, True]:
            for norm_mode in ["none", "shbnorm", "shnorm"]:
                share_norms = norm_mode == "shnorm"
                share_block_norms = norm_mode == "shbnorm"

                cfg = {
                    "arch": "rank1_out",
                    "d_model": 3, "ff": ff,
                    "n_heads": 1, "n_kv_heads": 1,
                    "head_dim": 4, "rope_theta": 3.0,
                    "tie_kv": tie_kv,
                    "share_norms": share_norms,
                    "share_block_norms": share_block_norms,
                }
                configs.append(cfg)

    # --- Standard Qwen3 ff=1 models (reference) ---
    for tie_kv in [False, True]:
        for tie_qo in [False, True]:
            for norm_mode in ["none", "shbnorm", "shnorm"]:
                share_norms = norm_mode == "shnorm"
                share_block_norms = norm_mode == "shbnorm"

                cfg = {
                    "arch": "qwen3",
                    "d_model": 3, "ff": 1,
                    "n_heads": 1, "n_kv_heads": 1,
                    "head_dim": 4, "rope_theta": 3.0,
                    "tie_kv": tie_kv, "tie_qo": tie_qo,
                    "share_norms": share_norms,
                    "share_block_norms": share_block_norms,
                }
                configs.append(cfg)

    return configs


def config_name(cfg):
    """Generate a human-readable name for a config."""
    parts = [cfg["arch"], f"ff{cfg['ff']}"]
    if cfg.get("tie_kv"):
        parts.append("tieKV")
    if cfg.get("tie_qo"):
        parts.append("tieQO")
    if cfg.get("share_norms"):
        parts.append("shnorm")
    if cfg.get("share_block_norms"):
        parts.append("shbnorm")
    return "_".join(parts)


def build_model(cfg, device):
    """Build a model from config dict."""
    arch = cfg["arch"]
    kwargs = {
        "d_model": cfg["d_model"],
        "n_heads": cfg["n_heads"],
        "n_kv_heads": cfg["n_kv_heads"],
        "head_dim": cfg["head_dim"],
        "ff": cfg["ff"],
        "rope_theta": cfg["rope_theta"],
    }

    if arch == "qwen3":
        kwargs.update({
            "tie_kv": cfg.get("tie_kv", False),
            "tie_qo": cfg.get("tie_qo", False),
            "share_norms": cfg.get("share_norms", False),
            "share_block_norms": cfg.get("share_block_norms", False),
        })
        model = Qwen3AdditionModel(**kwargs)
    elif arch == "circular_arc":
        kwargs.update({
            "tie_kv": cfg.get("tie_kv", False),
            "tie_qo": cfg.get("tie_qo", False),
            "share_norms": cfg.get("share_norms", False),
            "share_block_norms": cfg.get("share_block_norms", False),
        })
        model = CircularArcQwen3(**kwargs)
    elif arch == "rank1_out":
        kwargs.update({
            "tie_kv": cfg.get("tie_kv", False),
            "share_norms": cfg.get("share_norms", False),
            "share_block_norms": cfg.get("share_block_norms", False),
        })
        model = Rank1OutModel(**kwargs)
    else:
        raise ValueError(f"Unknown arch: {arch}")

    return model.to(device)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# ============================================================================
# Training
# ============================================================================

def train_and_evaluate(cfg, seed, device, max_steps, eval_pairs,
                       eval_interval=2000, batch_size=128):
    """Train a model and return metrics.

    Uses cosine LR from 0.01 to 0.001.
    Returns dict with best_acc, grok_step, final_loss, etc.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = build_model(cfg, device)
    n_params = count_params(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.01)

    best_acc = 0.0
    best_dig = 0.0
    grok_step = None  # Step where acc first > 0.5
    final_loss = None

    t0 = time.time()

    for step in range(1, max_steps + 1):
        # Cosine LR: 0.01 -> 0.001
        progress = step / max(max_steps, 1)
        cur_lr = 0.001 + 0.5 * (0.01 - 0.001) * (1 + math.cos(math.pi * progress))
        for pg in optimizer.param_groups:
            pg["lr"] = cur_lr

        model.train()
        full_seq, labels = generate_batch(batch_size, device, max_digits=10)
        optimizer.zero_grad()
        logits = model(full_seq)
        shift_logits = logits[:, :-1, :].reshape(-1, VOCAB_SIZE)
        shift_labels = labels[:, 1:].reshape(-1)
        loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        final_loss = loss.item()

        if step % eval_interval == 0:
            model.eval()
            with torch.no_grad():
                seq_acc, dig_acc = evaluate(model, device, test_pairs=eval_pairs[:500])

            elapsed = time.time() - t0
            print(f"  [{config_name(cfg)} s{seed}] step {step}/{max_steps} "
                  f"loss={final_loss:.4f} acc={seq_acc:.1%} best={best_acc:.1%} "
                  f"lr={cur_lr:.6f} [{elapsed:.0f}s]", flush=True)

            if seq_acc > best_acc:
                best_acc = seq_acc
            if dig_acc > best_dig:
                best_dig = dig_acc

            if grok_step is None and seq_acc > 0.5:
                grok_step = step

            # Early termination if already perfect
            if seq_acc >= 0.999:
                break

    elapsed = time.time() - t0

    return {
        "n_params": n_params,
        "best_exact_acc": best_acc,
        "best_digit_acc": best_dig,
        "grok_step": grok_step,
        "final_loss": final_loss,
        "final_step": step,
        "elapsed_s": elapsed,
    }


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Parameter search for sub-62p models")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max-steps", type=int, default=100000,
                        help="Max training steps per config/seed (default: 100K)")
    parser.add_argument("--seeds", type=int, default=3,
                        help="Number of seeds per config (default: 3)")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-interval", type=int, default=2000)
    parser.add_argument("--output", type=str, default="experiments/param_search_results.csv",
                        help="Output CSV path")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from existing CSV (skip completed entries)")
    parser.add_argument("--max-params", type=int, default=100,
                        help="Only test configs with <= this many params (default: 100)")
    parser.add_argument("--min-params", type=int, default=30,
                        help="Only test configs with >= this many params (default: 30)")
    args = parser.parse_args()

    device = torch.device(args.device)
    eval_pairs = generate_test_set(2000, seed=12345)

    # Generate all configs
    all_configs = generate_configs()
    print(f"Generated {len(all_configs)} total configurations")

    # Filter by param count
    filtered = []
    for cfg in all_configs:
        try:
            model = build_model(cfg, torch.device("cpu"))
            n = count_params(model)
            del model
            if args.min_params <= n <= args.max_params:
                filtered.append((cfg, n))
        except Exception as e:
            pass  # Skip invalid configs

    # Sort by param count (ascending)
    filtered.sort(key=lambda x: x[1])
    print(f"Filtered to {len(filtered)} configs with {args.min_params}-{args.max_params} params")

    # Load completed entries from resume file
    completed = set()
    if args.resume and os.path.exists(args.resume):
        with open(args.resume) as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row["config_name"], row["seed"])
                completed.add(key)
        print(f"Resuming: {len(completed)} entries already completed")

    # Setup output CSV
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    write_header = not os.path.exists(args.output) or not args.resume
    csv_file = open(args.output, "a" if args.resume else "w", newline="")
    writer = csv.writer(csv_file)
    if write_header:
        writer.writerow([
            "config_name", "arch", "ff", "tie_kv", "tie_qo",
            "share_norms", "share_block_norms", "n_params",
            "seed", "max_steps", "best_exact_acc", "best_digit_acc",
            "grok_step", "final_loss", "final_step", "elapsed_s",
        ])
        csv_file.flush()

    # Print search plan
    total_runs = len(filtered) * args.seeds
    print(f"\nSearch plan: {len(filtered)} configs x {args.seeds} seeds = {total_runs} runs")
    print(f"Max steps per run: {args.max_steps}")
    est_time = total_runs * args.max_steps * 0.005 / 3600  # ~5ms per step on CPU
    print(f"Estimated time: {est_time:.1f} hours")
    print(f"Output: {args.output}")
    print()

    # Run search
    run_count = 0
    for cfg, expected_params in filtered:
        name = config_name(cfg)

        for seed_idx in range(args.seeds):
            seed = seed_idx * 7 + 1  # Seeds: 1, 8, 15, 22, 29

            if (name, str(seed)) in completed:
                print(f"  SKIP {name} seed={seed} (already done)")
                continue

            run_count += 1
            print(f"[{run_count}/{total_runs}] {name} ({expected_params}p) seed={seed}")
            sys.stdout.flush()

            try:
                result = train_and_evaluate(
                    cfg, seed, device, args.max_steps, eval_pairs,
                    eval_interval=args.eval_interval,
                    batch_size=args.batch_size,
                )

                writer.writerow([
                    name, cfg["arch"], cfg["ff"],
                    cfg.get("tie_kv", False), cfg.get("tie_qo", False),
                    cfg.get("share_norms", False), cfg.get("share_block_norms", False),
                    result["n_params"], seed, args.max_steps,
                    f"{result['best_exact_acc']:.6f}",
                    f"{result['best_digit_acc']:.6f}",
                    result["grok_step"] or "",
                    f"{result['final_loss']:.6f}",
                    result["final_step"],
                    f"{result['elapsed_s']:.1f}",
                ])
                csv_file.flush()

                grok_str = f"grok@{result['grok_step']}" if result["grok_step"] else "no grok"
                print(f"  -> {result['n_params']}p | best={result['best_exact_acc']:.4f} "
                      f"| {grok_str} | {result['elapsed_s']:.0f}s")

            except Exception as e:
                print(f"  ERROR: {e}")
                writer.writerow([
                    name, cfg["arch"], cfg["ff"],
                    cfg.get("tie_kv", False), cfg.get("tie_qo", False),
                    cfg.get("share_norms", False), cfg.get("share_block_norms", False),
                    expected_params, seed, args.max_steps,
                    "", "", "", "", "", f"ERROR: {e}",
                ])
                csv_file.flush()

            sys.stdout.flush()

    csv_file.close()
    print(f"\nSearch complete! Results: {args.output}")
    print(f"Total runs: {run_count}")


if __name__ == "__main__":
    main()
