"""Reproduction pipeline for 62p CircularArcQwen3 model.

Self-contained script for the 4-stage pipeline:
  Stage 0: AdamW cosine lr=0.01, 200K steps, batch=128, seed=42
  Stage 1: AdamW const lr=0.001, 50K steps, batch=256, EMA=0.999
  Stage 2: AdamW const lr=0.0003, 30K steps, batch=256, seed=101, EMA=0.999
  Stage 3: Adam (NO weight decay) cosine lr=0.001, 50K steps, batch=256

Key insight: AdamW is needed for initial training (stages 0-2), but Adam-no-wd
is critical for the final push (stage 3). Weight decay actively prevents the
62-param model from learning the last ~1% of cases.

Usage:
    python experiments/reproduce_62p.py --seed 42 --device cuda
    python experiments/reproduce_62p.py --seed 42 --steps-override 200  # smoke test
"""

import argparse
import csv
import json
import math
import os
import random
import sys
import time

# Ensure src/ is on the import path (for models not in the installed package)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import torch.nn as nn
import torch.nn.functional as F

from minimal10digittransformer.model.circular_arc import CircularArcQwen3
from minimal10digittransformer.model.qwen3 import (
    VOCAB_SIZE,
    INPUT_LEN,
    OUTPUT_LEN,
    TOTAL_LEN,
)
from minimal10digittransformer.data.addition import (
    encode,
    expected_output,
    generate_batch,
    generate_test_set,
    load_test_set,
)
from minimal10digittransformer.evaluation.metrics import evaluate, evaluate_detailed


# ============================================================================
# Training utilities
# ============================================================================

RESERVED_SEEDS = {42, 2025, 123, 99}


def count_params(model: nn.Module) -> int:
    """Count unique trainable parameters."""
    return sum(p.numel() for p in model.parameters())


def train_step(model: nn.Module, full_seq: torch.Tensor,
               labels: torch.Tensor) -> torch.Tensor:
    """Single forward/backward step with teacher forcing."""
    logits = model(full_seq)
    shift_logits = logits[:, :-1, :].reshape(-1, VOCAB_SIZE)
    shift_labels = labels[:, 1:].reshape(-1)
    return F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)


def save_checkpoint(model: nn.Module, path: str, step: int,
                    accuracy: float, n_params: int,
                    extra: dict | None = None):
    """Save a checkpoint with metadata."""
    data = {
        "state_dict": model.state_dict(),
        "step": step,
        "accuracy": accuracy,
        "n_params": n_params,
    }
    if extra:
        data.update(extra)
    torch.save(data, path)


def load_checkpoint(model: nn.Module, path: str) -> dict:
    """Load checkpoint into model, return checkpoint dict."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["state_dict"])
    return ckpt


def update_ema(model: nn.Module, ema_params: list, decay: float):
    """Update EMA parameters."""
    for p, p_ema in zip(model.parameters(), ema_params):
        p_ema.data.mul_(decay).add_(p.data, alpha=1 - decay)


def apply_ema(model: nn.Module, ema_params: list):
    """Apply EMA parameters to model (for evaluation)."""
    for p, p_ema in zip(model.parameters(), ema_params):
        p.data.copy_(p_ema.data)


def save_ema_checkpoint(model: nn.Module, ema_params: list, path: str,
                        step: int, accuracy: float, n_params: int):
    """Save an EMA checkpoint by temporarily swapping weights."""
    # Save current weights
    original_params = [p.data.clone() for p in model.parameters()]
    # Apply EMA
    apply_ema(model, ema_params)
    # Save
    save_checkpoint(model, path, step, accuracy, n_params)
    # Restore original weights
    for p, orig in zip(model.parameters(), original_params):
        p.data.copy_(orig)


# ============================================================================
# Metrics CSV helper
# ============================================================================

class MetricsWriter:
    """Wrapper around csv.writer that tracks the file handle for flushing."""

    def __init__(self, path: str):
        self.file = open(path, "w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow([
            "stage", "global_step", "stage_step", "loss", "lr",
            "exact_acc", "digit_acc", "elapsed",
        ])

    def writerow(self, row):
        self.writer.writerow(row)

    def flush_file(self):
        self.file.flush()

    def close(self):
        self.file.close()


# ============================================================================
# Stage runners
# ============================================================================

def run_stage_cosine(model: nn.Module, device: torch.device,
                     lr: float, batch_size: int, steps: int,
                     eval_pairs: list, ckpt_dir: str, stage_num: int,
                     n_params: int, metrics: MetricsWriter, t0: float,
                     global_step: int, weight_decay: float = 0.01,
                     min_lr_ratio: float = 0.1,
                     optimizer_cls=None, ema_decay: float = 0.0,
                     eval_interval: int = 2000,
                     ) -> tuple[float, int]:
    """Run a cosine LR decay training stage.

    Args:
        min_lr_ratio: min_lr = lr * min_lr_ratio (default 0.1 for AdamW,
                      set to 0.0 for Adam-no-wd stage)
        optimizer_cls: override optimizer class (default: AdamW)
        ema_decay: if > 0, maintain EMA with this decay rate

    Returns (best_accuracy, updated_global_step).
    """
    if optimizer_cls is None:
        optimizer_cls = torch.optim.AdamW
    min_lr = lr * min_lr_ratio

    optimizer = optimizer_cls(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_acc = 0.0

    # EMA setup
    ema_params = None
    best_ema_acc = 0.0
    if ema_decay > 0:
        ema_params = [p.data.clone() for p in model.parameters()]

    print(f"\n  Stage {stage_num} [cosine]: lr={lr}, min_lr={min_lr}, "
          f"batch={batch_size}, steps={steps}, wd={weight_decay}")
    if ema_decay > 0:
        print(f"    EMA decay={ema_decay}")

    for step in range(1, steps + 1):
        # Cosine LR schedule
        progress = step / max(steps, 1)
        cur_lr = min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * progress))
        for pg in optimizer.param_groups:
            pg["lr"] = cur_lr

        model.train()
        full_seq, labels = generate_batch(batch_size, device, max_digits=10)

        optimizer.zero_grad()
        loss = train_step(model, full_seq, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        global_step += 1

        if ema_params is not None:
            update_ema(model, ema_params, ema_decay)

        if step % 100 == 0:
            elapsed = time.time() - t0
            print(f"    step {step:6d}/{steps} | loss {loss.item():.4f} | "
                  f"lr {cur_lr:.2e} | {elapsed:.0f}s")
            sys.stdout.flush()
            metrics.writerow([
                stage_num, global_step, step, f"{loss.item():.6f}",
                f"{cur_lr:.2e}", "", "", f"{elapsed:.1f}",
            ])
            if step % 1000 == 0:
                metrics.flush_file()

        if step % eval_interval == 0:
            quick_pairs = eval_pairs[:500]
            seq_acc, dig_acc = evaluate(model, device, test_pairs=quick_pairs)
            elapsed = time.time() - t0
            print(f"    EVAL step {step}: exact={seq_acc:.4f} "
                  f"digit={dig_acc:.4f} [{elapsed:.0f}s]")
            sys.stdout.flush()

            metrics.writerow([
                stage_num, global_step, step, f"{loss.item():.6f}",
                f"{cur_lr:.2e}", f"{seq_acc:.4f}", f"{dig_acc:.4f}",
                f"{elapsed:.1f}",
            ])
            metrics.flush_file()

            if seq_acc > best_acc:
                best_acc = seq_acc
                save_checkpoint(
                    model, f"{ckpt_dir}/stage{stage_num}_best.pt",
                    global_step, seq_acc, n_params,
                )
                print(f"    ** NEW BEST: {seq_acc:.4f} **")

            # EMA eval
            if ema_params is not None:
                original_params = [p.data.clone() for p in model.parameters()]
                apply_ema(model, ema_params)
                ema_acc, ema_dig = evaluate(model, device, test_pairs=quick_pairs)
                # Restore
                for p, orig in zip(model.parameters(), original_params):
                    p.data.copy_(orig)

                print(f"    EMA  step {step}: exact={ema_acc:.4f} "
                      f"digit={ema_dig:.4f}")
                if ema_acc > best_ema_acc:
                    best_ema_acc = ema_acc
                    save_ema_checkpoint(
                        model, ema_params,
                        f"{ckpt_dir}/stage{stage_num}_best_ema.pt",
                        global_step, ema_acc, n_params,
                    )
                    print(f"    ** NEW BEST EMA: {ema_acc:.4f} **")

    # Save end-of-stage checkpoints
    save_checkpoint(
        model, f"{ckpt_dir}/stage{stage_num}_final.pt",
        global_step, best_acc, n_params,
    )
    if ema_params is not None:
        save_ema_checkpoint(
            model, ema_params,
            f"{ckpt_dir}/stage{stage_num}_final_ema.pt",
            global_step, best_ema_acc, n_params,
        )

    return best_acc, global_step


def run_stage_constant(model: nn.Module, device: torch.device,
                       lr: float, batch_size: int, steps: int,
                       eval_pairs: list, ckpt_dir: str, stage_num: int,
                       n_params: int, metrics: MetricsWriter, t0: float,
                       global_step: int, weight_decay: float = 0.01,
                       ema_decay: float = 0.0,
                       eval_interval: int = 2000,
                       ) -> tuple[float, int]:
    """Run a constant LR training stage.

    Returns (best_accuracy, updated_global_step).
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=weight_decay)
    best_acc = 0.0

    # EMA setup
    ema_params = None
    best_ema_acc = 0.0
    if ema_decay > 0:
        ema_params = [p.data.clone() for p in model.parameters()]

    print(f"\n  Stage {stage_num} [constant]: lr={lr}, batch={batch_size}, "
          f"steps={steps}, wd={weight_decay}")
    if ema_decay > 0:
        print(f"    EMA decay={ema_decay}")

    for step in range(1, steps + 1):
        model.train()
        full_seq, labels = generate_batch(batch_size, device, max_digits=10)

        optimizer.zero_grad()
        loss = train_step(model, full_seq, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        global_step += 1

        if ema_params is not None:
            update_ema(model, ema_params, ema_decay)

        if step % 100 == 0:
            elapsed = time.time() - t0
            print(f"    step {step:6d}/{steps} | loss {loss.item():.4f} | "
                  f"lr {lr:.2e} | {elapsed:.0f}s")
            sys.stdout.flush()
            metrics.writerow([
                stage_num, global_step, step, f"{loss.item():.6f}",
                f"{lr:.2e}", "", "", f"{elapsed:.1f}",
            ])
            if step % 1000 == 0:
                metrics.flush_file()

        if step % eval_interval == 0:
            quick_pairs = eval_pairs[:500]
            seq_acc, dig_acc = evaluate(model, device, test_pairs=quick_pairs)
            elapsed = time.time() - t0
            print(f"    EVAL step {step}: exact={seq_acc:.4f} "
                  f"digit={dig_acc:.4f} [{elapsed:.0f}s]")
            sys.stdout.flush()

            metrics.writerow([
                stage_num, global_step, step, f"{loss.item():.6f}",
                f"{lr:.2e}", f"{seq_acc:.4f}", f"{dig_acc:.4f}",
                f"{elapsed:.1f}",
            ])
            metrics.flush_file()

            if seq_acc > best_acc:
                best_acc = seq_acc
                save_checkpoint(
                    model, f"{ckpt_dir}/stage{stage_num}_best.pt",
                    global_step, seq_acc, n_params,
                )
                print(f"    ** NEW BEST: {seq_acc:.4f} **")

            # EMA eval
            if ema_params is not None:
                original_params = [p.data.clone() for p in model.parameters()]
                apply_ema(model, ema_params)
                ema_acc, ema_dig = evaluate(model, device, test_pairs=quick_pairs)
                for p, orig in zip(model.parameters(), original_params):
                    p.data.copy_(orig)

                print(f"    EMA  step {step}: exact={ema_acc:.4f} "
                      f"digit={ema_dig:.4f}")
                if ema_acc > best_ema_acc:
                    best_ema_acc = ema_acc
                    save_ema_checkpoint(
                        model, ema_params,
                        f"{ckpt_dir}/stage{stage_num}_best_ema.pt",
                        global_step, ema_acc, n_params,
                    )
                    print(f"    ** NEW BEST EMA: {ema_acc:.4f} **")

    # Save end-of-stage checkpoints
    save_checkpoint(
        model, f"{ckpt_dir}/stage{stage_num}_final.pt",
        global_step, best_acc, n_params,
    )
    if ema_params is not None:
        save_ema_checkpoint(
            model, ema_params,
            f"{ckpt_dir}/stage{stage_num}_final_ema.pt",
            global_step, best_ema_acc, n_params,
        )

    return best_acc, global_step


# ============================================================================
# Validation
# ============================================================================

def run_validation(model: nn.Module, device: torch.device,
                   ckpt_dir: str) -> dict:
    """Run final validation matching verify.py protocol."""
    results = {}

    print(f"\n{'=' * 70}")
    print("Final Validation (simulating verify.py)")
    print(f"{'=' * 70}")

    edge_cases = [
        (0, 0), (0, 1), (9_999_999_999, 0), (9_999_999_999, 1),
        (9_999_999_999, 9_999_999_999), (5_000_000_000, 5_000_000_000),
        (1_111_111_111, 8_888_888_889), (1_234_567_890, 9_876_543_210),
        (9_999_999_999, 9_999_999_999), (1, 9_999_999_999),
    ]
    rng = random.Random(2025)
    random_cases = [
        (rng.randint(0, 9_999_999_999), rng.randint(0, 9_999_999_999))
        for _ in range(10000)
    ]
    verify_pairs = edge_cases + random_cases

    print(f"  verify-style: {len(verify_pairs)} pairs "
          f"(10 edge + 10000 random, seed=2025)")
    detailed = evaluate_detailed(model, device, verify_pairs)
    results["verify"] = {
        "n_samples": detailed["n_samples"],
        "exact_acc": detailed["exact_acc"],
        "n_errors": detailed["n_errors"],
        "digit_acc": detailed["digit_acc"],
        "qualified": detailed["exact_acc"] >= 0.99,
    }
    qualified_str = "QUALIFIED" if results["verify"]["qualified"] else "NOT QUALIFIED"
    print(f"  Result: {detailed['n_samples'] - detailed['n_errors']}"
          f"/{detailed['n_samples']} correct "
          f"({detailed['exact_acc'] * 100:.2f}%)")
    print(f"  Status: {qualified_str}")

    # Holdout 10K
    holdout_10k_path = "data/test_holdout_10k.json"
    if os.path.exists(holdout_10k_path):
        print(f"\n  Holdout 10K ({holdout_10k_path}):")
        holdout_10k = load_test_set(holdout_10k_path)
        h10k = evaluate_detailed(model, device, holdout_10k)
        results["holdout_10k"] = {
            "n_samples": h10k["n_samples"],
            "exact_acc": h10k["exact_acc"],
            "n_errors": h10k["n_errors"],
            "digit_acc": h10k["digit_acc"],
        }
        print(f"  Result: {h10k['n_samples'] - h10k['n_errors']}"
              f"/{h10k['n_samples']} correct "
              f"({h10k['exact_acc'] * 100:.2f}%)")

    # Holdout 50K
    holdout_50k_path = "data/test_50k_independent.json"
    if os.path.exists(holdout_50k_path):
        print(f"\n  Holdout 50K ({holdout_50k_path}):")
        holdout_50k = load_test_set(holdout_50k_path)
        h50k = evaluate_detailed(model, device, holdout_50k)
        results["holdout_50k"] = {
            "n_samples": h50k["n_samples"],
            "exact_acc": h50k["exact_acc"],
            "n_errors": h50k["n_errors"],
            "digit_acc": h50k["digit_acc"],
        }
        print(f"  Result: {h50k['n_samples'] - h50k['n_errors']}"
              f"/{h50k['n_samples']} correct "
              f"({h50k['exact_acc'] * 100:.2f}%)")

    val_path = os.path.join(ckpt_dir, "validation.json")
    with open(val_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Validation saved to {val_path}")

    return results


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Reproduction pipeline for 62p CircularArcQwen3.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python experiments/reproduce_62p.py --seed 42 --device cuda
  python experiments/reproduce_62p.py --seed 42 --steps-override 200  # smoke test
""",
    )
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for model init (default: 42)")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--steps-override", type=int, default=None,
                        help="Override step count for ALL stages (smoke test)")
    parser.add_argument("--no-validate", action="store_true",
                        help="Skip final validation")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: checkpoints/reproduce_62p_s{seed})")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Stage configurations
    stages = [
        {
            "name": "Stage 0: AdamW cosine",
            "type": "cosine", "lr": 0.01, "batch_size": 128,
            "steps": 200000, "weight_decay": 0.01,
            "min_lr_ratio": 0.1, "ema_decay": 0.0,
            "seed": args.seed, "eval_interval": 2000,
        },
        {
            "name": "Stage 1: AdamW constant + EMA",
            "type": "constant", "lr": 0.001, "batch_size": 256,
            "steps": 50000, "weight_decay": 0.01,
            "ema_decay": 0.999,
            "seed": args.seed, "eval_interval": 2000,
        },
        {
            "name": "Stage 2: AdamW constant + EMA (lower LR)",
            "type": "constant", "lr": 0.0003, "batch_size": 256,
            "steps": 30000, "weight_decay": 0.01,
            "ema_decay": 0.999,
            "seed": 101, "eval_interval": 2000,
        },
        {
            "name": "Stage 3: Adam NO weight decay, cosine",
            "type": "cosine", "lr": 0.001, "batch_size": 256,
            "steps": 50000, "weight_decay": 0.0,
            "min_lr_ratio": 0.0,  # decay to 0
            "optimizer_cls": "adam",
            "ema_decay": 0.0,
            "seed": args.seed, "eval_interval": 2000,
        },
    ]

    # Output directory
    ckpt_dir = args.output_dir or f"checkpoints/reproduce_62p_s{args.seed}"
    os.makedirs(ckpt_dir, exist_ok=True)

    # Eval pairs for progress tracking
    eval_pairs = generate_test_set(2000, seed=12345)

    # Print banner
    print(f"\n{'=' * 70}")
    print("Reproduction Pipeline: 62p CircularArcQwen3")
    print(f"{'=' * 70}")
    print(f"Parameters:  62 (expected)")
    print(f"Seed:        {args.seed}")
    print(f"Device:      {device}")
    print(f"Stages:      {len(stages)}")
    for i, s in enumerate(stages):
        step_count = args.steps_override or s["steps"]
        print(f"  Stage {i}: {s['name']} ({step_count} steps)")
    print(f"Output:      {ckpt_dir}/")
    print(f"{'=' * 70}\n")

    # Metrics CSV
    metrics = MetricsWriter(f"{ckpt_dir}/metrics.csv")

    t0 = time.time()
    global_step = 0
    stage_results = []

    for stage_num, stage_cfg in enumerate(stages):
        steps = args.steps_override or stage_cfg["steps"]

        # Set seed for this stage
        stage_seed = stage_cfg["seed"]
        random.seed(stage_seed)
        torch.manual_seed(stage_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(stage_seed)

        if stage_num == 0:
            # Create model from scratch
            model = CircularArcQwen3(
                d_model=3, ff=2,
                n_heads=1, n_kv_heads=1,
                head_dim=4, rope_theta=3.0,
                tie_kv=True, tie_qo=True,
            ).to(device)
            n_params = count_params(model)
            print(f"Model created: {n_params} params")
            if n_params != 62:
                print(f"WARNING: Expected 62 params, got {n_params}")
        else:
            # Load best checkpoint from previous stage
            best_path = f"{ckpt_dir}/stage{stage_num - 1}_best.pt"
            final_path = f"{ckpt_dir}/stage{stage_num - 1}_final.pt"
            # For 62p, always use best base (not EMA) for stage transitions
            for path in [best_path, final_path]:
                if os.path.exists(path):
                    ckpt = load_checkpoint(model, path)
                    acc = ckpt.get("accuracy", "?")
                    print(f"  Loaded {os.path.basename(path)} "
                          f"(accuracy={acc})")
                    break
            else:
                print(f"  WARNING: No checkpoint found for stage "
                      f"{stage_num - 1}, continuing with current weights")

        print(f"\n{'=' * 50}")
        print(f"{stage_cfg['name']}")
        print(f"{'=' * 50}")

        # Run stage
        optimizer_cls = None
        if stage_cfg.get("optimizer_cls") == "adam":
            optimizer_cls = torch.optim.Adam

        if stage_cfg["type"] == "cosine":
            best_acc, global_step = run_stage_cosine(
                model=model, device=device,
                lr=stage_cfg["lr"], batch_size=stage_cfg["batch_size"],
                steps=steps, eval_pairs=eval_pairs,
                ckpt_dir=ckpt_dir, stage_num=stage_num,
                n_params=n_params, metrics=metrics, t0=t0,
                global_step=global_step,
                weight_decay=stage_cfg["weight_decay"],
                min_lr_ratio=stage_cfg.get("min_lr_ratio", 0.1),
                optimizer_cls=optimizer_cls,
                ema_decay=stage_cfg.get("ema_decay", 0.0),
                eval_interval=stage_cfg.get("eval_interval", 2000),
            )
        elif stage_cfg["type"] == "constant":
            best_acc, global_step = run_stage_constant(
                model=model, device=device,
                lr=stage_cfg["lr"], batch_size=stage_cfg["batch_size"],
                steps=steps, eval_pairs=eval_pairs,
                ckpt_dir=ckpt_dir, stage_num=stage_num,
                n_params=n_params, metrics=metrics, t0=t0,
                global_step=global_step,
                weight_decay=stage_cfg["weight_decay"],
                ema_decay=stage_cfg.get("ema_decay", 0.0),
                eval_interval=stage_cfg.get("eval_interval", 2000),
            )

        elapsed = time.time() - t0
        stage_results.append({
            "stage": stage_num,
            "name": stage_cfg["name"],
            "best_acc": best_acc,
            "global_step": global_step,
            "elapsed": elapsed,
        })
        print(f"\n  Stage {stage_num} complete: best_acc={best_acc:.6f}, "
              f"global_step={global_step}, elapsed={elapsed:.0f}s")

    metrics.close()

    # Load final best checkpoint
    last_stage = len(stages) - 1
    best_path = f"{ckpt_dir}/stage{last_stage}_best.pt"
    if os.path.exists(best_path):
        load_checkpoint(model, best_path)
    save_checkpoint(
        model, f"{ckpt_dir}/final_best.pt",
        global_step, stage_results[-1]["best_acc"], n_params,
    )

    # Summary evaluation
    print(f"\n{'=' * 70}")
    print("Pipeline complete -- running summary evaluation")
    print(f"{'=' * 70}")
    summary_pairs = generate_test_set(2000, seed=12345)
    seq_acc, dig_acc = evaluate(model, device, test_pairs=summary_pairs)
    elapsed = time.time() - t0
    print(f"  Summary eval (2K random, seed=12345): "
          f"exact={seq_acc:.4f} digit={dig_acc:.4f}")
    print(f"  Total training time: {elapsed:.0f}s ({elapsed/3600:.1f}h)")

    # Validation
    validation = None
    if not args.no_validate:
        validation = run_validation(model, device, ckpt_dir)

    # Save run summary
    summary = {
        "config": "62p_circular_arc",
        "description": "CircularArcQwen3, d=3 ff=2, tieKV+tieQO, 4-stage pipeline",
        "n_params": n_params,
        "expected_params": 62,
        "seed": args.seed,
        "device": str(device),
        "steps_override": args.steps_override,
        "stages": stage_results,
        "summary_eval": {"exact_acc": seq_acc, "digit_acc": dig_acc},
        "validation": validation,
        "total_elapsed": time.time() - t0,
        "ckpt_dir": ckpt_dir,
    }
    summary_path = f"{ckpt_dir}/run_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Final output
    print(f"\n{'=' * 70}")
    print(f"DONE: 62p CircularArcQwen3 seed={args.seed}")
    print(f"{'=' * 70}")
    print(f"  Checkpoints: {ckpt_dir}/")
    print(f"  Metrics CSV: {ckpt_dir}/metrics.csv")
    print(f"  Run summary: {summary_path}")
    if validation:
        v = validation.get("verify", {})
        n_samples = v.get("n_samples", 0)
        n_errors = v.get("n_errors", 0)
        qualified_str = "QUALIFIED" if v.get("qualified") else "NOT QUALIFIED"
        print(f"  Verify: {n_samples - n_errors}/{n_samples} ({qualified_str})")
    print(f"  Total time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
