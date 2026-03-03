"""Reproduction pipeline for 96p Rank1OutModel.

Self-contained script for the 2-stage pipeline:
  Stage 0: AdamW cosine lr=0.01, 200K steps, batch=128, seed=9999
  Stage 1: Targeted FT, Adam (no wd) lr=0.0003, 3K steps/iter, max 5 iters

The Rank1OutModel replaces the dense O projection with a rank-1 factorization,
saving 5 params vs standard Qwen3. Only 8 targeted FT pairs needed for 100%.

Usage:
    python experiments/reproduce_96p.py --seed 9999 --device cuda
    python experiments/reproduce_96p.py --seed 9999 --steps-override 200  # smoke test
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

from minimal10digittransformer.model.rank1_out import Rank1OutModel
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


def find_errors(model: nn.Module, device: torch.device,
                test_pairs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Autoregressive decode on each test pair, return those predicted incorrectly."""
    model.eval()
    errors = []
    with torch.no_grad():
        for a, b in test_pairs:
            exp = expected_output(a, b)
            inp = torch.tensor([encode(a, b)], dtype=torch.long, device=device)
            x = inp
            pred = []
            for _ in range(OUTPUT_LEN):
                logits = model(x)
                next_tok = logits[0, -1, :].argmax().item()
                pred.append(next_tok)
                x = torch.cat([x, torch.tensor([[next_tok]], device=device)],
                              dim=1)
            if pred != exp:
                errors.append((a, b))
    return errors


def build_targeted_batch(error_pairs: list[tuple[int, int]], batch_size: int,
                         device: torch.device
                         ) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a training batch mixing error pairs with random pairs."""
    full_list = []
    label_list = []

    if len(error_pairs) >= batch_size:
        selected = random.sample(error_pairs, batch_size)
    else:
        selected = list(error_pairs)

    for a, b in selected:
        inp = encode(a, b)
        tgt = expected_output(a, b)
        full_list.append(inp + tgt)
        label_list.append([-100] * INPUT_LEN + tgt)

    n_random = batch_size - len(full_list)
    if n_random > 0:
        rand_seq, rand_labels = generate_batch(n_random, device, max_digits=10)
        error_seq = torch.tensor(full_list, dtype=torch.long, device=device)
        error_labels = torch.tensor(label_list, dtype=torch.long, device=device)
        full_seq_t = torch.cat([error_seq, rand_seq], dim=0)
        labels_t = torch.cat([error_labels, rand_labels], dim=0)
    else:
        full_seq_t = torch.tensor(full_list, dtype=torch.long, device=device)
        labels_t = torch.tensor(label_list, dtype=torch.long, device=device)

    perm = torch.randperm(batch_size)
    return full_seq_t[perm], labels_t[perm]


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
                     global_step: int,
                     eval_interval: int = 2000,
                     ) -> tuple[float, int]:
    """Run a cosine LR decay training stage (AdamW).

    Returns (best_accuracy, updated_global_step).
    """
    min_lr = lr * 0.1
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    best_acc = 0.0

    print(f"\n  Stage {stage_num} [cosine]: lr={lr}, min_lr={min_lr}, "
          f"batch={batch_size}, steps={steps}")

    for step in range(1, steps + 1):
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

    save_checkpoint(
        model, f"{ckpt_dir}/stage{stage_num}_final.pt",
        global_step, best_acc, n_params,
    )
    return best_acc, global_step


def run_stage_targeted(model: nn.Module, device: torch.device,
                       lr: float, batch_size: int,
                       steps_per_iter: int, max_iters: int,
                       eval_pairs: list, ckpt_dir: str, stage_num: int,
                       n_params: int, metrics: MetricsWriter, t0: float,
                       global_step: int,
                       eval_interval: int = 500,
                       ) -> tuple[float, int]:
    """Run iterated targeted fine-tuning with Adam (no weight decay).

    Returns (best_accuracy, updated_global_step).
    """
    print(f"\n  Stage {stage_num} [targeted]: lr={lr}, batch={batch_size}, "
          f"steps/iter={steps_per_iter}, max_iters={max_iters}")

    # Initial error finding
    print(f"    Finding initial errors on {len(eval_pairs)} pairs...")
    errors = find_errors(model, device, eval_pairs)
    print(f"    Initial errors: {len(errors)}")

    if len(errors) == 0:
        print("    Model is already perfect on eval set!")
        save_checkpoint(
            model, f"{ckpt_dir}/stage{stage_num}_best.pt",
            global_step, 1.0, n_params,
        )
        return 1.0, global_step

    cumulative_errors = set((a, b) for a, b in errors)
    best_n_errors = len(errors)
    best_acc = 1.0 - best_n_errors / len(eval_pairs)

    for iteration in range(1, max_iters + 1):
        error_list = list(cumulative_errors)
        print(f"\n    --- Targeted iteration {iteration} ---")
        print(f"    Training on {len(error_list)} cumulative error pairs")

        # Adam with NO weight decay
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=0.0,
        )

        for step in range(1, steps_per_iter + 1):
            model.train()
            full_seq, labels = build_targeted_batch(
                error_list, batch_size, device)

            optimizer.zero_grad()
            loss = train_step(model, full_seq, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            global_step += 1

            if step % 500 == 0:
                elapsed = time.time() - t0
                print(f"      [iter {iteration}] step {step:5d}/{steps_per_iter} "
                      f"| loss {loss.item():.6f} | errors={len(error_list)} "
                      f"| {elapsed:.0f}s")
                sys.stdout.flush()

            if step % eval_interval == 0:
                quick_pairs = eval_pairs[:200]
                seq_acc, dig_acc = evaluate(model, device,
                                           test_pairs=quick_pairs)
                elapsed = time.time() - t0
                print(f"      [iter {iteration}] EVAL step {step}: "
                      f"exact={seq_acc:.4f} digit={dig_acc:.4f} "
                      f"[{elapsed:.0f}s]")
                sys.stdout.flush()

                metrics.writerow([
                    stage_num, global_step, step, f"{loss.item():.6f}",
                    f"{lr:.2e}", f"{seq_acc:.4f}", f"{dig_acc:.4f}",
                    f"{elapsed:.1f}",
                ])
                metrics.flush_file()

        # Full evaluation after iteration
        print(f"    Full evaluation on {len(eval_pairs)} pairs...")
        new_errors = find_errors(model, device, eval_pairs)
        n_new_errors = len(new_errors)
        seq_acc = 1.0 - n_new_errors / len(eval_pairs)
        elapsed = time.time() - t0
        print(f"    Iteration {iteration} result: {n_new_errors} errors "
              f"(exact={seq_acc:.6f}) [{elapsed:.0f}s]")

        if n_new_errors < best_n_errors:
            best_n_errors = n_new_errors
            best_acc = seq_acc
            save_checkpoint(
                model, f"{ckpt_dir}/stage{stage_num}_best.pt",
                global_step, seq_acc, n_params,
                extra={"targeted_iteration": iteration,
                       "n_errors": n_new_errors,
                       "cumulative_pairs": len(cumulative_errors)},
            )
            print(f"    ** NEW BEST: {seq_acc:.6f} ({n_new_errors} errors) **")

        if n_new_errors == 0:
            print(f"    Perfect! 0 errors after iteration {iteration}.")
            break

        # Accumulate new errors
        prev_count = len(cumulative_errors)
        for a, b in new_errors:
            cumulative_errors.add((a, b))
        added = len(cumulative_errors) - prev_count
        print(f"    Cumulative errors: {prev_count} -> "
              f"{len(cumulative_errors)} (+{added} new)")

    save_checkpoint(
        model, f"{ckpt_dir}/stage{stage_num}_final.pt",
        global_step, best_acc, n_params,
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
        description="Reproduction pipeline for 96p Rank1OutModel.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python experiments/reproduce_96p.py --seed 9999 --device cuda
  python experiments/reproduce_96p.py --seed 9999 --steps-override 200
""",
    )
    parser.add_argument("--seed", type=int, default=9999,
                        help="Random seed for base training (default: 9999)")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--steps-override", type=int, default=None,
                        help="Override step count for ALL stages (smoke test)")
    parser.add_argument("--no-validate", action="store_true",
                        help="Skip final validation")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory "
                             "(default: checkpoints/reproduce_96p_s{seed})")
    parser.add_argument("--train-eval-seed", type=int, default=777,
                        help="Seed for generating targeted FT eval pairs "
                             "(default: 777)")
    parser.add_argument("--train-eval-size", type=int, default=10000,
                        help="Number of eval pairs for targeted FT "
                             "(default: 10000)")
    args = parser.parse_args()

    if args.train_eval_seed in RESERVED_SEEDS:
        print(f"ERROR: --train-eval-seed {args.train_eval_seed} is reserved!")
        sys.exit(1)

    device = torch.device(args.device)

    # Output directory
    ckpt_dir = args.output_dir or f"checkpoints/reproduce_96p_s{args.seed}"
    os.makedirs(ckpt_dir, exist_ok=True)

    # Eval pairs
    progress_pairs = generate_test_set(2000, seed=12345)
    targeted_pairs = generate_test_set(args.train_eval_size,
                                       seed=args.train_eval_seed)

    # Set seeds
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Create model
    model = Rank1OutModel(
        d_model=3, ff=2,
        n_heads=1, n_kv_heads=1,
        head_dim=4, rope_theta=3.0,
        tie_kv=True,
    ).to(device)
    n_params = count_params(model)

    # Print banner
    print(f"\n{'=' * 70}")
    print("Reproduction Pipeline: 96p Rank1OutModel")
    print(f"{'=' * 70}")
    print(f"Parameters:  {n_params} (expected: 96)")
    print(f"Seed:        {args.seed}")
    print(f"Device:      {device}")
    print(f"Pipeline:    cosine 200K -> targeted FT")
    print(f"Output:      {ckpt_dir}/")
    print(f"{'=' * 70}\n")

    if n_params != 96:
        print(f"WARNING: Expected 96 params, got {n_params}")

    # Metrics CSV
    metrics = MetricsWriter(f"{ckpt_dir}/metrics.csv")

    t0 = time.time()
    global_step = 0
    stage_results = []

    # ---- Stage 0: AdamW cosine lr=0.01, 200K steps ----
    steps_0 = args.steps_override or 200000
    print(f"\n{'=' * 50}")
    print("Stage 0: AdamW cosine lr=0.01")
    print(f"{'=' * 50}")
    best_acc, global_step = run_stage_cosine(
        model=model, device=device, lr=0.01, batch_size=128,
        steps=steps_0, eval_pairs=progress_pairs,
        ckpt_dir=ckpt_dir, stage_num=0, n_params=n_params,
        metrics=metrics, t0=t0, global_step=global_step,
    )
    elapsed = time.time() - t0
    stage_results.append({
        "stage": 0, "name": "AdamW cosine",
        "best_acc": best_acc, "global_step": global_step,
        "elapsed": elapsed,
    })
    print(f"\n  Stage 0 complete: best_acc={best_acc:.6f}, "
          f"elapsed={elapsed:.0f}s")

    # ---- Stage 1: Targeted FT from FINAL checkpoint (not best) ----
    # Load final.pt (end-of-training, more errors but better generalization)
    final_path = f"{ckpt_dir}/stage0_final.pt"
    best_path = f"{ckpt_dir}/stage0_best.pt"
    if os.path.exists(final_path):
        ckpt = load_checkpoint(model, final_path)
        print(f"  Loaded stage0_final.pt "
              f"(accuracy={ckpt.get('accuracy', '?')})")
    elif os.path.exists(best_path):
        ckpt = load_checkpoint(model, best_path)
        print(f"  Loaded stage0_best.pt (final not available, "
              f"accuracy={ckpt.get('accuracy', '?')})")

    steps_per_iter = args.steps_override or 3000
    print(f"\n{'=' * 50}")
    print("Stage 1: Targeted FT (Adam no-wd, lr=0.0003)")
    print(f"{'=' * 50}")
    best_acc, global_step = run_stage_targeted(
        model=model, device=device, lr=0.0003, batch_size=256,
        steps_per_iter=steps_per_iter, max_iters=5,
        eval_pairs=targeted_pairs,
        ckpt_dir=ckpt_dir, stage_num=1, n_params=n_params,
        metrics=metrics, t0=t0, global_step=global_step,
    )
    elapsed = time.time() - t0
    stage_results.append({
        "stage": 1, "name": "Targeted FT",
        "best_acc": best_acc, "global_step": global_step,
        "elapsed": elapsed,
    })
    print(f"\n  Stage 1 complete: best_acc={best_acc:.6f}, "
          f"elapsed={elapsed:.0f}s")

    metrics.close()

    # Load final best checkpoint
    best_path = f"{ckpt_dir}/stage1_best.pt"
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
        "config": "96p_rank1_out",
        "description": "Rank1OutModel, d=3 ff=2, tieKV, 2-stage pipeline",
        "n_params": n_params,
        "expected_params": 96,
        "seed": args.seed,
        "train_eval_seed": args.train_eval_seed,
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
    print(f"DONE: 96p Rank1OutModel seed={args.seed}")
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
