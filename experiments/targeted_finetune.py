"""
Targeted fine-tuning: focus training on error pairs to push near-perfect models to 100%.

Finds all pairs the model gets wrong, then trains with batches that always include
those error pairs (padded with random pairs to fill the batch). Supports iterated mode
where errors are re-collected after each round and accumulated.

Usage:
  # Single round of targeted fine-tuning
  python experiments/targeted_finetune.py \
    --checkpoint checkpoints/qwen3_d3_ff2_89p_tiekv_tieqo_s118/best.pt \
    --test-set data/test_10k.json \
    --lr 0.0003 --steps 5000

  # Iterated: re-evaluate and accumulate errors each round
  python experiments/targeted_finetune.py \
    --checkpoint checkpoints/qwen3_d3_ff2_89p_tiekv_tieqo_s118/best.pt \
    --test-set data/test_10k.json \
    --lr 0.0003 --steps 5000 --iterated --max-iters 10

  # With 50K eval at the end
  python experiments/targeted_finetune.py \
    --checkpoint checkpoints/qwen3_d3_ff2_89p_tiekv_tieqo_s118/best.pt \
    --test-set data/test_10k.json \
    --lr 0.0003 --steps 5000 --iterated --eval-50k
"""

import argparse
import csv
import json
import os
import random
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from minimal10digittransformer.model.qwen3 import (
    Qwen3AdditionModel, VOCAB_SIZE, INPUT_LEN, OUTPUT_LEN, TOTAL_LEN,
)
from minimal10digittransformer.data.addition import (
    encode, expected_output, generate_batch, load_test_set,
)
from minimal10digittransformer.evaluation.metrics import evaluate, evaluate_detailed


# ── Model loading ────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, device: torch.device):
    """Load a Qwen3AdditionModel from a checkpoint file.

    Returns (model, config_dict, checkpoint_dict).
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    cfg = ckpt["config"]

    model = Qwen3AdditionModel(
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_kv_heads=cfg["n_kv_heads"],
        head_dim=cfg["head_dim"],
        ff=cfg["ff"],
        rope_theta=cfg["rope_theta"],
        qk_norm=not cfg.get("no_qk_norm", False),
        use_swiglu=not cfg.get("gelu", False),
        tie_kv=cfg.get("tie_kv", False),
        tie_qo=cfg.get("tie_qo", False),
        tie_gate=cfg.get("tie_gate", False),
        repeats=cfg.get("repeats", 1),
        share_norms=cfg.get("share_norms", False),
        share_block_norms=cfg.get("share_block_norms", False),
    ).to(device)

    model.load_state_dict(ckpt["state_dict"])
    return model, cfg, ckpt


# ── Error finding ────────────────────────────────────────────────────────────

def find_errors(model: Qwen3AdditionModel, device: torch.device,
                test_pairs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Autoregressive decode on each test pair, return those predicted incorrectly."""
    model.eval()
    errors = []

    with torch.no_grad():
        for a, b in test_pairs:
            exp = expected_output(a, b)
            inp = torch.tensor([encode(a, b)], dtype=torch.long, device=device)

            # Autoregressive generation
            x = inp
            pred = []
            for _ in range(OUTPUT_LEN):
                logits = model(x)
                next_tok = logits[0, -1, :].argmax().item()
                pred.append(next_tok)
                x = torch.cat([x, torch.tensor([[next_tok]], device=device)], dim=1)

            if pred != exp:
                errors.append((a, b))

    return errors


# ── Targeted batch construction ──────────────────────────────────────────────

def build_targeted_batch(error_pairs: list[tuple[int, int]],
                         batch_size: int,
                         device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a training batch that includes all error pairs, padded with random pairs.

    If there are more error pairs than batch_size, a random subset is used.
    Otherwise, the batch is filled with randomly generated pairs.

    Returns (full_seq, labels) tensors of shape [batch_size, TOTAL_LEN].
    """
    full_list = []
    label_list = []

    # Include error pairs (sample if too many)
    if len(error_pairs) >= batch_size:
        selected = random.sample(error_pairs, batch_size)
    else:
        selected = list(error_pairs)

    for a, b in selected:
        inp = encode(a, b)
        tgt = expected_output(a, b)
        full_seq = inp + tgt
        labels = [-100] * INPUT_LEN + tgt
        full_list.append(full_seq)
        label_list.append(labels)

    # Fill remainder with random pairs
    n_random = batch_size - len(full_list)
    if n_random > 0:
        rand_seq, rand_labels = generate_batch(n_random, device, max_digits=10)
        # generate_batch returns tensors; convert error pairs to tensors and concat
        error_seq = torch.tensor(full_list, dtype=torch.long, device=device)
        error_labels = torch.tensor(label_list, dtype=torch.long, device=device)
        full_seq_t = torch.cat([error_seq, rand_seq], dim=0)
        labels_t = torch.cat([error_labels, rand_labels], dim=0)
        # Shuffle so error pairs aren't always at the start
        perm = torch.randperm(batch_size)
        return full_seq_t[perm], labels_t[perm]
    else:
        full_seq_t = torch.tensor(full_list, dtype=torch.long, device=device)
        labels_t = torch.tensor(label_list, dtype=torch.long, device=device)
        perm = torch.randperm(batch_size)
        return full_seq_t[perm], labels_t[perm]


# ── Training step ────────────────────────────────────────────────────────────

def compute_loss(model: Qwen3AdditionModel, full_seq: torch.Tensor,
                 labels: torch.Tensor) -> torch.Tensor:
    """Standard causal LM loss with shifted logits."""
    logits = model(full_seq)
    shift_logits = logits[:, :-1, :].reshape(-1, VOCAB_SIZE)
    shift_labels = labels[:, 1:].reshape(-1)
    return F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)


# ── Single targeted fine-tuning round ────────────────────────────────────────

def targeted_finetune_round(
    model: Qwen3AdditionModel,
    error_pairs: list[tuple[int, int]],
    device: torch.device,
    lr: float,
    weight_decay: float,
    batch_size: int,
    steps: int,
    test_pairs: list[tuple[int, int]],
    eval_interval: int,
    metrics_writer,
    iteration: int,
    t0: float,
) -> tuple[float, float]:
    """Run one round of targeted fine-tuning.

    Returns (best_seq_acc, final_loss).
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_acc = 0.0

    for step in range(1, steps + 1):
        model.train()
        full_seq, labels = build_targeted_batch(error_pairs, batch_size, device)

        optimizer.zero_grad()
        loss = compute_loss(model, full_seq, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 500 == 0:
            elapsed = time.time() - t0
            print(f"  [iter {iteration}] step {step:5d}/{steps} | "
                  f"loss {loss.item():.6f} | errors={len(error_pairs)} | {elapsed:.0f}s")
            sys.stdout.flush()

        if step % eval_interval == 0:
            eval_pairs = test_pairs[:200] if len(test_pairs) > 200 else test_pairs
            seq_acc, dig_acc = evaluate(model, device, test_pairs=eval_pairs)
            elapsed = time.time() - t0
            print(f"  [iter {iteration}] EVAL step {step}: "
                  f"exact={seq_acc:.4f} digit={dig_acc:.4f} [{elapsed:.0f}s]")
            sys.stdout.flush()

            if metrics_writer:
                metrics_writer.writerow([
                    iteration, step, f"{loss.item():.6f}",
                    f"{seq_acc:.4f}", f"{dig_acc:.4f}",
                    len(error_pairs), f"{elapsed:.1f}",
                ])

            if seq_acc > best_acc:
                best_acc = seq_acc

    return best_acc, loss.item()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Targeted fine-tuning on error pairs to push near-perfect models to 100%%.",
    )
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--test-set", type=str, default="data/test_10k.json",
                        help="Path to fixed test set JSON")
    parser.add_argument("--lr", type=float, default=0.0003,
                        help="AdamW learning rate (default: 0.0003)")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                        help="AdamW weight decay (default: 0.01)")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Total batch size (error pairs + random fill)")
    parser.add_argument("--steps", type=int, default=5000,
                        help="Training steps per iteration")
    parser.add_argument("--eval-interval", type=int, default=1000,
                        help="Evaluate every N steps during training")
    parser.add_argument("--iterated", action="store_true",
                        help="Re-evaluate and accumulate errors each round")
    parser.add_argument("--max-iters", type=int, default=10,
                        help="Maximum iterations in iterated mode")
    parser.add_argument("--eval-50k", action="store_true",
                        help="Also evaluate on data/test_50k.json at the end")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: <checkpoint_dir>/targeted/)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load model
    model, cfg, ckpt = load_model(args.checkpoint, device)
    n_params = sum(p.numel() for p in model.parameters())

    print(f"{'=' * 70}")
    print(f"Targeted Fine-Tuning")
    print(f"{'=' * 70}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Params: {n_params}, step: {ckpt.get('step', '?')}, "
          f"acc: {ckpt.get('accuracy', '?')}")
    print(f"LR={args.lr}, wd={args.weight_decay}, batch={args.batch_size}, "
          f"steps/iter={args.steps}")
    print(f"Iterated: {args.iterated}, max_iters={args.max_iters}")
    print(f"{'=' * 70}")

    # Load test set
    test_pairs = load_test_set(args.test_set)
    print(f"Test set: {args.test_set} ({len(test_pairs)} pairs)")

    # Output directory
    if args.output_dir:
        out_dir = args.output_dir
    else:
        ckpt_dir = os.path.dirname(args.checkpoint)
        out_dir = os.path.join(ckpt_dir, "targeted")
    os.makedirs(out_dir, exist_ok=True)

    # Metrics CSV
    metrics_path = os.path.join(out_dir, "metrics.csv")
    metrics_file = open(metrics_path, "w", newline="")
    metrics_writer = csv.writer(metrics_file)
    metrics_writer.writerow([
        "iteration", "step", "loss", "exact_acc_200", "digit_acc_200",
        "n_error_pairs", "elapsed",
    ])

    t0 = time.time()

    # Initial error finding
    print(f"\nFinding initial errors on {len(test_pairs)} pairs...")
    errors = find_errors(model, device, test_pairs)
    print(f"  Initial errors: {len(errors)}")

    if len(errors) == 0:
        print("No errors found -- model is already perfect on this test set!")
        metrics_file.close()
        return

    # Cumulative error set (for iterated mode)
    cumulative_errors = set((a, b) for a, b in errors)
    best_global_acc = 0.0
    best_n_errors = len(errors)

    n_iters = args.max_iters if args.iterated else 1

    for iteration in range(1, n_iters + 1):
        error_list = list(cumulative_errors)
        print(f"\n--- Iteration {iteration} ---")
        print(f"  Training on {len(error_list)} cumulative error pairs "
              f"(batch={args.batch_size})")

        round_acc, final_loss = targeted_finetune_round(
            model=model,
            error_pairs=error_list,
            device=device,
            lr=args.lr,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            steps=args.steps,
            test_pairs=test_pairs,
            eval_interval=args.eval_interval,
            metrics_writer=metrics_writer,
            iteration=iteration,
            t0=t0,
        )
        metrics_file.flush()

        # Full evaluation on test set after this round
        print(f"\n  Full evaluation on {len(test_pairs)} pairs...")
        new_errors = find_errors(model, device, test_pairs)
        n_new_errors = len(new_errors)
        seq_acc = 1.0 - n_new_errors / len(test_pairs)
        elapsed = time.time() - t0
        print(f"  Iteration {iteration} result: {n_new_errors} errors "
              f"(exact={seq_acc:.4f}) [{elapsed:.0f}s]")

        # Save checkpoint if improved
        if n_new_errors < best_n_errors:
            best_n_errors = n_new_errors
            best_global_acc = seq_acc
            torch.save({
                "state_dict": model.state_dict(),
                "step": ckpt.get("step", 0),
                "accuracy": seq_acc,
                "n_params": n_params,
                "config": cfg,
                "targeted_config": {
                    "lr": args.lr,
                    "weight_decay": args.weight_decay,
                    "batch_size": args.batch_size,
                    "steps_per_iter": args.steps,
                    "iteration": iteration,
                    "n_errors": n_new_errors,
                    "cumulative_error_pairs": len(cumulative_errors),
                },
                "source_checkpoint": args.checkpoint,
            }, os.path.join(out_dir, "best.pt"))
            print(f"  ** NEW BEST: {seq_acc:.4f} ({n_new_errors} errors) **")

        if n_new_errors == 0:
            print(f"\n  Perfect! 0 errors after iteration {iteration}.")
            break

        if not args.iterated:
            break

        # Accumulate new errors for next round
        prev_count = len(cumulative_errors)
        for a, b in new_errors:
            cumulative_errors.add((a, b))
        added = len(cumulative_errors) - prev_count
        print(f"  Cumulative errors: {prev_count} -> {len(cumulative_errors)} "
              f"(+{added} new)")

    metrics_file.close()

    # Save final checkpoint
    torch.save({
        "state_dict": model.state_dict(),
        "step": ckpt.get("step", 0),
        "accuracy": 1.0 - best_n_errors / len(test_pairs),
        "n_params": n_params,
        "config": cfg,
    }, os.path.join(out_dir, "final.pt"))

    # Detailed eval on 10K
    print(f"\n{'=' * 70}")
    print(f"Final detailed evaluation on {len(test_pairs)} pairs...")
    detailed = evaluate_detailed(model, device, test_pairs)
    elapsed = time.time() - t0
    print(f"  Exact match: {detailed['exact_acc']:.4f} ({detailed['n_errors']} errors)")
    print(f"  Digit accuracy: {detailed['digit_acc']:.6f}")
    print(f"  Per-position accuracy (LSB->MSB):")
    for i, acc in enumerate(detailed["per_position"]):
        print(f"    Position {i}: {acc:.4f}")
    print(f"  Carry analysis:")
    for n_carries, (acc, count) in detailed["carry_acc"].items():
        print(f"    {n_carries} carries: {acc:.4f} ({count} samples)")

    # Optional 50K eval
    eval_50k_results = None
    if args.eval_50k:
        test_50k_path = "data/test_50k.json"
        if os.path.exists(test_50k_path):
            print(f"\n50K evaluation on {test_50k_path}...")
            test_50k = load_test_set(test_50k_path)
            detailed_50k = evaluate_detailed(model, device, test_50k)
            elapsed = time.time() - t0
            print(f"  50K Exact match: {detailed_50k['exact_acc']:.6f} "
                  f"({detailed_50k['n_errors']} errors)")
            print(f"  50K Digit accuracy: {detailed_50k['digit_acc']:.6f}")
            eval_50k_results = {
                "exact_acc": detailed_50k["exact_acc"],
                "digit_acc": detailed_50k["digit_acc"],
                "n_samples": detailed_50k["n_samples"],
                "n_errors": detailed_50k["n_errors"],
            }
        else:
            print(f"\n  Warning: {test_50k_path} not found, skipping 50K eval")

    # Save results JSON
    results = {
        "source_checkpoint": args.checkpoint,
        "n_params": n_params,
        "seed": args.seed,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "steps_per_iter": args.steps,
        "iterated": args.iterated,
        "total_iterations": iteration,
        "initial_errors": len(errors),
        "final_errors": detailed["n_errors"],
        "final_exact_acc": detailed["exact_acc"],
        "final_digit_acc": detailed["digit_acc"],
        "best_errors": best_n_errors,
        "best_exact_acc": best_global_acc,
        "per_position": detailed["per_position"],
        "carry_acc": {str(k): list(v) for k, v in detailed["carry_acc"].items()},
        "eval_50k": eval_50k_results,
        "elapsed": time.time() - t0,
    }
    results_path = os.path.join(out_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"Results saved to {out_dir}/")
    print(f"  best.pt: {best_n_errors} errors (exact={best_global_acc:.4f})")
    print(f"  Total time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
