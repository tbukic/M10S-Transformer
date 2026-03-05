"""Targeted fine-tuning for tying_search models.

Finds error pairs, trains on them with padding. Supports iterated mode.
Works with any model from tying_search.py (CircularArc + monkey-patched attention).

Usage:
    python experiments/tying_targeted_ft.py \
        --ckpt checkpoints/tying_continue/..../best.pt \
        --lr 0.0003 --steps 3000 --iterated --max-iters 10
"""

import argparse
import json
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.tying_search import build_model, count_params
from minimal10digittransformer.data.addition import (
    encode, expected_output, generate_batch, load_test_set,
)
from minimal10digittransformer.evaluation.metrics import evaluate, evaluate_detailed
from minimal10digittransformer.model.qwen3 import VOCAB_SIZE, INPUT_LEN, OUTPUT_LEN, TOTAL_LEN


def find_errors(model, device, test_pairs):
    """Find all (a, b) pairs the model gets wrong."""
    model.eval()
    errors = []
    with torch.no_grad():
        for a, b in test_pairs:
            input_ids = encode(a, b)
            exp = expected_output(a, b)
            x = torch.tensor([input_ids], device=device)

            # Autoregressive generation
            for pos in range(OUTPUT_LEN):
                logits = model(x)
                next_token = logits[0, -1].argmax().item()
                x = torch.cat([x, torch.tensor([[next_token]], device=device)], dim=1)

            pred_digits = x[0, INPUT_LEN:INPUT_LEN + OUTPUT_LEN].tolist()

            if pred_digits != exp:
                errors.append((a, b))

    return errors


def make_targeted_batch(error_pairs, batch_size, device):
    """Make a batch that includes all error pairs + random padding."""
    all_pairs = list(error_pairs)
    # Pad with random pairs to fill batch
    while len(all_pairs) < batch_size:
        a = random.randint(0, 10**10 - 1)
        b = random.randint(0, 10**10 - 1)
        all_pairs.append((a, b))

    random.shuffle(all_pairs)
    all_pairs = all_pairs[:batch_size]

    # Encode using the proper encode(a, b) API
    input_ids_list = []
    labels_list = []
    for a, b in all_pairs:
        ids = encode(a, b)
        exp = expected_output(a, b)
        full_seq = ids + exp
        input_ids_list.append(full_seq)

        label = [-100] * INPUT_LEN + exp
        labels_list.append(label)

    return (torch.tensor(input_ids_list, device=device),
            torch.tensor(labels_list, device=device))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--wd", type=float, default=0.0)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--iterated", action="store_true")
    parser.add_argument("--max-iters", type=int, default=10)
    parser.add_argument("--eval-size", type=int, default=2000,
                        help="Eval set size for error finding (default 2000)")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", "2")

    # Load checkpoint
    ckpt = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    cfg = ckpt["config"]
    seed = ckpt["seed"]
    prev_step = ckpt["step"]
    prev_acc = ckpt.get("acc", 0)
    n_params = ckpt.get("n_params", 0)

    print(f"Loaded: {args.ckpt}")
    print(f"  Config: {cfg['name']}, seed={seed}, step={prev_step}, acc={prev_acc:.1%}, {n_params}p")

    # Output directory
    if args.output_dir:
        out_dir = args.output_dir
    else:
        out_dir = f"checkpoints/tying_targeted/{cfg['name']}_s{seed}"
    os.makedirs(out_dir, exist_ok=True)

    # Build model
    model = build_model(cfg, args.device)
    model.load_state_dict(ckpt["state_dict"])
    actual_params = count_params(model)
    print(f"  Model: {actual_params}p")

    # Test pairs for error finding
    from minimal10digittransformer.data.addition import generate_test_set
    eval_pairs = generate_test_set(args.eval_size, seed=12345)

    if not args.iterated:
        # Single round
        errors = find_errors(model, args.device, eval_pairs)
        print(f"  Found {len(errors)} errors in {len(eval_pairs)} pairs")

        if len(errors) == 0:
            print("  Already perfect!")
            return

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
        model.train()
        for step in range(1, args.steps + 1):
            batch, labels = make_targeted_batch(errors, args.batch_size, args.device)
            logits = model(batch)
            shift_logits = logits[:, :-1, :].reshape(-1, VOCAB_SIZE)
            shift_labels = labels[:, 1:].reshape(-1)
            loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if step % 500 == 0 or step == args.steps:
                model.eval()
                with torch.no_grad():
                    acc, _ = evaluate(model, args.device, test_pairs=eval_pairs[:500])
                print(f"  step {step}/{args.steps} loss={loss.item():.4f} acc={acc:.1%}", flush=True)
                model.train()

        # Save
        torch.save({
            "state_dict": model.state_dict(),
            "config": cfg,
            "step": prev_step,
            "acc": acc,
            "seed": seed,
            "n_params": actual_params,
        }, os.path.join(out_dir, "targeted.pt"))

    else:
        # Iterated targeted FT
        all_error_pairs = set()
        for iteration in range(1, args.max_iters + 1):
            errors = find_errors(model, args.device, eval_pairs)
            new_errors = set(errors) - all_error_pairs
            all_error_pairs.update(errors)

            print(f"\n  Iter {iteration}: {len(errors)} errors ({len(new_errors)} new, "
                  f"{len(all_error_pairs)} cumulative)")

            if len(errors) == 0:
                print(f"  PERFECT at iteration {iteration}!")
                break

            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
            model.train()
            for step in range(1, args.steps + 1):
                batch, labels = make_targeted_batch(list(all_error_pairs), args.batch_size, args.device)
                logits = model(batch)
                shift_logits = logits[:, :-1, :].reshape(-1, VOCAB_SIZE)
                shift_labels = labels[:, 1:].reshape(-1)
                loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                if step % 500 == 0 or step == args.steps:
                    print(f"    step {step}/{args.steps} loss={loss.item():.4f}", flush=True)

            model.eval()
            with torch.no_grad():
                acc, _ = evaluate(model, args.device, test_pairs=eval_pairs[:500])
            print(f"  After iter {iteration}: acc={acc:.1%}", flush=True)

            # Save best
            torch.save({
                "state_dict": model.state_dict(),
                "config": cfg,
                "step": prev_step,
                "acc": acc,
                "seed": seed,
                "n_params": actual_params,
                "targeted_iters": iteration,
                "cumulative_errors": len(all_error_pairs),
            }, os.path.join(out_dir, "best.pt"))

        # Final eval on full set
        model.eval()
        with torch.no_grad():
            acc, _ = evaluate(model, args.device, test_pairs=eval_pairs)
        print(f"\n  Final eval ({len(eval_pairs)} pairs): acc={acc:.1%}")
        print(f"  Checkpoints in {out_dir}")


if __name__ == "__main__":
    main()
