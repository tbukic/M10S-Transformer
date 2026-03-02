"""
L-BFGS fine-tuning for pushing near-perfect models to 100%.

Inspired by staghado's finding that AdamW converges to saddle points,
while L-BFGS uses curvature information to escape them.

Usage:
  python experiments/lbfgs_finetune.py checkpoints/qwen3_d3_ff2_89p_tiekv_tieqo_s118/best.pt
  python experiments/lbfgs_finetune.py <checkpoint> --lr 1.0 --max-iter 20 --steps 500
"""

import argparse
import csv
import json
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


def compute_loss(model, full_seq, labels):
    logits = model(full_seq)
    shift_logits = logits[:, :-1, :].reshape(-1, VOCAB_SIZE)
    shift_labels = labels[:, 1:].reshape(-1)
    return F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="Path to .pt checkpoint")
    parser.add_argument("--lr", type=float, default=1.0, help="L-BFGS learning rate")
    parser.add_argument("--max-iter", type=int, default=20, help="Max L-BFGS iterations per step")
    parser.add_argument("--history-size", type=int, default=10, help="L-BFGS history size")
    parser.add_argument("--steps", type=int, default=500, help="Number of L-BFGS steps (each with new batch)")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size (larger = better for L-BFGS)")
    parser.add_argument("--eval-interval", type=int, default=50, help="Eval every N steps")
    parser.add_argument("--test-set", type=str, default="data/test_10k.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
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
    n_params = sum(p.numel() for p in model.parameters())

    print(f"Loaded: {args.checkpoint}")
    print(f"Params: {n_params}, step: {ckpt.get('step', '?')}, acc: {ckpt.get('accuracy', '?')}")
    print(f"L-BFGS: lr={args.lr}, max_iter={args.max_iter}, history={args.history_size}")
    print(f"Steps: {args.steps}, batch={args.batch_size}, eval_interval={args.eval_interval}")

    test_pairs = load_test_set(args.test_set) if args.test_set else None

    # Initial eval
    seq_acc, dig_acc = evaluate(model, device, n_samples=200, test_pairs=test_pairs[:200] if test_pairs else None)
    print(f"Initial: exact={seq_acc:.4f} digit={dig_acc:.4f}")

    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=args.lr,
        max_iter=args.max_iter,
        history_size=args.history_size,
        line_search_fn="strong_wolfe",
    )

    # Output directory
    ckpt_name = os.path.basename(os.path.dirname(args.checkpoint))
    out_dir = f"checkpoints/{ckpt_name}_lbfgs_s{args.seed}"
    os.makedirs(out_dir, exist_ok=True)

    # Metrics CSV
    metrics_file = open(f"{out_dir}/metrics.csv", "w", newline="")
    metrics_writer = csv.writer(metrics_file)
    metrics_writer.writerow(["step", "loss", "exact_acc", "digit_acc", "elapsed"])

    best_acc = seq_acc
    t0 = time.time()

    for step in range(1, args.steps + 1):
        model.train()
        full_seq, labels = generate_batch(args.batch_size, device, 10)

        def closure():
            optimizer.zero_grad()
            loss = compute_loss(model, full_seq, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            return loss

        loss = optimizer.step(closure)

        if step % 10 == 0:
            elapsed = time.time() - t0
            print(f"  step {step:4d} | loss {loss.item():.6f} | {elapsed:.0f}s")
            sys.stdout.flush()

        if step % args.eval_interval == 0:
            eval_pairs = test_pairs[:200] if test_pairs else None
            seq_acc, dig_acc = evaluate(model, device, n_samples=200, test_pairs=eval_pairs)
            elapsed = time.time() - t0
            print(f"  EVAL step {step}: exact={seq_acc:.4f} digit={dig_acc:.4f} [{elapsed:.0f}s]")
            sys.stdout.flush()

            metrics_writer.writerow([step, f"{loss.item():.6f}", f"{seq_acc:.4f}", f"{dig_acc:.4f}", f"{elapsed:.1f}"])
            metrics_file.flush()

            if seq_acc > best_acc:
                best_acc = seq_acc
                torch.save({
                    "state_dict": model.state_dict(),
                    "step": step,
                    "accuracy": seq_acc,
                    "digit_accuracy": dig_acc,
                    "n_params": n_params,
                    "config": cfg,
                    "lbfgs_config": {"lr": args.lr, "max_iter": args.max_iter,
                                     "history_size": args.history_size},
                    "source_checkpoint": args.checkpoint,
                }, f"{out_dir}/best.pt")
                print(f"  ** NEW BEST: {seq_acc:.4f} **")

    metrics_file.close()

    # Full 10K eval
    if test_pairs:
        print(f"\nFull 10K evaluation...")
        seq_acc, dig_acc = evaluate(model, device, n_samples=10000, test_pairs=test_pairs)
        print(f"FINAL 10K: exact={seq_acc:.4f} digit={dig_acc:.4f}")
        print(f"Best 200-sample: {best_acc:.4f}")

    torch.save({
        "state_dict": model.state_dict(),
        "step": args.steps,
        "accuracy": seq_acc if test_pairs else best_acc,
        "n_params": n_params,
        "config": cfg,
    }, f"{out_dir}/final.pt")

    results = {
        "source": args.checkpoint,
        "n_params": n_params,
        "seed": args.seed,
        "best_200_acc": best_acc,
        "final_10k_acc": seq_acc if test_pairs else None,
        "lbfgs_lr": args.lr,
        "lbfgs_max_iter": args.max_iter,
        "steps": args.steps,
        "elapsed": time.time() - t0,
    }
    with open(f"{out_dir}/results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {out_dir}/")


if __name__ == "__main__":
    main()
