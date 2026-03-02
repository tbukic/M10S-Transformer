"""
Qwen3-style transformer training for 10-digit addition.

Usage:
  python experiments/qwen3_train.py --d-model 3 --ff 6 --seed 42  # 173p
  python experiments/qwen3_train.py --d-model 3 --ff 3 --n-heads 1 --n-kv-heads 1 --seed 42  # 122p
"""

import argparse
import math
import random
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from minimal10digittransformer.model.qwen3 import (
    Qwen3AdditionModel, RMSNorm, VOCAB_SIZE, INPUT_LEN, OUTPUT_LEN, TOTAL_LEN,
    MAX_ADDEND, SUM_DIGITS,
)
from minimal10digittransformer.data.addition import (
    encode, expected_output, generate_batch, load_test_set,
)
from minimal10digittransformer.evaluation.metrics import evaluate


# ── Training (single forward pass with teacher forcing) ──────────────────────

def train_step(model: Qwen3AdditionModel, full_seq: torch.Tensor,
               labels: torch.Tensor) -> torch.Tensor:
    """
    Single forward pass: causal attention handles autoregressive conditioning.
    full_seq: [B, 35] prompt + target tokens
    labels: [B, 35] with -100 for masked positions
    """
    logits = model(full_seq)  # [B, 35, V]
    # Shift: logits[t] predicts token[t+1]
    shift_logits = logits[:, :-1, :].reshape(-1, VOCAB_SIZE)
    shift_labels = labels[:, 1:].reshape(-1)
    return F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# ── Curriculum (optional, default off) ───────────────────────────────────────

CURRICULUM = [
    (0, 2000, 3),      # 1-3 digits
    (2000, 7000, 6),    # 1-6 digits
    (7000, float("inf"), 10),  # 1-10 digits
]


def get_max_digits(step: int, use_curriculum: bool) -> int:
    if not use_curriculum:
        return 10
    for start, end, d in CURRICULUM:
        if start <= step < end:
            return d
    return 10


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--d-model", type=int, default=3)
    parser.add_argument("--ff", type=int, default=6)
    parser.add_argument("--n-heads", type=int, default=2)
    parser.add_argument("--n-kv-heads", type=int, default=1)
    parser.add_argument("--head-dim", type=int, default=4)
    parser.add_argument("--rope-theta", type=float, default=3.0)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=15000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--curriculum", action="store_true")
    parser.add_argument("--no-qk-norm", action="store_true", help="Disable QK norms (saves 2*head_dim params)")
    parser.add_argument("--gelu", action="store_true", help="Use GELU MLP instead of SwiGLU (saves d*ff params)")
    parser.add_argument("--tie-kv", action="store_true", help="Tie K=V projections (saves d*n_kv_heads*head_dim params)")
    parser.add_argument("--tie-qo", action="store_true", help="Tie o_proj = q_proj.T (saves d*n_heads*head_dim params)")
    parser.add_argument("--tie-gate", action="store_true", help="Tie gate=up in SwiGLU (saves d*ff params)")
    parser.add_argument("--repeats", type=int, default=1, help="Apply block N times (shared weights)")
    parser.add_argument("--share-norms", action="store_true", help="Share all RMSNorm weights (saves 2*d params)")
    parser.add_argument("--share-block-norms", action="store_true", help="Share ln1/ln2 only (saves d params)")
    parser.add_argument("--cosine-lr", action="store_true", help="Cosine LR decay from lr to lr/10")
    parser.add_argument("--warmup", type=int, default=0, help="LR warmup steps")
    parser.add_argument("--eval-interval", type=int, default=2000)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint .pt file")
    parser.add_argument("--test-set", type=str, default=None, help="Path to fixed test set JSON")
    args = parser.parse_args()

    device = torch.device(args.device)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # If resuming, load config from checkpoint
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        cfg = ckpt["config"]
        args.d_model = cfg["d_model"]
        args.n_heads = cfg["n_heads"]
        args.n_kv_heads = cfg["n_kv_heads"]
        args.head_dim = cfg["head_dim"]
        args.ff = cfg["ff"]
        args.rope_theta = cfg["rope_theta"]
        args.no_qk_norm = cfg.get("no_qk_norm", False)
        args.gelu = cfg.get("gelu", False)
        args.tie_kv = cfg.get("tie_kv", False)
        args.tie_qo = cfg.get("tie_qo", False)
        args.tie_gate = cfg.get("tie_gate", False)
        args.repeats = cfg.get("repeats", 1)
        args.share_norms = cfg.get("share_norms", False)
        args.share_block_norms = cfg.get("share_block_norms", False)

    model = Qwen3AdditionModel(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        head_dim=args.head_dim,
        ff=args.ff,
        rope_theta=args.rope_theta,
        qk_norm=not args.no_qk_norm,
        use_swiglu=not args.gelu,
        tie_kv=args.tie_kv,
        tie_qo=args.tie_qo,
        tie_gate=args.tie_gate,
        repeats=args.repeats,
        share_norms=args.share_norms,
        share_block_norms=args.share_block_norms,
    ).to(device)

    if args.resume:
        model.load_state_dict(ckpt["state_dict"])
        print(f"Resumed from {args.resume} (step {ckpt.get('step', '?')}, acc {ckpt.get('accuracy', '?')})")

    n_params = count_params(model)
    print(f"{'='*70}")
    print(f"Qwen3 Addition: d={args.d_model}, ff={args.ff}, heads={args.n_heads}/"
          f"{args.n_kv_heads}kv, hd={args.head_dim}, theta={args.rope_theta}")
    print(f"Params: {n_params}")
    print(f"LR={args.lr}, batch={args.batch_size}, steps={args.steps}, seed={args.seed}")
    print(f"Curriculum: {args.curriculum}")
    print(f"{'='*70}")

    wandb_run = None
    if args.wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project="minimal-10digit-transformer",
                name=f"qwen3_d{args.d_model}_ff{args.ff}_{n_params}p_s{args.seed}",
                config=vars(args) | {"n_params": n_params},
                tags=["qwen3", "rope"],
                reinit=True,
            )
        except Exception:
            pass

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best_seq_acc = 0.0
    t0 = time.time()

    ablation = ""
    if args.no_qk_norm:
        ablation += "_noqkn"
    if args.gelu:
        ablation += "_gelu"
    if args.tie_kv:
        ablation += "_tiekv"
    if args.tie_qo:
        ablation += "_tieqo"
    if args.tie_gate:
        ablation += "_tiegate"
    if args.share_norms:
        ablation += "_shnorm"
    if args.share_block_norms:
        ablation += "_shbnorm"
    rep_str = f"_r{args.repeats}" if args.repeats > 1 else ""
    tag = f"qwen3_d{args.d_model}_ff{args.ff}_{n_params}p{ablation}{rep_str}"
    ckpt_dir = f"checkpoints/{tag}_s{args.seed}"
    import os
    os.makedirs(ckpt_dir, exist_ok=True)

    # Load test set if provided
    test_pairs = load_test_set(args.test_set) if args.test_set else None

    # Training metrics log (CSV for plotting)
    import csv
    metrics_path = f"{ckpt_dir}/metrics.csv"
    metrics_file = open(metrics_path, "w", newline="")
    metrics_writer = csv.writer(metrics_file)
    metrics_writer.writerow(["step", "loss", "lr", "exact_acc", "digit_acc", "elapsed"])

    for step in range(1, args.steps + 1):
        # LR schedule
        if args.cosine_lr or args.warmup > 0:
            if step <= args.warmup:
                lr = args.lr * step / max(args.warmup, 1)
            elif args.cosine_lr:
                progress = (step - args.warmup) / max(args.steps - args.warmup, 1)
                lr = args.lr / 10 + 0.5 * (args.lr - args.lr / 10) * (1 + math.cos(math.pi * progress))
            else:
                lr = args.lr
            for pg in optimizer.param_groups:
                pg["lr"] = lr
        else:
            lr = args.lr

        model.train()
        max_d = get_max_digits(step, args.curriculum)
        full_seq, labels = generate_batch(args.batch_size, device, max_d)

        optimizer.zero_grad()
        loss = train_step(model, full_seq, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 100 == 0:
            elapsed = time.time() - t0
            print(f"  step {step:6d} | loss {loss.item():.4f} | lr {lr:.2e} | {elapsed:.0f}s")
            sys.stdout.flush()
            metrics_writer.writerow([step, f"{loss.item():.6f}", f"{lr:.2e}", "", "", f"{elapsed:.1f}"])
            if step % 1000 == 0:
                metrics_file.flush()
            if wandb_run:
                wandb_run.log({"step": step, "loss": loss.item(), "lr": lr})

        if step % args.eval_interval == 0:
            eval_pairs = test_pairs[:200] if test_pairs else None
            seq_acc, dig_acc = evaluate(model, device, n_samples=200, test_pairs=eval_pairs)
            elapsed = time.time() - t0
            print(f"  EVAL step {step}: exact={seq_acc:.4f} digit={dig_acc:.4f} [{elapsed:.0f}s]")
            sys.stdout.flush()

            if wandb_run:
                wandb_run.log({"step": step, "exact_acc": seq_acc, "digit_acc": dig_acc})

            metrics_writer.writerow([step, f"{loss.item():.6f}", f"{lr:.2e}", f"{seq_acc:.4f}", f"{dig_acc:.4f}", f"{elapsed:.1f}"])
            metrics_file.flush()

            if seq_acc > best_seq_acc:
                best_seq_acc = seq_acc
                torch.save({
                    "state_dict": model.state_dict(),
                    "step": step,
                    "accuracy": seq_acc,
                    "digit_accuracy": dig_acc,
                    "n_params": n_params,
                    "config": vars(args),
                }, f"{ckpt_dir}/best.pt")
                print(f"  ** NEW BEST: {seq_acc:.4f} **")

    metrics_file.close()

    # Final eval with more samples
    seq_acc, dig_acc = evaluate(model, device, n_samples=1000, test_pairs=test_pairs)
    print(f"\nFINAL: exact={seq_acc:.4f} digit={dig_acc:.4f} params={n_params}")
    print(f"Best exact: {best_seq_acc:.4f}")

    # Save final
    torch.save({
        "state_dict": model.state_dict(),
        "step": args.steps,
        "accuracy": seq_acc,
        "digit_accuracy": dig_acc,
        "n_params": n_params,
        "config": vars(args),
    }, f"{ckpt_dir}/final.pt")

    import json
    results = {
        "tag": tag,
        "n_params": n_params,
        "seed": args.seed,
        "best_acc": best_seq_acc,
        "final_acc": seq_acc,
        "final_digit_acc": dig_acc,
        "steps": args.steps,
        "elapsed": time.time() - t0,
    }
    with open(f"experiments/{tag}_s{args.seed}_results.json", "w") as f:
        json.dump(results, f, indent=2)

    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()
