"""Continue training tying_search models from checkpoint.

Supports cosine and constant LR schedules, plus Grokfast-EMA.

Usage:
    # Cosine continuation from best checkpoint
    python experiments/tying_continue.py --ckpt checkpoints/tying_58p_long/best_s1.pt \
        --phase cosine --lr 0.001 --lr-min 0.0001 --max-steps 800000

    # Const LR from best checkpoint
    python experiments/tying_continue.py --ckpt checkpoints/tying_58p_long/best_s1.pt \
        --phase const --lr 0.0003 --max-steps 200000

    # Grokfast continuation
    python experiments/tying_continue.py --ckpt checkpoints/tying_58p_long/best_s1.pt \
        --phase cosine --lr 0.001 --lr-min 0.0001 --max-steps 400000 --grokfast
"""

import argparse
import csv
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.tying_search import build_model, count_params
from minimal10digittransformer.data.addition import generate_batch, generate_test_set
from minimal10digittransformer.evaluation.metrics import evaluate
from minimal10digittransformer.model.qwen3 import VOCAB_SIZE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint")
    parser.add_argument("--phase", choices=["cosine", "const"], default="cosine")
    parser.add_argument("--max-steps", type=int, default=800000)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--lr-min", type=float, default=0.0001)
    parser.add_argument("--wd", type=float, default=0.01)
    parser.add_argument("--eval-interval", type=int, default=5000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--grokfast", action="store_true")
    parser.add_argument("--gf-alpha", type=float, default=0.98)
    parser.add_argument("--gf-lambda", type=float, default=2.0)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", "2")

    # Load checkpoint
    ckpt = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    cfg = ckpt["config"]
    seed = ckpt["seed"]
    prev_step = ckpt["step"]
    prev_acc = ckpt.get("acc", 0)
    prev_loss = ckpt.get("loss", 0)
    n_params = ckpt.get("n_params", 0)

    print(f"Loaded checkpoint: {args.ckpt}")
    print(f"  Config: {cfg['name']}, seed={seed}, step={prev_step}")
    print(f"  Prev acc={prev_acc:.1%}, loss={prev_loss:.4f}, {n_params}p")

    gf_tag = f"_gf{args.gf_alpha}x{args.gf_lambda}" if args.grokfast else ""
    phase_tag = f"{args.phase}_lr{args.lr}{gf_tag}"
    print(f"  Phase: {phase_tag}, max_steps={args.max_steps}")

    # Output directory
    if args.output_dir:
        out_dir = args.output_dir
    else:
        out_dir = f"checkpoints/tying_continue/{cfg['name']}_s{seed}_{phase_tag}"
    os.makedirs(out_dir, exist_ok=True)

    metrics_dir = "experiments/tying_continue_metrics"
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_path = os.path.join(metrics_dir, f"{cfg['name']}_s{seed}_{phase_tag}.csv")

    # Build model and load state
    model = build_model(cfg, args.device)
    model.load_state_dict(ckpt["state_dict"])
    actual_params = count_params(model)
    print(f"  Model rebuilt: {actual_params}p")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    if args.phase == "cosine" and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
        print("  Loaded optimizer state")
    else:
        print(f"  Fresh optimizer (lr={args.lr}, wd={args.wd})")

    eval_pairs = generate_test_set(2000, seed=12345)

    # Grokfast-EMA
    gf_ema = {}
    if args.grokfast:
        print(f"  Grokfast-EMA: alpha={args.gf_alpha}, lambda={args.gf_lambda}")
        for name, p in model.named_parameters():
            if p.requires_grad:
                gf_ema[name] = torch.zeros_like(p.data)

    best_acc = prev_acc
    best_step = 0
    grok_step = None
    start_time = time.time()

    # Metrics file
    write_header = not os.path.exists(metrics_path) or os.path.getsize(metrics_path) == 0
    metrics_file = open(metrics_path, "a", newline="")
    mw = csv.writer(metrics_file)
    if write_header:
        mw.writerow(["global_step", "phase_step", "loss", "lr", "exact_acc", "elapsed"])

    for step in range(1, args.max_steps + 1):
        global_step = prev_step + step

        # LR schedule
        if args.phase == "cosine":
            progress = step / max(args.max_steps, 1)
            cur_lr = args.lr_min + 0.5 * (args.lr - args.lr_min) * (1 + math.cos(math.pi * progress))
        else:
            cur_lr = args.lr

        for pg in optimizer.param_groups:
            pg["lr"] = cur_lr

        model.train()
        batch, labels = generate_batch(128, args.device)
        logits = model(batch)
        shift_logits = logits[:, :-1, :].reshape(-1, VOCAB_SIZE)
        shift_labels = labels[:, 1:].reshape(-1)
        loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)

        optimizer.zero_grad()
        loss.backward()

        # Grokfast-EMA
        if args.grokfast:
            for name, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    gf_ema[name].mul_(args.gf_alpha).add_(p.grad, alpha=1 - args.gf_alpha)
                    p.grad.add_(gf_ema[name], alpha=args.gf_lambda)

        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Log loss every 1K steps
        if step % 1000 == 0 and step % args.eval_interval != 0:
            elapsed = time.time() - start_time
            mw.writerow([global_step, step, f"{loss.item():.6f}", f"{cur_lr:.6f}", "", f"{elapsed:.1f}"])
            metrics_file.flush()

        if step % args.eval_interval == 0 or step == args.max_steps:
            model.eval()
            with torch.no_grad():
                acc, _ = evaluate(model, device=args.device, test_pairs=eval_pairs[:500])

            elapsed = time.time() - start_time
            print(f"  [{cfg['name']} s{seed} {args.phase}] "
                  f"step {step}/{args.max_steps} (global {global_step}) "
                  f"loss={loss.item():.4f} acc={acc:.1%} best={best_acc:.1%} "
                  f"lr={cur_lr:.6f} [{elapsed:.0f}s]", flush=True)

            mw.writerow([global_step, step, f"{loss.item():.6f}", f"{cur_lr:.6f}",
                         f"{acc:.6f}", f"{elapsed:.1f}"])
            metrics_file.flush()

            if acc > best_acc:
                best_acc = acc
                best_step = global_step

            # Save checkpoints
            ckpt_data = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": cfg,
                "step": global_step,
                "phase_step": step,
                "acc": acc,
                "loss": loss.item(),
                "seed": seed,
                "n_params": actual_params,
                "phase": args.phase,
                "prev_ckpt": args.ckpt,
            }
            torch.save(ckpt_data, os.path.join(out_dir, "latest.pt"))
            if acc >= best_acc and acc > 0:
                torch.save(ckpt_data, os.path.join(out_dir, "best.pt"))

            if grok_step is None and acc > 0.5:
                grok_step = global_step

            if acc >= 0.999:
                print(f"  GROKKED at global step {global_step}!")
                break

    metrics_file.close()
    elapsed = time.time() - start_time
    grok = f"grok@{grok_step}" if grok_step else "no grok"
    print(f"\nDone. best={best_acc:.1%} (step {best_step}) {grok} [{elapsed:.0f}s]")
    print(f"Checkpoints in {out_dir}")
    print(f"Metrics in {metrics_path}")


if __name__ == "__main__":
    main()
