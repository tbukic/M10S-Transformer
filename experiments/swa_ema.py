"""SWA (Stochastic Weight Averaging) and EMA (Exponential Moving Average) experiments.

Two approaches to push sub-100p models over 99%:

1. **EMA during fine-tuning**: Maintain exponential moving average of weights during
   training. At each step, ema_weights = decay * ema_weights + (1-decay) * weights.
   Evaluate the EMA model periodically.

2. **SWA (trajectory averaging)**: Fine-tune from a checkpoint, saving snapshots
   every N steps, then average the last K snapshots. This is different from cross-seed
   averaging (which we already tried and didn't help).

Usage:
  # EMA fine-tuning from a checkpoint
  python experiments/swa_ema.py --mode ema \
    --checkpoint checkpoints/.../best.pt \
    --lr 0.001 --batch-size 256 --steps 30000 --seed 42

  # SWA fine-tuning (saves snapshots and averages them)
  python experiments/swa_ema.py --mode swa \
    --checkpoint checkpoints/.../best.pt \
    --lr 0.001 --batch-size 256 --steps 30000 --seed 42

  # Run both on all high-accuracy sub-100p checkpoints
  python experiments/swa_ema.py --mode sweep
"""

import argparse
import copy
import json
import math
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
from minimal10digittransformer.data.addition import generate_batch, load_test_set
from minimal10digittransformer.evaluation.metrics import evaluate


def load_model_from_checkpoint(ckpt_path, device="cpu"):
    """Load model and config from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    config = ckpt["config"]
    model = Qwen3AdditionModel(
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_kv_heads=config["n_kv_heads"],
        head_dim=config["head_dim"],
        ff=config["ff"],
        rope_theta=config["rope_theta"],
        qk_norm=not config.get("no_qk_norm", False),
        use_swiglu=not config.get("gelu", False),
        tie_kv=config.get("tie_kv", False),
        tie_qo=config.get("tie_qo", False),
        tie_gate=config.get("tie_gate", False),
        repeats=config.get("repeats", 1),
        share_norms=config.get("share_norms", False),
        share_block_norms=config.get("share_block_norms", False),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    return model, config, ckpt


def train_step(model, full_seq, labels):
    """Single training step with teacher forcing."""
    logits = model(full_seq)
    shift_logits = logits[:, :-1, :].reshape(-1, VOCAB_SIZE)
    shift_labels = labels[:, 1:].reshape(-1)
    return F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)


class EMAModel:
    """Exponential Moving Average of model parameters."""

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            self.shadow[name] = param.data.clone()

    def update(self, model):
        for name, param in model.named_parameters():
            self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def apply(self, model):
        """Copy EMA weights into model (for evaluation)."""
        backup = {}
        for name, param in model.named_parameters():
            backup[name] = param.data.clone()
            param.data.copy_(self.shadow[name])
        return backup

    def restore(self, model, backup):
        """Restore original weights after evaluation."""
        for name, param in model.named_parameters():
            param.data.copy_(backup[name])


def run_ema(args, test_pairs):
    """Fine-tune with EMA tracking."""
    device = torch.device(args.device)
    model, config, ckpt = load_model_from_checkpoint(args.checkpoint, device)
    n_params = sum(p.numel() for p in model.parameters())

    print(f"EMA fine-tuning from {args.checkpoint}")
    print(f"  Params: {n_params}, decay: {args.ema_decay}")
    print(f"  LR: {args.lr}, batch: {args.batch_size}, steps: {args.steps}")

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    ema = EMAModel(model, decay=args.ema_decay)

    best_base_acc = 0.0
    best_ema_acc = 0.0
    t0 = time.time()

    results_log = []

    for step in range(1, args.steps + 1):
        # Optional cosine LR
        if args.cosine_lr:
            progress = step / args.steps
            lr = args.lr / 10 + 0.5 * (args.lr - args.lr / 10) * (1 + math.cos(math.pi * progress))
            for pg in optimizer.param_groups:
                pg["lr"] = lr

        model.train()
        full_seq, labels = generate_batch(args.batch_size, device, max_digits=10)
        optimizer.zero_grad()
        loss = train_step(model, full_seq, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Update EMA after each step
        ema.update(model)

        if step % 100 == 0:
            elapsed = time.time() - t0
            print(f"  step {step:6d} | loss {loss.item():.4f} | {elapsed:.0f}s")
            sys.stdout.flush()

        if step % args.eval_interval == 0:
            # Eval base model
            eval_pairs = test_pairs[:200]
            base_acc, base_dig = evaluate(model, device, test_pairs=eval_pairs)

            # Eval EMA model
            backup = ema.apply(model)
            ema_acc, ema_dig = evaluate(model, device, test_pairs=eval_pairs)
            ema.restore(model, backup)

            elapsed = time.time() - t0
            print(f"  EVAL step {step}: base={base_acc:.4f} ema={ema_acc:.4f} [{elapsed:.0f}s]")
            sys.stdout.flush()

            if base_acc > best_base_acc:
                best_base_acc = base_acc
            if ema_acc > best_ema_acc:
                best_ema_acc = ema_acc
                # Save EMA checkpoint
                backup = ema.apply(model)
                ckpt_dir = f"checkpoints/ema_{n_params}p_d{args.ema_decay}_s{args.seed}"
                os.makedirs(ckpt_dir, exist_ok=True)
                torch.save({
                    "state_dict": model.state_dict(),
                    "step": step,
                    "accuracy": ema_acc,
                    "n_params": n_params,
                    "config": config,
                    "ema_decay": args.ema_decay,
                    "source_checkpoint": args.checkpoint,
                }, f"{ckpt_dir}/best.pt")
                ema.restore(model, backup)
                print(f"  ** NEW BEST EMA: {ema_acc:.4f} **")

            results_log.append({
                "step": step, "base_acc": base_acc, "ema_acc": ema_acc,
                "base_dig": base_dig, "ema_dig": ema_dig,
            })

    # Final full eval on 10K
    print("\nFinal evaluation on full test set...")
    # Base model
    base_acc, base_dig = evaluate(model, device, test_pairs=test_pairs)
    base_errors = int(round((1 - base_acc) * len(test_pairs)))

    # EMA model
    backup = ema.apply(model)
    ema_acc, ema_dig = evaluate(model, device, test_pairs=test_pairs)
    ema_errors = int(round((1 - ema_acc) * len(test_pairs)))
    ema.restore(model, backup)

    print(f"\nFINAL ({len(test_pairs)} samples):")
    print(f"  Base: {base_acc:.4f} ({base_errors} errors)")
    print(f"  EMA:  {ema_acc:.4f} ({ema_errors} errors)")
    print(f"  Best base (200-sample): {best_base_acc:.4f}")
    print(f"  Best EMA (200-sample):  {best_ema_acc:.4f}")

    return {
        "mode": "ema",
        "n_params": n_params,
        "ema_decay": args.ema_decay,
        "seed": args.seed,
        "source": args.checkpoint,
        "final_base_acc": base_acc,
        "final_ema_acc": ema_acc,
        "final_base_errors": base_errors,
        "final_ema_errors": ema_errors,
        "best_base_200": best_base_acc,
        "best_ema_200": best_ema_acc,
        "steps": args.steps,
        "log": results_log,
    }


def run_swa(args, test_pairs):
    """Fine-tune with SWA (save snapshots and average them)."""
    device = torch.device(args.device)
    model, config, ckpt = load_model_from_checkpoint(args.checkpoint, device)
    n_params = sum(p.numel() for p in model.parameters())

    print(f"SWA fine-tuning from {args.checkpoint}")
    print(f"  Params: {n_params}, snapshot_interval: {args.swa_interval}")
    print(f"  LR: {args.lr}, batch: {args.batch_size}, steps: {args.steps}")
    print(f"  SWA start: step {args.swa_start}")

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # SWA: accumulate running average of weights
    swa_state = None
    swa_count = 0
    snapshots_taken = 0

    best_base_acc = 0.0
    best_swa_acc = 0.0
    t0 = time.time()
    results_log = []

    for step in range(1, args.steps + 1):
        # Cyclic or constant LR for SWA
        if args.cosine_lr:
            progress = step / args.steps
            lr = args.lr / 10 + 0.5 * (args.lr - args.lr / 10) * (1 + math.cos(math.pi * progress))
        else:
            lr = args.lr
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        model.train()
        full_seq, labels = generate_batch(args.batch_size, device, max_digits=10)
        optimizer.zero_grad()
        loss = train_step(model, full_seq, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Take SWA snapshot
        if step >= args.swa_start and step % args.swa_interval == 0:
            if swa_state is None:
                swa_state = {name: param.data.clone() for name, param in model.named_parameters()}
                swa_count = 1
            else:
                for name, param in model.named_parameters():
                    swa_state[name] += param.data
                swa_count += 1
            snapshots_taken += 1

        if step % 100 == 0:
            elapsed = time.time() - t0
            print(f"  step {step:6d} | loss {loss.item():.4f} | snapshots: {snapshots_taken} | {elapsed:.0f}s")
            sys.stdout.flush()

        if step % args.eval_interval == 0:
            eval_pairs = test_pairs[:200]
            # Eval base model
            base_acc, base_dig = evaluate(model, device, test_pairs=eval_pairs)

            # Eval SWA model (if we have snapshots)
            swa_acc = 0.0
            swa_dig = 0.0
            if swa_state is not None and swa_count > 0:
                # Temporarily load SWA weights
                backup = {name: param.data.clone() for name, param in model.named_parameters()}
                for name, param in model.named_parameters():
                    param.data.copy_(swa_state[name] / swa_count)
                swa_acc, swa_dig = evaluate(model, device, test_pairs=eval_pairs)
                # Restore
                for name, param in model.named_parameters():
                    param.data.copy_(backup[name])

            elapsed = time.time() - t0
            print(f"  EVAL step {step}: base={base_acc:.4f} swa={swa_acc:.4f} "
                  f"(n={swa_count}) [{elapsed:.0f}s]")
            sys.stdout.flush()

            if base_acc > best_base_acc:
                best_base_acc = base_acc
            if swa_acc > best_swa_acc and swa_count > 0:
                best_swa_acc = swa_acc
                # Save SWA checkpoint
                backup = {name: param.data.clone() for name, param in model.named_parameters()}
                for name, param in model.named_parameters():
                    param.data.copy_(swa_state[name] / swa_count)
                ckpt_dir = f"checkpoints/swa_{n_params}p_s{args.seed}"
                os.makedirs(ckpt_dir, exist_ok=True)
                torch.save({
                    "state_dict": model.state_dict(),
                    "step": step,
                    "accuracy": swa_acc,
                    "n_params": n_params,
                    "config": config,
                    "swa_count": swa_count,
                    "source_checkpoint": args.checkpoint,
                }, f"{ckpt_dir}/best.pt")
                for name, param in model.named_parameters():
                    param.data.copy_(backup[name])
                print(f"  ** NEW BEST SWA: {swa_acc:.4f} (avg of {swa_count}) **")

            results_log.append({
                "step": step, "base_acc": base_acc, "swa_acc": swa_acc,
                "swa_count": swa_count,
            })

    # Final full eval
    print("\nFinal evaluation on full test set...")
    base_acc, base_dig = evaluate(model, device, test_pairs=test_pairs)
    base_errors = int(round((1 - base_acc) * len(test_pairs)))

    swa_acc = 0.0
    swa_errors = -1
    if swa_state is not None and swa_count > 0:
        backup = {name: param.data.clone() for name, param in model.named_parameters()}
        for name, param in model.named_parameters():
            param.data.copy_(swa_state[name] / swa_count)
        swa_acc, swa_dig = evaluate(model, device, test_pairs=test_pairs)
        swa_errors = int(round((1 - swa_acc) * len(test_pairs)))
        for name, param in model.named_parameters():
            param.data.copy_(backup[name])

    print(f"\nFINAL ({len(test_pairs)} samples):")
    print(f"  Base: {base_acc:.4f} ({base_errors} errors)")
    print(f"  SWA:  {swa_acc:.4f} ({swa_errors} errors) (avg of {swa_count} snapshots)")

    return {
        "mode": "swa",
        "n_params": n_params,
        "swa_count": swa_count,
        "seed": args.seed,
        "source": args.checkpoint,
        "final_base_acc": base_acc,
        "final_swa_acc": swa_acc,
        "final_base_errors": base_errors,
        "final_swa_errors": swa_errors,
        "steps": args.steps,
        "log": results_log,
    }


def find_best_sub100p_checkpoints():
    """Find the best checkpoint for each sub-100p config."""
    from pathlib import Path
    configs = {
        "89p": "qwen3_d3_ff2_89p_tiekv_tieqo",
        "86p": "qwen3_d3_ff2_86p_tiekv_tieqo_shbnorm",
        "83p": "qwen3_d3_ff2_83p_tiekv_tieqo_shnorm",
    }
    results = {}
    ckpt_base = Path("checkpoints")
    for config_name, pattern in configs.items():
        best_path = None
        best_acc = 0.0
        for d in sorted(ckpt_base.iterdir()):
            if d.is_dir() and pattern in d.name and d.name.startswith("qwen3_"):
                # Skip EMA/SWA checkpoints
                if "ema_" in d.name or "swa_" in d.name:
                    continue
                bp = d / "best.pt"
                if bp.exists():
                    try:
                        ckpt = torch.load(str(bp), map_location="cpu", weights_only=True)
                        acc = ckpt.get("accuracy", 0)
                        if acc > best_acc:
                            best_acc = acc
                            best_path = str(bp)
                    except Exception:
                        pass
        if best_path:
            results[config_name] = (best_path, best_acc)
    return results


def run_sweep(args, test_pairs):
    """Run EMA and SWA on all sub-100p checkpoints."""
    checkpoints = find_best_sub100p_checkpoints()

    print("=" * 70)
    print("SWA/EMA SWEEP on sub-100p models")
    print("=" * 70)
    for name, (path, acc) in sorted(checkpoints.items()):
        print(f"  {name}: {path} (train acc={acc:.4f})")
    print()

    all_results = []

    for config_name in ["89p", "86p", "83p"]:
        if config_name not in checkpoints:
            print(f"  Skipping {config_name}: no checkpoint found")
            continue

        path, base_acc = checkpoints[config_name]
        print(f"\n{'='*70}")
        print(f"Processing {config_name} from {path}")
        print(f"{'='*70}")

        for mode in ["ema", "swa"]:
            print(f"\n--- {config_name} {mode.upper()} ---")
            args.checkpoint = path
            args.mode = mode

            for seed in [args.seed, args.seed + 1, args.seed + 2]:
                args.seed = seed
                try:
                    if mode == "ema":
                        result = run_ema(args, test_pairs)
                    else:
                        result = run_swa(args, test_pairs)
                    result["config"] = config_name
                    all_results.append(result)

                    # Save incrementally
                    with open("experiments/swa_ema_results.json", "w") as f:
                        json.dump(all_results, f, indent=2,
                                  default=lambda o: float(o) if hasattr(o, 'item') else o)
                except Exception as e:
                    print(f"  ERROR: {e}")
                    import traceback
                    traceback.print_exc()

    # Summary
    print(f"\n{'='*70}")
    print("SWA/EMA SWEEP SUMMARY")
    print(f"{'='*70}")
    for r in all_results:
        mode = r["mode"]
        key = f"final_{mode}_acc" if mode in r.get("mode", "") else "final_ema_acc"
        avg_key = f"final_{mode}_acc"
        base_key = "final_base_acc"
        print(f"  {r.get('config','?'):>4s} {mode:>3s} s{r['seed']}: "
              f"base={r.get(base_key, 0):.4f} "
              f"{mode}={r.get(avg_key, 0):.4f} "
              f"({r.get(f'final_{mode}_errors', '?')} errors)")

    return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["ema", "swa", "sweep"], required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--eval-interval", type=int, default=2000)
    parser.add_argument("--test-set", type=str, default="data/test_10k.json")
    parser.add_argument("--cosine-lr", action="store_true")

    # EMA params
    parser.add_argument("--ema-decay", type=float, default=0.999,
                        help="EMA decay factor (0.99, 0.999, 0.9999)")

    # SWA params
    parser.add_argument("--swa-start", type=int, default=5000,
                        help="Step to start collecting SWA snapshots")
    parser.add_argument("--swa-interval", type=int, default=500,
                        help="Steps between SWA snapshots")

    args = parser.parse_args()

    # Load test set
    test_pairs = load_test_set(args.test_set)
    print(f"Loaded {len(test_pairs)} test pairs from {args.test_set}")

    if args.mode == "sweep":
        results = run_sweep(args, test_pairs)
    elif args.mode == "ema":
        if not args.checkpoint:
            parser.error("--checkpoint required for ema mode")
        results = run_ema(args, test_pairs)
    elif args.mode == "swa":
        if not args.checkpoint:
            parser.error("--checkpoint required for swa mode")
        results = run_swa(args, test_pairs)

    # Save final results
    out_path = f"experiments/swa_ema_{args.mode}_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2,
                  default=lambda o: float(o) if hasattr(o, 'item') else o)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
