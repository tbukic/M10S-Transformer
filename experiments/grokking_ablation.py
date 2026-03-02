"""Grokking vs EMA/SWA ablation experiment.

Key question: Does EMA/SWA accelerate or enable grokking during from-scratch training?

Experiment design:
1. Train 122p from scratch with seeds that ARE known to grok (0, 6, 8, 9)
   and seeds that DON'T grok (1, 2, 3, 4, 5)
2. For each seed, run 3 conditions:
   a) Standard training (baseline)
   b) EMA tracking (eval EMA model at each checkpoint)
   c) SWA tracking (eval averaged model at each checkpoint)
3. Compare: does EMA/SWA grok earlier or more reliably?

Also test on 89p (harder to grok) to see if EMA/SWA helps.

Usage:
  python experiments/grokking_ablation.py --config 122p --seeds 0,3 --steps 25000
  python experiments/grokking_ablation.py --config 89p --seeds 0,5 --steps 50000
"""

import argparse
import csv
import math
import os
import random
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from minimal10digittransformer.model.qwen3 import (
    Qwen3AdditionModel, VOCAB_SIZE,
)
from minimal10digittransformer.data.addition import generate_batch, load_test_set
from minimal10digittransformer.evaluation.metrics import evaluate


CONFIGS = {
    "122p": {"d_model": 3, "ff": 3, "n_heads": 1, "n_kv_heads": 1, "head_dim": 4,
             "rope_theta": 3.0, "tie_kv": False, "tie_qo": False},
    "89p": {"d_model": 3, "ff": 2, "n_heads": 1, "n_kv_heads": 1, "head_dim": 4,
            "rope_theta": 3.0, "tie_kv": True, "tie_qo": True},
    "101p": {"d_model": 3, "ff": 2, "n_heads": 1, "n_kv_heads": 1, "head_dim": 4,
             "rope_theta": 3.0, "tie_kv": False, "tie_qo": True},
}


def train_step(model, full_seq, labels):
    logits = model(full_seq)
    shift_logits = logits[:, :-1, :].reshape(-1, VOCAB_SIZE)
    shift_labels = labels[:, 1:].reshape(-1)
    return F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)


def run_grokking_ablation(config_name, seed, steps, lr, batch_size, eval_interval,
                          ema_decay, swa_start, swa_interval, test_pairs, device):
    """Run a single seed with EMA and SWA tracking."""
    cfg = CONFIGS[config_name]
    random.seed(seed)
    torch.manual_seed(seed)

    model = Qwen3AdditionModel(
        d_model=cfg["d_model"], n_heads=cfg["n_heads"],
        n_kv_heads=cfg["n_kv_heads"], head_dim=cfg["head_dim"],
        ff=cfg["ff"], rope_theta=cfg["rope_theta"],
        tie_kv=cfg.get("tie_kv", False), tie_qo=cfg.get("tie_qo", False),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # EMA shadow
    ema_shadow = {name: param.data.clone() for name, param in model.named_parameters()}

    # SWA accumulator
    swa_state = None
    swa_count = 0

    # Results CSV
    out_dir = f"plots/grokking_ablation/{config_name}_s{seed}"
    os.makedirs(out_dir, exist_ok=True)
    csv_path = f"{out_dir}/metrics.csv"
    csv_file = open(csv_path, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["step", "loss", "lr", "base_acc", "ema_acc", "swa_acc", "swa_count"])

    print(f"\n{'='*60}")
    print(f"{config_name} seed={seed} ({n_params}p)")
    print(f"Steps: {steps}, EMA decay: {ema_decay}")
    print(f"{'='*60}")

    t0 = time.time()

    for step in range(1, steps + 1):
        # Cosine LR
        progress = step / steps
        cur_lr = lr / 10 + 0.5 * (lr - lr / 10) * (1 + math.cos(math.pi * progress))
        for pg in optimizer.param_groups:
            pg["lr"] = cur_lr

        model.train()
        full_seq, labels = generate_batch(batch_size, device, max_digits=10)
        optimizer.zero_grad()
        loss = train_step(model, full_seq, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Update EMA
        for name, param in model.named_parameters():
            ema_shadow[name].mul_(ema_decay).add_(param.data, alpha=1 - ema_decay)

        # Update SWA
        if step >= swa_start and step % swa_interval == 0:
            if swa_state is None:
                swa_state = {name: param.data.clone() for name, param in model.named_parameters()}
                swa_count = 1
            else:
                for name, param in model.named_parameters():
                    swa_state[name] += param.data
                swa_count += 1

        # Log loss every 100 steps
        if step % 100 == 0:
            writer.writerow([step, f"{loss.item():.6f}", f"{cur_lr:.2e}", "", "", "", swa_count])

        # Eval every eval_interval steps
        if step % eval_interval == 0:
            eval_pairs = test_pairs[:200]

            # Base model
            base_acc, _ = evaluate(model, device, test_pairs=eval_pairs)

            # EMA model
            backup = {}
            for name, param in model.named_parameters():
                backup[name] = param.data.clone()
                param.data.copy_(ema_shadow[name])
            ema_acc, _ = evaluate(model, device, test_pairs=eval_pairs)
            for name, param in model.named_parameters():
                param.data.copy_(backup[name])

            # SWA model
            swa_acc = 0.0
            if swa_state is not None and swa_count > 0:
                backup = {}
                for name, param in model.named_parameters():
                    backup[name] = param.data.clone()
                    param.data.copy_(swa_state[name] / swa_count)
                swa_acc, _ = evaluate(model, device, test_pairs=eval_pairs)
                for name, param in model.named_parameters():
                    param.data.copy_(backup[name])

            elapsed = time.time() - t0
            writer.writerow([step, f"{loss.item():.6f}", f"{cur_lr:.2e}",
                             f"{base_acc:.4f}", f"{ema_acc:.4f}", f"{swa_acc:.4f}", swa_count])
            csv_file.flush()

            print(f"  step {step:6d} | base={base_acc:.4f} ema={ema_acc:.4f} "
                  f"swa={swa_acc:.4f} (n={swa_count}) | {elapsed:.0f}s")
            sys.stdout.flush()

    csv_file.close()

    # Final full eval on test set
    base_acc, _ = evaluate(model, device, test_pairs=test_pairs)
    backup = {}
    for name, param in model.named_parameters():
        backup[name] = param.data.clone()
        param.data.copy_(ema_shadow[name])
    ema_acc, _ = evaluate(model, device, test_pairs=test_pairs)
    for name, param in model.named_parameters():
        param.data.copy_(backup[name])

    swa_acc = 0.0
    if swa_state is not None and swa_count > 0:
        backup = {}
        for name, param in model.named_parameters():
            backup[name] = param.data.clone()
            param.data.copy_(swa_state[name] / swa_count)
        swa_acc, _ = evaluate(model, device, test_pairs=test_pairs)
        for name, param in model.named_parameters():
            param.data.copy_(backup[name])

    n_test = len(test_pairs)
    print(f"\nFINAL ({n_test} samples):")
    print(f"  Base: {base_acc:.4f} ({int(round((1-base_acc)*n_test))} errors)")
    print(f"  EMA:  {ema_acc:.4f} ({int(round((1-ema_acc)*n_test))} errors)")
    print(f"  SWA:  {swa_acc:.4f} ({int(round((1-swa_acc)*n_test))} errors, n={swa_count})")

    return {
        "config": config_name, "seed": seed, "n_params": n_params,
        "final_base": base_acc, "final_ema": ema_acc, "final_swa": swa_acc,
        "csv_path": csv_path,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="122p", choices=list(CONFIGS.keys()))
    parser.add_argument("--seeds", default="0,3", help="Comma-separated seeds")
    parser.add_argument("--steps", type=int, default=25000)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--swa-start", type=int, default=5000)
    parser.add_argument("--swa-interval", type=int, default=200)
    parser.add_argument("--test-set", type=str, default="data/test_10k.json")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    test_pairs = load_test_set(args.test_set)
    device = torch.device(args.device)

    print(f"Grokking vs EMA/SWA Ablation: {args.config}")
    print(f"Seeds: {seeds}, Steps: {args.steps}")
    print(f"EMA decay: {args.ema_decay}, SWA start: {args.swa_start}")

    results = []
    for seed in seeds:
        r = run_grokking_ablation(
            args.config, seed, args.steps, args.lr, args.batch_size,
            args.eval_interval, args.ema_decay, args.swa_start,
            args.swa_interval, test_pairs, device,
        )
        results.append(r)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for r in results:
        print(f"  {r['config']} s{r['seed']}: base={r['final_base']:.4f} "
              f"ema={r['final_ema']:.4f} swa={r['final_swa']:.4f}")

    import json
    with open(f"experiments/grokking_ablation_{args.config}_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
