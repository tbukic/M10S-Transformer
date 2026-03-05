"""Multi-layer (shared weights) search on known-working configs.

Tests repeats=2,4,8 on configs that already grok at 1 layer.
Same params — just applies the transformer block N times.

Usage:
    python experiments/multilayer_search.py --seeds 3 --max-steps 100000
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

from minimal10digittransformer.model.qwen3 import Qwen3AdditionModel
from minimal10digittransformer.model.circular_arc import CircularArcQwen3
from minimal10digittransformer.data.addition import generate_batch
from minimal10digittransformer.evaluation.metrics import evaluate


KNOWN_GOOD = [
    # (name_base, arch, model_kwargs, proven at rep=1)
    ("122p_ff3", "qwen3", dict(
        d_model=3, n_heads=1, n_kv_heads=1, head_dim=4, ff=3, rope_theta=3.0,
    ), "100% at 200K steps"),
    ("89p_tieKV_tieQO", "arc", dict(
        d_model=3, n_heads=1, n_kv_heads=1, head_dim=4, ff=2, rope_theta=3.0,
        tie_kv=True, tie_qo=True,
    ), "100% at 100K steps"),
    ("68p_tieQO_shnorm", "arc", dict(
        d_model=3, n_heads=1, n_kv_heads=1, head_dim=4, ff=2, rope_theta=3.0,
        tie_qo=True, share_norms=True,
    ), "97.2% at 100K steps"),
    ("62p_tieKV_tieQO", "arc", dict(
        d_model=3, n_heads=1, n_kv_heads=1, head_dim=4, ff=2, rope_theta=3.0,
        tie_kv=True, tie_qo=True,
    ), "100% at 336K steps (5-stage)"),
    ("95p_arc_ff3", "arc", dict(
        d_model=3, n_heads=1, n_kv_heads=1, head_dim=4, ff=3, rope_theta=3.0,
    ), "100% at 50K steps"),
]


def count_params(model):
    seen = set()
    total = 0
    for p in model.parameters():
        if p.data_ptr() not in seen:
            seen.add(p.data_ptr())
            total += p.numel()
    return total


def build_model(arch, kwargs):
    if arch == "qwen3":
        return Qwen3AdditionModel(**kwargs)
    elif arch == "arc":
        return CircularArcQwen3(**kwargs)
    raise ValueError(f"Unknown arch: {arch}")


def train_one(name, arch, model_kwargs, seed, max_steps, eval_interval, device, ckpt_dir):
    torch.manual_seed(seed)
    random.seed(seed)

    model = build_model(arch, model_kwargs)
    model = model.to(device)
    n_params = count_params(model)
    repeats = model_kwargs.get("repeats", 1)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps, eta_min=0.001)

    best_acc = 0.0
    best_step = 0
    loss_history = []
    start_time = time.time()

    for step in range(1, max_steps + 1):
        model.train()
        batch, labels = generate_batch(128, device)
        logits = model(batch)
        shift_logits = logits[:, :-1, :].reshape(-1, logits.size(-1))
        shift_labels = labels[:, 1:].reshape(-1)
        loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % eval_interval == 0 or step == max_steps:
            model.eval()
            with torch.no_grad():
                acc, _ = evaluate(model, device, n_samples=500)

            elapsed = time.time() - start_time
            loss_val = loss.item()
            loss_history.append((step, loss_val, acc))
            print(f"  [{name} rep={repeats} s{seed}] step {step}/{max_steps} "
                  f"loss={loss_val:.4f} acc={acc:.1%} best={best_acc:.1%} "
                  f"[{elapsed:.0f}s]", flush=True)

            if acc > best_acc:
                best_acc = acc
                best_step = step
                if acc > 0.10 and ckpt_dir:
                    path = os.path.join(ckpt_dir, f"{name}_rep{repeats}_s{seed}")
                    os.makedirs(path, exist_ok=True)
                    torch.save({
                        "state_dict": model.state_dict(),
                        "config": model_kwargs,
                        "arch": arch,
                        "step": step,
                        "acc": acc,
                        "seed": seed,
                    }, os.path.join(path, "best.pt"))

    elapsed = time.time() - start_time
    return {
        "config_name": name,
        "arch": arch,
        "repeats": repeats,
        "n_params": n_params,
        "seed": seed,
        "best_exact_acc": best_acc,
        "best_step": best_step,
        "final_loss": loss.item(),
        "final_step": max_steps,
        "elapsed_s": elapsed,
        "loss_at_10k": next((l for s, l, a in loss_history if s == 10000), ""),
        "loss_at_50k": next((l for s, l, a in loss_history if s == 50000), ""),
        "loss_at_100k": next((l for s, l, a in loss_history if s == 100000), ""),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=100000)
    parser.add_argument("--eval-interval", type=int, default=5000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="experiments/multilayer_results.csv")
    parser.add_argument("--ckpt-dir", default="checkpoints/multilayer")
    args = parser.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", "4")

    seed_list = [1, 8, 15][:args.seeds]
    repeat_list = [2, 4, 8]

    # Load completed
    completed = set()
    if os.path.exists(args.output):
        with open(args.output) as f:
            for row in csv.DictReader(f):
                completed.add((row["config_name"], int(row["repeats"]), int(row["seed"])))

    # Build run list
    runs = []
    for name, arch, base_kwargs, note in KNOWN_GOOD:
        for rep in repeat_list:
            kwargs = {**base_kwargs, "repeats": rep}
            model = build_model(arch, kwargs)
            n_params = count_params(model)
            for seed in seed_list:
                if (name, rep, seed) not in completed:
                    runs.append((name, arch, kwargs, n_params, rep, seed, note))

    print(f"Multi-layer search: {len(KNOWN_GOOD)} configs × {len(repeat_list)} repeats × {len(seed_list)} seeds")
    print(f"Remaining: {len(runs)} runs, {len(completed)} already done")
    print()
    for name, arch, _, _, rep, seed, note in runs[:15]:
        print(f"  {name} rep={rep} seed={seed}")
    if len(runs) > 15:
        print(f"  ... and {len(runs)-15} more")
    print()

    fieldnames = ["config_name", "arch", "repeats", "n_params", "seed",
                  "best_exact_acc", "best_step", "final_loss", "final_step",
                  "elapsed_s", "loss_at_10k", "loss_at_50k", "loss_at_100k"]

    write_header = not os.path.exists(args.output) or os.path.getsize(args.output) == 0
    done = 0

    for name, arch, kwargs, n_params, rep, seed, note in runs:
        done += 1
        print(f"\n[{done}/{len(runs)}] {name} rep={rep} ({n_params}p) seed={seed}  (1-layer: {note})")
        result = train_one(name, arch, kwargs, seed, args.max_steps, args.eval_interval,
                          args.device, args.ckpt_dir)

        with open(args.output, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
                write_header = False
            writer.writerow(result)

        print(f"  -> {result['best_exact_acc']:.1%} (step {result['best_step']})")

    print(f"\nDone. Results in {args.output}")


if __name__ == "__main__":
    main()
