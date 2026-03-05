"""Quick LR/batch sweep for d=2 models.

Tests whether d=2 models are dead due to architecture or just bad hyperparameters.

Current defaults: lr=0.01 cosine→0.001, batch=128
Try: {0.1, 0.03, 0.01, 0.003, 0.001} × {32, 128, 512}

Usage:
    python experiments/d2_lr_sweep.py
"""

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

from minimal10digittransformer.model.circular_arc import CircularArcQwen3
from minimal10digittransformer.data.addition import generate_batch
from minimal10digittransformer.evaluation.metrics import evaluate
from minimal10digittransformer.model.qwen3 import VOCAB_SIZE


def count_params(model):
    seen = set()
    total = 0
    for p in model.parameters():
        if p.data_ptr() not in seen:
            seen.add(p.data_ptr())
            total += p.numel()
    return total


def train_one(cfg, lr, batch_size, seed, max_steps, eval_interval, device):
    torch.manual_seed(seed)
    random.seed(seed)

    model = CircularArcQwen3(**{k: v for k, v in cfg.items() if k != "name"})
    model = model.to(device)
    n_params = count_params(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    best_acc = 0.0
    best_step = 0
    history = []  # (step, loss, acc)
    start_time = time.time()

    for step in range(1, max_steps + 1):
        # Cosine LR: lr -> lr/10
        progress = step / max(max_steps, 1)
        cur_lr = (lr / 10) + 0.5 * (lr - lr / 10) * (1 + math.cos(math.pi * progress))
        for pg in optimizer.param_groups:
            pg["lr"] = cur_lr

        model.train()
        batch, labels = generate_batch(batch_size, device)
        logits = model(batch)
        shift_logits = logits[:, :-1, :].reshape(-1, VOCAB_SIZE)
        shift_labels = labels[:, 1:].reshape(-1)
        loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % eval_interval == 0 or step == max_steps:
            model.eval()
            with torch.no_grad():
                acc, _ = evaluate(model, device, n_samples=500)

            history.append((step, loss.item(), acc))

            if acc > best_acc:
                best_acc = acc
                best_step = step

    elapsed = time.time() - start_time
    return {
        "n_params": n_params,
        "best_acc": best_acc,
        "best_step": best_step,
        "final_loss": loss.item(),
        "elapsed": elapsed,
        "history": history,
    }


# Configs to test
CONFIGS = [
    {"name": "d2_ff2_tieKV_tieQO", "d_model": 2, "ff": 2,
     "n_heads": 1, "n_kv_heads": 1, "head_dim": 4, "rope_theta": 3.0,
     "tie_kv": True, "tie_qo": True},
    {"name": "d2_ff2_tieQO_shnorm", "d_model": 2, "ff": 2,
     "n_heads": 1, "n_kv_heads": 1, "head_dim": 4, "rope_theta": 3.0,
     "tie_qo": True, "share_norms": True},
    {"name": "d2_ff2", "d_model": 2, "ff": 2,
     "n_heads": 1, "n_kv_heads": 1, "head_dim": 4, "rope_theta": 3.0},
]

LRS = [0.1, 0.03, 0.01, 0.003, 0.001]
BATCH_SIZES = [32, 128, 512]
SEED = 1
MAX_STEPS = 20000
EVAL_INTERVAL = 1000


def main():
    device = "cpu"
    os.environ.setdefault("OMP_NUM_THREADS", "4")

    output = "experiments/d2_lr_sweep_results.csv"
    log_dir = "experiments/d2_lr_sweep_logs"
    os.makedirs(log_dir, exist_ok=True)

    # Load completed
    completed = set()
    if os.path.exists(output):
        with open(output) as f:
            for row in csv.DictReader(f):
                completed.add((row["config"], row["lr"], row["batch_size"]))

    total = len(CONFIGS) * len(LRS) * len(BATCH_SIZES)
    remaining = total - len(completed)
    print(f"d=2 LR/batch sweep: {len(CONFIGS)} configs × {len(LRS)} LRs × {len(BATCH_SIZES)} batches = {total}")
    print(f"Remaining: {remaining}, {len(completed)} done")
    print()

    fieldnames = ["config", "n_params", "lr", "batch_size", "seed",
                  "best_acc", "best_step", "final_loss", "elapsed"]

    write_header = not os.path.exists(output) or os.path.getsize(output) == 0
    done = 0

    for cfg in CONFIGS:
        for lr in LRS:
            for bs in BATCH_SIZES:
                key = (cfg["name"], str(lr), str(bs))
                if key in completed:
                    continue

                done += 1
                print(f"[{done}/{remaining}] {cfg['name']} lr={lr} bs={bs}")

                result = train_one(cfg, lr, bs, SEED, MAX_STEPS, EVAL_INTERVAL, device)

                # Save trajectory
                log_path = os.path.join(log_dir, f"{cfg['name']}_lr{lr}_bs{bs}.csv")
                with open(log_path, "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["step", "loss", "acc"])
                    for step, loss, acc in result["history"]:
                        w.writerow([step, f"{loss:.6f}", f"{acc:.6f}"])

                # Append to main results
                with open(output, "a", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=fieldnames)
                    if write_header:
                        w.writeheader()
                        write_header = False
                    w.writerow({
                        "config": cfg["name"],
                        "n_params": result["n_params"],
                        "lr": lr,
                        "batch_size": bs,
                        "seed": SEED,
                        "best_acc": f"{result['best_acc']:.6f}",
                        "best_step": result["best_step"],
                        "final_loss": f"{result['final_loss']:.6f}",
                        "elapsed": f"{result['elapsed']:.1f}",
                    })

                print(f"  -> acc={result['best_acc']:.1%} loss={result['final_loss']:.4f} "
                      f"[{result['elapsed']:.0f}s]")

    print(f"\nDone. Results in {output}")
    print(f"Trajectories in {log_dir}/")


if __name__ == "__main__":
    main()
