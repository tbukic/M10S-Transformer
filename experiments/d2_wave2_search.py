"""Wave 2 search: ReLU MLP + multi-layer experiments.

Three experiment groups:
1. d=2 ReLU MLP (no gating) — all tying/norm combos
2. d=2 SwiGLU+ReLU (gating with ReLU instead of SiLU) — all tying/norm combos
3. Multi-layer (repeats=2,4,8) on known-converging d=3 configs

Usage:
    python experiments/d2_wave2_search.py --seeds 5 --max-steps 100000
    python experiments/d2_wave2_search.py --group relu_only --seeds 3
    python experiments/d2_wave2_search.py --group multilayer --seeds 3
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

from minimal10digittransformer.model.circular_arc import CircularArcQwen3
from minimal10digittransformer.data.addition import generate_batch
from minimal10digittransformer.evaluation.metrics import evaluate


def generate_configs(group="all"):
    configs = []

    tying_combos = []
    for tie_kv in [False, True]:
        for tie_qo in [False, True]:
            for sn, sbn in [(False, False), (False, True), (True, False)]:
                tying_combos.append((tie_kv, tie_qo, sn, sbn))

    if group in ("all", "relu_only"):
        # Group 1: d=2, ReLU MLP (no gate, 2 matrices)
        for tie_kv, tie_qo, sn, sbn in tying_combos:
            name_parts = ["arc_d2_ff2_relu"]
            if tie_kv: name_parts.append("tieKV")
            if tie_qo: name_parts.append("tieQO")
            if sn: name_parts.append("shnorm")
            if sbn: name_parts.append("shbnorm")
            configs.append({
                "name": "_".join(name_parts),
                "d_model": 2, "ff": 2,
                "n_heads": 1, "n_kv_heads": 1,
                "head_dim": 4, "rope_theta": 3.0,
                "tie_kv": tie_kv, "tie_qo": tie_qo,
                "share_norms": sn, "share_block_norms": sbn,
                "use_swiglu": False, "activation": "relu",
            })

    if group in ("all", "swiglu_relu"):
        # Group 2: d=2, SwiGLU with ReLU activation (3 matrices, but ReLU gate)
        for tie_kv, tie_qo, sn, sbn in tying_combos:
            name_parts = ["arc_d2_ff2_swirelu"]
            if tie_kv: name_parts.append("tieKV")
            if tie_qo: name_parts.append("tieQO")
            if sn: name_parts.append("shnorm")
            if sbn: name_parts.append("shbnorm")
            configs.append({
                "name": "_".join(name_parts),
                "d_model": 2, "ff": 2,
                "n_heads": 1, "n_kv_heads": 1,
                "head_dim": 4, "rope_theta": 3.0,
                "tie_kv": tie_kv, "tie_qo": tie_qo,
                "share_norms": sn, "share_block_norms": sbn,
                "use_swiglu": True, "activation": "relu",
            })

    if group in ("all", "relu_d3"):
        # Group 3: d=3, ReLU MLP (no gate) — compare with known SwiGLU results
        for tie_kv, tie_qo, sn, sbn in tying_combos:
            name_parts = ["arc_d3_ff2_relu"]
            if tie_kv: name_parts.append("tieKV")
            if tie_qo: name_parts.append("tieQO")
            if sn: name_parts.append("shnorm")
            if sbn: name_parts.append("shbnorm")
            configs.append({
                "name": "_".join(name_parts),
                "d_model": 3, "ff": 2,
                "n_heads": 1, "n_kv_heads": 1,
                "head_dim": 4, "rope_theta": 3.0,
                "tie_kv": tie_kv, "tie_qo": tie_qo,
                "share_norms": sn, "share_block_norms": sbn,
                "use_swiglu": False, "activation": "relu",
            })

    if group in ("all", "multilayer"):
        # Group 4: Multi-layer (shared weights) on known-converging configs
        # Test repeats=2,4,8 on configs that grok at d=3
        known_good = [
            # 68p equivalent: ff=2, tieQO, shnorm (97.2% at d=3)
            {"tie_kv": False, "tie_qo": True, "share_norms": True, "share_block_norms": False},
            # 62p equivalent: ff=2, tieKV+tieQO (our submission)
            {"tie_kv": True, "tie_qo": True, "share_norms": False, "share_block_norms": False},
            # 71p equivalent: ff=2, tieQO, shbnorm (97.2% at d=3)
            {"tie_kv": False, "tie_qo": True, "share_norms": False, "share_block_norms": True},
        ]
        for repeats in [2, 4, 8]:
            for base in known_good:
                name_parts = [f"arc_d3_ff2_rep{repeats}"]
                if base["tie_kv"]: name_parts.append("tieKV")
                if base["tie_qo"]: name_parts.append("tieQO")
                if base["share_norms"]: name_parts.append("shnorm")
                if base["share_block_norms"]: name_parts.append("shbnorm")
                configs.append({
                    "name": "_".join(name_parts),
                    "d_model": 3, "ff": 2,
                    "n_heads": 1, "n_kv_heads": 1,
                    "head_dim": 4, "rope_theta": 3.0,
                    "repeats": repeats,
                    **base,
                    "use_swiglu": True, "activation": "default",
                })

    return configs


def count_params(model):
    seen = set()
    total = 0
    for p in model.parameters():
        if p.data_ptr() not in seen:
            seen.add(p.data_ptr())
            total += p.numel()
    return total


def build_model(cfg):
    model_kwargs = {k: v for k, v in cfg.items() if k != "name"}
    return CircularArcQwen3(**model_kwargs)


def train_one(cfg, seed, max_steps, eval_interval, device, ckpt_dir):
    torch.manual_seed(seed)
    random.seed(seed)

    model = build_model(cfg)
    model = model.to(device)
    n_params = count_params(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps, eta_min=0.001)

    best_acc = 0.0
    best_step = 0
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
            print(f"  [{cfg['name']} s{seed}] step {step}/{max_steps} "
                  f"loss={loss.item():.4f} acc={acc:.1%} best={best_acc:.1%} "
                  f"[{elapsed:.0f}s]", flush=True)

            if acc > best_acc:
                best_acc = acc
                best_step = step
                if acc > 0.10 and ckpt_dir:
                    path = os.path.join(ckpt_dir, f"{cfg['name']}_s{seed}")
                    os.makedirs(path, exist_ok=True)
                    torch.save({
                        "state_dict": model.state_dict(),
                        "config": cfg,
                        "step": step,
                        "acc": acc,
                        "seed": seed,
                    }, os.path.join(path, "best.pt"))

    elapsed = time.time() - start_time
    final_loss = loss.item()

    return {
        "config_name": cfg["name"],
        "n_params": n_params,
        "seed": seed,
        "best_exact_acc": best_acc,
        "best_step": best_step,
        "final_loss": final_loss,
        "final_step": max_steps,
        "elapsed_s": elapsed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=100000)
    parser.add_argument("--eval-interval", type=int, default=5000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="experiments/d2_wave2_results.csv")
    parser.add_argument("--ckpt-dir", default="checkpoints/d2_wave2")
    parser.add_argument("--group", default="all",
                        choices=["all", "relu_only", "swiglu_relu", "relu_d3", "multilayer"])
    args = parser.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", "4")

    configs = generate_configs(args.group)
    seed_list = [1, 8, 15, 22, 29][:args.seeds]

    # Load completed runs
    completed = set()
    if os.path.exists(args.output):
        with open(args.output) as f:
            for row in csv.DictReader(f):
                completed.add((row["config_name"], int(row["seed"])))

    # Sort by param count
    configs_with_params = []
    for cfg in configs:
        m = build_model(cfg)
        n = count_params(m)
        configs_with_params.append((n, cfg))
    configs_with_params.sort(key=lambda x: (x[0], x[1]["name"]))

    total_runs = sum(1 for _, cfg in configs_with_params for s in seed_list
                     if (cfg["name"], s) not in completed)
    print(f"Wave 2 search ({args.group}): {len(configs_with_params)} configs × {len(seed_list)} seeds")
    print(f"Remaining: {total_runs} runs, {len(completed)} already done")
    print()

    # Print config summary
    for n, cfg in configs_with_params:
        print(f"  {n:3d}p  {cfg['name']}")
    print()

    fieldnames = ["config_name", "n_params", "seed", "best_exact_acc",
                  "best_step", "final_loss", "final_step", "elapsed_s"]

    write_header = not os.path.exists(args.output) or os.path.getsize(args.output) == 0
    done = 0

    for n_params, cfg in configs_with_params:
        for seed in seed_list:
            if (cfg["name"], seed) in completed:
                continue

            done += 1
            print(f"\n[{done}/{total_runs}] {cfg['name']} ({n_params}p) seed={seed}")
            result = train_one(cfg, seed, args.max_steps, args.eval_interval,
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
