"""Reproduce any submitted model from random initialization.

All model configs live in experiments/configs.yaml. This script reads them,
builds the model, and runs all training phases automatically.

Usage:
    uv run experiments/reproduce.py 89p
    uv run experiments/reproduce.py 122p --seed 42
    uv run experiments/reproduce.py --all
    uv run experiments/reproduce.py --list
    uv run experiments/reproduce.py 83p --steps-override 200   # smoke test
"""

import argparse
import csv
import json
import math
import os
import random
import sys
import time

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import torch.nn as nn
import torch.nn.functional as F

from minimal10digittransformer.model.qwen3 import (
    Qwen3AdditionModel,
    VOCAB_SIZE,
    INPUT_LEN,
    OUTPUT_LEN,
    TOTAL_LEN,
)
from minimal10digittransformer.model.circular_arc import CircularArcQwen3
from minimal10digittransformer.model.rank1_out import Rank1OutModel
from minimal10digittransformer.data.addition import (
    encode,
    expected_output,
    generate_batch,
    generate_test_set,
    load_test_set,
)
from minimal10digittransformer.evaluation.metrics import evaluate, evaluate_detailed


# ============================================================================
# Architecture registry
# ============================================================================

ARCH_CLASSES = {
    "qwen3": Qwen3AdditionModel,
    "circular_arc": CircularArcQwen3,
    "rank1_out": Rank1OutModel,
}

RESERVED_SEEDS = {42, 2025, 123, 99}


# ============================================================================
# Config loading
# ============================================================================

def load_configs(yaml_path=None):
    """Load model configs from YAML, merging defaults into each model."""
    if yaml_path is None:
        yaml_path = os.path.join(os.path.dirname(__file__), "configs.yaml")
    with open(yaml_path) as f:
        raw = yaml.safe_load(f)

    defaults = raw.get("defaults", {})
    configs = {}
    for name, cfg in raw["models"].items():
        # Merge defaults into model_args
        merged_args = dict(defaults)
        merged_args.update(cfg.get("model_args", {}))
        cfg["model_args"] = merged_args
        configs[name] = cfg
    return configs


# ============================================================================
# Training utilities
# ============================================================================

def count_params(model):
    return sum(p.numel() for p in model.parameters())


def train_step(model, full_seq, labels):
    logits = model(full_seq)
    shift_logits = logits[:, :-1, :].reshape(-1, VOCAB_SIZE)
    shift_labels = labels[:, 1:].reshape(-1)
    return F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)


def save_checkpoint(model, path, step, accuracy, n_params, extra=None):
    data = {
        "state_dict": model.state_dict(),
        "step": step, "accuracy": accuracy, "n_params": n_params,
    }
    if extra:
        data.update(extra)
    torch.save(data, path)


def load_checkpoint(model, path):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["state_dict"])
    return ckpt


# ============================================================================
# EMA helpers
# ============================================================================

def update_ema(model, ema_params, decay):
    for p, p_ema in zip(model.parameters(), ema_params):
        p_ema.data.mul_(decay).add_(p.data, alpha=1 - decay)


def apply_ema(model, ema_params):
    for p, p_ema in zip(model.parameters(), ema_params):
        p.data.copy_(p_ema.data)


def save_ema_checkpoint(model, ema_params, path, step, accuracy, n_params):
    original = [p.data.clone() for p in model.parameters()]
    apply_ema(model, ema_params)
    save_checkpoint(model, path, step, accuracy, n_params)
    for p, orig in zip(model.parameters(), original):
        p.data.copy_(orig)


# ============================================================================
# Targeted FT helpers
# ============================================================================

def find_errors(model, device, test_pairs):
    model.eval()
    errors = []
    with torch.no_grad():
        for a, b in test_pairs:
            exp = expected_output(a, b)
            inp = torch.tensor([encode(a, b)], dtype=torch.long, device=device)
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


def build_targeted_batch(error_pairs, batch_size, device):
    selected = (random.sample(error_pairs, batch_size)
                if len(error_pairs) >= batch_size else list(error_pairs))
    full_list, label_list = [], []
    for a, b in selected:
        inp = encode(a, b)
        tgt = expected_output(a, b)
        full_list.append(inp + tgt)
        label_list.append([-100] * INPUT_LEN + tgt)
    n_random = batch_size - len(full_list)
    if n_random > 0:
        rand_seq, rand_labels = generate_batch(n_random, device, max_digits=10)
        error_seq = torch.tensor(full_list, dtype=torch.long, device=device)
        error_labels = torch.tensor(label_list, dtype=torch.long, device=device)
        full_seq = torch.cat([error_seq, rand_seq], dim=0)
        labels = torch.cat([error_labels, rand_labels], dim=0)
    else:
        full_seq = torch.tensor(full_list, dtype=torch.long, device=device)
        labels = torch.tensor(label_list, dtype=torch.long, device=device)
    perm = torch.randperm(batch_size)
    return full_seq[perm], labels[perm]


# ============================================================================
# Metrics CSV
# ============================================================================

class MetricsWriter:
    COLUMNS = ["phase", "global_step", "phase_step", "loss", "lr",
               "exact_acc", "digit_acc", "elapsed"]

    def __init__(self, path):
        self.file = open(path, "w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow(self.COLUMNS)

    def log(self, phase, global_step, phase_step, loss, lr, elapsed,
            seq_acc=None, dig_acc=None):
        self.writer.writerow([
            phase, global_step, phase_step, f"{loss:.6f}", f"{lr:.2e}",
            f"{seq_acc:.4f}" if seq_acc is not None else "",
            f"{dig_acc:.4f}" if dig_acc is not None else "",
            f"{elapsed:.1f}",
        ])

    def flush(self):
        self.file.flush()

    def close(self):
        self.file.close()


# ============================================================================
# Phase runners
# ============================================================================

def _eval_and_log(model, device, eval_pairs, metrics, phase_num, global_step,
                  step, loss, lr, t0, ckpt_dir, n_params, best_acc_holder,
                  ema_params=None, best_ema_holder=None):
    """Run eval, log, save checkpoint if improved. Returns (seq_acc, dig_acc)."""
    seq_acc, dig_acc = evaluate(model, device, test_pairs=eval_pairs[:500])
    elapsed = time.time() - t0
    print(f"    EVAL step {step}: exact={seq_acc:.4f} "
          f"digit={dig_acc:.4f} [{elapsed:.0f}s]")
    sys.stdout.flush()
    metrics.log(phase_num, global_step, step, loss, lr, elapsed,
                seq_acc, dig_acc)
    metrics.flush()

    if seq_acc > best_acc_holder[0]:
        best_acc_holder[0] = seq_acc
        save_checkpoint(model, f"{ckpt_dir}/phase{phase_num}_best.pt",
                        global_step, seq_acc, n_params)
        print(f"    ** NEW BEST: {seq_acc:.4f} **")

    if ema_params is not None and best_ema_holder is not None:
        original = [p.data.clone() for p in model.parameters()]
        apply_ema(model, ema_params)
        ema_acc, _ = evaluate(model, device, test_pairs=eval_pairs[:500])
        for p, orig in zip(model.parameters(), original):
            p.data.copy_(orig)
        print(f"    EMA  step {step}: exact={ema_acc:.4f}")
        if ema_acc > best_ema_holder[0]:
            best_ema_holder[0] = ema_acc
            save_ema_checkpoint(model, ema_params,
                                f"{ckpt_dir}/phase{phase_num}_best_ema.pt",
                                global_step, ema_acc, n_params)

    return seq_acc, dig_acc


def run_phase_gradient(model, device, phase_cfg, eval_pairs, ckpt_dir,
                       phase_num, n_params, metrics, t0, global_step, steps):
    """Run a gradient-based phase (cosine, constant, or cosine_nowd).

    Unifies cosine/constant/cosine_nowd to eliminate code duplication.
    """
    ptype = phase_cfg["type"]
    lr = phase_cfg["lr"]
    batch_size = phase_cfg["batch_size"]
    eval_interval = phase_cfg.get("eval_interval", 2000)
    ema_decay = phase_cfg.get("ema_decay", 0.0)

    # Optimizer selection
    if ptype == "cosine_nowd":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        wd = 0.0
    else:
        wd = phase_cfg.get("weight_decay", 0.01)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                      weight_decay=wd)

    # LR schedule
    min_lr_ratio = phase_cfg.get("min_lr_ratio", 0.1)
    if ptype == "constant":
        get_lr = lambda step: lr
    elif ptype == "cosine_nowd":
        get_lr = lambda step: lr * 0.5 * (1 + math.cos(math.pi * step / max(steps, 1)))
    else:  # cosine
        min_lr = lr * min_lr_ratio
        get_lr = lambda step: min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * step / max(steps, 1)))

    # EMA
    ema_params = [p.data.clone() for p in model.parameters()] if ema_decay > 0 else None

    label = {"cosine": "cosine", "constant": "constant",
             "cosine_nowd": "cosine-nowd"}[ptype]
    print(f"\n  Phase {phase_num} [{label}]: lr={lr}, batch={batch_size}, "
          f"steps={steps}, wd={wd}")

    best_acc = [0.0]
    best_ema_acc = [0.0]

    for step in range(1, steps + 1):
        cur_lr = get_lr(step)
        for pg in optimizer.param_groups:
            pg["lr"] = cur_lr

        model.train()
        full_seq, labels = generate_batch(batch_size, device, max_digits=10)
        optimizer.zero_grad()
        loss = train_step(model, full_seq, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        global_step += 1

        if ema_params is not None:
            update_ema(model, ema_params, ema_decay)

        if step % 100 == 0 and step % eval_interval != 0:
            elapsed = time.time() - t0
            print(f"    step {step:6d}/{steps} | loss {loss.item():.4f} | "
                  f"lr {cur_lr:.2e} | {elapsed:.0f}s")
            sys.stdout.flush()
            metrics.log(phase_num, global_step, step, loss.item(), cur_lr,
                        elapsed)
            if step % 1000 == 0:
                metrics.flush()

        if step % eval_interval == 0:
            _eval_and_log(model, device, eval_pairs, metrics, phase_num,
                          global_step, step, loss.item(), cur_lr, t0,
                          ckpt_dir, n_params, best_acc,
                          ema_params=ema_params, best_ema_holder=best_ema_acc)

    # Save final
    save_checkpoint(model, f"{ckpt_dir}/phase{phase_num}_final.pt",
                    global_step, best_acc[0], n_params)
    if ema_params is not None:
        save_ema_checkpoint(model, ema_params,
                            f"{ckpt_dir}/phase{phase_num}_final_ema.pt",
                            global_step, best_ema_acc[0], n_params)
    return best_acc[0], global_step


def run_phase_lbfgs(model, device, phase_cfg, eval_pairs, ckpt_dir,
                    phase_num, n_params, metrics, t0, global_step, steps):
    """L-BFGS optimization phase."""
    lr = phase_cfg["lr"]
    batch_size = phase_cfg["batch_size"]
    max_iter = phase_cfg.get("max_iter", 30)
    history_size = phase_cfg.get("history_size", 10)
    eval_interval = phase_cfg.get("eval_interval", 50)

    optimizer = torch.optim.LBFGS(
        model.parameters(), lr=lr, max_iter=max_iter,
        history_size=history_size, line_search_fn="strong_wolfe",
    )

    print(f"\n  Phase {phase_num} [L-BFGS]: lr={lr}, batch={batch_size}, "
          f"steps={steps}, max_iter={max_iter}")

    best_acc = [0.0]

    for step in range(1, steps + 1):
        model.train()
        full_seq, labels = generate_batch(batch_size, device, max_digits=10)

        def closure():
            optimizer.zero_grad()
            loss = train_step(model, full_seq, labels)
            loss.backward()
            return loss

        loss = optimizer.step(closure)
        global_step += 1

        if step % 10 == 0 and step % eval_interval != 0:
            elapsed = time.time() - t0
            loss_val = loss.item() if torch.is_tensor(loss) else loss
            print(f"    step {step:6d}/{steps} | loss {loss_val:.6f} | "
                  f"{elapsed:.0f}s")
            sys.stdout.flush()
            metrics.log(phase_num, global_step, step, loss_val, lr, elapsed)
            if step % 50 == 0:
                metrics.flush()

        if step % eval_interval == 0:
            loss_val = loss.item() if torch.is_tensor(loss) else loss
            _eval_and_log(model, device, eval_pairs, metrics, phase_num,
                          global_step, step, loss_val, lr, t0,
                          ckpt_dir, n_params, best_acc)

    save_checkpoint(model, f"{ckpt_dir}/phase{phase_num}_final.pt",
                    global_step, best_acc[0], n_params)
    return best_acc[0], global_step


def run_phase_targeted(model, device, phase_cfg, targeted_pairs, ckpt_dir,
                       phase_num, n_params, metrics, t0, global_step,
                       steps_override=None):
    """Iterated targeted FT: find errors, train on them, repeat."""
    lr = phase_cfg["lr"]
    batch_size = phase_cfg["batch_size"]
    weight_decay = phase_cfg.get("weight_decay", 0.0)
    max_iters = phase_cfg.get("max_iters", 10)
    steps_per_iter = steps_override or phase_cfg.get("steps_per_iter", 3000)
    eval_interval = phase_cfg.get("eval_interval", 500)

    print(f"\n  Phase {phase_num} [targeted]: lr={lr}, batch={batch_size}, "
          f"steps/iter={steps_per_iter}, max_iters={max_iters}")

    print(f"    Finding initial errors on {len(targeted_pairs)} pairs...")
    errors = find_errors(model, device, targeted_pairs)
    print(f"    Initial errors: {len(errors)}")

    if len(errors) == 0:
        print("    Model is already perfect on eval set!")
        save_checkpoint(model, f"{ckpt_dir}/phase{phase_num}_best.pt",
                        global_step, 1.0, n_params)
        return 1.0, global_step

    cumulative_errors = set((a, b) for a, b in errors)
    best_n_errors = len(errors)
    best_acc = 1.0 - best_n_errors / len(targeted_pairs)
    phase_step = 0  # cumulative across iterations (never resets)

    for iteration in range(1, max_iters + 1):
        error_list = list(cumulative_errors)
        print(f"\n    --- Targeted iteration {iteration} ---")
        print(f"    Training on {len(error_list)} cumulative error pairs")

        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay)

        for step in range(1, steps_per_iter + 1):
            model.train()
            full_seq, labels = build_targeted_batch(
                error_list, batch_size, device)
            optimizer.zero_grad()
            loss = train_step(model, full_seq, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            global_step += 1
            phase_step += 1

            if step % 500 == 0:
                elapsed = time.time() - t0
                print(f"      [iter {iteration}] step {step:5d}/{steps_per_iter} "
                      f"| loss {loss.item():.6f} | errors={len(error_list)} "
                      f"| {elapsed:.0f}s")
                sys.stdout.flush()

            if step % eval_interval == 0:
                seq_acc, dig_acc = evaluate(model, device,
                                           test_pairs=targeted_pairs[:200])
                elapsed = time.time() - t0
                metrics.log(phase_num, global_step, phase_step, loss.item(),
                            lr, elapsed, seq_acc, dig_acc)
                metrics.flush()

        print(f"    Full evaluation on {len(targeted_pairs)} pairs...")
        new_errors = find_errors(model, device, targeted_pairs)
        n_new = len(new_errors)
        seq_acc = 1.0 - n_new / len(targeted_pairs)
        elapsed = time.time() - t0
        print(f"    Iteration {iteration}: {n_new} errors "
              f"(exact={seq_acc:.6f}) [{elapsed:.0f}s]")

        if n_new < best_n_errors:
            best_n_errors = n_new
            best_acc = seq_acc
            save_checkpoint(
                model, f"{ckpt_dir}/phase{phase_num}_best.pt",
                global_step, seq_acc, n_params,
                extra={"targeted_iteration": iteration,
                       "n_errors": n_new,
                       "cumulative_pairs": len(cumulative_errors)})
            print(f"    ** NEW BEST: {seq_acc:.6f} ({n_new} errors) **")

        if n_new == 0:
            print(f"    Perfect! 0 errors after iteration {iteration}.")
            break

        prev = len(cumulative_errors)
        for a, b in new_errors:
            cumulative_errors.add((a, b))
        added = len(cumulative_errors) - prev
        print(f"    Cumulative errors: {prev} -> "
              f"{len(cumulative_errors)} (+{added} new)")

    save_checkpoint(model, f"{ckpt_dir}/phase{phase_num}_final.pt",
                    global_step, best_acc, n_params)
    return best_acc, global_step


# ============================================================================
# Validation
# ============================================================================

def run_validation(model, device, ckpt_dir):
    """Run verify-style + holdout validation, save results."""
    results = {}

    print(f"\n{'=' * 70}")
    print("Final Validation")
    print(f"{'=' * 70}")

    # Verify-style: 10 edge + 10K random (seed=2025)
    edge_cases = [
        (0, 0), (0, 1), (9_999_999_999, 0), (9_999_999_999, 1),
        (9_999_999_999, 9_999_999_999), (5_000_000_000, 5_000_000_000),
        (1_111_111_111, 8_888_888_889), (1_234_567_890, 9_876_543_210),
        (9_999_999_999, 9_999_999_999), (1, 9_999_999_999),
    ]
    rng = random.Random(2025)
    random_cases = [
        (rng.randint(0, 9_999_999_999), rng.randint(0, 9_999_999_999))
        for _ in range(10000)
    ]
    verify_pairs = edge_cases + random_cases

    print(f"  verify-style: {len(verify_pairs)} pairs "
          f"(10 edge + 10000 random, seed=2025)")
    detailed = evaluate_detailed(model, device, verify_pairs)
    results["verify"] = {
        "n_samples": detailed["n_samples"],
        "exact_acc": detailed["exact_acc"],
        "n_errors": detailed["n_errors"],
        "digit_acc": detailed["digit_acc"],
        "qualified": detailed["exact_acc"] >= 0.99,
    }
    qualified = "QUALIFIED" if results["verify"]["qualified"] else "NOT QUALIFIED"
    correct = detailed["n_samples"] - detailed["n_errors"]
    print(f"  Result: {correct}/{detailed['n_samples']} correct "
          f"({detailed['exact_acc'] * 100:.2f}%)")
    print(f"  Status: {qualified}")

    # Holdout sets
    for name, path in [("holdout_10k", "data/test_holdout_10k.json"),
                       ("holdout_50k", "data/test_50k_independent.json")]:
        if os.path.exists(path):
            print(f"\n  {name} ({path}):")
            pairs = load_test_set(path)
            d = evaluate_detailed(model, device, pairs)
            results[name] = {
                "n_samples": d["n_samples"],
                "exact_acc": d["exact_acc"],
                "n_errors": d["n_errors"],
                "digit_acc": d["digit_acc"],
            }
            correct = d["n_samples"] - d["n_errors"]
            print(f"  Result: {correct}/{d['n_samples']} correct "
                  f"({d['exact_acc'] * 100:.2f}%)")

    val_path = os.path.join(ckpt_dir, "validation.json")
    with open(val_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Validation saved to {val_path}")
    return results


# ============================================================================
# Main pipeline
# ============================================================================

PHASE_RUNNERS = {
    "cosine": run_phase_gradient,
    "constant": run_phase_gradient,
    "cosine_nowd": run_phase_gradient,
    "lbfgs": run_phase_lbfgs,
}


def run_config(config_name, cfg, seed, device, ckpt_dir,
               steps_override=None, no_validate=False,
               train_eval_seed=777, train_eval_size=10000):
    """Run the full reproduction pipeline for one config."""
    os.makedirs(ckpt_dir, exist_ok=True)

    # Eval pairs (separate from test/holdout sets)
    progress_pairs = generate_test_set(2000, seed=12345)
    targeted_pairs = generate_test_set(train_eval_size, seed=train_eval_seed)

    # Seed and build model
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    arch_cls = ARCH_CLASSES[cfg["arch"]]
    model = arch_cls(**cfg["model_args"]).to(device)
    n_params = count_params(model)

    # Banner
    print(f"\n{'=' * 70}")
    print(f"Reproduction: {config_name} ({cfg['arch']})")
    print(f"{'=' * 70}")
    print(f"  Description: {cfg['desc']}")
    print(f"  Parameters:  {n_params} (expected: {cfg['params']})")
    print(f"  Seed:        {seed}")
    print(f"  Device:      {device}")
    phases = cfg["phases"]
    print(f"  Phases:      {len(phases)}")
    for i, p in enumerate(phases):
        if p["type"] == "targeted":
            s = f"{p.get('steps_per_iter', 3000)}/iter x {p.get('max_iters', 10)}"
        else:
            s = str(steps_override or p["steps"])
        print(f"    Phase {i}: {p['type']} ({s} steps)")
    print(f"  Output:      {ckpt_dir}/")
    print(f"{'=' * 70}\n")

    if n_params != cfg["params"]:
        print(f"WARNING: Expected {cfg['params']} params, got {n_params}")

    metrics = MetricsWriter(f"{ckpt_dir}/metrics.csv")
    t0 = time.time()
    global_step = 0
    phase_results = []

    for phase_num, phase_cfg in enumerate(phases):
        # Per-phase seed override
        if "stage_seed" in phase_cfg:
            random.seed(phase_cfg["stage_seed"])
            torch.manual_seed(phase_cfg["stage_seed"])

        # Load from previous phase
        if phase_num > 0:
            prev_cfg = phases[phase_num - 1]
            if prev_cfg.get("use_final"):
                load_path = f"{ckpt_dir}/phase{phase_num - 1}_final.pt"
            else:
                load_path = f"{ckpt_dir}/phase{phase_num - 1}_best.pt"
            fallback = f"{ckpt_dir}/phase{phase_num - 1}_final.pt"
            for path in [load_path, fallback]:
                if os.path.exists(path):
                    ckpt = load_checkpoint(model, path)
                    print(f"  Loaded {os.path.basename(path)} "
                          f"(accuracy={ckpt.get('accuracy', '?')})")
                    break
            else:
                print(f"  WARNING: No checkpoint for phase {phase_num - 1}")

        ptype = phase_cfg["type"]
        steps = (steps_override or phase_cfg.get("steps")) if ptype != "targeted" else None

        print(f"\n{'=' * 50}")
        print(f"Phase {phase_num}: {ptype}")
        print(f"{'=' * 50}")

        if ptype == "targeted":
            spi = steps_override or phase_cfg.get("steps_per_iter", 3000)
            phase_cfg_copy = dict(phase_cfg)
            if steps_override:
                phase_cfg_copy["steps_per_iter"] = spi
            best_acc, global_step = run_phase_targeted(
                model, device, phase_cfg_copy, targeted_pairs, ckpt_dir,
                phase_num, n_params, metrics, t0, global_step)
        elif ptype in PHASE_RUNNERS:
            best_acc, global_step = PHASE_RUNNERS[ptype](
                model, device, phase_cfg, progress_pairs, ckpt_dir,
                phase_num, n_params, metrics, t0, global_step, steps)
        else:
            raise ValueError(f"Unknown phase type: {ptype}")

        elapsed = time.time() - t0
        phase_results.append({
            "phase": phase_num, "type": ptype,
            "best_acc": best_acc, "global_step": global_step,
            "elapsed": elapsed,
        })
        print(f"\n  Phase {phase_num} complete: best_acc={best_acc:.6f}, "
              f"steps={global_step}, time={elapsed:.0f}s")

    metrics.close()

    # Load final best and save as final_best.pt
    last = len(phases) - 1
    best_path = f"{ckpt_dir}/phase{last}_best.pt"
    if os.path.exists(best_path):
        load_checkpoint(model, best_path)
    save_checkpoint(model, f"{ckpt_dir}/final_best.pt",
                    global_step, phase_results[-1]["best_acc"], n_params)

    # Summary eval
    print(f"\n{'=' * 70}")
    print("Pipeline complete — summary evaluation")
    print(f"{'=' * 70}")
    summary_pairs = generate_test_set(2000, seed=12345)
    seq_acc, dig_acc = evaluate(model, device, test_pairs=summary_pairs)
    elapsed = time.time() - t0
    print(f"  Summary (2K, seed=12345): exact={seq_acc:.4f} "
          f"digit={dig_acc:.4f}")
    print(f"  Total training time: {elapsed:.0f}s ({elapsed / 3600:.1f}h)")

    # Validation
    validation = None
    if not no_validate:
        validation = run_validation(model, device, ckpt_dir)

    # Save run summary
    summary = {
        "config": config_name,
        "description": cfg["desc"],
        "arch": cfg["arch"],
        "n_params": n_params,
        "expected_params": cfg["params"],
        "seed": seed,
        "train_eval_seed": train_eval_seed,
        "device": str(device),
        "steps_override": steps_override,
        "phases": phase_results,
        "summary_eval": {"exact_acc": seq_acc, "digit_acc": dig_acc},
        "validation": validation,
        "total_elapsed": time.time() - t0,
        "ckpt_dir": ckpt_dir,
    }
    with open(f"{ckpt_dir}/run_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"DONE: {config_name} ({cfg['arch']}) seed={seed}")
    print(f"{'=' * 70}")
    print(f"  Checkpoints: {ckpt_dir}/")
    if validation:
        v = validation.get("verify", {})
        ne = v.get("n_errors", "?")
        q = "QUALIFIED" if v.get("qualified") else "NOT QUALIFIED"
        print(f"  Verify: {v.get('n_samples', 0) - ne}/{v.get('n_samples', 0)} ({q})")
    print(f"  Total time: {time.time() - t0:.0f}s")

    return summary


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Reproduce any submitted model from random initialization.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  uv run experiments/reproduce.py 89p
  uv run experiments/reproduce.py 122p --seed 42
  uv run experiments/reproduce.py --all
  uv run experiments/reproduce.py --list
  uv run experiments/reproduce.py 83p --steps-override 200  # smoke test
""",
    )
    parser.add_argument("model", nargs="?", default=None,
                        help="Model name (e.g. 89p, 122p, 62p)")
    parser.add_argument("--list", action="store_true",
                        help="List all available configs")
    parser.add_argument("--all", action="store_true",
                        help="Reproduce ALL configs sequentially")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override seed (default: proven_seed from config)")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--steps-override", type=int, default=None,
                        help="Override step count for ALL phases (smoke test)")
    parser.add_argument("--no-validate", action="store_true",
                        help="Skip final validation")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory")
    parser.add_argument("--config-file", type=str, default=None,
                        help="Path to configs YAML (default: experiments/configs.yaml)")
    parser.add_argument("--train-eval-seed", type=int, default=777,
                        help="Seed for targeted FT eval pairs (default: 777)")
    parser.add_argument("--train-eval-size", type=int, default=10000,
                        help="Number of eval pairs for targeted FT (default: 10000)")
    args = parser.parse_args()

    configs = load_configs(args.config_file)

    if args.list:
        print(f"\n{'=' * 70}")
        print("Available configurations")
        print(f"{'=' * 70}")
        for name in sorted(configs, key=lambda k: configs[k]["params"],
                           reverse=True):
            c = configs[name]
            total_steps = sum(
                p.get("steps", p.get("steps_per_iter", 0) * p.get("max_iters", 1))
                for p in c["phases"])
            print(f"  {name:>5s}: {c['params']:3d}p | {c['arch']:13s} | "
                  f"{len(c['phases'])} phases | ~{total_steps:,} steps | "
                  f"seed={c['proven_seed']} | {c['desc']}")
        print(f"\nUsage: uv run experiments/reproduce.py <model>")
        return

    if args.train_eval_seed in RESERVED_SEEDS:
        print(f"ERROR: --train-eval-seed {args.train_eval_seed} is reserved!")
        sys.exit(1)

    # Determine which configs to run
    if args.all:
        to_run = sorted(configs.keys(),
                        key=lambda k: configs[k]["params"], reverse=True)
    elif args.model:
        if args.model not in configs:
            print(f"ERROR: Unknown model '{args.model}'. "
                  f"Available: {', '.join(sorted(configs.keys()))}")
            sys.exit(1)
        to_run = [args.model]
    else:
        parser.print_help()
        sys.exit(1)

    device = torch.device(args.device)
    all_results = []

    for name in to_run:
        cfg = configs[name]
        seed = args.seed if args.seed is not None else cfg["proven_seed"]
        ckpt_dir = (args.output_dir or
                    f"checkpoints/reproduce_{name}_s{seed}")

        result = run_config(
            name, cfg, seed, device, ckpt_dir,
            steps_override=args.steps_override,
            no_validate=args.no_validate,
            train_eval_seed=args.train_eval_seed,
            train_eval_size=args.train_eval_size,
        )
        all_results.append(result)

    if len(all_results) > 1:
        print(f"\n{'=' * 70}")
        print("ALL REPRODUCTIONS COMPLETE")
        print(f"{'=' * 70}")
        for r in all_results:
            v = (r.get("validation", {}) or {}).get("verify", {})
            q = "PASS" if v.get("qualified") else "FAIL"
            ne = v.get("n_errors", "?")
            print(f"  {r['config']:>5s}: {r['n_params']}p | "
                  f"verify={q} ({ne} errors) | "
                  f"{r['total_elapsed']:.0f}s")


if __name__ == "__main__":
    main()
