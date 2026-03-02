"""Sweep v5: Fixed-layout curriculum for 10-digit addition.

Critical insight: Previous experiments showed that digit-count curriculum
(2-digit -> 3-digit -> ... -> 10-digit) fails because changing the sequence
length destroys learned attention patterns. When trained on 2-digit (9 tokens),
the model learns position-specific attention. Switching to 3-digit (12 tokens)
shifts all position roles and the model cannot adapt.

Solution: FIXED-LAYOUT CURRICULUM
- Always use the 10-digit format (33 tokens: 10 + 1 + 10 + 1 + 11)
- Vary difficulty by controlling number magnitude (sig_digits)
- Level 1: a, b in [0, 99] -> 2 significant digits, 8 leading zeros
- Level 9: a, b in [0, 10^10 - 1] -> full 10-digit
- This preserves position roles and attention patterns across all levels

Phases:
  baseline: d_model=16 trained DIRECTLY on full 10-digit (no curriculum)
  seed_tournament: d7 x 3 seeds with fixed-layout curriculum
  all: Everything
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from minimal10digittransformer.model.transformer import (
    MinimalTransformer,
    TransformerConfig,
    count_parameters,
)

# ---------------------------------------------------------------------------
# Constants for fixed 10-digit layout
# ---------------------------------------------------------------------------
MAX_DIGITS = 10
INPUT_LEN = MAX_DIGITS * 2 + 2  # 22: 10 (A) + 1 (+) + 10 (B) + 1 (=)
OUTPUT_LEN = MAX_DIGITS + 1      # 11: result digits (can overflow by 1)
SEQ_LEN = INPUT_LEN + OUTPUT_LEN # 33: always


# ---------------------------------------------------------------------------
# Fixed-layout batch generation
# ---------------------------------------------------------------------------

def generate_batch_v5(batch_size, sig_digits, device):
    """Generate batch always in 10-digit format but with limited significant digits.

    sig_digits controls the magnitude: numbers are 0 to 10^sig_digits - 1,
    then zero-padded to 10 digits for the standard 33-token format.

    The sequence is ALWAYS 33 tokens regardless of sig_digits:
      [A9 A8 ... A0 + B9 B8 ... B0 = C0 C1 ... C10]
    where A,B are MSB-first (human readable) and C is LSB-first.
    """
    max_val = 10 ** sig_digits
    a = torch.randint(0, max_val, (batch_size,), device=device, dtype=torch.long)
    b = torch.randint(0, max_val, (batch_size,), device=device, dtype=torch.long)
    c = a + b

    # Extract ALL 10 digits for input, 11 for output
    a_digits, b_digits, c_digits = [], [], []
    a_tmp, b_tmp, c_tmp = a.clone(), b.clone(), c.clone()
    for _ in range(MAX_DIGITS):
        a_digits.append(a_tmp % 10)
        b_digits.append(b_tmp % 10)
        c_digits.append(c_tmp % 10)
        a_tmp //= 10
        b_tmp //= 10
        c_tmp //= 10
    c_digits.append(c_tmp % 10)  # 11th digit for overflow

    a_t = torch.stack(a_digits, dim=1).flip(1)  # MSB first for input
    b_t = torch.stack(b_digits, dim=1).flip(1)  # MSB first for input
    c_t = torch.stack(c_digits, dim=1)           # LSB first for output

    plus = torch.full((batch_size, 1), 10, device=device, dtype=torch.long)
    eq = torch.full((batch_size, 1), 11, device=device, dtype=torch.long)

    input_ids = torch.cat([a_t, plus, b_t, eq, c_t], dim=1)  # Always 33 tokens
    assert input_ids.shape[1] == SEQ_LEN, f"Expected {SEQ_LEN} tokens, got {input_ids.shape[1]}"

    labels = torch.full_like(input_ids, -100)
    labels[:, INPUT_LEN - 1:-1] = c_t  # Predict output tokens

    return input_ids, labels


# ---------------------------------------------------------------------------
# Evaluation (fixed 10-digit layout)
# ---------------------------------------------------------------------------

def evaluate_v5(model, n_samples, sig_digits, device, seed=42):
    """Evaluate exact-match accuracy with fixed 10-digit format.

    Numbers are up to sig_digits significant digits, but always in 33-token format.
    """
    model.eval()
    correct = 0
    total = 0
    digit_correct = torch.zeros(OUTPUT_LEN, device=device)
    digit_total = torch.zeros(OUTPUT_LEN, device=device)

    gen = torch.Generator(device=device if device != "cpu" else "cpu")
    gen.manual_seed(seed)

    with torch.no_grad():
        for _ in range(0, n_samples, 2048):
            bs = min(2048, n_samples - total)
            if bs <= 0:
                break
            input_ids, labels = generate_batch_v5(bs, sig_digits, device)
            logits = model(input_ids)
            preds = logits[:, INPUT_LEN - 1:-1, :].argmax(dim=-1)
            targets = input_ids[:, INPUT_LEN:]

            correct += (preds == targets).all(dim=1).sum().item()
            matches = (preds == targets)
            digit_correct += matches.sum(dim=0).float()
            digit_total += torch.full((OUTPUT_LEN,), float(bs), device=device)
            total += bs

    exact_acc = correct / total if total > 0 else 0.0
    per_digit = (digit_correct / digit_total.clamp(min=1)).cpu().tolist()

    return {
        "exact_match": exact_acc,
        "per_digit_accuracy": per_digit,
        "digit_accuracy_avg": sum(per_digit) / len(per_digit),
        "correct": correct,
        "total": total,
    }


def evaluate_all_sig_levels(model, device, max_sig=10, n_samples=10000):
    """Evaluate the model at each significance level (1 through max_sig)."""
    results = {}
    for s in range(1, max_sig + 1):
        r = evaluate_v5(model, n_samples, s, device, seed=12345 + s)
        results[s] = r
    return results


def get_example_predictions_v5(model, sig_digits, device, n_examples=5):
    """Get example predictions for logging, in fixed 10-digit format."""
    model.eval()
    with torch.no_grad():
        input_ids, labels = generate_batch_v5(n_examples, sig_digits, device)
        logits = model(input_ids)
        preds = logits[:, INPUT_LEN - 1:-1, :].argmax(dim=-1)
        targets = input_ids[:, INPUT_LEN:]

    examples = []
    for i in range(n_examples):
        seq = input_ids[i].cpu().tolist()
        a_digits = seq[:MAX_DIGITS]
        b_digits = seq[MAX_DIGITS + 1: MAX_DIGITS * 2 + 1]
        pred_digits = preds[i].cpu().tolist()
        target_digits = targets[i].cpu().tolist()

        a_str = "".join(str(d) for d in a_digits)
        b_str = "".join(str(d) for d in b_digits)
        # Output is LSB-first, reverse for human-readable
        pred_str = "".join(str(d) for d in reversed(pred_digits))
        target_str = "".join(str(d) for d in reversed(target_digits))
        match = "OK" if pred_digits == target_digits else "WRONG"

        examples.append(f"  {a_str} + {b_str} = {pred_str} (expected {target_str}) [{match}]")

    return "\n".join(examples)


# ---------------------------------------------------------------------------
# Configurations
# ---------------------------------------------------------------------------

def get_d7_config():
    """d_model=7, 15 repeats, ~420 params - best small architecture."""
    return ("d7_v5", TransformerConfig(
        d_model=7, n_heads=1, n_layers=1, d_ff=7,
        share_layers=True, n_layer_repeats=15,
        norm_type="rmsnorm", activation="relu",
        pe_period=11.0,
    ), 3e-3)


def get_d8_config():
    """d_model=8, 15 repeats, ~528 params."""
    return ("d8_v5", TransformerConfig(
        d_model=8, n_heads=1, n_layers=1, d_ff=8,
        share_layers=True, n_layer_repeats=15,
        norm_type="rmsnorm", activation="relu",
        pe_period=11.0,
    ), 1e-3)


def get_d16_config():
    """d_model=16, 10 repeats, ~1824 params - baseline."""
    return ("d16_v5", TransformerConfig(
        d_model=16, n_heads=1, n_layers=1, d_ff=16,
        share_layers=True, n_layer_repeats=10,
        norm_type="rmsnorm", activation="relu",
        pe_period=11.0,
    ), 1e-3)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_v5(config, name, seed=0, epochs=5000, lr=3e-3, wd=0.01, bs=1024,
                 device="cuda", eval_every=500, log_every=200, use_wandb=False,
                 curriculum=True, curriculum_threshold=0.80, start_sig=2,
                 warm_restart_period=3000):
    """Train a single (config, seed) with fixed-layout curriculum.

    Args:
        curriculum: If True, start at start_sig and advance. If False, train at sig_digits=10 directly.
        curriculum_threshold: Advance when exact_match exceeds this on current level.
        start_sig: Starting significance level for curriculum (default 2).
        warm_restart_period: T_0 for CosineAnnealingWarmRestarts.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    model = MinimalTransformer(config).to(device)
    n_params = count_parameters(model)

    current_sig = start_sig if curriculum else MAX_DIGITS

    print(f"\n{'=' * 70}")
    print(f"{name} (seed={seed}): {n_params} params | curriculum={curriculum} | start_sig={current_sig}")
    print(f"Config: d_model={config.d_model}, n_heads={config.n_heads}, d_ff={config.d_ff}, "
          f"n_layer_repeats={config.n_layer_repeats}, norm={config.norm_type}, "
          f"act={config.activation}, pe_period={config.pe_period}")
    print(f"Fixed layout: always {SEQ_LEN} tokens (10-digit format)")
    print(f"{'=' * 70}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=warm_restart_period, T_mult=1, eta_min=1e-5
    )

    wandb_run = None
    if use_wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project="minimal-10digit-transformer",
                name=f"{name}_s{seed}",
                config={
                    "n_params": n_params, "epochs": epochs, "lr": lr, "seed": seed,
                    "curriculum": curriculum, "start_sig": start_sig,
                    "curriculum_threshold": curriculum_threshold,
                    "d_model": config.d_model, "n_layer_repeats": config.n_layer_repeats,
                    "pe_period": config.pe_period, "norm_type": config.norm_type,
                    "activation": config.activation, "n_heads": config.n_heads,
                    "d_ff": config.d_ff, "warm_restart_period": warm_restart_period,
                    "fixed_layout": True, "seq_len": SEQ_LEN,
                },
                tags=["sweep_v5", "fixed_layout"],
                reinit=True,
            )
        except Exception as e:
            print(f"  wandb init failed: {e}")

    best_acc = 0.0
    best_epoch = 0
    best_sig = current_sig
    t0 = time.time()
    level_start_time = time.time()
    level_start_epoch = 0
    level_times = {}
    curriculum_history = []

    print(f"  Starting at sig_digits={current_sig} | T_0={warm_restart_period} | lr={lr:.2e}")

    for ep in range(epochs):
        model.train()
        total_loss = 0
        n_batches = 20

        for _ in range(n_batches):
            input_ids, labels = generate_batch_v5(bs, current_sig, device)
            logits = model(input_ids)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()

        sched.step()
        avg_loss = total_loss / n_batches

        if ep % log_every == 0:
            elapsed = time.time() - t0
            print(f"  ep {ep:6d} | loss {avg_loss:.4f} | lr {opt.param_groups[0]['lr']:.2e} | "
                  f"sig={current_sig} | {elapsed:.0f}s")
            if wandb_run:
                import wandb
                wandb.log({
                    "loss": avg_loss,
                    "lr": opt.param_groups[0]["lr"],
                    "current_sig": current_sig,
                    "wall_time": elapsed,
                }, step=ep)

        if ep % eval_every == 0 and ep > 0:
            # Evaluate on current significance level
            results = evaluate_v5(model, 10000, current_sig, device, seed=12345)
            acc = results["exact_match"]
            digit_acc = results["digit_accuracy_avg"]

            print(f"  EVAL ep {ep} (sig={current_sig}): exact={acc:.4f} digit_avg={digit_acc:.4f}")
            print(f"    per-digit: {['%.3f' % d for d in results['per_digit_accuracy']]}")

            # Also evaluate on full 10-digit periodically
            if current_sig < MAX_DIGITS:
                full_results = evaluate_v5(model, 10000, MAX_DIGITS, device, seed=12345)
                print(f"    full(sig=10): exact={full_results['exact_match']:.4f} "
                      f"digit_avg={full_results['digit_accuracy_avg']:.4f}")

            # Show examples at current level
            if ep % (eval_every * 2) == 0:
                examples = get_example_predictions_v5(model, current_sig, device)
                print(f"  Examples (sig={current_sig}):\n{examples}")
                if current_sig < MAX_DIGITS:
                    examples_full = get_example_predictions_v5(model, MAX_DIGITS, device)
                    print(f"  Examples (sig=10):\n{examples_full}")

            if acc > best_acc:
                best_acc = acc
                best_epoch = ep
                best_sig = current_sig
                print(f"  ** BEST: {acc:.4f} at ep {ep} (sig={current_sig}) **")
                ckpt_dir = Path(f"checkpoints/{name}_s{seed}")
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "state_dict": model.state_dict(), "config": config,
                    "accuracy": acc, "n_params": n_params,
                    "current_sig": current_sig, "epoch": ep, "seed": seed,
                }, str(ckpt_dir / "best.pt"))

            # Curriculum advancement
            if curriculum and acc > curriculum_threshold and current_sig < MAX_DIGITS:
                level_wall_time = time.time() - level_start_time
                level_epochs_used = ep - level_start_epoch
                level_times[current_sig] = {
                    "wall_time": level_wall_time,
                    "epochs": level_epochs_used,
                    "final_acc": acc,
                }
                curriculum_history.append({
                    "epoch": ep,
                    "from_sig": current_sig,
                    "to_sig": current_sig + 1,
                    "accuracy": acc,
                })

                old_sig = current_sig
                current_sig += 1

                print(f"\n  {'*' * 60}")
                print(f"  >>> CURRICULUM ADVANCE: sig={old_sig} -> sig={current_sig} <<<")
                print(f"  >>> Accuracy was {acc:.4f} after {level_epochs_used} epochs ({level_wall_time:.0f}s)")
                print(f"  >>> Format UNCHANGED: still {SEQ_LEN} tokens (10-digit layout)")

                # Save checkpoint at curriculum transition
                ckpt_dir = Path(f"checkpoints/{name}_s{seed}")
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "state_dict": model.state_dict(), "config": config,
                    "accuracy": acc, "n_params": n_params,
                    "current_sig": current_sig, "epoch": ep, "seed": seed,
                    "curriculum_history": curriculum_history,
                }, str(ckpt_dir / f"curriculum_sig{old_sig}.pt"))

                # LR warm restart on curriculum advance
                for param_group in opt.param_groups:
                    param_group['lr'] = lr
                sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    opt, T_0=warm_restart_period, T_mult=1, eta_min=1e-5
                )

                remaining_epochs = epochs - ep
                levels_remaining = MAX_DIGITS - current_sig + 1
                print(f"  >>> LR reset to {lr:.2e}, CosineWarmRestarts T_0={warm_restart_period}")
                print(f"  >>> Remaining epochs: {remaining_epochs}, sig levels to go: {levels_remaining}")
                print(f"  {'*' * 60}\n")

                level_start_time = time.time()
                level_start_epoch = ep

            if wandb_run:
                import wandb
                log_dict = {
                    "eval_acc": acc, "best_acc": best_acc,
                    "digit_accuracy_avg": digit_acc,
                    "current_sig": current_sig,
                }
                for i, d in enumerate(results["per_digit_accuracy"]):
                    log_dict[f"digit_{i}_acc"] = d
                if current_sig < MAX_DIGITS:
                    log_dict["full_10d_exact"] = full_results["exact_match"]
                    log_dict["full_10d_digit_avg"] = full_results["digit_accuracy_avg"]
                wandb.log(log_dict, step=ep)

    # Record final level time
    level_times[current_sig] = {
        "wall_time": time.time() - level_start_time,
        "epochs": epochs - level_start_epoch,
        "final_acc": best_acc,
    }

    total_wall_time = time.time() - t0

    # Final comprehensive evaluation
    print(f"\n{'=' * 70}")
    print(f"FINAL EVALUATION: {name} (seed={seed})")
    print(f"{'=' * 70}")

    per_sig_results = evaluate_all_sig_levels(model, device, max_sig=MAX_DIGITS, n_samples=10000)

    print(f"\n  Per-significance-level accuracy:")
    print(f"  {'Sig':>8} | {'Exact Match':>12} | {'Digit Avg':>10}")
    print(f"  {'-' * 8}-+-{'-' * 12}-+-{'-' * 10}")
    for s in range(1, MAX_DIGITS + 1):
        r = per_sig_results[s]
        marker = " <-- current" if s == current_sig else ""
        print(f"  {s:>8} | {r['exact_match']:>12.4f} | {r['digit_accuracy_avg']:>10.4f}{marker}")

    final_results = evaluate_v5(model, 50000, MAX_DIGITS, device, seed=99999)
    print(f"\n  FINAL (sig=10, 50k samples): exact={final_results['exact_match']:.4f} "
          f"digit_avg={final_results['digit_accuracy_avg']:.4f}")
    print(f"  best: {best_acc:.4f} at ep {best_epoch} (sig={best_sig})")
    print(f"  total wall time: {total_wall_time:.0f}s ({total_wall_time / 60:.1f}min)")

    if curriculum_history:
        print(f"\n  Curriculum history:")
        for h in curriculum_history:
            print(f"    ep {h['epoch']}: sig={h['from_sig']} -> sig={h['to_sig']} (acc={h['accuracy']:.4f})")

    if level_times:
        print(f"\n  Time per curriculum level:")
        for level, info in sorted(level_times.items()):
            if isinstance(info, dict):
                print(f"    sig={level}: {info['wall_time']:.0f}s ({info['wall_time'] / 60:.1f}min), "
                      f"{info['epochs']} epochs, final_acc={info['final_acc']:.4f}")
            else:
                print(f"    sig={level}: {info:.0f}s")

    examples = get_example_predictions_v5(model, MAX_DIGITS, device, n_examples=8)
    print(f"  Final examples (10-digit):\n{examples}")

    if wandb_run:
        import wandb
        for s, r in per_sig_results.items():
            wandb.log({f"final_sig{s}_exact": r["exact_match"],
                       f"final_sig{s}_digit_avg": r["digit_accuracy_avg"]})
        wandb.finish()

    return {
        "name": name, "seed": seed, "n_params": n_params,
        "best_acc": best_acc, "best_epoch": best_epoch, "best_sig": best_sig,
        "final_acc": final_results["exact_match"],
        "final_digit_acc": final_results["digit_accuracy_avg"],
        "final_per_digit": final_results["per_digit_accuracy"],
        "per_sig_results": {
            str(s): {"exact_match": r["exact_match"], "digit_accuracy_avg": r["digit_accuracy_avg"]}
            for s, r in per_sig_results.items()
        },
        "curriculum_history": curriculum_history,
        "level_times": {str(k): v for k, v in level_times.items()},
        "total_wall_time": total_wall_time,
        "max_sig_reached": current_sig,
        "curriculum": curriculum,
    }


# ---------------------------------------------------------------------------
# Phase runners
# ---------------------------------------------------------------------------

def run_baseline(device="cuda", use_wandb=False):
    """Baseline: d_model=16 trained DIRECTLY on full 10-digit (no curriculum).

    This validates whether the architecture can learn 10-digit addition at all
    when given enough capacity and no curriculum confusion.
    """
    print("\n" + "#" * 70)
    print("# BASELINE: d16 direct 10-digit training (NO curriculum)")
    print("# 10000 epochs, always sig_digits=10, fixed 33-token layout")
    print("#" * 70)

    cname, cfg, cfg_lr = get_d16_config()
    seed = 0

    tmp_model = MinimalTransformer(cfg)
    n_p = count_parameters(tmp_model)
    del tmp_model
    print(f"  Config: {cname} ({n_p} params, lr={cfg_lr})")

    try:
        r = train_one_v5(
            cfg, f"{cname}_baseline_direct", seed=seed,
            epochs=10000, lr=cfg_lr, wd=0.01, bs=1024,
            device=device, eval_every=500, log_every=200,
            use_wandb=use_wandb,
            curriculum=False,  # NO curriculum - train directly on 10-digit
            warm_restart_period=3000,
        )
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback; traceback.print_exc()
        r = {"name": cname, "seed": seed, "error": str(e)}

    # Save
    existing = load_results()
    existing["baseline"] = r
    save_results(existing)

    # Summary
    print(f"\n\n{'=' * 70}")
    print("BASELINE RESULTS (d16 direct 10-digit)")
    print(f"{'=' * 70}")
    if "error" not in r:
        print(f"  {r['name']} seed={r['seed']}: final_acc={r['final_acc']:.4f} "
              f"digit_avg={r['final_digit_acc']:.4f} "
              f"wall_time={r['total_wall_time']:.0f}s")
    else:
        print(f"  ERROR: {r['error']}")

    return r


def run_seed_tournament(device="cuda", use_wandb=False):
    """Seed tournament: d7 x 3 seeds with fixed-layout curriculum, 5000 epochs.

    Also includes d8 and d16 with curriculum for comparison.
    """
    print("\n" + "#" * 70)
    print("# SEED TOURNAMENT: Fixed-layout curriculum")
    print("# d7 x seeds {1,3,4}, d8 x seed 2, d16 x seed 0")
    print("# 5000 epochs with curriculum, sig_digits 2->10")
    print("#" * 70)

    runs = []

    # d7 with best seeds from v4
    d7_name, d7_cfg, d7_lr = get_d7_config()
    for seed in [1, 3, 4]:
        runs.append((d7_cfg, f"{d7_name}_curriculum", seed, d7_lr))

    # d8 baseline with curriculum
    d8_name, d8_cfg, d8_lr = get_d8_config()
    runs.append((d8_cfg, f"{d8_name}_curriculum", 2, d8_lr))

    # d16 with curriculum
    d16_name, d16_cfg, d16_lr = get_d16_config()
    runs.append((d16_cfg, f"{d16_name}_curriculum", 0, d16_lr))

    all_results = []
    global_t0 = time.time()

    for cfg, name, seed, lr in runs:
        tmp_model = MinimalTransformer(cfg)
        n_p = count_parameters(tmp_model)
        del tmp_model
        print(f"\n  --- {name} seed={seed} ({n_p} params) ---")

        try:
            r = train_one_v5(
                cfg, name, seed=seed,
                epochs=5000, lr=lr, wd=0.01, bs=1024,
                device=device, eval_every=500, log_every=200,
                use_wandb=use_wandb,
                curriculum=True, curriculum_threshold=0.80, start_sig=2,
                warm_restart_period=3000,
            )
            all_results.append(r)
        except Exception as e:
            print(f"  ERROR in {name} seed={seed}: {e}")
            import traceback; traceback.print_exc()
            all_results.append({"name": name, "seed": seed, "error": str(e)})

        # Save intermediate
        existing = load_results()
        existing["seed_tournament"] = all_results
        save_results(existing)

    total_time = time.time() - global_t0

    # Print summary
    print(f"\n\n{'=' * 70}")
    print("SEED TOURNAMENT RESULTS (sorted by max sig_digits reached)")
    print(f"{'=' * 70}")
    print(f"Total wall time: {total_time:.0f}s ({total_time / 60:.1f}min)\n")

    valid = [r for r in all_results if "error" not in r]
    valid.sort(key=lambda x: (x["max_sig_reached"], x["best_acc"]), reverse=True)

    header = (f"{'Rank':>4} | {'Config':>25} | {'Seed':>4} | {'Params':>6} | "
              f"{'Max Sig':>7} | {'Best Acc':>8} | {'Best Ep':>7} | "
              f"{'10d Exact':>9} | {'Time':>6}")
    print(header)
    print("-" * len(header))
    for i, r in enumerate(valid, 1):
        print(f"{i:>4} | {r['name']:>25} | {r['seed']:>4} | {r['n_params']:>6} | "
              f"{r['max_sig_reached']:>7} | {r['best_acc']:>8.4f} | {r['best_epoch']:>7} | "
              f"{r['final_acc']:>9.4f} | {r['total_wall_time']:>5.0f}s")

    return all_results


# ---------------------------------------------------------------------------
# Results I/O
# ---------------------------------------------------------------------------

RESULTS_PATH = Path(__file__).parent / "sweep_v5_results.json"


def save_results(data):
    """Save results to JSON file."""
    with open(RESULTS_PATH, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_results():
    """Load results from JSON file, or return empty dict."""
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            return json.load(f)
    return {}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Sweep v5: Fixed-layout curriculum")
    parser.add_argument("--phase", type=str, default="all",
                        choices=["baseline", "seed_tournament", "all"],
                        help="Which phase to run")
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Phase: {args.phase}")
    print(f"Fixed layout: {SEQ_LEN} tokens (10-digit format, always)")

    if args.phase == "baseline" or args.phase == "all":
        baseline_result = run_baseline(device=device, use_wandb=args.wandb)

    if args.phase == "seed_tournament" or args.phase == "all":
        tournament_results = run_seed_tournament(device=device, use_wandb=args.wandb)

    print(f"\nDone! Results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
