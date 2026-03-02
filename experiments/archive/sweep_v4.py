"""Sweep v4: Multi-seed tournament for finding optimal architecture + initialization.

Design:
  Phase 1 (Seed Tournament): 3 configs x 5 seeds, 3000 epochs on 2-digit addition.
      Identifies which (config, seed) combinations learn fastest.
  Phase 2 (Long Training): Top 3 combos from Phase 1, 30000 epochs with curriculum
      from 2-digit to 10-digit addition.
  Phase 3 (Baseline): d_model=16 baseline trained with curriculum to 10-digit.

Key improvements over sweep_v3:
  - Seed-aware: explores initialization space, not just architecture space
  - Tournament selection: cheap Phase 1 filters before expensive Phase 2
  - CosineAnnealingWarmRestarts with T_0=2000 for finer-grained LR cycling
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
# Reused from sweep_v3
# ---------------------------------------------------------------------------

def generate_batch(batch_size, max_digits, device):
    """Generate a batch of addition problems on GPU. Output is LSB-first (reversed)."""
    a = torch.randint(0, 10**max_digits, (batch_size,), device=device, dtype=torch.long)
    b = torch.randint(0, 10**max_digits, (batch_size,), device=device, dtype=torch.long)
    c = a + b

    a_digits = []
    b_digits = []
    c_digits = []
    a_tmp, b_tmp, c_tmp = a.clone(), b.clone(), c.clone()
    for _ in range(max_digits):
        a_digits.append(a_tmp % 10)
        b_digits.append(b_tmp % 10)
        c_digits.append(c_tmp % 10)
        a_tmp //= 10
        b_tmp //= 10
        c_tmp //= 10
    c_digits.append(c_tmp % 10)

    a_t = torch.stack(a_digits, dim=1).flip(1)  # MSB first for input
    b_t = torch.stack(b_digits, dim=1).flip(1)  # MSB first for input
    c_t = torch.stack(c_digits, dim=1)           # LSB first for output

    plus = torch.full((batch_size, 1), 10, device=device, dtype=torch.long)
    eq = torch.full((batch_size, 1), 11, device=device, dtype=torch.long)

    input_ids = torch.cat([a_t, plus, b_t, eq, c_t], dim=1)
    input_len = max_digits * 2 + 2

    labels = torch.full_like(input_ids, -100)
    labels[:, input_len - 1:-1] = c_t

    return input_ids, labels


def evaluate(model, n_samples, max_digits, device, seed=42):
    """Evaluate exact-match accuracy. Returns dict with overall and per-digit metrics."""
    model.eval()

    input_len = max_digits * 2 + 2
    output_len = max_digits + 1
    correct = 0
    total = 0
    digit_correct = torch.zeros(output_len, device=device)
    digit_total = torch.zeros(output_len, device=device)

    with torch.no_grad():
        for _ in range(0, n_samples, 2048):
            bs = min(2048, n_samples - total)
            if bs <= 0:
                break
            input_ids, labels = generate_batch(bs, max_digits, device)
            logits = model(input_ids)
            preds = logits[:, input_len - 1:-1, :].argmax(dim=-1)
            targets = input_ids[:, input_len:]
            correct += (preds == targets).all(dim=1).sum().item()

            matches = (preds == targets)
            digit_correct += matches.sum(dim=0).float()
            digit_total += torch.full((output_len,), float(bs), device=device)

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


def get_example_predictions(model, max_digits, device, n_examples=5, seed=99):
    """Get a few example predictions for logging."""
    model.eval()
    input_len = max_digits * 2 + 2

    with torch.no_grad():
        input_ids, labels = generate_batch(n_examples, max_digits, device)
        logits = model(input_ids)
        preds = logits[:, input_len - 1:-1, :].argmax(dim=-1)
        targets = input_ids[:, input_len:]

    examples = []
    for i in range(n_examples):
        seq = input_ids[i].cpu().tolist()
        a_digits = seq[:max_digits]
        b_digits = seq[max_digits + 1: max_digits * 2 + 1]
        pred_digits = preds[i].cpu().tolist()
        target_digits = targets[i].cpu().tolist()

        a_str = "".join(str(d) for d in a_digits)
        b_str = "".join(str(d) for d in b_digits)
        pred_str = "".join(str(d) for d in reversed(pred_digits))
        target_str = "".join(str(d) for d in reversed(target_digits))
        match = "OK" if pred_digits == target_digits else "WRONG"

        examples.append(f"  {a_str} + {b_str} = {pred_str} (expected {target_str}) [{match}]")

    return "\n".join(examples)


def evaluate_all_digit_levels(model, device, max_digits=10, n_samples=10000):
    """Evaluate the model on each digit count separately (1-digit through max_digits)."""
    results = {}
    for d in range(1, max_digits + 1):
        r = evaluate(model, n_samples, d, device, seed=12345 + d)
        results[d] = r
    return results


# ---------------------------------------------------------------------------
# Configurations
# ---------------------------------------------------------------------------

def get_phase1_configs():
    """Return the 3 tournament configs for Phase 1."""
    configs = []

    # d6_fr_15rep (324 params)
    configs.append(("d6_fr_15rep", TransformerConfig(
        d_model=6, n_heads=1, n_layers=1, d_ff=6,
        share_layers=True, n_layer_repeats=15,
        norm_type="rmsnorm", activation="relu",
        pe_period=11.0,
    ), 3e-3))

    # d7_fr_15rep (~441 params)
    configs.append(("d7_fr_15rep", TransformerConfig(
        d_model=7, n_heads=1, n_layers=1, d_ff=7,
        share_layers=True, n_layer_repeats=15,
        norm_type="rmsnorm", activation="relu",
        pe_period=11.0,
    ), 3e-3))

    # d8_fr_15rep (~528 params)
    configs.append(("d8_fr_15rep", TransformerConfig(
        d_model=8, n_heads=1, n_layers=1, d_ff=8,
        share_layers=True, n_layer_repeats=15,
        norm_type="rmsnorm", activation="relu",
        pe_period=11.0,
    ), 1e-3))

    return configs


def get_phase3_config():
    """Return the d16 baseline config for Phase 3."""
    return ("d16_fr_10rep", TransformerConfig(
        d_model=16, n_heads=1, n_layers=1, d_ff=16,
        share_layers=True, n_layer_repeats=10,
        norm_type="rmsnorm", activation="relu",
        pe_period=11.0,
    ), 1e-3)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one(config, name, seed=0, epochs=3000, lr=3e-3, wd=0.01, bs=1024,
              device="cuda", eval_every=500, log_every=200, use_wandb=False,
              curriculum=False, curriculum_threshold=0.5, target_digits=2,
              warm_restart_period=2000):
    """Train a single (config, seed) combination.

    Returns a dict with training results.
    """
    # Seed for reproducibility
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    model = MinimalTransformer(config).to(device)
    n_params = count_parameters(model)
    print(f"\n{'='*70}")
    print(f"{name} (seed={seed}): {n_params} params | curriculum={curriculum} | target_digits={target_digits}")
    print(f"Config: d_model={config.d_model}, n_heads={config.n_heads}, d_ff={config.d_ff}, "
          f"n_layer_repeats={config.n_layer_repeats}, norm={config.norm_type}, "
          f"act={config.activation}, pe_period={config.pe_period}")
    print(f"{'='*70}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=warm_restart_period, T_mult=1, eta_min=1e-5
    )

    current_digits = 2 if curriculum else target_digits

    wandb_run = None
    if use_wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project="minimal-10digit-transformer",
                name=f"{name}_s{seed}",
                config={
                    "n_params": n_params, "epochs": epochs, "lr": lr, "seed": seed,
                    "curriculum": curriculum, "target_digits": target_digits,
                    "d_model": config.d_model, "n_layer_repeats": config.n_layer_repeats,
                    "pe_period": config.pe_period, "norm_type": config.norm_type,
                    "activation": config.activation, "n_heads": config.n_heads,
                    "d_ff": config.d_ff, "warm_restart_period": warm_restart_period,
                },
                tags=["sweep_v4"],
                reinit=True,
            )
        except Exception as e:
            print(f"  wandb init failed: {e}")

    best_acc = 0.0
    best_epoch = 0
    t0 = time.time()
    level_start_time = time.time()
    level_start_epoch = 0
    level_times = {}
    curriculum_history = []

    print(f"  Starting at {current_digits}-digit | warm_restart_period={warm_restart_period} | lr={lr:.2e}")

    for ep in range(epochs):
        model.train()
        total_loss = 0
        n_batches = 20

        for _ in range(n_batches):
            input_ids, labels = generate_batch(bs, current_digits, device)
            logits = model(input_ids)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
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
                  f"digits={current_digits} | {elapsed:.0f}s")
            if wandb_run:
                import wandb
                wandb.log({
                    "loss": avg_loss,
                    "lr": opt.param_groups[0]["lr"],
                    "current_digits": current_digits,
                    "wall_time": elapsed,
                }, step=ep)

        if ep % eval_every == 0 and ep > 0:
            results = evaluate(model, 10000, current_digits, device, seed=12345)
            acc = results["exact_match"]
            digit_acc = results["digit_accuracy_avg"]

            print(f"  EVAL ep {ep} (digits={current_digits}): exact={acc:.4f} digit_avg={digit_acc:.4f}")
            print(f"    per-digit: {['%.3f' % d for d in results['per_digit_accuracy']]}")

            if current_digits != target_digits:
                target_results = evaluate(model, 10000, target_digits, device, seed=12345)
                print(f"    target({target_digits}d): exact={target_results['exact_match']:.4f} "
                      f"digit_avg={target_results['digit_accuracy_avg']:.4f}")

            if ep % (eval_every * 2) == 0:
                examples = get_example_predictions(model, current_digits, device)
                print(f"  Examples (digits={current_digits}):\n{examples}")

            if acc > best_acc:
                best_acc = acc
                best_epoch = ep
                print(f"  ** BEST: {acc:.4f} at ep {ep} **")
                ckpt_dir = Path(f"checkpoints/{name}_s{seed}")
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "state_dict": model.state_dict(), "config": config,
                    "accuracy": acc, "n_params": n_params,
                    "current_digits": current_digits, "epoch": ep, "seed": seed,
                }, str(ckpt_dir / "best.pt"))

            # Curriculum advancement
            if curriculum and acc > curriculum_threshold and current_digits < target_digits:
                level_wall_time = time.time() - level_start_time
                level_epochs_used = ep - level_start_epoch
                level_times[current_digits] = level_wall_time
                curriculum_history.append((ep, current_digits, acc))

                old_digits = current_digits
                current_digits += 1

                print(f"\n  {'*'*60}")
                print(f"  >>> CURRICULUM ADVANCE: {old_digits}-digit -> {current_digits}-digit <<<")
                print(f"  >>> Accuracy was {acc:.4f} after {level_epochs_used} epochs ({level_wall_time:.0f}s)")

                ckpt_dir = Path(f"checkpoints/{name}_s{seed}")
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "state_dict": model.state_dict(), "config": config,
                    "accuracy": acc, "n_params": n_params,
                    "current_digits": current_digits, "epoch": ep, "seed": seed,
                    "curriculum_history": curriculum_history,
                }, str(ckpt_dir / f"curriculum_{old_digits}d.pt"))

                # LR warm restart on curriculum advance
                for param_group in opt.param_groups:
                    param_group['lr'] = lr
                sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    opt, T_0=warm_restart_period, T_mult=1, eta_min=1e-5
                )

                remaining_epochs = epochs - ep
                levels_remaining = target_digits - current_digits + 1
                print(f"  >>> LR reset to {lr:.2e}, CosineWarmRestarts T_0={warm_restart_period}")
                print(f"  >>> Remaining epochs: {remaining_epochs}, levels to go: {levels_remaining}")
                print(f"  {'*'*60}\n")

                level_start_time = time.time()
                level_start_epoch = ep

            if wandb_run:
                import wandb
                log_dict = {
                    "eval_acc": acc, "best_acc": best_acc,
                    "digit_accuracy_avg": digit_acc,
                }
                for i, d in enumerate(results["per_digit_accuracy"]):
                    log_dict[f"digit_{i}_acc"] = d
                if current_digits != target_digits:
                    log_dict["target_eval_acc"] = target_results["exact_match"]
                wandb.log(log_dict, step=ep)

    # Record final level time
    level_times[current_digits] = time.time() - level_start_time

    total_wall_time = time.time() - t0

    # Final evaluation
    print(f"\n{'='*70}")
    print(f"FINAL EVALUATION: {name} (seed={seed})")
    print(f"{'='*70}")

    per_level_results = evaluate_all_digit_levels(model, device, max_digits=target_digits, n_samples=10000)

    print(f"\n  Per-digit-count accuracy:")
    print(f"  {'Digits':>8} | {'Exact Match':>12} | {'Digit Avg':>10}")
    print(f"  {'-'*8}-+-{'-'*12}-+-{'-'*10}")
    for d in range(1, target_digits + 1):
        r = per_level_results[d]
        print(f"  {d:>8} | {r['exact_match']:>12.4f} | {r['digit_accuracy_avg']:>10.4f}")

    final_results = evaluate(model, 50000, target_digits, device, seed=99999)
    print(f"\n  FINAL ({target_digits}d, 50k samples): exact={final_results['exact_match']:.4f} "
          f"digit_avg={final_results['digit_accuracy_avg']:.4f}")
    print(f"  best: {best_acc:.4f} at ep {best_epoch}")
    print(f"  total wall time: {total_wall_time:.0f}s ({total_wall_time/60:.1f}min)")

    if curriculum_history:
        print(f"\n  Curriculum history:")
        for ep_h, digits_h, acc_h in curriculum_history:
            print(f"    ep {ep_h}: advanced from {digits_h}-digit (acc={acc_h:.4f})")

    if level_times:
        print(f"\n  Time per curriculum level:")
        for level, t in sorted(level_times.items()):
            print(f"    {level}-digit: {t:.0f}s ({t/60:.1f}min)")

    examples = get_example_predictions(model, target_digits, device, n_examples=8)
    print(f"  Final examples ({target_digits}-digit):\n{examples}")

    if wandb_run:
        import wandb
        for d, r in per_level_results.items():
            wandb.log({f"final_{d}d_exact": r["exact_match"], f"final_{d}d_digit_avg": r["digit_accuracy_avg"]})
        wandb.finish()

    return {
        "name": name, "seed": seed, "n_params": n_params,
        "best_acc": best_acc, "best_epoch": best_epoch,
        "final_acc": final_results["exact_match"],
        "final_digit_acc": final_results["digit_accuracy_avg"],
        "final_per_digit": final_results["per_digit_accuracy"],
        "per_level_results": {str(d): {"exact_match": r["exact_match"], "digit_accuracy_avg": r["digit_accuracy_avg"]}
                              for d, r in per_level_results.items()},
        "curriculum_history": curriculum_history,
        "level_times": {str(k): v for k, v in level_times.items()},
        "total_wall_time": total_wall_time,
        "max_digits_reached": current_digits,
    }


# ---------------------------------------------------------------------------
# Phase runners
# ---------------------------------------------------------------------------

def run_phase1(device="cuda", use_wandb=False):
    """Phase 1: Seed tournament - 3 configs x 5 seeds, 3000 epochs on 2-digit."""
    print("\n" + "#" * 70)
    print("# PHASE 1: SEED TOURNAMENT")
    print("# 3 configs x 5 seeds = 15 runs, 3000 epochs each on 2-digit")
    print("#" * 70)

    configs = get_phase1_configs()
    seeds = list(range(5))
    all_results = []

    global_t0 = time.time()

    for cfg_name, cfg, cfg_lr in configs:
        tmp_model = MinimalTransformer(cfg)
        n_p = count_parameters(tmp_model)
        del tmp_model
        print(f"\n  Config: {cfg_name} ({n_p} params, lr={cfg_lr})")

        for seed in seeds:
            print(f"\n  --- {cfg_name} seed={seed} ---")
            try:
                r = train_one(
                    cfg, cfg_name, seed=seed,
                    epochs=3000, lr=cfg_lr, wd=0.01, bs=1024,
                    device=device, eval_every=500, log_every=200,
                    use_wandb=use_wandb,
                    curriculum=False, target_digits=2,
                    warm_restart_period=2000,
                )
                all_results.append(r)
            except Exception as e:
                print(f"  ERROR in {cfg_name} seed={seed}: {e}")
                import traceback; traceback.print_exc()
                all_results.append({"name": cfg_name, "seed": seed, "error": str(e)})

            # Save intermediate results
            save_results({"phase1": all_results})

    total_time = time.time() - global_t0

    # Print summary table sorted by accuracy
    print(f"\n\n{'='*70}")
    print("PHASE 1 RESULTS (sorted by 2-digit exact match accuracy)")
    print(f"{'='*70}")
    print(f"Total wall time: {total_time:.0f}s ({total_time/60:.1f}min)\n")

    valid = [r for r in all_results if "error" not in r]
    valid.sort(key=lambda x: x["best_acc"], reverse=True)

    header = f"{'Rank':>4} | {'Config':>15} | {'Seed':>4} | {'Params':>6} | {'Best Acc':>8} | {'Best Ep':>7} | {'Time':>6}"
    print(header)
    print("-" * len(header))
    for i, r in enumerate(valid, 1):
        print(f"{i:>4} | {r['name']:>15} | {r['seed']:>4} | {r['n_params']:>6} | "
              f"{r['best_acc']:>8.4f} | {r['best_epoch']:>7} | {r['total_wall_time']:>5.0f}s")

    # Identify top 3
    top3 = valid[:3]
    print(f"\nTop 3 for Phase 2:")
    for i, r in enumerate(top3, 1):
        print(f"  {i}. {r['name']} seed={r['seed']} acc={r['best_acc']:.4f}")

    return all_results, top3


def run_phase2(top3_combos, device="cuda", use_wandb=False):
    """Phase 2: Long training with top seeds from Phase 1.

    Args:
        top3_combos: list of dicts with 'name' and 'seed' keys from Phase 1
    """
    print("\n" + "#" * 70)
    print("# PHASE 2: LONG TRAINING WITH TOP SEEDS")
    print("# 30000 epochs with curriculum 2->10 digit")
    print("#" * 70)

    # Map config names to configs
    all_configs = {name: (cfg, lr) for name, cfg, lr in get_phase1_configs()}
    results = []

    for combo in top3_combos:
        cname = combo["name"]
        seed = combo["seed"]
        if cname not in all_configs:
            print(f"  WARNING: config {cname} not found, skipping")
            continue
        cfg, cfg_lr = all_configs[cname]

        print(f"\n  --- Phase 2: {cname} seed={seed} ---")
        try:
            r = train_one(
                cfg, f"{cname}_phase2", seed=seed,
                epochs=30000, lr=cfg_lr, wd=0.01, bs=1024,
                device=device, eval_every=500, log_every=200,
                use_wandb=use_wandb,
                curriculum=True, curriculum_threshold=0.5, target_digits=10,
                warm_restart_period=2000,
            )
            results.append(r)
        except Exception as e:
            print(f"  ERROR in {cname} seed={seed}: {e}")
            import traceback; traceback.print_exc()
            results.append({"name": cname, "seed": seed, "error": str(e)})

        # Save intermediate
        existing = load_results()
        existing["phase2"] = results
        save_results(existing)

    # Summary
    print(f"\n\n{'='*70}")
    print("PHASE 2 RESULTS")
    print(f"{'='*70}")
    for r in results:
        if "error" in r:
            print(f"  {r['name']} seed={r['seed']}: ERROR - {r['error']}")
        else:
            print(f"  {r['name']} seed={r['seed']}: max_digits={r['max_digits_reached']} "
                  f"final_acc={r['final_acc']:.4f} wall_time={r['total_wall_time']:.0f}s")
            if r.get("curriculum_history"):
                for ep_h, d_h, a_h in r["curriculum_history"]:
                    print(f"    ep {ep_h}: {d_h}d -> {d_h+1}d (acc={a_h:.4f})")

    return results


def run_phase3(device="cuda", use_wandb=False):
    """Phase 3: d_model=16 baseline with curriculum to 10-digit."""
    print("\n" + "#" * 70)
    print("# PHASE 3: BASELINE (d_model=16)")
    print("# 20000 epochs with curriculum 2->10 digit")
    print("#" * 70)

    cname, cfg, cfg_lr = get_phase3_config()
    seed = 0

    tmp_model = MinimalTransformer(cfg)
    n_p = count_parameters(tmp_model)
    del tmp_model
    print(f"  Config: {cname} ({n_p} params, lr={cfg_lr})")

    try:
        r = train_one(
            cfg, f"{cname}_baseline", seed=seed,
            epochs=20000, lr=cfg_lr, wd=0.01, bs=1024,
            device=device, eval_every=500, log_every=200,
            use_wandb=use_wandb,
            curriculum=True, curriculum_threshold=0.5, target_digits=10,
            warm_restart_period=2000,
        )
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback; traceback.print_exc()
        r = {"name": cname, "seed": seed, "error": str(e)}

    # Save
    existing = load_results()
    existing["phase3"] = r
    save_results(existing)

    # Summary
    print(f"\n\n{'='*70}")
    print("PHASE 3 RESULTS (baseline)")
    print(f"{'='*70}")
    if "error" not in r:
        print(f"  {r['name']} seed={r['seed']}: max_digits={r['max_digits_reached']} "
              f"final_acc={r['final_acc']:.4f} wall_time={r['total_wall_time']:.0f}s")
        if r.get("curriculum_history"):
            for ep_h, d_h, a_h in r["curriculum_history"]:
                print(f"    ep {ep_h}: {d_h}d -> {d_h+1}d (acc={a_h:.4f})")
    else:
        print(f"  ERROR: {r['error']}")

    return r


# ---------------------------------------------------------------------------
# Results I/O
# ---------------------------------------------------------------------------

RESULTS_PATH = Path(__file__).parent / "sweep_v4_results.json"


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
    parser = argparse.ArgumentParser(description="Sweep v4: Multi-seed tournament")
    parser.add_argument("--phase", type=str, default="all",
                        choices=["1", "2", "3", "all"],
                        help="Which phase to run (1=tournament, 2=long training, 3=baseline, all=everything)")
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Phase: {args.phase}")

    if args.phase == "1" or args.phase == "all":
        phase1_results, top3 = run_phase1(device=device, use_wandb=args.wandb)
        # Save top 3 info for Phase 2
        existing = load_results()
        existing["phase1"] = phase1_results
        existing["top3"] = [{"name": r["name"], "seed": r["seed"], "best_acc": r["best_acc"]} for r in top3]
        save_results(existing)

    if args.phase == "2" or args.phase == "all":
        # Load top 3 from saved results if running phase 2 separately
        if args.phase == "2":
            existing = load_results()
            top3 = existing.get("top3", [])
            if not top3:
                print("ERROR: No Phase 1 results found. Run phase 1 first.")
                return
        phase2_results = run_phase2(top3, device=device, use_wandb=args.wandb)

    if args.phase == "3" or args.phase == "all":
        phase3_result = run_phase3(device=device, use_wandb=args.wandb)

    print("\nDone! Results saved to experiments/sweep_v4_results.json")


if __name__ == "__main__":
    main()
