"""Sweep v3 with LR warm restarts on curriculum advancement.

Key improvements over sweep_v2:
1. LR warm restart: resets optimizer LR and creates a new cosine schedule when
   curriculum advances to a new digit level. This fixes the issue where the LR
   was already decayed to ~1e-5 when the model needed to learn harder problems.
2. Focused configurations: only the most promising architectures from sweep_v2.
3. Checkpoint on curriculum advancement (not just best accuracy).
4. Per-curriculum-level wall time tracking.
5. Final per-digit-count evaluation (1-digit through 10-digit).
6. Summary table at the end.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from minimal10digittransformer.model.transformer import MinimalTransformer, TransformerConfig, count_parameters


def generate_batch(batch_size, max_digits, device):
    """Generate a batch of addition problems on GPU. Output is LSB-first (reversed)."""
    a = torch.randint(0, 10**max_digits, (batch_size,), device=device, dtype=torch.long)
    b = torch.randint(0, 10**max_digits, (batch_size,), device=device, dtype=torch.long)
    c = a + b

    # Extract digits (LSB first)
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

    # For causal LM: labels[t] = input_ids[t+1] (next token prediction)
    # logits[input_len-1] should predict input_ids[input_len] = C0
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

            # Per-digit accuracy
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
        # Output is LSB-first, so reverse for display
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


def train_one(config, name, epochs=30000, lr=3e-3, wd=0.01, bs=1024,
              device="cuda", eval_every=500, log_every=100, use_wandb=False,
              curriculum=True, curriculum_threshold=0.5, target_digits=10):
    """Train a single configuration with LR warm restarts on curriculum advancement."""
    model = MinimalTransformer(config).to(device)
    n_params = count_parameters(model)
    print(f"\n{'='*70}")
    print(f"{name}: {n_params} params | curriculum={curriculum} | target_digits={target_digits}")
    print(f"Config: d_model={config.d_model}, n_heads={config.n_heads}, d_ff={config.d_ff}, "
          f"n_layer_repeats={config.n_layer_repeats}, norm={config.norm_type}, "
          f"act={config.activation}, pe={config.pe_type}, ffn={config.ffn_type}")
    print(f"{'='*70}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    # Use CosineAnnealingWarmRestarts for automatic LR cycling within each
    # curriculum level. T_0=3000 means the LR resets every 3000 epochs, so
    # even if the model hasn't advanced curriculum, it gets fresh learning rate.
    # On curriculum advance, we also explicitly reset the scheduler.
    current_digits = 2 if curriculum else target_digits
    warm_restart_period = 3000
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=warm_restart_period, T_mult=1, eta_min=1e-5
    )

    wandb_run = None
    if use_wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project="minimal-10digit-transformer",
                name=name,
                config={
                    "n_params": n_params, "epochs": epochs, "lr": lr,
                    "curriculum": curriculum, "target_digits": target_digits,
                    "d_model": config.d_model, "n_layer_repeats": config.n_layer_repeats,
                    "rank": config.rank, "pe_period": config.pe_period,
                    "ffn_type": config.ffn_type, "norm_type": config.norm_type,
                    "activation": config.activation, "pe_type": config.pe_type,
                    "n_heads": config.n_heads, "d_ff": config.d_ff,
                    "use_bias": config.use_bias,
                },
                tags=["sweep_v3"],
                reinit=True,
            )
        except Exception as e:
            print(f"  wandb init failed: {e}")

    best_acc = 0.0
    best_epoch = 0
    t0 = time.time()
    level_start_time = time.time()
    level_start_epoch = 0
    level_times = {}  # digit_level -> wall_time_seconds
    curriculum_history = []  # list of (epoch, digit_level, accuracy)

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
            # Evaluate on current curriculum level
            results = evaluate(model, 10000, current_digits, device, seed=12345)
            acc = results["exact_match"]
            digit_acc = results["digit_accuracy_avg"]

            print(f"  EVAL ep {ep} (digits={current_digits}): exact={acc:.4f} digit_avg={digit_acc:.4f}")
            print(f"    per-digit: {['%.3f' % d for d in results['per_digit_accuracy']]}")

            # Also evaluate on target digits for tracking
            if current_digits != target_digits:
                target_results = evaluate(model, 10000, target_digits, device, seed=12345)
                print(f"    target({target_digits}d): exact={target_results['exact_match']:.4f} "
                      f"digit_avg={target_results['digit_accuracy_avg']:.4f}")

            # Log examples
            if ep % (eval_every * 2) == 0:
                examples = get_example_predictions(model, current_digits, device)
                print(f"  Examples (digits={current_digits}):\n{examples}")

            if acc > best_acc:
                best_acc = acc
                best_epoch = ep
                print(f"  ** BEST: {acc:.4f} at ep {ep} **")
                Path(f"checkpoints/{name}").mkdir(parents=True, exist_ok=True)
                torch.save({
                    "state_dict": model.state_dict(), "config": config,
                    "accuracy": acc, "n_params": n_params,
                    "current_digits": current_digits, "epoch": ep,
                }, f"checkpoints/{name}/best.pt")

            # Curriculum: advance if accuracy > threshold
            if curriculum and acc > curriculum_threshold and current_digits < target_digits:
                # Record time spent on this level
                level_wall_time = time.time() - level_start_time
                level_epochs_used = ep - level_start_epoch
                level_times[current_digits] = level_wall_time
                curriculum_history.append((ep, current_digits, acc))

                old_digits = current_digits
                current_digits += 1

                print(f"\n  {'*'*60}")
                print(f"  >>> CURRICULUM ADVANCE: {old_digits}-digit -> {current_digits}-digit <<<")
                print(f"  >>> Accuracy was {acc:.4f} after {level_epochs_used} epochs ({level_wall_time:.0f}s)")

                # Save curriculum checkpoint
                Path(f"checkpoints/{name}").mkdir(parents=True, exist_ok=True)
                torch.save({
                    "state_dict": model.state_dict(), "config": config,
                    "accuracy": acc, "n_params": n_params,
                    "current_digits": current_digits, "epoch": ep,
                    "curriculum_history": curriculum_history,
                }, f"checkpoints/{name}/curriculum_{old_digits}d.pt")

                # ===== LR WARM RESTART =====
                # Reset optimizer LR to initial value
                for param_group in opt.param_groups:
                    param_group['lr'] = lr

                # Create fresh CosineAnnealingWarmRestarts scheduler.
                # T_0=3000 means auto-restart every 3000 epochs even without
                # curriculum advance, preventing the LR from staying at minimum.
                sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    opt, T_0=warm_restart_period, T_mult=1, eta_min=1e-5
                )

                remaining_epochs = epochs - ep
                levels_remaining = target_digits - current_digits + 1
                print(f"  >>> LR reset to {lr:.2e}, CosineWarmRestarts T_0={warm_restart_period}")
                print(f"  >>> Remaining epochs: {remaining_epochs}, levels to go: {levels_remaining}")
                print(f"  {'*'*60}\n")

                # Reset level tracking
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

    # ===== Final evaluation on each digit level separately =====
    print(f"\n{'='*70}")
    print(f"FINAL EVALUATION: {name}")
    print(f"{'='*70}")

    per_level_results = evaluate_all_digit_levels(model, device, max_digits=target_digits, n_samples=10000)

    print(f"\n  Per-digit-count accuracy:")
    print(f"  {'Digits':>8} | {'Exact Match':>12} | {'Digit Avg':>10}")
    print(f"  {'-'*8}-+-{'-'*12}-+-{'-'*10}")
    for d in range(1, target_digits + 1):
        r = per_level_results[d]
        print(f"  {d:>8} | {r['exact_match']:>12.4f} | {r['digit_accuracy_avg']:>10.4f}")

    # Overall eval on target digits
    final_results = evaluate(model, 50000, target_digits, device, seed=99999)
    print(f"\n  FINAL ({target_digits}d, 50k samples): exact={final_results['exact_match']:.4f} "
          f"digit_avg={final_results['digit_accuracy_avg']:.4f}")
    print(f"  best: {best_acc:.4f} at ep {best_epoch}")
    print(f"  total wall time: {total_wall_time:.0f}s ({total_wall_time/60:.1f}min)")

    # Curriculum history
    if curriculum_history:
        print(f"\n  Curriculum history:")
        for ep_h, digits_h, acc_h in curriculum_history:
            print(f"    ep {ep_h}: advanced from {digits_h}-digit (acc={acc_h:.4f})")

    # Level times
    if level_times:
        print(f"\n  Time per curriculum level:")
        for level, t in sorted(level_times.items()):
            print(f"    {level}-digit: {t:.0f}s ({t/60:.1f}min)")

    examples = get_example_predictions(model, target_digits, device, n_examples=8)
    print(f"  Final examples ({target_digits}-digit):\n{examples}")

    if wandb_run:
        import wandb
        # Log final per-level results
        for d, r in per_level_results.items():
            wandb.log({f"final_{d}d_exact": r["exact_match"], f"final_{d}d_digit_avg": r["digit_accuracy_avg"]})
        wandb.finish()

    return {
        "name": name, "n_params": n_params,
        "best_acc": best_acc, "best_epoch": best_epoch,
        "final_acc": final_results["exact_match"],
        "final_digit_acc": final_results["digit_accuracy_avg"],
        "final_per_digit": final_results["per_digit_accuracy"],
        "per_level_results": {str(d): {"exact_match": r["exact_match"], "digit_accuracy_avg": r["digit_accuracy_avg"]}
                              for d, r in per_level_results.items()},
        "curriculum_history": curriculum_history,
        "level_times": level_times,
        "total_wall_time": total_wall_time,
        "max_digits_reached": current_digits,
    }


def get_configs():
    """Return the focused configurations to test, ordered by priority."""
    configs = []

    # 1. d6_fr_15rep (324 params) - Best from sweep_v2
    configs.append(("d6_fr_15rep", TransformerConfig(
        d_model=6, n_heads=1, n_layers=1, d_ff=6,
        share_layers=True, n_layer_repeats=15,
        norm_type="rmsnorm", activation="relu",
        pe_period=11.0,
    )))

    # 2. d6_fr_25rep (~324 params) - More repeats
    configs.append(("d6_fr_25rep", TransformerConfig(
        d_model=6, n_heads=1, n_layers=1, d_ff=6,
        share_layers=True, n_layer_repeats=25,
        norm_type="rmsnorm", activation="relu",
        pe_period=11.0,
    )))

    # 3. d6_fr_15rep_gelu (324 params) - GELU activation
    configs.append(("d6_fr_15rep_gelu", TransformerConfig(
        d_model=6, n_heads=1, n_layers=1, d_ff=6,
        share_layers=True, n_layer_repeats=15,
        norm_type="rmsnorm", activation="gelu",
        pe_period=11.0,
    )))

    # 4. d4_fr_30rep (~156-180 params) - Tiny model
    configs.append(("d4_fr_30rep", TransformerConfig(
        d_model=4, n_heads=1, n_layers=1, d_ff=4,
        share_layers=True, n_layer_repeats=30,
        norm_type="none", use_bias=True, activation="relu",
        pe_period=11.0,
    )))

    # 5. d8_noffn_20rep (392 params) - No FFN, was slowly improving
    configs.append(("d8_noffn_20rep", TransformerConfig(
        d_model=8, n_heads=1, n_layers=1,
        ffn_type="none", share_layers=True, n_layer_repeats=20,
        norm_type="rmsnorm", activation="relu",
        pe_period=11.0,
    )))

    # 6. d6_fr_15rep_abacus (~324+ params) - Abacus PE
    # NOTE: AbacusPE mapping is hardcoded for 10-digit layout. During curriculum
    # training with fewer digits, the position mapping may not align correctly.
    # This config is experimental.
    configs.append(("d6_fr_15rep_abacus", TransformerConfig(
        d_model=6, n_heads=1, n_layers=1, d_ff=6,
        share_layers=True, n_layer_repeats=15,
        norm_type="rmsnorm", activation="relu",
        pe_type="abacus",
    )))

    return configs


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Sweep v3: LR warm restarts on curriculum advancement")
    parser.add_argument("--mode", type=str, default="all",
                        choices=["quick", "all"],
                        help="quick: only config #1; all: all configs")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--epochs", type=int, default=30000)
    parser.add_argument("--curriculum-threshold", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--target-digits", type=int, default=10)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Mode: {args.mode} | Epochs: {args.epochs} | LR: {args.lr}")
    print(f"Curriculum threshold: {args.curriculum_threshold} | Target digits: {args.target_digits}")

    all_configs = get_configs()
    if args.mode == "quick":
        configs_to_run = all_configs[:1]
        print(f"\nQuick mode: running only config #1 ({configs_to_run[0][0]})")
    else:
        configs_to_run = all_configs

    # Print config summary
    print(f"\n{'='*70}")
    print("CONFIGURATIONS")
    print(f"{'='*70}")
    for i, (cname, cfg) in enumerate(configs_to_run, 1):
        m = MinimalTransformer(cfg)
        n = count_parameters(m)
        print(f"  {i}. {cname}: {n} params | d_model={cfg.d_model} n_rep={cfg.n_layer_repeats} "
              f"ffn={cfg.ffn_type} norm={cfg.norm_type} act={cfg.activation} pe={cfg.pe_type}")
        del m

    results = []
    global_t0 = time.time()

    for i, (cname, cfg) in enumerate(configs_to_run, 1):
        print(f"\n\n{'#'*70}")
        print(f"# CONFIG {i}/{len(configs_to_run)}: {cname}")
        print(f"{'#'*70}")

        try:
            r = train_one(
                cfg, cname,
                epochs=args.epochs, lr=args.lr, wd=0.01, bs=args.batch_size,
                device=device, eval_every=args.eval_every, log_every=args.log_every,
                use_wandb=args.wandb,
                curriculum=True,
                curriculum_threshold=args.curriculum_threshold,
                target_digits=args.target_digits,
            )
            results.append(r)
        except Exception as e:
            print(f"ERROR in {cname}: {e}")
            import traceback; traceback.print_exc()
            results.append({"name": cname, "error": str(e)})

        # Save intermediate
        with open("experiments/sweep_v3_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

    # ===== FINAL SUMMARY =====
    total_time = time.time() - global_t0
    print(f"\n\n{'='*70}")
    print("SWEEP V3 RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"Total wall time: {total_time:.0f}s ({total_time/3600:.1f}h)")
    print()

    # Summary table
    header = f"{'Name':>25} | {'Params':>6} | {'Best Acc':>8} | {'Best Ep':>7} | {'Final Acc':>9} | {'Max Digits':>10} | {'Time':>8}"
    print(header)
    print("-" * len(header))
    for r in results:
        if "error" in r:
            print(f"{'  ' + r['name']:>25} | {'ERROR':>6} | {r['error'][:50]}")
        else:
            print(f"{r['name']:>25} | {r['n_params']:>6} | {r['best_acc']:>8.4f} | {r['best_epoch']:>7} | "
                  f"{r['final_acc']:>9.4f} | {r.get('max_digits_reached', '?'):>10} | "
                  f"{r['total_wall_time']:>7.0f}s")

    # Per-level breakdown for each config
    print(f"\n{'='*70}")
    print("PER-DIGIT-COUNT ACCURACY")
    print(f"{'='*70}")
    for r in results:
        if "error" in r:
            continue
        print(f"\n  {r['name']} ({r['n_params']}p):")
        plr = r.get("per_level_results", {})
        for d in range(1, 11):
            dr = plr.get(str(d), {})
            em = dr.get("exact_match", 0)
            da = dr.get("digit_accuracy_avg", 0)
            bar = "#" * int(em * 40)
            print(f"    {d:>2}d: exact={em:.4f} digit_avg={da:.4f} |{bar}")

    # Save final
    with open("experiments/sweep_v3_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to experiments/sweep_v3_results.json")


if __name__ == "__main__":
    main()
