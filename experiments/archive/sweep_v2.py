"""Corrected sweep v2 with curriculum learning and pipeline validation.

Fixes applied:
1. Eval off-by-one: targets = input_ids[:, input_len:] (not labels)
2. PE period: SinusoidalPE now uses config.pe_period (default 11.0)

New features:
- Curriculum learning: start with max_digits=2, increase by 1 when accuracy > 50%
- Pipeline validation: first trains a larger model on 3-digit addition
- Per-digit accuracy logging
- Example prediction logging
- wandb support via --wandb flag
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
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

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
            # FIX: use input_ids for targets, not labels (which are shifted by 1)
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
        # Decode input (A + B = )
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


def train_one(config, name, epochs=10000, lr=3e-3, wd=0.01, bs=1024,
              device="cuda", eval_every=500, log_every=100, use_wandb=False,
              curriculum=False, curriculum_threshold=0.5, target_digits=10):
    """Train a single configuration with optional curriculum learning."""
    model = MinimalTransformer(config).to(device)
    n_params = count_parameters(model)
    print(f"\n{'='*60}")
    print(f"{name}: {n_params} params | curriculum={curriculum} | target_digits={target_digits}")
    print(f"{'='*60}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)

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
                },
                tags=["sweep_v2"],
                reinit=True,
            )
        except Exception:
            pass

    best_acc = 0.0
    best_epoch = 0
    current_digits = 2 if curriculum else target_digits
    t0 = time.time()

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
            print(f"  ep {ep:6d} | loss {avg_loss:.4f} | lr {opt.param_groups[0]['lr']:.2e} | digits={current_digits} | {time.time()-t0:.0f}s")
            if wandb_run:
                import wandb
                wandb.log({"loss": avg_loss, "lr": opt.param_groups[0]["lr"], "current_digits": current_digits}, step=ep)

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
                print(f"    target({target_digits}d): exact={target_results['exact_match']:.4f} digit_avg={target_results['digit_accuracy_avg']:.4f}")

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
                    "accuracy": acc, "n_params": n_params, "current_digits": current_digits,
                }, f"checkpoints/{name}/best.pt")

            # Curriculum: advance if accuracy > threshold
            if curriculum and acc > curriculum_threshold and current_digits < target_digits:
                current_digits += 1
                print(f"  >>> CURRICULUM ADVANCE: now training on {current_digits}-digit addition <<<")

            if wandb_run:
                import wandb
                log_dict = {
                    "eval_acc": acc, "best_acc": best_acc,
                    "digit_accuracy_avg": digit_acc,
                }
                for i, d in enumerate(results["per_digit_accuracy"]):
                    log_dict[f"digit_{i}_acc"] = d
                wandb.log(log_dict, step=ep)

    # Final eval
    final_results = evaluate(model, 50000, target_digits, device, seed=99999)
    print(f"\n  FINAL ({target_digits}d): exact={final_results['exact_match']:.4f} digit_avg={final_results['digit_accuracy_avg']:.4f}")
    print(f"  best: {best_acc:.4f} at ep {best_epoch}")
    examples = get_example_predictions(model, target_digits, device, n_examples=8)
    print(f"  Final examples:\n{examples}")

    if wandb_run:
        import wandb
        wandb.finish()

    return {
        "name": name, "n_params": n_params,
        "best_acc": best_acc, "best_epoch": best_epoch,
        "final_acc": final_results["exact_match"],
        "final_digit_acc": final_results["digit_accuracy_avg"],
        "final_per_digit": final_results["per_digit_accuracy"],
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Corrected sweep v2 with curriculum learning")
    parser.add_argument("--mode", type=str, default="all",
                        choices=["validate", "medium", "all"],
                        help="validate: pipeline check; medium: medium configs; all: both")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--curriculum-threshold", type=float, default=0.5)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    results = []

    # ===== PHASE 1: Pipeline Validation =====
    if args.mode in ("validate", "all"):
        print("\n" + "=" * 70)
        print("PHASE 1: Pipeline Validation (d_model=32 on 3-digit addition)")
        print("=" * 70)

        validate_config = TransformerConfig(
            d_model=32, n_heads=1, n_layers=1, d_ff=32,
            share_layers=True, n_layer_repeats=10,
            norm_type="rmsnorm", activation="relu",
            pe_period=11.0,
        )
        validate_model = MinimalTransformer(validate_config)
        print(f"Validation model params: {count_parameters(validate_model)}")

        r = train_one(
            validate_config, "validate_d32_3digit",
            epochs=min(args.epochs, 2000), lr=3e-3, wd=0.01, bs=1024,
            device=device, eval_every=200, log_every=50,
            use_wandb=args.wandb,
            curriculum=False, target_digits=3,
        )
        results.append(r)

        if r["best_acc"] < 0.01:
            print("\n*** WARNING: Pipeline validation failed (<1% accuracy on 3-digit). Check for bugs. ***")
        elif r["best_acc"] > 0.5:
            print(f"\n*** Pipeline validation PASSED: {r['best_acc']:.4f} accuracy on 3-digit ***")
        else:
            print(f"\n*** Pipeline validation partial: {r['best_acc']:.4f} accuracy on 3-digit ***")

    # ===== PHASE 2: Medium configs with curriculum =====
    if args.mode in ("medium", "all"):
        print("\n" + "=" * 70)
        print("PHASE 2: Medium configs with curriculum learning")
        print("=" * 70)

        configs = {}

        configs["med_d8r1_20rep_curriculum"] = TransformerConfig(
            d_model=8, n_heads=1, n_layers=1, d_ff=4,
            share_layers=True, n_layer_repeats=20,
            rank=1, norm_type="none", activation="relu",
            pe_period=11.0,
        )
        configs["med_d8r2_20rep_curriculum"] = TransformerConfig(
            d_model=8, n_heads=1, n_layers=1, d_ff=4,
            share_layers=True, n_layer_repeats=20,
            rank=2, norm_type="rmsnorm", activation="relu",
            pe_period=11.0,
        )
        configs["med_d8_noffn_20rep_curriculum"] = TransformerConfig(
            d_model=8, n_heads=1, n_layers=1,
            ffn_type="none", share_layers=True, n_layer_repeats=20,
            norm_type="rmsnorm", activation="relu",
            pe_period=11.0,
        )
        configs["med_d6_15rep_curriculum"] = TransformerConfig(
            d_model=6, n_heads=1, n_layers=1, d_ff=6,
            share_layers=True, n_layer_repeats=15,
            norm_type="rmsnorm", activation="relu",
            pe_period=11.0,
        )

        for name, cfg in sorted(configs.items(), key=lambda x: count_parameters(MinimalTransformer(x[1]))):
            n = count_parameters(MinimalTransformer(cfg))
            print(f"  {name}: {n} params")

        for name, cfg in sorted(configs.items(), key=lambda x: count_parameters(MinimalTransformer(x[1]))):
            try:
                r = train_one(
                    cfg, name,
                    epochs=args.epochs, lr=3e-3, wd=0.01, bs=1024,
                    device=device, eval_every=500, log_every=100,
                    use_wandb=args.wandb,
                    curriculum=True,
                    curriculum_threshold=args.curriculum_threshold,
                    target_digits=10,
                )
                results.append(r)
            except Exception as e:
                print(f"ERROR in {name}: {e}")
                import traceback; traceback.print_exc()
                results.append({"name": name, "error": str(e)})

            # Save intermediate
            with open("experiments/sweep_v2_results.json", "w") as f:
                json.dump(results, f, indent=2)

    # ===== Summary =====
    print("\n" + "=" * 70)
    print("SWEEP V2 RESULTS")
    print("=" * 70)
    for r in results:
        if "error" in r:
            print(f"  {r['name']}: ERROR - {r['error']}")
        else:
            print(f"  {r['name']}: {r['n_params']}p | best={r['best_acc']:.4f} | final={r['final_acc']:.4f} | digit_avg={r.get('final_digit_acc', 0):.4f}")

    # Save final
    with open("experiments/sweep_v2_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
