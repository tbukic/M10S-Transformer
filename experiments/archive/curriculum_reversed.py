"""Fixed-layout curriculum with ALL-REVERSED (LSB-first) format.

Combines the best ideas:
1. Fixed 33-token layout (always 10-digit format)
2. All-reversed (LSB-first) for proper PE alignment
3. Curriculum: gradually increase significant digits
4. 2 attention heads for dual operand routing

The key insight: with reversed format and period-11 PE,
same-column digits (A_i, B_i, C_i) all share the same PE value.
This gives the model a strong structural prior for column-wise attention.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from minimal10digittransformer.model.transformer import (
    MinimalTransformer,
    TransformerConfig,
    count_parameters,
)

MAX_DIGITS = 10
INPUT_LEN = MAX_DIGITS * 2 + 2  # 22
OUTPUT_LEN = MAX_DIGITS + 1      # 11
SEQ_LEN = INPUT_LEN + OUTPUT_LEN  # 33


def generate_batch(batch_size, sig_digits, device):
    """Generate batch in ALL-REVERSED format with controlled magnitude.

    sig_digits controls number size: a, b in [0, 10^sig_digits).
    Always padded to 10-digit format (33 tokens).
    """
    max_val = 10**sig_digits
    a = torch.randint(0, max_val, (batch_size,), device=device, dtype=torch.long)
    b = torch.randint(0, max_val, (batch_size,), device=device, dtype=torch.long)
    c = a + b

    a_digits, b_digits, c_digits = [], [], []
    a_tmp, b_tmp, c_tmp = a.clone(), b.clone(), c.clone()
    for _ in range(MAX_DIGITS):
        a_digits.append(a_tmp % 10)
        b_digits.append(b_tmp % 10)
        c_digits.append(c_tmp % 10)
        a_tmp //= 10; b_tmp //= 10; c_tmp //= 10
    c_digits.append(c_tmp % 10)

    # ALL LSB-first
    a_t = torch.stack(a_digits, dim=1)
    b_t = torch.stack(b_digits, dim=1)
    c_t = torch.stack(c_digits, dim=1)

    plus = torch.full((batch_size, 1), 10, device=device, dtype=torch.long)
    eq = torch.full((batch_size, 1), 11, device=device, dtype=torch.long)

    input_ids = torch.cat([a_t, plus, b_t, eq, c_t], dim=1)
    labels = torch.full_like(input_ids, -100)
    labels[:, INPUT_LEN - 1:-1] = c_t

    return input_ids, labels


def evaluate(model, n_samples, sig_digits, device, seed=42):
    """Evaluate on specified sig_digits level."""
    model.eval()
    correct = 0
    total = 0
    digit_correct = torch.zeros(OUTPUT_LEN, device=device)
    digit_total = torch.zeros(OUTPUT_LEN, device=device)

    torch.manual_seed(seed)
    with torch.no_grad():
        for _ in range(0, n_samples, 2048):
            bs = min(2048, n_samples - total)
            if bs <= 0:
                break
            input_ids, _ = generate_batch(bs, sig_digits, device)
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
    return {"exact_match": exact_acc, "per_digit": per_digit,
            "digit_avg": sum(per_digit) / len(per_digit)}


def show_examples(model, device, sig_digits, n=5):
    """Show examples."""
    model.eval()
    with torch.no_grad():
        input_ids, _ = generate_batch(n, sig_digits, device)
        logits = model(input_ids)
        preds = logits[:, INPUT_LEN - 1:-1, :].argmax(dim=-1)
        targets = input_ids[:, INPUT_LEN:]

    for i in range(n):
        seq = input_ids[i].cpu().tolist()
        a_str = "".join(str(d) for d in reversed(seq[:MAX_DIGITS]))
        b_str = "".join(str(d) for d in reversed(seq[MAX_DIGITS+1:MAX_DIGITS*2+1]))
        pred_str = "".join(str(d) for d in reversed(preds[i].cpu().tolist()))
        tgt_str = "".join(str(d) for d in reversed(targets[i].cpu().tolist()))
        ok = "OK" if preds[i].cpu().tolist() == targets[i].cpu().tolist() else "WRONG"
        print(f"  {a_str} + {b_str} = {pred_str} (expected {tgt_str}) [{ok}]")


def train(d_model, n_heads, n_repeats, epochs, lr, seed=0, use_wandb=False,
          advance_threshold=0.80, mixed_ratio=0.3):
    """Train with reversed curriculum.

    mixed_ratio: fraction of each batch from previous difficulty levels
    (mixed training helps prevent catastrophic forgetting on advancement).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    config = TransformerConfig(
        d_model=d_model, n_heads=n_heads, n_layers=1, d_ff=d_model,
        share_layers=True, n_layer_repeats=n_repeats,
        norm_type="rmsnorm", activation="relu",
        pe_period=11.0,
    )
    model = MinimalTransformer(config).to(device)
    n_params = count_parameters(model)

    tag = f"currev_d{d_model}_h{n_heads}_r{n_repeats}"
    print(f"\n{'='*70}")
    print(f"CURRICULUM REVERSED: d={d_model}, heads={n_heads}, repeats={n_repeats}, params={n_params}")
    print(f"lr={lr}, epochs={epochs}, seed={seed}, advance_threshold={advance_threshold}")
    print(f"{'='*70}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=3000, T_mult=1, eta_min=1e-5
    )

    wandb_run = None
    if use_wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project="minimal-10digit-transformer",
                name=f"{tag}_{n_params}p_s{seed}",
                config={"d_model": d_model, "n_heads": n_heads, "n_repeats": n_repeats,
                        "n_params": n_params, "lr": lr, "epochs": epochs, "seed": seed,
                        "curriculum": True, "reversed": True, "mixed_ratio": mixed_ratio},
                tags=["curriculum_reversed"],
                reinit=True,
            )
        except Exception:
            pass

    sig = 2  # Start with 2 significant digits
    best_acc = 0.0
    best_sig10_acc = 0.0
    t0 = time.time()

    for ep in range(epochs):
        model.train()
        total_loss = 0
        n_batches = 20
        batch_size = 1024

        for _ in range(n_batches):
            # Mixed batch: most from current level, some from easier levels
            if sig > 2 and mixed_ratio > 0:
                n_mixed = int(batch_size * mixed_ratio)
                n_current = batch_size - n_mixed
                # Mixed portion: uniform over [2, sig-1]
                mix_sig = torch.randint(2, sig, (1,)).item()
                ids_mix, labels_mix = generate_batch(n_mixed, mix_sig, device)
                ids_cur, labels_cur = generate_batch(n_current, sig, device)
                input_ids = torch.cat([ids_cur, ids_mix], dim=0)
                labels = torch.cat([labels_cur, labels_mix], dim=0)
            else:
                input_ids, labels = generate_batch(batch_size, sig, device)

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

        if ep % 200 == 0:
            elapsed = time.time() - t0
            print(f"  ep {ep:6d} | sig={sig:2d} | loss {avg_loss:.4f} | lr {opt.param_groups[0]['lr']:.2e} | {elapsed:.0f}s")
            if wandb_run:
                import wandb
                wandb.log({"loss": avg_loss, "sig_digits": sig, "lr": opt.param_groups[0]["lr"]}, step=ep)

        if ep % 500 == 0 and ep > 0:
            results = evaluate(model, 10000, sig, device)
            acc = results["exact_match"]
            print(f"  EVAL sig={sig} ep {ep}: exact={acc:.4f} digit_avg={results['digit_avg']:.4f}")
            print(f"    per-digit: {['%.3f' % d for d in results['per_digit']]}")
            show_examples(model, device, sig, n=3)

            # Also eval on full 10-digit
            if sig < 10:
                r10 = evaluate(model, 5000, 10, device)
                print(f"  EVAL sig=10: exact={r10['exact_match']:.4f} digit_avg={r10['digit_avg']:.4f}")
                if r10["exact_match"] > best_sig10_acc:
                    best_sig10_acc = r10["exact_match"]
                if wandb_run:
                    import wandb
                    wandb.log({"eval_sig10_acc": r10["exact_match"], "eval_sig10_digit_avg": r10["digit_avg"]}, step=ep)

            # Advance if above threshold
            if acc >= advance_threshold and sig < 10:
                sig += 1
                print(f"  >> ADVANCED to sig={sig} (was {acc:.4f} on sig={sig-1})")
                # Reset scheduler for fresh cosine cycle
                sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    opt, T_0=3000, T_mult=1, eta_min=1e-5
                )

            if acc > best_acc:
                best_acc = acc

            if wandb_run:
                import wandb
                log_dict = {"eval_acc": acc, "best_acc": best_acc, "sig_digits": sig,
                           "digit_avg": results["digit_avg"]}
                for i, d in enumerate(results["per_digit"]):
                    log_dict[f"digit_{i}_acc"] = d
                wandb.log(log_dict, step=ep)

            # Save checkpoint
            ckpt_dir = Path(f"checkpoints/{tag}_s{seed}")
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                "state_dict": model.state_dict(), "config": config,
                "sig": sig, "accuracy": acc, "n_params": n_params,
            }, str(ckpt_dir / "best.pt"))

    # Final eval on full 10-digit
    final = evaluate(model, 50000, 10, device, seed=99999)
    print(f"\n  FINAL (sig=10): exact={final['exact_match']:.4f} digit_avg={final['digit_avg']:.4f}")
    print(f"    per-digit: {['%.3f' % d for d in final['per_digit']]}")
    show_examples(model, device, 10, n=10)

    if wandb_run:
        import wandb
        wandb.finish()

    return {
        "d_model": d_model, "n_heads": n_heads, "n_repeats": n_repeats,
        "n_params": n_params, "seed": seed, "best_sig_acc": best_acc,
        "final_sig": sig, "final_sig10_acc": final["exact_match"],
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--d-model", type=int, default=8)
    parser.add_argument("--n-heads", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--epochs", type=int, default=30000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.80)
    parser.add_argument("--mixed-ratio", type=float, default=0.3)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    result = train(args.d_model, args.n_heads, args.repeats, args.epochs, args.lr,
                   args.seed, args.wandb, args.threshold, args.mixed_ratio)

    out_file = f"experiments/curriculum_reversed_d{args.d_model}_h{args.n_heads}_results.json"
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {out_file}")


if __name__ == "__main__":
    main()
