"""Direct 10-digit training with ALL-REVERSED (LSB-first) format.

Critical fix: Previous experiments used MSB-first inputs + LSB-first outputs.
This breaks the period-11 PE alignment. With all LSB-first:
  PE(i) = PE(i+11) = PE(i+22) → same-column digits aligned perfectly.

Format: A0 A1...A9 + B0 B1...B9 = C0 C1...C10
Where digit 0 = ones, digit 9 = billions, etc.
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


def generate_batch(batch_size, device):
    """Generate full 10-digit addition batch — ALL LSB-first.

    With period-11 PE, position i, i+11, i+22 all share the same PE value.
    LSB-first means:
      pos 0 = A_ones, pos 11 = B_ones, pos 22 = C_ones  (all ones column)
      pos 1 = A_tens, pos 12 = B_tens, pos 23 = C_tens  (all tens column)
      etc.
    """
    a = torch.randint(0, 10**MAX_DIGITS, (batch_size,), device=device, dtype=torch.long)
    b = torch.randint(0, 10**MAX_DIGITS, (batch_size,), device=device, dtype=torch.long)
    c = a + b

    a_digits, b_digits, c_digits = [], [], []
    a_tmp, b_tmp, c_tmp = a.clone(), b.clone(), c.clone()
    for _ in range(MAX_DIGITS):
        a_digits.append(a_tmp % 10)
        b_digits.append(b_tmp % 10)
        c_digits.append(c_tmp % 10)
        a_tmp //= 10; b_tmp //= 10; c_tmp //= 10
    c_digits.append(c_tmp % 10)

    # KEY CHANGE: NO .flip(1) — inputs are LSB-first, matching output order
    a_t = torch.stack(a_digits, dim=1)  # LSB first (ones at position 0)
    b_t = torch.stack(b_digits, dim=1)  # LSB first
    c_t = torch.stack(c_digits, dim=1)  # LSB first (already was)

    plus = torch.full((batch_size, 1), 10, device=device, dtype=torch.long)
    eq = torch.full((batch_size, 1), 11, device=device, dtype=torch.long)

    input_ids = torch.cat([a_t, plus, b_t, eq, c_t], dim=1)

    labels = torch.full_like(input_ids, -100)
    labels[:, INPUT_LEN - 1:-1] = c_t

    return input_ids, labels


def evaluate(model, n_samples, device, seed=42):
    """Evaluate exact-match and per-digit accuracy."""
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
            input_ids, _ = generate_batch(bs, device)
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
        "per_digit": per_digit,
        "digit_avg": sum(per_digit) / len(per_digit),
    }


def show_examples(model, device, n=5):
    """Show example predictions (display as normal numbers, MSB-first)."""
    model.eval()
    with torch.no_grad():
        input_ids, _ = generate_batch(n, device)
        logits = model(input_ids)
        preds = logits[:, INPUT_LEN - 1:-1, :].argmax(dim=-1)
        targets = input_ids[:, INPUT_LEN:]

    for i in range(n):
        seq = input_ids[i].cpu().tolist()
        # Input is LSB-first, reverse for display
        a_str = "".join(str(d) for d in reversed(seq[:MAX_DIGITS]))
        b_str = "".join(str(d) for d in reversed(seq[MAX_DIGITS+1:MAX_DIGITS*2+1]))
        pred_str = "".join(str(d) for d in reversed(preds[i].cpu().tolist()))
        tgt_str = "".join(str(d) for d in reversed(targets[i].cpu().tolist()))
        ok = "OK" if preds[i].cpu().tolist() == targets[i].cpu().tolist() else "WRONG"
        print(f"  {a_str} + {b_str} = {pred_str} (expected {tgt_str}) [{ok}]")


def train(d_model, n_repeats, epochs, lr, seed=0, use_wandb=False, n_heads=1):
    """Train a single configuration."""
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

    tag = f"rev_d{d_model}_r{n_repeats}_h{n_heads}"
    print(f"\n{'='*70}")
    print(f"ALL-REVERSED 10-digit: d_model={d_model}, repeats={n_repeats}, heads={n_heads}, params={n_params}")
    print(f"lr={lr}, epochs={epochs}, seed={seed}")
    print(f"{'='*70}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=5000, T_mult=1, eta_min=1e-5
    )

    wandb_run = None
    if use_wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project="minimal-10digit-transformer",
                name=f"{tag}_{n_params}p_s{seed}",
                config={"d_model": d_model, "n_repeats": n_repeats, "n_params": n_params,
                        "n_heads": n_heads, "lr": lr, "epochs": epochs, "seed": seed,
                        "all_reversed": True},
                tags=["all_reversed", "direct_10digit"],
                reinit=True,
            )
        except Exception:
            pass

    best_acc = 0.0
    best_epoch = 0
    t0 = time.time()

    for ep in range(epochs):
        model.train()
        total_loss = 0
        n_batches = 20

        for _ in range(n_batches):
            input_ids, labels = generate_batch(1024, device)
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
            print(f"  ep {ep:6d} | loss {avg_loss:.4f} | lr {opt.param_groups[0]['lr']:.2e} | {elapsed:.0f}s")
            if wandb_run:
                import wandb
                wandb.log({"loss": avg_loss, "lr": opt.param_groups[0]["lr"]}, step=ep)

        if ep % 1000 == 0 and ep > 0:
            results = evaluate(model, 10000, device)
            acc = results["exact_match"]
            print(f"  EVAL ep {ep}: exact={acc:.4f} digit_avg={results['digit_avg']:.4f}")
            print(f"    per-digit: {['%.3f' % d for d in results['per_digit']]}")
            show_examples(model, device)

            if acc > best_acc:
                best_acc = acc
                best_epoch = ep
                print(f"  ** BEST: {acc:.4f} **")
                ckpt_dir = Path(f"checkpoints/{tag}_s{seed}")
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "state_dict": model.state_dict(), "config": config,
                    "accuracy": acc, "n_params": n_params,
                }, str(ckpt_dir / "best.pt"))

            if wandb_run:
                import wandb
                log_dict = {"eval_acc": acc, "best_acc": best_acc, "digit_avg": results["digit_avg"]}
                for i, d in enumerate(results["per_digit"]):
                    log_dict[f"digit_{i}_acc"] = d
                wandb.log(log_dict, step=ep)

        # Early stopping if perfect
        if best_acc >= 0.99:
            print(f"  Reached 99%+ at epoch {ep}, stopping early.")
            break

    # Final eval
    final = evaluate(model, 50000, device, seed=99999)
    print(f"\n  FINAL: exact={final['exact_match']:.4f} digit_avg={final['digit_avg']:.4f}")
    print(f"    per-digit: {['%.3f' % d for d in final['per_digit']]}")
    show_examples(model, device, n=10)

    if wandb_run:
        import wandb
        wandb.finish()

    return {
        "d_model": d_model, "n_repeats": n_repeats, "n_heads": n_heads,
        "n_params": n_params, "seed": seed, "best_acc": best_acc,
        "best_epoch": best_epoch, "final_acc": final["exact_match"],
        "final_per_digit": final["per_digit"],
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--epochs", type=int, default=30000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-heads", type=int, default=1)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    result = train(args.d_model, args.repeats, args.epochs, args.lr, args.seed,
                   args.wandb, args.n_heads)

    out_file = f"experiments/direct_reversed_d{args.d_model}_results.json"
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved results to {out_file}")


if __name__ == "__main__":
    main()
