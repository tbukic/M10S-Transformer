"""Knowledge distillation from large teacher to small student model.

The d64 teacher (25728p, 99.86% accuracy) provides soft targets
for training smaller student models on 10-digit addition.

Both teacher and student use ALL-REVERSED (LSB-first) format for
proper period-11 PE alignment.
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

from minimal10digittransformer.model.transformer import (
    MinimalTransformer,
    TransformerConfig,
    count_parameters,
)

MAX_DIGITS = 10
INPUT_LEN = MAX_DIGITS * 2 + 2  # 22
OUTPUT_LEN = MAX_DIGITS + 1      # 11
SEQ_LEN = INPUT_LEN + OUTPUT_LEN  # 33


def generate_batch_reversed(batch_size, device):
    """Generate batch in ALL-REVERSED (LSB-first) format."""
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

    a_t = torch.stack(a_digits, dim=1)   # LSB first
    b_t = torch.stack(b_digits, dim=1)   # LSB first
    c_t = torch.stack(c_digits, dim=1)   # LSB first

    plus = torch.full((batch_size, 1), 10, device=device, dtype=torch.long)
    eq = torch.full((batch_size, 1), 11, device=device, dtype=torch.long)

    input_ids = torch.cat([a_t, plus, b_t, eq, c_t], dim=1)
    labels = torch.full_like(input_ids, -100)
    labels[:, INPUT_LEN - 1:-1] = c_t

    return input_ids, labels


def generate_batch_msb(batch_size, device):
    """Generate batch in MSB-first format (for the teacher)."""
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

    a_t = torch.stack(a_digits, dim=1).flip(1)  # MSB first
    b_t = torch.stack(b_digits, dim=1).flip(1)   # MSB first
    c_t = torch.stack(c_digits, dim=1)            # LSB first

    plus = torch.full((batch_size, 1), 10, device=device, dtype=torch.long)
    eq = torch.full((batch_size, 1), 11, device=device, dtype=torch.long)

    input_ids = torch.cat([a_t, plus, b_t, eq, c_t], dim=1)
    return input_ids


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
            input_ids, _ = generate_batch_reversed(bs, device)
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


def show_examples(model, device, n=5):
    """Show example predictions."""
    model.eval()
    with torch.no_grad():
        input_ids, _ = generate_batch_reversed(n, device)
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


def train_distill(teacher_path, d_model, n_repeats, n_heads, epochs, lr,
                  temperature=4.0, alpha=0.5, seed=0, use_wandb=False):
    """Train student with knowledge distillation from teacher.

    Loss = alpha * KL(student || teacher) + (1-alpha) * CE(student, labels)

    The teacher uses MSB-first format, student uses reversed (LSB-first).
    We generate the same numbers for both but in different formats.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Load teacher
    ckpt = torch.load(teacher_path, map_location=device, weights_only=False)
    teacher_config = ckpt["config"]
    teacher = MinimalTransformer(teacher_config).to(device)
    teacher.load_state_dict(ckpt["state_dict"])
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Loaded teacher: {count_parameters(teacher)} params, acc={ckpt.get('accuracy', 'unknown')}")

    # Create student (reversed format)
    student_config = TransformerConfig(
        d_model=d_model, n_heads=n_heads, n_layers=1, d_ff=d_model,
        share_layers=True, n_layer_repeats=n_repeats,
        norm_type="rmsnorm", activation="relu",
        pe_period=11.0,
    )
    student = MinimalTransformer(student_config).to(device)
    n_params = count_parameters(student)

    tag = f"distill_d{d_model}_h{n_heads}_r{n_repeats}"
    print(f"\n{'='*70}")
    print(f"Distillation: student d_model={d_model}, heads={n_heads}, repeats={n_repeats}, params={n_params}")
    print(f"teacher={teacher_path}, T={temperature}, alpha={alpha}")
    print(f"lr={lr}, epochs={epochs}, seed={seed}")
    print(f"{'='*70}")

    opt = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=0.01)
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
                config={"d_model": d_model, "n_heads": n_heads, "n_repeats": n_repeats,
                        "n_params": n_params, "lr": lr, "epochs": epochs, "seed": seed,
                        "temperature": temperature, "alpha": alpha,
                        "teacher_path": str(teacher_path), "distillation": True},
                tags=["distillation", "all_reversed"],
                reinit=True,
            )
        except Exception:
            pass

    best_acc = 0.0
    best_epoch = 0
    t0 = time.time()

    for ep in range(epochs):
        student.train()
        total_loss = 0
        total_kl = 0
        total_ce = 0
        n_batches = 20

        for _ in range(n_batches):
            # Generate same numbers, different formats
            batch_size = 1024
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

            # Student: LSB-first
            a_rev = torch.stack(a_digits, dim=1)
            b_rev = torch.stack(b_digits, dim=1)
            c_t = torch.stack(c_digits, dim=1)
            plus = torch.full((batch_size, 1), 10, device=device, dtype=torch.long)
            eq = torch.full((batch_size, 1), 11, device=device, dtype=torch.long)
            student_ids = torch.cat([a_rev, plus, b_rev, eq, c_t], dim=1)

            # Teacher: MSB-first
            a_msb = torch.stack(a_digits, dim=1).flip(1)
            b_msb = torch.stack(b_digits, dim=1).flip(1)
            teacher_ids = torch.cat([a_msb, plus, b_msb, eq, c_t], dim=1)

            # Forward pass
            with torch.no_grad():
                teacher_logits = teacher(teacher_ids)
            student_logits = student(student_ids)

            # Extract output positions only (positions 21..31 predict positions 22..32)
            # Both teacher and student output C in LSB-first order at the same positions
            t_out = teacher_logits[:, INPUT_LEN - 1:-1, :]  # (B, 11, V)
            s_out = student_logits[:, INPUT_LEN - 1:-1, :]  # (B, 11, V)

            # KL divergence loss (soft targets)
            kl_loss = F.kl_div(
                F.log_softmax(s_out / temperature, dim=-1),
                F.softmax(t_out / temperature, dim=-1),
                reduction="batchmean"
            ) * (temperature ** 2)

            # Hard target CE loss
            labels = torch.full_like(student_ids, -100)
            labels[:, INPUT_LEN - 1:-1] = c_t
            ce_loss = F.cross_entropy(
                student_logits.view(-1, student_logits.size(-1)),
                labels.view(-1), ignore_index=-100
            )

            loss = alpha * kl_loss + (1 - alpha) * ce_loss

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
            total_kl += kl_loss.item()
            total_ce += ce_loss.item()

        sched.step()
        avg_loss = total_loss / n_batches

        if ep % 200 == 0:
            elapsed = time.time() - t0
            avg_kl = total_kl / n_batches
            avg_ce = total_ce / n_batches
            print(f"  ep {ep:6d} | loss {avg_loss:.4f} | kl {avg_kl:.4f} | ce {avg_ce:.4f} | lr {opt.param_groups[0]['lr']:.2e} | {elapsed:.0f}s")
            if wandb_run:
                import wandb
                wandb.log({"loss": avg_loss, "kl_loss": avg_kl, "ce_loss": avg_ce,
                          "lr": opt.param_groups[0]["lr"]}, step=ep)

        if ep % 1000 == 0 and ep > 0:
            results = evaluate(student, 10000, device)
            acc = results["exact_match"]
            print(f"  EVAL ep {ep}: exact={acc:.4f} digit_avg={results['digit_avg']:.4f}")
            print(f"    per-digit: {['%.3f' % d for d in results['per_digit']]}")
            show_examples(student, device)

            if acc > best_acc:
                best_acc = acc
                best_epoch = ep
                print(f"  ** BEST: {acc:.4f} **")
                ckpt_dir = Path(f"checkpoints/{tag}_s{seed}")
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "state_dict": student.state_dict(), "config": student_config,
                    "accuracy": acc, "n_params": n_params,
                }, str(ckpt_dir / "best.pt"))

            if wandb_run:
                import wandb
                log_dict = {"eval_acc": acc, "best_acc": best_acc, "digit_avg": results["digit_avg"]}
                for i, d in enumerate(results["per_digit"]):
                    log_dict[f"digit_{i}_acc"] = d
                wandb.log(log_dict, step=ep)

        if best_acc >= 0.99:
            print(f"  Reached 99%+ at epoch {ep}, stopping early.")
            break

    final = evaluate(student, 50000, device, seed=99999)
    print(f"\n  FINAL: exact={final['exact_match']:.4f} digit_avg={final['digit_avg']:.4f}")
    print(f"    per-digit: {['%.3f' % d for d in final['per_digit']]}")
    show_examples(student, device, n=10)

    if wandb_run:
        import wandb
        wandb.finish()

    return {
        "d_model": d_model, "n_heads": n_heads, "n_repeats": n_repeats,
        "n_params": n_params, "seed": seed, "temperature": temperature,
        "alpha": alpha, "best_acc": best_acc, "best_epoch": best_epoch,
        "final_acc": final["exact_match"], "final_per_digit": final["per_digit"],
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", type=str, default="checkpoints/direct_d64_r10_s0/best.pt")
    parser.add_argument("--d-model", type=int, default=8)
    parser.add_argument("--n-heads", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--epochs", type=int, default=50000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    result = train_distill(
        args.teacher, args.d_model, args.repeats, args.n_heads,
        args.epochs, args.lr, args.temperature, args.alpha, args.seed, args.wandb
    )

    out_file = f"experiments/distill_d{args.d_model}_h{args.n_heads}_results.json"
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved results to {out_file}")


if __name__ == "__main__":
    main()
