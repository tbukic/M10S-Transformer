"""Flexible training script for various architecture configurations.

Supports:
- vocab=10 (map + and = to token 0) or vocab=15
- Shared or non-shared layers
- Variable n_layers, n_repeats, d_ff, n_heads
- All-reversed (LSB-first) format
- Various LR, warmup, schedulers
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# Ensure unbuffered output for log redirection
os.environ.setdefault("PYTHONUNBUFFERED", "1")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

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


def generate_batch(batch_size, device, vocab10=False):
    """Generate full 10-digit addition batch — ALL LSB-first."""
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

    a_t = torch.stack(a_digits, dim=1)
    b_t = torch.stack(b_digits, dim=1)
    c_t = torch.stack(c_digits, dim=1)

    if vocab10:
        # Map + and = to token 0 (model distinguishes via PE)
        plus = torch.zeros((batch_size, 1), device=device, dtype=torch.long)
        eq = torch.zeros((batch_size, 1), device=device, dtype=torch.long)
    else:
        plus = torch.full((batch_size, 1), 10, device=device, dtype=torch.long)
        eq = torch.full((batch_size, 1), 11, device=device, dtype=torch.long)

    input_ids = torch.cat([a_t, plus, b_t, eq, c_t], dim=1)
    labels = torch.full_like(input_ids, -100)
    labels[:, INPUT_LEN - 1:-1] = c_t
    return input_ids, labels


def evaluate(model, n_samples, device, seed=42, vocab10=False):
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
            input_ids, _ = generate_batch(bs, device, vocab10=vocab10)
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
    return {"exact_match": exact_acc, "per_digit": per_digit, "digit_avg": sum(per_digit) / len(per_digit)}


def show_examples(model, device, n=5, vocab10=False):
    model.eval()
    with torch.no_grad():
        input_ids, _ = generate_batch(n, device, vocab10=vocab10)
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


def train(args):
    """Train a single configuration."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    vocab_size = 10 if args.vocab10 else 15
    d_ff = args.d_ff if args.d_ff > 0 else args.d_model
    rank = args.rank if args.rank > 0 else None

    config = TransformerConfig(
        vocab_size=vocab_size,
        d_model=args.d_model, n_heads=args.n_heads,
        n_layers=args.n_layers, d_ff=d_ff,
        share_layers=args.share_layers,
        n_layer_repeats=args.repeats if args.share_layers else 1,
        norm_type="rmsnorm", activation=args.activation,
        pe_period=11.0,
        use_bias=args.use_bias,
        rank=rank,
    )
    model = MinimalTransformer(config).to(device)
    n_params = count_parameters(model)

    # Build descriptive tag
    share_str = f"sh{args.repeats}" if args.share_layers else f"L{args.n_layers}"
    tag = f"flex_d{args.d_model}_h{args.n_heads}_{share_str}_v{vocab_size}"
    if d_ff != args.d_model:
        tag += f"_ff{d_ff}"
    if rank is not None:
        tag += f"_r{rank}"
    if args.activation != "relu":
        tag += f"_{args.activation}"

    print(f"\n{'='*70}")
    print(f"FLEX: d={args.d_model}, heads={args.n_heads}, "
          f"{'shared r='+str(args.repeats) if args.share_layers else str(args.n_layers)+' layers'}, "
          f"d_ff={d_ff}, vocab={vocab_size}, params={n_params}")
    print(f"lr={args.lr}, epochs={args.epochs}, seed={args.seed}")
    print(f"{'='*70}")

    # Optimizer
    ft_lr = args.finetune_lr
    if ft_lr is not None:
        # Fine-tuning: fresh cosine decay from ft_lr, short warmup, no warm restarts
        import math as _math
        opt = torch.optim.AdamW(model.parameters(), lr=ft_lr, weight_decay=args.wd)
        ft_warmup = 500
        ft_min_ratio = 0.01

        def lr_lambda(ep):
            if ep < ft_warmup:
                return (ep + 1) / max(1, ft_warmup)
            progress = (ep - ft_warmup) / max(1, args.epochs - ft_warmup)
            cosval = 0.5 * (1 + _math.cos(_math.pi * progress))
            return ft_min_ratio + (1.0 - ft_min_ratio) * cosval

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        print(f"  Fine-tune mode: lr={ft_lr}, {args.epochs} epochs, warmup={ft_warmup}")
    elif args.warmup > 0:
        # Use linear warmup + cosine decay
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

        def lr_lambda(ep):
            if ep < args.warmup:
                return ep / args.warmup
            progress = (ep - args.warmup) / max(1, args.epochs - args.warmup)
            return 0.5 * (1 + __import__('math').cos(progress * 3.14159))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=args.t0, T_mult=1, eta_min=1e-5
        )

    wandb_run = None
    if args.wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project="minimal-10digit-transformer",
                name=f"{tag}_{n_params}p_s{args.seed}",
                config=vars(args) | {"n_params": n_params, "tag": tag},
                tags=["flex", "all_reversed"],
                reinit=True,
            )
        except Exception:
            pass

    best_acc = 0.0
    best_epoch = 0
    start_ep = 0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        best_acc = ckpt.get("accuracy", 0.0)
        print(f"  Resumed from {args.resume}, acc={best_acc:.4f}")

    t0 = time.time()

    for ep in range(start_ep, args.epochs):
        model.train()
        total_loss = 0
        n_batches = 20

        for _ in range(n_batches):
            input_ids, labels = generate_batch(1024, device, vocab10=args.vocab10)
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
            results = evaluate(model, 10000, device, vocab10=args.vocab10)
            acc = results["exact_match"]
            print(f"  EVAL ep {ep}: exact={acc:.4f} digit_avg={results['digit_avg']:.4f}")
            print(f"    per-digit: {['%.3f' % d for d in results['per_digit']]}")
            show_examples(model, device, vocab10=args.vocab10)

            if acc > best_acc:
                best_acc = acc
                best_epoch = ep
                print(f"  ** BEST: {acc:.4f} **")

            # Save checkpoint on accuracy improvement or every 10000 epochs
            if acc >= best_acc or ep % 10000 == 0:
                ckpt_dir = Path(f"checkpoints/{tag}_s{args.seed}")
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "state_dict": model.state_dict(), "config": config,
                    "accuracy": acc, "n_params": n_params, "epoch": ep,
                    "digit_avg": results["digit_avg"],
                }, str(ckpt_dir / "best.pt"))

            if wandb_run:
                import wandb
                log_dict = {"eval_acc": acc, "best_acc": best_acc, "digit_avg": results["digit_avg"]}
                for i, d_acc in enumerate(results["per_digit"]):
                    log_dict[f"digit_{i}_acc"] = d_acc
                wandb.log(log_dict, step=ep)

        if best_acc >= 0.99:
            print(f"  Reached 99%+ at epoch {ep}, stopping early.")
            break

    final = evaluate(model, 50000, device, seed=99999, vocab10=args.vocab10)
    print(f"\n  FINAL: exact={final['exact_match']:.4f} digit_avg={final['digit_avg']:.4f}")
    print(f"    per-digit: {['%.3f' % d for d in final['per_digit']]}")
    show_examples(model, device, n=10, vocab10=args.vocab10)

    if wandb_run:
        import wandb
        wandb.finish()

    return {
        "d_model": args.d_model, "n_heads": args.n_heads,
        "n_layers": args.n_layers if not args.share_layers else 1,
        "n_repeats": args.repeats if args.share_layers else 0,
        "d_ff": d_ff, "vocab_size": vocab_size,
        "n_params": n_params, "seed": args.seed,
        "best_acc": best_acc, "best_epoch": best_epoch,
        "final_acc": final["exact_match"], "final_per_digit": final["per_digit"],
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--d-model", type=int, default=36)
    parser.add_argument("--n-heads", type=int, default=2)
    parser.add_argument("--n-layers", type=int, default=1, help="For non-shared mode")
    parser.add_argument("--d-ff", type=int, default=0, help="FFN dim (0=d_model)")
    parser.add_argument("--share-layers", action="store_true")
    parser.add_argument("--repeats", type=int, default=15, help="Repeats for shared mode")
    parser.add_argument("--vocab10", action="store_true", help="Use vocab=10 (map +/= to 0)")
    parser.add_argument("--rank", type=int, default=0, help="Low-rank factorization rank (0=full)")
    parser.add_argument("--use-bias", action="store_true")
    parser.add_argument("--activation", type=str, default="relu", choices=["relu", "gelu", "silu"])
    parser.add_argument("--epochs", type=int, default=15000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=0.01)
    parser.add_argument("--warmup", type=int, default=0, help="Warmup epochs")
    parser.add_argument("--t0", type=int, default=5000, help="CosineAnnealing T_0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume from")
    parser.add_argument("--finetune-lr", type=float, default=None, help="Fine-tune LR (fresh cosine decay, no warm restarts)")
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    result = train(args)

    tag = f"flex_d{args.d_model}_h{args.n_heads}"
    out_file = f"experiments/{tag}_results.json"
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved results to {out_file}")


if __name__ == "__main__":
    main()
