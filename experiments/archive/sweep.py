"""Systematic sweep of architectures for minimal 10-digit addition.

Runs experiments in parallel on GPU, focusing on the most promising
configurations from research:
1. Looped transformers with shared layers
2. Low-rank factorization
3. Reversed output (LSB-first)
4. Period-11 sinusoidal PE
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
    # We only want loss on output positions
    # logits[input_len-1] should predict input_ids[input_len] = C0
    # logits[input_len] should predict input_ids[input_len+1] = C1
    # So labels[input_len-1] = C0, labels[input_len] = C1, etc.
    labels = torch.full_like(input_ids, -100)
    labels[:, input_len - 1:-1] = c_t

    return input_ids, labels


def evaluate(model, n_samples, max_digits, device, seed=42):
    model.eval()
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    input_len = max_digits * 2 + 2
    correct = 0
    total = 0

    with torch.no_grad():
        for _ in range(0, n_samples, 2048):
            bs = min(2048, n_samples - total)
            if bs <= 0:
                break
            input_ids, labels = generate_batch(bs, max_digits, device)
            logits = model(input_ids)
            preds = logits[:, input_len - 1:-1, :].argmax(dim=-1)
            targets = labels[:, input_len:]
            correct += (preds == targets).all(dim=1).sum().item()
            total += bs

    return correct / total if total > 0 else 0.0


def train_one(config, name, epochs=10000, lr=3e-3, wd=0.01, bs=1024,
              device="cuda", eval_every=1000, log_every=200, use_wandb=False):
    """Train a single configuration."""
    model = MinimalTransformer(config).to(device)
    n_params = count_parameters(model)
    print(f"\n{'='*50}")
    print(f"{name}: {n_params} params")
    print(f"{'='*50}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)

    wandb_run = None
    if use_wandb:
        try:
            import wandb
            wandb_run = wandb.init(project="minimal-10digit-transformer", name=name,
                config={"n_params": n_params, "epochs": epochs, "lr": lr}, tags=["sweep"], reinit=True)
        except Exception:
            pass

    best_acc = 0.0
    best_epoch = 0
    t0 = time.time()

    for ep in range(epochs):
        model.train()
        total_loss = 0
        n_batches = 20  # Fixed batches per epoch for speed

        for _ in range(n_batches):
            input_ids, labels = generate_batch(bs, 10, device)
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
            print(f"  ep {ep:6d} | loss {avg_loss:.4f} | lr {opt.param_groups[0]['lr']:.2e} | {time.time()-t0:.0f}s")
            if wandb_run:
                import wandb
                wandb.log({"loss": avg_loss, "lr": opt.param_groups[0]["lr"]}, step=ep)

        if ep % eval_every == 0 and ep > 0:
            acc = evaluate(model, 10000, 10, device, seed=12345)
            print(f"  EVAL ep {ep}: {acc:.4f}")
            if acc > best_acc:
                best_acc = acc
                best_epoch = ep
                print(f"  ** BEST: {acc:.4f} **")
                Path(f"checkpoints/{name}").mkdir(parents=True, exist_ok=True)
                torch.save({"state_dict": model.state_dict(), "config": config,
                           "accuracy": acc, "n_params": n_params}, f"checkpoints/{name}/best.pt")
            if wandb_run:
                import wandb
                wandb.log({"eval_acc": acc, "best_acc": best_acc}, step=ep)

    # Final eval on larger set
    acc = evaluate(model, 50000, 10, device, seed=99999)
    print(f"  FINAL: {acc:.4f} | best: {best_acc:.4f} at ep {best_epoch}")

    if wandb_run:
        import wandb
        wandb.finish()

    return {"name": name, "n_params": n_params, "best_acc": best_acc, "final_acc": acc, "best_epoch": best_epoch}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", type=str, default="all", help="Experiment group: all, tiny, medium, large")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--epochs", type=int, default=15000)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    configs = {}

    # Group 1: TINY (<200 params) - can we learn with so few params?
    if args.group in ("all", "tiny"):
        configs["tiny_132p_d4r1_30rep"] = TransformerConfig(
            d_model=4, n_heads=1, n_layers=1, d_ff=4,
            share_layers=True, n_layer_repeats=30,
            rank=1, norm_type="none", use_bias=True, activation="relu",
        )
        configs["tiny_156p_d4r2_20rep"] = TransformerConfig(
            d_model=4, n_heads=1, n_layers=1, d_ff=4,
            share_layers=True, n_layer_repeats=20,
            rank=2, norm_type="none", activation="relu",
        )
        configs["tiny_172p_d4r2_30rep_bias"] = TransformerConfig(
            d_model=4, n_heads=1, n_layers=1, d_ff=4,
            share_layers=True, n_layer_repeats=30,
            rank=2, norm_type="none", use_bias=True, activation="relu",
        )

    # Group 2: MEDIUM (200-400 params) - sweet spot?
    if args.group in ("all", "medium"):
        configs["med_208p_d8r1_20rep"] = TransformerConfig(
            d_model=8, n_heads=1, n_layers=1, d_ff=4,
            share_layers=True, n_layer_repeats=20,
            rank=1, norm_type="none", activation="relu",
        )
        configs["med_320p_d8r2_20rep"] = TransformerConfig(
            d_model=8, n_heads=1, n_layers=1, d_ff=4,
            share_layers=True, n_layer_repeats=20,
            rank=2, norm_type="rmsnorm", activation="relu",
        )
        configs["med_324p_d6_15rep"] = TransformerConfig(
            d_model=6, n_heads=1, n_layers=1, d_ff=6,
            share_layers=True, n_layer_repeats=15,
            norm_type="rmsnorm", activation="relu",
        )
        configs["med_392p_d8_noffn_20rep"] = TransformerConfig(
            d_model=8, n_heads=1, n_layers=1,
            ffn_type="none", share_layers=True, n_layer_repeats=20,
            norm_type="rmsnorm", activation="relu",
        )

    # Group 3: LARGE (400-700 params) - should definitely work
    if args.group in ("all", "large"):
        configs["large_528p_d8_20rep"] = TransformerConfig(
            d_model=8, n_heads=1, n_layers=1, d_ff=8,
            share_layers=True, n_layer_repeats=20,
            norm_type="rmsnorm", activation="relu",
        )
        configs["large_656p_d8_ff16"] = TransformerConfig(
            d_model=8, n_heads=1, n_layers=1, d_ff=16,
            share_layers=True, n_layer_repeats=10,
            norm_type="rmsnorm", activation="relu",
        )

    # Print parameter counts
    print("Configurations:")
    for name, cfg in sorted(configs.items(), key=lambda x: count_parameters(MinimalTransformer(x[1]))):
        n = count_parameters(MinimalTransformer(cfg))
        print(f"  {name}: {n} params")

    results = []
    for name, cfg in sorted(configs.items(), key=lambda x: count_parameters(MinimalTransformer(x[1]))):
        try:
            r = train_one(cfg, name, epochs=args.epochs, use_wandb=args.wandb, device=device)
            results.append(r)
        except Exception as e:
            print(f"ERROR in {name}: {e}")
            import traceback; traceback.print_exc()
            results.append({"name": name, "error": str(e)})

        # Save intermediate
        with open("experiments/sweep_results.json", "w") as f:
            json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("SWEEP RESULTS")
    print("=" * 70)
    for r in results:
        if "error" in r:
            print(f"  {r['name']}: ERROR")
        else:
            print(f"  {r['name']}: {r['n_params']}p | best={r['best_acc']:.4f} | final={r['final_acc']:.4f}")


if __name__ == "__main__":
    main()
