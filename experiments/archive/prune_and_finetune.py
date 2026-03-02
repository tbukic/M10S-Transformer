"""Progressive pruning: Train d_src to 100%, prune to d_target, fine-tune.

Instead of training small models from scratch (which gets stuck at the 1.884
plateau), start from a trained larger model and structurally prune dimensions,
then fine-tune. This avoids the bad optimization landscape.

Approach:
1. Load trained model (e.g., d=36, 100% accuracy)
2. Score dimension importance (activation magnitude + weight magnitude)
3. Remove least important dimensions from all weight matrices
4. Fine-tune the pruned model with lower LR
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
    return {"exact_match": exact_acc, "per_digit": per_digit, "digit_avg": sum(per_digit) / len(per_digit)}


def show_examples(model, device, n=5):
    model.eval()
    with torch.no_grad():
        input_ids, _ = generate_batch(n, device)
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


def compute_dimension_importance(model, device, n_samples=10000):
    """Score each hidden dimension's importance by activation + weight magnitude.

    Returns importance scores per dimension (higher = more important).
    For attention heads: scores are per head_dim, replicated across heads.
    """
    model.eval()
    d = model.config.d_model
    n_heads = model.config.n_heads
    head_dim = d // n_heads

    # Activation-based importance: run data through model, measure output activations
    act_importance = torch.zeros(d, device=device)
    torch.manual_seed(123)
    with torch.no_grad():
        for _ in range(0, n_samples, 1024):
            input_ids, _ = generate_batch(1024, device)
            x = model.get_embeddings(input_ids)
            if model.config.share_layers:
                for _ in range(model.n_repeats):
                    x = model.shared_layer(x)
            else:
                for layer in model.layers:
                    x = layer(x)
            # Score by output position activation magnitude
            out_acts = x[:, INPUT_LEN:, :]  # (batch, 11, d)
            act_importance += out_acts.abs().sum(dim=(0, 1))

    # Weight-based importance: sum of absolute weight values touching each dimension
    weight_importance = torch.zeros(d, device=device)
    for name, param in model.named_parameters():
        if param.dim() == 2:
            rows, cols = param.shape
            if cols == d:
                weight_importance += param.abs().sum(dim=0)
            if rows == d:
                weight_importance += param.abs().sum(dim=1)
        elif param.dim() == 1 and param.shape[0] == d:
            weight_importance += param.abs()

    # Normalize and combine
    act_norm = act_importance / act_importance.max().clamp(min=1e-8)
    wt_norm = weight_importance / weight_importance.max().clamp(min=1e-8)
    combined = act_norm + wt_norm

    return combined


def prune_model_simple(src_model, target_d, device="cuda"):
    """Prune by keeping the most important dimensions.

    For attention: keeps top head_dim dimensions within each head.
    For FFN and embeddings: keeps top dimensions overall.
    """
    src_d = src_model.config.d_model
    n_heads = src_model.config.n_heads
    src_head_dim = src_d // n_heads
    tgt_head_dim = target_d // n_heads

    assert target_d < src_d
    assert target_d % n_heads == 0, f"target_d={target_d} must be divisible by n_heads={n_heads}"

    # Compute importance
    importance = compute_dimension_importance(src_model, device)

    # For general dimensions (embedding, FFN, norms): keep top target_d
    _, keep_general = torch.topk(importance, target_d)
    keep_general = keep_general.sort().values
    print(f"  General keep dims: {keep_general.cpu().tolist()[:10]}...")

    # For attention: keep top tgt_head_dim within each head
    keep_attn_out = []  # Indices into the d_model-sized output of q/k/v projections
    keep_attn_in = []   # Same for input of o_proj
    for h in range(n_heads):
        head_start = h * src_head_dim
        head_importance = importance[head_start:head_start + src_head_dim]
        _, head_keep = torch.topk(head_importance, tgt_head_dim)
        head_keep = head_keep.sort().values + head_start
        keep_attn_out.extend(head_keep.cpu().tolist())
    keep_attn_out = torch.tensor(keep_attn_out, device=device)

    # Create new model
    new_config = TransformerConfig(
        d_model=target_d, n_heads=n_heads,
        n_layers=src_model.config.n_layers, d_ff=target_d,
        share_layers=src_model.config.share_layers,
        n_layer_repeats=src_model.config.n_layer_repeats,
        norm_type=src_model.config.norm_type,
        activation=src_model.config.activation,
        pe_period=src_model.config.pe_period,
        vocab_size=src_model.config.vocab_size,
    )
    new_model = MinimalTransformer(new_config).to(device)

    with torch.no_grad():
        # Token embedding: (vocab, src_d) -> (vocab, target_d)
        new_model.tok_embed.weight.copy_(
            src_model.tok_embed.weight[:, keep_general]
        )

        src_layer = src_model.shared_layer if src_model.config.share_layers else src_model.layers[0]
        dst_layer = new_model.shared_layer if new_model.config.share_layers else new_model.layers[0]

        # Attention: q,k,v projections (d_out, d_in) where d_out has head structure
        for proj_name in ['q_proj', 'k_proj', 'v_proj']:
            src_w = getattr(src_layer.attn, proj_name).weight  # (src_d, src_d)
            dst_w = getattr(dst_layer.attn, proj_name).weight  # (target_d, target_d)
            # Select rows from keep_attn_out (output has head structure)
            # Select cols from keep_general (input is general d_model)
            dst_w.copy_(src_w[keep_attn_out][:, keep_general])

        # o_proj: (d_model, d_model) - input has head structure, output is general
        src_w = src_layer.attn.o_proj.weight
        dst_w = dst_layer.attn.o_proj.weight
        dst_w.copy_(src_w[keep_general][:, keep_attn_out])

        # FFN: up (d_ff, d_model), down (d_model, d_ff)
        dst_layer.ffn.up.weight.copy_(
            src_layer.ffn.up.weight[keep_general][:, keep_general]
        )
        dst_layer.ffn.down.weight.copy_(
            src_layer.ffn.down.weight[keep_general][:, keep_general]
        )

        # Norms
        dst_layer.norm1.weight.copy_(src_layer.norm1.weight[keep_general])
        dst_layer.norm2.weight.copy_(src_layer.norm2.weight[keep_general])
        new_model.output_norm.weight.copy_(src_model.output_norm.weight[keep_general])

    return new_model, keep_general


def finetune(model, epochs, lr, device, use_wandb=False, tag=""):
    """Fine-tune a pruned model."""
    n_params = count_parameters(model)
    d = model.config.d_model

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=2000, T_mult=1, eta_min=1e-5
    )

    wandb_run = None
    if use_wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project="minimal-10digit-transformer",
                name=f"prune_{tag}_{n_params}p",
                config={"d_model": d, "n_params": n_params, "lr": lr,
                        "epochs": epochs, "method": "progressive_prune"},
                tags=["prune", "all_reversed"],
                reinit=True,
            )
        except Exception:
            pass

    best_acc = 0.0
    best_epoch = -1
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

        if (ep % 500 == 0 and ep > 0) or ep == 100:
            results = evaluate(model, 10000, device)
            acc = results["exact_match"]
            print(f"  EVAL ep {ep}: exact={acc:.4f} digit_avg={results['digit_avg']:.4f}")
            print(f"    per-digit: {['%.3f' % d for d in results['per_digit']]}")
            show_examples(model, device)

            if acc > best_acc:
                best_acc = acc
                best_epoch = ep
                print(f"  ** BEST: {acc:.4f} **")
                ckpt_dir = Path(f"checkpoints/prune_{tag}")
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "state_dict": model.state_dict(),
                    "config": model.config,
                    "accuracy": acc,
                    "n_params": n_params,
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

    final = evaluate(model, 50000, device, seed=99999)
    print(f"\n  FINAL: exact={final['exact_match']:.4f} digit_avg={final['digit_avg']:.4f}")
    print(f"    per-digit: {['%.3f' % d for d in final['per_digit']]}")
    show_examples(model, device, n=10)

    if wandb_run:
        import wandb
        wandb.finish()

    return {
        "d_model": d, "n_params": n_params, "method": "prune_finetune",
        "best_acc": best_acc, "best_epoch": best_epoch,
        "final_acc": final["exact_match"], "final_per_digit": final["per_digit"],
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-ckpt", type=str, default="checkpoints/rev_d36_r15_h2_s0/best.pt")
    parser.add_argument("--target-d", type=int, default=34)
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load source model
    print(f"Loading source model from {args.source_ckpt}")
    ckpt = torch.load(args.source_ckpt, map_location=device, weights_only=False)
    src_config = ckpt["config"]
    src_model = MinimalTransformer(src_config).to(device)
    src_model.load_state_dict(ckpt["state_dict"])
    src_d = src_config.d_model
    src_params = count_parameters(src_model)
    print(f"  Source: d={src_d}, params={src_params}, accuracy={ckpt.get('accuracy', 'unknown')}")

    # Evaluate source
    src_results = evaluate(src_model, 10000, device)
    print(f"  Source eval: exact={src_results['exact_match']:.4f}")

    # Prune
    print(f"\nPruning from d={src_d} to d={args.target_d}...")
    pruned_model, keep_dims = prune_model_simple(src_model, args.target_d, device=device)
    pruned_params = count_parameters(pruned_model)
    print(f"  Pruned: d={args.target_d}, params={pruned_params}")

    # Evaluate immediately after pruning
    pruned_results = evaluate(pruned_model, 10000, device)
    print(f"  Post-prune eval: exact={pruned_results['exact_match']:.4f} digit_avg={pruned_results['digit_avg']:.4f}")
    print(f"    per-digit: {['%.3f' % d for d in pruned_results['per_digit']]}")
    show_examples(pruned_model, device)

    # Fine-tune
    print(f"\nFine-tuning d={args.target_d} ({pruned_params}p) with lr={args.lr}, epochs={args.epochs}...")
    tag = f"d{src_d}_to_d{args.target_d}"
    result = finetune(pruned_model, args.epochs, args.lr, device, args.wandb, tag)

    # Save results
    result["source_d"] = src_d
    result["source_params"] = src_params
    result["prune_dims_kept"] = keep_dims.cpu().tolist()
    out_file = f"experiments/prune_{tag}_results.json"
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved results to {out_file}")


if __name__ == "__main__":
    main()
