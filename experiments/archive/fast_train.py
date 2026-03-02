"""Fast training for minimal transformers.

Generates all data as GPU tensors directly using torch operations.
Avoids Python-level data generation overhead.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from minimal10digittransformer.model.transformer import MinimalTransformer, TransformerConfig, count_parameters


def generate_addition_batch_gpu(
    batch_size: int,
    max_digits: int = 10,
    device: str = "cuda",
    reversed_output: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a batch of addition problems directly on GPU.

    Format: AAAAAAAAAA+BBBBBBBBBB=CCCCCCCCCCC
    With reversed output: result digits in LSB-first order.

    Returns:
        input_ids: (batch, seq_len) - full sequence
        labels: (batch, seq_len) - -100 for input positions, digit tokens for output
    """
    # Generate random numbers
    a = torch.randint(0, 10**max_digits, (batch_size,), device=device, dtype=torch.long)
    b = torch.randint(0, 10**max_digits, (batch_size,), device=device, dtype=torch.long)
    c = a + b

    # Extract digits
    a_digits = []
    b_digits = []
    c_digits = []

    a_tmp = a.clone()
    b_tmp = b.clone()
    c_tmp = c.clone()

    for _ in range(max_digits):
        a_digits.append(a_tmp % 10)
        b_digits.append(b_tmp % 10)
        c_digits.append(c_tmp % 10)
        a_tmp = a_tmp // 10
        b_tmp = b_tmp // 10
        c_tmp = c_tmp // 10
    c_digits.append(c_tmp % 10)  # 11th digit for result

    # Stack: a_digits[0] is LSB
    a_tensor = torch.stack(a_digits, dim=1)  # (batch, max_digits) LSB first
    b_tensor = torch.stack(b_digits, dim=1)  # (batch, max_digits) LSB first
    c_tensor = torch.stack(c_digits, dim=1)  # (batch, max_digits+1) LSB first

    # Build input sequence
    plus_token = torch.full((batch_size, 1), 10, device=device, dtype=torch.long)  # '+'
    eq_token = torch.full((batch_size, 1), 11, device=device, dtype=torch.long)    # '='

    if reversed_output:
        # Input A in MSB-first (natural reading), output C in LSB-first
        a_input = a_tensor.flip(1)  # MSB first
        b_input = b_tensor.flip(1)  # MSB first
        c_output = c_tensor  # LSB first
    else:
        a_input = a_tensor.flip(1)  # MSB first
        b_input = b_tensor.flip(1)  # MSB first
        c_output = c_tensor.flip(1)  # MSB first

    # Full sequence: A + B = C
    input_ids = torch.cat([a_input, plus_token, b_input, eq_token, c_output], dim=1)

    # For causal LM: labels[t] = input_ids[t+1] (next token prediction)
    # logits[input_len-1] predicts input_ids[input_len] = C0
    input_len = max_digits * 2 + 2  # A + + + B + =
    labels = torch.full_like(input_ids, -100)
    labels[:, input_len - 1:-1] = c_output

    return input_ids, labels


def evaluate_fast(
    model: nn.Module,
    n_samples: int = 10000,
    max_digits: int = 10,
    device: str = "cuda",
    reversed_output: bool = True,
    seed: int = 42,
) -> dict:
    """Fast GPU-based evaluation."""
    model.eval()
    torch.manual_seed(seed)

    input_len = max_digits * 2 + 2
    correct = 0
    total = 0
    digit_correct = 0
    digit_total = 0

    batch_size = min(n_samples, 2048)

    with torch.no_grad():
        for start in range(0, n_samples, batch_size):
            bs = min(batch_size, n_samples - start)
            input_ids, labels = generate_addition_batch_gpu(bs, max_digits, device, reversed_output)

            logits = model(input_ids)
            # Get predictions for output positions
            pred_logits = logits[:, input_len - 1:-1, :]
            predictions = pred_logits.argmax(dim=-1)

            targets = labels[:, input_len:]

            # Exact match
            matches = (predictions == targets).all(dim=1)
            correct += matches.sum().item()
            total += bs

            # Per-digit accuracy
            digit_matches = (predictions == targets)
            digit_correct += digit_matches.sum().item()
            digit_total += digit_matches.numel()

    return {
        "exact_match": correct / total if total > 0 else 0.0,
        "digit_accuracy": digit_correct / digit_total if digit_total > 0 else 0.0,
        "correct": correct,
        "total": total,
    }


def train_fast(
    model_config: TransformerConfig,
    epochs: int = 5000,
    lr: float = 3e-3,
    weight_decay: float = 0.01,
    batch_size: int = 1024,
    eval_interval: int = 500,
    log_interval: int = 100,
    batches_per_epoch: int = 50,
    max_digits: int = 10,
    reversed_output: bool = True,
    device: str = "cuda",
    name: str = "experiment",
    use_wandb: bool = False,
    seed: int = 42,
    curriculum: bool = True,
    curriculum_step: int = 500,
    min_lr: float = 1e-5,
    save_best: bool = True,
) -> dict:
    """Fast training loop."""
    torch.manual_seed(seed)
    model = MinimalTransformer(model_config).to(device)
    n_params = count_parameters(model)

    print(f"Training {name}: {n_params} params")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)

    wandb_run = None
    if use_wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project="minimal-10digit-transformer",
                name=name,
                config={
                    "n_params": n_params,
                    "model_config": str(model_config),
                    "lr": lr, "epochs": epochs, "batch_size": batch_size,
                    "reversed_output": reversed_output, "curriculum": curriculum,
                },
                tags=["fast_train"],
            )
        except Exception:
            pass

    best_accuracy = 0.0
    best_epoch = 0
    start_time = time.time()

    for epoch in range(epochs):
        model.train()

        # Curriculum: gradually increase digit count
        if curriculum:
            current_digits = min(2 + epoch // curriculum_step, max_digits)
        else:
            current_digits = max_digits

        epoch_loss = 0.0
        for _ in range(batches_per_epoch):
            input_ids, labels = generate_addition_batch_gpu(
                batch_size, current_digits, device, reversed_output
            )

            logits = model(input_ids)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / batches_per_epoch

        if epoch % log_interval == 0:
            lr_now = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start_time
            print(f"  Epoch {epoch:6d} | Loss: {avg_loss:.4f} | LR: {lr_now:.2e} | Digits: {current_digits} | {elapsed:.0f}s")

            if wandb_run:
                import wandb
                wandb.log({"train/loss": avg_loss, "train/lr": lr_now, "train/digits": current_digits}, step=epoch)

        if epoch % eval_interval == 0 and epoch > 0:
            results = evaluate_fast(model, 10000, max_digits, device, reversed_output, seed=12345)
            acc = results["exact_match"]
            print(f"  EVAL Epoch {epoch}: exact={acc:.4f} digit={results['digit_accuracy']:.4f}")

            if acc > best_accuracy:
                best_accuracy = acc
                best_epoch = epoch
                print(f"  ** New best: {acc:.4f} **")
                if save_best:
                    save_path = Path("checkpoints") / name
                    save_path.mkdir(parents=True, exist_ok=True)
                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "model_config": model_config,
                        "accuracy": acc,
                        "epoch": epoch,
                        "n_params": n_params,
                    }, save_path / "best.pt")

            if wandb_run:
                import wandb
                wandb.log({"eval/exact_match": acc, "eval/digit_accuracy": results["digit_accuracy"], "eval/best": best_accuracy}, step=epoch)

    # Final eval
    results = evaluate_fast(model, 50000, max_digits, device, reversed_output, seed=99999)
    final_acc = results["exact_match"]
    print(f"\n  Final ({name}): {final_acc:.4f} exact, {results['digit_accuracy']:.4f} digit")
    print(f"  Best: {best_accuracy:.4f} at epoch {best_epoch}")

    if wandb_run:
        import wandb
        wandb.finish()

    return {
        "name": name,
        "n_params": n_params,
        "best_accuracy": best_accuracy,
        "best_epoch": best_epoch,
        "final_accuracy": final_acc,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--d-model", type=int, default=8)
    parser.add_argument("--n-heads", type=int, default=1)
    parser.add_argument("--d-ff", type=int, default=8)
    parser.add_argument("--n-repeats", type=int, default=15)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--embed-dim", type=int, default=None)
    parser.add_argument("--norm", type=str, default="rmsnorm")
    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--use-bias", action="store_true")
    parser.add_argument("--no-ffn", action="store_true")
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--name", type=str, default="fast_exp")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--no-curriculum", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = TransformerConfig(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=1,
        d_ff=args.d_ff if not args.no_ffn else 1,
        share_layers=True,
        n_layer_repeats=args.n_repeats,
        pe_type="sinusoidal",
        pe_period=11.0,
        rank=args.rank,
        embed_dim=args.embed_dim,
        norm_type=args.norm,
        activation=args.activation,
        use_bias=args.use_bias,
        ffn_type="none" if args.no_ffn else "standard",
    )

    train_fast(
        config,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        name=args.name,
        device=device,
        use_wandb=args.wandb,
        curriculum=not args.no_curriculum,
        seed=args.seed,
    )
