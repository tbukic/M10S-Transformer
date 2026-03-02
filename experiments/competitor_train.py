"""Competitor architecture training script.

Reimplements the top AdderBoard entries' architecture:
- Rank-factorized linear layers (LowRankLinear)
- Low-rank position embeddings (LowRankEmbedding)
- shareA_tieKV attention (shared bottleneck, K=V tied)
- Tied input/output embeddings
- 3-phase curriculum, cosine decay with warmup

References:
  rezabyt/digit-addition-311p (311p, d=4 ff=8 r=3)
  yinglunz (456p, d=7 ff=14 r=3 attn_out_r=2)
  h3nock/tiny-adder-lab (335p, d=4 ff=12 r=3)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTHONUNBUFFERED", "1")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------------

class LowRankLinear(nn.Module):
    """Low-rank factorized linear: y = x @ A @ B."""

    def __init__(self, in_features: int, out_features: int, rank: int):
        super().__init__()
        self.A = nn.Parameter(torch.empty(in_features, rank))
        self.B = nn.Parameter(torch.empty(rank, out_features))
        nn.init.normal_(self.A, std=math.sqrt(2.0 / (in_features + rank)))
        nn.init.normal_(self.B, std=math.sqrt(2.0 / (rank + out_features)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.A @ self.B


class LowRankEmbedding(nn.Module):
    """Low-rank positional embedding: PE[pos] = A[pos] @ B."""

    def __init__(self, seq_len: int, d_model: int, rank: int):
        super().__init__()
        self.A = nn.Parameter(torch.empty(seq_len, rank))
        self.B = nn.Parameter(torch.empty(rank, d_model))
        nn.init.normal_(self.A, std=0.02)
        nn.init.normal_(self.B, std=0.02)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        return self.A[positions] @ self.B


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (weight only, no bias)."""

    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class CompetitorAttention(nn.Module):
    """shareA_tieKV attention: shared bottleneck A, tied K=V.

    Params: qkv_A (d, qkv_rank), qkv_Bq (qkv_rank, d), qkv_Bkv (qkv_rank, d)
    Output projection: LowRankLinear(d, d, attn_out_rank)
    Single head, causal.
    """

    def __init__(self, d_model: int, qkv_rank: int, attn_out_rank: int):
        super().__init__()
        self.d_model = d_model

        # shareA_tieKV: shared A, separate B for Q and tied K=V
        self.qkv_A = nn.Parameter(torch.empty(d_model, qkv_rank))
        self.qkv_Bq = nn.Parameter(torch.empty(qkv_rank, d_model))
        self.qkv_Bkv = nn.Parameter(torch.empty(qkv_rank, d_model))

        nn.init.normal_(self.qkv_A, std=math.sqrt(2.0 / (d_model + qkv_rank)))
        nn.init.normal_(self.qkv_Bq, std=math.sqrt(2.0 / (qkv_rank + d_model)))
        nn.init.normal_(self.qkv_Bkv, std=math.sqrt(2.0 / (qkv_rank + d_model)))

        # Output projection
        self.proj = LowRankLinear(d_model, d_model, attn_out_rank)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        # Shared bottleneck
        h = x @ self.qkv_A          # (B, T, qkv_rank)
        q = h @ self.qkv_Bq         # (B, T, d_model)
        k = v = h @ self.qkv_Bkv    # (B, T, d_model) -- K and V are identical

        # Scaled dot-product attention (single head)
        scale = C ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale  # (B, T, T)

        # Causal mask
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        attn = attn.masked_fill(mask, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        out = attn @ v  # (B, T, d_model)

        return self.proj(out)


class CompetitorFFN(nn.Module):
    """Feed-forward network with low-rank up and down projections, ReLU activation."""

    def __init__(self, d_model: int, d_ff: int, ffn_rank: int):
        super().__init__()
        self.up = LowRankLinear(d_model, d_ff, ffn_rank)
        self.down = LowRankLinear(d_ff, d_model, ffn_rank)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.gelu(self.up(x)))


class CompetitorTransformerBlock(nn.Module):
    """Pre-norm residual transformer block."""

    def __init__(self, d_model: int, d_ff: int, qkv_rank: int,
                 attn_out_rank: int, ffn_rank: int):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = CompetitorAttention(d_model, qkv_rank, attn_out_rank)
        self.norm2 = RMSNorm(d_model)
        self.ffn = CompetitorFFN(d_model, d_ff, ffn_rank)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class CompetitorModel(nn.Module):
    """Complete competitor model: tied embeddings, low-rank PE, transformer block(s).

    Supports --share-layers --repeats N for weight sharing (repeat the single block N times).
    """

    def __init__(self, vocab_size: int, d_model: int, d_ff: int,
                 max_seq_len: int, pos_rank: int, qkv_rank: int,
                 attn_out_rank: int, ffn_rank: int,
                 share_layers: bool = False, repeats: int = 1):
        super().__init__()
        self.d_model = d_model
        self.share_layers = share_layers
        self.repeats = repeats

        # Token embedding (tied with output head)
        self.token_emb = nn.Embedding(vocab_size, d_model)

        # Low-rank positional embedding
        self.pos_emb = LowRankEmbedding(max_seq_len, d_model, pos_rank)

        # Transformer block(s)
        if share_layers:
            self.block = CompetitorTransformerBlock(
                d_model, d_ff, qkv_rank, attn_out_rank, ffn_rank)
        else:
            self.block = CompetitorTransformerBlock(
                d_model, d_ff, qkv_rank, attn_out_rank, ffn_rank)

        # Final norm
        self.final_norm = RMSNorm(d_model)

        # Output head: tied to token embedding
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

        # Match competitor init: normal_(std=0.02) for embeddings and linear layers
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape

        # Embeddings
        x = self.token_emb(input_ids)
        positions = torch.arange(T, device=input_ids.device)
        x = x + self.pos_emb(positions)

        # Transformer block(s)
        n_passes = self.repeats if self.share_layers else 1
        for _ in range(n_passes):
            x = self.block(x)

        # Output
        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits


def count_parameters(model: nn.Module) -> int:
    """Count unique trainable parameters (accounts for weight tying)."""
    seen = set()
    total = 0
    for p in model.parameters():
        if p.data_ptr() not in seen:
            seen.add(p.data_ptr())
            total += p.numel()
    return total


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

MAX_DIGITS = 10
SEQ_LEN = 33  # 22 prompt + 11 sum digits (EOS is the 34th in full, but model input is 33)

# Vocab=14: 0-9 digits, 10='+', 11='=', 12=PAD, 13=EOS
PLUS_TOK = 10
EQ_TOK = 11
PAD_TOK = 12
EOS_TOK = 13

# For vocab10 mode
INPUT_LEN_V10 = MAX_DIGITS * 2 + 2  # 22
OUTPUT_LEN_V10 = MAX_DIGITS + 1     # 11


def generate_batch_v14(batch_size: int, max_digits: int, device: torch.device):
    """Generate a batch in competitor format (vocab=14, MSD-first input, LSB-first output).

    Input: "0000000005+0000000007=" (MSD-first, zero-padded to 10 digits)
    Target: reversed sum digits + EOS (LSB-first)
    Full sequence: 22 prompt + 11 sum digits = 33 tokens for model input
    Labels: -100 for prompt, sum digits + EOS for output part
    """
    # Random operands up to max_digits
    upper = 10 ** max_digits
    a = torch.randint(0, upper, (batch_size,), device=device, dtype=torch.long)
    b = torch.randint(0, upper, (batch_size,), device=device, dtype=torch.long)
    c = a + b

    # Extract digits MSD-first for input operands (zero-padded to 10 digits)
    a_digits_msd = []
    b_digits_msd = []
    a_tmp, b_tmp = a.clone(), b.clone()
    for _ in range(MAX_DIGITS):
        a_digits_msd.append(a_tmp % 10)
        b_digits_msd.append(b_tmp % 10)
        a_tmp //= 10
        b_tmp //= 10
    # Reverse to get MSD-first
    a_digits_msd.reverse()
    b_digits_msd.reverse()

    # Extract digits LSB-first for sum (11 digits including carry)
    c_digits_lsb = []
    c_tmp = c.clone()
    for _ in range(MAX_DIGITS + 1):
        c_digits_lsb.append(c_tmp % 10)
        c_tmp //= 10

    a_t = torch.stack(a_digits_msd, dim=1)  # (B, 10) MSD-first
    b_t = torch.stack(b_digits_msd, dim=1)  # (B, 10) MSD-first
    c_t = torch.stack(c_digits_lsb, dim=1)  # (B, 11) LSB-first

    # Build full sequence: A + B = C_reversed EOS
    # Prompt: a_digits(10) + '+'(1) + b_digits(10) + '='(1) = 22 tokens
    # Output: c_digits(11) = 11 tokens
    # Total input to model: 33 tokens (we shift internally for causal LM)
    plus = torch.full((batch_size, 1), PLUS_TOK, device=device, dtype=torch.long)
    eq = torch.full((batch_size, 1), EQ_TOK, device=device, dtype=torch.long)

    # Full input: prompt + sum digits (33 tokens, last target is predicted from pos 32)
    input_ids = torch.cat([a_t, plus, b_t, eq, c_t], dim=1)  # (B, 33)

    # Labels: -100 for first 21 prompt tokens, then sum digits shifted
    # For causal LM: labels[i] = next token at position i
    # Positions 0..20: prompt tokens -> label = -100
    # Position 21 (= sign): label = c_t[:, 0] (first sum digit)
    # Position 21+k: label = c_t[:, k] for k in 0..10
    # But we only have 33 positions (0..32), and the last target is at position 32
    labels = torch.full_like(input_ids, -100)
    # The = sign is at position 21. After it comes c_t (positions 22..32).
    # For next-token prediction: label at position 21 = c_t[:, 0], label at pos 22 = c_t[:, 1], ...
    # label at pos 31 = c_t[:, 10]
    # We use the standard causal LM setup: logits[:, t, :] predicts input_ids[:, t+1]
    # So labels should be shifted: labels[:, t] = input_ids[:, t+1]
    # But we only want loss on sum digits.
    # Position 21 is '='. We want to predict c_t[:, 0] from position 21.
    # In standard shifted setup: labels = input_ids[:, 1:], input for logits = input_ids[:, :-1]
    # But here we keep full input_ids and handle shifting in loss.
    # Let's match the competitor approach: loss on positions where we predict sum digits.
    # labels[pos] = token the model should predict at position pos
    # At pos 21 (= sign), model should predict first sum digit c_t[:, 0]
    labels[:, 21:32] = c_t[:, :11]

    return input_ids, labels


def generate_batch_v10(batch_size: int, max_digits: int, device: torch.device):
    """Generate batch in our format (vocab=10, all-reversed, +/= mapped to 0)."""
    upper = 10 ** max_digits
    a = torch.randint(0, upper, (batch_size,), device=device, dtype=torch.long)
    b = torch.randint(0, upper, (batch_size,), device=device, dtype=torch.long)
    c = a + b

    # Extract digits LSB-first
    a_digits, b_digits, c_digits = [], [], []
    a_tmp, b_tmp, c_tmp = a.clone(), b.clone(), c.clone()
    for _ in range(MAX_DIGITS):
        a_digits.append(a_tmp % 10)
        b_digits.append(b_tmp % 10)
        c_digits.append(c_tmp % 10)
        a_tmp //= 10
        b_tmp //= 10
        c_tmp //= 10
    c_digits.append(c_tmp % 10)

    a_t = torch.stack(a_digits, dim=1)
    b_t = torch.stack(b_digits, dim=1)
    c_t = torch.stack(c_digits, dim=1)

    plus = torch.zeros((batch_size, 1), device=device, dtype=torch.long)
    eq = torch.zeros((batch_size, 1), device=device, dtype=torch.long)

    input_ids = torch.cat([a_t, plus, b_t, eq, c_t], dim=1)
    labels = torch.full_like(input_ids, -100)
    labels[:, INPUT_LEN_V10 - 1:-1] = c_t
    return input_ids, labels


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_v14(model, n_samples: int, device, seed: int = 42):
    """Evaluate with vocab=14 format. Returns exact-match and per-digit accuracy."""
    model.eval()
    correct = 0
    total = 0
    n_output = MAX_DIGITS + 1  # 11 sum digits
    digit_correct = torch.zeros(n_output, device=device)
    digit_total = torch.zeros(n_output, device=device)

    torch.manual_seed(seed)
    with torch.no_grad():
        for _ in range(0, n_samples, 2048):
            bs = min(2048, n_samples - total)
            if bs <= 0:
                break
            input_ids, labels = generate_batch_v14(bs, MAX_DIGITS, device)
            logits = model(input_ids)
            # Predictions at positions 21..31 (predicting sum digits)
            preds = logits[:, 21:32, :].argmax(dim=-1)  # (B, 11)
            targets = input_ids[:, 22:33]  # sum digits at positions 22..32
            correct += (preds == targets).all(dim=1).sum().item()
            matches = (preds == targets)
            digit_correct += matches.sum(dim=0).float()
            digit_total += torch.full((n_output,), float(bs), device=device)
            total += bs

    exact_acc = correct / total if total > 0 else 0.0
    per_digit = (digit_correct / digit_total.clamp(min=1)).cpu().tolist()
    return {
        "exact_match": exact_acc,
        "per_digit": per_digit,
        "digit_avg": sum(per_digit) / len(per_digit),
    }


def evaluate_v10(model, n_samples: int, device, seed: int = 42):
    """Evaluate with vocab=10 format."""
    model.eval()
    correct = 0
    total = 0
    n_output = OUTPUT_LEN_V10
    digit_correct = torch.zeros(n_output, device=device)
    digit_total = torch.zeros(n_output, device=device)

    torch.manual_seed(seed)
    with torch.no_grad():
        for _ in range(0, n_samples, 2048):
            bs = min(2048, n_samples - total)
            if bs <= 0:
                break
            input_ids, _ = generate_batch_v10(bs, MAX_DIGITS, device)
            logits = model(input_ids)
            preds = logits[:, INPUT_LEN_V10 - 1:-1, :].argmax(dim=-1)
            targets = input_ids[:, INPUT_LEN_V10:]
            correct += (preds == targets).all(dim=1).sum().item()
            matches = (preds == targets)
            digit_correct += matches.sum(dim=0).float()
            digit_total += torch.full((n_output,), float(bs), device=device)
            total += bs

    exact_acc = correct / total if total > 0 else 0.0
    per_digit = (digit_correct / digit_total.clamp(min=1)).cpu().tolist()
    return {
        "exact_match": exact_acc,
        "per_digit": per_digit,
        "digit_avg": sum(per_digit) / len(per_digit),
    }


def show_examples_v14(model, device, n: int = 5):
    """Show sample predictions in vocab=14 format."""
    model.eval()
    with torch.no_grad():
        input_ids, _ = generate_batch_v14(n, MAX_DIGITS, device)
        logits = model(input_ids)
        preds = logits[:, 21:32, :].argmax(dim=-1)
        targets = input_ids[:, 22:33]

    for i in range(n):
        seq = input_ids[i].cpu().tolist()
        a_str = "".join(str(d) for d in seq[:10])
        b_str = "".join(str(d) for d in seq[11:21])
        # Reverse LSB-first to get human-readable
        pred_digits = preds[i].cpu().tolist()
        tgt_digits = targets[i].cpu().tolist()
        pred_str = "".join(str(d) for d in reversed(pred_digits))
        tgt_str = "".join(str(d) for d in reversed(tgt_digits))
        ok = "OK" if pred_digits == tgt_digits else "WRONG"
        print(f"  {a_str} + {b_str} = {pred_str} (expected {tgt_str}) [{ok}]")


def show_examples_v10(model, device, n: int = 5):
    """Show sample predictions in vocab=10 format."""
    model.eval()
    with torch.no_grad():
        input_ids, _ = generate_batch_v10(n, MAX_DIGITS, device)
        logits = model(input_ids)
        preds = logits[:, INPUT_LEN_V10 - 1:-1, :].argmax(dim=-1)
        targets = input_ids[:, INPUT_LEN_V10:]

    for i in range(n):
        seq = input_ids[i].cpu().tolist()
        a_str = "".join(str(d) for d in reversed(seq[:MAX_DIGITS]))
        b_str = "".join(str(d) for d in reversed(seq[MAX_DIGITS + 1:MAX_DIGITS * 2 + 1]))
        pred_str = "".join(str(d) for d in reversed(preds[i].cpu().tolist()))
        tgt_str = "".join(str(d) for d in reversed(targets[i].cpu().tolist()))
        ok = "OK" if preds[i].cpu().tolist() == targets[i].cpu().tolist() else "WRONG"
        print(f"  {a_str} + {b_str} = {pred_str} (expected {tgt_str}) [{ok}]")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def get_curriculum_max_digits(step: int) -> int:
    """3-phase curriculum: 1-3 digits (0-2K), 1-6 (2K-7K), 1-10 (7K+)."""
    if step < 2000:
        return 3
    elif step < 7000:
        return 6
    else:
        return 10


def get_lr(step: int, warmup_steps: int, total_steps: int, base_lr: float,
           min_lr_ratio: float) -> float:
    """Linear warmup + cosine decay to min_lr."""
    min_lr = base_lr * min_lr_ratio
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(progress, 1.0)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def train(args):
    """Main training loop."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    vocab_size = 10 if args.vocab10 else 14

    model = CompetitorModel(
        vocab_size=vocab_size,
        d_model=args.d_model,
        d_ff=args.d_ff,
        max_seq_len=SEQ_LEN,
        pos_rank=args.pos_rank,
        qkv_rank=args.qkv_rank,
        attn_out_rank=args.attn_out_rank,
        ffn_rank=args.ffn_rank,
        share_layers=args.share_layers,
        repeats=args.repeats,
    ).to(device)

    n_params = count_parameters(model)

    # Build tag
    share_str = f"sh{args.repeats}" if args.share_layers else "1L"
    tag = (f"comp_d{args.d_model}_ff{args.d_ff}_v{vocab_size}"
           f"_pr{args.pos_rank}_qr{args.qkv_rank}_ar{args.attn_out_rank}_fr{args.ffn_rank}"
           f"_{share_str}")

    print(f"\n{'=' * 70}")
    print(f"COMPETITOR: d={args.d_model}, ff={args.d_ff}, vocab={vocab_size}, "
          f"pos_rank={args.pos_rank}, qkv_rank={args.qkv_rank}, "
          f"attn_out_rank={args.attn_out_rank}, ffn_rank={args.ffn_rank}")
    print(f"{'shared r=' + str(args.repeats) if args.share_layers else '1 layer'}, "
          f"params={n_params}")
    print(f"lr={args.lr}, steps={args.total_steps}, warmup={args.warmup_steps}, "
          f"seed={args.seed}")
    print(f"{'=' * 70}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01,
    )

    # LR scheduler
    if args.warm_restarts:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=args.t0, T_mult=1, eta_min=args.lr * args.min_lr_ratio)
        use_manual_lr = False
    else:
        scheduler = None
        use_manual_lr = True

    # Resume from checkpoint
    start_step = 0
    best_acc = 0.0
    ft_steps = ft_warmup = ft_min_lr_ratio = None  # set if fine-tuning
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        best_acc = ckpt.get("accuracy", 0.0)
        start_step = ckpt.get("step", 0)
        print(f"  Resumed from {args.resume}, acc={best_acc:.4f}, step={start_step}")
        if args.finetune_lr is not None:
            args.lr = args.finetune_lr
            for pg in optimizer.param_groups:
                pg["lr"] = args.finetune_lr
            # Fresh cosine schedule for fine-tuning (like competitor's finetune.py)
            # Steps counted from start_step, warmup=500, min_lr_ratio=0.01
            ft_steps = args.total_steps - start_step
            ft_warmup = 500
            ft_min_lr_ratio = 0.01
            print(f"  Fine-tune: lr={args.finetune_lr}, {ft_steps} steps, warmup={ft_warmup}, min_ratio={ft_min_lr_ratio}")

    # Wandb
    wandb_run = None
    if args.wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project="minimal-10digit-transformer",
                name=f"{tag}_{n_params}p_s{args.seed}",
                config=vars(args) | {"n_params": n_params, "tag": tag},
                tags=["competitor", "shareA_tieKV"],
                reinit=True,
            )
        except Exception:
            pass

    # Select data gen and eval functions
    if args.vocab10:
        gen_fn = lambda bs, max_d, dev: generate_batch_v10(bs, max_d, dev)
        eval_fn = evaluate_v10
        show_fn = show_examples_v10
    else:
        gen_fn = generate_batch_v14
        eval_fn = evaluate_v14
        show_fn = show_examples_v14

    t0 = time.time()
    running_loss = 0.0
    loss_count = 0

    for step in range(start_step, args.total_steps):
        model.train()

        # Set LR
        if use_manual_lr:
            if args.finetune_lr is not None:
                # Fresh cosine schedule for fine-tuning
                ft_step = step - start_step
                lr = get_lr(ft_step, ft_warmup, ft_steps, args.finetune_lr, ft_min_lr_ratio)
            else:
                lr = get_lr(step, args.warmup_steps, args.total_steps, args.lr,
                            args.min_lr_ratio)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

        # Curriculum: determine max digits for this step
        max_d = get_curriculum_max_digits(step)

        # Generate batch
        if args.vocab10:
            input_ids, labels = generate_batch_v10(args.batch_size, max_d, device)
        else:
            input_ids, labels = generate_batch_v14(args.batch_size, max_d, device)

        # Forward
        logits = model(input_ids)

        # Loss: labels[t] = what logits[t] should predict (the next token).
        # labels[21] = c_t[0] = input_ids[22], labels[22] = c_t[1], etc.
        # No additional shift needed — labels are already in next-token format.
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=-100,
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item()
        loss_count += 1

        # Logging
        if step % 100 == 0 and step > 0:
            avg_loss = running_loss / loss_count
            cur_lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t0
            print(f"  step {step:6d} | loss {avg_loss:.4f} | lr {cur_lr:.2e} | "
                  f"digits {max_d:2d} | {elapsed:.0f}s")
            if wandb_run:
                import wandb
                wandb.log({"loss": avg_loss, "lr": cur_lr, "max_digits": max_d}, step=step)
            running_loss = 0.0
            loss_count = 0

        # Evaluation
        if step % args.eval_interval == 0 and step > 0:
            results = eval_fn(model, 10000, device)
            acc = results["exact_match"]
            print(f"  EVAL step {step}: exact={acc:.4f} digit_avg={results['digit_avg']:.4f}")
            print(f"    per-digit: {['%.3f' % d for d in results['per_digit']]}")
            show_fn(model, device)

            if acc > best_acc:
                best_acc = acc
                print(f"  ** BEST: {acc:.4f} **")
                # Save checkpoint
                ckpt_dir = Path(f"checkpoints/{tag}_s{args.seed}")
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "state_dict": model.state_dict(),
                    "n_params": n_params,
                    "accuracy": acc,
                    "step": step,
                    "digit_avg": results["digit_avg"],
                    "args": vars(args),
                }, str(ckpt_dir / "best.pt"))

            if wandb_run:
                import wandb
                log_dict = {
                    "eval_acc": acc, "best_acc": best_acc,
                    "digit_avg": results["digit_avg"],
                }
                for i, d_acc in enumerate(results["per_digit"]):
                    log_dict[f"digit_{i}_acc"] = d_acc
                wandb.log(log_dict, step=step)

    # Final evaluation
    final = eval_fn(model, 50000, device, seed=99999)
    print(f"\n  FINAL: exact={final['exact_match']:.4f} digit_avg={final['digit_avg']:.4f}")
    print(f"    per-digit: {['%.3f' % d for d in final['per_digit']]}")
    show_fn(model, device, n=10)

    if wandb_run:
        import wandb
        wandb.finish()

    return {
        "tag": tag, "n_params": n_params, "seed": args.seed,
        "best_acc": best_acc,
        "final_acc": final["exact_match"],
        "final_per_digit": final["per_digit"],
    }


def main():
    parser = argparse.ArgumentParser(description="Competitor architecture training")
    parser.add_argument("--d-model", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=8)
    parser.add_argument("--pos-rank", type=int, default=3)
    parser.add_argument("--qkv-rank", type=int, default=3)
    parser.add_argument("--attn-out-rank", type=int, default=3)
    parser.add_argument("--ffn-rank", type=int, default=3)
    parser.add_argument("--total-steps", type=int, default=162000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--warmup-steps", type=int, default=1350)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=34)
    parser.add_argument("--share-layers", action="store_true")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--vocab10", action="store_true",
                        help="Use our format: vocab=10, all-reversed, +/= mapped to 0")
    parser.add_argument("--warm-restarts", action="store_true",
                        help="Use CosineAnnealingWarmRestarts instead of linear warmup + cosine decay")
    parser.add_argument("--t0", type=int, default=2000, help="T_0 for warm restarts")
    parser.add_argument("--resume", type=str, default="",
                        help="Path to checkpoint for resuming/fine-tuning")
    parser.add_argument("--finetune-lr", type=float, default=None,
                        help="Override LR for fine-tuning")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--eval-interval", type=int, default=1000)
    args = parser.parse_args()

    result = train(args)

    out_file = f"experiments/{result['tag']}_s{args.seed}_results.json"
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved results to {out_file}")


if __name__ == "__main__":
    main()
