"""Mechanistic interpretability visualizations for minimal addition transformers.

Generates 7 analysis plots for understanding how 62-122 param transformers solve
10-digit addition. Inspired by Nanda et al. (2023) grokking analysis and
Quirke & Barez (2024) cascading carry circuit analysis.

Usage:
  uv run experiments/interpretability.py                    # default: 89p model
  uv run experiments/interpretability.py --checkpoint checkpoints/qwen3_d3_ff3_122p_s6/best.pt
  uv run experiments/interpretability.py --all-models       # compare multiple models
  uv run experiments/interpretability.py --analysis 1 3 5   # specific analyses only

Analyses:
  1. Embedding geometry (3D scatter of digit embeddings)
  2. Complete weight atlas (heatmap of every weight matrix)
  3. Attention heatmap (average attention pattern)
  4. Logit lens (residual stream projections at each stage)
  5. Carry vs no-carry attention comparison
  6. Training dynamics (weight trajectories during training)
  7. Residual stream trajectories (3D paths through the network)
"""

import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# Add project root for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from minimal10digittransformer.model.qwen3 import (
    Qwen3AdditionModel, RMSNorm, apply_rope, VOCAB_SIZE, INPUT_LEN, OUTPUT_LEN, TOTAL_LEN,
)
from minimal10digittransformer.model.circular_arc import CircularArcQwen3
from minimal10digittransformer.model.rank1_out import Rank1OutModel
from minimal10digittransformer.data.addition import encode, expected_output

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch
    from mpl_toolkits.mplot3d import Axes3D, proj3d
    import matplotlib.gridspec as gridspec
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("ERROR: matplotlib required. Install: pip install matplotlib")
    sys.exit(1)

import numpy as np


# ── Helpers ────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, device="cpu"):
    """Load model from checkpoint, auto-detecting architecture."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    cfg = ckpt["config"]

    # Detect architecture from state dict keys
    state = ckpt.get("model", ckpt.get("state_dict", {}))
    is_arc = cfg.get("circular_arc", False) or "arc_A" in state
    is_rank1 = "block.attn.out_proj_A" in state

    common_kwargs = dict(
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_kv_heads=cfg["n_kv_heads"],
        head_dim=cfg["head_dim"],
        ff=cfg["ff"],
        rope_theta=cfg["rope_theta"],
        qk_norm=not cfg.get("no_qk_norm", False),
        use_swiglu=not cfg.get("gelu", False),
        tie_kv=cfg.get("tie_kv", False),
        tie_gate=cfg.get("tie_gate", False),
        share_norms=cfg.get("share_norms", False),
        share_block_norms=cfg.get("share_block_norms", False),
    )

    if is_arc:
        model = CircularArcQwen3(tie_qo=cfg.get("tie_qo", False), **common_kwargs)
        arch_name = "CircularArc"
    elif is_rank1:
        model = Rank1OutModel(**common_kwargs)
        arch_name = "Rank1Out"
    else:
        model = Qwen3AdditionModel(tie_qo=cfg.get("tie_qo", False), **common_kwargs)
        arch_name = "Qwen3"

    model.load_state_dict(state)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded {n_params}p model from {checkpoint_path}")
    print(f"  Architecture: {arch_name}, "
          f"d={cfg['d_model']}, ff={cfg['ff']}, "
          f"tieKV={cfg.get('tie_kv',False)}, tieQO={cfg.get('tie_qo',False)}")
    return model, cfg, is_arc


def get_embedding_table(model, is_arc):
    """Get the 10xd_model embedding table."""
    if is_arc:
        return model._compute_embedding_table().detach()
    else:
        return model.embed.weight.detach()


def count_carries(a: int, b: int) -> list[bool]:
    """Return per-position carry flags (11 positions, LSB first)."""
    carry = 0
    carries = []
    for _ in range(11):
        d_a = a % 10
        d_b = b % 10
        total = d_a + d_b + carry
        carries.append(total >= 10)
        carry = 1 if total >= 10 else 0
        a //= 10
        b //= 10
    return carries


def position_labels():
    """Return human-readable labels for each of the 35 token positions."""
    labels = []
    # Input: [0] rev(a, 10) [0,0] rev(b, 10) [0]
    labels.append("pad")
    for i in range(10):
        labels.append(f"a{i}")
    labels.append("sep")
    labels.append("sep")
    for i in range(10):
        labels.append(f"b{i}")
    labels.append("pad")
    # Output: 11 sum digits
    for i in range(11):
        labels.append(f"s{i}")
    return labels


def run_with_intermediates(model, input_ids, is_arc):
    """Run model and capture intermediate residual states.

    Returns dict with keys:
      'after_embed': (B, T, d)
      'after_attn':  (B, T, d)  -- residual after attention
      'after_mlp':   (B, T, d)  -- residual after MLP
      'after_norm':  (B, T, d)  -- after final norm
      'attn_weights': (B, n_heads, T, T) -- attention probabilities
    """
    B, T = input_ids.shape
    block = model.block

    # Embedding
    if is_arc:
        emb_table = model._compute_embedding_table()
        x = emb_table[input_ids]
    else:
        x = model.embed(input_ids)

    states = {"after_embed": x.detach().clone()}

    # Attention (manually to capture weights)
    residual = x
    x_normed = block.ln1(x)
    attn_module = block.attn

    q = attn_module.q_proj(x_normed).view(B, T, attn_module.n_heads, attn_module.head_dim).transpose(1, 2)
    k = attn_module.k_proj(x_normed).view(B, T, attn_module.n_kv_heads, attn_module.head_dim).transpose(1, 2)
    v_proj = attn_module.k_proj if attn_module.tie_kv else attn_module.v_proj
    v = v_proj(x_normed).view(B, T, attn_module.n_kv_heads, attn_module.head_dim).transpose(1, 2)

    if attn_module.use_qk_norm:
        q = attn_module.q_norm(q)
        k = attn_module.k_norm(k)

    q = apply_rope(q, attn_module.rope_cos, attn_module.rope_sin)
    k = apply_rope(k, attn_module.rope_cos, attn_module.rope_sin)

    if attn_module.n_rep > 1:
        k = k.repeat_interleave(attn_module.n_rep, dim=1)
        v = v.repeat_interleave(attn_module.n_rep, dim=1)

    scale = 1.0 / math.sqrt(attn_module.head_dim)
    attn_logits = (q @ k.transpose(-2, -1)) * scale
    mask = model.causal_mask[:T, :T] if hasattr(model, 'causal_mask') else block.attn.causal_mask[:T, :T]
    attn_logits = attn_logits + mask
    attn_weights = F.softmax(attn_logits, dim=-1)

    states["attn_weights"] = attn_weights.detach().clone()

    attn_out = (attn_weights @ v).transpose(1, 2).contiguous().view(B, T, -1)
    if attn_module.tie_qo:
        attn_out = F.linear(attn_out, attn_module.q_proj.weight.t())
    else:
        attn_out = attn_module.o_proj(attn_out)

    x = residual + attn_out
    states["after_attn"] = x.detach().clone()

    # MLP
    residual = x
    x_normed = block.ln2(x)
    mlp_out = block.mlp(x_normed)
    x = residual + mlp_out
    states["after_mlp"] = x.detach().clone()

    # Final norm
    x = model.final_norm(x)
    states["after_norm"] = x.detach().clone()

    return states


# ── Analysis 1: Embedding Geometry ─────────────────────────────────────────

def plot_embedding_geometry(model, is_arc, output_dir, tag=""):
    """Plot 3D scatter of digit embeddings (10 points in R3)."""
    emb = get_embedding_table(model, is_arc).numpy()
    d_model = emb.shape[1]

    fig = plt.figure(figsize=(12, 5))

    if d_model >= 3:
        # 3D scatter
        ax = fig.add_subplot(121, projection="3d")
        colors = plt.cm.tab10(np.arange(10))
        for i in range(10):
            ax.scatter(emb[i, 0], emb[i, 1], emb[i, 2],
                       c=[colors[i]], s=200, zorder=5)
            ax.text(emb[i, 0], emb[i, 1], emb[i, 2], f" {i}",
                    fontsize=12, fontweight="bold")

        # Connect consecutive digits with lines
        for i in range(9):
            ax.plot([emb[i, 0], emb[i+1, 0]],
                    [emb[i, 1], emb[i+1, 1]],
                    [emb[i, 2], emb[i+1, 2]],
                    c="gray", alpha=0.4, linewidth=1)

        ax.set_xlabel("dim 0")
        ax.set_ylabel("dim 1")
        ax.set_zlabel("dim 2")
        ax.set_title("Digit Embeddings in 3D")

        # 2D projection (dims 0 vs 1)
        ax2 = fig.add_subplot(122)
    else:
        ax2 = fig.add_subplot(111)

    colors = plt.cm.tab10(np.arange(10))
    for i in range(10):
        ax2.scatter(emb[i, 0], emb[i, 1], c=[colors[i]], s=200, zorder=5)
        ax2.annotate(f"{i}", (emb[i, 0], emb[i, 1]),
                     fontsize=14, fontweight="bold",
                     textcoords="offset points", xytext=(8, 4))

    for i in range(9):
        ax2.plot([emb[i, 0], emb[i+1, 0]],
                 [emb[i, 1], emb[i+1, 1]],
                 c="gray", alpha=0.4, linewidth=1)

    # Check for circular structure: fit circle
    if d_model >= 2:
        cx, cy = emb[:, 0].mean(), emb[:, 1].mean()
        radii = np.sqrt((emb[:, 0] - cx)**2 + (emb[:, 1] - cy)**2)
        r_mean = radii.mean()
        r_std = radii.std()
        circularity = 1.0 - r_std / r_mean if r_mean > 0 else 0

        # Draw best-fit circle
        theta = np.linspace(0, 2*np.pi, 100)
        ax2.plot(cx + r_mean * np.cos(theta), cy + r_mean * np.sin(theta),
                 "k--", alpha=0.2, linewidth=1)
        ax2.set_title(f"Embedding dims 0-1 (circularity={circularity:.3f})")

    ax2.set_xlabel("dim 0")
    ax2.set_ylabel("dim 1")
    ax2.set_aspect("equal")
    ax2.grid(True, alpha=0.3)

    # Print embedding values
    print("\n  Embedding table:")
    for i in range(10):
        vals = ", ".join(f"{v:+.4f}" for v in emb[i])
        print(f"    digit {i}: [{vals}]")

    if is_arc:
        A = model.arc_A.item()
        start = model.arc_start.item()
        stride = model.arc_stride.item()
        print(f"  Arc params: A={A:.4f}, start={start:.4f}, stride={stride:.4f}")

    plt.suptitle(f"Analysis 1: Embedding Geometry {tag}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = f"{output_dir}/1_embedding_geometry.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


# ── Analysis 2: Complete Weight Atlas ──────────────────────────────────────

def plot_weight_atlas(model, is_arc, output_dir, tag=""):
    """Heatmap of every weight matrix in the model."""
    # Collect all named parameters
    params = [(name, p.detach().numpy()) for name, p in model.named_parameters()]
    n = len(params)
    ncols = 4
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = axes.flatten() if n > ncols else ([axes] if n == 1 else list(axes))

    for idx, (name, p) in enumerate(params):
        ax = axes[idx]
        data = p.squeeze()

        if data.ndim == 1:
            # 1D param: show as horizontal bar
            im = ax.imshow(data.reshape(1, -1), cmap="RdBu_r", aspect="auto",
                           vmin=-abs(data).max(), vmax=abs(data).max())
            ax.set_yticks([])
            for j, v in enumerate(data):
                ax.text(j, 0, f"{v:.3f}", ha="center", va="center", fontsize=7)
        elif data.ndim == 2:
            vmax = abs(data).max()
            im = ax.imshow(data, cmap="RdBu_r", aspect="auto",
                           vmin=-vmax, vmax=vmax)
            # Annotate values if small enough
            if data.size <= 30:
                for i in range(data.shape[0]):
                    for j in range(data.shape[1]):
                        ax.text(j, i, f"{data[i,j]:.2f}", ha="center", va="center",
                                fontsize=6)
        else:
            ax.text(0.5, 0.5, f"shape={data.shape}", transform=ax.transAxes,
                    ha="center", va="center")

        ax.set_title(f"{name}\n{list(p.shape)}", fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(f"Analysis 2: Complete Weight Atlas {tag}\n"
                 f"({sum(p.size for _, p in params)} total params)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = f"{output_dir}/2_weight_atlas.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


# ── Analysis 3: Attention Heatmap ──────────────────────────────────────────

def plot_attention_heatmap(model, is_arc, output_dir, tag="", n_examples=50):
    """Average attention pattern over many addition problems."""
    import random
    rng = random.Random(42)
    device = next(model.parameters()).device

    attn_sum = None
    pos_labels = position_labels()

    for _ in range(n_examples):
        a = rng.randint(0, 10**10 - 1)
        b = rng.randint(0, 10**10 - 1)
        inp = encode(a, b)
        exp = expected_output(a, b)
        full_seq = inp + exp
        input_ids = torch.tensor([full_seq], dtype=torch.long, device=device)

        states = run_with_intermediates(model, input_ids, is_arc)
        attn = states["attn_weights"][0, 0].numpy()  # head 0

        if attn_sum is None:
            attn_sum = attn
        else:
            attn_sum += attn

    attn_avg = attn_sum / n_examples

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Full heatmap
    ax = axes[0]
    im = ax.imshow(attn_avg, cmap="Blues", aspect="auto")
    ax.set_xticks(range(35))
    ax.set_yticks(range(35))
    ax.set_xticklabels(pos_labels, rotation=90, fontsize=7)
    ax.set_yticklabels(pos_labels, fontsize=7)
    ax.set_xlabel("Key (attending to)")
    ax.set_ylabel("Query (position)")
    ax.set_title("Full Attention Pattern (avg)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Draw structural boxes
    for (y0, y1, x0, x1, label) in [
        (24, 34, 1, 10, "sum→a"),
        (24, 34, 13, 22, "sum→b"),
        (24, 34, 24, 34, "sum→sum"),
    ]:
        rect = plt.Rectangle((x0-0.5, y0-0.5), x1-x0+1, y1-y0+1,
                              fill=False, edgecolor="red", linewidth=1.5, linestyle="--")
        ax.add_patch(rect)
        ax.text(x0, y0-1, label, fontsize=8, color="red", fontweight="bold")

    # Zoomed: output positions only
    ax2 = axes[1]
    output_attn = attn_avg[24:35, :]
    im2 = ax2.imshow(output_attn, cmap="Blues", aspect="auto")
    ax2.set_xticks(range(35))
    ax2.set_xticklabels(pos_labels, rotation=90, fontsize=7)
    ax2.set_yticks(range(11))
    ax2.set_yticklabels([f"s{i}" for i in range(11)], fontsize=8)
    ax2.set_xlabel("Key (attending to)")
    ax2.set_ylabel("Output position")
    ax2.set_title("Output Positions Attention (zoomed)")
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    plt.suptitle(f"Analysis 3: Attention Heatmap {tag} (n={n_examples})",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = f"{output_dir}/3_attention_heatmap.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


# ── Analysis 4: Logit Lens ────────────────────────────────────────────────

def plot_logit_lens(model, is_arc, output_dir, tag=""):
    """Project residual stream through tied embeddings at each stage."""
    import random
    rng = random.Random(42)
    device = next(model.parameters()).device

    # Pick a single representative example
    a, b = 1234567890, 9876543210
    inp = encode(a, b)
    exp = expected_output(a, b)
    full_seq = inp + exp
    input_ids = torch.tensor([full_seq], dtype=torch.long, device=device)

    states = run_with_intermediates(model, input_ids, is_arc)
    emb_table = get_embedding_table(model, is_arc)  # (10, d)

    stage_names = ["After Embed", "After Attn", "After MLP", "After Norm"]
    stage_keys = ["after_embed", "after_attn", "after_mlp", "after_norm"]

    fig, axes = plt.subplots(len(stage_keys), 1, figsize=(18, 3.5 * len(stage_keys)))

    for idx, (key, name) in enumerate(zip(stage_keys, stage_names)):
        ax = axes[idx]
        residual = states[key][0]  # (T, d)
        # Project through embedding to get logits
        logits = residual @ emb_table.T  # (T, 10)
        probs = F.softmax(logits, dim=-1).numpy()

        im = ax.imshow(probs.T, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
        ax.set_yticks(range(10))
        ax.set_yticklabels([str(i) for i in range(10)])
        ax.set_ylabel("Digit")

        pos_labels = position_labels()
        ax.set_xticks(range(35))
        ax.set_xticklabels(pos_labels, rotation=90, fontsize=7)

        # Mark correct output digits
        for i in range(11):
            pos = 24 + i
            correct = exp[i]
            ax.plot(pos, correct, "ws", markersize=8, markeredgecolor="black",
                    markeredgewidth=2)

        ax.set_title(f"{name}: P(digit) at each position", fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.02, pad=0.01)

        # Vertical line separating input from output
        ax.axvline(x=23.5, color="white", linewidth=2, linestyle="--")

    plt.suptitle(f"Analysis 4: Logit Lens {tag}\n"
                 f"({a} + {b} = {a+b})",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = f"{output_dir}/4_logit_lens.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


# ── Analysis 5: Carry vs No-Carry Attention ────────────────────────────────

def plot_carry_attention(model, is_arc, output_dir, tag="", n_examples=200):
    """Compare attention patterns for carry vs no-carry digit positions."""
    import random
    rng = random.Random(42)
    device = next(model.parameters()).device

    # Collect attention patterns for carry and no-carry output positions
    carry_attn = []  # attention rows for positions where carry occurs
    nocarry_attn = []

    for _ in range(n_examples):
        a = rng.randint(0, 10**10 - 1)
        b = rng.randint(0, 10**10 - 1)
        inp = encode(a, b)
        exp = expected_output(a, b)
        full_seq = inp + exp
        input_ids = torch.tensor([full_seq], dtype=torch.long, device=device)

        states = run_with_intermediates(model, input_ids, is_arc)
        attn = states["attn_weights"][0, 0].numpy()  # head 0

        carries = count_carries(a, b)

        # For each output position (24-34 = s0-s10)
        for i in range(11):
            row = attn[24 + i, :]  # attention from output position i
            if carries[i]:
                carry_attn.append(row)
            else:
                nocarry_attn.append(row)

    carry_avg = np.mean(carry_attn, axis=0) if carry_attn else np.zeros(35)
    nocarry_avg = np.mean(nocarry_attn, axis=0) if nocarry_attn else np.zeros(35)

    fig, axes = plt.subplots(3, 1, figsize=(16, 10))
    pos_labels = position_labels()

    for ax, data, title, color in [
        (axes[0], nocarry_avg, f"No-Carry Positions (n={len(nocarry_attn)})", "Blues"),
        (axes[1], carry_avg, f"Carry Positions (n={len(carry_attn)})", "Reds"),
    ]:
        ax.bar(range(35), data, color=plt.colormaps.get_cmap(color)(0.6))
        ax.set_xticks(range(35))
        ax.set_xticklabels(pos_labels, rotation=90, fontsize=7)
        ax.set_ylabel("Attention Weight")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    # Difference: carry - no-carry
    diff = carry_avg - nocarry_avg
    colors = ["#d32f2f" if d > 0 else "#1976d2" for d in diff]
    axes[2].bar(range(35), diff, color=colors)
    axes[2].set_xticks(range(35))
    axes[2].set_xticklabels(pos_labels, rotation=90, fontsize=7)
    axes[2].set_ylabel("Attention Difference")
    axes[2].set_title("Carry − No-Carry (red=carry attends more, blue=less)")
    axes[2].grid(True, alpha=0.3)
    axes[2].axhline(0, color="black", linewidth=0.5)

    plt.suptitle(f"Analysis 5: Carry vs No-Carry Attention {tag}",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = f"{output_dir}/5_carry_attention.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")

    # Also: per-output-position breakdown
    fig2, axes2 = plt.subplots(3, 4, figsize=(20, 12))
    axes2 = axes2.flatten()
    for pos_i in range(11):
        ax = axes2[pos_i]
        carry_rows = []
        nocarry_rows = []

        rng2 = random.Random(42)
        for _ in range(n_examples):
            a = rng2.randint(0, 10**10 - 1)
            b = rng2.randint(0, 10**10 - 1)
            inp = encode(a, b)
            exp = expected_output(a, b)
            full_seq = inp + exp
            input_ids = torch.tensor([full_seq], dtype=torch.long, device=device)

            states = run_with_intermediates(model, input_ids, is_arc)
            attn = states["attn_weights"][0, 0].numpy()
            carries = count_carries(a, b)

            if carries[pos_i]:
                carry_rows.append(attn[24 + pos_i, :])
            else:
                nocarry_rows.append(attn[24 + pos_i, :])

        if carry_rows:
            ax.plot(np.mean(carry_rows, axis=0), "r-", alpha=0.7, label=f"carry (n={len(carry_rows)})")
        if nocarry_rows:
            ax.plot(np.mean(nocarry_rows, axis=0), "b-", alpha=0.7, label=f"no-carry (n={len(nocarry_rows)})")

        ax.set_title(f"s{pos_i} (output digit {pos_i})", fontsize=9)
        ax.legend(fontsize=7)
        ax.set_xlim(0, 34)
        ax.grid(True, alpha=0.2)

    axes2[11].set_visible(False)
    plt.suptitle(f"Analysis 5b: Per-Position Carry vs No-Carry {tag}",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path2 = f"{output_dir}/5b_carry_per_position.png"
    plt.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path2}")


# ── Analysis 6: Training Dynamics ──────────────────────────────────────────

def plot_training_dynamics(checkpoint_dir, output_dir, tag=""):
    """Plot weight values over training by loading multiple checkpoints.

    Looks for best.pt and final.pt, or numbered checkpoints.
    """
    import csv

    ckpt_dir = Path(checkpoint_dir)

    # Find all checkpoint files with step numbers
    ckpt_files = sorted(ckpt_dir.glob("step_*.pt"))
    if not ckpt_files:
        # Try metrics.csv for loss/accuracy dynamics at least
        csv_path = ckpt_dir / "metrics.csv"
        if not csv_path.exists():
            print(f"  No step_*.pt checkpoints or metrics.csv found in {ckpt_dir}")
            print("  Skipping analysis 6 (training dynamics)")
            return

    # If we have checkpoints, load weight trajectories
    if ckpt_files:
        trajectories = {}  # param_name -> list of (step, values)

        for cp in ckpt_files:
            step = int(cp.stem.split("_")[1])
            ckpt = torch.load(str(cp), map_location="cpu", weights_only=True)
            state = ckpt.get("model", ckpt)
            for name, tensor in state.items():
                if name not in trajectories:
                    trajectories[name] = []
                trajectories[name].append((step, tensor.numpy().flatten()))

        n = len(trajectories)
        ncols = 3
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
        axes = axes.flatten()

        for idx, (name, traj) in enumerate(trajectories.items()):
            ax = axes[idx]
            traj.sort(key=lambda x: x[0])
            steps = [t[0] for t in traj]
            n_weights = len(traj[0][1])

            for w_idx in range(n_weights):
                values = [t[1][w_idx] for t in traj]
                ax.plot(steps, values, alpha=0.6, linewidth=0.8)

            ax.set_title(f"{name} ({n_weights} weights)", fontsize=9)
            ax.grid(True, alpha=0.2)
            ax.set_xlabel("Step", fontsize=8)

        for idx in range(n, len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle(f"Analysis 6: Weight Trajectories {tag}",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        path = f"{output_dir}/6_training_dynamics.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved {path}")
    else:
        # Compare best vs final weights if both exist
        best_path = ckpt_dir / "best.pt"
        final_path = ckpt_dir / "final.pt"
        if best_path.exists() and final_path.exists():
            best_ckpt = torch.load(str(best_path), map_location="cpu", weights_only=True)
            final_ckpt = torch.load(str(final_path), map_location="cpu", weights_only=True)
            best = best_ckpt.get("model", best_ckpt.get("state_dict", best_ckpt))
            final = final_ckpt.get("model", final_ckpt.get("state_dict", final_ckpt))

            diffs = {}
            for name in best:
                diff = (final[name] - best[name]).abs().numpy()
                diffs[name] = diff

            n = len(diffs)
            ncols = 4
            nrows = (n + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
            axes = axes.flatten()

            for idx, (name, diff) in enumerate(diffs.items()):
                ax = axes[idx]
                data = diff.squeeze()
                if data.ndim <= 1:
                    ax.bar(range(data.size), data.flatten())
                else:
                    im = ax.imshow(data, cmap="hot", aspect="auto")
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                ax.set_title(f"|final - best|\n{name}", fontsize=8)

            for idx in range(n, len(axes)):
                axes[idx].set_visible(False)

            plt.suptitle(f"Analysis 6: Weight Change (best→final) {tag}",
                         fontsize=13, fontweight="bold")
            plt.tight_layout()
            path = f"{output_dir}/6_weight_change.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  Saved {path}")
        else:
            print("  Only one checkpoint found — skipping training dynamics")


# ── Analysis 7: Residual Stream Trajectories ───────────────────────────────

def plot_residual_trajectories(model, is_arc, output_dir, tag=""):
    """3D plot of how token representations evolve through the network."""
    device = next(model.parameters()).device
    d_model = model.d_model

    if d_model < 3:
        print("  d_model < 3, skipping 3D residual trajectory plot")
        return

    # Example problems: one with many carries, one without
    examples = [
        (9999999999, 1, "9999999999 + 1 (max carry)"),
        (1234567890, 1111111111, "1234567890 + 1111111111 (no carry)"),
        (5555555555, 4444444445, "5555555555 + 4444444445 (all carry)"),
    ]

    fig = plt.figure(figsize=(18, 6))
    stage_names = ["Embed", "Attn", "MLP", "Norm"]
    stage_keys = ["after_embed", "after_attn", "after_mlp", "after_norm"]
    stage_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for ex_idx, (a, b, desc) in enumerate(examples):
        ax = fig.add_subplot(1, 3, ex_idx + 1, projection="3d")

        inp = encode(a, b)
        exp = expected_output(a, b)
        full_seq = inp + exp
        input_ids = torch.tensor([full_seq], dtype=torch.long, device=device)

        states = run_with_intermediates(model, input_ids, is_arc)

        # Plot trajectories for output positions (s0-s10)
        for pos in range(24, 35):
            points = []
            for key in stage_keys:
                v = states[key][0, pos].numpy()
                points.append(v[:3])
            points = np.array(points)

            digit_idx = pos - 24
            color = plt.cm.tab10(digit_idx / 11)

            ax.plot(points[:, 0], points[:, 1], points[:, 2],
                    color=color, alpha=0.7, linewidth=1.5)
            # Mark stages
            for s_idx, (px, py, pz) in enumerate(points):
                marker = ["o", "s", "^", "D"][s_idx]
                ax.scatter(px, py, pz, c=[stage_colors[s_idx]],
                           marker=marker, s=30, zorder=5)

            # Label final position
            ax.text(points[-1, 0], points[-1, 1], points[-1, 2],
                    f" s{digit_idx}", fontsize=7)

        ax.set_xlabel("dim 0", fontsize=8)
        ax.set_ylabel("dim 1", fontsize=8)
        ax.set_zlabel("dim 2", fontsize=8)
        ax.set_title(f"{desc}\n(sum={a+b})", fontsize=9)

    # Legend for stages
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=stage_colors[0],
               markersize=8, label="After Embed"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=stage_colors[1],
               markersize=8, label="After Attn"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor=stage_colors[2],
               markersize=8, label="After MLP"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor=stage_colors[3],
               markersize=8, label="After Norm"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=4, fontsize=9)

    plt.suptitle(f"Analysis 7: Residual Stream Trajectories {tag}",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0.06, 1, 0.95])
    path = f"{output_dir}/7_residual_trajectories.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


# ── Multi-model comparison ─────────────────────────────────────────────────

def plot_embedding_comparison(output_dir):
    """Compare embedding geometry across multiple models."""
    ckpt_base = Path("checkpoints")
    models_to_compare = [
        ("89p", "qwen3_d3_ff2_89p_tiekv_tieqo_s11127/best.pt"),
        ("122p", "qwen3_d3_ff3_122p_s6/best.pt"),
        ("83p shnorm", "qwen3_d3_ff2_83p_tiekv_tieqo_shnorm_s905/best.pt"),
        ("62p arc", "qwen3_arc_62p_tiekv_tieqo_adam_nowd/best.pt"),
        ("95p arc", "qwen3_arc_95p_ft_s9999/best.pt"),
    ]

    fig = plt.figure(figsize=(20, 8))
    loaded = 0

    for idx, (label, rel_path) in enumerate(models_to_compare):
        path = ckpt_base / rel_path
        if not path.exists():
            continue

        try:
            model, cfg, is_arc = load_model(str(path))
        except Exception as e:
            print(f"  Failed to load {label}: {e}")
            continue

        loaded += 1
        emb = get_embedding_table(model, is_arc).numpy()
        d = emb.shape[1]

        if d >= 3:
            ax = fig.add_subplot(2, 3, loaded, projection="3d")
            colors = plt.cm.tab10(np.arange(10))
            for i in range(10):
                ax.scatter(emb[i, 0], emb[i, 1], emb[i, 2],
                           c=[colors[i]], s=150, zorder=5)
                ax.text(emb[i, 0], emb[i, 1], emb[i, 2], f" {i}", fontsize=10)
            for i in range(9):
                ax.plot([emb[i, 0], emb[i+1, 0]],
                        [emb[i, 1], emb[i+1, 1]],
                        [emb[i, 2], emb[i+1, 2]],
                        c="gray", alpha=0.4)
            ax.set_title(label, fontsize=11, fontweight="bold")
            ax.set_xlabel("d0", fontsize=8)
            ax.set_ylabel("d1", fontsize=8)
            ax.set_zlabel("d2", fontsize=8)

    if loaded == 0:
        print("  No models found for comparison")
        plt.close()
        return

    plt.suptitle("Embedding Geometry Comparison Across Models",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = f"{output_dir}/comparison_embeddings.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


# ── Main ───────────────────────────────────────────────────────────────────

DEFAULT_CHECKPOINT = "checkpoints/qwen3_d3_ff2_89p_tiekv_tieqo_s11127/best.pt"

def main():
    parser = argparse.ArgumentParser(description="Interpretability analysis for addition transformers")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                        help="Path to model checkpoint")
    parser.add_argument("--output-dir", default="plots/interpretability",
                        help="Output directory for plots")
    parser.add_argument("--analysis", nargs="+", type=int, default=list(range(1, 8)),
                        help="Which analyses to run (1-7)")
    parser.add_argument("--all-models", action="store_true",
                        help="Run embedding comparison across all models")
    parser.add_argument("--n-examples", type=int, default=50,
                        help="Number of examples for attention averaging")
    args = parser.parse_args()

    output_dir = args.output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"ERROR: Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    model, cfg, is_arc = load_model(str(ckpt_path))
    n_params = sum(p.numel() for p in model.parameters())
    tag = f"({n_params}p)"

    analyses = set(args.analysis)

    if 1 in analyses:
        print("\n── Analysis 1: Embedding Geometry ──")
        plot_embedding_geometry(model, is_arc, output_dir, tag)

    if 2 in analyses:
        print("\n── Analysis 2: Weight Atlas ──")
        plot_weight_atlas(model, is_arc, output_dir, tag)

    if 3 in analyses:
        print("\n── Analysis 3: Attention Heatmap ──")
        plot_attention_heatmap(model, is_arc, output_dir, tag, n_examples=args.n_examples)

    if 4 in analyses:
        print("\n── Analysis 4: Logit Lens ──")
        plot_logit_lens(model, is_arc, output_dir, tag)

    if 5 in analyses:
        print("\n── Analysis 5: Carry vs No-Carry Attention ──")
        plot_carry_attention(model, is_arc, output_dir, tag, n_examples=args.n_examples)

    if 6 in analyses:
        print("\n── Analysis 6: Training Dynamics ──")
        ckpt_dir = ckpt_path.parent
        plot_training_dynamics(str(ckpt_dir), output_dir, tag)

    if 7 in analyses:
        print("\n── Analysis 7: Residual Stream Trajectories ──")
        plot_residual_trajectories(model, is_arc, output_dir, tag)

    if args.all_models:
        print("\n── Embedding Comparison (all models) ──")
        plot_embedding_comparison(output_dir)

    print(f"\nDone! All plots saved to {output_dir}/")


if __name__ == "__main__":
    main()
