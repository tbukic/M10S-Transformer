"""Matrix tying experiments on 70-80p configs.

Tests novel weight-tying strategies to reduce parameter counts while
preserving grokking ability. Uses known-good base configs (74p, 80p).

Tying strategies tested:
  MLP tying:
    gate_alpha:  gate = α·up (scalar, saves 3*ff - 1)
    gate_eq_up:  gate = up (saves 3*ff)
    down_eq_upT: down = up^T (saves 3*ff)
    down_neg_upT: down = -up^T (saves 3*ff)
    gate_alpha+down_upT: both combined (saves 6*ff - 1)
    gate_eq_up+down_upT: both combined (saves 6*ff)

  Embedding:
    quadratic: e(d) = [c0 - c1*d², -d, c2*d] (3 params, compare vs arc)
    frozen_random: random init, frozen (0 params)

  RoPE:
    learnable_theta: make theta a learnable param (+1 param)
    theta_sweep: test theta ∈ {1, 2, 3, 5, 10, 19}

Usage:
    python experiments/tying_search.py --device cpu --max-steps 100000
    python experiments/tying_search.py --max-steps 500 --seeds 1  # smoke test
"""

import argparse
import csv
import math
import os
import random
import sys
import time
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import torch.nn as nn
import torch.nn.functional as F

from minimal10digittransformer.model.circular_arc import CircularArcQwen3
from minimal10digittransformer.model.qwen3 import (
    Qwen3Block, Qwen3MLP, Qwen3Attention, RMSNorm,
    precompute_rope_freqs, create_causal_mask,
    VOCAB_SIZE, TOTAL_LEN,
)
from minimal10digittransformer.data.addition import generate_batch, generate_test_set
from minimal10digittransformer.evaluation.metrics import evaluate


# ============================================================================
# Model building with novel tying
# ============================================================================

def count_params(model):
    """Count unique parameters (handles tied weights)."""
    seen = set()
    total = 0
    for p in model.parameters():
        if p.data_ptr() not in seen:
            seen.add(p.data_ptr())
            total += p.numel()
    return total


def apply_mlp_tying(model, tie_mode):
    """Apply MLP tying to an already-built model by modifying forward methods.

    tie_mode can be:
      'gate_alpha'      - gate = α·up (add scalar param)
      'gate_eq_up'      - gate = up
      'down_eq_upT'     - down = up^T
      'down_neg_upT'    - down = -up^T
      'gate_alpha+down_upT' - both gate_alpha and down_eq_upT
      'gate_eq_up+down_upT' - both gate_eq_up and down_eq_upT
    """
    mlp = model.block.mlp

    if 'gate_alpha' in tie_mode:
        # Add scalar α, remove gate_proj
        alpha = nn.Parameter(torch.tensor(1.0))
        mlp.gate_alpha = alpha
        if hasattr(mlp, 'gate_proj'):
            del mlp.gate_proj
        if 'down_upT' in tie_mode or 'down_neg_upT' in tie_mode:
            if hasattr(mlp, 'down_proj'):
                del mlp.down_proj
        mlp.tie_gate = True  # use up_proj for gate

        orig_forward = mlp.forward
        def gate_alpha_forward(self, x, _alpha=alpha):
            gate_out = self._act(self.up_proj(x) * _alpha)
            up_out = self.up_proj(x)
            if 'down_upT' in tie_mode:
                return F.linear(gate_out * up_out, self.up_proj.weight.t())
            elif 'down_neg_upT' in tie_mode:
                return F.linear(gate_out * up_out, -self.up_proj.weight.t())
            else:
                return self.down_proj(gate_out * up_out)
        mlp.forward = types.MethodType(gate_alpha_forward, mlp)

    elif 'gate_eq_up' in tie_mode:
        # gate = up (just use up_proj twice)
        if hasattr(mlp, 'gate_proj'):
            del mlp.gate_proj
        mlp.tie_gate = True

        def gate_eq_up_forward(self, x):
            up_out = self.up_proj(x)
            gate_out = self._act(up_out)
            if 'down_upT' in tie_mode:
                return F.linear(gate_out * up_out, self.up_proj.weight.t())
            elif 'down_neg_upT' in tie_mode:
                return F.linear(gate_out * up_out, -self.up_proj.weight.t())
            else:
                return self.down_proj(gate_out * up_out)
        mlp.forward = types.MethodType(gate_eq_up_forward, mlp)

    elif tie_mode == 'down_eq_upT':
        # down = up^T only
        del mlp.down_proj

        def down_upT_forward(self, x):
            gate_proj = self.up_proj if self.tie_gate else self.gate_proj
            return F.linear(self._act(gate_proj(x)) * self.up_proj(x),
                            self.up_proj.weight.t())
        mlp.forward = types.MethodType(down_upT_forward, mlp)

    elif tie_mode == 'down_neg_upT':
        # down = -up^T
        del mlp.down_proj

        def down_neg_upT_forward(self, x):
            gate_proj = self.up_proj if self.tie_gate else self.gate_proj
            return F.linear(self._act(gate_proj(x)) * self.up_proj(x),
                            -self.up_proj.weight.t())
        mlp.forward = types.MethodType(down_neg_upT_forward, mlp)

    return model


def apply_attn_tying(model, tie_mode):
    """Apply attention tying beyond standard tieKV/tieQO.

    tie_mode:
      'k_alpha_q'  - K = α·Q (scalar, saves head_dim*d - 1)
      'k_rot_q'    - K = R(φ)·Q (rotation, saves head_dim*d - 1)
    """
    attn = model.block.attn

    if tie_mode == 'k_alpha_q':
        alpha = nn.Parameter(torch.tensor(1.0))
        attn.k_alpha = alpha
        # Delete k_proj, derive from q_proj
        del attn.k_proj

        orig_forward = attn.forward
        def k_alpha_q_forward(self, x, mask=None, _alpha=alpha):
            B, T, _ = x.shape
            q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
            # K = α·Q (same projection, scaled)
            k = (self.q_proj(x) * _alpha).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
            # V uses q_proj too (since we're replacing k_proj)
            v_proj = self.q_proj if self.tie_kv else self.v_proj
            v = v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

            if self.use_qk_norm:
                q = self.q_norm(q)
                k = self.k_norm(k)

            q = apply_rope(q, self.rope_cos, self.rope_sin)
            k = apply_rope(k, self.rope_cos, self.rope_sin)

            if self.n_rep > 1:
                k = k.repeat_interleave(self.n_rep, dim=1)
                v = v.repeat_interleave(self.n_rep, dim=1)

            scale = 1.0 / math.sqrt(self.head_dim)
            scores = (q @ k.transpose(-2, -1)) * scale
            if mask is not None:
                scores = scores + mask[:T, :T]
            attn_weights = F.softmax(scores, dim=-1)

            out = (attn_weights @ v).transpose(1, 2).contiguous().view(B, T, -1)
            if self.tie_qo:
                return F.linear(out, self.q_proj.weight.t())
            return self.o_proj(out)

        # Need apply_rope in closure
        from minimal10digittransformer.model.qwen3 import apply_rope
        attn.forward = types.MethodType(k_alpha_q_forward, attn)

    elif tie_mode == 'v_eq_q':
        # Tie V projection to Q projection (both are head_dim × d_model)
        # V(x) = Q_proj(x) — same matrix, saves head_dim*d params (12p for hd=4, d=3)
        # Works because V and Q are used differently: Q gets QK-normed + RoPE, V goes to weighted sum
        attn = model.block.attn
        from minimal10digittransformer.model.qwen3 import apply_rope

        if hasattr(attn, 'v_proj'):
            del attn.v_proj

        def v_eq_q_forward(self, x, mask=None):
            B, T, _ = x.shape
            shared = self.q_proj(x)  # V and Q share this projection
            q = shared.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
            v = shared.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

            if hasattr(self, 'k_proj'):
                k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
            elif hasattr(self, 'k_alpha'):
                k = (self.q_proj(x) * self.k_alpha).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
            elif hasattr(self, 'k_rot_theta'):
                cos_t = torch.cos(self.k_rot_theta)
                sin_t = torch.sin(self.k_rot_theta)
                k_raw = shared.clone()
                k0 = k_raw[..., 0] * cos_t - k_raw[..., 1] * sin_t
                k1 = k_raw[..., 0] * sin_t + k_raw[..., 1] * cos_t
                k_raw = k_raw.clone()
                k_raw[..., 0] = k0
                k_raw[..., 1] = k1
                k = k_raw.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
            else:
                k = shared.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

            if self.use_qk_norm:
                q = self.q_norm(q)
                k = self.k_norm(k)

            q = apply_rope(q, self.rope_cos, self.rope_sin)
            k = apply_rope(k, self.rope_cos, self.rope_sin)

            if self.n_rep > 1:
                k = k.repeat_interleave(self.n_rep, dim=1)
                v = v.repeat_interleave(self.n_rep, dim=1)

            scale = 1.0 / math.sqrt(self.head_dim)
            scores = (q @ k.transpose(-2, -1)) * scale
            if mask is not None:
                scores = scores + mask[:T, :T]
            attn_weights = F.softmax(scores, dim=-1)

            out = (attn_weights @ v).transpose(1, 2).contiguous().view(B, T, -1)
            if self.tie_qo:
                return F.linear(out, self.q_proj.weight.t())
            return self.o_proj(out)

        attn.forward = types.MethodType(v_eq_q_forward, attn)

    elif tie_mode == 'k_rot_q':
        # MicroAdder-style: Q and K share projection, Q gets rotated by learnable angle
        # Saves head_dim*d - 1 params (same as k_alpha_q but rotation instead of scaling)
        theta_angle = nn.Parameter(torch.tensor(-0.5))  # ~-29° like MicroAdder converges to
        attn.k_rot_theta = theta_angle
        del attn.k_proj

        from minimal10digittransformer.model.qwen3 import apply_rope

        def k_rot_q_forward(self, x, mask=None, _theta=theta_angle):
            B, T, _ = x.shape
            shared = self.q_proj(x)  # (B, T, head_dim) via shared projection

            # K = shared (no rotation)
            k = shared.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

            # Q = rotate shared by theta on dims [0,1]
            q_raw = shared.clone()
            cos_t = torch.cos(_theta)
            sin_t = torch.sin(_theta)
            q0 = q_raw[..., 0] * cos_t - q_raw[..., 1] * sin_t
            q1 = q_raw[..., 0] * sin_t + q_raw[..., 1] * cos_t
            q_raw = q_raw.clone()
            q_raw[..., 0] = q0
            q_raw[..., 1] = q1
            q = q_raw.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

            v_proj = self.q_proj if self.tie_kv else self.v_proj
            v = v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

            if self.use_qk_norm:
                q = self.q_norm(q)
                k = self.k_norm(k)

            q = apply_rope(q, self.rope_cos, self.rope_sin)
            k = apply_rope(k, self.rope_cos, self.rope_sin)

            if self.n_rep > 1:
                k = k.repeat_interleave(self.n_rep, dim=1)
                v = v.repeat_interleave(self.n_rep, dim=1)

            scale = 1.0 / math.sqrt(self.head_dim)
            scores = (q @ k.transpose(-2, -1)) * scale
            if mask is not None:
                scores = scores + mask[:T, :T]
            attn_weights = F.softmax(scores, dim=-1)

            out = (attn_weights @ v).transpose(1, 2).contiguous().view(B, T, -1)
            if self.tie_qo:
                return F.linear(out, self.q_proj.weight.t())
            return self.o_proj(out)

        attn.forward = types.MethodType(k_rot_q_forward, attn)

    return model


def apply_rank1_projections(model):
    """Replace Q and K 4×3 linear layers with rank-1 factored: (4×1)·(1×3) = 7p instead of 12p.

    Saves 5p per projection. V and O keep their original form (or are tied).
    """
    attn = model.block.attn
    from minimal10digittransformer.model.qwen3 import apply_rope

    d_model = attn.q_proj.in_features  # 3
    head_dim = attn.q_proj.out_features  # 4

    # Rank-1 factors for Q: q = col_q @ row_q (4×1 · 1×3)
    col_q = nn.Parameter(torch.randn(head_dim, 1) * 0.1)
    row_q = nn.Parameter(torch.randn(1, d_model) * 0.1)
    attn.col_q = col_q
    attn.row_q = row_q

    # Rank-1 factors for K: k = col_k @ row_k
    col_k = nn.Parameter(torch.randn(head_dim, 1) * 0.1)
    row_k = nn.Parameter(torch.randn(1, d_model) * 0.1)
    attn.col_k = col_k
    attn.row_k = row_k

    # Delete original full-rank projections
    del attn.q_proj
    del attn.k_proj

    def rank1_forward(self, x, mask=None):
        B, T, _ = x.shape
        # Rank-1 Q: x @ row_q^T @ col_q^T = x @ (col_q @ row_q)^T
        q_weight = self.col_q @ self.row_q  # (4, 3)
        q = F.linear(x, q_weight).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        k_weight = self.col_k @ self.row_k  # (4, 3)
        k = F.linear(x, k_weight).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q = apply_rope(q, self.rope_cos, self.rope_sin)
        k = apply_rope(k, self.rope_cos, self.rope_sin)

        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = (q @ k.transpose(-2, -1)) * scale
        if mask is not None:
            scores = scores + mask[:T, :T]
        attn_weights = F.softmax(scores, dim=-1)

        out = (attn_weights @ v).transpose(1, 2).contiguous().view(B, T, -1)
        if self.tie_qo:
            # O = Q^T, but Q is rank-1 factored
            return F.linear(out, (self.col_q @ self.row_q).t())
        return self.o_proj(out)

    attn.forward = types.MethodType(rank1_forward, attn)
    return model


def apply_rank1_normalized(model):
    """Rank-1 Q/K with explicit factor normalization.

    Each projection is: W = scale * (col / ||col||) @ (row / ||row||)
    This prevents scale ambiguity between factors (col*10, row/10 = same W).
    The learnable 'scale' parameter controls the overall magnitude.
    Adds 1p per projection over basic rank-1 (still saves 4p each vs full 12p).
    """
    attn = model.block.attn
    from minimal10digittransformer.model.qwen3 import apply_rope

    d_model = attn.q_proj.in_features  # 3
    head_dim = attn.q_proj.out_features  # 4

    # Normalized rank-1 factors for Q
    col_q = nn.Parameter(torch.randn(head_dim, 1))
    row_q = nn.Parameter(torch.randn(1, d_model))
    scale_q = nn.Parameter(torch.tensor(1.0))
    attn.col_q = col_q
    attn.row_q = row_q
    attn.scale_q = scale_q

    # Normalized rank-1 factors for K
    col_k = nn.Parameter(torch.randn(head_dim, 1))
    row_k = nn.Parameter(torch.randn(1, d_model))
    scale_k = nn.Parameter(torch.tensor(1.0))
    attn.col_k = col_k
    attn.row_k = row_k
    attn.scale_k = scale_k

    del attn.q_proj
    del attn.k_proj

    def rank1_norm_forward(self, x, mask=None):
        B, T, _ = x.shape
        # Normalized rank-1: scale * (col/||col||) @ (row/||row||)
        col_q_n = self.col_q / (self.col_q.norm() + 1e-8)
        row_q_n = self.row_q / (self.row_q.norm() + 1e-8)
        q_weight = self.scale_q * (col_q_n @ row_q_n)
        q = F.linear(x, q_weight).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        col_k_n = self.col_k / (self.col_k.norm() + 1e-8)
        row_k_n = self.row_k / (self.row_k.norm() + 1e-8)
        k_weight = self.scale_k * (col_k_n @ row_k_n)
        k = F.linear(x, k_weight).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q = apply_rope(q, self.rope_cos, self.rope_sin)
        k = apply_rope(k, self.rope_cos, self.rope_sin)

        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = (q @ k.transpose(-2, -1)) * scale
        if mask is not None:
            scores = scores + mask[:T, :T]
        attn_weights = F.softmax(scores, dim=-1)

        out = (attn_weights @ v).transpose(1, 2).contiguous().view(B, T, -1)
        if self.tie_qo:
            return F.linear(out, (self.scale_q * (col_q_n @ row_q_n)).t())
        return self.o_proj(out)

    attn.forward = types.MethodType(rank1_norm_forward, attn)
    return model


def apply_noproj_qk(model):
    """Remove Q and K projections entirely — pad embedding to head_dim with zeros.

    Q and K are just the embedding padded from d_model=3 to head_dim=4.
    QK norms handle scale, RoPE handles position.
    Saves 24p (both Q and K projections removed).
    V projection is kept (needed for the value pathway).
    tieQO: O uses the padding operation transpose (just drops dim 4).
    """
    attn = model.block.attn
    from minimal10digittransformer.model.qwen3 import apply_rope

    head_dim = attn.head_dim  # 4
    d_model = attn.q_proj.in_features  # 3
    pad_dim = head_dim - d_model  # 1

    # Delete Q and K projections
    del attn.q_proj
    del attn.k_proj

    def noproj_forward(self, x, mask=None):
        B, T, D = x.shape
        # Pad d_model -> head_dim: [x0, x1, x2] -> [x0, x1, x2, 0]
        padded = F.pad(x, (0, pad_dim))  # (B, T, head_dim)
        q = padded.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = padded.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q = apply_rope(q, self.rope_cos, self.rope_sin)
        k = apply_rope(k, self.rope_cos, self.rope_sin)

        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = (q @ k.transpose(-2, -1)) * scale
        if mask is not None:
            scores = scores + mask[:T, :T]
        attn_weights = F.softmax(scores, dim=-1)

        out = (attn_weights @ v).transpose(1, 2).contiguous().view(B, T, -1)
        if self.tie_qo:
            # O is inverse of Q: just take first d_model dims
            return out[..., :D]
        return self.o_proj(out)

    attn.forward = types.MethodType(noproj_forward, attn)
    return model


def apply_noproj_qkv(model):
    """Remove ALL Q, K, V projections — pad embedding to head_dim for everything.

    Q = K = V = pad(embedding, head_dim).
    Attention becomes: softmax(RoPE(pad(x)) · RoPE(pad(x))^T) · pad(x).
    The MLP must do ALL the computation. Attention just does position-weighted averaging.
    tieQO: O just truncates back to d_model.
    Saves 36p (Q+K+V projections all removed).
    """
    attn = model.block.attn
    from minimal10digittransformer.model.qwen3 import apply_rope

    head_dim = attn.head_dim
    d_model = attn.q_proj.in_features
    pad_dim = head_dim - d_model

    del attn.q_proj
    del attn.k_proj
    del attn.v_proj

    def noproj_qkv_forward(self, x, mask=None):
        B, T, D = x.shape
        padded = F.pad(x, (0, pad_dim))
        q = padded.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = padded.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = padded.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q = apply_rope(q, self.rope_cos, self.rope_sin)
        k = apply_rope(k, self.rope_cos, self.rope_sin)

        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = (q @ k.transpose(-2, -1)) * scale
        if mask is not None:
            scores = scores + mask[:T, :T]
        attn_weights = F.softmax(scores, dim=-1)

        out = (attn_weights @ v).transpose(1, 2).contiguous().view(B, T, -1)
        if self.tie_qo:
            return out[..., :D]
        return self.o_proj(out)

    attn.forward = types.MethodType(noproj_qkv_forward, attn)
    return model


def apply_rank1_qk_noproj_v(model, normalized=False):
    """Rank-1 factored Q/K + projection-free V (pad embedding).

    Q, K: rank-1 factored (col·row), 7p each = 14p total
    V: just pad embedding from d_model=3 to head_dim=4, 0p
    Combines learned Q/K directions with zero-cost V.
    Saves 10p over rank1_qk (no V proj) or 22p over full (rank1 Q/K + no V).

    If normalized=True, uses scale * (col/||col||) @ (row/||row||) form.
    """
    attn = model.block.attn
    from minimal10digittransformer.model.qwen3 import apply_rope

    d_model = attn.q_proj.in_features  # 3
    head_dim = attn.q_proj.out_features  # 4
    pad_dim = head_dim - d_model  # 1

    # Rank-1 factors for Q
    col_q = nn.Parameter(torch.randn(head_dim, 1) * 0.1)
    row_q = nn.Parameter(torch.randn(1, d_model) * 0.1)
    attn.col_q = col_q
    attn.row_q = row_q

    # Rank-1 factors for K
    col_k = nn.Parameter(torch.randn(head_dim, 1) * 0.1)
    row_k = nn.Parameter(torch.randn(1, d_model) * 0.1)
    attn.col_k = col_k
    attn.row_k = row_k

    if normalized:
        attn.scale_q = nn.Parameter(torch.tensor(1.0))
        attn.scale_k = nn.Parameter(torch.tensor(1.0))

    del attn.q_proj
    del attn.k_proj
    del attn.v_proj

    _normalized = normalized

    def rank1_noproj_v_forward(self, x, mask=None):
        B, T, D = x.shape

        # Rank-1 Q
        if _normalized:
            cq = self.col_q / (self.col_q.norm() + 1e-8)
            rq = self.row_q / (self.row_q.norm() + 1e-8)
            q_weight = self.scale_q * (cq @ rq)
        else:
            q_weight = self.col_q @ self.row_q
        q = F.linear(x, q_weight).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Rank-1 K
        if _normalized:
            ck = self.col_k / (self.col_k.norm() + 1e-8)
            rk = self.row_k / (self.row_k.norm() + 1e-8)
            k_weight = self.scale_k * (ck @ rk)
        else:
            k_weight = self.col_k @ self.row_k
        k = F.linear(x, k_weight).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # V = pad(embedding) — no projection
        v = F.pad(x, (0, pad_dim)).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q = apply_rope(q, self.rope_cos, self.rope_sin)
        k = apply_rope(k, self.rope_cos, self.rope_sin)

        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = (q @ k.transpose(-2, -1)) * scale
        if mask is not None:
            scores = scores + mask[:T, :T]
        attn_weights = F.softmax(scores, dim=-1)

        out = (attn_weights @ v).transpose(1, 2).contiguous().view(B, T, -1)
        if self.tie_qo:
            # O = Q^T (rank-1 factored)
            if _normalized:
                cq = self.col_q / (self.col_q.norm() + 1e-8)
                rq = self.row_q / (self.row_q.norm() + 1e-8)
                return F.linear(out, (self.scale_q * (cq @ rq)).t())
            return F.linear(out, (self.col_q @ self.row_q).t())
        return self.o_proj(out)

    attn.forward = types.MethodType(rank1_noproj_v_forward, attn)
    return model


def apply_shared_rank1_qk(model, with_alpha=True, noproj_v=False):
    """Shared rank-1 factorization for Q and K.

    Q_weight = K_weight = col @ row (same 7p for both).
    Q and K get the SAME linear projection — differentiation comes from QK norms + RoPE.
    Optionally add scalar α: K_weight = α * Q_weight (like k_alpha_q but rank-1).

    with_alpha=True: adds 1p scalar between Q and K (+1p)
    noproj_v=True: V uses padding instead of projection (saves 12p more)

    Param count: 7p (shared col+row) + 1p (alpha, optional) + 12p (V, if kept)
    """
    attn = model.block.attn
    from minimal10digittransformer.model.qwen3 import apply_rope

    d_model = attn.q_proj.in_features  # 3
    head_dim = attn.q_proj.out_features  # 4
    pad_dim = head_dim - d_model  # 1

    # Shared rank-1 factors (same for Q and K)
    col = nn.Parameter(torch.randn(head_dim, 1) * 0.1)
    row = nn.Parameter(torch.randn(1, d_model) * 0.1)
    attn.shared_col = col
    attn.shared_row = row

    if with_alpha:
        attn.k_alpha = nn.Parameter(torch.tensor(1.0))

    del attn.q_proj
    del attn.k_proj
    if noproj_v:
        del attn.v_proj

    _with_alpha = with_alpha
    _noproj_v = noproj_v

    def shared_rank1_forward(self, x, mask=None):
        B, T, D = x.shape

        # Shared rank-1 weight
        w = self.shared_col @ self.shared_row  # (4, 3)

        q = F.linear(x, w).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        if _with_alpha:
            k = F.linear(x, w * self.k_alpha).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        else:
            k = F.linear(x, w).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if _noproj_v:
            v = F.pad(x, (0, pad_dim)).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        else:
            v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q = apply_rope(q, self.rope_cos, self.rope_sin)
        k = apply_rope(k, self.rope_cos, self.rope_sin)

        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = (q @ k.transpose(-2, -1)) * scale
        if mask is not None:
            scores = scores + mask[:T, :T]
        attn_weights = F.softmax(scores, dim=-1)

        out = (attn_weights @ v).transpose(1, 2).contiguous().view(B, T, -1)
        if self.tie_qo:
            # O = Q^T = (col @ row)^T
            return F.linear(out, w.t())
        return self.o_proj(out)

    attn.forward = types.MethodType(shared_rank1_forward, attn)
    return model


def apply_proj3x3_sin4th(model, drop_rope=False):
    """Replace 4×3 Q/K projections with 3×3 + sinusoidal 4th dim.

    Q,K project 3→3 (9p each instead of 12p), then a 4th coordinate is appended
    from a learnable sinusoidal: sin(pos * freq + phase).
    Saves 3p per Q/K projection. Adds 2p (freq, phase).

    If drop_rope=True, disables RoPE (sin4th provides position info instead).
    O projection: kept as full 3×4 (maps attention 4D output back to 3D model space).
    tieQO not supported (Q is 3×3, O needs to be 3×4).

    Args:
        drop_rope: if True, skip RoPE application (sin4th replaces it)
    """
    attn = model.block.attn
    from minimal10digittransformer.model.qwen3 import apply_rope

    d_model = attn.q_proj.in_features  # 3
    head_dim = attn.head_dim  # 4

    # Replace 4×3 with 3×3
    new_q = nn.Linear(d_model, d_model, bias=False)
    new_k = nn.Linear(d_model, d_model, bias=False)
    nn.init.normal_(new_q.weight, std=0.02)
    nn.init.normal_(new_k.weight, std=0.02)

    del attn.q_proj
    del attn.k_proj
    attn.q_proj_3 = new_q
    attn.k_proj_3 = new_k

    # If tieQO was set, we need a real O proj since Q is now 3×3 not 4×3
    if attn.tie_qo:
        attn.tie_qo = False
        attn.o_proj = nn.Linear(head_dim, d_model, bias=False)
        nn.init.normal_(attn.o_proj.weight, std=0.02)

    # Learnable sinusoidal for 4th dim (shared between Q and K)
    attn.sin4_freq = nn.Parameter(torch.tensor(1.0))
    attn.sin4_phase = nn.Parameter(torch.tensor(0.0))
    _drop_rope = drop_rope

    def proj3_sin4_forward(self, x, mask=None):
        B, T, _ = x.shape

        # 3×3 projection → 3D
        q3 = self.q_proj_3(x)  # (B, T, 3)
        k3 = self.k_proj_3(x)  # (B, T, 3)

        # 4th dim: sin(position * freq + phase)
        pos = torch.arange(T, device=x.device, dtype=x.dtype)
        sin4 = torch.sin(pos * self.sin4_freq + self.sin4_phase)  # (T,)
        sin4 = sin4.unsqueeze(0).unsqueeze(-1).expand(B, T, 1)  # (B, T, 1)

        # Concat to get 4D
        q = torch.cat([q3, sin4], dim=-1)  # (B, T, 4)
        k = torch.cat([k3, sin4], dim=-1)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if not _drop_rope:
            q = apply_rope(q, self.rope_cos, self.rope_sin)
            k = apply_rope(k, self.rope_cos, self.rope_sin)

        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = (q @ k.transpose(-2, -1)) * scale
        if mask is not None:
            scores = scores + mask[:T, :T]
        attn_weights = F.softmax(scores, dim=-1)

        out = (attn_weights @ v).transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(out)

    attn.forward = types.MethodType(proj3_sin4_forward, attn)
    return model


class QuadraticEmbeddingQwen3(nn.Module):
    """Model with quadratic embedding: e(d) = [c0 - c1*d², -d, c2*d] for d_model=3."""

    def __init__(self, d_model=3, n_heads=1, n_kv_heads=1, head_dim=4, ff=3,
                 rope_theta=3.0, max_len=TOTAL_LEN + 1, qk_norm=True,
                 tie_kv=False, tie_qo=False, tie_gate=False,
                 share_norms=False, share_block_norms=False,
                 window_size=0):
        super().__init__()
        self.d_model = d_model

        # Quadratic embedding: 3 params
        self.qe_c0 = nn.Parameter(torch.tensor(2.0))
        self.qe_c1 = nn.Parameter(torch.tensor(0.05))
        self.qe_c2 = nn.Parameter(torch.tensor(0.3))

        rope_cos, rope_sin = precompute_rope_freqs(head_dim, max_len, rope_theta)

        if share_norms:
            shared_norm = RMSNorm(d_model)
        elif share_block_norms:
            shared_norm = RMSNorm(d_model)
        else:
            shared_norm = None

        self.block = Qwen3Block(d_model, n_heads, n_kv_heads, head_dim, ff,
                                rope_cos, rope_sin, qk_norm=qk_norm,
                                tie_kv=tie_kv, tie_qo=tie_qo, tie_gate=tie_gate,
                                shared_norm=shared_norm)
        self.final_norm = shared_norm if share_norms else RMSNorm(d_model)

        mask = create_causal_mask(max_len, window_size)
        self.register_buffer("causal_mask", mask, persistent=False)
        self.apply(self._init_weights)

    def _compute_embedding_table(self):
        d = torch.arange(VOCAB_SIZE, device=self.qe_c0.device, dtype=self.qe_c0.dtype)
        if self.d_model == 3:
            return torch.stack([
                self.qe_c0 - self.qe_c1 * d * d,
                -d,
                self.qe_c2 * d,
            ], dim=1)
        elif self.d_model == 2:
            return torch.stack([
                self.qe_c0 - self.qe_c1 * d * d,
                -d,
            ], dim=1)
        else:
            emb = torch.zeros(VOCAB_SIZE, self.d_model, device=self.qe_c0.device)
            emb[:, 0] = self.qe_c0 - self.qe_c1 * d * d
            emb[:, 1] = -d
            if self.d_model >= 3:
                emb[:, 2] = self.qe_c2 * d
            return emb

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

    def forward(self, input_ids):
        emb_table = self._compute_embedding_table()
        x = emb_table[input_ids]
        x = self.block(x, self.causal_mask)
        x = self.final_norm(x)
        return F.linear(x, emb_table)


class FrozenRandomEmbeddingQwen3(CircularArcQwen3):
    """Like CircularArcQwen3 but with frozen random embedding (0 embed params)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Replace arc params with frozen random embedding table
        del self.arc_A
        del self.arc_start
        del self.arc_stride
        # Generate fixed random embedding
        torch.manual_seed(42)
        emb = torch.randn(VOCAB_SIZE, self.d_model) * 0.5
        self.register_buffer("frozen_embed", emb)

    def _compute_embedding_table(self):
        return self.frozen_embed


# ============================================================================
# Config generation
# ============================================================================

def generate_configs():
    """Generate all tying experiment configs."""
    configs = []

    # ── Base configs (known to grok) ──
    # 74p: arc_ff2_tieQO — 4/5 seeds grok, reliable
    base_74p = dict(
        d_model=3, ff=2, n_heads=1, n_kv_heads=1, head_dim=4,
        rope_theta=3.0, tie_qo=True,
    )
    # 80p: arc_ff3_tieKV_shbnorm — 3/5 seeds grok to 100%
    base_80p = dict(
        d_model=3, ff=3, n_heads=1, n_kv_heads=1, head_dim=4,
        rope_theta=3.0, tie_kv=True, share_block_norms=True,
    )
    # 68p: arc_ff2_tieQO_shnorm — 3/5 grok
    base_68p = dict(
        d_model=3, ff=2, n_heads=1, n_kv_heads=1, head_dim=4,
        rope_theta=3.0, tie_qo=True, share_norms=True,
    )

    # ── MLP tying experiments ──
    # Test on 74p base (ff=2, most tying savings applicable)
    mlp_ties = [
        ("gate_alpha",          "gate=α·up"),
        ("gate_eq_up",          "gate=up"),
        ("down_eq_upT",         "down=up^T"),
        ("down_neg_upT",        "down=-up^T"),
        ("gate_alpha+down_upT", "gate=α·up, down=up^T"),
        ("gate_eq_up+down_upT", "gate=up, down=up^T"),
    ]
    for tie_mode, desc in mlp_ties:
        configs.append({
            "name": f"74p_{tie_mode}",
            "base": "74p",
            "model_kwargs": {**base_74p},
            "mlp_tie": tie_mode,
            "attn_tie": None,
            "embed_type": "arc",
            "desc": f"74p base + {desc}",
        })

    # Also test MLP tying on 80p base (ff=3, bigger savings)
    for tie_mode, desc in mlp_ties:
        configs.append({
            "name": f"80p_{tie_mode}",
            "base": "80p",
            "model_kwargs": {**base_80p},
            "mlp_tie": tie_mode,
            "attn_tie": None,
            "embed_type": "arc",
            "desc": f"80p base + {desc}",
        })

    # ── Attention tying: K=α·Q on 74p (which doesn't have tieKV) ──
    configs.append({
        "name": "74p_k_alpha_q",
        "base": "74p",
        "model_kwargs": {**base_74p},
        "mlp_tie": None,
        "attn_tie": "k_alpha_q",
        "embed_type": "arc",
        "desc": "74p + K=α·Q (saves K proj)",
    })

    # ── Combo: K=α·Q + gate=α·up on 74p ──
    configs.append({
        "name": "74p_k_alpha_q+gate_alpha",
        "base": "74p",
        "model_kwargs": {**base_74p},
        "mlp_tie": "gate_alpha",
        "attn_tie": "k_alpha_q",
        "embed_type": "arc",
        "desc": "74p + K=α·Q + gate=α·up",
    })

    # ── Combo: K=α·Q + gate=α·up + down=up^T on 74p ──
    configs.append({
        "name": "74p_k_alpha_q+gate_alpha+down_upT",
        "base": "74p",
        "model_kwargs": {**base_74p},
        "mlp_tie": "gate_alpha+down_upT",
        "attn_tie": "k_alpha_q",
        "embed_type": "arc",
        "desc": "74p + K=α·Q + gate=α·up + down=up^T (max tying)",
    })

    # ── MicroAdder-style: shared Q/K with phase rotation ──
    configs.append({
        "name": "74p_k_rot_q",
        "base": "74p",
        "model_kwargs": {**base_74p},
        "mlp_tie": None,
        "attn_tie": "k_rot_q",
        "embed_type": "arc",
        "desc": "74p + K=R(θ)·shared_QK (MicroAdder-style rotation)",
    })
    # k_rot_q + gate_alpha combo (MicroAdder-style + our best MLP tying)
    configs.append({
        "name": "74p_k_rot_q+gate_alpha",
        "base": "74p",
        "model_kwargs": {**base_74p},
        "mlp_tie": "gate_alpha",
        "attn_tie": "k_rot_q",
        "embed_type": "arc",
        "desc": "74p + K=R(θ)·shared_QK + gate=α·up",
    })
    # k_rot_q + gate_alpha + down_upT (max combo with rotation)
    configs.append({
        "name": "74p_k_rot_q+gate_alpha+down_upT",
        "base": "74p",
        "model_kwargs": {**base_74p},
        "mlp_tie": "gate_alpha+down_upT",
        "attn_tie": "k_rot_q",
        "embed_type": "arc",
        "desc": "74p + K=R(θ)·shared + gate=α·up + down=up^T (max rot tying)",
    })
    # Also on 68p base (shnorm, known to grok)
    configs.append({
        "name": "68p_k_rot_q+gate_alpha",
        "base": "68p",
        "model_kwargs": {**base_68p},
        "mlp_tie": "gate_alpha",
        "attn_tie": "k_rot_q",
        "embed_type": "arc",
        "desc": "68p + K=R(θ)·shared + gate=α·up (shnorm base)",
    })

    # ── V = Q tying (saves 12p by sharing V and Q projections) ──
    configs.append({
        "name": "74p_v_eq_q",
        "base": "74p",
        "model_kwargs": {**base_74p},
        "mlp_tie": None,
        "attn_tie": "v_eq_q",
        "embed_type": "arc",
        "desc": "74p + V=Q (shared projection, saves 12p)",
    })
    # V=Q + our best MLP tying
    configs.append({
        "name": "74p_v_eq_q+k_alpha_q+gate_alpha",
        "base": "74p",
        "model_kwargs": {**base_74p},
        "mlp_tie": "gate_alpha",
        "attn_tie": "v_eq_q+k_alpha_q",
        "embed_type": "arc",
        "desc": "74p + V=Q + K=αQ + gate=αup (mega attention tie)",
    })
    # Mega combo: V=Q + K=αQ + gate=αup + down=upT
    configs.append({
        "name": "74p_mega_tie",
        "base": "74p",
        "model_kwargs": {**base_74p},
        "mlp_tie": "gate_alpha+down_upT",
        "attn_tie": "v_eq_q+k_alpha_q",
        "embed_type": "arc",
        "desc": "74p + V=Q + K=αQ + gate=αup + down=upT (max tie)",
    })
    # Mega combo with rotation
    configs.append({
        "name": "74p_mega_tie_rot",
        "base": "74p",
        "model_kwargs": {**base_74p},
        "mlp_tie": "gate_alpha+down_upT",
        "attn_tie": "v_eq_q+k_rot_q",
        "embed_type": "arc",
        "desc": "74p + V=Q + K=R(θ)Q + gate=αup + down=upT (max rot tie)",
    })

    # ── ff=3 mega ties (more MLP capacity, no deadly down_upT) ──
    # 83p base: arc ff=3 tieQO
    base_83p = dict(
        d_model=3, ff=3, n_heads=1, n_kv_heads=1, head_dim=4,
        rope_theta=3.0, tie_qo=True,
    )
    # 52p: V=Q + K=αQ + gate=αup on ff=3 base
    configs.append({
        "name": "83p_mega_ff3",
        "base": "83p",
        "model_kwargs": {**base_83p},
        "mlp_tie": "gate_alpha",
        "attn_tie": "v_eq_q+k_alpha_q",
        "embed_type": "arc",
        "desc": "83p (ff=3 tieQO) + V=Q + K=αQ + gate=αup (~52p, no down_upT)",
    })
    # 49p: + shbnorm
    base_83p_shbnorm = dict(
        d_model=3, ff=3, n_heads=1, n_kv_heads=1, head_dim=4,
        rope_theta=3.0, tie_qo=True, share_block_norms=True,
    )
    configs.append({
        "name": "80p_mega_ff3_shbnorm",
        "base": "80p_shbnorm",
        "model_kwargs": {**base_83p_shbnorm},
        "mlp_tie": "gate_alpha",
        "attn_tie": "v_eq_q+k_alpha_q",
        "embed_type": "arc",
        "desc": "80p (ff=3 tieQO shbnorm) + V=Q + K=αQ + gate=αup (~49p)",
    })
    # 46p: + shnorm
    base_83p_shnorm = dict(
        d_model=3, ff=3, n_heads=1, n_kv_heads=1, head_dim=4,
        rope_theta=3.0, tie_qo=True, share_norms=True,
    )
    configs.append({
        "name": "77p_mega_ff3_shnorm",
        "base": "77p_shnorm",
        "model_kwargs": {**base_83p_shnorm},
        "mlp_tie": "gate_alpha",
        "attn_tie": "v_eq_q+k_alpha_q",
        "embed_type": "arc",
        "desc": "77p (ff=3 tieQO shnorm) + V=Q + K=αQ + gate=αup (~46p)",
    })
    # 52p without V=Q: just K=αQ + gate=αup on ff=3 (more attention capacity)
    configs.append({
        "name": "83p_k_alpha_q+gate_alpha",
        "base": "83p",
        "model_kwargs": {**base_83p},
        "mlp_tie": "gate_alpha",
        "attn_tie": "k_alpha_q",
        "embed_type": "arc",
        "desc": "83p (ff=3 tieQO) + K=αQ + gate=αup (~64p, reference)",
    })

    # ── Embedding experiments on 68p base ──
    configs.append({
        "name": "68p_quadratic_embed",
        "base": "68p",
        "model_kwargs": {**base_68p},
        "mlp_tie": None,
        "attn_tie": None,
        "embed_type": "quadratic",
        "desc": "68p with quadratic embedding instead of arc",
    })
    configs.append({
        "name": "68p_frozen_random_embed",
        "base": "68p",
        "model_kwargs": {**base_68p},
        "mlp_tie": None,
        "attn_tie": None,
        "embed_type": "frozen_random",
        "desc": "68p with frozen random embedding (0 embed params)",
    })

    # ── RoPE theta sweep on 68p base ──
    for theta in [1.0, 2.0, 5.0, 10.0, 19.0]:
        configs.append({
            "name": f"68p_theta{theta:.0f}",
            "base": "68p",
            "model_kwargs": {**base_68p, "rope_theta": theta},
            "mlp_tie": None,
            "attn_tie": None,
            "embed_type": "arc",
            "desc": f"68p with RoPE theta={theta}",
        })

    # ── Rank-1 factored Q/K projections ──
    # On 122p base (no tying, most room): 4×3=12p → (4+3)=7p each, saves 10p
    base_122p = dict(
        d_model=3, ff=3, n_heads=1, n_kv_heads=1, head_dim=4,
        rope_theta=3.0,
    )
    configs.append({
        "name": "122p_rank1_qk",
        "base": "122p",
        "model_kwargs": {**base_122p},
        "mlp_tie": None,
        "attn_tie": "rank1_qk",
        "embed_type": "arc",
        "desc": "122p + rank-1 factored Q,K (7p each instead of 12p)",
    })
    # On 74p base (tieQO): rank-1 Q also determines O via tie
    configs.append({
        "name": "74p_rank1_qk",
        "base": "74p",
        "model_kwargs": {**base_74p},
        "mlp_tie": None,
        "attn_tie": "rank1_qk",
        "embed_type": "arc",
        "desc": "74p + rank-1 factored Q,K (tieQO uses rank-1 Q for O)",
    })

    # ── Rank-1 with normalization (prevents scale ambiguity) ──
    configs.append({
        "name": "74p_rank1_norm_qk",
        "base": "74p",
        "model_kwargs": {**base_74p},
        "mlp_tie": None,
        "attn_tie": "rank1_norm_qk",
        "embed_type": "arc",
        "desc": "74p + normalized rank-1 Q,K (scale * col/||col|| @ row/||row||)",
    })

    # ── Rank-1 QK + proven MLP ties (push param count lower) ──
    # rank-1 QK + gate_alpha on 74p: saves 10p (rank1) + 5p (gate) = 59p
    configs.append({
        "name": "74p_rank1_qk+gate_alpha",
        "base": "74p",
        "model_kwargs": {**base_74p},
        "mlp_tie": "gate_alpha",
        "attn_tie": "rank1_qk",
        "embed_type": "arc",
        "desc": "74p + rank-1 Q,K + gate=α·up (~59p)",
    })
    # Same with shnorm: ~53p
    configs.append({
        "name": "68p_rank1_qk+gate_alpha",
        "base": "68p",
        "model_kwargs": {**base_68p},
        "mlp_tie": "gate_alpha",
        "attn_tie": "rank1_qk",
        "embed_type": "arc",
        "desc": "68p (shnorm) + rank-1 Q,K + gate=α·up (~53p)",
    })
    # Normalized version of the 53p
    configs.append({
        "name": "68p_rank1_norm+gate_alpha",
        "base": "68p",
        "model_kwargs": {**base_68p},
        "mlp_tie": "gate_alpha",
        "attn_tie": "rank1_norm_qk",
        "embed_type": "arc",
        "desc": "68p (shnorm) + normalized rank-1 Q,K + gate=α·up (~55p)",
    })

    # ── Projection-free Q/K (pad embedding to head_dim, no Q/K matrices) ──
    # Saves 24p by removing both Q and K projections entirely
    configs.append({
        "name": "74p_noproj_qk",
        "base": "74p",
        "model_kwargs": {**base_74p},
        "mlp_tie": None,
        "attn_tie": "noproj_qk",
        "embed_type": "arc",
        "desc": "74p + no Q/K projection (pad emb to 4D, ~50p)",
    })
    configs.append({
        "name": "68p_noproj_qk",
        "base": "68p",
        "model_kwargs": {**base_68p},
        "mlp_tie": None,
        "attn_tie": "noproj_qk",
        "embed_type": "arc",
        "desc": "68p (shnorm) + no Q/K projection (~44p)",
    })
    # noproj + gate_alpha = radical savings
    configs.append({
        "name": "74p_noproj_qk+gate_alpha",
        "base": "74p",
        "model_kwargs": {**base_74p},
        "mlp_tie": "gate_alpha",
        "attn_tie": "noproj_qk",
        "embed_type": "arc",
        "desc": "74p + no Q/K + gate=α·up (~45p)",
    })
    configs.append({
        "name": "68p_noproj_qk+gate_alpha",
        "base": "68p",
        "model_kwargs": {**base_68p},
        "mlp_tie": "gate_alpha",
        "attn_tie": "noproj_qk",
        "embed_type": "arc",
        "desc": "68p (shnorm) + no Q/K + gate=α·up (~39p)",
    })

    # ── No QKV projections at all (pad embedding, V=pad too) ──
    configs.append({
        "name": "74p_noproj_qkv",
        "base": "74p",
        "model_kwargs": {**base_74p},
        "mlp_tie": None,
        "attn_tie": "noproj_qkv",
        "embed_type": "arc",
        "desc": "74p + no Q/K/V projection (MLP does all work, ~38p)",
    })
    configs.append({
        "name": "68p_noproj_qkv+gate_alpha",
        "base": "68p",
        "model_kwargs": {**base_68p},
        "mlp_tie": "gate_alpha",
        "attn_tie": "noproj_qkv",
        "embed_type": "arc",
        "desc": "68p (shnorm) + no Q/K/V + gate=αup (~27p!)",
    })
    # V-only: keep V projection, remove Q and K
    configs.append({
        "name": "74p_noproj_qk_vonly",
        "base": "74p",
        "model_kwargs": {**base_74p},
        "mlp_tie": "gate_alpha",
        "attn_tie": "noproj_qk",
        "embed_type": "arc",
        "desc": "74p + no Q/K + gate=αup (V kept, ~45p)",
    })

    # ── Rank-1 QK + no V projection (learned Q/K direction, free V) ──
    configs.append({
        "name": "74p_rank1_qk_noproj_v",
        "base": "74p",
        "model_kwargs": {**base_74p},
        "mlp_tie": None,
        "attn_tie": "rank1_qk_noproj_v",
        "embed_type": "arc",
        "desc": "74p + rank1 Q/K + pad V (~40p)",
    })
    configs.append({
        "name": "68p_rank1_qk_noproj_v",
        "base": "68p",
        "model_kwargs": {**base_68p},
        "mlp_tie": None,
        "attn_tie": "rank1_qk_noproj_v",
        "embed_type": "arc",
        "desc": "68p (shnorm) + rank1 Q/K + pad V (~34p)",
    })
    configs.append({
        "name": "74p_rank1_qk_noproj_v+gate_alpha",
        "base": "74p",
        "model_kwargs": {**base_74p},
        "mlp_tie": "gate_alpha",
        "attn_tie": "rank1_qk_noproj_v",
        "embed_type": "arc",
        "desc": "74p + rank1 Q/K + pad V + gate=α·up (~35p)",
    })
    configs.append({
        "name": "68p_rank1_qk_noproj_v+gate_alpha",
        "base": "68p",
        "model_kwargs": {**base_68p},
        "mlp_tie": "gate_alpha",
        "attn_tie": "rank1_qk_noproj_v",
        "embed_type": "arc",
        "desc": "68p (shnorm) + rank1 Q/K + pad V + gate=α·up (~29p)",
    })
    # Normalized rank-1 QK + no V
    configs.append({
        "name": "74p_rank1_norm_qk_noproj_v",
        "base": "74p",
        "model_kwargs": {**base_74p},
        "mlp_tie": None,
        "attn_tie": "rank1_norm_qk_noproj_v",
        "embed_type": "arc",
        "desc": "74p + normalized rank1 Q/K + pad V (~42p)",
    })

    # ── Shared rank-1 QK (same projection for Q and K) ──
    configs.append({
        "name": "74p_shared_rank1_qk",
        "base": "74p",
        "model_kwargs": {**base_74p},
        "mlp_tie": None,
        "attn_tie": "shared_rank1_qk",
        "embed_type": "arc",
        "desc": "74p + shared rank1 Q=K + alpha (~46p)",
    })
    configs.append({
        "name": "68p_shared_rank1_qk",
        "base": "68p",
        "model_kwargs": {**base_68p},
        "mlp_tie": None,
        "attn_tie": "shared_rank1_qk",
        "embed_type": "arc",
        "desc": "68p (shnorm) + shared rank1 Q=K + alpha (~40p)",
    })
    configs.append({
        "name": "74p_shared_rank1_qk+gate_alpha",
        "base": "74p",
        "model_kwargs": {**base_74p},
        "mlp_tie": "gate_alpha",
        "attn_tie": "shared_rank1_qk",
        "embed_type": "arc",
        "desc": "74p + shared rank1 QK + gate=α·up (~41p)",
    })
    configs.append({
        "name": "68p_shared_rank1_qk+gate_alpha",
        "base": "68p",
        "model_kwargs": {**base_68p},
        "mlp_tie": "gate_alpha",
        "attn_tie": "shared_rank1_qk",
        "embed_type": "arc",
        "desc": "68p (shnorm) + shared rank1 QK + gate=α·up (~35p)",
    })
    # Shared rank-1 QK + no V projection
    configs.append({
        "name": "74p_shared_rank1_qk_noproj_v",
        "base": "74p",
        "model_kwargs": {**base_74p},
        "mlp_tie": None,
        "attn_tie": "shared_rank1_qk_noproj_v",
        "embed_type": "arc",
        "desc": "74p + shared rank1 Q=K + pad V (~34p)",
    })
    configs.append({
        "name": "68p_shared_rank1_qk_noproj_v",
        "base": "68p",
        "model_kwargs": {**base_68p},
        "mlp_tie": None,
        "attn_tie": "shared_rank1_qk_noproj_v",
        "embed_type": "arc",
        "desc": "68p (shnorm) + shared rank1 Q=K + pad V (~28p)",
    })
    configs.append({
        "name": "74p_shared_rank1_qk_noproj_v+gate_alpha",
        "base": "74p",
        "model_kwargs": {**base_74p},
        "mlp_tie": "gate_alpha",
        "attn_tie": "shared_rank1_qk_noproj_v",
        "embed_type": "arc",
        "desc": "74p + shared rank1 QK + pad V + gate=α·up (~29p)",
    })
    configs.append({
        "name": "68p_shared_rank1_qk_noproj_v+gate_alpha",
        "base": "68p",
        "model_kwargs": {**base_68p},
        "mlp_tie": "gate_alpha",
        "attn_tie": "shared_rank1_qk_noproj_v",
        "embed_type": "arc",
        "desc": "68p (shnorm) + shared rank1 QK + pad V + gate=α·up (~23p!)",
    })
    # Shared rank-1 without alpha (Q = K exactly)
    configs.append({
        "name": "74p_shared_rank1_qk_noalpha",
        "base": "74p",
        "model_kwargs": {**base_74p},
        "mlp_tie": None,
        "attn_tie": "shared_rank1_qk_noalpha",
        "embed_type": "arc",
        "desc": "74p + shared rank1 Q=K exact (~45p)",
    })

    # ── 3×3 projections + sinusoidal 4th dim ──
    # On 122p base: saves 6p from Q/K (9p each vs 12p), adds 2p + 12p for O = net +2p
    # But on 122p without tieQO, O was already 12p, so net saves 4p
    configs.append({
        "name": "122p_proj3x3_sin4",
        "base": "122p",
        "model_kwargs": {**base_122p},
        "mlp_tie": None,
        "attn_tie": "proj3x3_sin4",
        "embed_type": "arc",
        "desc": "122p + 3×3 Q,K + sinusoidal 4th dim (with RoPE)",
    })
    # Same but DROP RoPE (sin4th provides position)
    configs.append({
        "name": "122p_proj3x3_sin4_norope",
        "base": "122p",
        "model_kwargs": {**base_122p},
        "mlp_tie": None,
        "attn_tie": "proj3x3_sin4_norope",
        "embed_type": "arc",
        "desc": "122p + 3×3 Q,K + sin 4th dim, NO RoPE",
    })
    # On 74p base (note: tieQO gets disabled, O becomes separate 3×4)
    configs.append({
        "name": "74p_proj3x3_sin4",
        "base": "74p",
        "model_kwargs": {**base_74p},
        "mlp_tie": None,
        "attn_tie": "proj3x3_sin4",
        "embed_type": "arc",
        "desc": "74p + 3×3 Q,K + sin 4th dim (tieQO disabled, O separate)",
    })
    configs.append({
        "name": "74p_proj3x3_sin4_norope",
        "base": "74p",
        "model_kwargs": {**base_74p},
        "mlp_tie": None,
        "attn_tie": "proj3x3_sin4_norope",
        "embed_type": "arc",
        "desc": "74p + 3×3 Q,K + sin 4th, NO RoPE (tieQO disabled)",
    })

    return configs


def build_model(cfg, device):
    """Build model with appropriate tying applied."""
    kwargs = cfg["model_kwargs"]
    embed_type = cfg["embed_type"]

    if embed_type == "quadratic":
        model = QuadraticEmbeddingQwen3(**kwargs)
    elif embed_type == "frozen_random":
        model = FrozenRandomEmbeddingQwen3(**kwargs)
    else:
        model = CircularArcQwen3(**kwargs)

    model = model.to(device)

    # Apply MLP tying
    if cfg["mlp_tie"]:
        model = apply_mlp_tying(model, cfg["mlp_tie"])

    # Apply attention tying / projection changes
    attn_tie = cfg["attn_tie"]
    if attn_tie == "rank1_qk":
        model = apply_rank1_projections(model)
    elif attn_tie == "rank1_norm_qk":
        model = apply_rank1_normalized(model)
    elif attn_tie == "noproj_qk":
        model = apply_noproj_qk(model)
    elif attn_tie == "noproj_qkv":
        model = apply_noproj_qkv(model)
    elif attn_tie == "rank1_qk_noproj_v":
        model = apply_rank1_qk_noproj_v(model, normalized=False)
    elif attn_tie == "rank1_norm_qk_noproj_v":
        model = apply_rank1_qk_noproj_v(model, normalized=True)
    elif attn_tie == "shared_rank1_qk":
        model = apply_shared_rank1_qk(model, with_alpha=True, noproj_v=False)
    elif attn_tie == "shared_rank1_qk_noalpha":
        model = apply_shared_rank1_qk(model, with_alpha=False, noproj_v=False)
    elif attn_tie == "shared_rank1_qk_noproj_v":
        model = apply_shared_rank1_qk(model, with_alpha=True, noproj_v=True)
    elif attn_tie == "proj3x3_sin4":
        model = apply_proj3x3_sin4th(model, drop_rope=False)
    elif attn_tie == "proj3x3_sin4_norope":
        model = apply_proj3x3_sin4th(model, drop_rope=True)
    elif attn_tie:
        # For compound ties with v_eq_q, need special handling since v_eq_q's
        # forward already handles k_alpha_q/k_rot_q internally.
        # Just set up the K attributes first, then install v_eq_q forward.
        if 'v_eq_q' in attn_tie:
            attn = model.block.attn
            if 'k_alpha_q' in attn_tie:
                alpha = nn.Parameter(torch.tensor(1.0))
                attn.k_alpha = alpha
                if hasattr(attn, 'k_proj'):
                    del attn.k_proj
            elif 'k_rot_q' in attn_tie:
                theta_angle = nn.Parameter(torch.tensor(-0.5))
                attn.k_rot_theta = theta_angle
                if hasattr(attn, 'k_proj'):
                    del attn.k_proj
            model = apply_attn_tying(model, 'v_eq_q')
        else:
            model = apply_attn_tying(model, attn_tie)

    return model


# ============================================================================
# Training
# ============================================================================

def train_one(cfg, seed, max_steps, eval_interval, device, eval_pairs):
    random.seed(seed)
    torch.manual_seed(seed)

    model = build_model(cfg, device)
    n_params = count_params(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.01)

    best_acc = 0.0
    best_step = 0
    grok_step = None
    start_time = time.time()

    for step in range(1, max_steps + 1):
        progress = step / max(max_steps, 1)
        cur_lr = 0.001 + 0.5 * (0.01 - 0.001) * (1 + math.cos(math.pi * progress))
        for pg in optimizer.param_groups:
            pg["lr"] = cur_lr

        model.train()
        batch, labels = generate_batch(128, device)
        logits = model(batch)
        shift_logits = logits[:, :-1, :].reshape(-1, VOCAB_SIZE)
        shift_labels = labels[:, 1:].reshape(-1)
        loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % eval_interval == 0 or step == max_steps:
            model.eval()
            with torch.no_grad():
                acc, _ = evaluate(model, device, test_pairs=eval_pairs[:500])

            elapsed = time.time() - start_time
            print(f"  [{cfg['name']} s{seed}] step {step}/{max_steps} "
                  f"loss={loss.item():.4f} acc={acc:.1%} best={best_acc:.1%} "
                  f"[{elapsed:.0f}s]", flush=True)

            if acc > best_acc:
                best_acc = acc
                best_step = step

            if grok_step is None and acc > 0.5:
                grok_step = step

            if acc >= 0.999:
                break

    elapsed = time.time() - start_time
    return {
        "config_name": cfg["name"],
        "base": cfg["base"],
        "mlp_tie": cfg["mlp_tie"] or "",
        "attn_tie": cfg["attn_tie"] or "",
        "embed_type": cfg["embed_type"],
        "n_params": n_params,
        "seed": seed,
        "best_exact_acc": best_acc,
        "best_step": best_step,
        "grok_step": grok_step or "",
        "final_loss": loss.item(),
        "final_step": step,
        "elapsed_s": elapsed,
        "desc": cfg["desc"],
    }


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Matrix tying search")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=100000)
    parser.add_argument("--eval-interval", type=int, default=5000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="experiments/tying_search_results.csv")
    parser.add_argument("--config", default=None, help="Run only this config name")
    args = parser.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", "4")

    configs = generate_configs()
    seed_list = [1, 8, 15][:args.seeds]
    eval_pairs = generate_test_set(2000, seed=12345)

    if args.config:
        # Support exact match OR substring filter (pipe-separated patterns)
        patterns = args.config.split("|")
        configs = [c for c in configs if any(p in c["name"] for p in patterns)]
        if not configs:
            all_names = [c["name"] for c in generate_configs()]
            print(f"Unknown config: {args.config}")
            print(f"Available: {all_names}")
            return

    # Load completed
    completed = set()
    if os.path.exists(args.output):
        with open(args.output) as f:
            for row in csv.DictReader(f):
                completed.add((row["config_name"], int(row["seed"])))

    # Sort by expected param count (build one to count, then sort)
    configs_with_params = []
    for cfg in configs:
        try:
            m = build_model(cfg, "cpu")
            n = count_params(m)
            configs_with_params.append((n, cfg))
            del m
        except Exception as e:
            print(f"  SKIP {cfg['name']}: {e}")
    configs_with_params.sort(key=lambda x: (x[0], x[1]["name"]))

    total = sum(1 for _, cfg in configs_with_params for s in seed_list
                if (cfg["name"], s) not in completed)
    print(f"Tying search: {len(configs_with_params)} configs × {len(seed_list)} seeds")
    print(f"Remaining: {total}, {len(completed)} done")
    print()

    for n, cfg in configs_with_params:
        print(f"  {n:3d}p  {cfg['name']:45s}  {cfg['desc']}")
    print()

    fieldnames = ["config_name", "base", "mlp_tie", "attn_tie", "embed_type",
                  "n_params", "seed", "best_exact_acc", "best_step", "grok_step",
                  "final_loss", "final_step", "elapsed_s", "desc"]

    write_header = not os.path.exists(args.output) or os.path.getsize(args.output) == 0
    done = 0

    for n_params, cfg in configs_with_params:
        for seed in seed_list:
            if (cfg["name"], seed) in completed:
                continue

            done += 1
            print(f"\n[{done}/{total}] {cfg['name']} ({n_params}p) seed={seed}")
            print(f"  {cfg['desc']}")

            try:
                result = train_one(cfg, seed, args.max_steps, args.eval_interval,
                                   args.device, eval_pairs)
            except Exception as e:
                print(f"  ERROR: {e}")
                continue

            with open(args.output, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if write_header:
                    writer.writeheader()
                    write_header = False
                writer.writerow(result)

            grok = f"grok@{result['grok_step']}" if result['grok_step'] else "no grok"
            print(f"  -> {result['best_exact_acc']:.1%} (step {result['best_step']}) {grok}")

    print(f"\nDone. Results in {args.output}")


if __name__ == "__main__":
    main()
