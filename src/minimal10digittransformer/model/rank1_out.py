"""Qwen3 model with rank-1 attention output projection.

Replaces the dense O projection (head_dim x d_model = 12 params for hd=4, d=3)
with a rank-1 factorization:
    out_proj_A: (head_dim, 1) = 4 params
    out_proj_B: (1, d_model) = 3 params
    Total: 7 params instead of 12 -> saves 5 params

NOT compatible with tie_qo (since O is no longer a dense matrix that can be Q^T).
Use with tieKV only.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from minimal10digittransformer.model.qwen3 import (
    Qwen3MLP,
    RMSNorm,
    precompute_rope_freqs,
    apply_rope,
    VOCAB_SIZE,
    TOTAL_LEN,
)


class Rank1OutAttention(nn.Module):
    """Qwen3-style attention with rank-1 output projection.

    Instead of o_proj = Linear(head_dim, d_model), uses:
        out = (attn_out @ out_proj_A) @ out_proj_B
    where out_proj_A: (head_dim, 1), out_proj_B: (1, d_model).
    """

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int,
                 head_dim: int, rope_cos: torch.Tensor, rope_sin: torch.Tensor,
                 qk_norm: bool = True, tie_kv: bool = False):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.n_rep = n_heads // n_kv_heads
        self.use_qk_norm = qk_norm
        self.tie_kv = tie_kv

        self.q_proj = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        if not tie_kv:
            self.v_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)

        # Rank-1 output projection (replaces dense o_proj)
        self.out_proj_A = nn.Parameter(torch.empty(n_heads * head_dim, 1))
        self.out_proj_B = nn.Parameter(torch.empty(1, d_model))
        nn.init.kaiming_uniform_(self.out_proj_A, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.out_proj_B, a=math.sqrt(5))

        if qk_norm:
            self.q_norm = RMSNorm(head_dim)
            self.k_norm = RMSNorm(head_dim)

        self.register_buffer("rope_cos", rope_cos, persistent=False)
        self.register_buffer("rope_sin", rope_sin, persistent=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v_proj = self.k_proj if self.tie_kv else self.v_proj
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
        attn = (q @ k.transpose(-2, -1)) * scale
        if mask is not None:
            attn = attn + mask[:T, :T]
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, -1)
        # Rank-1 output: (B, T, n_heads*head_dim) @ (n_heads*head_dim, 1) @ (1, d_model)
        out = (out @ self.out_proj_A) @ self.out_proj_B
        return out


class Rank1OutBlock(nn.Module):
    """Transformer block using rank-1 output attention."""

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int,
                 head_dim: int, ff: int, rope_cos, rope_sin,
                 qk_norm: bool = True, use_swiglu: bool = True,
                 tie_kv: bool = False, tie_gate: bool = False,
                 shared_norm: nn.Module = None):
        super().__init__()
        if shared_norm is not None:
            self.ln1 = shared_norm
            self.ln2 = shared_norm
        else:
            self.ln1 = RMSNorm(d_model)
            self.ln2 = RMSNorm(d_model)
        self.attn = Rank1OutAttention(
            d_model, n_heads, n_kv_heads, head_dim, rope_cos, rope_sin,
            qk_norm=qk_norm, tie_kv=tie_kv,
        )
        self.mlp = Qwen3MLP(d_model, ff, use_swiglu=use_swiglu, tie_gate=tie_gate)

    def forward(self, x, mask=None):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.mlp(self.ln2(x))
        return x


class Rank1OutModel(nn.Module):
    """Qwen3 model with rank-1 attention output projection."""

    def __init__(self, d_model: int, n_heads: int = 1, n_kv_heads: int = 1,
                 head_dim: int = 4, ff: int = 3, rope_theta: float = 3.0,
                 max_len: int = TOTAL_LEN + 1, qk_norm: bool = True,
                 use_swiglu: bool = True, tie_kv: bool = False,
                 tie_gate: bool = False, repeats: int = 1,
                 share_norms: bool = False, share_block_norms: bool = False):
        super().__init__()
        self.d_model = d_model
        self.repeats = repeats

        # Standard tied embedding
        self.embed = nn.Embedding(VOCAB_SIZE, d_model)
        self.lm_head_weight = self.embed.weight

        rope_cos, rope_sin = precompute_rope_freqs(head_dim, max_len, rope_theta)

        if share_norms:
            shared_norm = RMSNorm(d_model)
        elif share_block_norms:
            shared_norm = RMSNorm(d_model)
        else:
            shared_norm = None

        self.block = Rank1OutBlock(
            d_model, n_heads, n_kv_heads, head_dim, ff,
            rope_cos, rope_sin, qk_norm=qk_norm,
            use_swiglu=use_swiglu, tie_kv=tie_kv,
            tie_gate=tie_gate, shared_norm=shared_norm,
        )
        self.final_norm = shared_norm if share_norms else RMSNorm(d_model)

        mask = torch.full((max_len, max_len), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        self.register_buffer("causal_mask", mask, persistent=False)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        for _ in range(self.repeats):
            x = self.block(x, self.causal_mask)
        x = self.final_norm(x)
        logits = F.linear(x, self.lm_head_weight)
        return logits
