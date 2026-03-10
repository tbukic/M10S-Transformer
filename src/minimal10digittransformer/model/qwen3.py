"""Qwen3-style 1-layer transformer for 10-digit addition.

Architecture: Single Qwen3 block with GQA, RoPE, SwiGLU MLP, RMSNorm.
Key insight: RoPE theta=3 makes d_model=3 viable.
Supports aggressive weight tying (tie_kv, tie_qo, tie_gate, share_norms).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Constants ────────────────────────────────────────────────────────────────

NUM_DIGITS = 10
SUM_DIGITS = 11
MAX_ADDEND = 10**NUM_DIGITS - 1
VOCAB_SIZE = 10  # digits 0-9
# Input: [0] + reversed(a, 10 digits) + [0, 0] + reversed(b, 10 digits) + [0] = 24 tokens
# Output: 11 reversed sum digits (autoregressive)
INPUT_LEN = 24
OUTPUT_LEN = SUM_DIGITS
TOTAL_LEN = INPUT_LEN + OUTPUT_LEN  # 35


# ── Causal Mask ──────────────────────────────────────────────────────────────

def create_causal_mask(max_len: int, window_size: int = 0) -> torch.Tensor:
    """Causal mask, optionally with sliding window.

    window_size=0: standard full causal (default, no behavior change)
    window_size=W: each position attends to at most W previous positions
                   (including self)
    """
    mask = torch.full((max_len, max_len), float("-inf"))
    for i in range(max_len):
        start = max(0, i - window_size + 1) if window_size > 0 else 0
        mask[i, start:i + 1] = 0.0
    return mask


# ── RoPE ─────────────────────────────────────────────────────────────────────

def precompute_rope_freqs(head_dim: int, max_len: int, theta: float = 3.0):
    """Precompute RoPE frequency tensor."""
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)  # [max_len, head_dim/2]
    cos = freqs.cos()
    sin = freqs.sin()
    return cos, sin  # each [max_len, head_dim/2]


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to x of shape [B, n_heads, T, head_dim]."""
    T = x.shape[2]
    cos_t = cos[:T].unsqueeze(0).unsqueeze(0)  # [1, 1, T, hd/2]
    sin_t = sin[:T].unsqueeze(0).unsqueeze(0)
    # Split into pairs
    x1, x2 = x[..., ::2], x[..., 1::2]  # each [..., hd/2]
    out1 = x1 * cos_t - x2 * sin_t
    out2 = x1 * sin_t + x2 * cos_t
    return torch.stack([out1, out2], dim=-1).flatten(-2)


# ── RMSNorm ──────────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms).to(x.dtype) * self.weight


# ── GQA Attention with RoPE ──────────────────────────────────────────────────

class Qwen3Attention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int,
                 head_dim: int, rope_cos: torch.Tensor, rope_sin: torch.Tensor,
                 qk_norm: bool = True, tie_kv: bool = False, tie_qo: bool = False,
                 share_qk_norm: bool = False):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.n_rep = n_heads // n_kv_heads  # GQA repeat factor
        self.use_qk_norm = qk_norm
        self.tie_kv = tie_kv
        self.tie_qo = tie_qo

        self.q_proj = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        if not tie_kv:
            self.v_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        if not tie_qo:
            self.o_proj = nn.Linear(n_heads * head_dim, d_model, bias=False)

        # Qwen3 has Q/K norms (optional)
        if qk_norm:
            self.q_norm = RMSNorm(head_dim)
            self.k_norm = self.q_norm if share_qk_norm else RMSNorm(head_dim)

        self.register_buffer("rope_cos", rope_cos, persistent=False)
        self.register_buffer("rope_sin", rope_sin, persistent=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v_proj = self.k_proj if self.tie_kv else self.v_proj
        v = v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Q/K norms (Qwen3 style)
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # Apply RoPE
        q = apply_rope(q, self.rope_cos, self.rope_sin)
        k = apply_rope(k, self.rope_cos, self.rope_sin)

        # GQA: repeat KV heads
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) * scale
        if mask is not None:
            attn = attn + mask[:T, :T]
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, -1)
        if self.tie_qo:
            return F.linear(out, self.q_proj.weight.t())
        return self.o_proj(out)


# ── SwiGLU MLP ───────────────────────────────────────────────────────────────

class Qwen3MLP(nn.Module):
    def __init__(self, d_model: int, intermediate_size: int, use_swiglu: bool = True,
                 tie_gate: bool = False, activation: str = "default",
                 mlp_bias: bool = False):
        super().__init__()
        self.use_swiglu = use_swiglu
        self.tie_gate = tie_gate
        # activation: "default" = silu for swiglu / gelu for non-swiglu,
        #             "relu", "silu", "gelu"
        self.activation = activation
        if use_swiglu:
            if not tie_gate:
                self.gate_proj = nn.Linear(d_model, intermediate_size, bias=False)
            self.up_proj = nn.Linear(d_model, intermediate_size, bias=mlp_bias)
        else:
            self.up_proj = nn.Linear(d_model, intermediate_size, bias=mlp_bias)
        self.down_proj = nn.Linear(intermediate_size, d_model, bias=False)

    def _act(self, x):
        if self.activation == "relu":
            return F.relu(x)
        elif self.activation == "silu":
            return F.silu(x)
        elif self.activation == "gelu":
            return F.gelu(x)
        elif self.activation == "default":
            return F.silu(x) if self.use_swiglu else F.gelu(x)
        return F.silu(x) if self.use_swiglu else F.gelu(x)

    def forward(self, x):
        if self.use_swiglu:
            gate_proj = self.up_proj if self.tie_gate else self.gate_proj
            return self.down_proj(self._act(gate_proj(x)) * self.up_proj(x))
        else:
            return self.down_proj(self._act(self.up_proj(x)))


# ── Transformer Block ────────────────────────────────────────────────────────

class Qwen3Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int,
                 head_dim: int, ff: int, rope_cos, rope_sin,
                 qk_norm: bool = True, use_swiglu: bool = True, tie_kv: bool = False,
                 tie_qo: bool = False, tie_gate: bool = False,
                 shared_norm: nn.Module = None, activation: str = "default",
                 share_qk_norm: bool = False, mlp_bias: bool = False):
        super().__init__()
        if shared_norm is not None:
            self.ln1 = shared_norm
            self.ln2 = shared_norm
        else:
            self.ln1 = RMSNorm(d_model)
            self.ln2 = RMSNorm(d_model)
        self.attn = Qwen3Attention(d_model, n_heads, n_kv_heads, head_dim, rope_cos, rope_sin, qk_norm=qk_norm, tie_kv=tie_kv, tie_qo=tie_qo, share_qk_norm=share_qk_norm)
        self.mlp = Qwen3MLP(d_model, ff, use_swiglu=use_swiglu, tie_gate=tie_gate, activation=activation, mlp_bias=mlp_bias)

    def forward(self, x, mask=None):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.mlp(self.ln2(x))
        return x


# ── Full Model ───────────────────────────────────────────────────────────────

class Qwen3AdditionModel(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 2, n_kv_heads: int = 1,
                 head_dim: int = 4, ff: int = 6, rope_theta: float = 3.0,
                 max_len: int = TOTAL_LEN + 1, qk_norm: bool = True,
                 use_swiglu: bool = True, tie_kv: bool = False,
                 tie_qo: bool = False, tie_gate: bool = False, repeats: int = 1,
                 share_norms: bool = False, share_block_norms: bool = False,
                 activation: str = "default", window_size: int = 0,
                 share_qk_norm: bool = False, mlp_bias: bool = False):
        super().__init__()
        self.d_model = d_model
        self.repeats = repeats

        # Tied embedding
        self.embed = nn.Embedding(VOCAB_SIZE, d_model)
        self.lm_head_weight = self.embed.weight  # tied

        # RoPE
        rope_cos, rope_sin = precompute_rope_freqs(head_dim, max_len, rope_theta)

        # Shared norm (optional)
        # share_norms: all 3 norms (ln1, ln2, final) share one RMSNorm
        # share_block_norms: ln1 and ln2 share, final_norm is separate
        if share_norms:
            shared_norm = RMSNorm(d_model)
        elif share_block_norms:
            shared_norm = RMSNorm(d_model)
        else:
            shared_norm = None

        # Single transformer layer (applied `repeats` times with shared weights)
        self.block = Qwen3Block(d_model, n_heads, n_kv_heads, head_dim, ff,
                                rope_cos, rope_sin, qk_norm=qk_norm,
                                use_swiglu=use_swiglu, tie_kv=tie_kv,
                                tie_qo=tie_qo, tie_gate=tie_gate,
                                shared_norm=shared_norm, activation=activation,
                                share_qk_norm=share_qk_norm, mlp_bias=mlp_bias)
        self.final_norm = shared_norm if share_norms else RMSNorm(d_model)

        # Causal mask (with optional sliding window)
        mask = create_causal_mask(max_len, window_size)
        self.register_buffer("causal_mask", mask, persistent=False)

        # Init weights
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
        """
        input_ids: [B, T] token IDs
        Returns: logits [B, T, VOCAB_SIZE]
        """
        x = self.embed(input_ids)
        for _ in range(self.repeats):
            x = self.block(x, self.causal_mask)
        x = self.final_norm(x)
        logits = F.linear(x, self.lm_head_weight)  # tied
        return logits
