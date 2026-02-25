"""Minimal transformer architecture for 10-digit addition.

Design principles:
- Every parameter must earn its place
- Support for extreme parameter reduction techniques
- Configurable: causal/non-causal, various PE types, layer sharing
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TransformerConfig:
    """Configuration for the minimal transformer."""

    # Vocabulary and sequence
    vocab_size: int = 15  # 0-9, +, =, pad, bos, eos
    max_seq_len: int = 33  # 10 + 1 + 10 + 1 + 11 = 33 for 10-digit addition

    # Architecture
    d_model: int = 8  # Hidden dimension
    n_heads: int = 1  # Number of attention heads
    n_layers: int = 1  # Number of transformer layers
    d_ff: int = 16  # FFN intermediate dimension
    dropout: float = 0.0  # Dropout rate

    # Attention
    causal: bool = True  # Causal masking
    use_attention: bool = True  # Can disable attention for ablation

    # Positional encoding
    pe_type: str = "sinusoidal"  # sinusoidal, learned, abacus, none, custom_period
    pe_period: float = 11.0  # Period for custom_period PE
    pe_shared: bool = False  # Share PE params with embedding

    # Embeddings
    embed_dim: int | None = None  # If set, use factorized embedding (rank reduction)
    tie_weights: bool = True  # Tie input/output embeddings

    # Layer sharing
    share_layers: bool = False  # Share weights across all layers
    n_layer_repeats: int = 1  # How many times to repeat the shared layer

    # FFN
    ffn_type: str = "standard"  # standard, glu, none
    use_bias: bool = False  # Use bias in linear layers

    # Normalization
    norm_type: str = "rmsnorm"  # layernorm, rmsnorm, none
    pre_norm: bool = True  # Pre-norm vs post-norm

    # Low-rank
    rank: int | None = None  # If set, use low-rank factorization for all weight matrices

    # Skip connections
    use_residual: bool = True  # Use residual connections

    # Activation
    activation: str = "relu"  # relu, gelu, silu, relu2, sigmoid, tanh

    # Output
    output_length: int = 11  # Number of output tokens to predict


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class LowRankLinear(nn.Module):
    """Low-rank linear layer: W = A @ B where A is (out, rank), B is (rank, in)."""

    def __init__(self, in_features: int, out_features: int, rank: int, bias: bool = False):
        super().__init__()
        self.A = nn.Parameter(torch.randn(out_features, rank) * 0.01)
        self.B = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.A @ self.B)
        if self.bias is not None:
            out = out + self.bias
        return out


def make_linear(
    in_features: int, out_features: int, rank: int | None = None, bias: bool = False
) -> nn.Module:
    """Create a linear layer, optionally low-rank."""
    if rank is not None and rank < min(in_features, out_features):
        return LowRankLinear(in_features, out_features, rank, bias)
    return nn.Linear(in_features, out_features, bias=bias)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


def get_activation(name: str):
    """Get activation function by name."""
    activations = {
        "relu": F.relu,
        "gelu": F.gelu,
        "silu": F.silu,
        "relu2": lambda x: F.relu(x) ** 2,
        "sigmoid": torch.sigmoid,
        "tanh": torch.tanh,
    }
    return activations[name]


def get_norm(norm_type: str, dim: int) -> nn.Module:
    """Get normalization layer."""
    if norm_type == "layernorm":
        return nn.LayerNorm(dim)
    elif norm_type == "rmsnorm":
        return RMSNorm(dim)
    elif norm_type == "none":
        return nn.Identity()
    raise ValueError(f"Unknown norm type: {norm_type}")


class SinusoidalPE(nn.Module):
    """Sinusoidal positional encoding with configurable period."""

    def __init__(self, d_model: int, max_len: int, period: float = 10000.0):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(period) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model > 1:
            pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.size(1)]


class AbacusPE(nn.Module):
    """Abacus positional encoding - encodes digit position within number.

    Each position gets an embedding based on which digit column it represents
    (ones, tens, hundreds, etc.) rather than absolute position.
    """

    def __init__(self, d_model: int, max_digits: int = 10, max_seq_len: int = 33):
        super().__init__()
        self.d_model = d_model
        self.max_digits = max_digits
        # Learnable embeddings for each digit position
        self.digit_pos_embed = nn.Embedding(max_digits + 2, d_model)  # +2 for special positions

        # Build position-to-digit-column mapping
        # For "AAAAAAAAAA+BBBBBBBBBB=CCCCCCCCCCC"
        # A positions map to digit columns 9,8,...,0
        # B positions map to digit columns 9,8,...,0
        # C positions map to digit columns 10,9,...,0
        mapping = []
        for i in range(max_digits):
            mapping.append(max_digits - 1 - i)  # A digits
        mapping.append(max_digits)  # + sign -> special position
        for i in range(max_digits):
            mapping.append(max_digits - 1 - i)  # B digits
        mapping.append(max_digits + 1)  # = sign -> special position
        for i in range(max_digits + 1):
            mapping.append(max_digits - i)  # C digits (result)

        self.register_buffer("pos_mapping", torch.tensor(mapping[:max_seq_len], dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        pos_ids = self.pos_mapping[:seq_len]
        return x + self.digit_pos_embed(pos_ids)


class Attention(nn.Module):
    """Multi-head attention with optional low-rank projections."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_model = config.d_model
        self.head_dim = config.d_model // config.n_heads
        self.causal = config.causal

        self.q_proj = make_linear(config.d_model, config.d_model, config.rank, config.use_bias)
        self.k_proj = make_linear(config.d_model, config.d_model, config.rank, config.use_bias)
        self.v_proj = make_linear(config.d_model, config.d_model, config.rank, config.use_bias)
        self.o_proj = make_linear(config.d_model, config.d_model, config.rank, config.use_bias)

        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scale = self.head_dim**-0.5
        attn = (q @ k.transpose(-2, -1)) * scale

        if self.causal:
            mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
            attn = attn.masked_fill(mask, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(out)


class FFN(nn.Module):
    """Feed-forward network with configurable type."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.ffn_type = config.ffn_type
        act = get_activation(config.activation)
        self.act = act

        if config.ffn_type == "standard":
            self.up = make_linear(config.d_model, config.d_ff, config.rank, config.use_bias)
            self.down = make_linear(config.d_ff, config.d_model, config.rank, config.use_bias)
        elif config.ffn_type == "glu":
            self.gate = make_linear(config.d_model, config.d_ff, config.rank, config.use_bias)
            self.up = make_linear(config.d_model, config.d_ff, config.rank, config.use_bias)
            self.down = make_linear(config.d_ff, config.d_model, config.rank, config.use_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.ffn_type == "standard":
            return self.down(self.act(self.up(x)))
        elif self.ffn_type == "glu":
            return self.down(self.act(self.gate(x)) * self.up(x))
        return x  # ffn_type == "none"


class TransformerBlock(nn.Module):
    """Single transformer block."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        if config.use_attention:
            self.attn = Attention(config)
            self.norm1 = get_norm(config.norm_type, config.d_model)

        if config.ffn_type != "none":
            self.ffn = FFN(config)
            self.norm2 = get_norm(config.norm_type, config.d_model)

        self.use_residual = config.use_residual
        self.pre_norm = config.pre_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention
        if hasattr(self, "attn"):
            if self.pre_norm:
                attn_out = self.attn(self.norm1(x))
            else:
                attn_out = self.norm1(self.attn(x))

            if self.use_residual:
                x = x + attn_out
            else:
                x = attn_out

        # FFN
        if hasattr(self, "ffn"):
            if self.pre_norm:
                ffn_out = self.ffn(self.norm2(x))
            else:
                ffn_out = self.norm2(self.ffn(x))

            if self.use_residual:
                x = x + ffn_out
            else:
                x = ffn_out

        return x


class MinimalTransformer(nn.Module):
    """Minimal transformer for 10-digit addition.

    Supports:
    - Factorized embeddings (rank reduction)
    - Weight tying
    - Low-rank attention
    - Layer sharing
    - Configurable PE, FFN, normalization
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        # Token embedding
        if config.embed_dim is not None:
            # Factorized embedding: vocab -> embed_dim -> d_model
            self.tok_embed_low = nn.Embedding(config.vocab_size, config.embed_dim)
            self.tok_embed_proj = nn.Linear(config.embed_dim, config.d_model, bias=False)
        else:
            self.tok_embed = nn.Embedding(config.vocab_size, config.d_model)

        # Positional encoding
        if config.pe_type == "sinusoidal":
            self.pos_enc = SinusoidalPE(config.d_model, config.max_seq_len)
        elif config.pe_type == "learned":
            self.pos_enc = nn.Embedding(config.max_seq_len, config.d_model)
        elif config.pe_type == "abacus":
            self.pos_enc = AbacusPE(config.d_model, max_digits=10, max_seq_len=config.max_seq_len)
        elif config.pe_type == "custom_period":
            self.pos_enc = SinusoidalPE(config.d_model, config.max_seq_len, period=config.pe_period)
        elif config.pe_type == "none":
            self.pos_enc = None
        else:
            raise ValueError(f"Unknown PE type: {config.pe_type}")

        # Transformer layers
        if config.share_layers:
            self.shared_layer = TransformerBlock(config)
            self.n_repeats = config.n_layer_repeats if config.n_layer_repeats > 0 else config.n_layers
        else:
            self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])

        # Output head
        self.output_norm = get_norm(config.norm_type, config.d_model)
        if config.embed_dim is not None and config.tie_weights:
            self.output_proj = nn.Linear(config.d_model, config.embed_dim, bias=False)
            # Output logits = output_proj @ tok_embed_low.weight.T
        elif config.tie_weights and config.embed_dim is None:
            pass  # Will use tok_embed.weight for output
        else:
            self.output_head = make_linear(
                config.d_model, config.vocab_size, config.rank, config.use_bias
            )

    def get_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get token + positional embeddings."""
        if hasattr(self, "tok_embed_low"):
            x = self.tok_embed_proj(self.tok_embed_low(input_ids))
        else:
            x = self.tok_embed(input_ids)

        if self.pos_enc is not None:
            if isinstance(self.pos_enc, nn.Embedding):
                positions = torch.arange(input_ids.size(1), device=input_ids.device)
                x = x + self.pos_enc(positions)
            else:
                x = self.pos_enc(x)

        return x

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            input_ids: (batch, seq_len) token IDs

        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        x = self.get_embeddings(input_ids)

        # Apply transformer layers
        if self.config.share_layers:
            for _ in range(self.n_repeats):
                x = self.shared_layer(x)
        else:
            for layer in self.layers:
                x = layer(x)

        # Output projection
        x = self.output_norm(x)

        if hasattr(self, "output_head"):
            logits = self.output_head(x)
        elif self.config.embed_dim is not None and self.config.tie_weights:
            x = self.output_proj(x)
            logits = F.linear(x, self.tok_embed_low.weight)
        elif self.config.tie_weights:
            logits = F.linear(x, self.tok_embed.weight)
        else:
            logits = self.output_head(x)

        return logits

    def count_params(self) -> int:
        """Count trainable parameters."""
        return count_parameters(self)
