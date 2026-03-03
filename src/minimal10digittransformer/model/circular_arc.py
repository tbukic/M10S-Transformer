"""Qwen3 model with circular arc token embedding.

Replaces the 30-param nn.Embedding(10, 3) lookup table with a 3-param circular
arc parametrization:
    emb[d] = [A*cos(start + d*stride), A*sin(start + d*stride), 0]

This saves 27 params. Uses the standard Qwen3Block for attention and MLP.
The lm_head is tied to the dynamically computed embedding table.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from minimal10digittransformer.model.qwen3 import (
    Qwen3Block,
    RMSNorm,
    precompute_rope_freqs,
    VOCAB_SIZE,
    TOTAL_LEN,
)


class CircularArcQwen3(nn.Module):
    """Qwen3 model with circular arc token embedding instead of lookup table.

    Token embedding: emb[d] = [A*cos(start + d*stride), A*sin(start + d*stride), 0]
    Only 3 learnable params for the embedding instead of 30.
    The lm_head is tied to the dynamically computed embedding table.
    """

    def __init__(self, d_model: int, n_heads: int = 1, n_kv_heads: int = 1,
                 head_dim: int = 4, ff: int = 3, rope_theta: float = 3.0,
                 max_len: int = TOTAL_LEN + 1, qk_norm: bool = True,
                 use_swiglu: bool = True, tie_kv: bool = False,
                 tie_qo: bool = False, tie_gate: bool = False, repeats: int = 1,
                 share_norms: bool = False, share_block_norms: bool = False,
                 arc_init_A: float = 2.5, arc_init_start: float = -1.2,
                 arc_init_stride: float = 0.29):
        super().__init__()
        self.d_model = d_model
        self.repeats = repeats

        # Circular arc embedding params (3 params, replaces 30-param nn.Embedding)
        self.arc_A = nn.Parameter(torch.tensor(arc_init_A))
        self.arc_start = nn.Parameter(torch.tensor(arc_init_start))
        self.arc_stride = nn.Parameter(torch.tensor(arc_init_stride))

        # RoPE
        rope_cos, rope_sin = precompute_rope_freqs(head_dim, max_len, rope_theta)

        # Shared norm (optional)
        if share_norms:
            shared_norm = RMSNorm(d_model)
        elif share_block_norms:
            shared_norm = RMSNorm(d_model)
        else:
            shared_norm = None

        # Transformer block (same as standard Qwen3)
        self.block = Qwen3Block(d_model, n_heads, n_kv_heads, head_dim, ff,
                                rope_cos, rope_sin, qk_norm=qk_norm,
                                use_swiglu=use_swiglu, tie_kv=tie_kv,
                                tie_qo=tie_qo, tie_gate=tie_gate,
                                shared_norm=shared_norm)
        self.final_norm = shared_norm if share_norms else RMSNorm(d_model)

        # Causal mask
        mask = torch.full((max_len, max_len), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        self.register_buffer("causal_mask", mask, persistent=False)

        # Init weights (skip embedding since we use arc params)
        self.apply(self._init_weights)

    def _compute_embedding_table(self) -> torch.Tensor:
        """Compute token embedding table from arc parameters.

        Returns (VOCAB_SIZE, d_model) tensor.
        For d_model=3: [A*cos(start + d*stride), A*sin(start + d*stride), 0]
        """
        d = torch.arange(VOCAB_SIZE, device=self.arc_A.device, dtype=self.arc_A.dtype)
        angles = self.arc_start + d * self.arc_stride

        if self.d_model == 2:
            return torch.stack([
                self.arc_A * torch.cos(angles),
                self.arc_A * torch.sin(angles),
            ], dim=1)
        elif self.d_model == 3:
            return torch.stack([
                self.arc_A * torch.cos(angles),
                self.arc_A * torch.sin(angles),
                torch.zeros_like(angles),
            ], dim=1)
        else:
            # For d_model > 3, pad with zeros
            emb = torch.zeros(VOCAB_SIZE, self.d_model, device=self.arc_A.device,
                              dtype=self.arc_A.dtype)
            emb[:, 0] = self.arc_A * torch.cos(angles)
            emb[:, 1] = self.arc_A * torch.sin(angles)
            return emb

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        input_ids: [B, T] token IDs
        Returns: logits [B, T, VOCAB_SIZE]
        """
        emb_table = self._compute_embedding_table()  # (10, d_model)
        x = emb_table[input_ids]  # (B, T, d_model)

        for _ in range(self.repeats):
            x = self.block(x, self.causal_mask)
        x = self.final_norm(x)

        # Tied lm_head: logits = x @ emb_table.T
        logits = F.linear(x, emb_table)
        return logits
