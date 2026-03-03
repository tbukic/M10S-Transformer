"""Transformer model for addition."""

from minimal10digittransformer.model.qwen3 import (
    Qwen3AdditionModel,
    Qwen3Attention,
    Qwen3Block,
    Qwen3MLP,
    RMSNorm,
    precompute_rope_freqs,
    apply_rope,
    NUM_DIGITS,
    SUM_DIGITS,
    MAX_ADDEND,
    VOCAB_SIZE,
    INPUT_LEN,
    OUTPUT_LEN,
    TOTAL_LEN,
)
from minimal10digittransformer.model.circular_arc import CircularArcQwen3
from minimal10digittransformer.model.rank1_out import (
    Rank1OutModel,
    Rank1OutAttention,
    Rank1OutBlock,
)

__all__ = [
    "Qwen3AdditionModel",
    "Qwen3Attention",
    "Qwen3Block",
    "Qwen3MLP",
    "RMSNorm",
    "precompute_rope_freqs",
    "apply_rope",
    "NUM_DIGITS",
    "SUM_DIGITS",
    "MAX_ADDEND",
    "VOCAB_SIZE",
    "INPUT_LEN",
    "OUTPUT_LEN",
    "TOTAL_LEN",
    "CircularArcQwen3",
    "Rank1OutModel",
    "Rank1OutAttention",
    "Rank1OutBlock",
]
