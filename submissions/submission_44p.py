"""AdderBoard submission: M10S-44p (44 parameters).

1-layer decoder-only transformer with circular arc embedding + K=Q + V=Q + tieQO + shared norms.
d=3, 1h/1kv, hd=4, ff=2, RoPE theta=3, SwiGLU, RMSNorm, tied embed.
"""

import sys
import os
from pathlib import Path

import torch

# Need to import obsolete module for KeqQ build
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))
import sub50_sweep_obsolete as sweep
_ORIG_CONFIGS = list(sweep.CONFIGS)
_orig_build = sweep.build_model

from minimal10digittransformer.model.qwen3 import OUTPUT_LEN
from minimal10digittransformer.data.addition import encode

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_CHECKPOINT = _PROJECT_ROOT / "checkpoints" / "qwen3_arc_44p_KeqQ_s27" / "best.pt"

METADATA = {
    "name": "M10S-44p",
    "author": "Tom Bukic",
    "params": 44,
    "architecture": "1L decoder-only transformer + circular arc embed, d=3, 1h/1kv, hd=4, ff=2, K=Q, V=Q, tieQO, shnorm",
    "tricks": [
        "Circular arc embedding (3 params instead of 30)",
        "K = Q (key shares query weights, 0 extra params)",
        "V = Q (value shares query weights, 0 extra params)",
        "Tied Q=O projections (output = Q transposed)",
        "All 3 RMSNorms shared (-6 params)",
        "RoPE (zero params)",
        "QK norms",
        "Curriculum + Grokfast-EMA + adaptive weight decay",
    ],
}


def build_model():
    """Load checkpoint and return (model, metadata)."""
    device = torch.device("cpu")
    cfg_name = "44p_ff2_tieQO_KeqQ_shnorm_VeqQ"
    cfg = [c for c in _ORIG_CONFIGS if c["name"] == cfg_name][0]
    model, n_params = _orig_build(cfg, device)

    ckpt = torch.load(str(_CHECKPOINT), map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, METADATA


def add(model, a: int, b: int) -> int:
    """Add two integers using the model."""
    device = next(model.parameters()).device
    inp = torch.tensor([encode(a, b)], dtype=torch.long, device=device)

    with torch.no_grad():
        x = inp
        digits = []
        for _ in range(OUTPUT_LEN):
            logits = model(x)
            next_tok = logits[0, -1, :].argmax().item()
            digits.append(next_tok)
            x = torch.cat([x, torch.tensor([[next_tok]], device=device)], dim=1)

    result = sum(d * (10 ** i) for i, d in enumerate(digits))
    return result
