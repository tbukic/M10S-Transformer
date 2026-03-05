"""AdderBoard submission: M10S-58p (58 parameters).

1-layer decoder-only transformer with circular arc embedding + monkey-patched attention and MLP tying.
d=3, 1h/1kv, hd=4, ff=2, RoPE theta=3, SwiGLU.
K=alpha*Q (scalar replaces key projection), gate=alpha*up (scalar replaces gate projection).
Tied Q=O projections.
"""

from pathlib import Path

import torch

from experiments.tying_search import build_model as _build_model
from minimal10digittransformer.model.qwen3 import OUTPUT_LEN
from minimal10digittransformer.data.addition import encode

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_CHECKPOINT = _PROJECT_ROOT / "checkpoints" / "qwen3_arc_58p_tying_s1_targeted" / "best.pt"

METADATA = {
    "name": "M10S-58p",
    "author": "Tom Bukic",
    "params": 58,
    "architecture": "1L decoder-only transformer + circular arc embed, d=3, 1h/1kv, hd=4, ff=2, K=aQ, gate=a*up, tieQO",
    "tricks": [
        "Circular arc embedding (3 params instead of 30)",
        "K = alpha * Q (scalar replaces 12-param key projection)",
        "gate = alpha * up (scalar replaces 6-param gate projection)",
        "Tied Q=O projections (output = Q transposed)",
        "RoPE (zero params)",
        "QK norms",
        "Grokfast-EMA (alpha=0.98, lambda=2.0)",
        "Iterated targeted fine-tuning (1 iteration, 9 error pairs)",
    ],
}


def build_model():
    """Load checkpoint and return (model, metadata)."""
    device = torch.device("cpu")
    ckpt = torch.load(str(_CHECKPOINT), map_location=device, weights_only=False)
    cfg = ckpt["config"]

    model = _build_model(cfg, device)
    model.load_state_dict(ckpt["state_dict"])
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
