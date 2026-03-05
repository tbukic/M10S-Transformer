"""AdderBoard submission: M10S-95p (95 parameters).

1-layer decoder-only transformer with circular arc embedding, d=3, 1h/1kv, hd=4, ff=3, RoPE theta=3, SwiGLU.
Circular arc embedding replaces 30-param lookup table with 3 learnable params.
"""

from pathlib import Path

import torch

from minimal10digittransformer.model.circular_arc import CircularArcQwen3
from minimal10digittransformer.model.qwen3 import OUTPUT_LEN
from minimal10digittransformer.data.addition import encode

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_CHECKPOINT = _PROJECT_ROOT / "checkpoints" / "qwen3_arc_95p_ft_s9999" / "best.pt"

METADATA = {
    "name": "M10S-95p",
    "author": "Tom Bukic",
    "params": 95,
    "architecture": "1L decoder-only transformer + circular arc embedding, d=3, 1h/1kv, hd=4, ff=3, RoPE theta=3, SwiGLU",
    "tricks": [
        "Circular arc embedding (3 params instead of 30)",
        "Tied lm_head to dynamic embedding table",
        "RoPE (zero params)",
        "QK norms",
        "Cosine LR + fine-tuning",
    ],
}


def build_model():
    """Load checkpoint and return (model, metadata)."""
    device = torch.device("cpu")
    ckpt = torch.load(str(_CHECKPOINT), map_location=device, weights_only=True)
    cfg = ckpt["config"]

    model = CircularArcQwen3(
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_kv_heads=cfg["n_kv_heads"],
        head_dim=cfg["head_dim"],
        ff=cfg["ff"],
        rope_theta=cfg["rope_theta"],
        qk_norm=True,
        use_swiglu=True,
        tie_kv=cfg.get("tie_kv", False),
        tie_qo=cfg.get("tie_qo", False),
        tie_gate=cfg.get("tie_gate", False),
        repeats=cfg.get("repeats", 1),
        share_norms=cfg.get("share_norms", False),
        share_block_norms=cfg.get("share_block_norms", False),
    ).to(device)
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
