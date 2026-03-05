"""AdderBoard submission: M10S-101p (101 parameters).

1-layer decoder-only transformer, d=3, 1h/1kv, hd=4, ff=2, RoPE theta=3, SwiGLU.
Tied O=Q^T, targeted fine-tuning.
"""

from pathlib import Path

import torch

from minimal10digittransformer.model.qwen3 import Qwen3AdditionModel, OUTPUT_LEN
from minimal10digittransformer.data.addition import encode

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_CHECKPOINT = _PROJECT_ROOT / "checkpoints" / "qwen3_d3_ff2_101p_tieqo_s13_targeted" / "best.pt"

METADATA = {
    "name": "M10S-101p",
    "author": "Tom Bukic",
    "params": 101,
    "architecture": "1L decoder-only transformer, d=3, 1h/1kv, hd=4, ff=2, RoPE theta=3, SwiGLU",
    "tricks": [
        "Tied embeddings",
        "Tied O=Q^T",
        "RoPE (zero params)",
        "QK norms",
        "Targeted fine-tuning",
    ],
}


def build_model():
    """Load checkpoint and return (model, metadata)."""
    device = torch.device("cpu")
    ckpt = torch.load(str(_CHECKPOINT), map_location=device, weights_only=True)
    cfg = ckpt["config"]

    model = Qwen3AdditionModel(
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_kv_heads=cfg["n_kv_heads"],
        head_dim=cfg["head_dim"],
        ff=cfg["ff"],
        rope_theta=cfg["rope_theta"],
        qk_norm=not cfg.get("no_qk_norm", False),
        use_swiglu=not cfg.get("gelu", False),
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
