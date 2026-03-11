"""AdderBoard submission: M10S-36p (36 parameters).

1-layer decoder-only transformer with circular arc embedding + K=rotation(Q) + V=Q
+ tieQO + shared norms + tied QK norms + down=rotation(up^T).
d=3, 1h/1kv, hd=4, ff=2, RoPE theta=3, SwiGLU, RMSNorm, tied embed.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))
import sub50_sweep as sweep

from minimal10digittransformer.model.qwen3 import OUTPUT_LEN
from minimal10digittransformer.data.addition import encode

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_CHECKPOINT = _PROJECT_ROOT / "checkpoints" / "qwen3_arc_36p_KrotQ_downRotUpT_s8_targeted" / "best.pt"

METADATA = {
    "name": "M10S-36p",
    "author": "Tom Bukic",
    "params": 36,
    "architecture": "1L decoder-only transformer + circular arc embed, d=3, 1h/1kv, hd=4, ff=2, K=rot(Q), V=Q, tieQO, shnorm, tieQKnorm, down=rot(up^T)",
    "tricks": [
        "Circular arc embedding (3 params instead of 30)",
        "K = rotation(Q) (1-param 2D rotation replaces 12-param key projection)",
        "V = Q (value shares query weights, 0 extra params)",
        "Tied Q=O projections (output = Q transposed)",
        "All 3 RMSNorms shared (ln1=ln2=final_norm, saves 6 params)",
        "Tied QK norms (q_norm and k_norm share weights, saves 4 params)",
        "down = rotation(up^T) (1-param rotation + transpose replaces 6-param down projection)",
        "RoPE (zero params)",
        "SwiGLU MLP at ff=2",
        "Grokfast-EMA gradient filter (alpha=0.98, lambda=3.0)",
        "Iterated targeted fine-tuning (2 rounds, 70 cumulative error pairs, lr=1e-4)",
    ],
}


def build_model():
    """Load checkpoint and return (model, metadata)."""
    device = torch.device("cpu")
    cfg_name = "36p_ff2_tieQO_KrotQ_shnorm_VeqQ_downRotUpT_tieQKnorm"
    cfg = [c for c in sweep.CONFIGS if c["name"] == cfg_name][0]
    model, n_params = sweep.build_model(cfg, device)

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
