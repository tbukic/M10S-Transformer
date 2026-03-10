"""Sub-50p architecture sweep using declarative tie_groups.

Defines model configs with tie_groups lists (no monkey-patching).
Training loop and algorithms are imported from sub50_sweep_obsolete.

Usage:
    python experiments/sub50_sweep.py                           # run all on CPU
    python experiments/sub50_sweep.py --device cuda             # run all on GPU
    python experiments/sub50_sweep.py --config 44p --algo adamw  # single config+algo
    python experiments/sub50_sweep.py --summary                 # print summary
"""

import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_SCRIPT_DIR, "..", "src"))
sys.path.insert(0, _SCRIPT_DIR)

from minimal10digittransformer.model.circular_arc import CircularArcQwen3
from minimal10digittransformer.model.tie_groups import apply_tie_groups, count_unique_params

# Import training infrastructure from the old module (training loop, algorithms,
# data generation, LR schedules, multiprocessing, summary reporting).
import sub50_sweep_obsolete as _impl

ALGORITHMS = _impl.ALGORITHMS
SEEDS = _impl.SEEDS
SCRIPT_DIR = _impl.SCRIPT_DIR
RESULTS_DIR = _impl.RESULTS_DIR


# ============================================================================
# Architecture configurations — declarative tie_groups
# ============================================================================

BASE_ARGS = {
    "d_model": 3,
    "n_heads": 1,
    "n_kv_heads": 1,
    "head_dim": 4,
    "rope_theta": 3.0,
}

BASE_ARGS_D2 = {
    "d_model": 2,
    "n_heads": 1,
    "n_kv_heads": 1,
    "head_dim": 4,
    "rope_theta": 3.0,
}

BASE_ARGS_D2_HD2 = {
    "d_model": 2,
    "n_heads": 1,
    "n_kv_heads": 1,
    "head_dim": 2,
    "rope_theta": 3.0,
}

# Shorthand helpers for common tie group patterns
_KaQ = "scalar:block.attn.k_proj"       # K = alpha * Q (1 param)
_KrotQ = "rotation:block.attn.k_proj"   # K = rotate(Q) (1 param)
_KeqQ = "block.attn.k_proj"             # K = Q (0 params)
_VeqQ = "block.attn.v_proj"             # V = Q (0 params)
_gateA = "scalar:block.mlp.gate_proj"   # gate = alpha * up (1 param)
_gateEqUp = "block.mlp.gate_proj"       # gate = up (0 params)
_downUpT = "transpose:block.mlp.down_proj"  # down = up^T (0 params)
_downRotUpT = "rottranspose:block.mlp.down_proj"  # down = up^T ∘ R(θ) (1 param)
_gateDownT = "transpose:block.mlp.gate_proj"  # gate = down^T (0 params)

def _attn(*followers):
    """Tie group with q_proj as master."""
    return ["block.attn.q_proj"] + list(followers)

def _mlp(*followers):
    """Tie group with up_proj as master."""
    return ["block.mlp.up_proj"] + list(followers)

def _mlp_down(*followers):
    """Tie group with down_proj as master."""
    return ["block.mlp.down_proj"] + list(followers)

_TIEQK = ["block.attn.q_norm", "block.attn.k_norm"]  # share QK norms


CONFIGS = [
    # ── ff=2 configs (more MLP capacity) ──
    {
        "name": "52p_ff2_tieQO_KaQ_gateA_shnorm",
        "params": 52,
        "model_args": {**BASE_ARGS, "ff": 2, "tie_qo": True, "share_norms": True},
        "tie_groups": [_attn(_KaQ), _mlp(_gateA)],
    },
    {
        "name": "40p_ff2_tieQO_KaQ_gateA_shnorm_VeqQ",
        "params": 40,
        "model_args": {**BASE_ARGS, "ff": 2, "tie_qo": True, "share_norms": True},
        "tie_groups": [_attn(_KaQ, _VeqQ), _mlp(_gateA)],
    },
    {
        "name": "57p_ff2_tieQO_KaQ_shnorm",
        "params": 57,
        "model_args": {**BASE_ARGS, "ff": 2, "tie_qo": True, "share_norms": True},
        "tie_groups": [_attn(_KaQ)],
    },
    {
        "name": "45p_ff2_tieQO_KaQ_shnorm_VeqQ",
        "params": 45,
        "model_args": {**BASE_ARGS, "ff": 2, "tie_qo": True, "share_norms": True},
        "tie_groups": [_attn(_KaQ, _VeqQ)],
    },

    # ── ff=2 with shbnorm ──
    {
        "name": "43p_ff2_tieQO_KaQ_gateA_shbnorm_VeqQ",
        "params": 43,
        "model_args": {**BASE_ARGS, "ff": 2, "tie_qo": True, "share_block_norms": True},
        "tie_groups": [_attn(_KaQ, _VeqQ), _mlp(_gateA)],
    },
    {
        "name": "48p_ff2_tieQO_KaQ_shbnorm_VeqQ",
        "params": 48,
        "model_args": {**BASE_ARGS, "ff": 2, "tie_qo": True, "share_block_norms": True},
        "tie_groups": [_attn(_KaQ, _VeqQ)],
    },

    # ── ff=1 configs ──
    {
        "name": "49p_ff1_tieQO_KaQ_gateA_shbnorm",
        "params": 49,
        "model_args": {**BASE_ARGS, "ff": 1, "tie_qo": True, "share_block_norms": True},
        "tie_groups": [_attn(_KaQ), _mlp(_gateA)],
    },
    {
        "name": "46p_ff1_tieQO_KaQ_gateA_shnorm",
        "params": 46,
        "model_args": {**BASE_ARGS, "ff": 1, "tie_qo": True, "share_norms": True},
        "tie_groups": [_attn(_KaQ), _mlp(_gateA)],
    },
    {
        "name": "34p_ff1_tieQO_KaQ_gateA_shnorm_VeqQ",
        "params": 34,
        "model_args": {**BASE_ARGS, "ff": 1, "tie_qo": True, "share_norms": True},
        "tie_groups": [_attn(_KaQ, _VeqQ), _mlp(_gateA)],
    },

    # ── ff=1 without gate tying ──
    {
        "name": "48p_ff1_tieQO_KaQ_shnorm",
        "params": 48,
        "model_args": {**BASE_ARGS, "ff": 1, "tie_qo": True, "share_norms": True},
        "tie_groups": [_attn(_KaQ)],
    },
    {
        "name": "36p_ff1_tieQO_KaQ_shnorm_VeqQ",
        "params": 36,
        "model_args": {**BASE_ARGS, "ff": 1, "tie_qo": True, "share_norms": True},
        "tie_groups": [_attn(_KaQ, _VeqQ)],
    },

    # ── ff=1 with shbnorm ──
    {
        "name": "39p_ff1_tieQO_KaQ_shbnorm_VeqQ",
        "params": 39,
        "model_args": {**BASE_ARGS, "ff": 1, "tie_qo": True, "share_block_norms": True},
        "tie_groups": [_attn(_KaQ, _VeqQ)],
    },
    {
        "name": "37p_ff1_tieQO_KaQ_gateA_shbnorm_VeqQ",
        "params": 37,
        "model_args": {**BASE_ARGS, "ff": 1, "tie_qo": True, "share_block_norms": True},
        "tie_groups": [_attn(_KaQ, _VeqQ), _mlp(_gateA)],
    },

    # ── ff=1 with down=up^T ──
    {
        "name": "51p_ff1_tieQO_KaQ_downUpT",
        "params": 51,
        "model_args": {**BASE_ARGS, "ff": 1, "tie_qo": True},
        "tie_groups": [_attn(_KaQ), _mlp(_downUpT)],
    },
    {
        "name": "49p_ff1_tieQO_KaQ_gateA_downUpT",
        "params": 49,
        "model_args": {**BASE_ARGS, "ff": 1, "tie_qo": True},
        "tie_groups": [_attn(_KaQ), _mlp(_gateA, _downUpT)],
    },
    {
        "name": "45p_ff1_tieQO_KaQ_shnorm_downUpT",
        "params": 45,
        "model_args": {**BASE_ARGS, "ff": 1, "tie_qo": True, "share_norms": True},
        "tie_groups": [_attn(_KaQ), _mlp(_downUpT)],
    },
    {
        "name": "43p_ff1_tieQO_KaQ_gateA_shnorm_downUpT",
        "params": 43,
        "model_args": {**BASE_ARGS, "ff": 1, "tie_qo": True, "share_norms": True},
        "tie_groups": [_attn(_KaQ), _mlp(_gateA, _downUpT)],
    },
    {
        "name": "39p_ff1_tieQO_KaQ_VeqQ_downUpT",
        "params": 39,
        "model_args": {**BASE_ARGS, "ff": 1, "tie_qo": True},
        "tie_groups": [_attn(_KaQ, _VeqQ), _mlp(_downUpT)],
    },
    {
        "name": "33p_ff1_tieQO_KaQ_shnorm_VeqQ_downUpT",
        "params": 33,
        "model_args": {**BASE_ARGS, "ff": 1, "tie_qo": True, "share_norms": True},
        "tie_groups": [_attn(_KaQ, _VeqQ), _mlp(_downUpT)],
    },
    {
        "name": "31p_ff1_tieQO_KaQ_gateA_shnorm_VeqQ_downUpT",
        "params": 31,
        "model_args": {**BASE_ARGS, "ff": 1, "tie_qo": True, "share_norms": True},
        "tie_groups": [_attn(_KaQ, _VeqQ), _mlp(_gateA, _downUpT)],
    },

    # ── NEW: ff=2 sub-45p variants (from 45p winner) ──

    # 44p: drop k_alpha scalar (K=Q instead of K=aQ)
    {
        "name": "44p_ff2_tieQO_KeqQ_shnorm_VeqQ",
        "params": 44,
        "model_args": {**BASE_ARGS, "ff": 2, "tie_qo": True, "share_norms": True},
        "tie_groups": [_attn(_KeqQ, _VeqQ)],
    },
    # 41p: tie QK norms (q_norm = k_norm, saves 4p)
    {
        "name": "41p_ff2_tieQO_KaQ_shnorm_VeqQ_tieQKnorm",
        "params": 41,
        "model_args": {**BASE_ARGS, "ff": 2, "tie_qo": True, "share_norms": True},
        "tie_groups": [_attn(_KaQ, _VeqQ), _TIEQK],
    },
    # 40p: K=Q + tie QK norms
    {
        "name": "40p_ff2_tieQO_KeqQ_shnorm_VeqQ_tieQKnorm",
        "params": 40,
        "model_args": {**BASE_ARGS, "ff": 2, "tie_qo": True, "share_norms": True},
        "tie_groups": [_attn(_KeqQ, _VeqQ), _TIEQK],
    },
    # 39p: down=up^T (saves 6p from 45p)
    {
        "name": "39p_ff2_tieQO_KaQ_shnorm_VeqQ_downUpT",
        "params": 39,
        "model_args": {**BASE_ARGS, "ff": 2, "tie_qo": True, "share_norms": True},
        "tie_groups": [_attn(_KaQ, _VeqQ), _mlp(_downUpT)],
    },
    # 39p: gate=up identity (saves 6p from 45p)
    {
        "name": "39p_ff2_tieQO_KaQ_shnorm_VeqQ_gateEqUp",
        "params": 39,
        "model_args": {**BASE_ARGS, "ff": 2, "tie_qo": True, "share_norms": True},
        "tie_groups": [_attn(_KaQ, _VeqQ), _mlp(_gateEqUp)],
    },
    # 38p: down=up^T + K=Q
    {
        "name": "38p_ff2_tieQO_KeqQ_shnorm_VeqQ_downUpT",
        "params": 38,
        "model_args": {**BASE_ARGS, "ff": 2, "tie_qo": True, "share_norms": True},
        "tie_groups": [_attn(_KeqQ, _VeqQ), _mlp(_downUpT)],
    },
    # 35p: down=up^T + tie QK norm
    {
        "name": "35p_ff2_tieQO_KaQ_shnorm_VeqQ_downUpT_tieQKnorm",
        "params": 35,
        "model_args": {**BASE_ARGS, "ff": 2, "tie_qo": True, "share_norms": True},
        "tie_groups": [_attn(_KaQ, _VeqQ), _mlp(_downUpT), _TIEQK],
    },
    # 34p: down=up^T + K=Q + tie QK norm
    {
        "name": "34p_ff2_tieQO_KeqQ_shnorm_VeqQ_downUpT_tieQKnorm",
        "params": 34,
        "model_args": {**BASE_ARGS, "ff": 2, "tie_qo": True, "share_norms": True},
        "tie_groups": [_attn(_KeqQ, _VeqQ), _mlp(_downUpT), _TIEQK],
    },
    # 29p: down=up^T + gate=up + tie QK norm (45 - 6 - 6 - 4)
    {
        "name": "29p_ff2_tieQO_KaQ_shnorm_VeqQ_downUpT_gateEqUp_tieQKnorm",
        "params": 29,
        "model_args": {**BASE_ARGS, "ff": 2, "tie_qo": True, "share_norms": True},
        "tie_groups": [_attn(_KaQ, _VeqQ), _mlp(_gateEqUp, _downUpT), _TIEQK],
    },

    # ── MLP tying: all 4 combinations of pairing {up, down, gate} ──

    # gate=up already above (39p_ff2_..._gateEqUp)
    # down=up^T already above (39p_ff2_..._downUpT)

    # 39p: gate=down^T (down_proj is master, gate is transpose follower)
    {
        "name": "39p_ff2_tieQO_KaQ_shnorm_VeqQ_gateDownT",
        "params": 39,
        "model_args": {**BASE_ARGS, "ff": 2, "tie_qo": True, "share_norms": True},
        "tie_groups": [_attn(_KaQ, _VeqQ), _mlp_down(_gateDownT)],
    },
    # 33p: gate=up + down=up^T (all 3 MLP projections share up_proj, standalone)
    {
        "name": "33p_ff2_tieQO_KaQ_shnorm_VeqQ_gateEqUp_downUpT",
        "params": 33,
        "model_args": {**BASE_ARGS, "ff": 2, "tie_qo": True, "share_norms": True},
        "tie_groups": [_attn(_KaQ, _VeqQ), _mlp(_gateEqUp, _downUpT)],
    },

    # ── K=rotation(Q) configs (2D rotation in head_dim space, 1 param) ──

    # 45p: KrotQ baseline (same as KaQ but rotation instead of scalar)
    {
        "name": "45p_ff2_tieQO_KrotQ_shnorm_VeqQ",
        "params": 45,
        "model_args": {**BASE_ARGS, "ff": 2, "tie_qo": True, "share_norms": True},
        "tie_groups": [_attn(_KrotQ, _VeqQ)],
    },
    # 41p: KrotQ + tieQKnorm (our world record config)
    {
        "name": "41p_ff2_tieQO_KrotQ_shnorm_VeqQ_tieQKnorm",
        "params": 41,
        "model_args": {**BASE_ARGS, "ff": 2, "tie_qo": True, "share_norms": True},
        "tie_groups": [_attn(_KrotQ, _VeqQ), _TIEQK],
    },
    # 39p: KrotQ + down=up^T
    {
        "name": "39p_ff2_tieQO_KrotQ_shnorm_VeqQ_downUpT",
        "params": 39,
        "model_args": {**BASE_ARGS, "ff": 2, "tie_qo": True, "share_norms": True},
        "tie_groups": [_attn(_KrotQ, _VeqQ), _mlp(_downUpT)],
    },
    # 35p: KrotQ + down=up^T + tieQKnorm
    {
        "name": "35p_ff2_tieQO_KrotQ_shnorm_VeqQ_downUpT_tieQKnorm",
        "params": 35,
        "model_args": {**BASE_ARGS, "ff": 2, "tie_qo": True, "share_norms": True},
        "tie_groups": [_attn(_KrotQ, _VeqQ), _mlp(_downUpT), _TIEQK],
    },
    # 36p: down=R(θ)·up^T (rotation adds 1 param vs pure transpose)
    {
        "name": "36p_ff2_tieQO_KrotQ_shnorm_VeqQ_downRotUpT_tieQKnorm",
        "params": 36,
        "model_args": {**BASE_ARGS, "ff": 2, "tie_qo": True, "share_norms": True},
        "tie_groups": [_attn(_KrotQ, _VeqQ), _mlp(_downRotUpT), _TIEQK],
    },
    # 34p: fix arc_A=1.0 (2 embedding params instead of 3)
    {
        "name": "34p_ff2_tieQO_KrotQ_shnorm_VeqQ_downUpT_tieQKnorm_fixA",
        "params": 34,
        "model_args": {**BASE_ARGS, "ff": 2, "tie_qo": True, "share_norms": True, "fix_arc_A": 1.0},
        "tie_groups": [_attn(_KrotQ, _VeqQ), _mlp(_downUpT), _TIEQK],
    },

    # ── Gate reparameterisations (log-space and vector) ──

    # 30p log-space gate: exp(log_alpha) * up(x) prevents collapse to zero
    {
        "name": "30p_ff2_tieQO_KrotQ_shnorm_VeqQ_gateLogA_downUpT_tieQKnorm",
        "params": 30,
        "model_args": {**BASE_ARGS, "ff": 2, "tie_qo": True, "share_norms": True},
        "tie_groups": [_attn(_KrotQ, _VeqQ), _mlp("logscalar:block.mlp.gate_proj", _downUpT), _TIEQK],
    },
    # 31p vector gate: per-element alpha_vec (2 elements for ff=2)
    {
        "name": "31p_ff2_tieQO_KrotQ_shnorm_VeqQ_gateVec_downUpT_tieQKnorm",
        "params": 31,
        "model_args": {**BASE_ARGS, "ff": 2, "tie_qo": True, "share_norms": True},
        "tie_groups": [_attn(_KrotQ, _VeqQ), _mlp("vecscalar:block.mlp.gate_proj", _downUpT), _TIEQK],
    },

    # ── ff=1 SwiGLU with KrotQ (minimal MLP) ──

    # 36p: ff=1 + KrotQ + VeqQ + tieQO + shnorm (full gate+up+down at ff=1)
    {
        "name": "36p_ff1_tieQO_KrotQ_shnorm_VeqQ",
        "params": 36,
        "model_args": {**BASE_ARGS, "ff": 1, "tie_qo": True, "share_norms": True},
        "tie_groups": [_attn(_KrotQ, _VeqQ)],
    },
    # 32p: ff=1 + KrotQ + VeqQ + tieQO + shnorm + tieQKnorm
    {
        "name": "32p_ff1_tieQO_KrotQ_shnorm_VeqQ_tieQKnorm",
        "params": 32,
        "model_args": {**BASE_ARGS, "ff": 1, "tie_qo": True, "share_norms": True},
        "tie_groups": [_attn(_KrotQ, _VeqQ), _TIEQK],
    },
    # 27p: ff=1 + KrotQ + gate=a*up + down=up^T + tieQKnorm (near-minimal)
    {
        "name": "27p_ff1_tieQO_KrotQ_shnorm_VeqQ_gateA_downUpT_tieQKnorm",
        "params": 27,
        "model_args": {**BASE_ARGS, "ff": 1, "tie_qo": True, "share_norms": True},
        "tie_groups": [_attn(_KrotQ, _VeqQ), _mlp(_gateA, _downUpT), _TIEQK],
    },

    # ── ff=2 + gate=alpha*up + down=up^T (proven ties combined) ──

    # 30p: ff=2 + KrotQ + gate=a*up + down=up^T + tieQKnorm
    {
        "name": "30p_ff2_tieQO_KrotQ_shnorm_VeqQ_gateA_downUpT_tieQKnorm",
        "params": 30,
        "model_args": {**BASE_ARGS, "ff": 2, "tie_qo": True, "share_norms": True},
        "tie_groups": [_attn(_KrotQ, _VeqQ), _mlp(_gateA, _downUpT), _TIEQK],
    },
    # 34p: ff=2 + KrotQ + gate=a*up + down=up^T (no tieQKnorm)
    {
        "name": "34p_ff2_tieQO_KrotQ_shnorm_VeqQ_gateA_downUpT",
        "params": 34,
        "model_args": {**BASE_ARGS, "ff": 2, "tie_qo": True, "share_norms": True},
        "tie_groups": [_attn(_KrotQ, _VeqQ), _mlp(_gateA, _downUpT)],
    },

    # ── ReLU MLP (no gate_proj, saves ff*d params vs SwiGLU) ──

    {"name": "35p_relu_ff2", "params": 35,
     "model_args": {**BASE_ARGS, "ff": 2, "tie_qo": True, "share_norms": True, "use_swiglu": False, "activation": "relu"},
     "tie_groups": [_attn(_KrotQ, _VeqQ), _TIEQK]},
    {"name": "29p_relu_ff2_tied", "params": 29,
     "model_args": {**BASE_ARGS, "ff": 2, "tie_qo": True, "share_norms": True, "use_swiglu": False, "activation": "relu"},
     "tie_groups": [_attn(_KrotQ, _VeqQ), _mlp(_downUpT), _TIEQK]},
    {"name": "37p_relu_ff2_bias", "params": 37,
     "model_args": {**BASE_ARGS, "ff": 2, "tie_qo": True, "share_norms": True, "use_swiglu": False, "activation": "relu", "mlp_bias": True},
     "tie_groups": [_attn(_KrotQ, _VeqQ), _TIEQK]},
    {"name": "32p_relu_ff1", "params": 32,
     "model_args": {**BASE_ARGS, "ff": 1, "tie_qo": True, "share_block_norms": True, "use_swiglu": False, "activation": "relu"},
     "tie_groups": [_attn(_KrotQ, _VeqQ), _TIEQK]},
    {"name": "26p_relu_ff1_tied", "params": 26,
     "model_args": {**BASE_ARGS, "ff": 1, "tie_qo": True, "share_norms": True, "use_swiglu": False, "activation": "relu"},
     "tie_groups": [_attn(_KrotQ, _VeqQ), _mlp(_downUpT), _TIEQK]},

    # ── d=2 configs ──

    {"name": "30p_d2_swiglu_ff2", "params": 30,
     "model_args": {**BASE_ARGS_D2, "ff": 2, "tie_qo": True, "share_norms": True},
     "tie_groups": [_attn(_KrotQ, _VeqQ), _TIEQK]},
    {"name": "26p_d2_relu_ff2", "params": 26,
     "model_args": {**BASE_ARGS_D2, "ff": 2, "tie_qo": True, "share_norms": True, "use_swiglu": False, "activation": "relu"},
     "tie_groups": [_attn(_KrotQ, _VeqQ), _TIEQK]},
    {"name": "22p_d2_relu_ff2_tied", "params": 22,
     "model_args": {**BASE_ARGS_D2, "ff": 2, "tie_qo": True, "share_norms": True, "use_swiglu": False, "activation": "relu"},
     "tie_groups": [_attn(_KrotQ, _VeqQ), _mlp(_downUpT), _TIEQK]},
    {"name": "24p_d2_relu_ff1", "params": 24,
     "model_args": {**BASE_ARGS_D2, "ff": 1, "tie_qo": True, "share_block_norms": True, "use_swiglu": False, "activation": "relu"},
     "tie_groups": [_attn(_KrotQ, _VeqQ), _TIEQK]},
    {"name": "22p_d2_relu_ff1_tied", "params": 22,
     "model_args": {**BASE_ARGS_D2, "ff": 1, "tie_qo": True, "share_block_norms": True, "use_swiglu": False, "activation": "relu"},
     "tie_groups": [_attn(_KrotQ, _VeqQ), _mlp(_downUpT), _TIEQK]},

    # ── No-rotation variants (K=Q identity, drops 1 param) ──

    # d=3
    {"name": "34p_relu_ff2_norot", "params": 34,
     "model_args": {**BASE_ARGS, "ff": 2, "tie_qo": True, "share_norms": True, "use_swiglu": False, "activation": "relu"},
     "tie_groups": [_attn(_KeqQ, _VeqQ), _TIEQK]},
    {"name": "28p_relu_ff2_tied_norot", "params": 28,
     "model_args": {**BASE_ARGS, "ff": 2, "tie_qo": True, "share_norms": True, "use_swiglu": False, "activation": "relu"},
     "tie_groups": [_attn(_KeqQ, _VeqQ), _mlp(_downUpT), _TIEQK]},
    {"name": "31p_relu_ff1_norot", "params": 31,
     "model_args": {**BASE_ARGS, "ff": 1, "tie_qo": True, "share_block_norms": True, "use_swiglu": False, "activation": "relu"},
     "tie_groups": [_attn(_KeqQ, _VeqQ), _TIEQK]},
    {"name": "25p_relu_ff1_tied_norot", "params": 25,
     "model_args": {**BASE_ARGS, "ff": 1, "tie_qo": True, "share_norms": True, "use_swiglu": False, "activation": "relu"},
     "tie_groups": [_attn(_KeqQ, _VeqQ), _mlp(_downUpT), _TIEQK]},
    # d=2
    {"name": "29p_d2_swiglu_ff2_norot", "params": 29,
     "model_args": {**BASE_ARGS_D2, "ff": 2, "tie_qo": True, "share_norms": True},
     "tie_groups": [_attn(_KeqQ, _VeqQ), _TIEQK]},
    {"name": "25p_d2_relu_ff2_norot", "params": 25,
     "model_args": {**BASE_ARGS_D2, "ff": 2, "tie_qo": True, "share_norms": True, "use_swiglu": False, "activation": "relu"},
     "tie_groups": [_attn(_KeqQ, _VeqQ), _TIEQK]},
    {"name": "21p_d2_relu_ff2_tied_norot", "params": 21,
     "model_args": {**BASE_ARGS_D2, "ff": 2, "tie_qo": True, "share_norms": True, "use_swiglu": False, "activation": "relu"},
     "tie_groups": [_attn(_KeqQ, _VeqQ), _mlp(_downUpT), _TIEQK]},
    {"name": "23p_d2_relu_ff1_norot", "params": 23,
     "model_args": {**BASE_ARGS_D2, "ff": 1, "tie_qo": True, "share_block_norms": True, "use_swiglu": False, "activation": "relu"},
     "tie_groups": [_attn(_KeqQ, _VeqQ), _TIEQK]},
    {"name": "21p_d2_relu_ff1_tied_norot", "params": 21,
     "model_args": {**BASE_ARGS_D2, "ff": 1, "tie_qo": True, "share_block_norms": True, "use_swiglu": False, "activation": "relu"},
     "tie_groups": [_attn(_KeqQ, _VeqQ), _mlp(_downUpT), _TIEQK]},

    # ── d=2, hd=2 configs (true 2×2 attention projections) ──

    {"name": "20p_d2hd2_relu_ff2", "params": 20,
     "model_args": {**BASE_ARGS_D2_HD2, "ff": 2, "tie_qo": True, "share_norms": True, "use_swiglu": False, "activation": "relu"},
     "tie_groups": [_attn(_KrotQ, _VeqQ), _TIEQK]},
    {"name": "16p_d2hd2_relu_ff2_tied", "params": 16,
     "model_args": {**BASE_ARGS_D2_HD2, "ff": 2, "tie_qo": True, "share_norms": True, "use_swiglu": False, "activation": "relu"},
     "tie_groups": [_attn(_KrotQ, _VeqQ), _mlp(_downUpT), _TIEQK]},
    {"name": "16p_d2hd2_relu_ff1", "params": 16,
     "model_args": {**BASE_ARGS_D2_HD2, "ff": 1, "tie_qo": True, "share_norms": True, "use_swiglu": False, "activation": "relu"},
     "tie_groups": [_attn(_KrotQ, _VeqQ), _TIEQK]},
    {"name": "14p_d2hd2_relu_ff1_tied", "params": 14,
     "model_args": {**BASE_ARGS_D2_HD2, "ff": 1, "tie_qo": True, "share_norms": True, "use_swiglu": False, "activation": "relu"},
     "tie_groups": [_attn(_KrotQ, _VeqQ), _mlp(_downUpT), _TIEQK]},
    {"name": "24p_d2hd2_swiglu_ff2", "params": 24,
     "model_args": {**BASE_ARGS_D2_HD2, "ff": 2, "tie_qo": True, "share_norms": True},
     "tie_groups": [_attn(_KrotQ, _VeqQ), _TIEQK]},
    {"name": "19p_d2hd2_relu_ff2_norot", "params": 19,
     "model_args": {**BASE_ARGS_D2_HD2, "ff": 2, "tie_qo": True, "share_norms": True, "use_swiglu": False, "activation": "relu"},
     "tie_groups": [_attn(_KeqQ, _VeqQ), _TIEQK]},
    {"name": "15p_d2hd2_relu_ff2_tied_norot", "params": 15,
     "model_args": {**BASE_ARGS_D2_HD2, "ff": 2, "tie_qo": True, "share_norms": True, "use_swiglu": False, "activation": "relu"},
     "tie_groups": [_attn(_KeqQ, _VeqQ), _mlp(_downUpT), _TIEQK]},
]


# ============================================================================
# Model building
# ============================================================================

def build_model(cfg, device):
    """Build model with tie_groups (new) or tying dict (old compat)."""
    model_args = dict(cfg["model_args"])
    tying = cfg.get("tying", {})

    # Handle tying flags that modify model construction (not tie_groups)
    if tying.get("relu_mlp"):
        model_args["use_swiglu"] = False
        model_args["activation"] = "relu"
    if tying.get("relu_bias"):
        model_args["mlp_bias"] = True

    model = CircularArcQwen3(**model_args)

    if "tie_groups" in cfg:
        apply_tie_groups(model, cfg["tie_groups"])
    elif tying:
        groups = _tying_to_tie_groups(tying)
        apply_tie_groups(model, groups)

    model = model.to(device)

    return model, count_unique_params(model)


def count_params(model):
    """Count unique parameters (alias for count_unique_params)."""
    return count_unique_params(model)


def _tying_to_tie_groups(tying):
    """Convert old tying dict to tie_groups list."""
    groups = []

    # Attention ties (q_proj is master)
    attn_followers = []
    if tying.get("k_rot_q"):
        attn_followers.append(_KrotQ)
    elif tying.get("k_alpha_q"):
        attn_followers.append(_KaQ)
    elif tying.get("k_eq_q"):
        attn_followers.append(_KeqQ)
    if tying.get("v_eq_q"):
        attn_followers.append(_VeqQ)
    if attn_followers:
        groups.append(_attn(*attn_followers))

    # MLP ties (up_proj is master)
    mlp_followers = []
    if tying.get("gate_alpha"):
        mlp_followers.append(_gateA)
    elif tying.get("gate_eq_up"):
        mlp_followers.append(_gateEqUp)
    if tying.get("down_eq_upT"):
        mlp_followers.append(_downUpT)
    if mlp_followers:
        groups.append(_mlp(*mlp_followers))

    # QK norm sharing
    if tying.get("tie_qk_norm"):
        groups.append(_TIEQK)

    return groups


def remap_old_state_dict(state_dict):
    """Remap old monkey-patched state dict keys to tie_groups format.

    Old: block.attn.k_alpha → New: block.attn.k_proj.alpha
    Old: block.mlp.gate_alpha → New: block.mlp.gate_proj.alpha
    """
    KEY_MAP = {
        "block.attn.k_alpha": "block.attn.k_proj.alpha",
        "block.mlp.gate_alpha": "block.mlp.gate_proj.alpha",
    }
    return {KEY_MAP.get(k, k): v for k, v in state_dict.items()}


# Re-export training infrastructure
get_lr = _impl.get_lr

# Patch the old module so train_one/main use our build_model and CONFIGS
_impl.build_model = build_model
_impl.CONFIGS = CONFIGS
_impl.count_params = count_params
# Fix subprocess script path: when _impl.main spawns workers, it uses __file__
# which would point to sub50_sweep_obsolete.py.  Override so workers run this script.
_impl.__dict__["__file__"] = os.path.abspath(__file__)

train_one = _impl.train_one
run_worker = _impl.run_worker
print_summary = _impl.print_summary
main = _impl.main

if __name__ == "__main__":
    main()
