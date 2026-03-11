"""Sub-50p architecture sweep using declarative tie_groups.

Defines model configs with tie_groups lists and a full training engine
(AdamW, L-BFGS, CMA-ES, Newton) with subprocess-based parallelism.

Usage:
    python experiments/sub50_sweep.py                           # run all on CPU
    python experiments/sub50_sweep.py --device cuda             # run all on GPU
    python experiments/sub50_sweep.py --config 44p --algo adamw  # single config+algo
    python experiments/sub50_sweep.py --summary                 # print summary
"""

import argparse
import csv
import math
import multiprocessing
import os
import random
import signal
import subprocess
import sys
import time

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_SCRIPT_DIR, "..", "src"))
sys.path.insert(0, _SCRIPT_DIR)

import torch
import torch.nn as nn
import torch.nn.functional as F

from minimal10digittransformer.model.circular_arc import CircularArcQwen3
from minimal10digittransformer.model.qwen3 import (
    VOCAB_SIZE, TOTAL_LEN, MAX_ADDEND, INPUT_LEN, apply_rope,
)
from minimal10digittransformer.data.addition import (
    encode, expected_output, generate_batch,
)
from minimal10digittransformer.evaluation.metrics import evaluate
from minimal10digittransformer.model.tie_groups import apply_tie_groups, count_unique_params

SCRIPT_DIR = _SCRIPT_DIR
RESULTS_DIR = os.path.join(SCRIPT_DIR, "sweep_results_sub50")

# Resolve the venv python to avoid zombie processes when launched via `uv run`.
# When `uv` wraps the process, it exits without calling wait() on its children,
# causing them to become zombies reparented to PID 1.  By using the venv python
# directly we eliminate the extra wrapper layer.
_VENV_PYTHON = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    ".venv", "bin", "python3",
)
PYTHON = _VENV_PYTHON if os.path.exists(_VENV_PYTHON) else sys.executable


# ============================================================================
# Architecture configurations -- declarative tie_groups
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
_downRotUpT = "rottranspose:block.mlp.down_proj"  # down = up^T . R(theta) (1 param)
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
    # -- ff=2 configs (more MLP capacity) --
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

    # -- ff=2 with shbnorm --
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

    # -- ff=1 configs --
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

    # -- ff=1 without gate tying --
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

    # -- ff=1 with shbnorm --
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

    # -- ff=1 with down=up^T --
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

    # -- NEW: ff=2 sub-45p variants (from 45p winner) --

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

    # -- MLP tying: all 4 combinations of pairing {up, down, gate} --

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

    # -- K=rotation(Q) configs (2D rotation in head_dim space, 1 param) --

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
    # 36p: down=R(theta).up^T (rotation adds 1 param vs pure transpose)
    {
        "name": "36p_ff2_tieQO_KrotQ_shnorm_VeqQ_downRotUpT_tieQKnorm",
        "params": 36,
        "model_args": {**BASE_ARGS, "ff": 2, "tie_qo": True, "share_norms": True},
        "tie_groups": [_attn(_KrotQ, _VeqQ), _mlp(_downRotUpT), _TIEQK],
    },
    # 34p: drop A from embedding (2 params: start, stride)
    {
        "name": "34p_ff2_tieQO_KrotQ_shnorm_VeqQ_downUpT_tieQKnorm_dropA",
        "params": 34,
        "model_args": {**BASE_ARGS, "ff": 2, "tie_qo": True, "share_norms": True, "drop_A": True},
        "tie_groups": [_attn(_KrotQ, _VeqQ), _mlp(_downUpT), _TIEQK],
    },
    # Keep old name as alias for running experiments (results dir compat)
    {
        "name": "34p_ff2_tieQO_KrotQ_shnorm_VeqQ_downUpT_tieQKnorm_fixA",
        "params": 34,
        "model_args": {**BASE_ARGS, "ff": 2, "tie_qo": True, "share_norms": True, "drop_A": True},
        "tie_groups": [_attn(_KrotQ, _VeqQ), _mlp(_downUpT), _TIEQK],
    },

    # -- Gate reparameterisations (log-space and vector) --

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

    # -- ff=1 SwiGLU with KrotQ (minimal MLP) --

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

    # -- ff=2 + gate=alpha*up + down=up^T (proven ties combined) --

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

    # -- ReLU MLP (no gate_proj, saves ff*d params vs SwiGLU) --

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

    # -- d=2 configs --

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

    # -- No-rotation variants (K=Q identity, drops 1 param) --

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

    # -- d=2, hd=2 configs (true 2x2 attention projections) --

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

def apply_init_scheme(model, init_scheme):
    """Apply per-layer initialization from an init_scheme dict.

    init_scheme maps parameter name patterns to distribution configs:
        {"q_proj": {"dist": "normal", "mean": 0, "std": 5.0},
         "norm":   {"dist": "uniform", "low": -10, "high": 15},
         "theta":  {"dist": "uniform", "low": -4, "high": 2},
         ...}

    Supported distributions:
        normal(mean, std), uniform(low, high), laplace(loc, scale),
        ones(), zeros(), constant(value)

    Pattern matching: a parameter is matched if the pattern string appears
    anywhere in the parameter name. First match wins.
    """
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        for pattern, spec in init_scheme.items():
            if pattern not in name:
                continue
            dist = spec.get("dist", "normal")
            with torch.no_grad():
                if dist == "normal":
                    p.normal_(mean=spec.get("mean", 0.0), std=spec.get("std", 0.02))
                elif dist == "uniform":
                    p.uniform_(spec.get("low", -1.0), spec.get("high", 1.0))
                elif dist == "laplace":
                    # Laplace: loc + scale * (Exponential - Exponential)
                    loc = spec.get("loc", 0.0)
                    scale = spec.get("scale", 1.0)
                    p.copy_(torch.distributions.Laplace(loc, scale).sample(p.shape))
                elif dist == "ones":
                    p.fill_(1.0)
                elif dist == "zeros":
                    p.zero_()
                elif dist == "constant":
                    p.fill_(spec.get("value", 0.0))
            break  # first match wins


def build_model(cfg, device, init_scheme=None):
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

    # Apply custom init AFTER tie_groups (so tied params get re-inited together)
    if init_scheme:
        apply_init_scheme(model, init_scheme)

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

    Old: block.attn.k_alpha -> New: block.attn.k_proj.alpha
    Old: block.mlp.gate_alpha -> New: block.mlp.gate_proj.alpha
    """
    KEY_MAP = {
        "block.attn.k_alpha": "block.attn.k_proj.alpha",
        "block.mlp.gate_alpha": "block.mlp.gate_proj.alpha",
    }
    return {KEY_MAP.get(k, k): v for k, v in state_dict.items()}


# ============================================================================
# Training algorithm configurations
# ============================================================================

ALGORITHMS = [
    # Core optimizers
    {"name": "adamw", "optimizer": "adamw", "lr": 0.01, "weight_decay": 0.01},
    {"name": "adamw_lr03", "optimizer": "adamw", "lr": 0.03, "weight_decay": 0.01},
    {"name": "adamw_lr02", "optimizer": "adamw", "lr": 0.02, "weight_decay": 0.01},
    {"name": "adamw_nowd", "optimizer": "adamw", "lr": 0.01, "weight_decay": 0.0},
    {"name": "adamw_highbeta", "optimizer": "adamw_high_beta1", "lr": 0.01, "weight_decay": 0.01},

    # Grokfast variants (our key technique for small models)
    {"name": "adamw+gf", "optimizer": "adamw", "lr": 0.01, "weight_decay": 0.01,
     "grokfast": True},
    {"name": "adamw_lr02+gf", "optimizer": "adamw", "lr": 0.02, "weight_decay": 0.01,
     "grokfast": True},
    {"name": "gf_nowd_lam1", "optimizer": "adamw", "lr": 0.003, "weight_decay": 0.0,
     "grokfast": True, "grokfast_lambda": 1.0},

    # Adaptive WD
    {"name": "adamw+awd", "optimizer": "adamw", "lr": 0.01, "weight_decay": 0.01,
     "adaptive_wd": True},

    # Curriculum + grokfast + adaptive_wd (our universal recipe for small models)
    {"name": "curr+gf+awd", "optimizer": "adamw", "lr": 0.01, "weight_decay": 0.01,
     "grokfast": True, "adaptive_wd": True,
     "digit_curriculum": [(0, 3), (2000, 6), (7000, 10)]},

    # Curriculum + cosine_wd (best for 83p)
    {"name": "curr+coswd", "optimizer": "adamw", "lr": 0.01, "weight_decay": 0.01,
     "cosine_wd": True,
     "digit_curriculum": [(0, 3), (2000, 6), (7000, 10)]},

    # Inverse cosine WD (best for 122p)
    {"name": "inv_coswd", "optimizer": "adamw", "lr": 0.01, "weight_decay": 0.01,
     "inverse_cosine_wd": True},

    # Carry-mix (MicroAdder's key technique)
    {"name": "carry_mix", "optimizer": "adamw", "lr": 0.01, "weight_decay": 0.01,
     "carry_mix": True, "carry_ratio": 0.8},

    # Carry-mix + grokfast
    {"name": "carry+gf", "optimizer": "adamw", "lr": 0.01, "weight_decay": 0.01,
     "carry_mix": True, "carry_ratio": 0.8, "grokfast": True},

    # RMSprop no WD (dark horse in ablation)
    {"name": "rmsprop_nowd", "optimizer": "rmsprop", "lr": 0.01, "weight_decay": 0.0},

    # All micro (MicroAdder-style: carry+curriculum+lr02+warmup)
    {"name": "all_micro", "optimizer": "adamw", "lr": 0.02, "weight_decay": 0.01,
     "carry_mix": True, "carry_ratio": 0.8,
     "digit_curriculum": [(0, 3), (2000, 6), (7000, 10)],
     "warmup_steps": 1000},

    # All micro + grokfast
    {"name": "all_micro+gf", "optimizer": "adamw", "lr": 0.02, "weight_decay": 0.01,
     "carry_mix": True, "carry_ratio": 0.8,
     "digit_curriculum": [(0, 3), (2000, 6), (7000, 10)],
     "warmup_steps": 1000, "grokfast": True},

    # -- Research-derived optimizers --

    # perpGrad: project gradients orthogonal to weights (accelerates grokking 32-60%)
    {"name": "adamw+perpgrad", "optimizer": "adamw", "lr": 0.01, "weight_decay": 0.01,
     "perpgrad": True},
    {"name": "adamw+perpgrad+gf", "optimizer": "adamw", "lr": 0.01, "weight_decay": 0.01,
     "perpgrad": True, "grokfast": True},

    # SGLD: adds scaled noise to gradients (escapes local minima)
    {"name": "sgld", "optimizer": "adamw", "lr": 0.01, "weight_decay": 0.01,
     "sgld_temp": 1.0},
    {"name": "sgld+gf", "optimizer": "adamw", "lr": 0.01, "weight_decay": 0.01,
     "sgld_temp": 1.0, "grokfast": True},

    # Cycling directional perturbation v1 (DEPRECATED: in-place, wrong)
    {"name": "adamw+cyc_perturb", "optimizer": "adamw", "lr": 0.01, "weight_decay": 0.01,
     "cyclic_perturb": True, "perturb_gamma": 0.01, "perturb_cycle": 5000},
    {"name": "adamw+cyc_perturb+gf", "optimizer": "adamw", "lr": 0.01, "weight_decay": 0.01,
     "cyclic_perturb": True, "perturb_gamma": 0.01, "perturb_cycle": 5000,
     "grokfast": True},

    # CDP v2: frozen overlay perturbation (W_eff = W_base + overlay, overlay frozen)
    {"name": "adamw+cdp", "optimizer": "adamw", "lr": 0.01, "weight_decay": 0.01,
     "cdp": True, "cdp_scale": 0.1, "cdp_cycle": 10000, "cdp_target": "proj"},
    {"name": "adamw+cdp+gf", "optimizer": "adamw", "lr": 0.01, "weight_decay": 0.01,
     "cdp": True, "cdp_scale": 0.1, "cdp_cycle": 10000, "cdp_target": "proj",
     "grokfast": True, "gf_alpha": 0.98, "gf_lambda": 2.0},
    {"name": "adamw+cdp_attn", "optimizer": "adamw", "lr": 0.01, "weight_decay": 0.01,
     "cdp": True, "cdp_scale": 0.1, "cdp_cycle": 10000, "cdp_target": "attn"},
    {"name": "adamw+cdp_all", "optimizer": "adamw", "lr": 0.01, "weight_decay": 0.01,
     "cdp": True, "cdp_scale": 0.1, "cdp_cycle": 10000, "cdp_target": "all"},

    # -- Low-LR + high-WD variants (for ultra-small models with scale degeneracy) --

    {"name": "adamw_lr001+gf", "optimizer": "adamw", "lr": 0.001, "weight_decay": 0.01,
     "grokfast": True},
    {"name": "adamw_lr001_wd01+gf", "optimizer": "adamw", "lr": 0.001, "weight_decay": 0.1,
     "grokfast": True},
    {"name": "adamw_lr003_wd01+gf", "optimizer": "adamw", "lr": 0.003, "weight_decay": 0.1,
     "grokfast": True},
    {"name": "adamw_wd01+gf", "optimizer": "adamw", "lr": 0.01, "weight_decay": 0.1,
     "grokfast": True},
    {"name": "adamw_lr001_wd03+gf", "optimizer": "adamw", "lr": 0.001, "weight_decay": 0.3,
     "grokfast": True},

    # -- Simple baselines (no WD, no adaptive LR) --
    {"name": "sgd", "optimizer": "sgd", "lr": 0.01, "weight_decay": 0.0},
    {"name": "sgd_lr001", "optimizer": "sgd", "lr": 0.001, "weight_decay": 0.0},
    {"name": "sgd_mom", "optimizer": "sgd_momentum", "lr": 0.01, "weight_decay": 0.0},
    {"name": "adam_nowd", "optimizer": "adam", "lr": 0.01, "weight_decay": 0.0},
    {"name": "adam_nowd+gf", "optimizer": "adam", "lr": 0.01, "weight_decay": 0.0,
     "grokfast": True},

    # -- Explicit L2 regularization (added to loss, flows through Adam momentum) --
    {"name": "adamw+l2_01+gf", "optimizer": "adamw", "lr": 0.01, "weight_decay": 0.0,
     "l2_lambda": 0.1, "grokfast": True},
    {"name": "adamw+l2_001+gf", "optimizer": "adamw", "lr": 0.01, "weight_decay": 0.0,
     "l2_lambda": 0.01, "grokfast": True},
    {"name": "adamw+l2_1+gf", "optimizer": "adamw", "lr": 0.01, "weight_decay": 0.0,
     "l2_lambda": 1.0, "grokfast": True},

    # -- 2nd-order / derivative-free methods (for ultra-small models <=50 params) --

    # L-BFGS: quasi-Newton, approximates inverse Hessian
    {"name": "lbfgs", "optimizer": "lbfgs", "lr": 1.0, "batch_size": 512},
    {"name": "lbfgs_lr01", "optimizer": "lbfgs", "lr": 0.1, "batch_size": 512},
    {"name": "lbfgs_lr001", "optimizer": "lbfgs", "lr": 0.01, "batch_size": 512},

    # CMA-ES: covariance matrix adaptation evolution strategy (gradient-free)
    {"name": "cmaes", "optimizer": "cmaes", "sigma0": 0.5, "batch_size": 512},
    {"name": "cmaes_sig01", "optimizer": "cmaes", "sigma0": 0.1, "batch_size": 512},
    {"name": "cmaes_sig1", "optimizer": "cmaes", "sigma0": 1.0, "batch_size": 512},

    # -- SAM (Sharpness-Aware Minimization) --
    {"name": "sam", "optimizer": "adamw", "lr": 0.01, "weight_decay": 0.01,
     "sam": True, "sam_rho": 0.05},
    {"name": "sam+gf", "optimizer": "adamw", "lr": 0.01, "weight_decay": 0.01,
     "sam": True, "sam_rho": 0.05, "grokfast": True},
    {"name": "sam_rho02", "optimizer": "adamw", "lr": 0.01, "weight_decay": 0.01,
     "sam": True, "sam_rho": 0.2},
    {"name": "sam_rho02+gf", "optimizer": "adamw", "lr": 0.01, "weight_decay": 0.01,
     "sam": True, "sam_rho": 0.2, "grokfast": True},

    # -- Per-param LR groups --
    {"name": "pplr", "optimizer": "adamw_pplr", "lr": 0.01, "weight_decay": 0.01,
     "lr_gate_mult": 0.1, "lr_norm_mult": 3.0},
    {"name": "pplr+gf", "optimizer": "adamw_pplr", "lr": 0.01, "weight_decay": 0.01,
     "lr_gate_mult": 0.1, "lr_norm_mult": 3.0, "grokfast": True},
    {"name": "pplr_gm03+gf", "optimizer": "adamw_pplr", "lr": 0.01, "weight_decay": 0.01,
     "lr_gate_mult": 0.3, "lr_norm_mult": 3.0, "grokfast": True},

    # -- Per-param LR v2 — gradient-analysis-informed --
    {"name": "pplr2+gf", "optimizer": "adamw_pplr2", "lr": 0.001, "weight_decay": 0.01,
     "lr_arc_mult": 0.5, "lr_norm_mult": 3.0, "lr_up_mult": 1.5, "wd_qproj": 0.01,
     "grokfast": True},
    {"name": "pplr2_lr01+gf", "optimizer": "adamw_pplr2", "lr": 0.01, "weight_decay": 0.01,
     "lr_arc_mult": 0.5, "lr_norm_mult": 3.0, "lr_up_mult": 1.5, "wd_qproj": 0.01,
     "grokfast": True},
    {"name": "pplr2_curr+gf", "optimizer": "adamw_pplr2", "lr": 0.001, "weight_decay": 0.01,
     "lr_arc_mult": 0.5, "lr_norm_mult": 3.0, "lr_up_mult": 1.5, "wd_qproj": 0.01,
     "grokfast": True, "curriculum": True, "adaptive_wd": True},
    # Selective weight decay on exploding params only
    {"name": "adamw_selwd+gf", "optimizer": "adamw_pplr2", "lr": 0.001, "weight_decay": 0.0,
     "lr_arc_mult": 0.5, "lr_norm_mult": 3.0, "lr_up_mult": 1.5, "wd_qproj": 0.03,
     "grokfast": True},

    # -- Per-layer init (scaled distributions from gradient analysis) --
    # sinit_v1: normal for most, uniform for norms/scalars, laplace for q_proj
    {"name": "sinit_v1+gf", "optimizer": "adamw_pplr2", "lr": 0.001, "weight_decay": 0.01,
     "lr_arc_mult": 0.5, "lr_norm_mult": 3.0, "lr_up_mult": 1.5, "wd_qproj": 0.01,
     "grokfast": True, "init_scheme": {
         "q_proj.weight": {"dist": "laplace", "loc": 0, "scale": 3.0},
         "up_proj": {"dist": "normal", "std": 3.0},
         "gate_proj": {"dist": "normal", "std": 0.5},
         "ln1": {"dist": "uniform", "low": -10, "high": 15},
         "norm": {"dist": "uniform", "low": -10, "high": 15},
         "arc_stride": {"dist": "uniform", "low": 0.04, "high": 0.12},
         "theta": {"dist": "uniform", "low": -4, "high": 2},
     }},
    # sinit_v2: wider q_proj, tighter norms
    {"name": "sinit_v2+gf", "optimizer": "adamw_pplr2", "lr": 0.001, "weight_decay": 0.01,
     "lr_arc_mult": 0.5, "lr_norm_mult": 3.0, "lr_up_mult": 1.5, "wd_qproj": 0.01,
     "grokfast": True, "init_scheme": {
         "q_proj.weight": {"dist": "laplace", "loc": 0, "scale": 8.0},
         "up_proj": {"dist": "normal", "std": 8.0},
         "gate_proj": {"dist": "normal", "std": 1.0},
         "ln1": {"dist": "uniform", "low": -5, "high": 20},
         "norm": {"dist": "uniform", "low": -5, "high": 20},
         "arc_stride": {"dist": "uniform", "low": 0.03, "high": 0.15},
         "theta": {"dist": "uniform", "low": -5, "high": 3},
     }},
    # sinit_v1 with curr+awd
    {"name": "sinit_v1_curr+gf", "optimizer": "adamw_pplr2", "lr": 0.001, "weight_decay": 0.01,
     "lr_arc_mult": 0.5, "lr_norm_mult": 3.0, "lr_up_mult": 1.5, "wd_qproj": 0.01,
     "grokfast": True, "curriculum": True, "adaptive_wd": True, "init_scheme": {
         "q_proj.weight": {"dist": "laplace", "loc": 0, "scale": 3.0},
         "up_proj": {"dist": "normal", "std": 3.0},
         "gate_proj": {"dist": "normal", "std": 0.5},
         "ln1": {"dist": "uniform", "low": -10, "high": 15},
         "norm": {"dist": "uniform", "low": -10, "high": 15},
         "arc_stride": {"dist": "uniform", "low": 0.04, "high": 0.12},
         "theta": {"dist": "uniform", "low": -4, "high": 2},
     }},

    # -- Weight norm constraint (OmniGrok) --
    {"name": "omnigrok", "optimizer": "adamw", "lr": 0.01, "weight_decay": 0.0,
     "norm_constraint": True, "norm_radius": 1.0},
    {"name": "omnigrok+gf", "optimizer": "adamw", "lr": 0.01, "weight_decay": 0.0,
     "norm_constraint": True, "norm_radius": 1.0, "grokfast": True},
    {"name": "omnigrok_r05+gf", "optimizer": "adamw", "lr": 0.01, "weight_decay": 0.0,
     "norm_constraint": True, "norm_radius": 0.5, "grokfast": True},
    {"name": "omnigrok_r2+gf", "optimizer": "adamw", "lr": 0.01, "weight_decay": 0.0,
     "norm_constraint": True, "norm_radius": 2.0, "grokfast": True},

    # -- Full Newton (exact Hessian, tractable for <=50 params) --
    {"name": "newton", "optimizer": "newton", "lr": 0.1, "batch_size": 512},
    {"name": "newton_lr01", "optimizer": "newton", "lr": 0.01, "batch_size": 512},
    {"name": "newton_lr1", "optimizer": "newton", "lr": 1.0, "batch_size": 512},
    {"name": "newton_damp", "optimizer": "newton", "lr": 0.1, "batch_size": 512,
     "newton_damping": 0.1},
]

SEEDS = list(range(50))  # 50 seeds


# ============================================================================
# Data generation
# ============================================================================

def generate_carry_heavy_pair(max_digits=10):
    """Generate a pair (a, b) where the addition involves carries."""
    if max_digits < 10:
        n_d = random.randint(1, max_digits)
        limit = 10 ** n_d - 1
    else:
        n_d = 10
        limit = MAX_ADDEND

    a_digits = []
    b_digits = []
    has_carry = False
    for _ in range(n_d):
        a_d = random.randint(0, 9)
        if random.random() < 0.5:
            min_b = max(0, 10 - a_d)
            if min_b <= 9:
                b_d = random.randint(min_b, 9)
                has_carry = True
            else:
                b_d = random.randint(0, 9)
        else:
            b_d = random.randint(0, 9)
        a_digits.append(a_d)
        b_digits.append(b_d)

    a = sum(d * (10 ** i) for i, d in enumerate(a_digits))
    b = sum(d * (10 ** i) for i, d in enumerate(b_digits))
    a = min(a, limit)
    b = min(b, limit)

    if not has_carry:
        carry = 0
        aa, bb = a, b
        for _ in range(n_d):
            if (aa % 10) + (bb % 10) + carry >= 10:
                has_carry = True
                break
            carry = 0
            aa //= 10
            bb //= 10
        if not has_carry:
            return generate_carry_heavy_pair(max_digits)

    return a, b


def generate_carry_mix_batch(batch_size, device, carry_ratio, max_digits=10):
    """Generate batch with carry_ratio fraction of carry-heavy examples."""
    n_carry = int(batch_size * carry_ratio)
    n_random = batch_size - n_carry

    full_list, label_list = [], []

    for _ in range(n_carry):
        a, b = generate_carry_heavy_pair(max_digits)
        inp = encode(a, b)
        tgt = expected_output(a, b)
        full_list.append(inp + tgt)
        label_list.append([-100] * INPUT_LEN + tgt)

    for _ in range(n_random):
        if max_digits < 10:
            n_d = random.randint(1, max_digits)
            a = random.randint(0, 10 ** n_d - 1)
            b = random.randint(0, 10 ** n_d - 1)
        else:
            a = random.randint(0, MAX_ADDEND)
            b = random.randint(0, MAX_ADDEND)
        inp = encode(a, b)
        tgt = expected_output(a, b)
        full_list.append(inp + tgt)
        label_list.append([-100] * INPUT_LEN + tgt)

    full_seq = torch.tensor(full_list, dtype=torch.long, device=device)
    labels = torch.tensor(label_list, dtype=torch.long, device=device)
    perm = torch.randperm(batch_size)
    return full_seq[perm], labels[perm]


# ============================================================================
# Training utilities
# ============================================================================

def get_max_digits(step, curriculum):
    if curriculum is None:
        return 10
    result = 10
    for threshold, digits in sorted(curriculum, reverse=True):
        if step >= threshold:
            result = digits
            break
    return result


def get_carry_ratio(step, base_ratio, fade_start, fade_end):
    if step < fade_start:
        return base_ratio
    if step >= fade_end:
        return 0.0
    progress = (step - fade_start) / max(fade_end - fade_start, 1)
    return base_ratio * (1.0 - progress)


def get_lr(step, total_steps, base_lr, warmup_steps=0, min_lr_ratio=0.1):
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * step / warmup_steps
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    min_lr = base_lr * min_lr_ratio
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


# ============================================================================
# Single run
# ============================================================================

def train_one(arch_cfg, algo_cfg, seed, max_steps, eval_interval, device, results_dir):
    """Train one (architecture, algorithm, seed) combination."""
    config_name = arch_cfg["name"]
    algo_name = algo_cfg["name"]
    run_name = f"{config_name}__{algo_name}"

    # Check if already done
    csv_path = os.path.join(results_dir, f"{run_name}_s{seed}.csv")
    if os.path.exists(csv_path):
        try:
            with open(csv_path) as f:
                lines = f.readlines()
            if len(lines) > 2:  # header + at least 2 data rows
                return None  # skip
        except Exception:
            pass

    random.seed(seed)
    torch.manual_seed(seed)

    # Build model (with optional per-layer init from algo config)
    init_scheme = algo_cfg.get("init_scheme", None)
    model, n_params = build_model(arch_cfg, device, init_scheme=init_scheme)

    # Extract algo settings
    lr = algo_cfg.get("lr", 0.01)
    weight_decay = algo_cfg.get("weight_decay", 0.01)
    opt_name = algo_cfg.get("optimizer", "adamw")
    warmup_steps = algo_cfg.get("warmup_steps", 0)
    use_grokfast = algo_cfg.get("grokfast", False)
    grokfast_alpha = algo_cfg.get("grokfast_alpha", 0.98)
    grokfast_lambda = algo_cfg.get("grokfast_lambda", 2.0)
    adaptive_wd = algo_cfg.get("adaptive_wd", False)
    cosine_wd = algo_cfg.get("cosine_wd", False)
    inverse_cosine_wd = algo_cfg.get("inverse_cosine_wd", False)
    carry_mix = algo_cfg.get("carry_mix", False)
    carry_ratio = algo_cfg.get("carry_ratio", 0.8)
    carry_fade_start = algo_cfg.get("carry_fade_start", 15000)
    carry_fade_end = algo_cfg.get("carry_fade_end", 45000)
    digit_curriculum = algo_cfg.get("digit_curriculum", None)
    batch_size = algo_cfg.get("batch_size", 128)
    use_perpgrad = algo_cfg.get("perpgrad", False)
    sgld_temp = algo_cfg.get("sgld_temp", 0.0)
    use_cyclic_perturb = algo_cfg.get("cyclic_perturb", False)
    perturb_gamma_init = algo_cfg.get("perturb_gamma", 0.01)
    perturb_cycle = algo_cfg.get("perturb_cycle", 5000)

    # Explicit L2 regularization (added to loss, NOT weight decay)
    l2_lambda = algo_cfg.get("l2_lambda", 0.0)

    # CDP v2: frozen overlay cycling directional perturbation
    use_cdp = algo_cfg.get("cdp", False)
    cdp_scale = algo_cfg.get("cdp_scale", 0.1)
    cdp_cycle = algo_cfg.get("cdp_cycle", 10000)
    cdp_target = algo_cfg.get("cdp_target", "proj")

    # SAM (Sharpness-Aware Minimization)
    use_sam = algo_cfg.get("sam", False)
    sam_rho = algo_cfg.get("sam_rho", 0.05)

    # Weight norm constraint (OmniGrok)
    use_norm_constraint = algo_cfg.get("norm_constraint", False)
    norm_radius = algo_cfg.get("norm_radius", 1.0)

    # Per-param LR group multipliers
    lr_gate_mult = algo_cfg.get("lr_gate_mult", 1.0)
    lr_norm_mult = algo_cfg.get("lr_norm_mult", 1.0)
    lr_arc_mult = algo_cfg.get("lr_arc_mult", 1.0)
    lr_up_mult = algo_cfg.get("lr_up_mult", 1.0)
    wd_qproj = algo_cfg.get("wd_qproj", weight_decay)

    # Build optimizer
    if opt_name == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr,
                                         weight_decay=weight_decay)
    elif opt_name == "adamw_high_beta1":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                       betas=(0.98, 0.999),
                                       weight_decay=weight_decay)
    elif opt_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                     weight_decay=weight_decay)
    elif opt_name == "sgd_momentum":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                                     weight_decay=weight_decay)
    elif opt_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                      weight_decay=weight_decay)
    elif opt_name == "adamw_pplr":
        # Per-parameter LR groups: gate/scalar params get lower LR, norms get higher
        gate_params, norm_params, other_params = [], [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if "alpha" in name or "log_alpha" in name or "alpha_vec" in name or "theta" in name:
                gate_params.append(p)
            elif "norm" in name:
                norm_params.append(p)
            else:
                other_params.append(p)
        param_groups = [
            {"params": other_params, "lr": lr},
            {"params": gate_params, "lr": lr * lr_gate_mult},
            {"params": norm_params, "lr": lr * lr_norm_mult},
        ]
        # Filter empty groups
        param_groups = [g for g in param_groups if g["params"]]
        optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    elif opt_name == "adamw_pplr2":
        # Fine-grained per-parameter LR from gradient analysis
        arc_params, gate_params, norm_params, up_params, qproj_params, other_params = [], [], [], [], [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if "arc_" in name:
                arc_params.append(p)
            elif "alpha" in name or "log_alpha" in name or "alpha_vec" in name:
                gate_params.append(p)
            elif "norm" in name:
                norm_params.append(p)
            elif "up_proj" in name:
                up_params.append(p)
            elif "q_proj" in name:
                qproj_params.append(p)
            else:
                other_params.append(p)
        param_groups = [
            {"params": arc_params, "lr": lr * lr_arc_mult},
            {"params": gate_params, "lr": lr * lr_gate_mult},
            {"params": norm_params, "lr": lr * lr_norm_mult},
            {"params": up_params, "lr": lr * lr_up_mult},
            {"params": qproj_params, "lr": lr, "weight_decay": wd_qproj},
            {"params": other_params, "lr": lr},
        ]
        param_groups = [g for g in param_groups if g["params"]]
        optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    else:  # adamw
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                       weight_decay=weight_decay)

    # Grokfast-EMA state
    gf_ema = {}
    if use_grokfast:
        for pname, p in model.named_parameters():
            if p.requires_grad:
                gf_ema[pname] = torch.zeros_like(p.data)

    # Adaptive WD tracking
    wd_thresholds = set()
    current_wd = weight_decay

    # Cyclic directional perturbation state (v1 - deprecated)
    perturb_dirs = {}
    perturb_gamma = perturb_gamma_init
    if use_cyclic_perturb:
        for pname, p in model.named_parameters():
            if p.requires_grad:
                d = torch.randn_like(p.data)
                d.div_(d.norm() + 1e-8)
                perturb_dirs[pname] = d

    # CDP v2: overlay-based perturbation state
    cdp_params = {}
    cdp_dirs = {}
    cdp_overlays = {}
    if use_cdp:
        for pname, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if cdp_target == "all":
                include = True
            elif cdp_target == "proj":
                include = "proj" in pname
            elif cdp_target == "attn":
                include = any(x in pname for x in ["q_proj", "k_proj", "v_proj", "o_proj"])
            elif cdp_target == "mlp":
                include = any(x in pname for x in ["up_proj", "gate_proj", "down_proj"])
            else:
                include = cdp_target in pname
            if include:
                cdp_params[pname] = p
                d = torch.randn_like(p.data)
                d.div_(d.norm() + 1e-8)
                cdp_dirs[pname] = d
                cdp_overlays[pname] = torch.zeros_like(p.data)

    # CSV output
    os.makedirs(results_dir, exist_ok=True)
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["step", "loss", "acc", "lr", "wd"])

    # Gradient/param logging for small models (<=50 params)
    log_gradients = n_params <= 50
    grad_log_file = None
    grad_log_writer = None
    grad_buffer = []
    param_names = []
    if log_gradients:
        param_names = [n for n, p in model.named_parameters() if p.requires_grad]
        grad_csv_path = csv_path.replace(".csv", ".grad.csv")
        grad_log_file = open(grad_csv_path, "w", newline="")
        grad_log_writer = csv.writer(grad_log_file)
        header = ["step", "loss"]
        for pn in param_names:
            short = pn.replace("block.", "").replace("attn.", "").replace("mlp.", "")
            header.extend([f"{short}_val", f"{short}_grad", f"{short}_upd"])
        grad_log_writer.writerow(header)

    t0 = time.time()
    best_acc = 0.0

    for step in range(1, max_steps + 1):
        # LR schedule
        cur_lr = get_lr(step, max_steps, lr, warmup_steps)
        for pg in optimizer.param_groups:
            pg["lr"] = cur_lr

        # WD schedule
        if cosine_wd or inverse_cosine_wd:
            progress = step / max(max_steps, 1)
            if cosine_wd:
                wd_scale = 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
            else:
                wd_scale = 0.1 + 0.9 * 0.5 * (1 - math.cos(math.pi * progress))
            current_wd = weight_decay * wd_scale
            for pg in optimizer.param_groups:
                pg["weight_decay"] = current_wd

        # Data generation
        max_digits = get_max_digits(step, digit_curriculum)
        if carry_mix:
            cur_carry_ratio = get_carry_ratio(step, carry_ratio, carry_fade_start, carry_fade_end)
            full_seq, labels = generate_carry_mix_batch(batch_size, device, cur_carry_ratio, max_digits)
        else:
            full_seq, labels = generate_batch(batch_size, device, max_digits)

        # CDP v2: update frozen overlay
        if use_cdp:
            cdp_half = cdp_cycle // 2
            cycle_pos = (step - 1) % cdp_cycle
            in_perturb_phase = cycle_pos < cdp_half

            if cycle_pos == 0:
                for pname in cdp_params:
                    d = torch.randn_like(cdp_params[pname].data)
                    d.div_(d.norm() + 1e-8)
                    cdp_dirs[pname] = d

            with torch.no_grad():
                for pname, p in cdp_params.items():
                    p.sub_(cdp_overlays[pname])

                    if in_perturb_phase:
                        lr_ratio = cur_lr / lr if lr > 0 else 0
                        cdp_gamma = (lr_ratio ** 0.1)
                        w_norm = p.data.norm()
                        magnitude = cdp_gamma * cdp_scale * w_norm
                        cdp_overlays[pname] = magnitude * cdp_dirs[pname]
                    else:
                        cdp_overlays[pname].zero_()

                    p.add_(cdp_overlays[pname])

        # Forward + backward
        model.train()
        logits = model(full_seq)
        shift_logits = logits[:, :-1, :].reshape(-1, VOCAB_SIZE)
        shift_labels = labels[:, 1:].reshape(-1)
        loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)

        # Explicit L2 regularization (added to loss, flows through Adam momentum)
        if l2_lambda > 0:
            l2_reg = sum(p.pow(2).sum() for p in model.parameters() if p.requires_grad)
            loss = loss + l2_lambda * l2_reg

        optimizer.zero_grad()
        loss.backward()

        # SAM: Sharpness-Aware Minimization (double forward-backward)
        if use_sam:
            _sam_grads = {}
            _sam_orig = {}
            grad_norm = 0.0
            for pname, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    _sam_grads[pname] = p.grad.clone()
                    _sam_orig[pname] = p.data.clone()
                    grad_norm += p.grad.pow(2).sum().item()
            grad_norm = math.sqrt(grad_norm) + 1e-12

            with torch.no_grad():
                for pname, p in model.named_parameters():
                    if pname in _sam_grads:
                        epsilon = sam_rho * _sam_grads[pname] / grad_norm
                        p.add_(epsilon)

            optimizer.zero_grad()
            logits2 = model(full_seq)
            shift_logits2 = logits2[:, :-1, :].reshape(-1, VOCAB_SIZE)
            loss_sam = F.cross_entropy(shift_logits2, shift_labels, ignore_index=-100)
            if l2_lambda > 0:
                l2_reg2 = sum(p.pow(2).sum() for p in model.parameters() if p.requires_grad)
                loss_sam = loss_sam + l2_lambda * l2_reg2
            loss_sam.backward()

            with torch.no_grad():
                for pname, p in model.named_parameters():
                    if pname in _sam_orig:
                        p.copy_(_sam_orig[pname])

        # Grokfast-EMA gradient filter
        if use_grokfast:
            for pname, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    gf_ema[pname].mul_(grokfast_alpha).add_(p.grad, alpha=1 - grokfast_alpha)
                    p.grad.add_(gf_ema[pname], alpha=grokfast_lambda)

        # perpGrad: project gradients orthogonal to weights
        if use_perpgrad:
            for p in model.parameters():
                if p.requires_grad and p.grad is not None and p.data.numel() > 1:
                    w_flat = p.data.view(-1)
                    g_flat = p.grad.view(-1)
                    proj = (g_flat @ w_flat) / (w_flat @ w_flat + 1e-8)
                    p.grad.sub_(proj * p.data)

        # SGLD: add scaled Gaussian noise to gradients
        if sgld_temp > 0:
            for p in model.parameters():
                if p.requires_grad and p.grad is not None:
                    noise = torch.randn_like(p.grad) * math.sqrt(2 * cur_lr * sgld_temp)
                    p.grad.add_(noise)

        nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Snapshot for gradient logging (before optimizer step)
        if log_gradients:
            _pre_vals = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
            _grads = {n: p.grad.clone() if p.grad is not None else torch.zeros_like(p.data)
                      for n, p in model.named_parameters() if p.requires_grad}

        optimizer.step()

        # Weight norm constraint (OmniGrok): project weights onto norm ball
        if use_norm_constraint:
            with torch.no_grad():
                for p in model.parameters():
                    if p.requires_grad and p.data.numel() > 1:
                        w_norm = p.data.norm()
                        if w_norm > norm_radius:
                            p.data.mul_(norm_radius / w_norm)

        # Log gradient info after every step (buffer in RAM)
        if log_gradients:
            row = [step, f"{loss.item():.6f}"]
            _named = dict(model.named_parameters())
            for pn in param_names:
                p = _named[pn]
                val = p.data.flatten()
                grad = _grads[pn].flatten()
                upd = (p.data - _pre_vals[pn]).flatten()
                if val.numel() == 1:
                    row.extend([f"{val.item():.6f}", f"{grad.item():.6f}", f"{upd.item():.6f}"])
                else:
                    row.extend([f"{val.norm().item():.6f}", f"{grad.norm().item():.6f}", f"{upd.norm().item():.6f}"])
            grad_buffer.append(row)
            if step % 10000 == 0:
                grad_log_writer.writerows(grad_buffer)
                grad_log_file.flush()
                grad_buffer.clear()

        # Cyclic directional perturbation v1 (deprecated, kept for running experiments)
        if use_cyclic_perturb:
            if step % perturb_cycle == 0:
                perturb_gamma = perturb_gamma_init
                for pname, p in model.named_parameters():
                    if p.requires_grad:
                        d = torch.randn_like(p.data)
                        d.div_(d.norm() + 1e-8)
                        perturb_dirs[pname] = d
            else:
                perturb_gamma *= 0.99995

            with torch.no_grad():
                for pname, p in model.named_parameters():
                    if p.requires_grad and pname in perturb_dirs:
                        p.add_(perturb_dirs[pname], alpha=perturb_gamma)

        # Evaluate
        if step % eval_interval == 0 or step == max_steps:
            model.eval()
            with torch.no_grad():
                seq_acc, _ = evaluate(model, device, n_samples=500)

            loss_val = loss.item()
            csv_writer.writerow([
                step, f"{loss_val:.6f}", f"{seq_acc:.4f}",
                f"{cur_lr:.2e}", f"{current_wd:.6f}",
            ])
            csv_file.flush()

            if seq_acc > best_acc:
                best_acc = seq_acc
                ckpt_dir = os.path.join(results_dir, "checkpoints")
                os.makedirs(ckpt_dir, exist_ok=True)
                ckpt_path = os.path.join(ckpt_dir, f"{run_name}_s{seed}_best.pt")
                torch.save({"model_state_dict": model.state_dict(),
                            "accuracy": best_acc, "step": step,
                            "config": config_name, "algo": algo_name,
                            "seed": seed}, ckpt_path)

            # Periodic checkpoint every 10K steps
            if step > 0 and step % 10000 == 0:
                ckpt_dir = os.path.join(results_dir, "checkpoints")
                os.makedirs(ckpt_dir, exist_ok=True)
                ckpt_path = os.path.join(ckpt_dir, f"{run_name}_s{seed}_step{step}.pt")
                torch.save({"model_state_dict": model.state_dict(),
                            "accuracy": seq_acc, "step": step,
                            "config": config_name, "algo": algo_name,
                            "seed": seed}, ckpt_path)

            # Adaptive WD
            if adaptive_wd:
                if seq_acc > 0.01 and "1pct" not in wd_thresholds:
                    wd_thresholds.add("1pct")
                    current_wd *= 0.1
                    for pg in optimizer.param_groups:
                        pg["weight_decay"] = current_wd
                if seq_acc > 0.05 and "5pct" not in wd_thresholds:
                    wd_thresholds.add("5pct")
                    current_wd *= 0.01
                    for pg in optimizer.param_groups:
                        pg["weight_decay"] = current_wd

            if step % (eval_interval * 5) == 0:
                elapsed = time.time() - t0
                print(f"    [{run_name} s{seed}] step {step}/{max_steps} "
                      f"loss={loss_val:.4f} acc={seq_acc:.4f} best={best_acc:.4f} "
                      f"[{elapsed:.0f}s]", flush=True)

            # Early stop if grokked
            if seq_acc >= 0.999:
                print(f"    [{run_name} s{seed}] GROKKED at step {step}!", flush=True)
                break

    csv_file.close()
    if grad_log_file is not None:
        if grad_buffer:
            grad_log_writer.writerows(grad_buffer)
            grad_buffer.clear()
        grad_log_file.close()
    elapsed = time.time() - t0

    return {
        "config": config_name,
        "algo": algo_name,
        "seed": seed,
        "params": n_params,
        "best_acc": best_acc,
        "elapsed": elapsed,
    }


def train_one_lbfgs(arch_cfg, algo_cfg, seed, max_steps, eval_interval, device, results_dir):
    """Train with L-BFGS (quasi-Newton, closure-based)."""
    config_name = arch_cfg["name"]
    algo_name = algo_cfg["name"]
    run_name = f"{config_name}__{algo_name}"

    csv_path = os.path.join(results_dir, f"{run_name}_s{seed}.csv")
    if os.path.exists(csv_path):
        try:
            with open(csv_path) as f:
                lines = f.readlines()
            if len(lines) > 2:
                return None
        except Exception:
            pass

    random.seed(seed)
    torch.manual_seed(seed)

    model, n_params = build_model(arch_cfg, device)

    lr = algo_cfg.get("lr", 1.0)
    batch_size = algo_cfg.get("batch_size", 512)

    optimizer = torch.optim.LBFGS(
        model.parameters(), lr=lr, max_iter=20,
        history_size=min(100, n_params * 5),
        line_search_fn="strong_wolfe",
    )

    os.makedirs(results_dir, exist_ok=True)
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["step", "loss", "acc", "lr", "wd"])

    t0 = time.time()
    best_acc = 0.0

    for step in range(1, max_steps + 1):
        model.train()
        full_seq, labels = generate_batch(batch_size, device)

        def closure():
            optimizer.zero_grad()
            logits = model(full_seq)
            shift_logits = logits[:, :-1, :].reshape(-1, VOCAB_SIZE)
            shift_labels = labels[:, 1:].reshape(-1)
            loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)
            loss.backward()
            return loss

        loss = optimizer.step(closure)

        if step % eval_interval == 0 or step == max_steps:
            model.eval()
            with torch.no_grad():
                seq_acc, _ = evaluate(model, device, n_samples=500)

            loss_val = loss.item()
            csv_writer.writerow([step, f"{loss_val:.6f}", f"{seq_acc:.4f}", f"{lr:.2e}", "0.0"])
            csv_file.flush()

            if seq_acc > best_acc:
                best_acc = seq_acc
                ckpt_dir = os.path.join(results_dir, "checkpoints")
                os.makedirs(ckpt_dir, exist_ok=True)
                torch.save({"model_state_dict": model.state_dict(),
                            "accuracy": best_acc, "step": step,
                            "config": config_name, "algo": algo_name,
                            "seed": seed},
                           os.path.join(ckpt_dir, f"{run_name}_s{seed}_best.pt"))

            # Periodic checkpoint every 10K steps
            if step > 0 and step % 10000 == 0:
                ckpt_dir = os.path.join(results_dir, "checkpoints")
                os.makedirs(ckpt_dir, exist_ok=True)
                torch.save({"model_state_dict": model.state_dict(),
                            "accuracy": seq_acc, "step": step,
                            "config": config_name, "algo": algo_name,
                            "seed": seed},
                           os.path.join(ckpt_dir, f"{run_name}_s{seed}_step{step}.pt"))

            if step % (eval_interval * 5) == 0:
                elapsed = time.time() - t0
                print(f"    [{run_name} s{seed}] step {step}/{max_steps} "
                      f"loss={loss_val:.4f} acc={seq_acc:.4f} best={best_acc:.4f} "
                      f"[{elapsed:.0f}s]", flush=True)

            if seq_acc >= 0.999:
                print(f"    [{run_name} s{seed}] GROKKED at step {step}!", flush=True)
                break

    csv_file.close()
    return {"config": config_name, "algo": algo_name, "seed": seed,
            "params": n_params, "best_acc": best_acc, "elapsed": time.time() - t0}


def train_one_cmaes(arch_cfg, algo_cfg, seed, max_steps, eval_interval, device, results_dir):
    """Train with CMA-ES (gradient-free, population-based)."""
    import cma

    config_name = arch_cfg["name"]
    algo_name = algo_cfg["name"]
    run_name = f"{config_name}__{algo_name}"

    csv_path = os.path.join(results_dir, f"{run_name}_s{seed}.csv")
    if os.path.exists(csv_path):
        try:
            with open(csv_path) as f:
                lines = f.readlines()
            if len(lines) > 2:
                return None
        except Exception:
            pass

    random.seed(seed)
    torch.manual_seed(seed)

    model, n_params = build_model(arch_cfg, device)
    sigma0 = algo_cfg.get("sigma0", 0.5)
    batch_size = algo_cfg.get("batch_size", 512)

    # Flatten initial params
    x0 = torch.cat([p.data.flatten() for p in model.parameters() if p.requires_grad]).cpu().numpy()
    trainable_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]

    def set_params(x_vec):
        """Set model params from flat numpy vector."""
        offset = 0
        for _, p in trainable_params:
            numel = p.data.numel()
            p.data.copy_(torch.tensor(x_vec[offset:offset + numel],
                         dtype=p.dtype).reshape(p.shape).to(device))
            offset += numel

    def eval_loss(x_vec):
        """Evaluate loss for a candidate solution."""
        set_params(x_vec)
        model.eval()
        with torch.no_grad():
            full_seq, labels = generate_batch(batch_size, device)
            logits = model(full_seq)
            shift_logits = logits[:, :-1, :].reshape(-1, VOCAB_SIZE)
            shift_labels = labels[:, 1:].reshape(-1)
            loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)
        return loss.item()

    # CMA-ES options
    popsize = max(4 + int(3 * math.log(n_params)), 8)
    opts = {
        "seed": seed,
        "maxiter": max_steps,
        "popsize": popsize,
        "verb_disp": 0,
        "verb_log": 0,
        "verb_filenameprefix": "/dev/null",
        "tolfun": 1e-8,
    }
    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)

    os.makedirs(results_dir, exist_ok=True)
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["step", "loss", "acc", "lr", "wd"])

    t0 = time.time()
    best_acc = 0.0
    step = 0

    while not es.stop() and step < max_steps:
        solutions = es.ask()
        fitnesses = [eval_loss(x) for x in solutions]
        es.tell(solutions, fitnesses)
        step += 1

        if step % eval_interval == 0 or step == max_steps:
            set_params(es.result.xbest)
            model.eval()
            with torch.no_grad():
                seq_acc, _ = evaluate(model, device, n_samples=500)

            loss_val = min(fitnesses)
            csv_writer.writerow([step, f"{loss_val:.6f}", f"{seq_acc:.4f}",
                                 f"{sigma0:.2e}", "0.0"])
            csv_file.flush()

            if seq_acc > best_acc:
                best_acc = seq_acc
                ckpt_dir = os.path.join(results_dir, "checkpoints")
                os.makedirs(ckpt_dir, exist_ok=True)
                torch.save({"model_state_dict": model.state_dict(),
                            "accuracy": best_acc, "step": step,
                            "config": config_name, "algo": algo_name,
                            "seed": seed},
                           os.path.join(ckpt_dir, f"{run_name}_s{seed}_best.pt"))

            # Periodic checkpoint every 10K steps
            if step > 0 and step % 10000 == 0:
                ckpt_dir = os.path.join(results_dir, "checkpoints")
                os.makedirs(ckpt_dir, exist_ok=True)
                torch.save({"model_state_dict": model.state_dict(),
                            "accuracy": seq_acc, "step": step,
                            "config": config_name, "algo": algo_name,
                            "seed": seed},
                           os.path.join(ckpt_dir, f"{run_name}_s{seed}_step{step}.pt"))

            if step % (eval_interval * 5) == 0:
                elapsed = time.time() - t0
                print(f"    [{run_name} s{seed}] gen {step}/{max_steps} "
                      f"loss={loss_val:.4f} acc={seq_acc:.4f} best={best_acc:.4f} "
                      f"sigma={es.sigma:.4f} [{elapsed:.0f}s]", flush=True)

            if seq_acc >= 0.999:
                print(f"    [{run_name} s{seed}] GROKKED at gen {step}!", flush=True)
                break

    csv_file.close()
    return {"config": config_name, "algo": algo_name, "seed": seed,
            "params": n_params, "best_acc": best_acc, "elapsed": time.time() - t0}


def train_one_newton(arch_cfg, algo_cfg, seed, max_steps, eval_interval, device, results_dir):
    """Train with exact Newton's method (full Hessian, tractable for <=50 params).

    Uses standard backward + per-grad-element backward for Hessian rows.
    Optional Tikhonov damping: (H + lambda*I)^-1 g.
    """
    config_name = arch_cfg["name"]
    algo_name = algo_cfg["name"]
    run_name = f"{config_name}__{algo_name}"

    csv_path = os.path.join(results_dir, f"{run_name}_s{seed}.csv")
    if os.path.exists(csv_path):
        try:
            with open(csv_path) as f:
                lines = f.readlines()
            if len(lines) > 2:
                return None
        except Exception:
            pass

    random.seed(seed)
    torch.manual_seed(seed)

    model, n_params = build_model(arch_cfg, device)
    lr = algo_cfg.get("lr", 0.1)
    batch_size = algo_cfg.get("batch_size", 512)
    damping = algo_cfg.get("newton_damping", 0.0)

    trainable = [p for p in model.parameters() if p.requires_grad]
    total_params = sum(p.numel() for p in trainable)

    os.makedirs(results_dir, exist_ok=True)
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["step", "loss", "acc", "lr", "wd"])

    t0 = time.time()
    best_acc = 0.0

    for step in range(1, max_steps + 1):
        model.train()
        full_seq, labels = generate_batch(batch_size, device)

        # Forward + backward with create_graph for 2nd order
        logits = model(full_seq)
        shift_logits = logits[:, :-1, :].reshape(-1, VOCAB_SIZE)
        shift_labels = labels[:, 1:].reshape(-1)
        loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)

        # Compute gradient with graph retained
        grads = torch.autograd.grad(loss, trainable, create_graph=True)
        flat_grad = torch.cat([g.flatten() for g in grads])

        # Compute Hessian row-by-row
        hessian = torch.zeros(total_params, total_params, device=device)
        for i in range(total_params):
            hess_row_grads = torch.autograd.grad(
                flat_grad[i], trainable, retain_graph=(i < total_params - 1))
            hessian[i] = torch.cat([h.flatten() for h in hess_row_grads])

        # Detach for the linear solve
        flat_grad_d = flat_grad.detach()

        # Add damping: (H + lambda*I)
        if damping > 0:
            hessian.add_(torch.eye(total_params, device=device) * damping)

        # Solve H * delta = -grad
        try:
            delta = torch.linalg.solve(hessian, -flat_grad_d)
        except Exception:
            delta = -flat_grad_d

        # Clip newton step
        delta_norm = delta.norm()
        if delta_norm > 10.0:
            delta.mul_(10.0 / delta_norm)

        # Apply update to model params
        offset = 0
        with torch.no_grad():
            for p in trainable:
                numel = p.numel()
                p.add_(lr * delta[offset:offset + numel].reshape(p.shape))
                offset += numel

        # Eval
        if step % eval_interval == 0 or step == max_steps:
            model.eval()
            with torch.no_grad():
                seq_acc, _ = evaluate(model, device, n_samples=500)

            loss_val = loss.item()
            csv_writer.writerow([step, f"{loss_val:.6f}", f"{seq_acc:.4f}", f"{lr:.2e}", "0.0"])
            csv_file.flush()

            if seq_acc > best_acc:
                best_acc = seq_acc
                ckpt_dir = os.path.join(results_dir, "checkpoints")
                os.makedirs(ckpt_dir, exist_ok=True)
                torch.save({"model_state_dict": model.state_dict(),
                            "accuracy": best_acc, "step": step,
                            "config": config_name, "algo": algo_name,
                            "seed": seed},
                           os.path.join(ckpt_dir, f"{run_name}_s{seed}_best.pt"))

            # Periodic checkpoint every 10K steps
            if step > 0 and step % 10000 == 0:
                ckpt_dir = os.path.join(results_dir, "checkpoints")
                os.makedirs(ckpt_dir, exist_ok=True)
                torch.save({"model_state_dict": model.state_dict(),
                            "accuracy": seq_acc, "step": step,
                            "config": config_name, "algo": algo_name,
                            "seed": seed},
                           os.path.join(ckpt_dir, f"{run_name}_s{seed}_step{step}.pt"))

            if step % (eval_interval * 5) == 0:
                elapsed = time.time() - t0
                print(f"    [{run_name} s{seed}] step {step}/{max_steps} "
                      f"loss={loss_val:.4f} acc={seq_acc:.4f} best={best_acc:.4f} "
                      f"[{elapsed:.0f}s]", flush=True)

            if seq_acc >= 0.999:
                print(f"    [{run_name} s{seed}] GROKKED at step {step}!", flush=True)
                break

    csv_file.close()
    return {"config": config_name, "algo": algo_name, "seed": seed,
            "params": n_params, "best_acc": best_acc, "elapsed": time.time() - t0}


def run_worker(args):
    """Multiprocessing worker."""
    arch_cfg, algo_cfg, seed, max_steps, eval_interval, device, results_dir = args
    try:
        opt_type = algo_cfg.get("optimizer", "adamw")
        if opt_type == "lbfgs":
            return train_one_lbfgs(arch_cfg, algo_cfg, seed, max_steps, eval_interval, device, results_dir)
        elif opt_type == "cmaes":
            return train_one_cmaes(arch_cfg, algo_cfg, seed, max_steps, eval_interval, device, results_dir)
        elif opt_type == "newton":
            return train_one_newton(arch_cfg, algo_cfg, seed, max_steps, eval_interval, device, results_dir)
        else:
            return train_one(arch_cfg, algo_cfg, seed, max_steps, eval_interval, device, results_dir)
    except Exception as e:
        run_name = f"{arch_cfg['name']}__{algo_cfg['name']}"
        print(f"    [{run_name} s{seed}] ERROR: {e}", flush=True)
        return None


# ============================================================================
# Summary
# ============================================================================

def print_summary(results_dir):
    """Print summary from all CSV files in results_dir."""
    if not os.path.isdir(results_dir):
        print(f"No results directory: {results_dir}")
        return

    from collections import defaultdict
    data = defaultdict(lambda: defaultdict(list))

    for fname in sorted(os.listdir(results_dir)):
        if not fname.endswith(".csv") or fname == "summary.csv":
            continue
        base = fname[:-4]
        parts = base.rsplit("_s", 1)
        if len(parts) != 2:
            continue
        run_name = parts[0]
        try:
            seed = int(parts[1])
        except ValueError:
            continue

        csv_path = os.path.join(results_dir, fname)
        best_acc = 0.0
        try:
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        acc = float(row["acc"])
                        best_acc = max(best_acc, acc)
                    except (ValueError, KeyError):
                        pass
        except Exception:
            continue

        data[run_name][seed] = best_acc

    if not data:
        print("No results found.")
        return

    # Group by config
    from collections import defaultdict as dd
    config_results = dd(list)
    for run_name, seeds in sorted(data.items()):
        parts = run_name.split("__", 1)
        if len(parts) == 2:
            config, algo = parts
        else:
            config = run_name
            algo = "unknown"

        accs = sorted(seeds.values(), reverse=True)
        n_seeds = len(accs)
        n_alive = sum(1 for a in accs if a > 0.01)
        n_grokked = sum(1 for a in accs if a > 0.95)
        best = accs[0] if accs else 0
        median = accs[len(accs)//2] if accs else 0

        config_results[config].append({
            "algo": algo,
            "n_seeds": n_seeds,
            "n_alive": n_alive,
            "n_grokked": n_grokked,
            "best": best,
            "median": median,
        })

    # Print
    print(f"\n{'='*90}")
    print(f"Sub-50p Sweep Results ({results_dir})")
    print(f"{'='*90}")

    for config in sorted(config_results.keys()):
        results = config_results[config]
        results.sort(key=lambda r: r["best"], reverse=True)

        print(f"\n--- {config} ---")
        print(f"{'Algorithm':<25} {'Seeds':>5} {'Alive':>5} {'Grok':>5} {'Best':>8} {'Median':>8}")
        for r in results:
            print(f"  {r['algo']:<23} {r['n_seeds']:>5} {r['n_alive']:>5} "
                  f"{r['n_grokked']:>5} {r['best']:>7.1%} {r['median']:>7.1%}")

    # Overall summary
    print(f"\n{'='*90}")
    print("OVERALL: Best per architecture")
    print(f"{'Config':<45} {'Best Algo':<25} {'Best':>8} {'Alive':>10}")
    for config in sorted(config_results.keys()):
        results = config_results[config]
        results.sort(key=lambda r: r["best"], reverse=True)
        best = results[0]
        total_alive = sum(r["n_alive"] for r in results)
        total_seeds = sum(r["n_seeds"] for r in results)
        print(f"  {config:<43} {best['algo']:<23} {best['best']:>7.1%} "
              f"{total_alive}/{total_seeds}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Sub-50p massive architecture sweep")
    parser.add_argument("--max-steps", type=int, default=100000)
    parser.add_argument("--eval-interval", type=int, default=2000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--config", default=None, help="Run only this config name (or prefix)")
    parser.add_argument("--algo", default=None, help="Run only this algorithm name (or prefix)")
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--max-parallel", type=int, default=None,
                        help="Max parallel processes (default: auto)")
    parser.add_argument("--max-params", type=int, default=None,
                        help="Only run configs with <= this many params")
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    parser.add_argument("--summary", action="store_true", help="Print summary and exit")
    parser.add_argument("--dry-run", action="store_true", help="Show what would run")
    args = parser.parse_args()

    if args.summary:
        print_summary(args.results_dir)
        return

    # Filter configs
    configs = CONFIGS
    if args.max_params:
        configs = [c for c in configs if c.get("params", 999) <= args.max_params]
    if args.config:
        configs = [c for c in CONFIGS if args.config in c["name"]]
        if not configs:
            print(f"No configs matching '{args.config}'")
            print(f"Available: {[c['name'] for c in CONFIGS]}")
            return

    # Filter algorithms
    algos = ALGORITHMS
    if args.algo:
        algos = [a for a in ALGORITHMS if args.algo in a["name"]]
        if not algos:
            print(f"No algorithms matching '{args.algo}'")
            print(f"Available: {[a['name'] for a in ALGORITHMS]}")
            return

    # Seeds
    seeds = args.seeds if args.seeds else SEEDS

    # Build task list
    tasks = []
    for cfg in configs:
        for algo in algos:
            for seed in seeds:
                run_name = f"{cfg['name']}__{algo['name']}"
                csv_path = os.path.join(args.results_dir, f"{run_name}_s{seed}.csv")
                if os.path.exists(csv_path):
                    try:
                        with open(csv_path) as f:
                            lines = f.readlines()
                        if len(lines) > 2:
                            continue
                    except Exception:
                        pass
                tasks.append((cfg, algo, seed, args.max_steps, args.eval_interval,
                             args.device, args.results_dir))

    # Sort tasks by param count (smallest first), then algo, then seed
    tasks.sort(key=lambda t: (t[0].get("params", 999), t[1]["name"], t[2]))

    total_possible = len(configs) * len(algos) * len(seeds)
    print(f"Sub-50p Sweep: {len(configs)} configs x {len(algos)} algos x {len(seeds)} seeds = {total_possible} total")
    print(f"Remaining: {len(tasks)} ({total_possible - len(tasks)} already done)")
    print(f"Steps: {args.max_steps}, eval every {args.eval_interval}, device: {args.device}")
    print()

    # Verify param counts
    for cfg in configs:
        model, n_params = build_model(cfg, "cpu")
        expected = cfg.get("params", "?")
        status = "OK" if n_params == expected else f"MISMATCH (expected {expected})"
        print(f"  {cfg['name']}: {n_params}p {status}")
        del model

    if args.dry_run:
        print(f"\nDry run: would launch {len(tasks)} tasks")
        for cfg, algo, seed, *_ in tasks[:10]:
            print(f"  {cfg['name']}__{algo['name']}_s{seed}")
        if len(tasks) > 10:
            print(f"  ... and {len(tasks) - 10} more")
        return

    if not tasks:
        print("\nAll tasks already completed!")
        print_summary(args.results_dir)
        return

    print()

    # Determine parallelism
    if args.max_parallel:
        n_workers = args.max_parallel
    elif args.device == "cpu":
        n_workers = min(len(tasks), max(1, multiprocessing.cpu_count() // 2))
    else:
        n_workers = min(len(tasks), 20)

    print(f"Launching {len(tasks)} tasks with {n_workers} workers...")

    # Sequential mode: run in-process (for debugging or single GPU)
    if n_workers == 1:
        os.environ["OMP_NUM_THREADS"] = "2"
        for i, task_args in enumerate(tasks):
            cfg, algo, seed = task_args[:3]
            print(f"\n[{i+1}/{len(tasks)}] {cfg['name']}__{algo['name']}_s{seed}", flush=True)
            result = run_worker(task_args)
            if result:
                status = "dead"
                if result["best_acc"] > 0.01:
                    status = "ALIVE"
                if result["best_acc"] > 0.5:
                    status = f"GROK {result['best_acc']:.1%}"
                print(f"  -> {result['best_acc']:.4f} ({status}) [{result['elapsed']:.0f}s]", flush=True)
        print(f"\nDone! Results in {args.results_dir}")
        print_summary(args.results_dir)
        return

    # Use subprocess-based parallelism to avoid torch multiprocessing issues
    script_path = os.path.abspath(__file__)
    active = {}  # pid -> (process, cfg_name, algo_name, seed)
    task_iter = iter(tasks)
    done = 0

    def launch_next():
        try:
            cfg, algo, seed, max_steps, eval_interval, device, results_dir = next(task_iter)
        except StopIteration:
            return False
        cmd = [
            PYTHON, script_path,
            "--config", cfg["name"],
            "--algo", algo["name"],
            "--seeds", str(seed),
            "--max-steps", str(max_steps),
            "--eval-interval", str(eval_interval),
            "--device", device,
            "--results-dir", results_dir,
            "--max-parallel", "1",
        ]
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = env.get("OMP_NUM_THREADS", "2")
        proc = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
        active[proc.pid] = (proc, cfg["name"], algo["name"], seed)
        return True

    def _reap_children(signum=None, frame=None):
        """Reap all child processes on exit to prevent zombies."""
        for pid, (proc, _, _, _) in list(active.items()):
            try:
                proc.terminate()
            except OSError:
                pass
        for pid, (proc, _, _, _) in list(active.items()):
            try:
                proc.wait(timeout=5)
            except Exception:
                try:
                    proc.kill()
                except OSError:
                    pass

    signal.signal(signal.SIGTERM, _reap_children)
    signal.signal(signal.SIGINT, _reap_children)

    # Fill initial batch
    for _ in range(n_workers):
        if not launch_next():
            break

    # Wait for completions and launch new tasks
    while active:
        for pid in list(active.keys()):
            proc, cfg_name, algo_name, seed = active[pid]
            ret = proc.poll()
            if ret is not None:
                # Reap the child properly to avoid zombies
                proc.wait()
                del active[pid]
                done += 1

                # Check result CSV
                run_name = f"{cfg_name}__{algo_name}"
                csv_path = os.path.join(args.results_dir, f"{run_name}_s{seed}.csv")
                best_acc = 0.0
                try:
                    with open(csv_path) as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            try:
                                best_acc = max(best_acc, float(row["acc"]))
                            except (ValueError, KeyError):
                                pass
                except Exception:
                    pass

                status = "dead"
                if best_acc > 0.01:
                    status = "ALIVE"
                if best_acc > 0.5:
                    status = f"GROK {best_acc:.1%}"

                print(f"  [{done}/{len(tasks)}] {run_name}_s{seed}: "
                      f"{best_acc:.4f} ({status})", flush=True)

                # Launch replacement
                launch_next()

        if active:
            time.sleep(2)

    print(f"\nDone! Results in {args.results_dir}")
    print_summary(args.results_dir)


if __name__ == "__main__":
    main()
