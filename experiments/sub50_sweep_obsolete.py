"""Massive sub-50p architecture sweep.

Tests all feasible sub-50p configurations with K=alpha*Q + gate=alpha*up tying
(NOT share_qk_norm which kills grokking). ~50 seeds × all promising optimizers × 100K steps.

Key configs:
  ff=2: 52p (arc+tieQO+K=aQ+gate=a*up+shnorm)
  ff=1: 49p (arc+tieQO+K=aQ+gate=a*up+shbnorm)
  ff=1: 46p (arc+tieQO+K=aQ+gate=a*up+shnorm)
  ff=1: 43p (arc+tieQO+K=aQ+gate=a*up+shnorm+tieKV)
  ff=2: 40p (arc+tieQO+K=aQ+gate=a*up+shnorm+tieKV)

Usage:
    python experiments/sub50_sweep.py                           # run all on CPU
    python experiments/sub50_sweep.py --device cuda             # run all on GPU
    python experiments/sub50_sweep.py --config 46p --algo adamw  # single config+algo
    python experiments/sub50_sweep.py --summary                 # print summary
    python experiments/sub50_sweep.py --max-parallel 30         # limit parallelism
"""

import argparse
import csv
import math
import multiprocessing
import os
import random
import sys
import time
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "sweep_results_sub50")


# ============================================================================
# Architecture configurations
# ============================================================================

# Base model args shared by all configs
BASE_ARGS = {
    "d_model": 3,
    "n_heads": 1,
    "n_kv_heads": 1,
    "head_dim": 4,
    "rope_theta": 3.0,
}

# Architecture configs: (name, model_args, tying_spec)
# tying_spec: dict with keys like 'k_alpha_q', 'gate_alpha', 'v_eq_q'
CONFIGS = [
    # ── ff=2 configs (more MLP capacity) ──
    # 55p reference (our world record architecture, without shbnorm → shnorm variant)
    {
        "name": "52p_ff2_tieQO_KaQ_gateA_shnorm",
        "params": 52,
        "model_args": {**BASE_ARGS, "ff": 2, "tie_qo": True, "share_norms": True},
        "tying": {"k_alpha_q": True, "gate_alpha": True},
    },
    {
        "name": "40p_ff2_tieQO_KaQ_gateA_shnorm_VeqQ",
        "params": 40,
        "model_args": {**BASE_ARGS, "ff": 2, "tie_qo": True, "share_norms": True},
        "tying": {"k_alpha_q": True, "gate_alpha": True, "v_eq_q": True},
        # V = Q, K = alpha*Q — only q_proj matrix remains for attention
    },
    {
        "name": "57p_ff2_tieQO_KaQ_shnorm",
        "params": 57,
        "model_args": {**BASE_ARGS, "ff": 2, "tie_qo": True, "share_norms": True},
        "tying": {"k_alpha_q": True},
        # No gate tying — more MLP freedom
    },
    {
        "name": "45p_ff2_tieQO_KaQ_shnorm_VeqQ",
        "params": 45,
        "model_args": {**BASE_ARGS, "ff": 2, "tie_qo": True, "share_norms": True},
        "tying": {"k_alpha_q": True, "v_eq_q": True},
        # No gate tying + V=Q
    },

    # ── ff=2 with shbnorm (fills 40-45 gap) ──
    # 43p = 55p world record arch + V=Q — the most promising sub-45 config!
    {
        "name": "43p_ff2_tieQO_KaQ_gateA_shbnorm_VeqQ",
        "params": 43,
        "model_args": {**BASE_ARGS, "ff": 2, "tie_qo": True, "share_block_norms": True},
        "tying": {"k_alpha_q": True, "gate_alpha": True, "v_eq_q": True},
    },
    # 48p = ff=2 + V=Q + shbnorm, no gate tying — more MLP freedom
    {
        "name": "48p_ff2_tieQO_KaQ_shbnorm_VeqQ",
        "params": 48,
        "model_args": {**BASE_ARGS, "ff": 2, "tie_qo": True, "share_block_norms": True},
        "tying": {"k_alpha_q": True, "v_eq_q": True},
    },

    # 45p with K=rotation(Q) instead of K=alpha*Q
    {
        "name": "45p_ff2_tieQO_KrotQ_shnorm_VeqQ",
        "params": 45,
        "model_args": {**BASE_ARGS, "ff": 2, "tie_qo": True, "share_norms": True},
        "tying": {"k_rot_q": True, "v_eq_q": True},
        # K = rotation(Q) instead of alpha*Q, no gate tying
    },

    # ── ff=2 with K rotation (MicroAdder-inspired) ──
    {
        "name": "52p_ff2_tieQO_KrotQ_gateA_shnorm",
        "params": 52,
        "model_args": {**BASE_ARGS, "ff": 2, "tie_qo": True, "share_norms": True},
        "tying": {"k_rot_q": True, "gate_alpha": True},
    },
    {
        "name": "40p_ff2_tieQO_KrotQ_gateA_shnorm_VeqQ",
        "params": 40,
        "model_args": {**BASE_ARGS, "ff": 2, "tie_qo": True, "share_norms": True},
        "tying": {"k_rot_q": True, "gate_alpha": True, "v_eq_q": True},
    },
    # 43p KrotQ variant
    {
        "name": "43p_ff2_tieQO_KrotQ_gateA_shbnorm_VeqQ",
        "params": 43,
        "model_args": {**BASE_ARGS, "ff": 2, "tie_qo": True, "share_block_norms": True},
        "tying": {"k_rot_q": True, "gate_alpha": True, "v_eq_q": True},
    },

    # ── ff=1 configs (minimal MLP, hardest to grok) ──
    {
        "name": "49p_ff1_tieQO_KaQ_gateA_shbnorm",
        "params": 49,
        "model_args": {**BASE_ARGS, "ff": 1, "tie_qo": True, "share_block_norms": True},
        "tying": {"k_alpha_q": True, "gate_alpha": True},
    },
    {
        "name": "46p_ff1_tieQO_KaQ_gateA_shnorm",
        "params": 46,
        "model_args": {**BASE_ARGS, "ff": 1, "tie_qo": True, "share_norms": True},
        "tying": {"k_alpha_q": True, "gate_alpha": True},
    },
    {
        "name": "34p_ff1_tieQO_KaQ_gateA_shnorm_VeqQ",
        "params": 34,
        "model_args": {**BASE_ARGS, "ff": 1, "tie_qo": True, "share_norms": True},
        "tying": {"k_alpha_q": True, "gate_alpha": True, "v_eq_q": True},
        # Extreme: only q_proj for attention, 1 scalar for K, 1 scalar for gate
    },

    # ── ff=1 without gate tying ──
    {
        "name": "48p_ff1_tieQO_KaQ_shnorm",
        "params": 48,
        "model_args": {**BASE_ARGS, "ff": 1, "tie_qo": True, "share_norms": True},
        "tying": {"k_alpha_q": True},
    },
    {
        "name": "36p_ff1_tieQO_KaQ_shnorm_VeqQ",
        "params": 36,
        "model_args": {**BASE_ARGS, "ff": 1, "tie_qo": True, "share_norms": True},
        "tying": {"k_alpha_q": True, "v_eq_q": True},
    },

    # ── ff=1 with shbnorm (fills more gaps) ──
    # 39p = ff=1 + K=aQ + V=Q + shbnorm, no gate tying
    {
        "name": "39p_ff1_tieQO_KaQ_shbnorm_VeqQ",
        "params": 39,
        "model_args": {**BASE_ARGS, "ff": 1, "tie_qo": True, "share_block_norms": True},
        "tying": {"k_alpha_q": True, "v_eq_q": True},
    },
    # 37p = ff=1 + K=aQ + gate=a*up + V=Q + shbnorm
    {
        "name": "37p_ff1_tieQO_KaQ_gateA_shbnorm_VeqQ",
        "params": 37,
        "model_args": {**BASE_ARGS, "ff": 1, "tie_qo": True, "share_block_norms": True},
        "tying": {"k_alpha_q": True, "gate_alpha": True, "v_eq_q": True},
    },

    # ── ff=1 with K rotation ──
    {
        "name": "46p_ff1_tieQO_KrotQ_gateA_shnorm",
        "params": 46,
        "model_args": {**BASE_ARGS, "ff": 1, "tie_qo": True, "share_norms": True},
        "tying": {"k_rot_q": True, "gate_alpha": True},
    },
    {
        "name": "34p_ff1_tieQO_KrotQ_gateA_shnorm_VeqQ",
        "params": 34,
        "model_args": {**BASE_ARGS, "ff": 1, "tie_qo": True, "share_norms": True},
        "tying": {"k_rot_q": True, "gate_alpha": True, "v_eq_q": True},
    },

    # ── ff=1 with down=up^T (NEW — avoids V=Q death trap) ──
    # 51p: K=aQ + down=up^T, NO V=Q! Independent V, structured MLP
    {
        "name": "51p_ff1_tieQO_KaQ_downUpT",
        "params": 51,
        "model_args": {**BASE_ARGS, "ff": 1, "tie_qo": True},
        "tying": {"k_alpha_q": True, "down_eq_upT": True},
    },
    # 49p: K=aQ + gate=a*up + down=up^T (no V=Q, no norm sharing)
    {
        "name": "49p_ff1_tieQO_KaQ_gateA_downUpT",
        "params": 49,
        "model_args": {**BASE_ARGS, "ff": 1, "tie_qo": True},
        "tying": {"k_alpha_q": True, "gate_alpha": True, "down_eq_upT": True},
    },
    # 45p: K=aQ + shnorm + down=up^T (no V=Q, no gate tying — sub-47!)
    {
        "name": "45p_ff1_tieQO_KaQ_shnorm_downUpT",
        "params": 45,
        "model_args": {**BASE_ARGS, "ff": 1, "tie_qo": True, "share_norms": True},
        "tying": {"k_alpha_q": True, "down_eq_upT": True},
    },
    # 43p: K=aQ + gate + shnorm + down=up^T (no V=Q — sub-45!)
    {
        "name": "43p_ff1_tieQO_KaQ_gateA_shnorm_downUpT",
        "params": 43,
        "model_args": {**BASE_ARGS, "ff": 1, "tie_qo": True, "share_norms": True},
        "tying": {"k_alpha_q": True, "gate_alpha": True, "down_eq_upT": True},
    },
    # 39p: K=aQ + V=Q + down=up^T (no norm sharing)
    {
        "name": "39p_ff1_tieQO_KaQ_VeqQ_downUpT",
        "params": 39,
        "model_args": {**BASE_ARGS, "ff": 1, "tie_qo": True},
        "tying": {"k_alpha_q": True, "v_eq_q": True, "down_eq_upT": True},
    },
    # 33p: K=aQ + shnorm + V=Q + down=up^T
    {
        "name": "33p_ff1_tieQO_KaQ_shnorm_VeqQ_downUpT",
        "params": 33,
        "model_args": {**BASE_ARGS, "ff": 1, "tie_qo": True, "share_norms": True},
        "tying": {"k_alpha_q": True, "v_eq_q": True, "down_eq_upT": True},
    },
    # 31p: K=aQ + gate + shnorm + V=Q + down=up^T (extreme)
    {
        "name": "31p_ff1_tieQO_KaQ_gateA_shnorm_VeqQ_downUpT",
        "params": 31,
        "model_args": {**BASE_ARGS, "ff": 1, "tie_qo": True, "share_norms": True},
        "tying": {"k_alpha_q": True, "gate_alpha": True, "v_eq_q": True, "down_eq_upT": True},
    },

    # ── ff=2 sub-45p variants (from 45p winner: K=aQ, V=Q, tieQO, shnorm) ──

    # 44p: drop k_alpha scalar (K=Q instead of K=aQ)
    {
        "name": "44p_ff2_tieQO_KeqQ_shnorm_VeqQ",
        "params": 44,
        "model_args": {**BASE_ARGS, "ff": 2, "tie_qo": True, "share_norms": True},
        "tying": {"k_eq_q": True, "v_eq_q": True},
    },
    # 41p: tie QK norms (q_norm = k_norm, saves 4p)
    {
        "name": "41p_ff2_tieQO_KaQ_shnorm_VeqQ_tieQKnorm",
        "params": 41,
        "model_args": {**BASE_ARGS, "ff": 2, "tie_qo": True, "share_norms": True},
        "tying": {"k_alpha_q": True, "v_eq_q": True, "tie_qk_norm": True},
    },
    {
        "name": "41p_ff2_tieQO_KrotQ_shnorm_VeqQ_tieQKnorm",
        "params": 41,
        "model_args": {**BASE_ARGS, "ff": 2, "tie_qo": True, "share_norms": True},
        "tying": {"k_rot_q": True, "v_eq_q": True, "tie_qk_norm": True},
        # K = rotation(Q) + shared QK norms — rotation more expressive than alpha
    },
    # 40p: K=Q + tie QK norms (drop alpha + tie norms)
    {
        "name": "40p_ff2_tieQO_KeqQ_shnorm_VeqQ_tieQKnorm",
        "params": 40,
        "model_args": {**BASE_ARGS, "ff": 2, "tie_qo": True, "share_norms": True},
        "tying": {"k_eq_q": True, "v_eq_q": True, "tie_qk_norm": True},
    },
    # 39p: down=up^T (saves 6p from 45p)
    {
        "name": "39p_ff2_tieQO_KaQ_shnorm_VeqQ_downUpT",
        "params": 39,
        "model_args": {**BASE_ARGS, "ff": 2, "tie_qo": True, "share_norms": True},
        "tying": {"k_alpha_q": True, "v_eq_q": True, "down_eq_upT": True},
    },
    # 39p: gate=up identity (saves 6p from 45p)
    {
        "name": "39p_ff2_tieQO_KaQ_shnorm_VeqQ_gateEqUp",
        "params": 39,
        "model_args": {**BASE_ARGS, "ff": 2, "tie_qo": True, "share_norms": True},
        "tying": {"k_alpha_q": True, "v_eq_q": True, "gate_eq_up": True},
    },
    # 38p: down=up^T + K=Q (drop alpha)
    {
        "name": "38p_ff2_tieQO_KeqQ_shnorm_VeqQ_downUpT",
        "params": 38,
        "model_args": {**BASE_ARGS, "ff": 2, "tie_qo": True, "share_norms": True},
        "tying": {"k_eq_q": True, "v_eq_q": True, "down_eq_upT": True},
    },
    # 35p: down=up^T + tie QK norm
    {
        "name": "35p_ff2_tieQO_KaQ_shnorm_VeqQ_downUpT_tieQKnorm",
        "params": 35,
        "model_args": {**BASE_ARGS, "ff": 2, "tie_qo": True, "share_norms": True},
        "tying": {"k_alpha_q": True, "v_eq_q": True, "down_eq_upT": True, "tie_qk_norm": True},
    },
    # 34p: down=up^T + K=Q + tie QK norm (all three savings)
    {
        "name": "34p_ff2_tieQO_KeqQ_shnorm_VeqQ_downUpT_tieQKnorm",
        "params": 34,
        "model_args": {**BASE_ARGS, "ff": 2, "tie_qo": True, "share_norms": True},
        "tying": {"k_eq_q": True, "v_eq_q": True, "down_eq_upT": True, "tie_qk_norm": True},
    },
    # 33p: down=up^T + gate=up + tie QK norm
    {
        "name": "33p_ff2_tieQO_KaQ_shnorm_VeqQ_downUpT_gateEqUp_tieQKnorm",
        "params": 33,
        "model_args": {**BASE_ARGS, "ff": 2, "tie_qo": True, "share_norms": True},
        "tying": {"k_alpha_q": True, "v_eq_q": True, "down_eq_upT": True, "gate_eq_up": True, "tie_qk_norm": True},
    },
]


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

    # ── NEW: Research-derived optimizers ──

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
    # gamma tied to LR (decays 10x slower), scale = fraction of ||W|| (default 0.1)
    # Full cycle = perturbation_phase + recovery_phase (equal length)
    {"name": "adamw+cdp", "optimizer": "adamw", "lr": 0.01, "weight_decay": 0.01,
     "cdp": True, "cdp_scale": 0.1, "cdp_cycle": 10000, "cdp_target": "proj"},
    {"name": "adamw+cdp+gf", "optimizer": "adamw", "lr": 0.01, "weight_decay": 0.01,
     "cdp": True, "cdp_scale": 0.1, "cdp_cycle": 10000, "cdp_target": "proj",
     "grokfast": True, "gf_alpha": 0.98, "gf_lambda": 2.0},
    {"name": "adamw+cdp_attn", "optimizer": "adamw", "lr": 0.01, "weight_decay": 0.01,
     "cdp": True, "cdp_scale": 0.1, "cdp_cycle": 10000, "cdp_target": "attn"},
    {"name": "adamw+cdp_all", "optimizer": "adamw", "lr": 0.01, "weight_decay": 0.01,
     "cdp": True, "cdp_scale": 0.1, "cdp_cycle": 10000, "cdp_target": "all"},

    # ── Low-LR + high-WD variants (for ultra-small models with scale degeneracy) ──

    # Low LR (0.001) + grokfast: 10x smaller steps, WD becomes proportionally stronger
    {"name": "adamw_lr001+gf", "optimizer": "adamw", "lr": 0.001, "weight_decay": 0.01,
     "grokfast": True},

    # Low LR + high WD (0.1): aggressively penalize weight growth
    {"name": "adamw_lr001_wd01+gf", "optimizer": "adamw", "lr": 0.001, "weight_decay": 0.1,
     "grokfast": True},

    # Medium LR (0.003) + high WD: compromise
    {"name": "adamw_lr003_wd01+gf", "optimizer": "adamw", "lr": 0.003, "weight_decay": 0.1,
     "grokfast": True},

    # Standard LR + high WD: same steps but stronger regularization
    {"name": "adamw_wd01+gf", "optimizer": "adamw", "lr": 0.01, "weight_decay": 0.1,
     "grokfast": True},

    # Very high WD (0.3) + low LR: extreme regularization
    {"name": "adamw_lr001_wd03+gf", "optimizer": "adamw", "lr": 0.001, "weight_decay": 0.3,
     "grokfast": True},

    # ── Simple baselines (no WD, no adaptive LR) ──
    {"name": "sgd", "optimizer": "sgd", "lr": 0.01, "weight_decay": 0.0},
    {"name": "sgd_lr001", "optimizer": "sgd", "lr": 0.001, "weight_decay": 0.0},
    {"name": "sgd_mom", "optimizer": "sgd_momentum", "lr": 0.01, "weight_decay": 0.0},
    {"name": "adam_nowd", "optimizer": "adam", "lr": 0.01, "weight_decay": 0.0},
    {"name": "adam_nowd+gf", "optimizer": "adam", "lr": 0.01, "weight_decay": 0.0,
     "grokfast": True},

    # ── Explicit L2 regularization (added to loss, flows through Adam momentum) ──
    # Different from weight decay because L2 gradient goes through Adam's adaptive LR
    {"name": "adamw+l2_01+gf", "optimizer": "adamw", "lr": 0.01, "weight_decay": 0.0,
     "l2_lambda": 0.1, "grokfast": True},
    {"name": "adamw+l2_001+gf", "optimizer": "adamw", "lr": 0.01, "weight_decay": 0.0,
     "l2_lambda": 0.01, "grokfast": True},
    {"name": "adamw+l2_1+gf", "optimizer": "adamw", "lr": 0.01, "weight_decay": 0.0,
     "l2_lambda": 1.0, "grokfast": True},

    # ── 2nd-order / derivative-free methods (for ultra-small models ≤50 params) ──

    # L-BFGS: quasi-Newton, approximates inverse Hessian
    {"name": "lbfgs", "optimizer": "lbfgs", "lr": 1.0, "batch_size": 512},
    {"name": "lbfgs_lr01", "optimizer": "lbfgs", "lr": 0.1, "batch_size": 512},
    {"name": "lbfgs_lr001", "optimizer": "lbfgs", "lr": 0.01, "batch_size": 512},

    # CMA-ES: covariance matrix adaptation evolution strategy (gradient-free)
    {"name": "cmaes", "optimizer": "cmaes", "sigma0": 0.5, "batch_size": 512},
    {"name": "cmaes_sig01", "optimizer": "cmaes", "sigma0": 0.1, "batch_size": 512},
    {"name": "cmaes_sig1", "optimizer": "cmaes", "sigma0": 1.0, "batch_size": 512},

    # ── SAM (Sharpness-Aware Minimization) ──
    # Double forward+backward: ascend by rho * grad/||grad||, then descend on perturbed loss
    {"name": "sam", "optimizer": "adamw", "lr": 0.01, "weight_decay": 0.01,
     "sam": True, "sam_rho": 0.05},
    {"name": "sam+gf", "optimizer": "adamw", "lr": 0.01, "weight_decay": 0.01,
     "sam": True, "sam_rho": 0.05, "grokfast": True},
    {"name": "sam_rho02", "optimizer": "adamw", "lr": 0.01, "weight_decay": 0.01,
     "sam": True, "sam_rho": 0.2},
    {"name": "sam_rho02+gf", "optimizer": "adamw", "lr": 0.01, "weight_decay": 0.01,
     "sam": True, "sam_rho": 0.2, "grokfast": True},

    # ── Per-param LR groups ──
    # Gate/scalar params get lower LR, norms get higher LR
    {"name": "pplr", "optimizer": "adamw_pplr", "lr": 0.01, "weight_decay": 0.01,
     "lr_gate_mult": 0.1, "lr_norm_mult": 3.0},
    {"name": "pplr+gf", "optimizer": "adamw_pplr", "lr": 0.01, "weight_decay": 0.01,
     "lr_gate_mult": 0.1, "lr_norm_mult": 3.0, "grokfast": True},
    {"name": "pplr_gm03+gf", "optimizer": "adamw_pplr", "lr": 0.01, "weight_decay": 0.01,
     "lr_gate_mult": 0.3, "lr_norm_mult": 3.0, "grokfast": True},

    # ── Weight norm constraint (OmniGrok) ──
    # After optimizer step, project all weight matrices onto norm ball of radius R
    {"name": "omnigrok", "optimizer": "adamw", "lr": 0.01, "weight_decay": 0.0,
     "norm_constraint": True, "norm_radius": 1.0},
    {"name": "omnigrok+gf", "optimizer": "adamw", "lr": 0.01, "weight_decay": 0.0,
     "norm_constraint": True, "norm_radius": 1.0, "grokfast": True},
    {"name": "omnigrok_r05+gf", "optimizer": "adamw", "lr": 0.01, "weight_decay": 0.0,
     "norm_constraint": True, "norm_radius": 0.5, "grokfast": True},
    {"name": "omnigrok_r2+gf", "optimizer": "adamw", "lr": 0.01, "weight_decay": 0.0,
     "norm_constraint": True, "norm_radius": 2.0, "grokfast": True},

    # ── Full Newton (exact Hessian, tractable for ≤50 params) ──
    {"name": "newton", "optimizer": "newton", "lr": 0.1, "batch_size": 512},
    {"name": "newton_lr01", "optimizer": "newton", "lr": 0.01, "batch_size": 512},
    {"name": "newton_lr1", "optimizer": "newton", "lr": 1.0, "batch_size": 512},
    {"name": "newton_damp", "optimizer": "newton", "lr": 0.1, "batch_size": 512,
     "newton_damping": 0.1},
]

SEEDS = list(range(50))  # 50 seeds


# ============================================================================
# Model building with monkey-patched tying
# ============================================================================

def count_params(model):
    """Count unique parameters (handles tied weights)."""
    seen = set()
    total = 0
    for p in model.parameters():
        if p.data_ptr() not in seen:
            seen.add(p.data_ptr())
            total += p.numel()
    return total


def apply_k_alpha_q(model):
    """K = alpha * Q: replace k_proj with scalar times q_proj."""
    attn = model.block.attn
    alpha = nn.Parameter(torch.tensor(1.0))
    attn.k_alpha = alpha
    if hasattr(attn, 'k_proj'):
        del attn.k_proj

    def k_alpha_q_forward(self, x, mask=None, _alpha=alpha):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = (self.q_proj(x) * _alpha).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v_proj = self.q_proj if self.tie_kv else self.v_proj
        v = v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q = apply_rope(q, self.rope_cos, self.rope_sin)
        k = apply_rope(k, self.rope_cos, self.rope_sin)

        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = (q @ k.transpose(-2, -1)) * scale
        if mask is not None:
            scores = scores + mask[:T, :T]
        attn_weights = F.softmax(scores, dim=-1)

        out = (attn_weights @ v).transpose(1, 2).contiguous().view(B, T, -1)
        if self.tie_qo:
            return F.linear(out, self.q_proj.weight.t())
        return self.o_proj(out)

    attn.forward = types.MethodType(k_alpha_q_forward, attn)
    return model


def apply_k_rot_q(model):
    """K = R(phi)*Q: rotation-based K tying."""
    attn = model.block.attn
    theta_angle = nn.Parameter(torch.tensor(-0.5))
    attn.k_rot_theta = theta_angle
    if hasattr(attn, 'k_proj'):
        del attn.k_proj

    def k_rot_q_forward(self, x, mask=None, _theta=theta_angle):
        B, T, _ = x.shape
        shared = self.q_proj(x)

        k = shared.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q_raw = shared.clone()
        cos_t = torch.cos(_theta)
        sin_t = torch.sin(_theta)
        q0 = q_raw[..., 0] * cos_t - q_raw[..., 1] * sin_t
        q1 = q_raw[..., 0] * sin_t + q_raw[..., 1] * cos_t
        q_raw = q_raw.clone()
        q_raw[..., 0] = q0
        q_raw[..., 1] = q1
        q = q_raw.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        v_proj = self.q_proj if self.tie_kv else self.v_proj
        v = v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q = apply_rope(q, self.rope_cos, self.rope_sin)
        k = apply_rope(k, self.rope_cos, self.rope_sin)

        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = (q @ k.transpose(-2, -1)) * scale
        if mask is not None:
            scores = scores + mask[:T, :T]
        attn_weights = F.softmax(scores, dim=-1)

        out = (attn_weights @ v).transpose(1, 2).contiguous().view(B, T, -1)
        if self.tie_qo:
            return F.linear(out, self.q_proj.weight.t())
        return self.o_proj(out)

    attn.forward = types.MethodType(k_rot_q_forward, attn)
    return model


def apply_v_eq_q(model):
    """V = Q: tie V projection to Q projection."""
    attn = model.block.attn

    if hasattr(attn, 'v_proj'):
        del attn.v_proj

    # We need to modify the forward to use q_proj for V
    # If k_alpha_q or k_rot_q is already applied, we need to handle it
    # This is applied AFTER k_alpha_q, so the forward already handles K
    # We just need to mark that V should use q_proj
    attn.tie_kv = True  # This tells the already-patched forward to use q_proj for V
    return model


def apply_gate_alpha(model):
    """gate = alpha * up: replace gate_proj with scalar times up_proj."""
    mlp = model.block.mlp
    alpha = nn.Parameter(torch.tensor(1.0))
    mlp.gate_alpha = alpha
    if hasattr(mlp, 'gate_proj'):
        del mlp.gate_proj

    def gate_alpha_forward(self, x, _alpha=alpha):
        gate_out = self._act(self.up_proj(x) * _alpha)
        up_out = self.up_proj(x)
        return self.down_proj(gate_out * up_out)

    mlp.forward = types.MethodType(gate_alpha_forward, mlp)
    return model


def apply_down_eq_upT(model):
    """down = up^T: tie down_proj to transpose of up_proj."""
    mlp = model.block.mlp
    if hasattr(mlp, 'down_proj'):
        del mlp.down_proj

    # Get the existing forward (may already be patched by gate_alpha)
    old_forward = mlp.forward

    if hasattr(mlp, 'gate_alpha'):
        # gate_alpha is already applied — need combined forward
        _alpha = mlp.gate_alpha

        def down_upT_gate_alpha_forward(self, x, _a=_alpha):
            gate_out = self._act(self.up_proj(x) * _a)
            up_out = self.up_proj(x)
            hidden = gate_out * up_out
            return F.linear(hidden, self.up_proj.weight.t())

        mlp.forward = types.MethodType(down_upT_gate_alpha_forward, mlp)
    else:
        # Standard SwiGLU with down=up^T
        def down_upT_forward(self, x):
            gate_out = self._act(self.gate_proj(x))
            up_out = self.up_proj(x)
            hidden = gate_out * up_out
            return F.linear(hidden, self.up_proj.weight.t())

        mlp.forward = types.MethodType(down_upT_forward, mlp)

    return model


def apply_k_eq_q(model):
    """K = Q: use q_proj directly for K, no alpha scalar."""
    attn = model.block.attn
    if hasattr(attn, 'k_proj'):
        del attn.k_proj

    def k_eq_q_forward(self, x, mask=None):
        B, T, _ = x.shape
        shared = self.q_proj(x)
        q = shared.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = shared.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v_proj = self.q_proj if self.tie_kv else self.v_proj
        v = v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q = apply_rope(q, self.rope_cos, self.rope_sin)
        k = apply_rope(k, self.rope_cos, self.rope_sin)

        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = (q @ k.transpose(-2, -1)) * scale
        if mask is not None:
            scores = scores + mask[:T, :T]
        attn_weights = F.softmax(scores, dim=-1)

        out = (attn_weights @ v).transpose(1, 2).contiguous().view(B, T, -1)
        if self.tie_qo:
            return F.linear(out, self.q_proj.weight.t())
        return self.o_proj(out)

    attn.forward = types.MethodType(k_eq_q_forward, attn)
    return model


def apply_tie_qk_norm(model):
    """Tie q_norm = k_norm: share the same RMSNorm for Q and K."""
    attn = model.block.attn
    # Point k_norm to q_norm (same module, shared weights)
    attn.k_norm = attn.q_norm
    return model


def apply_gate_eq_up(model):
    """gate = up: gate_proj is identity to up_proj (no scalar)."""
    mlp = model.block.mlp
    if hasattr(mlp, 'gate_proj'):
        del mlp.gate_proj

    def gate_eq_up_forward(self, x):
        up_out = self.up_proj(x)
        gate_out = self._act(up_out)
        return self.down_proj(gate_out * up_out)

    mlp.forward = types.MethodType(gate_eq_up_forward, mlp)
    return model


def build_model(cfg, device):
    """Build model with architecture config and apply tying."""
    model = CircularArcQwen3(**cfg["model_args"]).to(device)
    tying = cfg.get("tying", {})

    # Apply attention tying
    if tying.get("k_alpha_q"):
        apply_k_alpha_q(model)
    elif tying.get("k_rot_q"):
        apply_k_rot_q(model)
    elif tying.get("k_eq_q"):
        apply_k_eq_q(model)

    # Apply V=Q tying (after K tying so forward is already patched)
    if tying.get("v_eq_q"):
        apply_v_eq_q(model)

    # Apply MLP tying
    if tying.get("gate_alpha"):
        apply_gate_alpha(model)
    elif tying.get("gate_eq_up"):
        apply_gate_eq_up(model)

    # Apply down=up^T (AFTER gate tying so combined forward works)
    if tying.get("down_eq_upT"):
        apply_down_eq_upT(model)

    # Apply QK norm tying (after attention forward is set)
    if tying.get("tie_qk_norm"):
        apply_tie_qk_norm(model)

    n_params = count_params(model)
    return model, n_params


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
        # Check if it has enough data
        try:
            with open(csv_path) as f:
                lines = f.readlines()
            if len(lines) > 2:  # header + at least 2 data rows
                return None  # skip
        except Exception:
            pass

    random.seed(seed)
    torch.manual_seed(seed)

    # Build model
    model, n_params = build_model(arch_cfg, device)

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
    cdp_scale = algo_cfg.get("cdp_scale", 0.1)  # fraction of ||W|| for overlay magnitude
    cdp_cycle = algo_cfg.get("cdp_cycle", 10000)
    cdp_target = algo_cfg.get("cdp_target", "proj")  # "proj", "attn", "mlp", "all"

    # SAM (Sharpness-Aware Minimization)
    use_sam = algo_cfg.get("sam", False)
    sam_rho = algo_cfg.get("sam_rho", 0.05)

    # Weight norm constraint (OmniGrok)
    use_norm_constraint = algo_cfg.get("norm_constraint", False)
    norm_radius = algo_cfg.get("norm_radius", 1.0)

    # Per-param LR group multipliers
    lr_gate_mult = algo_cfg.get("lr_gate_mult", 1.0)
    lr_norm_mult = algo_cfg.get("lr_norm_mult", 1.0)

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
    cdp_params = {}  # {name: param} for targeted params
    cdp_dirs = {}    # normalized random directions
    cdp_overlays = {}  # current overlay tensors
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
                include = cdp_target in pname  # custom filter string
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

    # Gradient/param logging for small models (≤50 params)
    # Logs every step in RAM, flushes to disk every eval_interval steps
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
        # Header: step, loss, then for each param: val, grad, update
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
        # Cycle: [0..half-1] = perturbation, [half..cycle-1] = recovery (no overlay)
        # Overlay is frozen (not trained) — optimizer naturally preserves it
        if use_cdp:
            cdp_half = cdp_cycle // 2
            cycle_pos = (step - 1) % cdp_cycle
            in_perturb_phase = cycle_pos < cdp_half

            if cycle_pos == 0:
                # Start of new cycle: resample random directions
                for pname in cdp_params:
                    d = torch.randn_like(cdp_params[pname].data)
                    d.div_(d.norm() + 1e-8)
                    cdp_dirs[pname] = d

            with torch.no_grad():
                for pname, p in cdp_params.items():
                    # Remove old overlay
                    p.sub_(cdp_overlays[pname])

                    if in_perturb_phase:
                        # Gamma tied to LR: decays 10x slower
                        lr_ratio = cur_lr / lr if lr > 0 else 0
                        cdp_gamma = (lr_ratio ** 0.1)
                        # Magnitude: gamma * scale * ||W|| * r_normalized
                        w_norm = p.data.norm()
                        magnitude = cdp_gamma * cdp_scale * w_norm
                        cdp_overlays[pname] = magnitude * cdp_dirs[pname]
                    else:
                        # Recovery phase: no overlay
                        cdp_overlays[pname].zero_()

                    # Apply new overlay (stays frozen through fwd/bwd/optim)
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
            # Save original gradients and params
            _sam_grads = {}
            _sam_orig = {}
            grad_norm = 0.0
            for pname, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    _sam_grads[pname] = p.grad.clone()
                    _sam_orig[pname] = p.data.clone()
                    grad_norm += p.grad.pow(2).sum().item()
            grad_norm = math.sqrt(grad_norm) + 1e-12

            # Ascend: perturb weights by epsilon = rho * grad / ||grad||
            with torch.no_grad():
                for pname, p in model.named_parameters():
                    if pname in _sam_grads:
                        epsilon = sam_rho * _sam_grads[pname] / grad_norm
                        p.add_(epsilon)

            # Recompute loss and gradients at perturbed point
            optimizer.zero_grad()
            logits2 = model(full_seq)
            shift_logits2 = logits2[:, :-1, :].reshape(-1, VOCAB_SIZE)
            loss_sam = F.cross_entropy(shift_logits2, shift_labels, ignore_index=-100)
            if l2_lambda > 0:
                l2_reg2 = sum(p.pow(2).sum() for p in model.parameters() if p.requires_grad)
                loss_sam = loss_sam + l2_lambda * l2_reg2
            loss_sam.backward()

            # Restore original weights (optimizer will update from restored position)
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
            # Flush to disk every 10K steps (small write, ~1MB)
            if step % 10000 == 0:
                grad_log_writer.writerows(grad_buffer)
                grad_log_file.flush()
                grad_buffer.clear()
        # CDP: overlay is frozen — optimizer naturally preserves it in p.data

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

        # Evaluate (CDP: overlay stays if active, removed only if in recovery phase)
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
                # Save best checkpoint (always, no accuracy threshold)
                ckpt_dir = os.path.join(results_dir, "checkpoints")
                os.makedirs(ckpt_dir, exist_ok=True)
                ckpt_path = os.path.join(ckpt_dir, f"{run_name}_s{seed}_best.pt")
                torch.save({"model_state_dict": model.state_dict(),
                            "accuracy": best_acc, "step": step,
                            "config": config_name, "algo": algo_name,
                            "seed": seed}, ckpt_path)

            # Periodic checkpoint every 10K steps (always)
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

            # Periodic checkpoint every 10K steps (always)
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
            # Set to best found so far
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

            # Periodic checkpoint every 10K steps (always)
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
    """Train with exact Newton's method (full Hessian, tractable for ≤50 params).

    Uses standard backward + per-grad-element backward for Hessian rows.
    Optional Tikhonov damping: (H + λI)⁻¹g.
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

        # Add damping: (H + λI)
        if damping > 0:
            hessian.add_(torch.eye(total_params, device=device) * damping)

        # Solve H * delta = -grad  →  delta = -H⁻¹g
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

    # Parse all CSVs
    from collections import defaultdict
    data = defaultdict(lambda: defaultdict(list))  # config__algo -> seed -> best_acc

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
    config_results = defaultdict(list)
    for run_name, seeds in sorted(data.items()):
        parts = run_name.split("__", 1)
        if len(parts) == 2:
            config, algo = parts
        else:
            config = run_name
            algo = "unknown"

        accs = sorted(seeds.values(), reverse=True)
        n_seeds = len(accs)
        n_alive = sum(1 for a in accs if a > 0.01)  # >1% = alive
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
        # Sort by best acc
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
    print(f"Sub-50p Sweep: {len(configs)} configs × {len(algos)} algos × {len(seeds)} seeds = {total_possible} total")
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
        # GPU: limit to avoid OOM
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
    import subprocess
    active = {}  # pid -> (process, cfg_name, algo_name, seed)
    task_iter = iter(tasks)
    done = 0

    def launch_next():
        try:
            cfg, algo, seed, max_steps, eval_interval, device, results_dir = next(task_iter)
        except StopIteration:
            return False
        cmd = [
            sys.executable, __file__,
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
