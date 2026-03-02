"""Validate tracked checkpoints: count parameters and run seeded evaluation.

Loads each best-per-param checkpoint, verifies parameter count matches
expectations, and evaluates on the fixed 10K test set to confirm reported
accuracy.

Usage:
    python scripts/validate_checkpoints.py
    python scripts/validate_checkpoints.py --test-set data/test_50k.json
    python scripts/validate_checkpoints.py --detailed
"""

import argparse
import json
import sys
from pathlib import Path

import torch

from minimal10digittransformer.model.qwen3 import Qwen3AdditionModel
from minimal10digittransformer.data.addition import load_test_set, generate_test_set
from minimal10digittransformer.evaluation.metrics import evaluate, evaluate_detailed

# Best-per-param checkpoints tracked in the repo
CHECKPOINTS = {
    "83p": {
        "path": "checkpoints/qwen3_d3_ff2_83p_tiekv_tieqo_shnorm_s905_targeted/best.pt",
        "expected_params": 83,
        "desc": "tieKV+tieQO+shnorm, ff=2, iterated targeted FT",
    },
    "86p": {
        "path": "checkpoints/qwen3_d3_ff2_86p_tiekv_tieqo_shbnorm_s1_targeted/best.pt",
        "expected_params": 86,
        "desc": "tieKV+tieQO+shbnorm, ff=2, targeted FT",
    },
    "89p": {
        "path": "checkpoints/qwen3_d3_ff2_89p_tiekv_tieqo_s11127/best.pt",
        "expected_params": 89,
        "desc": "tieKV+tieQO, ff=2, natural 4-stage FT (no test-set intervention)",
    },
    "101p": {
        "path": "checkpoints/qwen3_d3_ff2_101p_tieqo_s13_targeted/best.pt",
        "expected_params": 101,
        "desc": "tieQO, ff=2, targeted FT",
    },
    "122p": {
        "path": "checkpoints/qwen3_d3_ff3_122p_s6/best.pt",
        "expected_params": 122,
        "desc": "base, ff=3, 200K cosine",
    },
}


def load_model(ckpt_path, device):
    """Load model from checkpoint, return (model, config, checkpoint)."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
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
    return model, cfg, ckpt


def main():
    parser = argparse.ArgumentParser(description="Validate tracked checkpoints")
    parser.add_argument("--test-set", default="data/test_10k.json",
                        help="Test set for evaluation (default: 10K)")
    parser.add_argument("--detailed", action="store_true",
                        help="Run detailed per-position evaluation")
    parser.add_argument("--model", type=str, default=None,
                        help="Validate only this model (e.g., 89p)")
    args = parser.parse_args()

    device = torch.device("cpu")

    # Load test set
    test_path = Path(args.test_set)
    if test_path.exists():
        test_pairs = load_test_set(str(test_path))
        print(f"Test set: {test_path} ({len(test_pairs)} pairs)")
    else:
        print(f"ERROR: Test set not found: {test_path}")
        sys.exit(1)

    models_to_check = CHECKPOINTS
    if args.model:
        if args.model not in CHECKPOINTS:
            print(f"Unknown model: {args.model}. Available: {list(CHECKPOINTS.keys())}")
            sys.exit(1)
        models_to_check = {args.model: CHECKPOINTS[args.model]}

    print(f"\nValidating {len(models_to_check)} checkpoints...")
    print("=" * 70)

    all_pass = True
    results = {}

    for name, info in models_to_check.items():
        ckpt_path = Path(info["path"])
        print(f"\n{name}: {info['desc']}")
        print(f"  Path: {ckpt_path}")

        if not ckpt_path.exists():
            print(f"  FAIL: Checkpoint not found!")
            all_pass = False
            continue

        model, cfg, ckpt = load_model(str(ckpt_path), device)
        n_params = sum(p.numel() for p in model.parameters())

        # Check parameter count
        param_ok = n_params == info["expected_params"]
        param_status = "PASS" if param_ok else "FAIL"
        print(f"  Params: {n_params} (expected {info['expected_params']}) [{param_status}]")
        if not param_ok:
            all_pass = False

        # Print config
        flags = []
        if cfg.get("tie_kv"): flags.append("tieKV")
        if cfg.get("tie_qo"): flags.append("tieQO")
        if cfg.get("share_norms"): flags.append("shnorm")
        if cfg.get("share_block_norms"): flags.append("shbnorm")
        print(f"  Config: d={cfg['d_model']} ff={cfg['ff']} "
              f"h={cfg['n_heads']}/{cfg['n_kv_heads']} hd={cfg['head_dim']} "
              f"theta={cfg['rope_theta']} {'+'.join(flags) if flags else '(base)'}")
        print(f"  Step: {ckpt.get('step', '?')}")

        # Evaluate
        if args.detailed:
            res = evaluate_detailed(model, device, test_pairs)
            errors = res["n_errors"]
            acc = res["exact_acc"]
            print(f"  Accuracy: {acc:.6f} ({errors} errors / {res['n_samples']})")
            print(f"  Per-position (LSB→MSB): {' '.join(f'{p:.4f}' for p in res['per_position'])}")
        else:
            seq_acc, dig_acc = evaluate(model, device, test_pairs=test_pairs)
            errors = int(round((1 - seq_acc) * len(test_pairs)))
            acc = seq_acc
            print(f"  Accuracy: {seq_acc:.6f} ({errors} errors / {len(test_pairs)})")

        results[name] = {
            "params": n_params,
            "params_ok": param_ok,
            "accuracy": float(acc),
            "errors": errors,
            "step": ckpt.get("step", None),
        }

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Model':<8} {'Params':>6} {'Check':>6} {'Errors':>7} {'Accuracy':>10}")
    print("-" * 45)
    for name, res in results.items():
        check = "OK" if res["params_ok"] else "FAIL"
        print(f"{name:<8} {res['params']:>6} {check:>6} {res['errors']:>7} {res['accuracy']:>10.6f}")

    if all_pass:
        print(f"\nAll {len(results)} checkpoints validated successfully.")
    else:
        print(f"\nSome checks FAILED!")
        sys.exit(1)


if __name__ == "__main__":
    main()
