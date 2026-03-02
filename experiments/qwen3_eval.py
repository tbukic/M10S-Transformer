"""Evaluate a saved Qwen3 checkpoint on 10K samples."""

import argparse
import json
import time

import torch

from minimal10digittransformer.model.qwen3 import Qwen3AdditionModel
from minimal10digittransformer.data.addition import load_test_set, generate_test_set
from minimal10digittransformer.evaluation.metrics import evaluate, evaluate_detailed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="Path to checkpoint .pt file")
    parser.add_argument("--n-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--test-set", type=str, default=None, help="Path to fixed test set JSON")
    parser.add_argument("--detailed", action="store_true", help="Run detailed evaluation")
    args = parser.parse_args()

    device = torch.device("cpu")

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    config = ckpt["config"]

    model = Qwen3AdditionModel(
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_kv_heads=config["n_kv_heads"],
        head_dim=config["head_dim"],
        ff=config["ff"],
        rope_theta=config["rope_theta"],
        qk_norm=not config.get("no_qk_norm", False),
        use_swiglu=not config.get("gelu", False),
        tie_kv=config.get("tie_kv", False),
        tie_qo=config.get("tie_qo", False),
        tie_gate=config.get("tie_gate", False),
        repeats=config.get("repeats", 1),
        share_norms=config.get("share_norms", False),
        share_block_norms=config.get("share_block_norms", False),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded: {args.checkpoint}")
    print(f"Params: {n_params}, step: {ckpt.get('step', '?')}, "
          f"train_acc: {ckpt.get('accuracy', '?')}")

    # Load or generate test pairs
    if args.test_set:
        test_pairs = load_test_set(args.test_set)
        print(f"Using fixed test set: {args.test_set} ({len(test_pairs)} pairs)")
    else:
        test_pairs = generate_test_set(args.n_samples, seed=args.seed)
        print(f"Generated {len(test_pairs)} random pairs (seed={args.seed})")

    t0 = time.time()

    if args.detailed:
        results = evaluate_detailed(model, device, test_pairs)
        elapsed = time.time() - t0
        print(f"\nDetailed results on {results['n_samples']} samples:")
        print(f"  Exact match: {results['exact_acc']:.4f} ({results['n_errors']} errors)")
        print(f"  Digit accuracy: {results['digit_acc']:.4f}")
        print(f"\n  Per-position accuracy (LSB->MSB):")
        for i, acc in enumerate(results['per_position']):
            print(f"    Position {i}: {acc:.4f}")
        print(f"\n  Carry analysis:")
        for n_carries, (acc, count) in results['carry_acc'].items():
            print(f"    {n_carries} carries: {acc:.4f} ({count} samples)")
        print(f"  Time: {elapsed:.1f}s")

        # Save detailed results
        out_path = args.checkpoint.replace(".pt", "_detailed_eval.json")
        save_results = {
            "exact_acc": results["exact_acc"],
            "digit_acc": results["digit_acc"],
            "n_samples": results["n_samples"],
            "n_errors": results["n_errors"],
            "per_position": results["per_position"],
            "carry_acc": {str(k): list(v) for k, v in results["carry_acc"].items()},
        }
        with open(out_path, "w") as f:
            json.dump(save_results, f, indent=2)
        print(f"\n  Saved to {out_path}")
    else:
        seq_acc, dig_acc = evaluate(model, device, test_pairs=test_pairs)
        elapsed = time.time() - t0
        errors = int(round((1 - seq_acc) * len(test_pairs)))
        print(f"\nResults on {len(test_pairs)} samples:")
        print(f"  Exact match: {seq_acc:.4f} ({errors} errors)")
        print(f"  Digit accuracy: {dig_acc:.4f}")
        print(f"  Time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
