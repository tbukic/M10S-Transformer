"""Reproduce all claimed results from random initialization.

Usage:
    python experiments/reproduce.py --config 89p --seeds 0,1,2 --steps 50000
    python experiments/reproduce.py --config 122p --seeds 42 --steps 200  # smoke test
    python experiments/reproduce.py --list  # show all configs
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

# All reproduction configs
# Each maps to CLI args for qwen3_train.py
CONFIGS = {
    "122p": {
        "desc": "d=3 ff=3, 1h/1kv — baseline, no weight tying",
        "params": 122,
        "args": [
            "--d-model", "3", "--ff", "3",
            "--n-heads", "1", "--n-kv-heads", "1",
            "--head-dim", "4", "--rope-theta", "3.0",
            "--cosine-lr",
        ],
        "steps": 50000,
    },
    "113p": {
        "desc": "d=3 ff=2, 1h/1kv — reduced MLP",
        "params": 113,
        "args": [
            "--d-model", "3", "--ff", "2",
            "--n-heads", "1", "--n-kv-heads", "1",
            "--head-dim", "4", "--rope-theta", "3.0",
            "--cosine-lr",
        ],
        "steps": 50000,
    },
    "101p": {
        "desc": "d=3 ff=2, 1h/1kv, tieQO — output = Q^T",
        "params": 101,
        "args": [
            "--d-model", "3", "--ff", "2",
            "--n-heads", "1", "--n-kv-heads", "1",
            "--head-dim", "4", "--rope-theta", "3.0",
            "--tie-qo", "--cosine-lr",
        ],
        "steps": 50000,
    },
    "89p": {
        "desc": "d=3 ff=2, 1h/1kv, tieKV+tieQO — star result",
        "params": 89,
        "args": [
            "--d-model", "3", "--ff", "2",
            "--n-heads", "1", "--n-kv-heads", "1",
            "--head-dim", "4", "--rope-theta", "3.0",
            "--tie-kv", "--tie-qo", "--cosine-lr",
        ],
        "steps": 50000,
    },
    "86p": {
        "desc": "d=3 ff=2, 1h/1kv, tieKV+tieQO+share_block_norms",
        "params": 86,
        "args": [
            "--d-model", "3", "--ff", "2",
            "--n-heads", "1", "--n-kv-heads", "1",
            "--head-dim", "4", "--rope-theta", "3.0",
            "--tie-kv", "--tie-qo", "--share-block-norms", "--cosine-lr",
        ],
        "steps": 100000,
    },
    "83p": {
        "desc": "d=3 ff=2, 1h/1kv, tieKV+tieQO+share_norms",
        "params": 83,
        "args": [
            "--d-model", "3", "--ff", "2",
            "--n-heads", "1", "--n-kv-heads", "1",
            "--head-dim", "4", "--rope-theta", "3.0",
            "--tie-kv", "--tie-qo", "--share-norms", "--cosine-lr",
        ],
        "steps": 100000,
    },
}


def main():
    parser = argparse.ArgumentParser(description="Reproduce training results")
    parser.add_argument("--config", type=str, help="Config name (e.g., 89p, 122p)")
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4",
                        help="Comma-separated seeds")
    parser.add_argument("--steps", type=int, default=None,
                        help="Override training steps")
    parser.add_argument("--test-set", type=str, default="data/test_10k.json",
                        help="Fixed test set for evaluation")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--eval-interval", type=int, default=2000)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--list", action="store_true", help="List all configs")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running")
    args = parser.parse_args()

    if args.list:
        print("Available configs:")
        for name, cfg in CONFIGS.items():
            print(f"  {name:>5s}: {cfg['desc']} ({cfg['params']} params, {cfg['steps']} steps)")
        return

    if not args.config:
        parser.error("--config is required (use --list to see options)")

    if args.config not in CONFIGS:
        parser.error(f"Unknown config: {args.config}. Use --list to see options.")

    cfg = CONFIGS[args.config]
    seeds = [int(s) for s in args.seeds.split(",")]
    steps = args.steps or cfg["steps"]

    print(f"Reproducing: {args.config} — {cfg['desc']}")
    print(f"Expected params: {cfg['params']}")
    print(f"Seeds: {seeds}")
    print(f"Steps: {steps}")
    print(f"Test set: {args.test_set}")
    print()

    results = []
    for seed in seeds:
        cmd = [
            sys.executable, "experiments/qwen3_train.py",
            *cfg["args"],
            "--seed", str(seed),
            "--steps", str(steps),
            "--lr", str(args.lr),
            "--batch-size", str(args.batch_size),
            "--eval-interval", str(args.eval_interval),
            "--device", args.device,
        ]
        if args.test_set:
            cmd.extend(["--test-set", args.test_set])

        print(f"{'='*70}")
        print(f"Seed {seed}: {' '.join(cmd)}")
        print(f"{'='*70}")

        if args.dry_run:
            print("  (dry run — skipping)")
            continue

        result = subprocess.run(cmd, capture_output=False)
        results.append({
            "config": args.config,
            "seed": seed,
            "returncode": result.returncode,
        })

    if not args.dry_run and results:
        summary_path = f"experiments/reproduce_{args.config}_summary.json"
        with open(summary_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
