"""Extended reproduction runs with much longer training.

Strategy:
  1. Resume from-scratch with 200K cosine for seeds that didn't grok
  2. Resume from grokked checkpoints with 300K FT steps
  3. Evaluate all on fixed 10K test set

Usage:
    python experiments/run_extended.py                # run all
    python experiments/run_extended.py --stage fresh   # only from-scratch 200K
    python experiments/run_extended.py --stage ft      # only fine-tuning 300K
    python experiments/run_extended.py --eval-only     # only final eval
"""

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# ── From-scratch 200K cosine runs ───────────────────────────────────────────
# These are the configs + seeds that need more time to grok

FRESH_RUNS = [
    # 89p: 10 seeds, 200K cosine (4x longer than before)
    *[{"name": "89p", "seed": s, "steps": 200000, "args": [
        "--d-model", "3", "--ff", "2", "--n-heads", "1", "--n-kv-heads", "1",
        "--head-dim", "4", "--rope-theta", "3.0", "--tie-kv", "--tie-qo",
        "--cosine-lr", "--lr", "0.01", "--batch-size", "128",
    ]} for s in range(10)],

    # 122p: seeds that didn't grok at 50K, try 200K
    *[{"name": "122p", "seed": s, "steps": 200000, "args": [
        "--d-model", "3", "--ff", "3", "--n-heads", "1", "--n-kv-heads", "1",
        "--head-dim", "4", "--rope-theta", "3.0",
        "--cosine-lr", "--lr", "0.01", "--batch-size", "128",
    ]} for s in range(10)],

    # 101p: 10 seeds, 200K cosine (none grokked at 50K)
    *[{"name": "101p", "seed": s, "steps": 200000, "args": [
        "--d-model", "3", "--ff", "2", "--n-heads", "1", "--n-kv-heads", "1",
        "--head-dim", "4", "--rope-theta", "3.0", "--tie-qo",
        "--cosine-lr", "--lr", "0.01", "--batch-size", "128",
    ]} for s in range(10)],
]

# ── Fine-tuning from grokked checkpoints ────────────────────────────────────

# Config name → (tag_pattern, FT args)
FT_BASE = {
    "89p": "qwen3_d3_ff2_89p_tiekv_tieqo",
    "86p": "qwen3_d3_ff2_86p_tiekv_tieqo_shbnorm",
    "83p": "qwen3_d3_ff2_83p_tiekv_tieqo_shnorm",
    "122p": "qwen3_d3_ff3_122p",
    "101p": "qwen3_d3_ff2_101p_tieqo",
}


def find_best_checkpoints(config_name: str, min_acc: float = 0.0) -> list[tuple[str, int, float]]:
    """Find all best.pt checkpoints for a config with accuracy above threshold.
    Returns list of (path, seed, accuracy)."""
    pattern = FT_BASE.get(config_name, "")
    results = []
    ckpt_base = Path("checkpoints")
    if not ckpt_base.exists():
        return results

    import torch
    for d in sorted(ckpt_base.iterdir()):
        if d.is_dir() and pattern in d.name:
            best = d / "best.pt"
            if best.exists():
                try:
                    ckpt = torch.load(str(best), map_location="cpu", weights_only=True)
                    acc = ckpt.get("accuracy", 0)
                    # Extract seed from directory name
                    seed = int(d.name.split("_s")[-1])
                    if acc >= min_acc:
                        results.append((str(best), seed, acc))
                except Exception:
                    pass
    return results


def run_training(cmd_args: list[str], env: dict) -> dict:
    """Run a training command and parse results."""
    t0 = time.time()
    result = subprocess.run(cmd_args, capture_output=True, text=True, env=env)
    elapsed = time.time() - t0

    best_acc = 0.0
    final_acc = 0.0
    for line in result.stdout.split("\n"):
        if "NEW BEST:" in line:
            try:
                best_acc = float(line.split("NEW BEST:")[1].split("**")[0].strip())
            except (ValueError, IndexError):
                pass
        if "FINAL:" in line:
            try:
                final_acc = float(line.split("exact=")[1].split()[0])
            except (ValueError, IndexError):
                pass
        if "Best exact:" in line:
            try:
                best_acc = max(best_acc, float(line.split(":")[1].strip()))
            except (ValueError, IndexError):
                pass

    return {
        "best_acc": best_acc,
        "final_acc": final_acc,
        "elapsed": elapsed,
        "returncode": result.returncode,
    }


def _get_omp_threads(max_parallel: int) -> str:
    """Calculate OMP_NUM_THREADS based on available cores and parallelism."""
    import multiprocessing
    n_cores = multiprocessing.cpu_count()
    threads = max(1, n_cores // max(max_parallel, 1))
    return str(threads)


# Module-level default (updated by main)
_MAX_PARALLEL = 8


def run_one_fresh(run_spec: dict) -> dict:
    """Run a single from-scratch training."""
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = _get_omp_threads(_MAX_PARALLEL)
    cmd = [
        sys.executable, "experiments/qwen3_train.py",
        *run_spec["args"],
        "--seed", str(run_spec["seed"]),
        "--steps", str(run_spec["steps"]),
        "--eval-interval", "5000",
        "--test-set", "data/test_10k.json",
    ]
    r = run_training(cmd, env)
    r["name"] = run_spec["name"]
    r["seed"] = run_spec["seed"]
    r["steps"] = run_spec["steps"]
    r["stage"] = "fresh"
    return r


def run_one_ft(ckpt_path: str, name: str, seed: int, ft_seed: int) -> dict:
    """Run fine-tuning from a checkpoint."""
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = _get_omp_threads(_MAX_PARALLEL)
    cmd = [
        sys.executable, "experiments/qwen3_train.py",
        "--resume", ckpt_path,
        "--lr", "0.001",
        "--batch-size", "256",
        "--steps", "300000",
        "--seed", str(ft_seed),
        "--eval-interval", "5000",
        "--test-set", "data/test_10k.json",
    ]
    r = run_training(cmd, env)
    r["name"] = name
    r["seed"] = seed
    r["ft_seed"] = ft_seed
    r["stage"] = "ft"
    r["checkpoint"] = ckpt_path
    return r


def run_fresh_stage(max_parallel: int):
    """Run all from-scratch 200K cosine runs."""
    print(f"{'='*70}")
    print(f"FROM-SCRATCH 200K COSINE ({len(FRESH_RUNS)} runs, {max_parallel} parallel)")
    print(f"{'='*70}\n")
    sys.stdout.flush()

    results = []
    completed = 0
    total = len(FRESH_RUNS)

    with ProcessPoolExecutor(max_workers=max_parallel) as pool:
        futures = {pool.submit(run_one_fresh, run): run for run in FRESH_RUNS}
        for future in as_completed(futures):
            run = futures[future]
            try:
                r = future.result()
                results.append(r)
                completed += 1
                status = f"best={r['best_acc']:.4f}" if r['returncode'] == 0 else "FAILED"
                print(f"  [{completed}/{total}] {r['name']} seed={r['seed']}: "
                      f"{status} ({r['elapsed']:.0f}s)")
                sys.stdout.flush()
            except Exception as e:
                completed += 1
                print(f"  [{completed}/{total}] {run['name']} seed={run['seed']}: ERROR {e}")
                sys.stdout.flush()

    with open("experiments/extended_fresh_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    print(f"\n{'='*70}")
    print("FROM-SCRATCH SUMMARY")
    print(f"{'='*70}")
    by_config = {}
    for r in results:
        by_config.setdefault(r["name"], []).append(r)
    for name in ["89p", "122p", "101p"]:
        runs = by_config.get(name, [])
        if not runs:
            continue
        accs = [r.get("best_acc", 0) for r in runs if r.get("returncode") == 0]
        grokked = sum(1 for a in accs if a > 0.90)
        best = max(accs) if accs else 0
        mean = sum(accs) / len(accs) if accs else 0
        print(f"  {name}: {grokked}/{len(runs)} grokked (>90%), best={best:.4f}, mean={mean:.4f}")
    sys.stdout.flush()

    return results


def run_ft_stage(max_parallel: int):
    """Fine-tune all grokked checkpoints with 300K steps."""
    print(f"\n{'='*70}")
    print("FINE-TUNING 300K STEPS (from grokked checkpoints)")
    print(f"{'='*70}\n")
    sys.stdout.flush()

    # Find all checkpoints with >10% accuracy
    ft_jobs = []
    for config_name in ["89p", "86p", "83p", "122p", "101p"]:
        ckpts = find_best_checkpoints(config_name, min_acc=0.10)
        for path, seed, acc in ckpts:
            ft_jobs.append({
                "ckpt_path": path, "name": config_name,
                "seed": seed, "acc": acc, "ft_seed": seed + 1000,
            })

    print(f"  Found {len(ft_jobs)} checkpoints to fine-tune:")
    for j in ft_jobs:
        print(f"    {j['name']} s{j['seed']}: {j['acc']:.4f}")
    print()
    sys.stdout.flush()

    if not ft_jobs:
        return []

    results = []
    completed = 0
    total = len(ft_jobs)

    with ProcessPoolExecutor(max_workers=max_parallel) as pool:
        futures = {
            pool.submit(run_one_ft, j["ckpt_path"], j["name"], j["seed"], j["ft_seed"]): j
            for j in ft_jobs
        }
        for future in as_completed(futures):
            job = futures[future]
            try:
                r = future.result()
                results.append(r)
                completed += 1
                status = f"best={r['best_acc']:.4f}" if r.get("returncode") == 0 else "FAILED"
                print(f"  [{completed}/{total}] {r['name']} s{r['seed']}: "
                      f"{status} ({r.get('elapsed', 0):.0f}s)")
                sys.stdout.flush()
            except Exception as e:
                completed += 1
                print(f"  [{completed}/{total}] {job['name']} s{job['seed']}: ERROR {e}")
                sys.stdout.flush()

    with open("experiments/extended_ft_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def run_eval(max_parallel: int):
    """Evaluate all new best checkpoints on fixed 10K test set."""
    print(f"\n{'='*70}")
    print("FINAL EVALUATION — new reproduction checkpoints on fixed 10K")
    print(f"{'='*70}\n")
    sys.stdout.flush()

    # Only evaluate checkpoints from this round (recent final.pt)
    configs_to_eval = ["89p", "122p", "101p", "86p", "83p"]
    eval_jobs = []
    for config_name in configs_to_eval:
        ckpts = find_best_checkpoints(config_name, min_acc=0.01)
        for path, seed, acc in ckpts:
            eval_jobs.append({"path": path, "name": config_name, "seed": seed, "train_acc": acc})

    print(f"  Evaluating {len(eval_jobs)} checkpoints")
    sys.stdout.flush()

    def eval_one(job):
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = _get_omp_threads(max_parallel)
        cmd = [
            sys.executable, "experiments/qwen3_eval.py",
            job["path"], "--test-set", "data/test_10k.json",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        # Parse accuracy
        for line in result.stdout.split("\n"):
            if "Exact match:" in line:
                try:
                    acc_str = line.split("Exact match:")[1].strip().split()[0]
                    errors_str = line.split("(")[1].split()[0]
                    return {**job, "eval_acc": float(acc_str), "errors": int(errors_str)}
                except (ValueError, IndexError):
                    pass
        return {**job, "eval_acc": 0.0, "errors": -1}

    results = []
    with ProcessPoolExecutor(max_workers=max_parallel) as pool:
        futures = {pool.submit(eval_one, j): j for j in eval_jobs}
        for future in as_completed(futures):
            r = future.result()
            results.append(r)

    # Sort by config and accuracy
    results.sort(key=lambda x: (-x.get("eval_acc", 0), x["name"]))

    print(f"\n{'='*70}")
    print("EVALUATION RESULTS (fixed 10K test set)")
    print(f"{'='*70}")
    for r in results:
        if r.get("eval_acc", 0) > 0.01:
            print(f"  {r['name']:>5s} s{r['seed']:>4d}: {r['eval_acc']:.4f} "
                  f"({r.get('errors', '?')} errors)")
    sys.stdout.flush()

    with open("experiments/extended_eval_results.json", "w") as f:
        json.dump(results, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["fresh", "ft", "all"], default="all")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--max-parallel", type=int, default=8)
    args = parser.parse_args()

    global _MAX_PARALLEL
    _MAX_PARALLEL = args.max_parallel

    t0 = time.time()

    if args.eval_only:
        run_eval(args.max_parallel)
        print(f"\nTotal time: {(time.time() - t0)/60:.1f} minutes")
        return

    if args.stage in ("all", "fresh"):
        run_fresh_stage(args.max_parallel)

    if args.stage in ("all", "ft"):
        run_ft_stage(args.max_parallel)

    run_eval(args.max_parallel)

    print(f"\n{'='*70}")
    print(f"TOTAL TIME: {(time.time() - t0)/60:.1f} minutes")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
