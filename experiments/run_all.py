"""Run all reproduction experiments with parallel execution.

Handles the full pipeline:
  Stage 1: Cosine LR training from random init
  Stage 2: Fine-tuning from best checkpoint (for sub-100p models that grokked)
  Final:   Evaluate all best checkpoints on fixed 10K test set

Usage:
    python experiments/run_all.py                    # run everything
    python experiments/run_all.py --stage 1          # stage 1 only
    python experiments/run_all.py --stage 2          # stage 2 only (after stage 1)
    python experiments/run_all.py --eval-only        # evaluate existing checkpoints
    python experiments/run_all.py --max-parallel 4   # limit parallelism
"""

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# ── Reproduction configs ────────────────────────────────────────────────────

STAGE1_RUNS = [
    # Priority 1: 89p — star result, 10 seeds, 50K cosine
    *[{"name": "89p", "seed": s, "steps": 50000, "args": [
        "--d-model", "3", "--ff", "2", "--n-heads", "1", "--n-kv-heads", "1",
        "--head-dim", "4", "--rope-theta", "3.0", "--tie-kv", "--tie-qo",
        "--cosine-lr", "--lr", "0.01", "--batch-size", "128",
    ]} for s in range(10)],

    # Priority 2: 122p — baseline, 5 seeds, 50K cosine
    *[{"name": "122p", "seed": s, "steps": 50000, "args": [
        "--d-model", "3", "--ff", "3", "--n-heads", "1", "--n-kv-heads", "1",
        "--head-dim", "4", "--rope-theta", "3.0",
        "--cosine-lr", "--lr", "0.01", "--batch-size", "128",
    ]} for s in range(5)],

    # Priority 3: 101p — tieQO, 5 seeds, 50K cosine
    *[{"name": "101p", "seed": s, "steps": 50000, "args": [
        "--d-model", "3", "--ff", "2", "--n-heads", "1", "--n-kv-heads", "1",
        "--head-dim", "4", "--rope-theta", "3.0", "--tie-qo",
        "--cosine-lr", "--lr", "0.01", "--batch-size", "128",
    ]} for s in range(5)],

    # Priority 4: 86p — tieKV+tieQO+shbnorm, 10 seeds, 100K cosine
    *[{"name": "86p", "seed": s, "steps": 100000, "args": [
        "--d-model", "3", "--ff", "2", "--n-heads", "1", "--n-kv-heads", "1",
        "--head-dim", "4", "--rope-theta", "3.0", "--tie-kv", "--tie-qo",
        "--share-block-norms", "--cosine-lr", "--lr", "0.01", "--batch-size", "128",
    ]} for s in range(10)],

    # Priority 5: 83p — tieKV+tieQO+shnorm, 15 seeds, 100K cosine
    *[{"name": "83p", "seed": s, "steps": 100000, "args": [
        "--d-model", "3", "--ff", "2", "--n-heads", "1", "--n-kv-heads", "1",
        "--head-dim", "4", "--rope-theta", "3.0", "--tie-kv", "--tie-qo",
        "--share-norms", "--cosine-lr", "--lr", "0.01", "--batch-size", "128",
    ]} for s in range(15)],
]

# Fine-tuning configs: which stage 1 results to fine-tune
FT_CONFIGS = {
    "89p": {"lr": 0.001, "batch_size": 256, "steps": 30000, "min_grok_acc": 0.10},
    "86p": {"lr": 0.001, "batch_size": 256, "steps": 30000, "min_grok_acc": 0.10},
    "83p": {"lr": 0.001, "batch_size": 256, "steps": 30000, "min_grok_acc": 0.10},
}


def run_training(run_spec: dict) -> dict:
    """Run a single training job. Returns result dict."""
    env = os.environ.copy()
    import multiprocessing
    n_cores = multiprocessing.cpu_count()
    max_par = run_spec.get("_max_parallel", 8)
    env["OMP_NUM_THREADS"] = str(max(1, n_cores // max(max_par, 1)))

    cmd = [
        sys.executable, "experiments/qwen3_train.py",
        *run_spec["args"],
        "--seed", str(run_spec["seed"]),
        "--steps", str(run_spec["steps"]),
        "--eval-interval", "2000",
        "--test-set", "data/test_10k.json",
    ]

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    elapsed = time.time() - t0

    # Parse final accuracy from output
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
        "name": run_spec["name"],
        "seed": run_spec["seed"],
        "steps": run_spec["steps"],
        "best_acc": best_acc,
        "final_acc": final_acc,
        "elapsed": elapsed,
        "returncode": result.returncode,
        "stage": run_spec.get("stage", 1),
    }


def find_checkpoint(name: str, seed: int, stage: int = 1) -> str | None:
    """Find the best.pt checkpoint for a given config/seed."""
    # Scan checkpoints/ for matching directory
    ckpt_base = Path("checkpoints")
    if not ckpt_base.exists():
        return None

    # Build expected tag patterns
    tag_patterns = {
        "89p": "qwen3_d3_ff2_89p_tiekv_tieqo",
        "86p": "qwen3_d3_ff2_86p_tiekv_tieqo_shbnorm",
        "83p": "qwen3_d3_ff2_83p_tiekv_tieqo_shnorm",
        "122p": "qwen3_d3_ff3_122p",
        "101p": "qwen3_d3_ff2_101p_tieqo",
    }

    pattern = tag_patterns.get(name, "")
    for d in sorted(ckpt_base.iterdir()):
        if d.is_dir() and pattern in d.name and f"_s{seed}" in d.name:
            best = d / "best.pt"
            if best.exists():
                return str(best)
    return None


def run_finetune(name: str, seed: int, ft_config: dict, ft_seed: int,
                 max_parallel: int = 8) -> dict:
    """Run fine-tuning from a stage 1 checkpoint."""
    ckpt_path = find_checkpoint(name, seed)
    if not ckpt_path:
        return {"name": name, "seed": seed, "ft_seed": ft_seed,
                "error": "no checkpoint found", "stage": 2}

    env = os.environ.copy()
    import multiprocessing
    n_cores = multiprocessing.cpu_count()
    env["OMP_NUM_THREADS"] = str(max(1, n_cores // max(max_parallel, 1)))

    cmd = [
        sys.executable, "experiments/qwen3_train.py",
        "--resume", ckpt_path,
        "--lr", str(ft_config["lr"]),
        "--batch-size", str(ft_config["batch_size"]),
        "--steps", str(ft_config["steps"]),
        "--seed", str(ft_seed),
        "--eval-interval", "2000",
        "--test-set", "data/test_10k.json",
    ]

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
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
        "name": name,
        "seed": seed,
        "ft_seed": ft_seed,
        "best_acc": best_acc,
        "final_acc": final_acc,
        "elapsed": elapsed,
        "returncode": result.returncode,
        "stage": 2,
        "checkpoint": ckpt_path,
    }


def run_eval(ckpt_path: str, test_set: str, max_parallel: int = 8) -> dict:
    """Evaluate a checkpoint on a test set."""
    env = os.environ.copy()
    import multiprocessing
    n_cores = multiprocessing.cpu_count()
    env["OMP_NUM_THREADS"] = str(max(1, n_cores // max(max_parallel, 1)))

    cmd = [
        sys.executable, "experiments/qwen3_eval.py",
        ckpt_path,
        "--test-set", test_set,
        "--detailed",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    return {"checkpoint": ckpt_path, "test_set": test_set,
            "output": result.stdout, "returncode": result.returncode}


def stage1(max_parallel: int):
    """Run all stage 1 training."""
    print(f"{'='*70}")
    print(f"STAGE 1: Training from random init ({len(STAGE1_RUNS)} runs, {max_parallel} parallel)")
    print(f"{'='*70}\n")

    results = []
    completed = 0
    total = len(STAGE1_RUNS)

    # Pass max_parallel to each run for thread calculation
    runs_with_par = [{**run, "_max_parallel": max_parallel} for run in STAGE1_RUNS]

    with ProcessPoolExecutor(max_workers=max_parallel) as pool:
        futures = {pool.submit(run_training, run): run for run in runs_with_par}
        for future in as_completed(futures):
            run = futures[future]
            try:
                r = future.result()
                results.append(r)
                completed += 1
                status = f"best={r['best_acc']:.4f}" if r['returncode'] == 0 else "FAILED"
                print(f"  [{completed}/{total}] {r['name']} seed={r['seed']}: "
                      f"{status} ({r['elapsed']:.0f}s)")
            except Exception as e:
                completed += 1
                print(f"  [{completed}/{total}] {run['name']} seed={run['seed']}: ERROR {e}")
                results.append({"name": run["name"], "seed": run["seed"],
                                "error": str(e), "stage": 1})

    # Save results
    with open("experiments/reproduction_stage1_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    print(f"\n{'='*70}")
    print("STAGE 1 SUMMARY")
    print(f"{'='*70}")
    by_config = {}
    for r in results:
        by_config.setdefault(r["name"], []).append(r)
    for name in ["89p", "122p", "101p", "86p", "83p"]:
        runs = by_config.get(name, [])
        if not runs:
            continue
        accs = [r.get("best_acc", 0) for r in runs if r.get("returncode") == 0]
        grokked = sum(1 for a in accs if a > 0.5)
        best = max(accs) if accs else 0
        mean = sum(accs) / len(accs) if accs else 0
        print(f"  {name}: {grokked}/{len(runs)} grokked, best={best:.4f}, mean={mean:.4f}")

    return results


def stage2(stage1_results: list, max_parallel: int):
    """Run fine-tuning on grokked seeds."""
    print(f"\n{'='*70}")
    print("STAGE 2: Fine-tuning grokked seeds")
    print(f"{'='*70}\n")

    ft_jobs = []
    for r in stage1_results:
        name = r.get("name", "")
        if name not in FT_CONFIGS:
            continue
        ft_cfg = FT_CONFIGS[name]
        if r.get("best_acc", 0) >= ft_cfg["min_grok_acc"] and r.get("returncode") == 0:
            ft_jobs.append({
                "name": name,
                "seed": r["seed"],
                "ft_config": ft_cfg,
                "ft_seed": r["seed"] + 100,  # different seed for FT randomness
            })

    if not ft_jobs:
        print("  No seeds to fine-tune (none grokked above threshold)")
        return []

    print(f"  {len(ft_jobs)} fine-tuning jobs")

    results = []
    completed = 0
    total = len(ft_jobs)

    with ProcessPoolExecutor(max_workers=max_parallel) as pool:
        futures = {pool.submit(run_finetune, job["name"], job["seed"],
                               job["ft_config"], job["ft_seed"],
                               max_parallel): job
                   for job in ft_jobs}
        for future in as_completed(futures):
            job = futures[future]
            try:
                r = future.result()
                results.append(r)
                completed += 1
                status = f"best={r['best_acc']:.4f}" if r.get('returncode') == 0 else "FAILED"
                print(f"  [{completed}/{total}] {r['name']} seed={r['seed']}: "
                      f"{status} ({r.get('elapsed', 0):.0f}s)")
            except Exception as e:
                completed += 1
                print(f"  [{completed}/{total}] {job['name']} seed={job['seed']}: ERROR {e}")

    # Save results
    with open("experiments/reproduction_stage2_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def final_eval(max_parallel: int):
    """Evaluate all best checkpoints on fixed test sets."""
    print(f"\n{'='*70}")
    print("FINAL EVALUATION on fixed test sets")
    print(f"{'='*70}\n")

    ckpt_base = Path("checkpoints")
    if not ckpt_base.exists():
        print("  No checkpoints found")
        return

    # Find all best.pt files
    best_pts = sorted(ckpt_base.glob("*/best.pt"))
    print(f"  Found {len(best_pts)} checkpoints")

    # Evaluate on 10K test set
    results = []
    with ProcessPoolExecutor(max_workers=max_parallel) as pool:
        futures = {pool.submit(run_eval, str(pt), "data/test_10k.json",
                               max_parallel): pt
                   for pt in best_pts}
        for future in as_completed(futures):
            r = future.result()
            results.append(r)

    with open("experiments/reproduction_eval_results.json", "w") as f:
        json.dump([{"checkpoint": r["checkpoint"], "output": r["output"]}
                   for r in results], f, indent=2)

    for r in sorted(results, key=lambda x: x["checkpoint"]):
        # Extract accuracy from output
        for line in r["output"].split("\n"):
            if "Exact match:" in line:
                print(f"  {Path(r['checkpoint']).parent.name}: {line.strip()}")
                break


def main():
    parser = argparse.ArgumentParser(description="Run all reproduction experiments")
    parser.add_argument("--stage", type=int, default=0,
                        help="Run specific stage (1 or 2). 0 = all stages")
    parser.add_argument("--eval-only", action="store_true",
                        help="Only run final evaluation")
    parser.add_argument("--max-parallel", type=int, default=8,
                        help="Max parallel training jobs")
    args = parser.parse_args()

    t0 = time.time()

    if args.eval_only:
        final_eval(args.max_parallel)
        return

    if args.stage == 0 or args.stage == 1:
        s1_results = stage1(args.max_parallel)
    else:
        # Load previous stage 1 results
        with open("experiments/reproduction_stage1_results.json") as f:
            s1_results = json.load(f)

    if args.stage == 0 or args.stage == 2:
        stage2(s1_results, args.max_parallel)

    if args.stage == 0:
        final_eval(args.max_parallel)

    total_elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"TOTAL TIME: {total_elapsed/60:.1f} minutes")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
