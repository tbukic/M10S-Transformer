"""Generate training plots for the report.

Plots:
1. Loss curves: training loss vs step for different configs
2. Grokking curves: exact match accuracy vs step for different configs/seeds
3. Accuracy vs params: final accuracy as a function of parameter count
4. Carry analysis: accuracy by number of carries for top models
5. EMA/SWA comparison: base vs averaged model accuracy during fine-tuning

Usage:
  python experiments/plot_training.py                    # all plots from CSV logs
  python experiments/plot_training.py --from-checkpoints # rebuild from checkpoint files
  python experiments/plot_training.py --swa-ema          # SWA/EMA comparison only
"""

import argparse
import csv
import json
import os
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("matplotlib not installed. Install with: pip install matplotlib")


def load_metrics_csv(path):
    """Load a metrics.csv file into lists.

    Supports both old format (step column) and new reproduce.py format
    (global_step column with phase tracking).
    """
    losses, accs, lrs = [], [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Support both old ("step") and new ("global_step") formats
            if "global_step" in row:
                step = int(row["global_step"])
            elif "step" in row:
                step = int(row["step"])
            else:
                continue
            loss = float(row["loss"]) if row.get("loss") and row["loss"] else None
            acc = float(row["exact_acc"]) if row.get("exact_acc") and row["exact_acc"] else None
            lr = float(row["lr"]) if row.get("lr") and row["lr"] else None
            if loss is not None:
                losses.append((step, loss))
            if acc is not None:
                accs.append((step, acc))
            if lr is not None:
                lrs.append((step, lr))
    return losses, accs, lrs


def find_all_metrics():
    """Find all metrics.csv files in checkpoints/."""
    results = {}
    ckpt_base = Path("checkpoints")
    for d in sorted(ckpt_base.iterdir()):
        if d.is_dir():
            csv_path = d / "metrics.csv"
            if csv_path.exists():
                tag = d.name
                results[tag] = str(csv_path)
    return results


def collect_checkpoint_data():
    """Collect accuracy data from all checkpoints for accuracy-vs-params plot."""
    import torch
    data = []
    ckpt_base = Path("checkpoints")
    for d in sorted(ckpt_base.iterdir()):
        if d.is_dir() and d.name.startswith("qwen3_"):
            bp = d / "best.pt"
            if bp.exists():
                try:
                    ckpt = torch.load(str(bp), map_location="cpu", weights_only=True)
                    data.append({
                        "tag": d.name,
                        "n_params": ckpt.get("n_params", 0),
                        "accuracy": ckpt.get("accuracy", 0),
                        "step": ckpt.get("step", 0),
                    })
                except Exception:
                    pass
    return data


def plot_grokking_curves(metrics_dict, output_dir="plots"):
    """Plot accuracy vs step for different configs showing grokking dynamics."""
    if not HAS_MPL:
        return

    os.makedirs(output_dir, exist_ok=True)

    # Group by config (param count)
    configs = {}
    for tag, path in metrics_dict.items():
        # Extract param count from tag
        for part in tag.split("_"):
            if part.endswith("p") and part[:-1].isdigit():
                n_params = int(part[:-1])
                configs.setdefault(n_params, []).append((tag, path))
                break

    for n_params in sorted(configs.keys()):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        fig.suptitle(f"{n_params}p Grokking Dynamics", fontsize=14)

        for tag, path in configs[n_params]:
            losses, accs, _lrs = load_metrics_csv(path)
            seed = tag.split("_s")[-1]

            if losses:
                loss_steps, loss_vals = zip(*losses)
                ax1.plot(loss_steps, loss_vals, alpha=0.5, label=f"s{seed}", linewidth=0.8)

            if accs:
                acc_steps, acc_vals = zip(*accs)
                ax2.plot(acc_steps, acc_vals, alpha=0.7, label=f"s{seed}", linewidth=1.5,
                         marker="o", markersize=3)

        ax1.set_ylabel("Training Loss")
        ax1.set_yscale("log")
        ax1.legend(fontsize=8, ncol=3)
        ax1.grid(True, alpha=0.3)

        ax2.set_ylabel("Exact Match Accuracy")
        ax2.set_xlabel("Training Step")
        ax2.set_ylim(-0.05, 1.05)
        ax2.axhline(y=0.99, color="red", linestyle="--", alpha=0.3, label="99%")
        ax2.legend(fontsize=8, ncol=3)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/grokking_{n_params}p.png", dpi=150)
        plt.close()
        print(f"  Saved {output_dir}/grokking_{n_params}p.png")


def plot_accuracy_vs_params(output_dir="plots"):
    """Plot accuracy vs parameter count."""
    if not HAS_MPL:
        return

    os.makedirs(output_dir, exist_ok=True)
    data = collect_checkpoint_data()

    if not data:
        print("  No checkpoint data found")
        return

    # Group by param count, take best accuracy for each
    by_params = {}
    for d in data:
        n = d["n_params"]
        by_params.setdefault(n, []).append(d["accuracy"])

    params_list = sorted(by_params.keys())
    best_accs = [max(by_params[n]) for n in params_list]
    mean_accs = [sum(by_params[n]) / len(by_params[n]) for n in params_list]
    n_seeds = [len(by_params[n]) for n in params_list]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(params_list, best_accs, s=50, c="blue", alpha=0.7, label="Best seed")
    ax.scatter(params_list, mean_accs, s=30, c="orange", alpha=0.5, label="Mean across seeds")

    ax.set_xlabel("Parameter Count")
    ax.set_ylabel("Exact Match Accuracy (200-sample eval)")
    ax.set_title("Accuracy vs Parameter Count")
    ax.axhline(y=0.99, color="red", linestyle="--", alpha=0.3, label="99%")
    ax.axhline(y=1.0, color="green", linestyle="--", alpha=0.3, label="100%")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    # Annotate key configs
    for n, best in zip(params_list, best_accs):
        if best > 0.9 and n <= 140:
            ax.annotate(f"{n}p", (n, best), textcoords="offset points",
                        xytext=(5, 5), fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/accuracy_vs_params.png", dpi=150)
    plt.close()
    print(f"  Saved {output_dir}/accuracy_vs_params.png")


def plot_carry_analysis(output_dir="plots"):
    """Plot carry analysis from detailed eval results."""
    if not HAS_MPL:
        return

    os.makedirs(output_dir, exist_ok=True)

    # Look for detailed eval JSON files
    eval_files = list(Path("checkpoints").glob("**/best_detailed_eval.json"))
    if not eval_files:
        print("  No detailed eval files found. Run: python experiments/qwen3_eval.py --detailed")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    for eval_file in eval_files:
        tag = eval_file.parent.name
        with open(eval_file) as f:
            data = json.load(f)

        carries = sorted(data["carry_acc"].keys(), key=int)
        accs = [data["carry_acc"][c][0] for c in carries]
        counts = [data["carry_acc"][c][1] for c in carries]

        n_params = tag.split("_")
        for part in n_params:
            if part.endswith("p") and part[:-1].isdigit():
                label = part
                break
        else:
            label = tag[:20]

        ax.plot([int(c) for c in carries], accs, marker="o", label=label)

    ax.set_xlabel("Number of Carries")
    ax.set_ylabel("Exact Match Accuracy")
    ax.set_title("Accuracy by Carry Count")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.95, 1.005)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/carry_analysis.png", dpi=150)
    plt.close()
    print(f"  Saved {output_dir}/carry_analysis.png")


def plot_swa_ema(output_dir="plots"):
    """Plot SWA/EMA comparison from experiment results."""
    if not HAS_MPL:
        return

    os.makedirs(output_dir, exist_ok=True)

    results_file = Path("experiments/swa_ema_results.json")
    if not results_file.exists():
        # Try individual result files
        results = []
        for f in Path("experiments").glob("swa_ema_*_results.json"):
            with open(f) as fh:
                data = json.load(fh)
                if isinstance(data, list):
                    results.extend(data)
                else:
                    results.append(data)
    else:
        with open(results_file) as f:
            results = json.load(f)

    if not results:
        print("  No SWA/EMA results found")
        return

    # Plot training curves for each result
    for r in results:
        if "log" not in r or not r["log"]:
            continue

        config = r.get("config", f"{r['n_params']}p")
        mode = r["mode"]
        seed = r["seed"]

        fig, ax = plt.subplots(figsize=(8, 5))

        steps = [e["step"] for e in r["log"]]
        base_accs = [e["base_acc"] for e in r["log"]]

        ax.plot(steps, base_accs, label="Base", alpha=0.7)

        if mode == "ema":
            ema_accs = [e["ema_acc"] for e in r["log"]]
            ax.plot(steps, ema_accs, label=f"EMA (decay={r.get('ema_decay', '?')})",
                    linewidth=2)
        elif mode == "swa":
            swa_accs = [e["swa_acc"] for e in r["log"]]
            ax.plot(steps, swa_accs, label=f"SWA (n={r['log'][-1].get('swa_count', '?')})",
                    linewidth=2)

        ax.set_xlabel("Fine-tuning Step")
        ax.set_ylabel("Exact Match Accuracy (200 samples)")
        ax.set_title(f"{config} {mode.upper()} — seed {seed}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.9, 1.005)

        plt.tight_layout()
        fname = f"{output_dir}/{mode}_{config}_s{seed}.png"
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"  Saved {fname}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="plots")
    parser.add_argument("--from-checkpoints", action="store_true",
                        help="Rebuild data from checkpoint files")
    parser.add_argument("--swa-ema", action="store_true",
                        help="Only plot SWA/EMA comparisons")
    args = parser.parse_args()

    if not HAS_MPL:
        print("ERROR: matplotlib is required. Install with: pip install matplotlib")
        return

    if args.swa_ema:
        plot_swa_ema(args.output_dir)
        return

    print("Generating plots...")

    # 1. Grokking curves (from CSV logs)
    metrics = find_all_metrics()
    if metrics:
        print(f"\n1. Grokking curves ({len(metrics)} runs with CSV logs)")
        plot_grokking_curves(metrics, args.output_dir)
    else:
        print("\n1. No metrics.csv files found (need to retrain with logging)")

    # 2. Accuracy vs params
    print("\n2. Accuracy vs parameter count")
    plot_accuracy_vs_params(args.output_dir)

    # 3. Carry analysis
    print("\n3. Carry analysis")
    plot_carry_analysis(args.output_dir)

    # 4. SWA/EMA comparison
    print("\n4. SWA/EMA comparison")
    plot_swa_ema(args.output_dir)

    print(f"\nDone! Plots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
