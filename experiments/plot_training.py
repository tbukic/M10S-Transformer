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


def plot_all_models_combined(output_dir="plots"):
    """Plot loss and accuracy for all submitted models on a single figure.

    Two panels sharing the x-axis:
      Top:    Training loss (log scale) vs global step
      Bottom: Exact-match accuracy (%) vs global step

    Colors are consistent across both panels.
    Uses the best (longest) metrics file per model.
    """
    if not HAS_MPL:
        return

    os.makedirs(output_dir, exist_ok=True)

    ckpt_base = Path("checkpoints")

    # (label, csv_path) — pick the best metrics file per model.
    # Prefer runs that actually grokked (high accuracy) and have many data points.
    MODEL_CANDIDATES = {
        "62p":  ["proof_62p_s42", "reproduce_62p_s42_v2"],
        "83p":  ["proof_83p_s5", "reproduce_83p_s5"],
        "86p":  ["proof_86p_s1", "reproduce_86p_s1"],
        "89p":  ["proof_89p_s5", "reproduce_89p_s5"],
        "95p":  ["proof_95p_s0", "reproduce_95p_s9999"],
        "96p":  ["proof_96p_s9999", "reproduce_96p_s9999"],
        "101p": ["proof_101p_s13", "reproduce_101p_s13"],
        "113p": ["reproduce_113p_s1"],
        "122p": ["proof_122p_s6", "reproduce_122p_s6_proof"],
    }

    models = []
    for label, candidates in MODEL_CANDIDATES.items():
        best_path, best_score = None, (-1, 0)  # (max_acc, n_lines)
        for c in candidates:
            p = ckpt_base / c / "metrics.csv"
            if p.exists():
                losses, accs, _ = load_metrics_csv(str(p))
                n_lines = len(losses)
                max_acc = max((a for _, a in accs), default=0) if accs else 0
                score = (max_acc, n_lines)
                if score > best_score:
                    best_score = score
                    best_path = str(p)
        if best_path and best_score[1] > 1:
            models.append((label, best_path))

    if not models:
        print("  No metrics data found")
        return

    # Sort by param count
    models.sort(key=lambda x: int(x[0].replace("p", "")))

    # Consistent color palette (colorblind-friendly, ordered small → large)
    COLORS = {
        "62p":  "#e41a1c",  # red
        "83p":  "#ff7f00",  # orange
        "86p":  "#b8860b",  # dark goldenrod
        "89p":  "#4daf4a",  # green
        "95p":  "#377eb8",  # blue
        "96p":  "#984ea3",  # purple
        "101p": "#8b4513",  # saddle brown
        "113p": "#e91e9c",  # hot pink
        "122p": "#555555",  # dark gray
    }

    fig, (ax_loss, ax_acc) = plt.subplots(
        2, 1, figsize=(14, 9), sharex=True,
        gridspec_kw={"hspace": 0.08})

    for label, path in models:
        losses, accs, _lrs = load_metrics_csv(path)
        if not losses:
            continue

        color = COLORS.get(label, "#000000")

        # Smooth loss with exponential moving average for cleaner plot
        loss_steps, loss_vals = zip(*losses)
        loss_steps = list(loss_steps)
        loss_vals = list(loss_vals)

        # EMA smoothing
        smoothed = []
        alpha = 0.02  # small alpha = more smoothing
        s = loss_vals[0]
        for v in loss_vals:
            s = alpha * v + (1 - alpha) * s
            smoothed.append(s)

        # Subsample for rendering
        if len(loss_steps) > 2000:
            stride = max(1, len(loss_steps) // 1000)
            loss_steps = loss_steps[::stride]
            smoothed = smoothed[::stride]

        ax_loss.plot(loss_steps, smoothed, color=color, label=label,
                     linewidth=1.5, alpha=0.9)

        if accs:
            acc_steps, acc_vals = zip(*accs)
            ax_acc.plot(acc_steps, [a * 100 for a in acc_vals], color=color,
                        label=label, linewidth=1.8, marker="o", markersize=3,
                        alpha=0.9)

        best_acc = max(acc_vals) if accs else 0
        print(f"  {label}: {len(losses)} loss pts, {len(accs)} acc pts, "
              f"max_step={losses[-1][0]}, final_loss={losses[-1][1]:.4f}, "
              f"best_acc={best_acc:.2%}")

    # Loss panel
    ax_loss.set_ylabel("Training Loss", fontsize=12)
    ax_loss.set_yscale("log")
    ax_loss.set_ylim(bottom=0.005, top=3.0)
    ax_loss.legend(loc="upper right", fontsize=9, ncol=3, framealpha=0.9)
    ax_loss.grid(True, alpha=0.3, which="both")
    ax_loss.set_title("Training Curves — All Submitted Models",
                      fontsize=14, fontweight="bold")

    # Accuracy panel
    ax_acc.set_ylabel("Exact-Match Accuracy (%)", fontsize=12)
    ax_acc.set_xlabel("Training Step", fontsize=12)
    ax_acc.set_ylim(-2, 105)
    ax_acc.axhline(y=99, color="black", linestyle="--", linewidth=0.8,
                   alpha=0.4, label="99% threshold")
    ax_acc.legend(loc="lower right", fontsize=9, ncol=3, framealpha=0.9)
    ax_acc.grid(True, alpha=0.3)
    ax_acc.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{x/1000:.0f}K" if x >= 1000 else f"{x:.0f}"))

    # Annotate eval source
    ax_acc.text(0.01, 0.02,
                "Accuracy: 500-sample progress eval (seed=12345, separate from training & verify sets)",
                transform=ax_acc.transAxes, fontsize=7, alpha=0.5, style="italic")

    fig.subplots_adjust(left=0.07, right=0.97, top=0.94, bottom=0.07)
    out_path = f"{output_dir}/all_models_training.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")


def load_metrics_by_phase(path):
    """Load metrics CSV grouped by phase, deduplicating rows at same step.

    When duplicate rows exist at the same (phase, global_step), keeps
    the one with accuracy data (if any). This avoids double-plotting.

    Returns dict: phase_num -> {steps, losses, acc_steps, accs, phase_steps}
    """
    # First pass: deduplicate — prefer rows with accuracy
    raw = {}  # (phase, global_step) -> best row
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (int(row["phase"]), int(row["global_step"]))
            has_acc = bool(row.get("exact_acc", "").strip())
            if key not in raw or has_acc:
                raw[key] = row

    # Second pass: build phase data from deduplicated rows
    phases = {}
    for (p, gs) in sorted(raw.keys()):
        row = raw[(p, gs)]
        ps = int(row["phase_step"])
        loss = float(row["loss"])
        if p not in phases:
            phases[p] = {"steps": [], "losses": [], "acc_steps": [],
                         "accs": [], "phase_steps": []}
        phases[p]["steps"].append(gs)
        phases[p]["losses"].append(loss)
        phases[p]["phase_steps"].append(ps)
        if row.get("exact_acc", "").strip():
            phases[p]["acc_steps"].append(gs)
            phases[p]["accs"].append(float(row["exact_acc"]))
    return phases


def plot_all_models_phased(output_dir="plots"):
    """Single combined plot with functional stage bands on x-axis.

    Instead of per-model phase numbers, the x-axis is divided into universal
    functional stages:
      1. Cosine LR (base grokking)    — all models
      2. Constant LR (refinement)     — 62p, 83p, 113p
      3. L-BFGS (2nd-order opt)       — 86p
      4. Cosine no-WD (Adam FT)       — 62p
      5. Targeted FT (error correct)  — all except 122p

    Each model's per-phase data is mapped into the matching functional band.
    Models that skip a stage have empty space in that band.
    """
    if not HAS_MPL:
        return

    import yaml

    os.makedirs(output_dir, exist_ok=True)
    ckpt_base = Path("checkpoints")

    with open("experiments/configs.yaml") as f:
        all_cfg = yaml.safe_load(f)

    MODEL_CANDIDATES = {
        "62p":  ["proof_62p_s42", "reproduce_62p_s42_v2"],
        "83p":  ["proof_83p_s5", "reproduce_83p_s5"],
        "86p":  ["proof_86p_s1", "reproduce_86p_s1"],
        "89p":  ["proof_89p_s5", "reproduce_89p_s5"],
        "95p":  ["proof_95p_s0", "reproduce_95p_s9999"],
        "96p":  ["proof_96p_s9999", "reproduce_96p_s9999"],
        "101p": ["proof_101p_s13", "reproduce_101p_s13"],
        "113p": ["reproduce_113p_s1"],
        "122p": ["proof_122p_s6", "reproduce_122p_s6_proof"],
    }
    MODEL_ORDER = ["62p", "83p", "86p", "89p", "95p", "96p", "101p", "113p", "122p"]
    COLORS = {
        "62p":  "#e41a1c",  # red
        "83p":  "#ff7f00",  # orange
        "86p":  "#b8860b",  # dark goldenrod (distinct from orange)
        "89p":  "#4daf4a",  # green
        "95p":  "#377eb8",  # blue
        "96p":  "#984ea3",  # purple
        "101p": "#8b4513",  # saddle brown (distinct from orange/gold)
        "113p": "#e91e9c",  # hot pink (more vivid)
        "122p": "#555555",  # dark gray
    }

    # Universal functional stages (order matters)
    STAGES = [
        ("cosine",     "Cosine LR",     "#d4e6f1"),  # light blue
        ("constant",   "Constant LR",   "#d5f5e3"),  # light green
        ("lbfgs",      "L-BFGS",        "#f9e79f"),  # light yellow
        ("cosine_nowd","Cosine (no WD)","#fdebd0"),  # light orange
        ("targeted",   "Targeted FT",   "#fadbd8"),  # light pink
    ]

    # Load all model data (only CSVs with phase column)
    model_phases = {}
    for label in MODEL_ORDER:
        best_path, best_score = None, (-1, 0)
        for c in MODEL_CANDIDATES[label]:
            p = ckpt_base / c / "metrics.csv"
            if p.exists():
                # Check for phase column before considering
                with open(p) as f:
                    header = f.readline()
                if "phase" not in header:
                    continue
                losses, accs, _ = load_metrics_csv(str(p))
                max_acc = max((a for _, a in accs), default=0) if accs else 0
                if (max_acc, len(losses)) > best_score:
                    best_score = (max_acc, len(losses))
                    best_path = str(p)
        if best_path:
            model_phases[label] = load_metrics_by_phase(best_path)

    if not model_phases:
        print("  No data found")
        return

    # Map each model's actual phases to functional stages.
    # model_stage_data[label][stage_key] = list of phase_data dicts
    # (a model can have multiple phases of the same type, e.g. two "constant" phases)
    model_stage_data = {}
    for label in MODEL_ORDER:
        if label not in model_phases or label not in all_cfg["models"]:
            continue
        cfg_phases = all_cfg["models"][label]["phases"]
        pd = model_phases[label]
        stage_map = {}  # stage_key -> [phase_data, ...]
        for p_num, pcfg in enumerate(cfg_phases):
            ptype = pcfg["type"]
            if p_num in pd:
                stage_map.setdefault(ptype, []).append(pd[p_num])
        model_stage_data[label] = stage_map

    # Compute max width per functional stage (sum of phase_steps for models
    # that have multiple phases of the same type in that stage)
    stage_widths = {}
    for stage_key, _, _ in STAGES:
        max_w = 0
        for label, sdata in model_stage_data.items():
            if stage_key in sdata:
                # Sum phase_steps across all phases of this type
                total = 0
                for pdata in sdata[stage_key]:
                    if pdata["phase_steps"]:
                        total += max(pdata["phase_steps"])
                max_w = max(max_w, total)
        stage_widths[stage_key] = max_w

    # Remove stages that no model uses
    active_stages = [(k, name, bg) for k, name, bg in STAGES
                     if stage_widths.get(k, 0) > 0]

    # Compute x-offsets
    GAP = 3000
    stage_offsets = {}
    x = 0
    for stage_key, _, _ in active_stages:
        stage_offsets[stage_key] = x
        x += stage_widths[stage_key] + GAP

    # Build the figure
    fig, (ax_loss, ax_acc) = plt.subplots(
        2, 1, figsize=(16, 9), sharex=True,
        gridspec_kw={"hspace": 0.06})

    # Draw stage backgrounds, separators, and labels
    for idx, (stage_key, stage_name, bg_color) in enumerate(active_stages):
        x_start = stage_offsets[stage_key]
        x_end = x_start + stage_widths[stage_key]

        for ax in (ax_loss, ax_acc):
            ax.axvspan(x_start, x_end, alpha=0.3, color=bg_color, zorder=0)

        if idx > 0:
            for ax in (ax_loss, ax_acc):
                ax.axvline(x=x_start, color="#333333", linewidth=1.5,
                           linestyle="-", alpha=0.6, zorder=5)

        # Which models participate in this stage?
        participants = [l for l in MODEL_ORDER
                        if l in model_stage_data and stage_key in model_stage_data[l]]

        # Stage name above plot
        ax_loss.text((x_start + x_end) / 2, 1.10,
                     stage_name,
                     transform=ax_loss.get_xaxis_transform(),
                     ha="center", va="bottom", fontsize=10,
                     fontweight="bold", alpha=0.6)

        # Participant list below stage name
        if len(participants) == len(model_stage_data):
            desc = "All models"
        else:
            desc = ", ".join(participants)
        ax_loss.text((x_start + x_end) / 2, 1.08,
                     desc,
                     transform=ax_loss.get_xaxis_transform(),
                     ha="center", va="top", fontsize=7,
                     alpha=0.45, style="italic")


    # Plot each model — separate line segment per stage (no cross-stage connections)
    # Draw largest models first (background), smallest last (foreground) so
    # lower-param models are always visible on top.
    draw_order = list(reversed(MODEL_ORDER))
    for z_idx, label in enumerate(draw_order):
        if label not in model_stage_data:
            continue
        sdata = model_stage_data[label]
        color = COLORS.get(label, "#333333")
        first_segment = True  # for legend: only label the first segment
        z = 10 + z_idx  # higher z for smaller models
        ema_state = None  # carry EMA across stages for visual continuity

        for stage_key, _, _ in active_stages:
            if stage_key not in sdata:
                continue
            base_offset = stage_offsets[stage_key]

            # Collect data for this model in this stage
            seg_x_loss, seg_loss = [], []
            seg_x_acc, seg_acc = [], []

            inner_offset = 0
            for pdata in sdata[stage_key]:
                for ps, loss in zip(pdata["phase_steps"], pdata["losses"]):
                    seg_x_loss.append(base_offset + inner_offset + ps)
                    seg_loss.append(loss)

                with_acc_idx = 0
                for i, ps in enumerate(pdata["phase_steps"]):
                    gs = pdata["steps"][i]
                    if (with_acc_idx < len(pdata["acc_steps"]) and
                            gs == pdata["acc_steps"][with_acc_idx]):
                        seg_x_acc.append(base_offset + inner_offset + ps)
                        seg_acc.append(pdata["accs"][with_acc_idx])
                        with_acc_idx += 1

                if pdata["phase_steps"]:
                    inner_offset += max(pdata["phase_steps"])

            # Smooth loss — carry EMA state from previous stage for continuity
            if seg_loss:
                smoothed = []
                alpha_ema = 0.03
                s = ema_state if ema_state is not None else seg_loss[0]
                for v in seg_loss:
                    s = alpha_ema * v + (1 - alpha_ema) * s
                    smoothed.append(s)
                ema_state = s  # carry to next stage

                if len(seg_x_loss) > 2000:
                    stride = max(1, len(seg_x_loss) // 1000)
                    px = seg_x_loss[::stride]
                    py = smoothed[::stride]
                else:
                    px, py = seg_x_loss, smoothed

                ax_loss.plot(px, py, color=color,
                             label=label if first_segment else None,
                             linewidth=1.6, alpha=0.9, zorder=z)

            if seg_acc:
                # Subsample accuracy if too dense
                if len(seg_x_acc) > 30:
                    stride = max(1, len(seg_x_acc) // 30)
                    sa_x = seg_x_acc[::stride]
                    sa_y = [a * 100 for a in seg_acc[::stride]]
                else:
                    sa_x = seg_x_acc
                    sa_y = [a * 100 for a in seg_acc]
                ax_acc.plot(sa_x, sa_y, color=color,
                            label=label if first_segment else None,
                            linewidth=1.5, marker="o", markersize=2.5,
                            alpha=0.85, zorder=z)

            if seg_loss or seg_acc:
                first_segment = False

    # Format loss panel
    ax_loss.set_ylabel("Training Loss", fontsize=12)
    ax_loss.set_yscale("log")
    ax_loss.set_ylim(0.0003, 3.0)
    # Reverse legend order so smallest params appear first (matching visual top-to-bottom)
    handles, labels = ax_loss.get_legend_handles_labels()
    ax_loss.legend(handles[::-1], labels[::-1],
                   loc="lower left", fontsize=8, ncol=3, framealpha=0.9)
    ax_loss.grid(True, alpha=0.2, which="both")
    fig.suptitle("Training Pipeline — All Submitted Models",
                 fontsize=14, fontweight="bold", y=0.98)

    # Format accuracy panel
    ax_acc.set_ylabel("Exact-Match Accuracy (%)", fontsize=12)
    ax_acc.set_xlabel("Step (within stage)", fontsize=12)
    ax_acc.set_ylim(-2, 105)
    ax_acc.axhline(y=99, color="black", linestyle="--", linewidth=0.8,
                   alpha=0.3, label="99% threshold")
    handles, labels = ax_acc.get_legend_handles_labels()
    ax_acc.legend(handles[::-1], labels[::-1],
                  loc="lower right", fontsize=8, ncol=3, framealpha=0.9)
    ax_acc.grid(True, alpha=0.2)

    ax_acc.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{x/1000:.0f}K" if x >= 1000 else f"{x:.0f}"))

    fig.subplots_adjust(left=0.06, right=0.97, top=0.82, bottom=0.07)
    out_path = f"{output_dir}/all_models_phased.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")


def plot_all_models_by_phase(output_dir="plots"):
    """Plot a 3x3 grid: one subplot per model, with phase-colored backgrounds.

    Each subplot has loss (left y-axis) and accuracy (right y-axis).
    Background bands show training phases (cosine, constant, targeted, etc.).
    """
    if not HAS_MPL:
        return

    import yaml

    os.makedirs(output_dir, exist_ok=True)
    ckpt_base = Path("checkpoints")

    # Load phase types from configs.yaml
    cfg_path = Path("experiments/configs.yaml")
    with open(cfg_path) as f:
        all_cfg = yaml.safe_load(f)

    # Same model selection as combined plot
    MODEL_CANDIDATES = {
        "62p":  ["proof_62p_s42", "reproduce_62p_s42_v2"],
        "83p":  ["proof_83p_s5", "reproduce_83p_s5"],
        "86p":  ["proof_86p_s1", "reproduce_86p_s1"],
        "89p":  ["proof_89p_s5", "reproduce_89p_s5"],
        "95p":  ["proof_95p_s0", "reproduce_95p_s9999"],
        "96p":  ["proof_96p_s9999", "reproduce_96p_s9999"],
        "101p": ["proof_101p_s13", "reproduce_101p_s13"],
        "113p": ["reproduce_113p_s1"],
        "122p": ["proof_122p_s6", "reproduce_122p_s6_proof"],
    }

    # Model order: smallest to largest
    MODEL_ORDER = ["62p", "83p", "86p", "89p", "95p", "96p", "101p", "113p", "122p"]

    # Phase type colors (background bands)
    PHASE_COLORS = {
        "cosine":     "#d4e6f1",  # light blue
        "constant":   "#d5f5e3",  # light green
        "cosine_nowd": "#fdebd0", # light orange
        "lbfgs":      "#f9e79f",  # light yellow
        "targeted":   "#fadbd8",  # light red/pink
    }

    # Model accent color for curves
    MODEL_COLORS = {
        "62p":  "#e41a1c",
        "83p":  "#ff7f00",
        "86p":  "#c4a000",
        "89p":  "#4daf4a",
        "95p":  "#377eb8",
        "96p":  "#984ea3",
        "101p": "#a65628",
        "113p": "#f781bf",
        "122p": "#555555",
    }

    models = []
    for label in MODEL_ORDER:
        candidates = MODEL_CANDIDATES[label]
        best_path, best_score = None, (-1, 0)
        for c in candidates:
            p = ckpt_base / c / "metrics.csv"
            if p.exists():
                losses, accs, _ = load_metrics_csv(str(p))
                n_lines = len(losses)
                max_acc = max((a for _, a in accs), default=0) if accs else 0
                if (max_acc, n_lines) > best_score:
                    best_score = (max_acc, n_lines)
                    best_path = str(p)
        if best_path:
            models.append((label, best_path))

    n = len(models)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 4.5 * nrows))
    axes = axes.flatten()

    for idx, (label, path) in enumerate(models):
        ax_loss = axes[idx]
        ax_acc = ax_loss.twinx()

        phase_data = load_metrics_by_phase(path)

        # Get phase types from config
        cfg_key = label.replace("p", "")  # "89p" -> "89"
        phase_types = []
        if label in all_cfg["models"]:
            phase_types = [p["type"] for p in all_cfg["models"][label]["phases"]]

        color = MODEL_COLORS.get(label, "#333333")

        # Draw phase background bands
        for p_num in sorted(phase_data.keys()):
            pd = phase_data[p_num]
            if not pd["steps"]:
                continue
            x_min, x_max = pd["steps"][0], pd["steps"][-1]
            ptype = phase_types[p_num] if p_num < len(phase_types) else "unknown"
            bg_color = PHASE_COLORS.get(ptype, "#eeeeee")
            ax_loss.axvspan(x_min, x_max, alpha=0.45, color=bg_color, zorder=0)

            # Phase label at top
            x_mid = (x_min + x_max) / 2
            short_label = {"cosine": "cos", "constant": "const",
                           "cosine_nowd": "cos-nwd", "lbfgs": "L-BFGS",
                           "targeted": "targeted"}.get(ptype, ptype)
            ax_loss.text(x_mid, 0.97, f"P{p_num}: {short_label}",
                         transform=ax_loss.get_xaxis_transform(),
                         ha="center", va="top", fontsize=7, alpha=0.7,
                         fontweight="bold")

        # Plot loss (all phases concatenated)
        all_steps, all_losses = [], []
        for p_num in sorted(phase_data.keys()):
            pd = phase_data[p_num]
            all_steps.extend(pd["steps"])
            all_losses.extend(pd["losses"])

        # EMA smooth
        if all_losses:
            smoothed = []
            alpha_ema = 0.03
            s = all_losses[0]
            for v in all_losses:
                s = alpha_ema * v + (1 - alpha_ema) * s
                smoothed.append(s)

            # Subsample
            if len(all_steps) > 1500:
                stride = max(1, len(all_steps) // 800)
                plot_steps = all_steps[::stride]
                plot_losses = smoothed[::stride]
            else:
                plot_steps, plot_losses = all_steps, smoothed

            ax_loss.plot(plot_steps, plot_losses, color=color,
                         linewidth=1.3, alpha=0.9, label="Loss")

        # Plot accuracy
        all_acc_steps, all_accs = [], []
        for p_num in sorted(phase_data.keys()):
            pd = phase_data[p_num]
            all_acc_steps.extend(pd["acc_steps"])
            all_accs.extend(pd["accs"])

        if all_accs:
            ax_acc.plot(all_acc_steps, [a * 100 for a in all_accs],
                        color=color, linewidth=1.8, marker="o", markersize=3,
                        alpha=0.7, linestyle="--", label="Accuracy")
            ax_acc.axhline(y=99, color="black", linestyle=":", linewidth=0.6,
                           alpha=0.3)

        # Formatting
        ax_loss.set_yscale("log")
        ax_loss.set_ylim(0.0003, 3.0)
        ax_acc.set_ylim(-5, 108)

        best_acc = max(all_accs) if all_accs else 0
        n_phases = len(phase_data)
        ax_loss.set_title(f"{label}  ({n_phases} phase{'s' if n_phases > 1 else ''}, "
                          f"best={best_acc:.0%})",
                          fontsize=11, fontweight="bold", color=color)

        ax_loss.xaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, _: f"{x/1000:.0f}K" if x >= 1000 else f"{x:.0f}"))

        if idx % ncols == 0:
            ax_loss.set_ylabel("Loss", fontsize=9)
        ax_acc.set_ylabel("Accuracy (%)", fontsize=9, color=color, alpha=0.7)

        if idx >= (nrows - 1) * ncols:
            ax_loss.set_xlabel("Training Step", fontsize=9)

        ax_loss.tick_params(labelsize=8)
        ax_acc.tick_params(labelsize=8)
        ax_loss.grid(True, alpha=0.2, which="both")

    # Hide unused subplots
    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    # Legend for phase colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=PHASE_COLORS["cosine"], alpha=0.6, label="Cosine LR"),
        Patch(facecolor=PHASE_COLORS["constant"], alpha=0.6, label="Constant LR"),
        Patch(facecolor=PHASE_COLORS["cosine_nowd"], alpha=0.6, label="Cosine (no WD)"),
        Patch(facecolor=PHASE_COLORS["lbfgs"], alpha=0.6, label="L-BFGS"),
        Patch(facecolor=PHASE_COLORS["targeted"], alpha=0.6, label="Targeted FT"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=5,
               fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, -0.01))

    fig.suptitle("Training Phases — All Submitted Models",
                 fontsize=15, fontweight="bold", y=1.01)
    fig.subplots_adjust(hspace=0.35, wspace=0.3)

    out_path = f"{output_dir}/all_models_by_phase.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="plots")
    parser.add_argument("--from-checkpoints", action="store_true",
                        help="Rebuild data from checkpoint files")
    parser.add_argument("--swa-ema", action="store_true",
                        help="Only plot SWA/EMA comparisons")
    parser.add_argument("--combined", action="store_true",
                        help="Only plot combined all-models figure")
    parser.add_argument("--by-phase", action="store_true",
                        help="Only plot per-model phase breakdown")
    args = parser.parse_args()

    if not HAS_MPL:
        print("ERROR: matplotlib is required. Install with: pip install matplotlib")
        return

    if args.swa_ema:
        plot_swa_ema(args.output_dir)
        return

    if args.combined:
        print("Generating combined all-models plot...")
        plot_all_models_combined(args.output_dir)
        print("Done!")
        return

    if args.by_phase:
        print("Generating phased combined plot...")
        plot_all_models_phased(args.output_dir)
        print("Done!")
        return

    print("Generating plots...")

    # 0. Combined all-models plot
    print("\n0. Combined all-models training curves")
    plot_all_models_combined(args.output_dir)

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
