"""Ablation study for minimal addition transformers.

For each named parameter in the model, measures the accuracy impact of:
  1. Zeroing it out (set all values to 0)
  2. Randomizing it (replace with random values from same distribution)
  3. Per-element ablation (zero out each weight individually) — optional, for tiny models

Also supports component-level ablation (zero out entire attention, entire MLP, etc.)
and per-weight sensitivity analysis.

Usage:
  uv run experiments/ablation.py                                     # default: 89p
  uv run experiments/ablation.py --checkpoint checkpoints/.../best.pt
  uv run experiments/ablation.py --all-models                        # all submitted models
  uv run experiments/ablation.py --per-weight                        # element-level sensitivity
  uv run experiments/ablation.py --n-random 5                        # 5 random seeds per param
  uv run experiments/ablation.py --n-eval 500                        # faster eval (fewer samples)
"""

import argparse
import copy
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from minimal10digittransformer.evaluation.metrics import evaluate
from experiments.interpretability import load_model

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ── Core ablation functions ────────────────────────────────────────────────

def eval_accuracy(model, device, n_eval=500, seed=12345):
    """Quick accuracy evaluation."""
    exact_acc, digit_acc = evaluate(model, device, n_samples=n_eval, seed=seed)
    return exact_acc, digit_acc


def ablate_zero(model, param_name):
    """Return a copy of the model with the named parameter zeroed out."""
    ablated = copy.deepcopy(model)
    param = dict(ablated.named_parameters())[param_name]
    with torch.no_grad():
        param.zero_()
    return ablated


def ablate_random(model, param_name, seed=0):
    """Return a copy with the named parameter replaced by random values.

    Random values are drawn from N(0, std) where std matches the original
    parameter's standard deviation, preserving the scale.
    """
    ablated = copy.deepcopy(model)
    param = dict(ablated.named_parameters())[param_name]
    orig_std = param.data.std().item()
    orig_mean = param.data.mean().item()
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        param.data = torch.randn_like(param.data, generator=gen) * max(orig_std, 1e-6) + orig_mean
    return ablated


def ablate_element_zero(model, param_name, flat_idx):
    """Return a copy with a single element of the named parameter zeroed."""
    ablated = copy.deepcopy(model)
    param = dict(ablated.named_parameters())[param_name]
    with torch.no_grad():
        param.data.view(-1)[flat_idx] = 0.0
    return ablated


def ablate_component(model, component_prefix):
    """Return a copy with all parameters matching the prefix zeroed."""
    ablated = copy.deepcopy(model)
    for name, param in ablated.named_parameters():
        if name.startswith(component_prefix):
            with torch.no_grad():
                param.zero_()
    return ablated


# ── Component groups ───────────────────────────────────────────────────────

def get_component_groups(model):
    """Return dict of component_name -> list of param_name prefixes.

    Works for any Qwen3 or CircularArc model by inspecting the actual
    parameter names.
    """
    all_names = [n for n, _ in model.named_parameters()]
    groups = {}

    # Detect components by prefix patterns
    prefixes = {
        "embedding": ["embed.", "arc_"],
        "attention": ["block.attn."],
        "attention.qk": ["block.attn.q_proj.", "block.attn.k_proj.",
                         "block.attn.q_norm.", "block.attn.k_norm."],
        "attention.vo": ["block.attn.v_proj.", "block.attn.o_proj."],
        "mlp": ["block.mlp."],
        "mlp.gate": ["block.mlp.gate_proj."],
        "mlp.up": ["block.mlp.up_proj."],
        "mlp.down": ["block.mlp.down_proj."],
        "norms": ["block.ln1.", "block.ln2.", "final_norm."],
        "block.ln1": ["block.ln1."],
        "block.ln2": ["block.ln2."],
        "final_norm": ["final_norm."],
    }

    for group_name, group_prefixes in prefixes.items():
        matching = [n for n in all_names
                    if any(n.startswith(p) for p in group_prefixes)]
        if matching:
            groups[group_name] = matching

    return groups


# ── Main ablation runner ───────────────────────────────────────────────────

def run_ablation(model, device, n_eval=500, n_random=3, per_weight=False,
                 eval_seed=12345):
    """Run full ablation study on a model.

    Returns dict with:
      baseline: (exact_acc, digit_acc)
      per_param: {name: {zero: (exact, digit), random: [(exact, digit), ...], shape, numel}}
      per_component: {name: {zero: (exact, digit), params: [...]}}
      per_weight: {name: [(flat_idx, zero_acc), ...]}  (if per_weight=True)
    """
    print(f"\n  Baseline evaluation ({n_eval} samples)...")
    baseline = eval_accuracy(model, device, n_eval, eval_seed)
    print(f"  Baseline: {baseline[0]:.4f} exact, {baseline[1]:.4f} digit")

    results = {
        "baseline": {"exact_acc": baseline[0], "digit_acc": baseline[1]},
        "per_param": {},
        "per_component": {},
    }

    # Per-parameter ablation
    param_names = [n for n, _ in model.named_parameters()]
    print(f"\n  Per-parameter ablation ({len(param_names)} params)...")

    for name in param_names:
        param = dict(model.named_parameters())[name]
        shape = list(param.shape)
        numel = param.numel()

        # Zero ablation
        ablated = ablate_zero(model, name)
        zero_acc = eval_accuracy(ablated, device, n_eval, eval_seed)

        # Random ablation (multiple seeds)
        random_accs = []
        for seed in range(n_random):
            ablated = ablate_random(model, name, seed=seed)
            rand_acc = eval_accuracy(ablated, device, n_eval, eval_seed)
            random_accs.append({"exact_acc": rand_acc[0], "digit_acc": rand_acc[1]})

        mean_random_exact = (sum(r["exact_acc"] for r in random_accs) / len(random_accs)
                             if random_accs else None)
        zero_drop = baseline[0] - zero_acc[0]
        random_drop = baseline[0] - mean_random_exact if mean_random_exact is not None else 0.0

        results["per_param"][name] = {
            "shape": shape,
            "numel": numel,
            "zero": {"exact_acc": zero_acc[0], "digit_acc": zero_acc[1]},
            "random": random_accs,
            "zero_drop": zero_drop,
            "random_drop": random_drop,
        }

        rand_str = (f"rand={mean_random_exact:.4f} (drop={random_drop:+.4f})"
                    if mean_random_exact is not None else "rand=N/A")
        print(f"    {name:40s} [{numel:3d}] "
              f"zero={zero_acc[0]:.4f} (drop={zero_drop:+.4f})  {rand_str}")

    # Component-level ablation
    groups = get_component_groups(model)
    print(f"\n  Component-level ablation ({len(groups)} groups)...")

    for group_name, param_names_in_group in groups.items():
        ablated = copy.deepcopy(model)
        total_numel = 0
        for pname in param_names_in_group:
            p = dict(ablated.named_parameters())[pname]
            total_numel += p.numel()
            with torch.no_grad():
                p.zero_()

        comp_acc = eval_accuracy(ablated, device, n_eval, eval_seed)
        drop = baseline[0] - comp_acc[0]

        results["per_component"][group_name] = {
            "zero": {"exact_acc": comp_acc[0], "digit_acc": comp_acc[1]},
            "params": param_names_in_group,
            "total_numel": total_numel,
            "drop": drop,
        }

        print(f"    {group_name:25s} [{total_numel:3d} params] "
              f"zero={comp_acc[0]:.4f} (drop={drop:+.4f})")

    # Per-weight element sensitivity (optional — expensive for larger models)
    if per_weight:
        total_params = sum(p.numel() for _, p in model.named_parameters())
        print(f"\n  Per-weight sensitivity ({total_params} individual ablations)...")

        results["per_weight"] = {}
        for name, param in model.named_parameters():
            numel = param.numel()
            sensitivities = []
            for idx in range(numel):
                ablated = ablate_element_zero(model, name, idx)
                acc = eval_accuracy(ablated, device, n_eval, eval_seed)
                drop = baseline[0] - acc[0]
                sensitivities.append({
                    "flat_idx": idx,
                    "orig_value": param.data.view(-1)[idx].item(),
                    "exact_acc": acc[0],
                    "drop": drop,
                })

            results["per_weight"][name] = sensitivities

            # Summary
            drops = [s["drop"] for s in sensitivities]
            max_drop = max(drops)
            max_idx = drops.index(max_drop)
            print(f"    {name:40s} max_drop={max_drop:+.4f} at idx={max_idx} "
                  f"(val={sensitivities[max_idx]['orig_value']:.4f})")

    return results


# ── Plotting ───────────────────────────────────────────────────────────────

def plot_ablation_results(results, output_dir, tag="", model=None):
    """Generate ablation plots."""
    if not HAS_MPL:
        print("  matplotlib not available, skipping plots")
        return

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    baseline_exact = results["baseline"]["exact_acc"]

    # Plot 1: Per-parameter zero/random comparison
    per_param = results["per_param"]
    names = list(per_param.keys())
    short_names = [n.replace("block.", "").replace(".weight", "") for n in names]

    zero_drops = [per_param[n]["zero_drop"] for n in names]
    random_drops = [per_param[n]["random_drop"] for n in names]
    numels = [per_param[n]["numel"] for n in names]

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(names))
    width = 0.35

    bars1 = ax.bar(x - width/2, zero_drops, width, label="Zero ablation",
                   color="#d32f2f", alpha=0.8)
    bars2 = ax.bar(x + width/2, random_drops, width, label="Random ablation (mean)",
                   color="#1976d2", alpha=0.8)

    ax.set_xlabel("Parameter")
    ax.set_ylabel("Accuracy Drop (higher = more critical)")
    ax.set_title(f"Per-Parameter Ablation {tag}\n"
                 f"(baseline={baseline_exact:.4f})")
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=8)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(0, color="black", linewidth=0.5)

    # Annotate parameter count on each bar
    for i, (bar, n) in enumerate(zip(bars1, numels)):
        ax.text(bar.get_x() + bar.get_width()/2, -0.02,
                f"n={n}", ha="center", va="top", fontsize=7, color="gray")

    plt.tight_layout()
    path = f"{output_dir}/ablation_per_param.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")

    # Plot 2: Component-level ablation
    per_comp = results["per_component"]
    comp_names = list(per_comp.keys())
    comp_drops = [per_comp[c]["drop"] for c in comp_names]
    comp_numels = [per_comp[c]["total_numel"] for c in comp_names]

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#d32f2f" if d > 0.5 else "#ff9800" if d > 0.1 else "#4caf50"
              for d in comp_drops]
    bars = ax.barh(comp_names, comp_drops, color=colors, alpha=0.8)

    for bar, n in zip(bars, comp_numels):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f"{n}p", ha="left", va="center", fontsize=9)

    ax.set_xlabel("Accuracy Drop (higher = more critical)")
    ax.set_title(f"Component-Level Ablation {tag}\n"
                 f"(red=catastrophic, orange=significant, green=minor)")
    ax.grid(True, alpha=0.3, axis="x")
    ax.axvline(0, color="black", linewidth=0.5)

    plt.tight_layout()
    path = f"{output_dir}/ablation_components.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")

    # Plot 3: Per-weight sensitivity heatmap (if available)
    if "per_weight" in results:
        per_weight = results["per_weight"]
        n = len(per_weight)
        ncols = 4
        nrows = (n + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
        axes = axes.flatten() if n > ncols else (list(axes) if n > 1 else [axes])

        for idx, (name, sensitivities) in enumerate(per_weight.items()):
            ax = axes[idx]
            drops = np.array([s["drop"] for s in sensitivities])

            # Get shape from per_param results
            shape = tuple(results["per_param"][name]["shape"]) if name in results["per_param"] else (len(drops),)

            if len(shape) == 1:
                # 1D: bar chart
                ax.bar(range(len(drops)), drops,
                       color=["#d32f2f" if d > 0.01 else "#4caf50" for d in drops])
            elif len(shape) == 2:
                # 2D: heatmap
                drop_matrix = drops.reshape(shape)
                vmax = max(abs(drops.max()), 0.01)
                im = ax.imshow(drop_matrix, cmap="Reds", aspect="auto", vmin=0, vmax=vmax)
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                # Annotate if small enough
                if drops.size <= 20:
                    for i in range(shape[0]):
                        for j in range(shape[1]):
                            ax.text(j, i, f"{drop_matrix[i,j]:.3f}",
                                    ha="center", va="center", fontsize=6)
            else:
                ax.text(0.5, 0.5, f"shape={list(shape)}", transform=ax.transAxes,
                        ha="center")

            short_name = name.replace("block.", "").replace(".weight", "")
            ax.set_title(f"{short_name}\n{list(shape)}", fontsize=9)

        for idx in range(n, len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle(f"Per-Weight Sensitivity {tag}\n"
                     f"(drop in accuracy when single weight zeroed)",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        path = f"{output_dir}/ablation_per_weight.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved {path}")

    # Plot 4: Summary ranking
    fig, ax = plt.subplots(figsize=(10, 6))

    # Combine param and component data, sorted by drop
    all_items = []
    for n in names:
        all_items.append((
            n.replace("block.", "").replace(".weight", ""),
            per_param[n]["zero_drop"],
            per_param[n]["numel"],
            "param"
        ))
    for c in comp_names:
        all_items.append((
            f"[{c}]",
            per_comp[c]["drop"],
            per_comp[c]["total_numel"],
            "component"
        ))

    all_items.sort(key=lambda x: x[1], reverse=True)

    labels = [f"{item[0]} ({item[2]}p)" for item in all_items]
    drops = [item[1] for item in all_items]
    colors = ["#d32f2f" if item[3] == "component" else "#1976d2" for item in all_items]

    ax.barh(range(len(labels)), drops, color=colors, alpha=0.8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Accuracy Drop when Zeroed")
    ax.set_title(f"Ablation Ranking {tag}\n"
                 f"(blue=individual param, red=component group)")
    ax.grid(True, alpha=0.3, axis="x")
    ax.invert_yaxis()

    plt.tight_layout()
    path = f"{output_dir}/ablation_ranking.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


# ── All-models runner ──────────────────────────────────────────────────────

SUBMITTED_MODELS = {
    "62p": "checkpoints/qwen3_arc_62p_tiekv_tieqo_adam_nowd/best.pt",
    "83p": "checkpoints/qwen3_d3_ff2_83p_tiekv_tieqo_shnorm_s905/best.pt",
    "86p": "checkpoints/qwen3_d3_ff2_86p_tiekv_tieqo_shbnorm_s1_targeted/best.pt",
    "89p": "checkpoints/qwen3_d3_ff2_89p_tiekv_tieqo_s11127/best.pt",
    "95p": "checkpoints/qwen3_arc_95p_ft_s9999/best.pt",
    "96p": "checkpoints/qwen3_rank1_96p_tiekv_s9999/best.pt",
    "101p": "checkpoints/qwen3_d3_ff2_101p_tieqo_s13_targeted/best.pt",
    "122p": "checkpoints/qwen3_d3_ff3_122p_s6/best.pt",
}


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Ablation study for addition transformers")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to model checkpoint (default: 89p)")
    parser.add_argument("--output-dir", default="plots/ablation",
                        help="Output directory for plots")
    parser.add_argument("--all-models", action="store_true",
                        help="Run ablation on all submitted models")
    parser.add_argument("--per-weight", action="store_true",
                        help="Run per-element weight sensitivity (slow)")
    parser.add_argument("--n-random", type=int, default=3,
                        help="Number of random seeds per parameter")
    parser.add_argument("--n-eval", type=int, default=500,
                        help="Number of samples for evaluation")
    parser.add_argument("--save-json", action="store_true",
                        help="Save raw results as JSON")
    args = parser.parse_args()

    device = torch.device("cpu")
    output_dir = args.output_dir

    if args.all_models:
        for label, ckpt_path in sorted(SUBMITTED_MODELS.items()):
            if not Path(ckpt_path).exists():
                print(f"\n  Skipping {label}: checkpoint not found")
                continue

            print(f"\n{'='*60}")
            print(f"  ABLATION: {label}")
            print(f"{'='*60}")

            model, cfg, is_arc = load_model(ckpt_path, device)
            n_params = sum(p.numel() for p in model.parameters())
            tag = f"({label}, {n_params}p)"

            model_output_dir = f"{output_dir}/{label}"
            results = run_ablation(model, device, n_eval=args.n_eval,
                                   n_random=args.n_random,
                                   per_weight=args.per_weight)
            plot_ablation_results(results, model_output_dir, tag, model)

            if args.save_json:
                json_path = f"{model_output_dir}/ablation_results.json"
                Path(model_output_dir).mkdir(parents=True, exist_ok=True)
                with open(json_path, "w") as f:
                    json.dump(results, f, indent=2, default=str)
                print(f"  Saved {json_path}")

    else:
        ckpt_path = args.checkpoint or SUBMITTED_MODELS["89p"]
        if not Path(ckpt_path).exists():
            print(f"ERROR: checkpoint not found: {ckpt_path}")
            sys.exit(1)

        model, cfg, is_arc = load_model(ckpt_path, device)
        n_params = sum(p.numel() for p in model.parameters())
        tag = f"({n_params}p)"

        results = run_ablation(model, device, n_eval=args.n_eval,
                               n_random=args.n_random,
                               per_weight=args.per_weight)
        plot_ablation_results(results, output_dir, tag, model)

        if args.save_json:
            json_path = f"{output_dir}/ablation_results.json"
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            with open(json_path, "w") as f:
                json.dump(results, f, indent=2, default=str)
            print(f"  Saved {json_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
