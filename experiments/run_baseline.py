"""Baseline experiments to establish parameter count vs accuracy tradeoffs.

Run multiple configurations to find the Pareto frontier of params vs accuracy.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from minimal10digittransformer.evaluation.evaluator import evaluate_model
from minimal10digittransformer.model.transformer import MinimalTransformer, TransformerConfig, count_parameters
from minimal10digittransformer.training.trainer import Trainer, TrainingConfig


def run_experiment(
    name: str,
    model_config: TransformerConfig,
    training_config: TrainingConfig,
    device: str = "cuda",
) -> dict:
    """Run a single experiment and return results."""
    model = MinimalTransformer(model_config)
    n_params = count_parameters(model)
    print(f"\n{'='*60}")
    print(f"Experiment: {name} | Params: {n_params}")
    print(f"{'='*60}")

    training_config.experiment_name = name

    trainer = Trainer(model, model_config, training_config, device=device)
    best_accuracy = trainer.train()

    return {
        "name": name,
        "n_params": n_params,
        "best_accuracy": best_accuracy,
        "model_config": asdict(model_config),
        "training_config": asdict(training_config),
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = []

    # Baseline configs exploring the parameter-accuracy frontier
    experiments = [
        # Experiment 1: Standard small transformer (baseline)
        (
            "baseline_656p",
            TransformerConfig(
                d_model=8, n_heads=1, n_layers=1, d_ff=16,
                pe_type="sinusoidal", pe_period=11.0,
                activation="relu", norm_type="rmsnorm",
            ),
            TrainingConfig(
                lr=3e-3, weight_decay=0.01, epochs=2000,
                batch_size=512, train_samples=200000, eval_samples=10000,
                eval_interval=100, format="plain",
                use_wandb=True, wandb_tags=["baseline"],
            ),
        ),
        # Experiment 2: Reversed output format
        (
            "reversed_656p",
            TransformerConfig(
                d_model=8, n_heads=1, n_layers=1, d_ff=16,
                pe_type="sinusoidal", pe_period=11.0,
                activation="relu", norm_type="rmsnorm",
            ),
            TrainingConfig(
                lr=3e-3, weight_decay=0.01, epochs=2000,
                batch_size=512, train_samples=200000, eval_samples=10000,
                eval_interval=100, format="plain_reversed_output",
                use_wandb=True, wandb_tags=["baseline", "reversed_output"],
            ),
        ),
        # Experiment 3: Low-rank attention
        (
            "lowrank_r2_336p",
            TransformerConfig(
                d_model=8, n_heads=1, n_layers=1, d_ff=8,
                pe_type="sinusoidal", pe_period=11.0,
                rank=2, activation="relu", norm_type="rmsnorm",
            ),
            TrainingConfig(
                lr=3e-3, weight_decay=0.01, epochs=3000,
                batch_size=512, train_samples=200000, eval_samples=10000,
                eval_interval=100, format="plain_reversed_output",
                use_wandb=True, wandb_tags=["lowrank"],
            ),
        ),
        # Experiment 4: No FFN, attention only
        (
            "no_ffn_attn_only",
            TransformerConfig(
                d_model=8, n_heads=1, n_layers=1,
                ffn_type="none", pe_type="sinusoidal", pe_period=11.0,
                activation="relu", norm_type="rmsnorm",
            ),
            TrainingConfig(
                lr=3e-3, weight_decay=0.01, epochs=3000,
                batch_size=512, train_samples=200000, eval_samples=10000,
                eval_interval=100, format="plain_reversed_output",
                use_wandb=True, wandb_tags=["no_ffn"],
            ),
        ),
        # Experiment 5: Layer sharing with repeats
        (
            "shared_3repeats",
            TransformerConfig(
                d_model=8, n_heads=1, n_layers=1, d_ff=8,
                share_layers=True, n_layer_repeats=3,
                pe_type="sinusoidal", pe_period=11.0,
                activation="relu", norm_type="rmsnorm",
            ),
            TrainingConfig(
                lr=3e-3, weight_decay=0.01, epochs=3000,
                batch_size=512, train_samples=200000, eval_samples=10000,
                eval_interval=100, format="plain_reversed_output",
                use_wandb=True, wandb_tags=["shared_layers"],
            ),
        ),
        # Experiment 6: Ultra-tiny with low rank and factored embed
        (
            "ultra_tiny",
            TransformerConfig(
                d_model=4, n_heads=1, n_layers=1, d_ff=4,
                pe_type="sinusoidal", pe_period=11.0,
                rank=2, embed_dim=2, activation="relu",
                norm_type="none",
            ),
            TrainingConfig(
                lr=5e-3, weight_decay=0.005, epochs=5000,
                batch_size=512, train_samples=200000, eval_samples=10000,
                eval_interval=200, format="plain_reversed_output",
                use_wandb=True, wandb_tags=["ultra_tiny"],
            ),
        ),
        # Experiment 7: Curriculum learning
        (
            "curriculum_656p",
            TransformerConfig(
                d_model=8, n_heads=1, n_layers=1, d_ff=16,
                pe_type="sinusoidal", pe_period=11.0,
                activation="relu", norm_type="rmsnorm",
            ),
            TrainingConfig(
                lr=3e-3, weight_decay=0.01, epochs=3000,
                batch_size=512, train_samples=200000, eval_samples=10000,
                eval_interval=100, format="plain_reversed_output",
                use_curriculum=True, curriculum_start_digits=1,
                curriculum_epochs_per_level=200,
                use_wandb=True, wandb_tags=["curriculum"],
            ),
        ),
        # Experiment 8: Abacus PE
        (
            "abacus_pe",
            TransformerConfig(
                d_model=8, n_heads=1, n_layers=1, d_ff=16,
                pe_type="abacus",
                activation="relu", norm_type="rmsnorm",
            ),
            TrainingConfig(
                lr=3e-3, weight_decay=0.01, epochs=2000,
                batch_size=512, train_samples=200000, eval_samples=10000,
                eval_interval=100, format="plain",
                use_wandb=True, wandb_tags=["abacus_pe"],
            ),
        ),
    ]

    for name, model_config, train_config in experiments:
        try:
            result = run_experiment(name, model_config, train_config, device)
            results.append(result)

            # Save intermediate results
            results_file = Path("experiments/baseline_results.json")
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2, default=str)

        except Exception as e:
            print(f"ERROR in {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append({"name": name, "error": str(e)})

    # Print summary
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    for r in results:
        if "error" in r:
            print(f"  {r['name']}: ERROR - {r['error']}")
        else:
            print(f"  {r['name']}: {r['n_params']} params, best_accuracy={r['best_accuracy']:.4f}")


if __name__ == "__main__":
    main()
