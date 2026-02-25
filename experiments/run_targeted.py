"""Targeted experiments based on research insights.

Key insights from literature review:
1. Reversed output (LSB-first) is critical for carry propagation
2. Period-11 sinusoidal PE aligns digit columns
3. Looped/shared layers are most parameter-efficient
4. Low-rank factorization reduces parameters dramatically
5. ~4 ReLU carry neurons are theoretically sufficient for carry logic
6. Curriculum learning from 1-digit to 10-digit helps
7. Very long training (grokking) can be crucial
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from minimal10digittransformer.evaluation.evaluator import evaluate_model
from minimal10digittransformer.model.transformer import MinimalTransformer, TransformerConfig, count_parameters
from minimal10digittransformer.training.trainer import Trainer, TrainingConfig


def run_single(name, model_config, train_config, device="cuda"):
    model = MinimalTransformer(model_config)
    n = count_parameters(model)
    print(f"\n{'='*60}")
    print(f"{name} | {n} params")
    print(f"{'='*60}")
    train_config.experiment_name = name
    trainer = Trainer(model, model_config, train_config, device=device)
    return trainer.train()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default="all")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_wandb = not args.no_wandb

    # Base training config optimized for tiny models
    def make_train_config(name, epochs=5000, lr=3e-3, curriculum=True, **kwargs):
        return TrainingConfig(
            lr=lr,
            weight_decay=0.01,
            optimizer="adamw",
            scheduler="cosine",
            warmup_steps=200,
            epochs=epochs,
            batch_size=512,
            train_samples=200000,
            eval_samples=10000,
            eval_interval=200,
            log_interval=50,
            format="plain_reversed_output",
            use_curriculum=curriculum,
            curriculum_start_digits=2,
            curriculum_epochs_per_level=300,
            seed=42,
            experiment_name=name,
            use_wandb=use_wandb,
            wandb_tags=["targeted", "v2"],
            **kwargs,
        )

    experiments = {
        # --- LOOPED TRANSFORMER EXPERIMENTS ---
        # Based on research: looped transformers achieve <10% params with equivalent accuracy

        "looped_d8_r10": (
            TransformerConfig(
                d_model=8, n_heads=1, n_layers=1, d_ff=8,
                share_layers=True, n_layer_repeats=10,
                pe_type="sinusoidal", pe_period=11.0,
                activation="relu", norm_type="rmsnorm",
            ),
            make_train_config("looped_d8_r10", epochs=10000),
        ),

        "looped_d8_r20": (
            TransformerConfig(
                d_model=8, n_heads=1, n_layers=1, d_ff=8,
                share_layers=True, n_layer_repeats=20,
                pe_type="sinusoidal", pe_period=11.0,
                activation="relu", norm_type="rmsnorm",
            ),
            make_train_config("looped_d8_r20", epochs=10000),
        ),

        "looped_d4_r20_lowrank": (
            TransformerConfig(
                d_model=4, n_heads=1, n_layers=1, d_ff=4,
                share_layers=True, n_layer_repeats=20,
                pe_type="sinusoidal", pe_period=11.0,
                rank=2, activation="relu", norm_type="none",
            ),
            make_train_config("looped_d4_r20_lowrank", epochs=15000, lr=5e-3),
        ),

        "looped_d6_r15": (
            TransformerConfig(
                d_model=6, n_heads=1, n_layers=1, d_ff=6,
                share_layers=True, n_layer_repeats=15,
                pe_type="sinusoidal", pe_period=11.0,
                activation="relu", norm_type="rmsnorm",
            ),
            make_train_config("looped_d6_r15", epochs=10000),
        ),

        # --- FACTORIZED EMBEDDING EXPERIMENTS ---
        "factored_d8_e2_looped": (
            TransformerConfig(
                d_model=8, n_heads=1, n_layers=1, d_ff=8,
                share_layers=True, n_layer_repeats=15,
                pe_type="sinusoidal", pe_period=11.0,
                embed_dim=2, activation="relu", norm_type="rmsnorm",
            ),
            make_train_config("factored_d8_e2_looped", epochs=10000),
        ),

        # --- LOW-RANK + LOOPED EXPERIMENTS ---
        "lowrank_r1_looped_d8": (
            TransformerConfig(
                d_model=8, n_heads=1, n_layers=1, d_ff=4,
                share_layers=True, n_layer_repeats=20,
                pe_type="sinusoidal", pe_period=11.0,
                rank=1, activation="relu", norm_type="none",
            ),
            make_train_config("lowrank_r1_looped_d8", epochs=15000, lr=5e-3),
        ),

        "lowrank_r2_looped_d8": (
            TransformerConfig(
                d_model=8, n_heads=1, n_layers=1, d_ff=4,
                share_layers=True, n_layer_repeats=20,
                pe_type="sinusoidal", pe_period=11.0,
                rank=2, activation="relu", norm_type="rmsnorm",
            ),
            make_train_config("lowrank_r2_looped_d8", epochs=10000),
        ),

        # --- NO FFN EXPERIMENTS (attention only) ---
        "attn_only_looped_d8_r20": (
            TransformerConfig(
                d_model=8, n_heads=1, n_layers=1,
                ffn_type="none", share_layers=True, n_layer_repeats=20,
                pe_type="sinusoidal", pe_period=11.0,
                activation="relu", norm_type="rmsnorm",
            ),
            make_train_config("attn_only_looped_d8_r20", epochs=10000),
        ),

        # --- ABACUS PE ---
        "abacus_looped_d8_r15": (
            TransformerConfig(
                d_model=8, n_heads=1, n_layers=1, d_ff=8,
                share_layers=True, n_layer_repeats=15,
                pe_type="abacus",
                activation="relu", norm_type="rmsnorm",
            ),
            make_train_config("abacus_looped_d8_r15", epochs=10000, curriculum=False),
        ),

        # --- VERY LONG TRAINING (grokking) ---
        "grokking_d8_looped": (
            TransformerConfig(
                d_model=8, n_heads=1, n_layers=1, d_ff=8,
                share_layers=True, n_layer_repeats=15,
                pe_type="sinusoidal", pe_period=11.0,
                activation="relu", norm_type="rmsnorm",
            ),
            make_train_config("grokking_d8_looped", epochs=30000, lr=1e-3),
        ),

        # --- ULTRA MINIMAL ---
        "ultra_d4_r1_looped30": (
            TransformerConfig(
                d_model=4, n_heads=1, n_layers=1, d_ff=4,
                share_layers=True, n_layer_repeats=30,
                pe_type="sinusoidal", pe_period=11.0,
                rank=1, activation="relu", norm_type="none",
                use_bias=True,
            ),
            make_train_config("ultra_d4_r1_looped30", epochs=20000, lr=1e-2),
        ),
    }

    if args.experiment != "all":
        exp_names = args.experiment.split(",")
        experiments = {k: v for k, v in experiments.items() if k in exp_names}

    results = []
    for name, (model_config, train_config) in experiments.items():
        model = MinimalTransformer(model_config)
        n = count_parameters(model)
        print(f"\n  Config {name}: {n} params")

    print(f"\nRunning {len(experiments)} experiments on {device}")

    for name, (model_config, train_config) in experiments.items():
        try:
            best_acc = run_single(name, model_config, train_config, device)
            model = MinimalTransformer(model_config)
            results.append({
                "name": name,
                "n_params": count_parameters(model),
                "best_accuracy": best_acc,
            })
        except Exception as e:
            print(f"ERROR in {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append({"name": name, "error": str(e)})

        # Save intermediate
        with open("experiments/targeted_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    for r in sorted(results, key=lambda x: x.get("n_params", 99999)):
        if "error" in r:
            print(f"  {r['name']}: ERROR")
        else:
            print(f"  {r['name']}: {r['n_params']} params, accuracy={r['best_accuracy']:.4f}")


if __name__ == "__main__":
    main()
