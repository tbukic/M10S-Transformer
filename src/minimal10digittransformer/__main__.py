"""Command-line entrypoint for Minimal10DigitTransformer experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from minimal10digittransformer.evaluation.evaluator import evaluate_model
from minimal10digittransformer.model.transformer import MinimalTransformer, TransformerConfig, count_parameters
from minimal10digittransformer.training.trainer import Trainer, TrainingConfig


def create_config_from_args(args) -> tuple[TransformerConfig, TrainingConfig]:
    """Create configs from command-line args or config file."""
    if args.config:
        with open(args.config) as f:
            cfg = json.load(f)
        model_config = TransformerConfig(**cfg.get("model", {}))
        training_config = TrainingConfig(**cfg.get("training", {}))
    else:
        model_config = TransformerConfig(
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            d_ff=args.d_ff,
            causal=args.causal,
            pe_type=args.pe_type,
            pe_period=args.pe_period,
            tie_weights=args.tie_weights,
            share_layers=args.share_layers,
            n_layer_repeats=args.n_layer_repeats,
            ffn_type=args.ffn_type,
            use_bias=args.use_bias,
            norm_type=args.norm_type,
            pre_norm=args.pre_norm,
            rank=args.rank,
            use_residual=args.use_residual,
            activation=args.activation,
            embed_dim=args.embed_dim,
            dropout=args.dropout,
        )
        training_config = TrainingConfig(
            lr=args.lr,
            weight_decay=args.weight_decay,
            optimizer=args.optimizer,
            scheduler=args.scheduler,
            warmup_steps=args.warmup_steps,
            epochs=args.epochs,
            batch_size=args.batch_size,
            train_samples=args.train_samples,
            eval_samples=args.eval_samples,
            eval_interval=args.eval_interval,
            use_curriculum=args.use_curriculum,
            format=args.format,
            seed=args.seed,
            experiment_name=args.name,
            use_wandb=not args.no_wandb,
            wandb_tags=args.tags.split(",") if args.tags else None,
        )

    return model_config, training_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal Transformer for 10-digit Addition")
    parser.add_argument("--config", type=str, help="Path to config JSON file")
    parser.add_argument("--name", type=str, default="experiment", help="Experiment name")

    # Model args
    parser.add_argument("--d-model", type=int, default=8)
    parser.add_argument("--n-heads", type=int, default=1)
    parser.add_argument("--n-layers", type=int, default=1)
    parser.add_argument("--d-ff", type=int, default=16)
    parser.add_argument("--causal", action="store_true", default=True)
    parser.add_argument("--no-causal", dest="causal", action="store_false")
    parser.add_argument("--pe-type", type=str, default="sinusoidal")
    parser.add_argument("--pe-period", type=float, default=11.0)
    parser.add_argument("--tie-weights", action="store_true", default=True)
    parser.add_argument("--no-tie-weights", dest="tie_weights", action="store_false")
    parser.add_argument("--share-layers", action="store_true", default=False)
    parser.add_argument("--n-layer-repeats", type=int, default=1)
    parser.add_argument("--ffn-type", type=str, default="standard")
    parser.add_argument("--use-bias", action="store_true", default=False)
    parser.add_argument("--norm-type", type=str, default="rmsnorm")
    parser.add_argument("--pre-norm", action="store_true", default=True)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--use-residual", action="store_true", default=True)
    parser.add_argument("--no-residual", dest="use_residual", action="store_false")
    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--embed-dim", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=0.0)

    # Training args
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--optimizer", type=str, default="adamw")
    parser.add_argument("--scheduler", type=str, default="cosine")
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--train-samples", type=int, default=100000)
    parser.add_argument("--eval-samples", type=int, default=10000)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--use-curriculum", action="store_true", default=False)
    parser.add_argument("--format", type=str, default="plain")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-wandb", action="store_true", default=False)
    parser.add_argument("--tags", type=str, default=None)

    # Eval only
    parser.add_argument("--eval-only", type=str, default=None, help="Path to checkpoint for eval")

    args = parser.parse_args()
    model_config, training_config = create_config_from_args(args)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create model
    model = MinimalTransformer(model_config)
    n_params = count_parameters(model)
    print(f"Model has {n_params} trainable parameters")
    print(f"Config: {model_config}")

    if args.eval_only:
        checkpoint = torch.load(args.eval_only, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)
        results = evaluate_model(
            model, n_samples=10000, device=device, format=training_config.format, verbose=True
        )
        print(f"Results: {json.dumps(results, indent=2, default=str)}")
        return

    # Train
    trainer = Trainer(model, model_config, training_config, device=device)
    trainer.train()


if __name__ == "__main__":
    main()
