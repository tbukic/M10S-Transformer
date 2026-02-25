"""Training loop with wandb logging and comprehensive metrics."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from minimal10digittransformer.data.dataset import AdditionDataset
from minimal10digittransformer.evaluation.evaluator import evaluate_by_difficulty, evaluate_model
from minimal10digittransformer.model.transformer import TransformerConfig, count_parameters


@dataclass
class TrainingConfig:
    """Training configuration."""

    # Optimization
    lr: float = 1e-3
    weight_decay: float = 0.01
    optimizer: str = "adamw"  # adamw, adam, sgd
    betas: tuple[float, float] = (0.9, 0.999)

    # Schedule
    scheduler: str = "cosine"  # cosine, linear, none, cyclic
    warmup_steps: int = 100
    min_lr: float = 1e-5

    # Training
    epochs: int = 1000
    batch_size: int = 256
    train_samples: int = 100000
    eval_samples: int = 10000
    eval_interval: int = 50  # Evaluate every N epochs
    save_interval: int = 100
    log_interval: int = 10

    # Curriculum
    use_curriculum: bool = False
    curriculum_start_digits: int = 1
    curriculum_end_digits: int = 10
    curriculum_epochs_per_level: int = 100

    # Data
    max_digits: int = 10
    format: str = "plain"  # plain, reversed, plain_reversed_output

    # Reproducibility
    seed: int = 42
    eval_seed: int = 12345  # Different seed for eval

    # Output
    output_dir: str = "checkpoints"
    experiment_name: str = "minimal_addition"

    # Wandb
    use_wandb: bool = True
    wandb_project: str = "minimal-10digit-transformer"
    wandb_tags: list[str] | None = None


class Trainer:
    """Training loop for minimal transformer."""

    def __init__(
        self,
        model: nn.Module,
        model_config: TransformerConfig,
        training_config: TrainingConfig,
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.model_config = model_config
        self.config = training_config
        self.device = device
        self.n_params = count_parameters(model)

        # Setup output directory
        self.output_dir = Path(training_config.output_dir) / training_config.experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup optimizer
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()

        # Setup wandb
        self.wandb_run = None
        if training_config.use_wandb:
            self._init_wandb()

        # Training state
        self.global_step = 0
        self.best_accuracy = 0.0
        self.best_epoch = 0

    def _create_optimizer(self) -> torch.optim.Optimizer:
        params = [p for p in self.model.parameters() if p.requires_grad]
        if self.config.optimizer == "adamw":
            return torch.optim.AdamW(
                params,
                lr=self.config.lr,
                weight_decay=self.config.weight_decay,
                betas=self.config.betas,
            )
        elif self.config.optimizer == "adam":
            return torch.optim.Adam(params, lr=self.config.lr, betas=self.config.betas)
        elif self.config.optimizer == "sgd":
            return torch.optim.SGD(
                params, lr=self.config.lr, weight_decay=self.config.weight_decay, momentum=0.9
            )
        raise ValueError(f"Unknown optimizer: {self.config.optimizer}")

    def _create_scheduler(self):
        if self.config.scheduler == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs,
                eta_min=self.config.min_lr,
            )
        elif self.config.scheduler == "linear":
            return torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=self.config.min_lr / self.config.lr,
                total_iters=self.config.epochs,
            )
        elif self.config.scheduler == "cyclic":
            return torch.optim.lr_scheduler.CyclicLR(
                self.optimizer,
                base_lr=self.config.min_lr,
                max_lr=self.config.lr,
                step_size_up=self.config.epochs // 10,
                mode="triangular2",
            )
        return None

    def _init_wandb(self):
        try:
            import wandb

            config_dict = {
                "model": asdict(self.model_config),
                "training": asdict(self.config),
                "n_params": self.n_params,
            }

            self.wandb_run = wandb.init(
                project=self.config.wandb_project,
                name=self.config.experiment_name,
                config=config_dict,
                tags=self.config.wandb_tags or [],
            )
        except Exception as e:
            print(f"Warning: wandb init failed: {e}")
            self.wandb_run = None

    def _get_curriculum_digits(self, epoch: int) -> int:
        """Get current max digits for curriculum learning."""
        if not self.config.use_curriculum:
            return self.config.max_digits
        level = min(
            epoch // self.config.curriculum_epochs_per_level,
            self.config.curriculum_end_digits - self.config.curriculum_start_digits,
        )
        return self.config.curriculum_start_digits + level

    def train(self):
        """Run full training loop."""
        torch.manual_seed(self.config.seed)

        print(f"Training {self.config.experiment_name}")
        print(f"Parameters: {self.n_params}")
        print(f"Model config: {self.model_config}")
        print(f"Device: {self.device}")

        # Save configs
        with open(self.output_dir / "config.json", "w") as f:
            json.dump(
                {
                    "model": asdict(self.model_config),
                    "training": asdict(self.config),
                    "n_params": self.n_params,
                },
                f,
                indent=2,
                default=str,
            )

        start_time = time.time()

        for epoch in range(self.config.epochs):
            # Curriculum learning
            current_max_digits = self._get_curriculum_digits(epoch)

            # Create training dataset (fresh each epoch for no data leakage)
            train_dataset = AdditionDataset(
                size=self.config.train_samples,
                max_digits=current_max_digits,
                format=self.config.format,
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=0,
                drop_last=True,
            )

            # Training epoch
            epoch_loss = self._train_epoch(train_loader, epoch)

            # LR scheduling
            if self.scheduler is not None:
                self.scheduler.step()

            # Logging
            if epoch % self.config.log_interval == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                elapsed = time.time() - start_time
                print(
                    f"Epoch {epoch:5d} | Loss: {epoch_loss:.4f} | "
                    f"LR: {lr:.2e} | Digits: {current_max_digits} | "
                    f"Time: {elapsed:.0f}s"
                )

                if self.wandb_run:
                    import wandb

                    wandb.log(
                        {
                            "train/loss": epoch_loss,
                            "train/lr": lr,
                            "train/epoch": epoch,
                            "train/max_digits": current_max_digits,
                        },
                        step=self.global_step,
                    )

            # Evaluation
            if epoch % self.config.eval_interval == 0 and epoch > 0:
                self._evaluate_and_log(epoch)

            # Save checkpoint
            if epoch % self.config.save_interval == 0 and epoch > 0:
                self._save_checkpoint(epoch)

        # Final evaluation
        self._evaluate_and_log(self.config.epochs, final=True)
        self._save_checkpoint(self.config.epochs, final=True)

        if self.wandb_run:
            import wandb

            wandb.finish()

        print(f"\nTraining complete! Best accuracy: {self.best_accuracy:.4f} at epoch {self.best_epoch}")
        return self.best_accuracy

    def _train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)

            # Forward
            logits = self.model(input_ids)

            # Compute loss only on output positions (where labels != -100)
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )

            # Backward
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            self.global_step += 1

        return total_loss / max(n_batches, 1)

    def _evaluate_and_log(self, epoch: int, final: bool = False):
        """Run evaluation and log results."""
        # Full evaluation
        results = evaluate_model(
            self.model,
            n_samples=self.config.eval_samples,
            max_digits=self.config.max_digits,
            seed=self.config.eval_seed,
            device=self.device,
            format=self.config.format,
            verbose=final,
        )

        accuracy = results["exact_match_accuracy"]
        digit_acc = results["digit_accuracy"]

        print(
            f"  EVAL Epoch {epoch}: Exact={accuracy:.4f} ({results['correct']}/{results['total']}), "
            f"Digit={digit_acc:.4f}"
        )

        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.best_epoch = epoch
            self._save_checkpoint(epoch, best=True)
            print(f"  ** New best accuracy: {accuracy:.4f} **")

        # Per-digit breakdown
        if final or accuracy > 0.5:
            difficulty_results = evaluate_by_difficulty(
                self.model,
                max_digits=self.config.max_digits,
                n_per_level=1000,
                seed=self.config.eval_seed,
                device=self.device,
                format=self.config.format,
            )
            for n_digits, acc in difficulty_results.items():
                print(f"    {n_digits}-digit: {acc:.4f}")

            if self.wandb_run:
                import wandb

                for n_digits, acc in difficulty_results.items():
                    wandb.log(
                        {f"eval/accuracy_{n_digits}digit": acc},
                        step=self.global_step,
                    )

        if self.wandb_run:
            import wandb

            wandb.log(
                {
                    "eval/exact_match_accuracy": accuracy,
                    "eval/digit_accuracy": digit_acc,
                    "eval/best_accuracy": self.best_accuracy,
                    "eval/epoch": epoch,
                },
                step=self.global_step,
            )

        if final and "wrong_examples" in results:
            print("  Wrong examples:")
            for ex in results["wrong_examples"][:10]:
                print(f"    {ex}")

    def _save_checkpoint(self, epoch: int, best: bool = False, final: bool = False):
        """Save model checkpoint."""
        if best:
            path = self.output_dir / "best_model.pt"
        elif final:
            path = self.output_dir / "final_model.pt"
        else:
            path = self.output_dir / f"checkpoint_epoch{epoch}.pt"

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_accuracy": self.best_accuracy,
                "n_params": self.n_params,
                "model_config": asdict(self.model_config),
                "training_config": asdict(self.config),
            },
            path,
        )
