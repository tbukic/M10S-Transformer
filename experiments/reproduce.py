"""Unified reproduction pipeline for all claimed results.

Reproduces ANY claimed model from random initialization with a single command.
Automates all training phases (base, fine-tuning, targeted FT) in one run.

Usage:
    # List all available configs
    python experiments/reproduce.py --list

    # Reproduce 122p baseline (simple, single phase)
    python experiments/reproduce.py --config 122p --seed 6 --device cuda

    # Reproduce 89p star result (4-phase pipeline)
    python experiments/reproduce.py --config 89p --seed 11127 --device cuda

    # Reproduce 83p with targeted FT (base + FT + iterated targeted)
    python experiments/reproduce.py --config 83p --seed 905 --train-eval-seed 888 --device cuda

    # Smoke test (override step counts)
    python experiments/reproduce.py --config 122p --seed 42 --steps-override 200 --device cpu

    # Skip final validation
    python experiments/reproduce.py --config 89p --seed 0 --no-validate --device cuda
"""

import argparse
import csv
import json
import math
import os
import random
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from minimal10digittransformer.model.qwen3 import (
    Qwen3AdditionModel,
    VOCAB_SIZE,
    INPUT_LEN,
    OUTPUT_LEN,
    TOTAL_LEN,
    MAX_ADDEND,
)
from minimal10digittransformer.data.addition import (
    encode,
    expected_output,
    generate_batch,
    generate_test_set,
    load_test_set,
)
from minimal10digittransformer.evaluation.metrics import evaluate, evaluate_detailed


# ============================================================================
# Configuration registry
# ============================================================================

# Reserved seeds (NEVER use for train-eval):
#   42    = test_10k.json generation
#   2025  = verify.py official test
#   123   = holdout 10K
#   99    = holdout 50K independent
RESERVED_SEEDS = {42, 2025, 123, 99}

# Each config defines:
#   model_args: kwargs for Qwen3AdditionModel
#   params: expected unique parameter count
#   phases: list of phase dicts, each with:
#       type: "cosine" | "constant" | "lbfgs" | "targeted"
#       lr, batch_size, steps, weight_decay (optional, default 0.01)
#       warmup (optional, default 0)
#       eval_interval (optional, default 2000)
#       use_final (optional, default False): load final.pt instead of best.pt
#           for the NEXT phase's starting checkpoint
#       -- lbfgs-specific --
#       max_iter (optional, default 30): L-BFGS iterations per step
#       history_size (optional, default 10): L-BFGS history size
#       -- targeted-specific --
#       max_iters (optional, default 10)
#       steps_per_iter (optional, default 5000)
CONFIGS = {
    "122p": {
        "desc": "d=3 ff=3, 1h/1kv -- baseline, no weight tying",
        "params": 122,
        "model_args": {
            "d_model": 3, "ff": 3,
            "n_heads": 1, "n_kv_heads": 1,
            "head_dim": 4, "rope_theta": 3.0,
        },
        "phases": [
            {"type": "cosine", "lr": 0.01, "batch_size": 128,
             "steps": 200000, "eval_interval": 5000},
        ],
    },
    "113p": {
        "desc": "d=3 ff=2, 1h/1kv -- reduced MLP",
        "params": 113,
        "model_args": {
            "d_model": 3, "ff": 2,
            "n_heads": 1, "n_kv_heads": 1,
            "head_dim": 4, "rope_theta": 3.0,
        },
        "phases": [
            {"type": "cosine", "lr": 0.01, "batch_size": 128,
             "steps": 50000, "eval_interval": 2000},
            {"type": "constant", "lr": 0.001, "batch_size": 256,
             "steps": 30000, "eval_interval": 2000},
        ],
    },
    "101p": {
        "desc": "d=3 ff=2, 1h/1kv, tieQO -- output = Q^T",
        "params": 101,
        "model_args": {
            "d_model": 3, "ff": 2,
            "n_heads": 1, "n_kv_heads": 1,
            "head_dim": 4, "rope_theta": 3.0,
            "tie_qo": True,
        },
        "phases": [
            {"type": "cosine", "lr": 0.01, "batch_size": 128,
             "steps": 50000, "eval_interval": 2000},
            {"type": "targeted", "lr": 0.0003, "batch_size": 256,
             "steps_per_iter": 1000, "max_iters": 10, "eval_interval": 500},
        ],
    },
    "89p": {
        "desc": "d=3 ff=2, 1h/1kv, tieKV+tieQO -- star result (5-phase natural)",
        "params": 89,
        "model_args": {
            "d_model": 3, "ff": 2,
            "n_heads": 1, "n_kv_heads": 1,
            "head_dim": 4, "rope_theta": 3.0,
            "tie_kv": True, "tie_qo": True,
        },
        "phases": [
            {"type": "cosine", "lr": 0.01, "batch_size": 128,
             "steps": 50000, "eval_interval": 2000},
            {"type": "constant", "lr": 0.001, "batch_size": 256,
             "steps": 30000, "eval_interval": 2000},
            {"type": "constant", "lr": 0.0003, "batch_size": 128,
             "steps": 30000, "eval_interval": 2000},
            {"type": "constant", "lr": 0.001, "batch_size": 256,
             "steps": 300000, "eval_interval": 5000},
            {"type": "constant", "lr": 0.0003, "batch_size": 256,
             "steps": 20000, "eval_interval": 500},
        ],
    },
    "86p": {
        "desc": "d=3 ff=2, 1h/1kv, tieKV+tieQO+share_block_norms -- L-BFGS/targeted",
        "params": 86,
        "model_args": {
            "d_model": 3, "ff": 2,
            "n_heads": 1, "n_kv_heads": 1,
            "head_dim": 4, "rope_theta": 3.0,
            "tie_kv": True, "tie_qo": True,
            "share_block_norms": True,
        },
        "phases": [
            {"type": "cosine", "lr": 0.01, "batch_size": 128,
             "steps": 100000, "eval_interval": 2000},
            {"type": "lbfgs", "lr": 0.5, "batch_size": 512,
             "steps": 300, "max_iter": 30, "history_size": 10,
             "eval_interval": 50},
            {"type": "targeted", "lr": 0.0003, "batch_size": 256,
             "steps_per_iter": 500, "max_iters": 10, "eval_interval": 100},
        ],
    },
    "83p": {
        "desc": "d=3 ff=2, 1h/1kv, tieKV+tieQO+share_norms -- iterated targeted",
        "params": 83,
        "model_args": {
            "d_model": 3, "ff": 2,
            "n_heads": 1, "n_kv_heads": 1,
            "head_dim": 4, "rope_theta": 3.0,
            "tie_kv": True, "tie_qo": True,
            "share_norms": True,
        },
        "phases": [
            {"type": "cosine", "lr": 0.01, "batch_size": 128,
             "steps": 100000, "eval_interval": 2000, "use_final": True},
            {"type": "constant", "lr": 0.001, "batch_size": 128,
             "steps": 30000, "eval_interval": 2000},
            {"type": "constant", "lr": 0.0003, "batch_size": 128,
             "steps": 30000, "eval_interval": 2000},
            {"type": "targeted", "lr": 0.0003, "batch_size": 256,
             "steps_per_iter": 5000, "max_iters": 10, "eval_interval": 1000},
        ],
    },
}


# ============================================================================
# Training utilities
# ============================================================================

def count_params(model: nn.Module) -> int:
    """Count unique trainable parameters."""
    return sum(p.numel() for p in model.parameters())


def train_step(model: Qwen3AdditionModel, full_seq: torch.Tensor,
               labels: torch.Tensor) -> torch.Tensor:
    """Single forward/backward step with teacher forcing.

    full_seq: [B, 35] prompt + target tokens
    labels: [B, 35] with -100 for masked positions
    """
    logits = model(full_seq)  # [B, 35, V]
    # Shift: logits[t] predicts token[t+1]
    shift_logits = logits[:, :-1, :].reshape(-1, VOCAB_SIZE)
    shift_labels = labels[:, 1:].reshape(-1)
    return F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)


def save_checkpoint(model: Qwen3AdditionModel, path: str, step: int,
                    accuracy: float, n_params: int, config: dict,
                    extra: dict | None = None):
    """Save a checkpoint with metadata."""
    data = {
        "state_dict": model.state_dict(),
        "step": step,
        "accuracy": accuracy,
        "n_params": n_params,
        "config": config,
    }
    if extra:
        data.update(extra)
    torch.save(data, path)


# ============================================================================
# Phase runners
# ============================================================================

def run_cosine_phase(model: Qwen3AdditionModel, device: torch.device,
                     phase_cfg: dict, eval_pairs: list[tuple[int, int]],
                     ckpt_dir: str, phase_num: int, model_config: dict,
                     n_params: int, metrics_writer, t0: float,
                     global_step: int, steps_override: int | None = None,
                     ) -> tuple[float, int]:
    """Run a cosine LR decay training phase.

    Returns (best_accuracy, updated_global_step).
    """
    lr = phase_cfg["lr"]
    batch_size = phase_cfg["batch_size"]
    steps = steps_override if steps_override is not None else phase_cfg["steps"]
    warmup = phase_cfg.get("warmup", 0)
    eval_interval = phase_cfg.get("eval_interval", 2000)
    weight_decay = phase_cfg.get("weight_decay", 0.01)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_acc = 0.0

    print(f"\n  Phase {phase_num} [cosine]: lr={lr}, batch={batch_size}, "
          f"steps={steps}, warmup={warmup}")

    for step in range(1, steps + 1):
        # Cosine LR with optional warmup
        if step <= warmup:
            cur_lr = lr * step / max(warmup, 1)
        else:
            progress = (step - warmup) / max(steps - warmup, 1)
            cur_lr = lr / 10 + 0.5 * (lr - lr / 10) * (1 + math.cos(math.pi * progress))
        for pg in optimizer.param_groups:
            pg["lr"] = cur_lr

        model.train()
        full_seq, labels = generate_batch(batch_size, device, max_digits=10)

        optimizer.zero_grad()
        loss = train_step(model, full_seq, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        global_step += 1

        is_eval_step = (step % eval_interval == 0)

        if step % 100 == 0 and not is_eval_step:
            elapsed = time.time() - t0
            print(f"    step {step:6d}/{steps} | loss {loss.item():.4f} | "
                  f"lr {cur_lr:.2e} | {elapsed:.0f}s")
            sys.stdout.flush()
            metrics_writer.writerow([
                phase_num, global_step, step, f"{loss.item():.6f}",
                f"{cur_lr:.2e}", "", "", f"{elapsed:.1f}",
            ])
            if step % 1000 == 0:
                metrics_writer.flush_file()

        if is_eval_step:
            # Quick eval on subset
            quick_pairs = eval_pairs[:500]
            seq_acc, dig_acc = evaluate(model, device, test_pairs=quick_pairs)
            elapsed = time.time() - t0
            print(f"    step {step:6d}/{steps} | loss {loss.item():.4f} | "
                  f"lr {cur_lr:.2e} | {elapsed:.0f}s")
            print(f"    EVAL step {step}: exact={seq_acc:.4f} "
                  f"digit={dig_acc:.4f} [{elapsed:.0f}s]")
            sys.stdout.flush()

            metrics_writer.writerow([
                phase_num, global_step, step, f"{loss.item():.6f}",
                f"{cur_lr:.2e}", f"{seq_acc:.4f}", f"{dig_acc:.4f}",
                f"{elapsed:.1f}",
            ])
            metrics_writer.flush_file()

            if seq_acc > best_acc:
                best_acc = seq_acc
                save_checkpoint(
                    model, f"{ckpt_dir}/phase{phase_num}_best.pt",
                    global_step, seq_acc, n_params, model_config,
                )
                print(f"    ** NEW BEST: {seq_acc:.4f} **")

    # Save end-of-phase checkpoint
    save_checkpoint(
        model, f"{ckpt_dir}/phase{phase_num}_final.pt",
        global_step, best_acc, n_params, model_config,
    )
    return best_acc, global_step


def run_constant_phase(model: Qwen3AdditionModel, device: torch.device,
                       phase_cfg: dict, eval_pairs: list[tuple[int, int]],
                       ckpt_dir: str, phase_num: int, model_config: dict,
                       n_params: int, metrics_writer, t0: float,
                       global_step: int, steps_override: int | None = None,
                       ) -> tuple[float, int]:
    """Run a constant LR training phase.

    Returns (best_accuracy, updated_global_step).
    """
    lr = phase_cfg["lr"]
    batch_size = phase_cfg["batch_size"]
    steps = steps_override if steps_override is not None else phase_cfg["steps"]
    eval_interval = phase_cfg.get("eval_interval", 2000)
    weight_decay = phase_cfg.get("weight_decay", 0.01)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_acc = 0.0

    print(f"\n  Phase {phase_num} [constant]: lr={lr}, batch={batch_size}, steps={steps}")

    for step in range(1, steps + 1):
        model.train()
        full_seq, labels = generate_batch(batch_size, device, max_digits=10)

        optimizer.zero_grad()
        loss = train_step(model, full_seq, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        global_step += 1

        is_eval_step = (step % eval_interval == 0)

        if step % 100 == 0 and not is_eval_step:
            elapsed = time.time() - t0
            print(f"    step {step:6d}/{steps} | loss {loss.item():.4f} | "
                  f"lr {lr:.2e} | {elapsed:.0f}s")
            sys.stdout.flush()
            metrics_writer.writerow([
                phase_num, global_step, step, f"{loss.item():.6f}",
                f"{lr:.2e}", "", "", f"{elapsed:.1f}",
            ])
            if step % 1000 == 0:
                metrics_writer.flush_file()

        if is_eval_step:
            quick_pairs = eval_pairs[:500]
            seq_acc, dig_acc = evaluate(model, device, test_pairs=quick_pairs)
            elapsed = time.time() - t0
            print(f"    step {step:6d}/{steps} | loss {loss.item():.4f} | "
                  f"lr {lr:.2e} | {elapsed:.0f}s")
            print(f"    EVAL step {step}: exact={seq_acc:.4f} "
                  f"digit={dig_acc:.4f} [{elapsed:.0f}s]")
            sys.stdout.flush()

            metrics_writer.writerow([
                phase_num, global_step, step, f"{loss.item():.6f}",
                f"{lr:.2e}", f"{seq_acc:.4f}", f"{dig_acc:.4f}",
                f"{elapsed:.1f}",
            ])
            metrics_writer.flush_file()

            if seq_acc > best_acc:
                best_acc = seq_acc
                save_checkpoint(
                    model, f"{ckpt_dir}/phase{phase_num}_best.pt",
                    global_step, seq_acc, n_params, model_config,
                )
                print(f"    ** NEW BEST: {seq_acc:.4f} **")

    save_checkpoint(
        model, f"{ckpt_dir}/phase{phase_num}_final.pt",
        global_step, best_acc, n_params, model_config,
    )
    return best_acc, global_step


def run_lbfgs_phase(model: Qwen3AdditionModel, device: torch.device,
                    phase_cfg: dict, eval_pairs: list[tuple[int, int]],
                    ckpt_dir: str, phase_num: int, model_config: dict,
                    n_params: int, metrics_writer, t0: float,
                    global_step: int, steps_override: int | None = None,
                    ) -> tuple[float, int]:
    """Run an L-BFGS training phase.

    L-BFGS uses curvature information (Hessian approximation) to escape
    saddle points that AdamW converges to. Uses full-batch training with
    strong Wolfe line search.

    Returns (best_accuracy, updated_global_step).
    """
    lr = phase_cfg.get("lr", 0.5)
    batch_size = phase_cfg.get("batch_size", 512)
    steps = steps_override if steps_override is not None else phase_cfg["steps"]
    max_iter = phase_cfg.get("max_iter", 30)
    history_size = phase_cfg.get("history_size", 10)
    eval_interval = phase_cfg.get("eval_interval", 50)

    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=lr,
        max_iter=max_iter,
        history_size=history_size,
        line_search_fn="strong_wolfe",
    )
    best_acc = 0.0

    print(f"\n  Phase {phase_num} [lbfgs]: lr={lr}, batch={batch_size}, "
          f"steps={steps}, max_iter={max_iter}, history={history_size}")

    for step in range(1, steps + 1):
        model.train()
        full_seq, labels = generate_batch(batch_size, device, max_digits=10)

        def closure():
            optimizer.zero_grad()
            loss = train_step(model, full_seq, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            return loss

        loss = optimizer.step(closure)
        global_step += 1

        is_eval_step = (step % eval_interval == 0)

        if step % 10 == 0 and not is_eval_step:
            elapsed = time.time() - t0
            print(f"    step {step:6d}/{steps} | loss {loss.item():.6f} | "
                  f"lr {lr:.2e} | {elapsed:.0f}s")
            sys.stdout.flush()
            metrics_writer.writerow([
                phase_num, global_step, step, f"{loss.item():.6f}",
                f"{lr:.2e}", "", "", f"{elapsed:.1f}",
            ])
            if step % 50 == 0:
                metrics_writer.flush_file()

        if is_eval_step:
            quick_pairs = eval_pairs[:500]
            seq_acc, dig_acc = evaluate(model, device, test_pairs=quick_pairs)
            elapsed = time.time() - t0
            print(f"    step {step:6d}/{steps} | loss {loss.item():.6f} | "
                  f"lr {lr:.2e} | {elapsed:.0f}s")
            print(f"    EVAL step {step}: exact={seq_acc:.4f} "
                  f"digit={dig_acc:.4f} [{elapsed:.0f}s]")
            sys.stdout.flush()

            metrics_writer.writerow([
                phase_num, global_step, step, f"{loss.item():.6f}",
                f"{lr:.2e}", f"{seq_acc:.4f}", f"{dig_acc:.4f}",
                f"{elapsed:.1f}",
            ])
            metrics_writer.flush_file()

            if seq_acc > best_acc:
                best_acc = seq_acc
                save_checkpoint(
                    model, f"{ckpt_dir}/phase{phase_num}_best.pt",
                    global_step, seq_acc, n_params, model_config,
                )
                print(f"    ** NEW BEST: {seq_acc:.4f} **")

    save_checkpoint(
        model, f"{ckpt_dir}/phase{phase_num}_final.pt",
        global_step, best_acc, n_params, model_config,
    )
    return best_acc, global_step


def find_errors(model: Qwen3AdditionModel, device: torch.device,
                test_pairs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Autoregressive decode on each test pair, return those predicted incorrectly."""
    model.eval()
    errors = []
    with torch.no_grad():
        for a, b in test_pairs:
            exp = expected_output(a, b)
            inp = torch.tensor([encode(a, b)], dtype=torch.long, device=device)
            x = inp
            pred = []
            for _ in range(OUTPUT_LEN):
                logits = model(x)
                next_tok = logits[0, -1, :].argmax().item()
                pred.append(next_tok)
                x = torch.cat([x, torch.tensor([[next_tok]], device=device)], dim=1)
            if pred != exp:
                errors.append((a, b))
    return errors


def build_targeted_batch(error_pairs: list[tuple[int, int]], batch_size: int,
                         device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a training batch mixing error pairs with random pairs.

    If there are more error pairs than batch_size, a random subset is used.
    Otherwise, the batch is filled with randomly generated pairs.
    """
    full_list = []
    label_list = []

    if len(error_pairs) >= batch_size:
        selected = random.sample(error_pairs, batch_size)
    else:
        selected = list(error_pairs)

    for a, b in selected:
        inp = encode(a, b)
        tgt = expected_output(a, b)
        full_list.append(inp + tgt)
        label_list.append([-100] * INPUT_LEN + tgt)

    n_random = batch_size - len(full_list)
    if n_random > 0:
        rand_seq, rand_labels = generate_batch(n_random, device, max_digits=10)
        error_seq = torch.tensor(full_list, dtype=torch.long, device=device)
        error_labels = torch.tensor(label_list, dtype=torch.long, device=device)
        full_seq_t = torch.cat([error_seq, rand_seq], dim=0)
        labels_t = torch.cat([error_labels, rand_labels], dim=0)
    else:
        full_seq_t = torch.tensor(full_list, dtype=torch.long, device=device)
        labels_t = torch.tensor(label_list, dtype=torch.long, device=device)

    perm = torch.randperm(batch_size)
    return full_seq_t[perm], labels_t[perm]


def run_targeted_phase(model: Qwen3AdditionModel, device: torch.device,
                       phase_cfg: dict, eval_pairs: list[tuple[int, int]],
                       ckpt_dir: str, phase_num: int, model_config: dict,
                       n_params: int, metrics_writer, t0: float,
                       global_step: int, steps_override: int | None = None,
                       ) -> tuple[float, int]:
    """Run iterated targeted fine-tuning phase.

    1. Find errors in eval_pairs
    2. Train on error+random mixed batches
    3. Re-evaluate, accumulate errors, repeat
    4. Stop at 0 errors or max iterations

    Returns (best_accuracy, updated_global_step).
    """
    lr = phase_cfg.get("lr", 0.0003)
    batch_size = phase_cfg.get("batch_size", 256)
    weight_decay = phase_cfg.get("weight_decay", 0.01)
    max_iters = phase_cfg.get("max_iters", 10)
    steps_per_iter = phase_cfg.get("steps_per_iter", 5000)
    eval_interval = phase_cfg.get("eval_interval", 1000)

    if steps_override is not None:
        steps_per_iter = steps_override

    print(f"\n  Phase {phase_num} [targeted]: lr={lr}, batch={batch_size}, "
          f"steps/iter={steps_per_iter}, max_iters={max_iters}")

    # Initial error finding on full eval set
    print(f"    Finding initial errors on {len(eval_pairs)} pairs...")
    errors = find_errors(model, device, eval_pairs)
    print(f"    Initial errors: {len(errors)}")

    if len(errors) == 0:
        print("    Model is already perfect on train-eval set!")
        save_checkpoint(
            model, f"{ckpt_dir}/phase{phase_num}_best.pt",
            global_step, 1.0, n_params, model_config,
        )
        return 1.0, global_step

    cumulative_errors = set((a, b) for a, b in errors)
    best_n_errors = len(errors)
    best_acc = 1.0 - best_n_errors / len(eval_pairs)

    for iteration in range(1, max_iters + 1):
        error_list = list(cumulative_errors)
        print(f"\n    --- Targeted iteration {iteration} ---")
        print(f"    Training on {len(error_list)} cumulative error pairs")

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay,
        )

        for step in range(1, steps_per_iter + 1):
            model.train()
            full_seq, labels = build_targeted_batch(error_list, batch_size, device)

            optimizer.zero_grad()
            loss = train_step(model, full_seq, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            global_step += 1

            if step % 500 == 0:
                elapsed = time.time() - t0
                print(f"      [iter {iteration}] step {step:5d}/{steps_per_iter} | "
                      f"loss {loss.item():.6f} | errors={len(error_list)} | "
                      f"{elapsed:.0f}s")
                sys.stdout.flush()

            if step % eval_interval == 0:
                quick_pairs = eval_pairs[:200]
                seq_acc, dig_acc = evaluate(model, device, test_pairs=quick_pairs)
                elapsed = time.time() - t0
                print(f"      [iter {iteration}] EVAL step {step}: "
                      f"exact={seq_acc:.4f} digit={dig_acc:.4f} [{elapsed:.0f}s]")
                sys.stdout.flush()

                metrics_writer.writerow([
                    phase_num, global_step, step, f"{loss.item():.6f}",
                    f"{lr:.2e}", f"{seq_acc:.4f}", f"{dig_acc:.4f}",
                    f"{elapsed:.1f}",
                ])
                metrics_writer.flush_file()

        # Full evaluation on train-eval set after this iteration
        print(f"    Full evaluation on {len(eval_pairs)} pairs...")
        new_errors = find_errors(model, device, eval_pairs)
        n_new_errors = len(new_errors)
        seq_acc = 1.0 - n_new_errors / len(eval_pairs)
        elapsed = time.time() - t0
        print(f"    Iteration {iteration} result: {n_new_errors} errors "
              f"(exact={seq_acc:.6f}) [{elapsed:.0f}s]")

        if n_new_errors < best_n_errors:
            best_n_errors = n_new_errors
            best_acc = seq_acc
            save_checkpoint(
                model, f"{ckpt_dir}/phase{phase_num}_best.pt",
                global_step, seq_acc, n_params, model_config,
                extra={"targeted_iteration": iteration,
                       "n_errors": n_new_errors,
                       "cumulative_pairs": len(cumulative_errors)},
            )
            print(f"    ** NEW BEST: {seq_acc:.6f} ({n_new_errors} errors) **")

        if n_new_errors == 0:
            print(f"    Perfect! 0 errors after iteration {iteration}.")
            break

        # Accumulate new errors for next round
        prev_count = len(cumulative_errors)
        for a, b in new_errors:
            cumulative_errors.add((a, b))
        added = len(cumulative_errors) - prev_count
        print(f"    Cumulative errors: {prev_count} -> {len(cumulative_errors)} "
              f"(+{added} new)")

    save_checkpoint(
        model, f"{ckpt_dir}/phase{phase_num}_final.pt",
        global_step, best_acc, n_params, model_config,
    )
    return best_acc, global_step


# ============================================================================
# Metrics CSV helper
# ============================================================================

class MetricsWriter:
    """Wrapper around csv.writer that tracks the file handle for flushing."""

    def __init__(self, path: str):
        self.file = open(path, "w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow([
            "phase", "global_step", "phase_step", "loss", "lr",
            "exact_acc", "digit_acc", "elapsed",
        ])

    def writerow(self, row):
        self.writer.writerow(row)

    def flush_file(self):
        self.file.flush()

    def close(self):
        self.file.close()


# ============================================================================
# Validation (simulates verify.py + holdout sets)
# ============================================================================

def run_validation(model: Qwen3AdditionModel, device: torch.device,
                   ckpt_dir: str) -> dict:
    """Run final validation matching verify.py protocol.

    - verify-style: seed=2025, 10000 random + 10 edge cases
    - holdout 10K: seed=123 (if file exists)
    - holdout 50K: seed=99 (if file exists)

    Returns validation results dict.
    """
    results = {}

    # 1. Verify-style evaluation (seed=2025, 10000 random + 10 edge cases)
    print(f"\n{'=' * 70}")
    print("Final Validation (simulating verify.py)")
    print(f"{'=' * 70}")

    edge_cases = [
        (0, 0), (0, 1), (9_999_999_999, 0), (9_999_999_999, 1),
        (9_999_999_999, 9_999_999_999), (5_000_000_000, 5_000_000_000),
        (1_111_111_111, 8_888_888_889), (1_234_567_890, 9_876_543_210),
        (9_999_999_999, 9_999_999_999), (1, 9_999_999_999),
    ]
    rng = random.Random(2025)
    random_cases = [
        (rng.randint(0, 9_999_999_999), rng.randint(0, 9_999_999_999))
        for _ in range(10000)
    ]
    verify_pairs = edge_cases + random_cases

    print(f"  verify-style: {len(verify_pairs)} pairs (10 edge + 10000 random, seed=2025)")
    detailed = evaluate_detailed(model, device, verify_pairs)
    results["verify"] = {
        "n_samples": detailed["n_samples"],
        "exact_acc": detailed["exact_acc"],
        "n_errors": detailed["n_errors"],
        "digit_acc": detailed["digit_acc"],
        "qualified": detailed["exact_acc"] >= 0.99,
    }
    qualified_str = "QUALIFIED" if results["verify"]["qualified"] else "NOT QUALIFIED"
    print(f"  Result: {detailed['n_samples'] - detailed['n_errors']}/{detailed['n_samples']} "
          f"correct ({detailed['exact_acc'] * 100:.2f}%)")
    print(f"  Status: {qualified_str}")

    # 2. Holdout 10K (seed=123)
    holdout_10k_path = "data/test_holdout_10k.json"
    if os.path.exists(holdout_10k_path):
        print(f"\n  Holdout 10K ({holdout_10k_path}):")
        holdout_10k = load_test_set(holdout_10k_path)
        h10k = evaluate_detailed(model, device, holdout_10k)
        results["holdout_10k"] = {
            "n_samples": h10k["n_samples"],
            "exact_acc": h10k["exact_acc"],
            "n_errors": h10k["n_errors"],
            "digit_acc": h10k["digit_acc"],
        }
        print(f"  Result: {h10k['n_samples'] - h10k['n_errors']}/{h10k['n_samples']} "
              f"correct ({h10k['exact_acc'] * 100:.2f}%)")
    else:
        print(f"\n  Holdout 10K: {holdout_10k_path} not found, skipping")

    # 3. Holdout 50K (seed=99)
    holdout_50k_path = "data/test_50k_independent.json"
    if os.path.exists(holdout_50k_path):
        print(f"\n  Holdout 50K ({holdout_50k_path}):")
        holdout_50k = load_test_set(holdout_50k_path)
        h50k = evaluate_detailed(model, device, holdout_50k)
        results["holdout_50k"] = {
            "n_samples": h50k["n_samples"],
            "exact_acc": h50k["exact_acc"],
            "n_errors": h50k["n_errors"],
            "digit_acc": h50k["digit_acc"],
        }
        print(f"  Result: {h50k['n_samples'] - h50k['n_errors']}/{h50k['n_samples']} "
              f"correct ({h50k['exact_acc'] * 100:.2f}%)")
    else:
        print(f"\n  Holdout 50K: {holdout_50k_path} not found, skipping")

    # Save validation JSON
    val_path = os.path.join(ckpt_dir, "validation.json")
    with open(val_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Validation saved to {val_path}")

    return results


# ============================================================================
# Phase dispatch and best-checkpoint loading
# ============================================================================

PHASE_RUNNERS = {
    "cosine": run_cosine_phase,
    "constant": run_constant_phase,
    "lbfgs": run_lbfgs_phase,
    "targeted": run_targeted_phase,
}


def load_best_checkpoint(model: Qwen3AdditionModel, ckpt_dir: str,
                         phase_num: int, use_final: bool = False) -> bool:
    """Load the best checkpoint from a completed phase.

    If use_final is True, loads phase{N}_final.pt first (for cases where
    the end-of-phase weights are preferred over the best-eval checkpoint).
    Otherwise tries phase{N}_best.pt first, falls back to phase{N}_final.pt.
    Returns True if a checkpoint was loaded.
    """
    best_path = f"{ckpt_dir}/phase{phase_num}_best.pt"
    final_path = f"{ckpt_dir}/phase{phase_num}_final.pt"

    if use_final:
        paths = [final_path, best_path]
    else:
        paths = [best_path, final_path]

    for path in paths:
        if os.path.exists(path):
            ckpt = torch.load(path, map_location="cpu", weights_only=True)
            model.load_state_dict(ckpt["state_dict"])
            acc = ckpt.get("accuracy", "?")
            label = "final" if "final" in os.path.basename(path) else "best"
            print(f"  Loaded {os.path.basename(path)} ({label}, accuracy={acc})")
            return True
    return False


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Unified reproduction pipeline for all claimed results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python experiments/reproduce.py --list
  python experiments/reproduce.py --config 122p --seed 6 --device cuda
  python experiments/reproduce.py --config 89p --seed 11127 --device cuda
  python experiments/reproduce.py --config 83p --seed 905 --train-eval-seed 888 --device cuda
  python experiments/reproduce.py --config 122p --seed 42 --steps-override 200  # smoke test
""",
    )
    parser.add_argument("--config", type=str,
                        help="Config name (e.g., 89p, 122p). Use --list to see all.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for model initialization and training (default: 0)")
    parser.add_argument("--train-eval-seed", type=int, default=777,
                        help="Seed for generating train-eval pairs used in targeted FT "
                             "(default: 777). Must NOT be 42, 2025, 123, or 99.")
    parser.add_argument("--train-eval-size", type=int, default=50000,
                        help="Number of train-eval pairs for targeted FT (default: 50000)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device (cpu, cuda, cuda:0, etc.)")
    parser.add_argument("--steps-override", type=int, default=None,
                        help="Override step count for ALL phases (useful for smoke tests)")
    parser.add_argument("--no-validate", action="store_true",
                        help="Skip final validation (verify.py + holdout sets)")
    parser.add_argument("--list", action="store_true",
                        help="List all available configs and exit")
    args = parser.parse_args()

    # -- List mode --
    if args.list:
        print("Available reproduction configs:\n")
        print(f"  {'Config':<8s} {'Params':<8s} {'Phases':<8s} Description")
        print(f"  {'-'*6:<8s} {'-'*6:<8s} {'-'*6:<8s} {'-'*50}")
        for name, cfg in CONFIGS.items():
            n_phases = len(cfg["phases"])
            phase_types = " -> ".join(p["type"] for p in cfg["phases"])
            total_steps = sum(
                p.get("steps", 0) or p.get("steps_per_iter", 0) * p.get("max_iters", 1)
                for p in cfg["phases"]
            )
            print(f"  {name:<8s} {cfg['params']:<8d} {n_phases:<8d} {cfg['desc']}")
            print(f"  {'':8s} {'':8s} {'':8s} Pipeline: {phase_types} "
                  f"(~{total_steps:,} steps)")
        print(f"\nReserved seeds (never use as --train-eval-seed): {sorted(RESERVED_SEEDS)}")
        print(f"\nUsage: python experiments/reproduce.py --config <name> --seed <N> "
              f"[--device cuda]")
        return

    # -- Validate arguments --
    if not args.config:
        parser.error("--config is required (use --list to see options)")

    if args.config not in CONFIGS:
        parser.error(f"Unknown config: {args.config}. Use --list to see options.")

    if args.train_eval_seed in RESERVED_SEEDS:
        parser.error(
            f"--train-eval-seed {args.train_eval_seed} is reserved! "
            f"Reserved seeds: {sorted(RESERVED_SEEDS)}. "
            f"Use a different seed (default: 777)."
        )

    cfg = CONFIGS[args.config]
    device = torch.device(args.device)

    # -- Set seeds --
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # -- Create model --
    model = Qwen3AdditionModel(**cfg["model_args"]).to(device)
    n_params = count_params(model)

    if n_params != cfg["params"]:
        print(f"WARNING: Expected {cfg['params']} params, got {n_params}")

    # -- Setup output directory --
    ckpt_dir = f"checkpoints/reproduce_{args.config}_s{args.seed}"
    os.makedirs(ckpt_dir, exist_ok=True)

    # -- Build config dict for checkpoint metadata --
    # Flatten model_args into the format expected by load_model() in targeted_finetune.py
    model_config = {
        "d_model": cfg["model_args"]["d_model"],
        "n_heads": cfg["model_args"]["n_heads"],
        "n_kv_heads": cfg["model_args"]["n_kv_heads"],
        "head_dim": cfg["model_args"]["head_dim"],
        "ff": cfg["model_args"]["ff"],
        "rope_theta": cfg["model_args"]["rope_theta"],
        "no_qk_norm": not cfg["model_args"].get("qk_norm", True),
        "gelu": not cfg["model_args"].get("use_swiglu", True),
        "tie_kv": cfg["model_args"].get("tie_kv", False),
        "tie_qo": cfg["model_args"].get("tie_qo", False),
        "tie_gate": cfg["model_args"].get("tie_gate", False),
        "repeats": cfg["model_args"].get("repeats", 1),
        "share_norms": cfg["model_args"].get("share_norms", False),
        "share_block_norms": cfg["model_args"].get("share_block_norms", False),
    }

    # -- Generate train-eval pairs for targeted FT --
    has_targeted = any(p["type"] == "targeted" for p in cfg["phases"])
    if has_targeted:
        print(f"Generating {args.train_eval_size:,} train-eval pairs "
              f"(seed={args.train_eval_seed})...")
        eval_pairs = generate_test_set(args.train_eval_size, seed=args.train_eval_seed)
    else:
        # For non-targeted configs, generate a smaller eval set for progress tracking
        eval_pairs = generate_test_set(2000, seed=args.train_eval_seed)

    # -- Print banner --
    print(f"\n{'=' * 70}")
    print(f"Reproduction Pipeline: {args.config}")
    print(f"{'=' * 70}")
    print(f"Description: {cfg['desc']}")
    print(f"Parameters:  {n_params} (expected: {cfg['params']})")
    print(f"Seed:        {args.seed}")
    print(f"Device:      {device}")
    print(f"Phases:      {len(cfg['phases'])}")
    for i, p in enumerate(cfg["phases"], 1):
        phase_steps = p.get("steps", None) or f"{p.get('steps_per_iter', '?')}/iter x {p.get('max_iters', '?')}"
        if args.steps_override is not None:
            phase_steps = f"{args.steps_override} (overridden)"
        print(f"  Phase {i}: {p['type']} lr={p.get('lr', '?')} steps={phase_steps}")
    if has_targeted:
        print(f"Train-eval:  {len(eval_pairs):,} pairs (seed={args.train_eval_seed})")
    print(f"Output:      {ckpt_dir}/")
    print(f"{'=' * 70}\n")

    # -- Metrics CSV --
    metrics = MetricsWriter(f"{ckpt_dir}/metrics.csv")

    # -- Run phases --
    t0 = time.time()
    global_step = 0
    phase_results = []

    for phase_num, phase_cfg in enumerate(cfg["phases"], 1):
        phase_type = phase_cfg["type"]
        runner = PHASE_RUNNERS[phase_type]

        # For phases after the first, load checkpoint from previous phase
        if phase_num > 1:
            prev_phase_cfg = cfg["phases"][phase_num - 2]  # 0-indexed
            use_final = prev_phase_cfg.get("use_final", False)
            label = "final" if use_final else "best"
            print(f"\n  Loading {label} checkpoint from phase {phase_num - 1}...")
            loaded = load_best_checkpoint(model, ckpt_dir, phase_num - 1,
                                          use_final=use_final)
            if not loaded:
                print(f"  WARNING: No checkpoint found for phase {phase_num - 1}, "
                      f"continuing with current weights")

        best_acc, global_step = runner(
            model=model,
            device=device,
            phase_cfg=phase_cfg,
            eval_pairs=eval_pairs,
            ckpt_dir=ckpt_dir,
            phase_num=phase_num,
            model_config=model_config,
            n_params=n_params,
            metrics_writer=metrics,
            t0=t0,
            global_step=global_step,
            steps_override=args.steps_override,
        )

        elapsed = time.time() - t0
        phase_results.append({
            "phase": phase_num,
            "type": phase_type,
            "best_acc": best_acc,
            "global_step": global_step,
            "elapsed": elapsed,
        })
        print(f"\n  Phase {phase_num} complete: best_acc={best_acc:.6f}, "
              f"global_step={global_step}, elapsed={elapsed:.0f}s")

    metrics.close()

    # -- Save final best checkpoint --
    # Load the best from the last phase for final checkpoint and validation
    last_phase = len(cfg["phases"])
    load_best_checkpoint(model, ckpt_dir, last_phase)
    save_checkpoint(
        model, f"{ckpt_dir}/final_best.pt",
        global_step, phase_results[-1]["best_acc"], n_params, model_config,
    )

    # -- Final 2K eval for summary --
    print(f"\n{'=' * 70}")
    print("Pipeline complete -- running summary evaluation")
    print(f"{'=' * 70}")
    summary_pairs = generate_test_set(2000, seed=12345)
    seq_acc, dig_acc = evaluate(model, device, test_pairs=summary_pairs)
    elapsed = time.time() - t0
    print(f"  Summary eval (2K random, seed=12345): "
          f"exact={seq_acc:.4f} digit={dig_acc:.4f}")
    print(f"  Total training time: {elapsed:.0f}s ({elapsed/3600:.1f}h)")

    # -- Validation --
    validation = None
    if not args.no_validate:
        validation = run_validation(model, device, ckpt_dir)

    # -- Save run summary --
    summary = {
        "config": args.config,
        "description": cfg["desc"],
        "n_params": n_params,
        "expected_params": cfg["params"],
        "seed": args.seed,
        "train_eval_seed": args.train_eval_seed,
        "device": str(device),
        "steps_override": args.steps_override,
        "phases": phase_results,
        "summary_eval": {"exact_acc": seq_acc, "digit_acc": dig_acc},
        "validation": validation,
        "total_elapsed": time.time() - t0,
        "ckpt_dir": ckpt_dir,
    }
    summary_path = f"{ckpt_dir}/run_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # -- Final output --
    print(f"\n{'=' * 70}")
    print(f"DONE: {args.config} seed={args.seed}")
    print(f"{'=' * 70}")
    print(f"  Checkpoints: {ckpt_dir}/")
    print(f"  Metrics CSV: {ckpt_dir}/metrics.csv")
    print(f"  Run summary: {summary_path}")
    if validation:
        v = validation.get("verify", {})
        n_samples = v.get("n_samples", 0)
        n_errors = v.get("n_errors", 0)
        qualified_str = "QUALIFIED" if v.get("qualified") else "NOT QUALIFIED"
        print(f"  Verify: {n_samples - n_errors}/{n_samples} ({qualified_str})")
    print(f"  Total time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
