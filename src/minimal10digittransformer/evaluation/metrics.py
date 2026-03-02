"""Evaluation metrics for 10-digit addition models."""

import random

import torch

from minimal10digittransformer.model.qwen3 import (
    MAX_ADDEND, OUTPUT_LEN, Qwen3AdditionModel,
)
from minimal10digittransformer.data.addition import encode, expected_output


def evaluate(model: Qwen3AdditionModel, device: torch.device,
             n_samples: int = 10000, seed: int | None = None,
             test_pairs: list[tuple[int, int]] | None = None) -> tuple[float, float]:
    """Evaluate with autoregressive generation.

    Args:
        model: The model to evaluate.
        device: Device to run on.
        n_samples: Number of samples (ignored if test_pairs provided).
        seed: Random seed for generating pairs (ignored if test_pairs provided).
        test_pairs: Pre-generated list of (a, b) pairs. If None, generates randomly.

    Returns:
        (exact_match_accuracy, digit_accuracy)
    """
    model.eval()
    correct_seq = 0
    correct_dig = 0
    total_dig = 0

    if test_pairs is None:
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random
        test_pairs = [(rng.randint(0, MAX_ADDEND), rng.randint(0, MAX_ADDEND))
                      for _ in range(n_samples)]

    with torch.no_grad():
        for a, b in test_pairs:
            exp = expected_output(a, b)
            inp = torch.tensor([encode(a, b)], dtype=torch.long, device=device)

            # Autoregressive generation
            x = inp
            pred = []
            for _ in range(OUTPUT_LEN):
                logits = model(x)
                next_tok = logits[0, -1, :].argmax().item()
                pred.append(next_tok)
                x = torch.cat([x, torch.tensor([[next_tok]], device=device)], dim=1)

            matches = sum(p == e for p, e in zip(pred, exp))
            correct_dig += matches
            total_dig += OUTPUT_LEN
            correct_seq += (matches == OUTPUT_LEN)

    return correct_seq / len(test_pairs), correct_dig / total_dig


def evaluate_detailed(model: Qwen3AdditionModel, device: torch.device,
                      test_pairs: list[tuple[int, int]]) -> dict:
    """Detailed evaluation with per-digit-position accuracy and carry analysis.

    Returns dict with:
        exact_acc: overall exact match accuracy
        digit_acc: overall digit accuracy
        per_position: list of 11 per-position accuracies (LSB to MSB)
        carry_acc: dict mapping n_carries -> (exact_acc, count)
        errors: list of (a, b, predicted, expected) for incorrect samples
    """
    model.eval()
    correct_seq = 0
    per_pos_correct = [0] * OUTPUT_LEN
    carry_buckets: dict[int, list[bool]] = {}
    errors = []

    with torch.no_grad():
        for a, b in test_pairs:
            exp = expected_output(a, b)
            inp = torch.tensor([encode(a, b)], dtype=torch.long, device=device)

            # Autoregressive generation
            x = inp
            pred = []
            for _ in range(OUTPUT_LEN):
                logits = model(x)
                next_tok = logits[0, -1, :].argmax().item()
                pred.append(next_tok)
                x = torch.cat([x, torch.tensor([[next_tok]], device=device)], dim=1)

            # Per-position accuracy
            is_correct = True
            for i in range(OUTPUT_LEN):
                if pred[i] == exp[i]:
                    per_pos_correct[i] += 1
                else:
                    is_correct = False

            if is_correct:
                correct_seq += 1
            else:
                errors.append((a, b, pred, exp))

            # Count carries
            n_carries = _count_carries(a, b)
            carry_buckets.setdefault(n_carries, []).append(is_correct)

    n = len(test_pairs)
    carry_acc = {
        k: (sum(v) / len(v), len(v))
        for k, v in sorted(carry_buckets.items())
    }

    return {
        "exact_acc": correct_seq / n,
        "digit_acc": sum(per_pos_correct) / (n * OUTPUT_LEN),
        "per_position": [c / n for c in per_pos_correct],
        "carry_acc": carry_acc,
        "errors": errors[:100],  # cap at 100 for memory
        "n_samples": n,
        "n_errors": n - correct_seq,
    }


def _count_carries(a: int, b: int) -> int:
    """Count number of carry operations in a + b."""
    carry = 0
    n_carries = 0
    for _ in range(11):
        d_a = a % 10
        d_b = b % 10
        total = d_a + d_b + carry
        if total >= 10:
            n_carries += 1
            carry = 1
        else:
            carry = 0
        a //= 10
        b //= 10
    return n_carries
