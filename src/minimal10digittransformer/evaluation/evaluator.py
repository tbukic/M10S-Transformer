"""Evaluation for addition models.

Provides exact-match accuracy on addition problems with configurable test sets.
"""

from __future__ import annotations

import random

import torch

from minimal10digittransformer.data.tokenizer import AdditionTokenizer


def evaluate_model(
    model: torch.nn.Module,
    n_samples: int = 10000,
    max_digits: int = 10,
    seed: int = 42,
    device: str = "cuda",
    batch_size: int = 512,
    format: str = "plain",
    verbose: bool = False,
) -> dict:
    """Evaluate model on exact-match addition accuracy.

    Args:
        model: The model to evaluate
        n_samples: Number of test samples
        max_digits: Maximum digits per operand
        seed: Random seed for reproducibility
        device: Device to run on
        batch_size: Batch size for evaluation
        format: Tokenizer format
        verbose: Print wrong examples

    Returns:
        Dictionary with accuracy metrics
    """
    model.eval()
    tokenizer = AdditionTokenizer(max_digits=max_digits, format=format)
    rng = random.Random(seed)

    correct = 0
    total = 0
    digit_correct = 0
    digit_total = 0
    wrong_examples = []

    # Generate test pairs
    pairs = []
    for _ in range(n_samples):
        a = rng.randint(0, 10**max_digits - 1)
        b = rng.randint(0, 10**max_digits - 1)
        pairs.append((a, b))

    with torch.no_grad():
        for batch_start in range(0, len(pairs), batch_size):
            batch_pairs = pairs[batch_start : batch_start + batch_size]
            batch_inputs = []
            batch_targets = []

            for a, b in batch_pairs:
                input_ids, target_ids = tokenizer.encode_addition(a, b)
                full_seq = input_ids + target_ids
                batch_inputs.append(full_seq)
                batch_targets.append(target_ids)

            # Pad and create tensor
            input_tensor = torch.tensor(batch_inputs, dtype=torch.long, device=device)

            # Forward pass
            logits = model(input_tensor)

            # Get predictions for output positions only
            input_len = tokenizer.input_length
            pred_logits = logits[:, input_len - 1:-1, :]
            predictions = pred_logits.argmax(dim=-1)

            # Check accuracy
            for i, (a, b) in enumerate(batch_pairs):
                expected = torch.tensor(batch_targets[i], dtype=torch.long, device=device)
                pred = predictions[i][: len(expected)]

                # Exact match
                if torch.equal(pred, expected):
                    correct += 1
                elif verbose and len(wrong_examples) < 20:
                    pred_str = tokenizer.decode(pred.tolist())
                    exp_str = tokenizer.decode(expected.tolist())
                    wrong_examples.append(f"{a}+{b}: pred={pred_str}, expected={exp_str}")

                # Per-digit accuracy
                matches = (pred == expected).sum().item()
                digit_correct += matches
                digit_total += len(expected)
                total += 1

    results = {
        "exact_match_accuracy": correct / total if total > 0 else 0.0,
        "digit_accuracy": digit_correct / digit_total if digit_total > 0 else 0.0,
        "correct": correct,
        "total": total,
        "n_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }

    if verbose and wrong_examples:
        results["wrong_examples"] = wrong_examples

    return results


def evaluate_by_difficulty(
    model: torch.nn.Module,
    max_digits: int = 10,
    n_per_level: int = 1000,
    seed: int = 42,
    device: str = "cuda",
    format: str = "plain",
) -> dict:
    """Evaluate accuracy broken down by number of digits."""
    results = {}
    for n_digits in range(1, max_digits + 1):
        tokenizer = AdditionTokenizer(max_digits=max_digits, format=format)
        rng = random.Random(seed + n_digits)

        correct = 0
        total = 0

        pairs = []
        for _ in range(n_per_level):
            a = rng.randint(10 ** (n_digits - 1), 10**n_digits - 1)
            b = rng.randint(10 ** (n_digits - 1), 10**n_digits - 1)
            pairs.append((a, b))

        with torch.no_grad():
            batch_inputs = []
            batch_targets = []
            for a, b in pairs:
                input_ids, target_ids = tokenizer.encode_addition(a, b)
                full_seq = input_ids + target_ids
                batch_inputs.append(full_seq)
                batch_targets.append(target_ids)

            input_tensor = torch.tensor(batch_inputs, dtype=torch.long, device=device)
            logits = model(input_tensor)

            input_len = tokenizer.input_length
            pred_logits = logits[:, input_len - 1:-1, :]
            predictions = pred_logits.argmax(dim=-1)

            for i, (a, b) in enumerate(pairs):
                expected = torch.tensor(batch_targets[i], dtype=torch.long, device=device)
                pred = predictions[i][: len(expected)]
                if torch.equal(pred, expected):
                    correct += 1
                total += 1

        results[n_digits] = correct / total if total > 0 else 0.0

    return results
