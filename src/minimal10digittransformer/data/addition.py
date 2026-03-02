"""Data generation for 10-digit addition task.

Encoding: LSB-first (reversed digit order).
Format: [0] rev(a, 10 digits) [0,0] rev(b, 10 digits) [0] → 11 reversed sum digits
"""

import json
import random
from pathlib import Path

import torch

from minimal10digittransformer.model.qwen3 import (
    NUM_DIGITS, SUM_DIGITS, MAX_ADDEND, VOCAB_SIZE, INPUT_LEN, OUTPUT_LEN,
)


def encode(a: int, b: int) -> list[int]:
    """Encode a, b into LSB-first format: [0] rev(a) [0,0] rev(b) [0]"""
    pa = f"{a:010d}"
    pb = f"{b:010d}"
    return (
        [0]
        + [int(c) for c in reversed(pa)]
        + [0, 0]
        + [int(c) for c in reversed(pb)]
        + [0]
    )


def expected_output(a: int, b: int) -> list[int]:
    """LSB-first reversed sum digits."""
    s = str(a + b)[::-1].ljust(SUM_DIGITS, "0")
    return [int(c) for c in s]


def generate_batch(batch_size: int, device: torch.device, max_digits: int = 10):
    """Generate batch with full sequence for single-pass teacher forcing.
    Returns (full_seq, labels) where:
      full_seq: [B, 35] = prompt(24) + target(11)
      labels: [B, 35] with -100 for prompt positions
    """
    full_list, label_list = [], []
    for _ in range(batch_size):
        if max_digits < 10:
            n_d = random.randint(1, max_digits)
            a = random.randint(0, 10**n_d - 1)
            b = random.randint(0, 10**n_d - 1)
        else:
            a = random.randint(0, MAX_ADDEND)
            b = random.randint(0, MAX_ADDEND)
        inp = encode(a, b)  # 24 tokens
        tgt = expected_output(a, b)  # 11 tokens
        full_seq = inp + tgt  # 35 tokens
        # Labels: logits[t] predicts token[t+1] after shift.
        # We want loss on positions 23-33 (predicting tokens 24-34 = tgt[0:11]).
        labels = [-100] * INPUT_LEN + tgt  # 35 tokens
        full_list.append(full_seq)
        label_list.append(labels)

    return (
        torch.tensor(full_list, dtype=torch.long, device=device),
        torch.tensor(label_list, dtype=torch.long, device=device),
    )


def generate_test_set(n_samples: int, seed: int = 42) -> list[tuple[int, int]]:
    """Generate a fixed, deterministic test set of (a, b) pairs.

    Uses its own RNG to avoid polluting global state.
    """
    rng = random.Random(seed)
    return [(rng.randint(0, MAX_ADDEND), rng.randint(0, MAX_ADDEND))
            for _ in range(n_samples)]


def save_test_set(pairs: list[tuple[int, int]], path: str | Path):
    """Save test set to JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(pairs, f)


def load_test_set(path: str | Path) -> list[tuple[int, int]]:
    """Load test set from JSON file."""
    with open(path) as f:
        return [tuple(pair) for pair in json.load(f)]
