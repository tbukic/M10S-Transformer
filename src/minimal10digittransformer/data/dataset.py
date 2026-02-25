"""Dataset for 10-digit addition training."""

from __future__ import annotations

import random

import torch
from torch.utils.data import Dataset

from minimal10digittransformer.data.tokenizer import AdditionTokenizer


def generate_addition_pair(
    max_digits: int = 10,
    min_digits: int = 1,
    rng: random.Random | None = None,
) -> tuple[int, int]:
    """Generate a random addition pair with specified digit range."""
    if rng is None:
        rng = random.Random()
    num_digits_a = rng.randint(min_digits, max_digits)
    num_digits_b = rng.randint(min_digits, max_digits)
    a = rng.randint(10 ** (num_digits_a - 1), 10**num_digits_a - 1)
    b = rng.randint(10 ** (num_digits_b - 1), 10**num_digits_b - 1)
    return a, b


class AdditionDataset(Dataset):
    """On-the-fly addition dataset - generates samples dynamically.

    This avoids data leakage since each sample is generated fresh.
    For evaluation, use a fixed seed to get deterministic test sets.
    """

    def __init__(
        self,
        size: int,
        max_digits: int = 10,
        min_digits: int = 1,
        seed: int | None = None,
        format: str = "plain",
    ):
        self.size = size
        self.max_digits = max_digits
        self.min_digits = min_digits
        self.seed = seed
        self.tokenizer = AdditionTokenizer(max_digits=max_digits, format=format)

        # For fixed (eval) datasets, pre-generate all pairs
        if seed is not None:
            rng = random.Random(seed)
            self.pairs = [
                generate_addition_pair(max_digits, min_digits, rng) for _ in range(size)
            ]
        else:
            self.pairs = None

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if self.pairs is not None:
            a, b = self.pairs[idx]
        else:
            a, b = generate_addition_pair(self.max_digits, self.min_digits)

        full_seq = self.tokenizer.encode_full_sequence(a, b)

        input_len = self.tokenizer.input_length
        total_len = self.tokenizer.total_length

        # For causal LM: input is the full sequence, target is shifted
        tokens = torch.tensor(full_seq, dtype=torch.long)

        # Create target: -100 for input positions (don't compute loss), actual tokens for output
        target = torch.full((total_len,), -100, dtype=torch.long)
        target[input_len:] = tokens[input_len:]

        return {
            "input_ids": tokens,
            "labels": target,
            "a": a,
            "b": b,
        }
