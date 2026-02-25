"""Tokenizer for addition sequences.

Supports multiple input formats:
- 'plain': "1234567890+9876543210=11111111100"
- 'reversed': digits reversed (LSB first) for both input and output
- 'plain_reversed_output': normal input, reversed output
- 'reversed_all': everything reversed (LSB first), format used by top challenge solutions
"""

from __future__ import annotations


class AdditionTokenizer:
    """Minimal tokenizer for digit addition sequences.

    The challenge solutions use reversed digit order with minimal vocab (0-9 only).
    We support both the academic format and the challenge format.
    """

    DIGIT_TOKENS = {str(i): i for i in range(10)}
    SPECIAL_TOKENS = {"+": 10, "=": 11, "<pad>": 12, "<bos>": 13, "<eos>": 14}

    def __init__(self, max_digits: int = 10, format: str = "plain", vocab_mode: str = "full"):
        """
        Args:
            max_digits: Maximum digits per operand
            format: 'plain', 'reversed', 'plain_reversed_output', 'reversed_all'
            vocab_mode: 'full' (0-9 + special), 'digits_only' (0-9 only, uses 0 as padding)
        """
        self.max_digits = max_digits
        self.format = format
        self.vocab_mode = vocab_mode

        if vocab_mode == "digits_only":
            self.token_to_id = {str(i): i for i in range(10)}
            self.vocab_size = 10
            self.pad_id = 0
        else:
            self.token_to_id = {**self.DIGIT_TOKENS, **self.SPECIAL_TOKENS}
            self.vocab_size = len(self.token_to_id)
            self.pad_id = self.token_to_id["<pad>"]

        self.id_to_token = {v: k for k, v in self.token_to_id.items()}

    def encode_addition(self, a: int, b: int) -> tuple[list[int], list[int]]:
        """Encode an addition problem as input and target token sequences."""
        result = a + b
        a_str = str(a).zfill(self.max_digits)
        b_str = str(b).zfill(self.max_digits)
        r_str = str(result).zfill(self.max_digits + 1)

        if self.format == "reversed" or self.format == "reversed_all":
            a_str = a_str[::-1]
            b_str = b_str[::-1]
            r_str = r_str[::-1]
        elif self.format == "plain_reversed_output":
            r_str = r_str[::-1]

        if self.vocab_mode == "digits_only":
            # Challenge format: [0] + reversed_a + [0, 0] + reversed_b + [0]
            # Output: 11 result digits
            input_tokens = [0]  # padding/separator
            input_tokens.extend(int(c) for c in a_str)
            input_tokens.extend([0, 0])  # separator
            input_tokens.extend(int(c) for c in b_str)
            input_tokens.append(0)  # separator
            target_tokens = [int(c) for c in r_str]
        else:
            # Standard format with special tokens
            input_tokens = [self.token_to_id[c] for c in a_str]
            input_tokens.append(self.token_to_id["+"])
            input_tokens.extend(self.token_to_id[c] for c in b_str)
            input_tokens.append(self.token_to_id["="])
            target_tokens = [self.token_to_id[c] for c in r_str]

        return input_tokens, target_tokens

    def encode_full_sequence(self, a: int, b: int) -> list[int]:
        """Encode the full sequence including input and output for causal training."""
        input_ids, target_ids = self.encode_addition(a, b)
        return input_ids + target_ids

    def decode(self, token_ids: list[int]) -> str:
        """Decode token IDs back to string."""
        return "".join(self.id_to_token.get(t, "?") for t in token_ids)

    @property
    def input_length(self) -> int:
        """Length of the input portion."""
        if self.vocab_mode == "digits_only":
            # [0] + digits + [0, 0] + digits + [0]
            return 1 + self.max_digits + 2 + self.max_digits + 1
        return self.max_digits * 2 + 2  # digits + '+' + digits + '='

    @property
    def output_length(self) -> int:
        """Length of the output portion."""
        return self.max_digits + 1

    @property
    def total_length(self) -> int:
        """Total sequence length."""
        return self.input_length + self.output_length
