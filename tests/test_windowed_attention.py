"""Tests for windowed (sliding window) attention mask.

Verifies that create_causal_mask and window_size parameter work correctly
in both Qwen3AdditionModel and CircularArcQwen3.
"""

import sys
import os
import math

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from minimal10digittransformer.model.qwen3 import (
    Qwen3AdditionModel,
    create_causal_mask,
    TOTAL_LEN,
)
from minimal10digittransformer.model.circular_arc import CircularArcQwen3


# ── Test create_causal_mask directly ─────────────────────────────────────────

class TestCreateCausalMask:
    def test_window0_is_full_causal(self):
        """window_size=0 produces standard upper-triangular causal mask."""
        n = 8
        mask = create_causal_mask(n, window_size=0)
        expected = torch.full((n, n), float("-inf"))
        expected = torch.triu(expected, diagonal=1)
        assert torch.equal(mask, expected)

    def test_window_covers_full_seq(self):
        """window_size >= seq_len is equivalent to full causal."""
        n = 8
        full = create_causal_mask(n, window_size=0)
        wide = create_causal_mask(n, window_size=n)
        wider = create_causal_mask(n, window_size=n + 10)
        assert torch.equal(full, wide)
        assert torch.equal(full, wider)

    def test_window1_diagonal(self):
        """window_size=1: each position only sees itself."""
        n = 6
        mask = create_causal_mask(n, window_size=1)
        for i in range(n):
            for j in range(n):
                if i == j:
                    assert mask[i, j] == 0.0, f"mask[{i},{j}] should be 0"
                else:
                    assert mask[i, j] == float("-inf"), f"mask[{i},{j}] should be -inf"

    def test_window2_pattern(self):
        """window_size=2: position i sees i-1 and i."""
        n = 6
        mask = create_causal_mask(n, window_size=2)
        for i in range(n):
            start = max(0, i - 1)
            for j in range(n):
                if start <= j <= i:
                    assert mask[i, j] == 0.0, f"mask[{i},{j}] should be 0"
                else:
                    assert mask[i, j] == float("-inf"), f"mask[{i},{j}] should be -inf"

    def test_window3_specific_values(self):
        """window_size=3: position 5 sees [3,4,5], not [0,1,2]."""
        mask = create_causal_mask(8, window_size=3)
        # Position 5 should see positions 3, 4, 5
        assert mask[5, 3] == 0.0
        assert mask[5, 4] == 0.0
        assert mask[5, 5] == 0.0
        # Position 5 should NOT see 0, 1, 2, 6, 7
        assert mask[5, 0] == float("-inf")
        assert mask[5, 1] == float("-inf")
        assert mask[5, 2] == float("-inf")
        assert mask[5, 6] == float("-inf")
        assert mask[5, 7] == float("-inf")

    def test_window_at_start(self):
        """Early positions with window_size > position should still work."""
        mask = create_causal_mask(6, window_size=4)
        # Position 0: can only see itself (no earlier positions)
        assert mask[0, 0] == 0.0
        for j in range(1, 6):
            assert mask[0, j] == float("-inf")
        # Position 2: sees 0, 1, 2 (only 3 despite window=4)
        assert mask[2, 0] == 0.0
        assert mask[2, 1] == 0.0
        assert mask[2, 2] == 0.0


# ── Test model integration ──────────────────────────────────────────────────

class TestQwen3WindowedAttention:
    def _make_model(self, window_size=0, **kwargs):
        defaults = dict(d_model=3, n_heads=1, n_kv_heads=1, head_dim=4,
                        ff=3, rope_theta=3.0)
        defaults.update(kwargs)
        return Qwen3AdditionModel(window_size=window_size, **defaults)

    def test_window0_matches_default(self):
        """window_size=0 model produces identical output to default model."""
        torch.manual_seed(42)
        m0 = self._make_model(window_size=0)
        torch.manual_seed(42)
        m_default = Qwen3AdditionModel(d_model=3, n_heads=1, n_kv_heads=1,
                                        head_dim=4, ff=3, rope_theta=3.0)

        x = torch.randint(0, 10, (2, 35))
        with torch.no_grad():
            out0 = m0(x)
            out_d = m_default(x)
        assert torch.allclose(out0, out_d, atol=1e-6), \
            f"Max diff: {(out0 - out_d).abs().max()}"

    def test_windowed_model_runs(self):
        """Windowed model runs forward pass without error."""
        m = self._make_model(window_size=4)
        x = torch.randint(0, 10, (2, 35))
        out = m(x)
        assert out.shape == (2, 35, 10)
        assert torch.isfinite(out).all()

    def test_param_count_unchanged(self):
        """window_size doesn't add parameters."""
        m0 = self._make_model(window_size=0)
        m4 = self._make_model(window_size=4)
        p0 = sum(p.numel() for p in m0.parameters())
        p4 = sum(p.numel() for p in m4.parameters())
        assert p0 == p4

    def test_gradient_flow(self):
        """Gradients flow through windowed model (no NaN/zero grads)."""
        m = self._make_model(window_size=3)
        x = torch.randint(0, 10, (4, 35))
        out = m(x)
        loss = out.sum()
        loss.backward()
        for name, p in m.named_parameters():
            assert p.grad is not None, f"No grad for {name}"
            assert torch.isfinite(p.grad).all(), f"Non-finite grad for {name}"
            assert p.grad.abs().sum() > 0, f"Zero grad for {name}"

    def test_repeats_with_window(self):
        """Model with repeats=5, window_size=2 runs without error."""
        m = self._make_model(window_size=2, repeats=5)
        x = torch.randint(0, 10, (2, 35))
        out = m(x)
        assert out.shape == (2, 35, 10)
        assert torch.isfinite(out).all()

    def test_different_windows_different_outputs(self):
        """Different window sizes produce different outputs (not degenerate)."""
        torch.manual_seed(42)
        m_full = self._make_model(window_size=0)
        # Copy weights to windowed model
        m_win = self._make_model(window_size=2)
        m_win.load_state_dict(m_full.state_dict(), strict=False)

        x = torch.randint(0, 10, (2, 35))
        with torch.no_grad():
            out_full = m_full(x)
            out_win = m_win(x)
        # Position 0 should be the same (only sees itself either way with causal)
        # But later positions should differ
        assert not torch.allclose(out_full[:, 10:, :], out_win[:, 10:, :], atol=1e-5), \
            "Windowed and full models should produce different outputs"


class TestCircularArcWindowedAttention:
    def _make_model(self, window_size=0, **kwargs):
        defaults = dict(d_model=3, n_heads=1, n_kv_heads=1, head_dim=4,
                        ff=3, rope_theta=3.0)
        defaults.update(kwargs)
        return CircularArcQwen3(window_size=window_size, **defaults)

    def test_window0_matches_default(self):
        """window_size=0 arc model matches default arc model."""
        torch.manual_seed(42)
        m0 = self._make_model(window_size=0)
        torch.manual_seed(42)
        m_default = CircularArcQwen3(d_model=3, n_heads=1, n_kv_heads=1,
                                      head_dim=4, ff=3, rope_theta=3.0)

        x = torch.randint(0, 10, (2, 35))
        with torch.no_grad():
            out0 = m0(x)
            out_d = m_default(x)
        assert torch.allclose(out0, out_d, atol=1e-6)

    def test_windowed_arc_runs(self):
        """Windowed arc model runs forward pass."""
        m = self._make_model(window_size=4)
        x = torch.randint(0, 10, (2, 35))
        out = m(x)
        assert out.shape == (2, 35, 10)
        assert torch.isfinite(out).all()

    def test_repeats_with_window(self):
        """Arc model with repeats=5, window_size=2 runs without error."""
        m = self._make_model(window_size=2, repeats=5)
        x = torch.randint(0, 10, (2, 35))
        out = m(x)
        assert out.shape == (2, 35, 10)
        assert torch.isfinite(out).all()

    def test_param_count_unchanged(self):
        """window_size doesn't add parameters to arc model."""
        m0 = self._make_model(window_size=0)
        m4 = self._make_model(window_size=4)
        p0 = sum(p.numel() for p in m0.parameters())
        p4 = sum(p.numel() for p in m4.parameters())
        assert p0 == p4

    def test_gradient_flow(self):
        """Gradients flow through windowed arc model."""
        m = self._make_model(window_size=3)
        x = torch.randint(0, 10, (4, 35))
        out = m(x)
        loss = out.sum()
        loss.backward()
        for name, p in m.named_parameters():
            assert p.grad is not None, f"No grad for {name}"
            assert torch.isfinite(p.grad).all(), f"Non-finite grad for {name}"
