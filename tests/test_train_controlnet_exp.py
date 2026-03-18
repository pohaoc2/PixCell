# tests/test_train_controlnet_exp.py
"""
Unit tests for CFG dropout and channel reliability weighting logic.
Tests pure tensor operations, independent of the full training loop.
"""
import torch
import pytest


# ── CFG dropout ───────────────────────────────────────────────────────────────

def _apply_cfg_dropout(y: torch.Tensor, prob: float, rng: torch.Generator) -> torch.Tensor:
    if torch.rand(1, generator=rng).item() < prob:
        return torch.zeros_like(y)
    return y


def test_cfg_dropout_never_at_zero_prob():
    y   = torch.ones(1, 1, 1, 1536)
    rng = torch.Generator().manual_seed(0)
    for _ in range(100):
        out = _apply_cfg_dropout(y, prob=0.0, rng=rng)
        assert not torch.all(out == 0), "dropout should never fire at prob=0"


def test_cfg_dropout_always_at_one_prob():
    y   = torch.ones(1, 1, 1, 1536)
    rng = torch.Generator().manual_seed(0)
    for _ in range(20):
        out = _apply_cfg_dropout(y, prob=1.0, rng=rng)
        assert torch.all(out == 0), "dropout should always fire at prob=1"


def test_cfg_dropout_approximate_rate():
    """With prob=0.15 over 1000 trials, rate should be ~15% ± 3%."""
    y     = torch.ones(1, 1, 1, 1536)
    rng   = torch.Generator().manual_seed(42)
    drops = sum(
        1 for _ in range(1000)
        if torch.all(_apply_cfg_dropout(y, prob=0.15, rng=rng) == 0)
    )
    assert 120 <= drops <= 180, f"Expected ~150 drops, got {drops}"


def test_cfg_dropout_output_is_zeros_not_noise():
    y   = torch.ones(1, 1, 1, 1536) * 99.0
    rng = torch.Generator().manual_seed(0)
    out = _apply_cfg_dropout(y, prob=1.0, rng=rng)
    assert out.shape == y.shape
    assert torch.all(out == 0.0)


# ── Channel reliability weighting ────────────────────────────────────────────

def _apply_channel_weights(
    tme_channels: torch.Tensor,   # [B, C, H, W]
    weights: list[float],
) -> torch.Tensor:
    w = torch.tensor(weights, dtype=tme_channels.dtype, device=tme_channels.device)
    return tme_channels * w.view(1, -1, 1, 1)


def test_channel_weights_shape_preserved():
    x = torch.ones(2, 9, 256, 256)
    w = [1.0] * 6 + [0.5] * 3
    out = _apply_channel_weights(x, w)
    assert out.shape == x.shape


def test_channel_weights_values():
    x = torch.ones(1, 9, 4, 4)
    w = [1.0] * 6 + [0.5] * 3
    out = _apply_channel_weights(x, w)
    assert torch.all(out[:, :6] == 1.0), "reliable channels unchanged"
    assert torch.allclose(out[:, 6:], torch.full_like(out[:, 6:], 0.5)), \
        "approximate channels halved"


def test_channel_weights_count_must_match_channels():
    x = torch.ones(1, 9, 4, 4)
    with pytest.raises(Exception):
        _apply_channel_weights(x, [1.0] * 8)   # wrong count → broadcast error


def test_cfg_dropout_batch_independence():
    """Each sample in a batch must be dropped independently."""
    B = 32
    y   = torch.ones(B, 1, 1, 1536)
    rng = torch.Generator().manual_seed(7)
    dropped = [
        torch.all(
            _apply_cfg_dropout(y[b : b + 1], prob=0.5, rng=rng) == 0
        ).item()
        for b in range(B)
    ]
    # With prob=0.5 over 32 samples, all dropped or all kept is astronomically unlikely
    assert any(dropped), "at least one sample should be dropped"
    assert not all(dropped), "at least one sample should survive"
