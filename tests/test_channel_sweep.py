"""Tests for channel_sweep ctrl manipulation helpers."""
from __future__ import annotations

import pytest
import torch


def test_build_scaled_ctrl_scales_target_channel():
    from tools.stage3.channel_sweep import build_scaled_ctrl

    ctrl = torch.ones(10, 4, 4)
    result = build_scaled_ctrl(ctrl, channel_idx=2, scale=0.5)

    assert torch.allclose(result[2], ctrl[2] * 0.5)
    assert torch.allclose(result[0], ctrl[0])
    assert torch.allclose(result[9], ctrl[9])
    assert torch.allclose(ctrl[2], torch.ones(4, 4))


def test_build_2d_scaled_ctrl():
    from tools.stage3.channel_sweep import build_2d_scaled_ctrl

    ctrl = torch.ones(10, 4, 4)
    result = build_2d_scaled_ctrl(
        ctrl,
        idx_o2=8,
        idx_glucose=9,
        o2_scale=0.0,
        glucose_scale=0.5,
    )

    assert torch.allclose(result[8], torch.zeros(4, 4))
    assert torch.allclose(result[9], ctrl[9] * 0.5)
    assert torch.allclose(result[0], ctrl[0])


def test_build_relabeled_ctrl_copies_and_zeros():
    from tools.stage3.channel_sweep import build_relabeled_ctrl

    ctrl = torch.zeros(10, 4, 4)
    ctrl[2] = 0.8

    result = build_relabeled_ctrl(ctrl, idx_source=2, idx_target=3)

    assert torch.allclose(result[2], torch.zeros(4, 4))
    assert torch.allclose(result[3], ctrl[2])
    assert float(ctrl[2].mean()) == pytest.approx(0.8)


def test_build_relabeled_ctrl_does_not_affect_other_channels():
    from tools.stage3.channel_sweep import build_relabeled_ctrl

    ctrl = torch.rand(10, 4, 4)
    ctrl[2] = 1.0
    result = build_relabeled_ctrl(ctrl, idx_source=2, idx_target=3)

    for ch in range(10):
        if ch in (2, 3):
            continue
        assert torch.allclose(result[ch], ctrl[ch]), f"channel {ch} should be unchanged"


def test_sweep_scales_list():
    from tools.stage3.channel_sweep import SWEEP_SCALES

    assert SWEEP_SCALES == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert len(SWEEP_SCALES) == 5
