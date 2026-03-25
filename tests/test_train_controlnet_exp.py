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


# ── Split-optimizer (TME proj LR fix) ────────────────────────────────────────

def _make_tme_module():
    from diffusion.model.nets.multi_group_tme import MultiGroupTMEModule
    channel_groups = [
        dict(name="cell_identity", n_channels=3),
        dict(name="cell_state",    n_channels=3),
        dict(name="vasculature",   n_channels=1),
        dict(name="microenv",      n_channels=2),
    ]
    return MultiGroupTMEModule(channel_groups=channel_groups)


def test_proj_param_filter_captures_all_proj_layers():
    """'cross_attn.proj' filter must capture exactly weight+bias for all 4 groups."""
    tme = _make_tme_module()
    proj_names  = [n for n, _ in tme.named_parameters() if "cross_attn.proj" in n]
    other_names = [n for n, _ in tme.named_parameters() if "cross_attn.proj" not in n]

    # 4 groups × (proj.weight + proj.bias) = 8
    assert len(proj_names) == 8, f"Expected 8 proj params, got {len(proj_names)}: {proj_names}"
    # filter is exhaustive — no param lost or double-counted
    total = sum(1 for _ in tme.named_parameters())
    assert len(proj_names) + len(other_names) == total


def test_split_tme_optimizer_has_correct_lrs():
    """Two-group AdamW must assign tme_proj_lr to proj params and tme_lr to the rest."""
    tme = _make_tme_module()
    proj_params  = [p for n, p in tme.named_parameters() if "cross_attn.proj" in n]
    other_params = [p for n, p in tme.named_parameters() if "cross_attn.proj" not in n]

    opt = torch.optim.AdamW(
        [{"params": proj_params,  "lr": 3e-4},
         {"params": other_params, "lr": 1e-5}],
        weight_decay=0.0,
    )

    total = sum(1 for _ in tme.parameters())
    assert len(proj_params) + len(other_params) == total, "split must cover all params"
    assert len(opt.param_groups) == 2
    assert opt.param_groups[0]["lr"] == pytest.approx(3e-4), "proj group LR"
    assert opt.param_groups[1]["lr"] == pytest.approx(1e-5), "other group LR"


def test_build_tme_creates_split_optimizer_when_proj_lr_set():
    """_build_tme_module_and_optimizers creates two param groups when tme_proj_lr is set."""
    from train_scripts.training_utils import _build_tme_module_and_optimizers
    from unittest.mock import MagicMock, patch
    import types

    config = types.SimpleNamespace(
        tme_model="MultiGroupTMEModule",
        tme_base_ch=32,
        tme_proj_lr=3e-4,
        tme_lr=1e-5,
        channel_groups=[
            dict(name="cell_identity", channels=["a", "b", "c"]),
            dict(name="microenv",      channels=["x", "y"]),
        ],
        optimizer={"type": "AdamW", "lr": 5e-6, "weight_decay": 0.0,
                   "betas": (0.9, 0.999), "eps": 1e-8},
        lr_schedule_args={"num_warmup_steps": 10},
        num_epochs=1,
    )
    controlnet = MagicMock()
    dataloader = MagicMock()
    dataloader.__len__ = lambda self: 10
    logger     = MagicMock()
    active_ch  = ["cell_masks", "a", "b", "c", "x", "y"]

    # Patch the controlnet optimizer and scheduler builders so only the TME path runs
    with patch("train_scripts.training_utils.build_optimizer") as mock_opt, \
         patch("train_scripts.training_utils.build_lr_scheduler") as mock_sched:
        mock_opt.return_value   = MagicMock()
        mock_sched.return_value = MagicMock()
        result = _build_tme_module_and_optimizers(
            config, controlnet, dataloader, active_ch, logger
        )

    opt = result["optimizer_tme"]
    assert len(opt.param_groups) == 2, f"Expected 2 param groups, got {len(opt.param_groups)}"
    assert opt.param_groups[0]["lr"] == pytest.approx(3e-4), "proj group should be 3e-4"
    assert opt.param_groups[1]["lr"] == pytest.approx(1e-5), "other group should be 1e-5"
