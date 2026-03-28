"""Tests for stage3_inference channel loading and zero_mask_latent logic."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from PIL import Image


def _write_gray_png(path, value: int, size: int = 32):
    arr = np.full((size, size), value, dtype=np.uint8)
    Image.fromarray(arr, mode="L").save(path)


def _make_sim_channels_dir(tmp_path, channels: list[str], pixel_value: int = 200, size: int = 32):
    for ch in channels:
        d = tmp_path / ch
        d.mkdir(parents=True, exist_ok=True)
        _write_gray_png(d / "t.png", pixel_value, size)
    return tmp_path


# --- load_sim_channels ---

def test_load_sim_channels_binary_thresholding(tmp_path):
    """cell_masks channel is thresholded to {0.0, 1.0}; vasculature is not."""
    _make_sim_channels_dir(tmp_path, ["cell_masks", "vasculature"], pixel_value=200)

    from stage3_inference import load_sim_channels

    result = load_sim_channels(
        sim_channels_dir=tmp_path,
        sim_id="t",
        active_channels=["cell_masks", "vasculature"],
        resolution=32,
    )

    assert result.shape == (2, 32, 32)
    assert result.dtype == torch.float32

    binary_vals = result[0].unique().tolist()
    assert all(v in (0.0, 1.0) for v in binary_vals), f"Binary channel has non-binary values: {binary_vals}"

    cont_vals = result[1].unique()
    assert not all(v.item() in (0.0, 1.0) for v in cont_vals), "Continuous channel was unexpectedly binarized"


def test_load_sim_channels_cell_mask_alias(tmp_path):
    """Falls back to cell_mask/ dir when cell_masks/ is absent."""
    _make_sim_channels_dir(tmp_path, ["cell_mask"], pixel_value=255)

    from stage3_inference import load_sim_channels

    result = load_sim_channels(
        sim_channels_dir=tmp_path,
        sim_id="t",
        active_channels=["cell_masks"],
        resolution=32,
    )
    assert result.shape == (1, 32, 32)
    assert result.max().item() == 1.0


# --- encode_ctrl_mask_latent ---

def test_encode_ctrl_mask_latent_shape():
    """Output shape is [1, 16, H/8, W/8]."""
    from train_scripts.inference_controlnet import encode_ctrl_mask_latent

    H, W = 32, 32
    ctrl_full = torch.rand(4, H, W)

    fake_vae = MagicMock()
    fake_vae.encode.return_value.latent_dist.mean = torch.zeros(1, 16, H // 8, W // 8)

    result = encode_ctrl_mask_latent(
        ctrl_full, fake_vae, vae_shift=0.0, vae_scale=1.0, device="cpu", dtype=torch.float32
    )
    assert result.shape == (1, 16, H // 8, W // 8)


def test_encode_ctrl_mask_latent_scaling():
    """Output equals (mean - vae_shift) * vae_scale."""
    from train_scripts.inference_controlnet import encode_ctrl_mask_latent

    ctrl_full = torch.rand(2, 16, 16)
    raw_mean = torch.ones(1, 16, 2, 2) * 5.0

    fake_vae = MagicMock()
    fake_vae.encode.return_value.latent_dist.mean = raw_mean

    result = encode_ctrl_mask_latent(
        ctrl_full, fake_vae, vae_shift=1.0, vae_scale=2.0, device="cpu", dtype=torch.float32
    )
    expected = (raw_mean - 1.0) * 2.0
    assert torch.allclose(result, expected)


# --- generate zero_mask_latent ---

def _make_fake_models(vae_mean: torch.Tensor, tme_out: torch.Tensor) -> dict:
    fake_vae = MagicMock()
    fake_vae.encode.return_value.latent_dist.mean = vae_mean
    fake_vae.decode = MagicMock(return_value=[torch.zeros(1, 3, 32, 32)])
    fake_tme = MagicMock(return_value=tme_out)
    return dict(vae=fake_vae, controlnet=MagicMock(), base_model=MagicMock(), tme_module=fake_tme)


def _make_config(zero_mask_latent: bool):
    return SimpleNamespace(
        data=SimpleNamespace(active_channels=["cell_masks", "vasculature"]),
        scale_factor=1.0,
        shift_factor=0.0,
        image_size=32,
        channel_groups=None,
        zero_mask_latent=zero_mask_latent,
    )


def _make_scheduler():
    sched = MagicMock()
    sched.init_noise_sigma = 1.0
    sched.timesteps = []
    return sched


def test_generate_zero_mask_latent_applied(tmp_path):
    """When zero_mask_latent=True, controlnet receives tme_out - vae_mask."""
    from unittest.mock import patch
    from stage3_inference import generate

    _make_sim_channels_dir(tmp_path, ["cell_masks", "vasculature"], pixel_value=200)

    vae_mean = torch.ones(1, 16, 4, 4)
    tme_out  = torch.full((1, 16, 4, 4), 3.0)
    captured = {}

    def fake_denoise(**kwargs):
        captured["cil"] = kwargs["controlnet_input_latent"].clone()
        return torch.zeros(1, 16, 4, 4)

    with patch("stage3_inference.denoise", side_effect=fake_denoise):
        generate(
            sim_channels_dir=tmp_path,
            sim_id="t",
            models=_make_fake_models(vae_mean, tme_out),
            config=_make_config(zero_mask_latent=True),
            uni_embeds=torch.zeros(1, 1, 1, 1536),
            scheduler=_make_scheduler(),
            guidance_scale=1.0,
            device="cpu",
        )

    expected = tme_out - vae_mean  # 3.0 - 1.0 = 2.0
    assert torch.allclose(captured["cil"].float(), expected.float())


def test_generate_zero_mask_latent_off(tmp_path):
    """When zero_mask_latent=False, controlnet receives raw tme_out."""
    from unittest.mock import patch
    from stage3_inference import generate

    _make_sim_channels_dir(tmp_path, ["cell_masks", "vasculature"], pixel_value=200)

    vae_mean = torch.ones(1, 16, 4, 4)
    tme_out  = torch.full((1, 16, 4, 4), 3.0)
    captured = {}

    def fake_denoise(**kwargs):
        captured["cil"] = kwargs["controlnet_input_latent"].clone()
        return torch.zeros(1, 16, 4, 4)

    with patch("stage3_inference.denoise", side_effect=fake_denoise):
        generate(
            sim_channels_dir=tmp_path,
            sim_id="t",
            models=_make_fake_models(vae_mean, tme_out),
            config=_make_config(zero_mask_latent=False),
            uni_embeds=torch.zeros(1, 1, 1, 1536),
            scheduler=_make_scheduler(),
            guidance_scale=1.0,
            device="cpu",
        )

    assert torch.allclose(captured["cil"].float(), tme_out.float())
