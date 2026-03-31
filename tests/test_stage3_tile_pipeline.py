"""Tests for tools/stage3_tile_pipeline.py channel loading and generate_tile logic."""
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


# --- load_channel ---

def test_load_channel_binary_thresholding(tmp_path):
    """Binary=True: pixel 200 → 1.0, pixel 50 → 0.0."""
    from tools.stage3_tile_pipeline import load_channel

    _write_gray_png(tmp_path / "high.png", 200)
    _write_gray_png(tmp_path / "low.png",   50)

    high = load_channel(tmp_path, "high", resolution=32, binary=True)
    low  = load_channel(tmp_path, "low",  resolution=32, binary=True)

    assert high.dtype == np.float32
    assert set(np.unique(high).tolist()) == {1.0}
    assert set(np.unique(low).tolist())  == {0.0}


def test_load_channel_continuous_not_binarized(tmp_path):
    """Binary=False: pixel 128 → ~0.5, not snapped to {0,1}."""
    from tools.stage3_tile_pipeline import load_channel

    _write_gray_png(tmp_path / "cont.png", 128)

    result = load_channel(tmp_path, "cont", resolution=32, binary=False)

    assert result.dtype == np.float32
    unique = set(np.unique(result).tolist())
    assert unique != {0.0, 1.0}, "Continuous channel was unexpectedly binarized"
    assert 0.4 < result.mean() < 0.6


def test_load_channel_reflect_pad_output_size(tmp_path):
    """Non-binary channel is reflect-padded then resized back to requested resolution."""
    from tools.stage3_tile_pipeline import load_channel

    _write_gray_png(tmp_path / "pad.png", 100, size=32)

    result = load_channel(tmp_path, "pad", resolution=32, binary=False)

    assert result.shape == (32, 32)
    assert result.dtype == np.float32


# --- resolve_data_layout ---

def test_resolve_data_layout_orion(tmp_path):
    """ORION-style root: exp_channels/, features/, he/ all exist."""
    from tools.stage3_tile_pipeline import resolve_data_layout

    (tmp_path / "exp_channels").mkdir()
    (tmp_path / "features").mkdir()
    (tmp_path / "he").mkdir()

    ch_dir, feat_dir, he_dir = resolve_data_layout(tmp_path)

    assert ch_dir   == tmp_path / "exp_channels"
    assert feat_dir == tmp_path / "features"
    assert he_dir   == tmp_path / "he"


def test_resolve_data_layout_flat(tmp_path):
    """Flat sim-style root: no exp_channels/, features/, he/ — all fall back to data_root."""
    from tools.stage3_tile_pipeline import resolve_data_layout

    ch_dir, feat_dir, he_dir = resolve_data_layout(tmp_path)

    assert ch_dir   == tmp_path
    assert feat_dir == tmp_path
    assert he_dir   == tmp_path


# --- generate_tile zero_mask_latent ---

def _setup_exp_channels(tmp_path):
    exp_ch_dir = tmp_path / "exp_channels"
    for ch in ("cell_masks", "vasculature"):
        (exp_ch_dir / ch).mkdir(parents=True)
        _write_gray_png(exp_ch_dir / ch / "t.png", 200)
    return exp_ch_dir


def _make_config(zero_mask_latent: bool):
    return SimpleNamespace(
        data=SimpleNamespace(active_channels=["cell_masks", "vasculature"]),
        scale_factor=1.0,
        shift_factor=0.0,
        image_size=32,
        channel_groups=[{"name": "g1", "channels": ["vasculature"]}],
        zero_mask_latent=zero_mask_latent,
    )


def _make_four_group_config(zero_mask_latent: bool = False):
    return SimpleNamespace(
        data=SimpleNamespace(
            active_channels=[
                "cell_masks",
                "cell_type_healthy", "cell_type_cancer", "cell_type_immune",
                "cell_state_prolif", "cell_state_nonprolif", "cell_state_dead",
                "vasculature",
                "oxygen", "glucose",
            ]
        ),
        scale_factor=1.0,
        shift_factor=0.0,
        image_size=32,
        channel_groups=[
            {
                "name": "cell_identity",
                "channels": ["cell_type_healthy", "cell_type_cancer", "cell_type_immune"],
            },
            {
                "name": "cell_state",
                "channels": ["cell_state_prolif", "cell_state_nonprolif", "cell_state_dead"],
            },
            {"name": "vasculature", "channels": ["vasculature"]},
            {"name": "microenv", "channels": ["oxygen", "glucose"]},
        ],
        zero_mask_latent=zero_mask_latent,
    )


def test_generate_tile_zero_mask_latent_applied(tmp_path):
    """zero_mask_latent=True: controlnet receives tme_out - vae_mask."""
    from unittest.mock import patch
    from tools.stage3_tile_pipeline import generate_tile

    exp_ch_dir = _setup_exp_channels(tmp_path)

    vae_mean = torch.ones(1, 16, 4, 4)
    tme_out  = torch.full((1, 16, 4, 4), 3.0)
    fake_tme = MagicMock(return_value=tme_out)
    fake_vae = MagicMock()
    fake_vae.decode = MagicMock(return_value=[torch.zeros(1, 3, 32, 32)])

    models = dict(vae=fake_vae, controlnet=MagicMock(), base_model=MagicMock(), tme_module=fake_tme)
    scheduler = MagicMock()
    scheduler.init_noise_sigma = 1.0
    scheduler.timesteps = []

    captured = {}

    def fake_denoise(**kwargs):
        captured["cil"] = kwargs["controlnet_input_latent"].clone()
        return torch.zeros(1, 16, 4, 4)

    with patch("train_scripts.inference_controlnet.encode_ctrl_mask_latent", return_value=vae_mean), \
         patch("tools.channel_group_utils.split_channels_to_groups", return_value={}), \
         patch("train_scripts.inference_controlnet.denoise", side_effect=fake_denoise):
        generate_tile(
            tile_id="t",
            models=models,
            config=_make_config(zero_mask_latent=True),
            scheduler=scheduler,
            uni_embeds=torch.zeros(1, 1, 1, 1536),
            device="cpu",
            exp_channels_dir=exp_ch_dir,
            guidance_scale=1.0,
        )

    expected = tme_out - vae_mean  # 3.0 - 1.0 = 2.0
    assert torch.allclose(captured["cil"].float(), expected.float())


def test_generate_tile_zero_mask_latent_off(tmp_path):
    """zero_mask_latent=False: controlnet receives raw tme_out."""
    from unittest.mock import patch
    from tools.stage3_tile_pipeline import generate_tile

    exp_ch_dir = _setup_exp_channels(tmp_path)

    vae_mean = torch.ones(1, 16, 4, 4)
    tme_out  = torch.full((1, 16, 4, 4), 3.0)
    fake_tme = MagicMock(return_value=tme_out)
    fake_vae = MagicMock()
    fake_vae.decode = MagicMock(return_value=[torch.zeros(1, 3, 32, 32)])

    models = dict(vae=fake_vae, controlnet=MagicMock(), base_model=MagicMock(), tme_module=fake_tme)
    scheduler = MagicMock()
    scheduler.init_noise_sigma = 1.0
    scheduler.timesteps = []

    captured = {}

    def fake_denoise(**kwargs):
        captured["cil"] = kwargs["controlnet_input_latent"].clone()
        return torch.zeros(1, 16, 4, 4)

    with patch("train_scripts.inference_controlnet.encode_ctrl_mask_latent", return_value=vae_mean), \
         patch("tools.channel_group_utils.split_channels_to_groups", return_value={}), \
         patch("train_scripts.inference_controlnet.denoise", side_effect=fake_denoise):
        generate_tile(
            tile_id="t",
            models=models,
            config=_make_config(zero_mask_latent=False),
            scheduler=scheduler,
            uni_embeds=torch.zeros(1, 1, 1, 1536),
            device="cpu",
            exp_channels_dir=exp_ch_dir,
            guidance_scale=1.0,
        )

    assert torch.allclose(captured["cil"].float(), tme_out.float())


def test_group_ablation_plan_counts_cover_requested_full_suite():
    from tools.stage3_ablation import build_progressive_order_conditions, build_subset_conditions

    group_names = ("cell_identity", "cell_state", "vasculature", "microenv")

    singles = build_subset_conditions(group_names, subset_size=1)
    pairs = build_subset_conditions(group_names, subset_size=2)
    triples = build_subset_conditions(group_names, subset_size=3)
    order_sweeps = build_progressive_order_conditions(group_names, zero_mask_latent=False)

    assert len(singles) == 4
    assert len(pairs) == 6
    assert len(triples) == 4
    assert len(order_sweeps) == 24
    assert any("nutrient" in cond.label for cond in singles)
    assert all(len(conditions) == 5 for _, conditions in order_sweeps)


def test_generate_ablation_images_respects_requested_group_conditions(tmp_path):
    from unittest.mock import patch

    from tools.stage3_ablation import AblationCondition
    from tools.stage3_tile_pipeline import generate_ablation_images

    config = _make_four_group_config()
    fake_vae = MagicMock()
    fake_vae.to.return_value = fake_vae
    fake_vae.eval.return_value = fake_vae
    fake_vae.decode = MagicMock(return_value=[torch.zeros(1, 3, 32, 32)])

    class FakeTME:
        def __init__(self):
            self.active_groups_seen = []

        def __call__(self, vae_mask, tme_dict, active_groups=None):
            self.active_groups_seen.append(tuple(sorted(active_groups or ())))
            return torch.full_like(vae_mask, float(len(active_groups or ())))

    fake_tme = FakeTME()
    models = dict(
        vae=fake_vae,
        controlnet=MagicMock(),
        base_model=MagicMock(),
        tme_module=fake_tme,
    )
    scheduler = MagicMock()
    scheduler.init_noise_sigma = 1.0
    scheduler.timesteps = []

    captured = []

    def fake_denoise(**kwargs):
        captured.append(kwargs["controlnet_input_latent"].clone())
        return torch.zeros(1, 16, 4, 4)

    fake_ctrl_full = torch.zeros(len(config.data.active_channels), 32, 32)
    fake_tme_dict = {
        group["name"]: torch.zeros(1, len(group["channels"]), 32, 32)
        for group in config.channel_groups
    }
    conditions = [
        AblationCondition(label="identity only", active_groups=("cell_identity",)),
        AblationCondition(label="state + nutrient", active_groups=("cell_state", "microenv")),
    ]

    with patch("tools.stage3_tile_pipeline.load_exp_channels", return_value=fake_ctrl_full), \
         patch("train_scripts.inference_controlnet.encode_ctrl_mask_latent", return_value=torch.ones(1, 16, 4, 4)), \
         patch("tools.channel_group_utils.split_channels_to_groups", return_value=fake_tme_dict), \
         patch("train_scripts.inference_controlnet.denoise", side_effect=fake_denoise):
        results = generate_ablation_images(
            tile_id="t",
            models=models,
            config=config,
            scheduler=scheduler,
            uni_embeds=torch.zeros(1, 1, 1, 1536),
            device="cpu",
            exp_channels_dir=tmp_path,
            guidance_scale=1.0,
            seed=123,
            conditions=conditions,
        )

    assert [label for label, _ in results] == ["identity only", "state + nutrient"]
    assert fake_tme.active_groups_seen == [
        ("cell_identity",),
        ("cell_state", "microenv"),
    ]
    assert len(captured) == 2
