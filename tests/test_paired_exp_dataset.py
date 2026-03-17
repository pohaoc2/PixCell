# tests/test_paired_exp_dataset.py
import h5py
import numpy as np
import pytest
import cv2
from pathlib import Path


TILE_IDS = ["tile_0001", "tile_0002"]
RESOLUTION = 256
LT_SZ = RESOLUTION // 8   # 32

BINARY_CHANNELS = [
    "cell_mask",
    "cell_type_healthy", "cell_type_cancer", "cell_type_immune",
    "cell_state_prolif", "cell_state_nonprolif", "cell_state_dead",
]
FLOAT_CHANNELS = ["vasculature", "oxygen", "glucose"]
ALL_TME_CHANNELS = BINARY_CHANNELS + FLOAT_CHANNELS  # cell_mask first

ACTIVE_CHANNELS = ALL_TME_CHANNELS


def _write_png(path: Path, binary: bool):
    """Write a tiny synthetic 256×256 PNG."""
    if binary:
        arr = np.zeros((RESOLUTION, RESOLUTION), dtype=np.uint8)
        arr[10:50, 10:50] = 255
    else:
        arr = np.random.randint(0, 256, (RESOLUTION, RESOLUTION), dtype=np.uint8)
    cv2.imwrite(str(path), arr)


@pytest.fixture()
def exp_root(tmp_path):
    """Build a minimal paired-exp dataset directory in tmp_path."""
    # --- exp_channels ---
    for ch in ALL_TME_CHANNELS:
        ch_dir = tmp_path / "exp_channels" / ch
        ch_dir.mkdir(parents=True)
        for tid in TILE_IDS:
            _write_png(ch_dir / f"{tid}.png", binary=(ch in BINARY_CHANNELS))

    # --- features (UNI embeddings) — 1536-dim to match UNI-2h embed_dim ---
    feat_dir = tmp_path / "features"
    feat_dir.mkdir()
    for tid in TILE_IDS:
        np.save(feat_dir / f"{tid}_uni.npy", np.random.randn(1536).astype(np.float32))

    # --- vae_features ---
    vae_dir = tmp_path / "vae_features"
    vae_dir.mkdir()
    for tid in TILE_IDS:
        # mean+std stacked on channel dim → shape [32, LT_SZ, LT_SZ]
        arr = np.random.randn(32, LT_SZ, LT_SZ).astype(np.float32)
        np.save(vae_dir / f"{tid}_sd3_vae.npy", arr)
        np.save(vae_dir / f"{tid}_mask_sd3_vae.npy", arr)

    # --- metadata/exp_index.hdf5 ---
    meta_dir = tmp_path / "metadata"
    meta_dir.mkdir()
    with h5py.File(meta_dir / "exp_index.hdf5", "w") as h5:
        dt = h5py.special_dtype(vlen=str)
        ds = h5.create_dataset(f"exp_{RESOLUTION}", shape=(len(TILE_IDS),), dtype=dt)
        for i, tid in enumerate(TILE_IDS):
            ds[i] = tid

    return tmp_path


def test_fixture_structure(exp_root):
    """Smoke test: fixture creates expected files."""
    assert (exp_root / "metadata" / "exp_index.hdf5").exists()
    for ch in ALL_TME_CHANNELS:
        for tid in TILE_IDS:
            assert (exp_root / "exp_channels" / ch / f"{tid}.png").exists()
    for tid in TILE_IDS:
        assert (exp_root / "features" / f"{tid}_uni.npy").exists()
        assert (exp_root / "vae_features" / f"{tid}_sd3_vae.npy").exists()


import torch


def test_dataset_len(exp_root):
    from diffusion.data.datasets.paired_exp_controlnet_dataset import PairedExpControlNetData
    ds = PairedExpControlNetData(
        root=str(exp_root),
        resolution=RESOLUTION,
        active_channels=ACTIVE_CHANNELS,
    )
    assert len(ds) == len(TILE_IDS)


def test_dataset_item_shapes(exp_root):
    from diffusion.data.datasets.paired_exp_controlnet_dataset import PairedExpControlNetData
    ds = PairedExpControlNetData(
        root=str(exp_root),
        resolution=RESOLUTION,
        active_channels=ACTIVE_CHANNELS,
    )
    vae_feat, ssl_feat, ctrl_tensor, vae_mask, data_info = ds[0]

    assert vae_feat.shape == (16, LT_SZ, LT_SZ), f"Got {vae_feat.shape}"
    assert ssl_feat.shape == (1, 1, 1536), f"Got {ssl_feat.shape}"
    assert ctrl_tensor.shape == (len(ACTIVE_CHANNELS), RESOLUTION, RESOLUTION), \
        f"Got {ctrl_tensor.shape}"
    assert vae_mask.shape == (16, LT_SZ, LT_SZ), f"Got {vae_mask.shape}"
    assert "img_hw" in data_info
    assert "aspect_ratio" in data_info
    assert "tile_id" in data_info


def test_dataset_is_paired(exp_root):
    """ssl_feat (UNI embed) must come from the same tile as vae_feat — not random."""
    from diffusion.data.datasets.paired_exp_controlnet_dataset import PairedExpControlNetData
    ds = PairedExpControlNetData(
        root=str(exp_root),
        resolution=RESOLUTION,
        active_channels=ACTIVE_CHANNELS,
    )
    expected_uni = torch.from_numpy(
        np.load(exp_root / "features" / "tile_0001_uni.npy")
    ).view(1, 1, -1)

    _, ssl_feat, _, _, info = ds[0]
    assert info["tile_id"] == "tile_0001"
    assert torch.allclose(ssl_feat, expected_uni), "ssl_feat is not paired with vae_feat"


def test_binary_channels_are_binary(exp_root):
    """One-hot cell type/state channels must contain only 0.0 and 1.0."""
    from diffusion.data.datasets.paired_exp_controlnet_dataset import PairedExpControlNetData
    ds = PairedExpControlNetData(
        root=str(exp_root),
        resolution=RESOLUTION,
        active_channels=ACTIVE_CHANNELS,
    )
    _, _, ctrl_tensor, _, _ = ds[0]
    for i, ch in enumerate(ACTIVE_CHANNELS):
        if ch in BINARY_CHANNELS:
            vals = ctrl_tensor[i].unique()
            assert set(vals.tolist()).issubset({0.0, 1.0}), \
                f"Channel '{ch}' is not binary: unique values = {vals.tolist()}"
