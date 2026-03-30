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
    "cell_masks",
    "cell_type_healthy", "cell_type_cancer", "cell_type_immune",
    "cell_state_prolif", "cell_state_nonprolif", "cell_state_dead",
    "vasculature",
]
FLOAT_CHANNELS = ["oxygen", "glucose"]
ALL_TME_CHANNELS = BINARY_CHANNELS + FLOAT_CHANNELS  # cell_masks first

ACTIVE_CHANNELS = ALL_TME_CHANNELS


def _write_png(path: Path, binary: bool):
    """Write a tiny synthetic 256×256 PNG."""
    if binary:
        arr = np.zeros((RESOLUTION, RESOLUTION), dtype=np.uint8)
        arr[10:50, 10:50] = 255
    else:
        arr = np.random.randint(0, 256, (RESOLUTION, RESOLUTION), dtype=np.uint8)
    cv2.imwrite(str(path), arr)


def _write_npy(path: Path, arr: np.ndarray):
    np.save(path, np.asarray(arr, dtype=np.float32))


@pytest.fixture()
def exp_root(tmp_path):
    """Build a minimal paired-exp dataset directory in tmp_path."""
    # --- exp_channels ---
    for ch in ALL_TME_CHANNELS:
        ch_dir = tmp_path / "exp_channels" / ch
        ch_dir.mkdir(parents=True)
        for tid in TILE_IDS:
            if ch in {"oxygen", "glucose"}:
                base = 0.2 if ch == "oxygen" else 0.6
                arr = np.linspace(
                    base,
                    base + 0.2,
                    RESOLUTION * RESOLUTION,
                    dtype=np.float32,
                ).reshape(RESOLUTION, RESOLUTION)
                _write_npy(ch_dir / f"{tid}.npy", arr)
            elif ch == "vasculature":
                arr = np.zeros((RESOLUTION, RESOLUTION), dtype=np.float32)
                arr[10:50, 10:50] = 1.0
                np.save(ch_dir / f"{tid}.npy", arr.astype(bool))
            else:
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
            ext = ".npy" if ch in {"vasculature", "oxygen", "glucose"} else ".png"
            assert (exp_root / "exp_channels" / ch / f"{tid}{ext}").exists()
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
    assert "tile_idx" in data_info


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
    assert ds.get_ids(0)["tile_id"] == "tile_0001"
    assert info["tile_idx"].item() == 0
    assert torch.allclose(ssl_feat, expected_uni), "ssl_feat is not paired with vae_feat"


def test_binary_channels_are_binary(exp_root):
    """Binary channels, including vasculature, must contain only 0.0 and 1.0."""
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


def test_oxygen_and_glucose_npy_preserve_global_scale(exp_root):
    """Continuous nutrient .npy channels should keep raw [0,1] values."""
    from diffusion.data.datasets.paired_exp_controlnet_dataset import PairedExpControlNetData

    ds = PairedExpControlNetData(
        root=str(exp_root),
        resolution=RESOLUTION,
        active_channels=ACTIVE_CHANNELS,
    )
    _, _, ctrl_tensor, _, _ = ds[0]

    oxygen = ctrl_tensor[ACTIVE_CHANNELS.index("oxygen")]
    glucose = ctrl_tensor[ACTIVE_CHANNELS.index("glucose")]

    assert float(oxygen.min()) == pytest.approx(0.2, abs=1e-6)
    assert float(oxygen.max()) == pytest.approx(0.4, abs=1e-6)
    assert float(glucose.min()) == pytest.approx(0.6, abs=1e-6)
    assert float(glucose.max()) == pytest.approx(0.8, abs=1e-6)


def test_dataset_supports_legacy_npy_channel_directories(exp_root):
    """Legacy oxygen_npy/glucose_npy/vasculature_npy folders should still load."""
    from diffusion.data.datasets.paired_exp_controlnet_dataset import PairedExpControlNetData

    for ch in ("oxygen", "glucose", "vasculature"):
        (exp_root / "exp_channels" / ch).rename(exp_root / "exp_channels" / f"{ch}_npy")

    ds = PairedExpControlNetData(
        root=str(exp_root),
        resolution=RESOLUTION,
        active_channels=ACTIVE_CHANNELS,
    )
    _, _, ctrl_tensor, _, _ = ds[0]
    assert ctrl_tensor.shape == (len(ACTIVE_CHANNELS), RESOLUTION, RESOLUTION)


def test_missing_vae_mask_returns_zeros(exp_root):
    """When the VAE cell mask file is absent, _load_vae_mask must return a zero tensor."""
    from diffusion.data.datasets.paired_exp_controlnet_dataset import PairedExpControlNetData
    # Remove the mask VAE file for tile_0001 only
    mask_path = exp_root / "vae_features" / "tile_0001_mask_sd3_vae.npy"
    mask_path.unlink()

    ds = PairedExpControlNetData(
        root=str(exp_root),
        resolution=RESOLUTION,
        active_channels=ACTIVE_CHANNELS,
    )
    _, _, _, vae_mask, info = ds[0]
    assert info["tile_idx"].item() == 0
    assert vae_mask.shape == (16, LT_SZ, LT_SZ)
    assert torch.all(vae_mask == 0.0), "Expected zeros when mask file is missing"


def test_legacy_cell_mask_alias_is_accepted(exp_root):
    from diffusion.data.datasets.paired_exp_controlnet_dataset import PairedExpControlNetData

    legacy_channels = ["cell_mask", *ACTIVE_CHANNELS[1:]]
    ds = PairedExpControlNetData(
        root=str(exp_root),
        resolution=RESOLUTION,
        active_channels=legacy_channels,
    )

    assert ds.active_channels[0] == "cell_masks"
    _, _, ctrl_tensor, _, _ = ds[0]
    assert ctrl_tensor.shape == (len(ACTIVE_CHANNELS), RESOLUTION, RESOLUTION)
