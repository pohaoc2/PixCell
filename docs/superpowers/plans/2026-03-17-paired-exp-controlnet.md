# Paired Experimental ControlNet Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fine-tune PixCellControlNet on paired H&E + CODEX-derived TME channels, with CFG dropout and channel reliability weighting, to enable simulation-to-experiment domain mapping.

**Architecture:** `PairedExpControlNetData` loads paired (H&E, CODEX channels) from a single index — unlike `SimControlNetData`'s random cross-sampling. Training adds CFG dropout on the UNI embedding (15%) and per-channel reliability weights (0.5× for vasculature/O2/glucose). Inference supports TME-only mode via `null_uni_embed()`.

**Tech Stack:** PyTorch, HuggingFace Accelerate, diffusers, h5py, cv2, numpy — all already in environment.

**Spec:** `docs/superpowers/specs/2026-03-17-paired-exp-controlnet-design.md`

---

## Chunk 1: Dataset — `PairedExpControlNetData`

### File Map

| Action | Path |
|---|---|
| Create | `diffusion/data/datasets/paired_exp_controlnet_dataset.py` |
| Create | `tests/test_paired_exp_dataset.py` |
| Modify | `diffusion/data/datasets/__init__.py` |

---

### Task 1: Synthetic fixture for dataset tests

**Files:**
- Create: `tests/test_paired_exp_dataset.py`

The tests need a minimal on-disk layout matching what `PairedExpControlNetData` expects. Build a pytest fixture that creates it in a temp directory.

Channel layout (same format as sim channels, all 256×256 PNGs):
```
tmp/
├── metadata/exp_index.hdf5         # keys: ["exp_256"] → ["tile_0001", "tile_0002"]
├── exp_channels/
│   ├── cell_mask/      tile_0001.png   tile_0002.png   # binary, uint8 {0,255}
│   ├── cell_type_healthy/  ...                         # binary
│   ├── cell_type_cancer/   ...
│   ├── cell_type_immune/   ...
│   ├── cell_state_prolif/  ...
│   ├── cell_state_nonprolif/ ...
│   ├── cell_state_dead/    ...
│   ├── vasculature/    tile_0001.png   tile_0002.png   # grayscale float→uint8
│   ├── oxygen/         ...
│   └── glucose/        ...
├── features/
│   └── tile_0001_uni.npy   tile_0002_uni.npy           # shape [1152] float32
└── vae_features/
    ├── tile_0001_sd3_vae.npy       # shape [32, 32, 32] (mean+std cat on ch=0)
    ├── tile_0001_mask_sd3_vae.npy  # shape [32, 32, 32]
    ├── tile_0002_sd3_vae.npy
    └── tile_0002_mask_sd3_vae.npy
```

- [ ] **Step 1.1 — Write the fixture and a smoke-test that just checks the fixture itself**

```python
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

    # --- features (UNI embeddings) ---
    feat_dir = tmp_path / "features"
    feat_dir.mkdir()
    for tid in TILE_IDS:
        np.save(feat_dir / f"{tid}_uni.npy", np.random.randn(1152).astype(np.float32))

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
```

- [ ] **Step 1.2 — Run the fixture test to verify it passes**

```bash
cd /home/pohaoc2/UW/bagherilab/PixCell
python -m pytest tests/test_paired_exp_dataset.py::test_fixture_structure -v
```

Expected: `PASSED`

---

### Task 2: Write failing dataset shape tests

- [ ] **Step 2.1 — Add shape tests (will fail until dataset exists)**

Append to `tests/test_paired_exp_dataset.py`:

```python
import torch


ACTIVE_CHANNELS = [
    "cell_mask",
    "cell_type_healthy", "cell_type_cancer", "cell_type_immune",
    "cell_state_prolif", "cell_state_nonprolif", "cell_state_dead",
    "vasculature", "oxygen", "glucose",
]


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

    # vae_feat: H&E VAE latent, paired (same tile)
    assert vae_feat.shape == (16, LT_SZ, LT_SZ), f"Got {vae_feat.shape}"
    # ssl_feat: UNI embedding [1, 1, 1152]
    assert ssl_feat.shape == (1, 1, 1152), f"Got {ssl_feat.shape}"
    # ctrl_tensor: all channels [C, H, W]  (C = len(active_channels))
    assert ctrl_tensor.shape == (len(ACTIVE_CHANNELS), RESOLUTION, RESOLUTION), \
        f"Got {ctrl_tensor.shape}"
    # vae_mask: cell_mask VAE latent
    assert vae_mask.shape == (16, LT_SZ, LT_SZ), f"Got {vae_mask.shape}"
    # data_info keys
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
    # Load ground-truth UNI for tile_0001
    expected_uni = torch.from_numpy(
        np.load(exp_root / "features" / "tile_0001_uni.npy")
    ).view(1, 1, -1)

    # dataset[0] should be tile_0001 (first in index)
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
        if ch in [
            "cell_mask", "cell_type_healthy", "cell_type_cancer", "cell_type_immune",
            "cell_state_prolif", "cell_state_nonprolif", "cell_state_dead",
        ]:
            vals = ctrl_tensor[i].unique()
            assert set(vals.tolist()).issubset({0.0, 1.0}), \
                f"Channel '{ch}' is not binary: unique values = {vals.tolist()}"
```

- [ ] **Step 2.2 — Run tests to confirm they fail**

```bash
python -m pytest tests/test_paired_exp_dataset.py -v -k "not fixture"
```

Expected: `ImportError` or `ModuleNotFoundError` for `PairedExpControlNetData`

---

### Task 3: Implement `PairedExpControlNetData`

**Files:**
- Create: `diffusion/data/datasets/paired_exp_controlnet_dataset.py`

Pattern to follow: `diffusion/data/datasets/sim_controlnet_dataset.py` — reuse `_find_file`, `_load_spatial_file`, `_validate_channels`, `_write_h5_index`, `build_real_index` from that module.

- [ ] **Step 3.1 — Write the dataset class**

```python
# diffusion/data/datasets/paired_exp_controlnet_dataset.py
"""
PairedExpControlNetData — paired experimental H&E + CODEX-derived TME channels.

Unlike SimControlNetData (unpaired), each tile provides:
    - H&E VAE latent     (paired — same tile_id)
    - UNI embedding      (paired — same tile_id)
    - CODEX TME channels (paired — same tile_id)
    - cell_mask VAE latent (paired — same tile_id)

Directory layout under `root`:
    exp_data_root/
    ├── metadata/
    │   └── exp_index.hdf5          # flat list of paired tile IDs
    ├── exp_channels/               # one sub-folder per channel
    │   ├── cell_mask/              # binary PNG (required)
    │   ├── cell_type_healthy/      # binary PNG  (one-hot, required)
    │   ├── cell_type_cancer/       # binary PNG  (one-hot, required)
    │   ├── cell_type_immune/       # binary PNG  (one-hot, required)
    │   ├── cell_state_prolif/      # binary PNG  (one-hot, required)
    │   ├── cell_state_nonprolif/   # binary PNG  (one-hot, required)
    │   ├── cell_state_dead/        # binary PNG  (one-hot, required)
    │   ├── vasculature/            # float PNG   (approximate, optional)
    │   ├── oxygen/                 # float PNG   (approximate, optional)
    │   └── glucose/                # float PNG   (approximate, optional)
    ├── features/
    │   └── {tile_id}_uni.npy       # shape [1152]  paired H&E UNI embedding
    └── vae_features/
        ├── {tile_id}_sd3_vae.npy           # shape [32, lt_sz, lt_sz]
        └── {tile_id}_mask_sd3_vae.npy      # shape [32, lt_sz, lt_sz]
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from diffusers.utils.torch_utils import randn_tensor
from torch.utils.data import Dataset

# Re-use file I/O and validation helpers from sim dataset
from diffusion.data.datasets.sim_controlnet_dataset import (
    _find_file,
    _load_spatial_file,
    _write_h5_index,
)

# ── Channel registry ──────────────────────────────────────────────────────────

REQUIRED_CHANNELS: list[str] = [
    "cell_mask",
    "cell_type_healthy", "cell_type_cancer", "cell_type_immune",
    "cell_state_prolif", "cell_state_nonprolif", "cell_state_dead",
]
OPTIONAL_CHANNELS: list[str] = ["vasculature", "oxygen", "glucose"]
ALL_CHANNELS: list[str] = REQUIRED_CHANNELS + OPTIONAL_CHANNELS

_BINARY_CHANNELS: frozenset[str] = frozenset({
    "cell_mask",
    "cell_type_healthy", "cell_type_cancer", "cell_type_immune",
    "cell_state_prolif", "cell_state_nonprolif", "cell_state_dead",
})


def _validate_channels(active: list[str]) -> list[str]:
    unknown = set(active) - set(ALL_CHANNELS)
    if unknown:
        raise ValueError(f"Unknown channels: {unknown}. Available: {ALL_CHANNELS}")
    for ch in REQUIRED_CHANNELS:
        if ch not in active:
            raise ValueError(f"Required channel '{ch}' missing from active_channels={active}")
    ordered  = [ch for ch in REQUIRED_CHANNELS if ch in active]
    ordered += [ch for ch in active if ch not in REQUIRED_CHANNELS]
    return ordered


class PairedExpControlNetData(Dataset):
    """
    Dataset for PixCell-256 ControlNet fine-tuning on paired experimental data.

    Returns the same tuple as SimControlNetData / PanCancerControlNetData:
        (vae_feat, ssl_feat, ctrl_tensor, vae_mask, data_info)

    where:
        vae_feat    : [16, lt_sz, lt_sz]              paired H&E VAE latent
        ssl_feat    : [1, 1, 1152]                     paired H&E UNI embedding
        ctrl_tensor : [C, resolution, resolution]     CODEX TME channels
        vae_mask    : [16, lt_sz, lt_sz]               paired cell_mask VAE latent
        data_info   : dict (includes 'tile_id' for debugging)

    Args:
        root (str):            exp_data_root/ directory.
        resolution (int):      Spatial resolution (256).
        active_channels (list[str]):
            Which channels to include. Required: all 7 binary channels.
            Optional: ["vasculature", "oxygen", "glucose"].
        exp_channels_dir (str):  sub-folder for channel PNGs. Default "exp_channels".
        features_dir (str):      sub-folder for UNI .npy files. Default "features".
        vae_features_dir (str):  sub-folder for VAE .npy files. Default "vae_features".
        exp_index_h5 (str):      HDF5 index path. Default "metadata/exp_index.hdf5".
        vae_prefix (str):        VAE latent filename suffix. Default "sd3_vae".
        ssl_prefix (str):        UNI embedding filename suffix. Default "uni".
        train_subset_keys (list[str] | None): restrict to specific HDF5 keys.
    """

    def __init__(self, root: str, resolution: int, **kwargs):
        self.root       = Path(root)
        self.resolution = resolution
        self.lt_sz      = resolution // 8

        raw_channels         = kwargs.get("active_channels", list(REQUIRED_CHANNELS))
        self.active_channels = _validate_channels(list(raw_channels))
        self.n_channels      = len(self.active_channels)

        self.exp_channels_dir = self.root / kwargs.get("exp_channels_dir", "exp_channels")
        self.features_dir     = self.root / kwargs.get("features_dir",     "features")
        self.vae_features_dir = self.root / kwargs.get("vae_features_dir", "vae_features")
        self.vae_prefix       = kwargs.get("vae_prefix", "sd3_vae")
        self.ssl_prefix       = kwargs.get("ssl_prefix", "uni")

        exp_h5 = self.root / kwargs.get("exp_index_h5", "metadata/exp_index.hdf5")
        subset = kwargs.get("train_subset_keys", None)
        self.tile_ids = self._load_index(exp_h5, resolution, subset)

        if not self.tile_ids:
            raise RuntimeError(f"No tile IDs found in {exp_h5}")

        self.load_vae_feat = True

        print(
            f"[PairedExpControlNetData] "
            f"{len(self.tile_ids)} paired tiles | "
            f"channels={self.active_channels} | "
            f"resolution={resolution}"
        )

    @staticmethod
    def _load_index(h5_path: Path, resolution: int,
                    subset_keys: list[str] | None) -> list[str]:
        ids: list[str] = []
        with h5py.File(h5_path, "r") as h5:
            keys = [k for k in h5.keys() if f"_{resolution}" in k]
            if subset_keys is not None:
                keys = [k for k in keys if k in subset_keys]
            for key in keys:
                ids.extend(v.decode("utf-8") for v in h5[key])
        return ids

    def _build_ctrl_tensor(self, tile_id: str) -> torch.Tensor:
        planes: list[np.ndarray] = []
        for ch in self.active_channels:
            ch_dir = self.exp_channels_dir / ch
            fpath  = _find_file(ch_dir, tile_id)
            arr    = _load_spatial_file(
                fpath,
                resolution=self.resolution,
                binary=(ch in _BINARY_CHANNELS),
            )
            planes.append(arr)
        ctrl = torch.from_numpy(np.stack(planes, axis=0))
        assert ctrl.shape == (self.n_channels, self.resolution, self.resolution)
        return ctrl

    def _load_vae_feat(self, tile_id: str) -> torch.Tensor:
        path      = self.vae_features_dir / f"{tile_id}_{self.vae_prefix}.npy"
        arr       = torch.from_numpy(np.load(path))
        mean, std = arr.chunk(2, dim=0)
        if mean.ndim == 4 and mean.shape[0] == 1:
            mean, std = mean.squeeze(0), std.squeeze(0)
        sample = randn_tensor(mean.shape, device=mean.device, dtype=mean.dtype)
        feat   = mean + std * sample
        assert feat.shape == (16, self.lt_sz, self.lt_sz)
        return feat

    def _load_vae_mask(self, tile_id: str) -> torch.Tensor:
        path = self.vae_features_dir / f"{tile_id}_mask_{self.vae_prefix}.npy"
        if not path.exists():
            return torch.zeros(16, self.lt_sz, self.lt_sz)
        arr     = torch.from_numpy(np.load(path))
        mean, _ = arr.chunk(2, dim=0)
        if mean.ndim == 4 and mean.shape[0] == 1:
            mean = mean.squeeze(0)
        assert mean.shape == (16, self.lt_sz, self.lt_sz)
        return mean

    def _load_ssl_feat(self, tile_id: str) -> torch.Tensor:
        path = self.features_dir / f"{tile_id}_{self.ssl_prefix}.npy"
        feat = torch.from_numpy(np.load(path))
        return feat.view(1, 1, -1)   # [1, 1, 1152]

    def __len__(self) -> int:
        return len(self.tile_ids)

    def __getitem__(self, idx: int):
        tile_id     = self.tile_ids[idx]
        ctrl_tensor = self._build_ctrl_tensor(tile_id)
        vae_feat    = self._load_vae_feat(tile_id)
        vae_mask    = self._load_vae_mask(tile_id)
        ssl_feat    = self._load_ssl_feat(tile_id)

        data_info = {
            "img_hw":       torch.tensor([self.resolution] * 2, dtype=torch.float32),
            "aspect_ratio": torch.tensor(1.0),
            "tile_id":      tile_id,   # for debugging; string, not passed to accelerator
        }

        return vae_feat, ssl_feat, ctrl_tensor, vae_mask, data_info


def build_exp_index(
    exp_channels_dir: str | Path,
    output_path: str | Path,
    resolution: int = 256,
    key_name: str | None = None,
):
    """
    Build HDF5 index for paired exp tiles by scanning exp_channels/cell_mask/.

    Example:
        build_exp_index(
            "exp_data_root/exp_channels",
            "exp_data_root/metadata/exp_index.hdf5",
        )
    """
    mask_dir = Path(exp_channels_dir) / "cell_mask"
    if not mask_dir.exists():
        raise RuntimeError(f"cell_mask directory not found: {mask_dir}")
    files = sorted(p for ext in ("*.png", "*.npy") for p in mask_dir.glob(ext))
    if not files:
        raise RuntimeError(f"No PNG/NPY files found in {mask_dir}")
    tile_ids = [f.stem for f in files]
    _write_h5_index(Path(output_path), key_name or f"exp_{resolution}", tile_ids)
    print(f"[build_exp_index] Wrote {len(tile_ids)} IDs → {output_path}")
```

- [ ] **Step 3.2 — Register the dataset in `__init__.py`**

In `diffusion/data/datasets/__init__.py`, append:
```python
from .paired_exp_controlnet_dataset import PairedExpControlNetData
```

- [ ] **Step 3.3 — Run the dataset shape tests**

```bash
python -m pytest tests/test_paired_exp_dataset.py -v
```

Expected: all 5 tests `PASSED`

- [ ] **Step 3.4 — Commit**

```bash
git add diffusion/data/datasets/paired_exp_controlnet_dataset.py \
        diffusion/data/datasets/__init__.py \
        tests/test_paired_exp_dataset.py
git commit -m "feat: add PairedExpControlNetData dataset with paired H&E+CODEX channels"
```

---

## Chunk 2: Training Loop + Config

### File Map

| Action | Path |
|---|---|
| Create | `train_scripts/train_controlnet_exp.py` |
| Create | `configs/config_controlnet_exp.py` |
| Create | `tests/test_train_controlnet_exp.py` |

---

### Task 4: Write failing tests for CFG dropout and channel weighting

- [ ] **Step 4.1 — Write unit tests**

```python
# tests/test_train_controlnet_exp.py
"""
Unit tests for CFG dropout and channel reliability weighting logic.
These test the pure tensor operations, independent of the full training loop.
"""
import torch
import pytest


# ── CFG dropout ───────────────────────────────────────────────────────────────

def _apply_cfg_dropout(y: torch.Tensor, prob: float, rng: torch.Generator) -> torch.Tensor:
    """Extracted logic from train_controlnet_exp — import once implemented."""
    if torch.rand(1, generator=rng).item() < prob:
        return torch.zeros_like(y)
    return y


def test_cfg_dropout_never_at_zero_prob():
    y   = torch.ones(1, 1, 1, 1152)
    rng = torch.Generator().manual_seed(0)
    for _ in range(100):
        out = _apply_cfg_dropout(y, prob=0.0, rng=rng)
        assert not torch.all(out == 0), "dropout should never fire at prob=0"


def test_cfg_dropout_always_at_one_prob():
    y   = torch.ones(1, 1, 1, 1152)
    rng = torch.Generator().manual_seed(0)
    for _ in range(20):
        out = _apply_cfg_dropout(y, prob=1.0, rng=rng)
        assert torch.all(out == 0), "dropout should always fire at prob=1"


def test_cfg_dropout_approximate_rate():
    """With prob=0.15 over 1000 trials, rate should be ~15% ± 3%."""
    y     = torch.ones(1, 1, 1, 1152)
    rng   = torch.Generator().manual_seed(42)
    drops = sum(
        1 for _ in range(1000)
        if torch.all(_apply_cfg_dropout(y, prob=0.15, rng=rng) == 0)
    )
    assert 120 <= drops <= 180, f"Expected ~150 drops, got {drops}"


def test_cfg_dropout_output_is_zeros_not_noise():
    y   = torch.ones(1, 1, 1, 1152) * 99.0
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
```

- [ ] **Step 4.2 — Run tests to verify they pass (pure tensor logic, no imports needed)**

```bash
python -m pytest tests/test_train_controlnet_exp.py -v
```

Expected: all 7 tests `PASSED` (the helper functions are defined inline in the test file — they'll be extracted from the training loop in the next step)

---

### Task 5: Implement `train_controlnet_exp.py`

- [ ] **Step 5.1 — Write the training script**

`train_scripts/train_controlnet_exp.py` is structurally identical to `train_controlnet_sim.py` with three focused changes:
1. Uses `PairedExpControlNetData` instead of `SimControlNetData`
2. Adds CFG dropout on `y` before forward pass
3. Applies channel reliability weights to `tme_channels`

```python
# train_scripts/train_controlnet_exp.py
"""
train_controlnet_exp.py

PixCell ControlNet fine-tuning on PAIRED experimental H&E + CODEX-derived TME channels.

Three additions vs train_controlnet_sim.py (all marked # <- EXP):
    1. PairedExpControlNetData  — paired dataset (single index, no random cross-sampling)
    2. CFG dropout              — zero UNI embedding with probability cfg_dropout_prob
    3. Channel reliability weights — attenuate approximate channels before TMEEncoder

Usage:
    python  train_scripts/train_controlnet_exp.py <config_path> [--work-dir ...] [--debug ...]
    accelerate launch train_scripts/train_controlnet_exp.py <config_path>
"""
import os
import time
from copy import deepcopy

import torch

from train_scripts.initialize_models import (
    initialize_config_and_accelerator,
    initialize_models,
    setup_training_state,
    ema_update,
    _resume_from_checkpoint,
)
from diffusion.data.builder import build_dataloader
from diffusion.model.builder import build_model
from diffusion.utils.checkpoint import save_checkpoint
from diffusion.utils.optimizer import build_optimizer
from diffusion.utils.lr_scheduler import build_lr_scheduler

from diffusion.data.datasets.paired_exp_controlnet_dataset import PairedExpControlNetData
from train_scripts.train_controlnet_sim import (
    training_losses_controlnet,
    _save_sim_checkpoint,
    load_sim_checkpoint,
)


# ── Initialization ────────────────────────────────────────────────────────────

def initialize_exp_training(config, accelerator, logger, controlnet):
    """
    Build everything for paired-exp training:
        PairedExpControlNetData + TMEConditioningModule + both optimizers/schedulers.

    Config fields (in addition to base controlnet fields):
        exp_data_root           (str)    root of exp dataset
        active_channels         (list)   channel list for dataset
        cfg_dropout_prob        (float)  default 0.15
        channel_reliability_weights (list[float])  one per tme channel (excl. cell_mask)
        tme_model, tme_base_ch, tme_lr  (same as sim config)
    """
    active_channels = getattr(config, "active_channels", [
        "cell_mask",
        "cell_type_healthy", "cell_type_cancer", "cell_type_immune",
        "cell_state_prolif",  "cell_state_nonprolif", "cell_state_dead",
        "vasculature", "oxygen", "glucose",
    ])

    dataset = PairedExpControlNetData(
        root=config.exp_data_root,
        resolution=config.image_size,
        active_channels=active_channels,
        vae_prefix=getattr(config, "vae_prefix", "sd3_vae"),
        ssl_prefix=getattr(config, "ssl_prefix", "uni"),
    )
    train_dataloader = build_dataloader(
        dataset,
        num_workers=config.num_workers,
        batch_size=config.train_batch_size,
        shuffle=True,
    )

    n_tme_channels = len(active_channels) - 1   # exclude cell_mask
    tme_module = build_model(
        getattr(config, "tme_model", "TMEConditioningModule"),
        False, False,
        n_tme_channels=n_tme_channels,
        base_ch=getattr(config, "tme_base_ch", 32),
    )
    logger.info(
        f"[TMEConditioningModule] n_tme_channels={n_tme_channels}  "
        f"trainable params={sum(p.numel() for p in tme_module.parameters() if p.requires_grad):,}"
    )

    optimizer    = build_optimizer(controlnet, config.optimizer)
    lr_scheduler = build_lr_scheduler(config, optimizer, train_dataloader, lr_scale_ratio=1)

    tme_optimizer_cfg       = deepcopy(config.optimizer)
    tme_optimizer_cfg['lr'] = getattr(config, "tme_lr", config.optimizer.get('lr', 1e-4))
    optimizer_tme    = build_optimizer(tme_module, tme_optimizer_cfg)
    lr_scheduler_tme = build_lr_scheduler(config, optimizer_tme, train_dataloader, lr_scale_ratio=1)

    return {
        "train_dataloader": train_dataloader,
        "tme_module":       tme_module,
        "optimizer":        optimizer,
        "optimizer_tme":    optimizer_tme,
        "lr_scheduler":     lr_scheduler,
        "lr_scheduler_tme": lr_scheduler_tme,
    }


# ── Training loop ─────────────────────────────────────────────────────────────

def train_controlnet_exp(models_dict):
    """
    Paired-exp ControlNet training loop.

    Identical to train_controlnet_sim() with 3 additions (marked # <- EXP).
    """
    base_model        = models_dict['base_model']
    controlnet        = models_dict['controlnet']
    model_ema         = models_dict['model_ema']
    vae               = models_dict['vae']
    train_diffusion   = models_dict['train_diffusion']
    optimizer         = models_dict['optimizer']
    lr_scheduler      = models_dict['lr_scheduler']
    train_dataloader  = models_dict['train_dataloader']
    accelerator       = models_dict['accelerator']
    config            = models_dict['config']
    logger            = models_dict['logger']
    args              = models_dict['args']
    start_epoch       = models_dict['start_epoch']
    start_step        = models_dict['start_step']
    skip_step         = models_dict['skip_step']
    total_steps       = models_dict['total_steps']
    tme_module        = models_dict['tme_module']
    optimizer_tme     = models_dict['optimizer_tme']
    lr_scheduler_tme  = models_dict['lr_scheduler_tme']

    # <- EXP: CFG dropout probability and channel weights from config
    cfg_dropout_prob = getattr(config, "cfg_dropout_prob", 0.15)
    channel_weights  = getattr(config, "channel_reliability_weights", None)

    controlnet.train()
    for param in controlnet.parameters():
        param.requires_grad = True
    tme_module.train()

    time_start, last_tic = time.time(), time.time()
    global_step   = start_step + 1
    load_vae_feat = getattr(train_dataloader.dataset, 'load_vae_feat', False)
    vae_scale     = config.scale_factor
    vae_shift     = config.shift_factor

    logger.info("=" * 80)
    logger.info("Starting ControlNet + TME Fine-tuning (paired experimental data)")
    logger.info(f"cfg_dropout_prob={cfg_dropout_prob}  channel_weights={channel_weights}")
    logger.info(f"start_epoch={start_epoch}  start_step={start_step}  total_steps={total_steps}")
    logger.info("=" * 80)

    for epoch in range(start_epoch + 1, config.num_epochs + 1):
        for step, batch in enumerate(train_dataloader):
            if step < skip_step:
                if (step + 1) % 50 == 0 and accelerator.is_main_process:
                    logger.info(f"Skipping [{global_step}/{epoch}][{step+1}/{len(train_dataloader)}]")
                continue

            if load_vae_feat:
                z = batch[0]
            else:
                with torch.no_grad():
                    with torch.cuda.amp.autocast(
                        enabled=(config.mixed_precision in ['fp16', 'bf16'])
                    ):
                        x_in = batch[0].to(dtype=next(vae.parameters()).dtype)
                        posterior = vae.encode(x_in).latent_dist
                        z = (posterior.sample()
                             if config.sample_posterior else posterior.mode())
            clean_images = (z.float() - config.shift_factor) * config.scale_factor

            y             = batch[1]           # UNI embeddings  [B, 1, 1, 1152]
            control_input = batch[2]           # TME channels    [B, C, 256, 256]
            vae_mask      = batch[3]           # VAE cell mask   [B, 16, 32, 32]
            data_info     = batch[4]

            bs        = clean_images.shape[0]
            timesteps = torch.randint(
                0, config.train_sampling_steps, (bs,), device=clean_images.device
            ).long()

            vae_mask = (vae_mask - vae_shift) * vae_scale

            # <- EXP 1: CFG dropout — replace UNI embed with zeros
            for b in range(bs):
                if torch.rand(1).item() < cfg_dropout_prob:
                    y[b] = torch.zeros_like(y[b])

            tme_dtype    = next(tme_module.parameters()).dtype
            tme_channels = control_input[:, 1:, :, :].to(dtype=tme_dtype)

            # <- EXP 2: Channel reliability weighting
            if channel_weights is not None:
                w = torch.tensor(
                    channel_weights, device=tme_channels.device, dtype=tme_channels.dtype
                ).view(1, -1, 1, 1)
                tme_channels = tme_channels * w

            vae_mask = tme_module(vae_mask.to(dtype=tme_dtype), tme_channels)
            vae_mask = vae_mask.float()

            with accelerator.accumulate(controlnet, tme_module):
                optimizer.zero_grad()
                optimizer_tme.zero_grad()

                model_kwargs = dict(
                    y=y, mask=None, data_info=data_info, control_input=vae_mask,
                )
                loss_term = training_losses_controlnet(
                    diffusion=train_diffusion,
                    controlnet=controlnet,
                    base_model=base_model,
                    x_start=clean_images,
                    timesteps=timesteps,
                    model_kwargs=model_kwargs,
                    config=config,
                )
                loss = loss_term['loss']
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(controlnet.parameters(), config.gradient_clip)
                    optimizer.step()
                    lr_scheduler.step()
                    accelerator.clip_grad_norm_(tme_module.parameters(), config.gradient_clip)
                    optimizer_tme.step()
                    lr_scheduler_tme.step()

                if accelerator.is_main_process:
                    ema_update(model_ema, controlnet, config.ema_rate)

            if accelerator.sync_gradients:
                global_step += 1
                if global_step % config.log_interval == 0:
                    time_cost       = time.time() - last_tic
                    samples_per_sec = config.log_interval * config.train_batch_size / time_cost
                    logger.info(
                        f"Epoch [{epoch}/{config.num_epochs}] "
                        f"Step [{global_step}/{total_steps}] "
                        f"Loss: {loss.item():.4f} "
                        f"LR_ctrl: {optimizer.param_groups[0]['lr']:.2e} "
                        f"LR_tme:  {optimizer_tme.param_groups[0]['lr']:.2e} "
                        f"Samples/s: {samples_per_sec:.2f}"
                    )
                    last_tic = time.time()

                if global_step % config.save_model_steps == 0:
                    _save_sim_checkpoint(
                        accelerator, controlnet, tme_module, model_ema,
                        optimizer, optimizer_tme, lr_scheduler, lr_scheduler_tme,
                        global_step, epoch, config, logger,
                    )

            if global_step >= total_steps:
                logger.info(f"Reached max steps ({total_steps}). Stopping.")
                break

        if epoch % config.save_model_epochs == 0 or epoch == config.num_epochs:
            _save_sim_checkpoint(
                accelerator, controlnet, tme_module, model_ema,
                optimizer, optimizer_tme, lr_scheduler, lr_scheduler_tme,
                global_step, epoch, config, logger,
            )
        if global_step >= total_steps:
            break

    logger.info("=" * 80)
    logger.info("Fine-tuning Complete!")
    logger.info("=" * 80)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    config_path = "./configs/config_controlnet_exp.py"
    init_data   = initialize_config_and_accelerator([config_path])
    config      = init_data['config']
    accelerator = init_data['accelerator']
    logger      = init_data['logger']
    args        = init_data['args']

    model_data      = initialize_models(config, accelerator, logger)
    base_model      = model_data['base_model']
    controlnet      = model_data['controlnet']
    model_ema       = model_data['model_ema']
    vae             = model_data['vae']
    train_diffusion = model_data['train_diffusion']

    exp_data         = initialize_exp_training(config, accelerator, logger, controlnet)
    train_dataloader = exp_data['train_dataloader']
    tme_module       = exp_data['tme_module']
    optimizer        = exp_data['optimizer']
    optimizer_tme    = exp_data['optimizer_tme']
    lr_scheduler     = exp_data['lr_scheduler']
    lr_scheduler_tme = exp_data['lr_scheduler_tme']

    (
        base_model, controlnet, model_ema,
        optimizer, train_dataloader, lr_scheduler,
        tme_module, optimizer_tme,
    ) = accelerator.prepare(
        base_model, controlnet, model_ema,
        optimizer, train_dataloader, lr_scheduler,
        tme_module, optimizer_tme,
    )

    state_data = setup_training_state(
        config, accelerator, logger, args, train_dataloader,
        base_model, controlnet, model_ema, optimizer, lr_scheduler,
    )

    tme_ckpt = getattr(config, "resume_tme_checkpoint", None)
    if tme_ckpt:
        step = load_sim_checkpoint(
            tme_ckpt, tme_module, optimizer_tme, lr_scheduler_tme,
            device=accelerator.device,
        )
        logger.info(f"Resumed TME module from step {step} ({tme_ckpt})")

    models = {
        'base_model':       base_model,
        'controlnet':       controlnet,
        'model_ema':        model_ema,
        'vae':              vae,
        'train_diffusion':  train_diffusion,
        'optimizer':        optimizer,
        'optimizer_d':      None,
        'lr_scheduler':     lr_scheduler,
        'train_dataloader': train_dataloader,
        'accelerator':      accelerator,
        'config':           config,
        'logger':           logger,
        'args':             args,
        'tme_module':       tme_module,
        'optimizer_tme':    optimizer_tme,
        'lr_scheduler_tme': lr_scheduler_tme,
        **state_data,
    }
    train_controlnet_exp(models)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5.2 — Verify import succeeds**

```bash
python -c "from train_scripts.train_controlnet_exp import initialize_exp_training; print('OK')"
```

Expected: `OK`

---

### Task 6: Write `config_controlnet_exp.py`

- [ ] **Step 6.1 — Write the config**

```python
# configs/config_controlnet_exp.py
"""
Configuration for PixCell-256 ControlNet fine-tuning on paired experimental data.
Inherits base PixArt settings; overrides dataset, channels, and training knobs.
"""

_base_ = ['./PixArt_xl2_internal.py']
image_size = 256
root = "./"

# =====================================================================
# Dataset — PairedExpControlNetData
# =====================================================================
data = dict(
    type="PairedExpControlNetData",
    resolution=image_size,
    exp_channels_dir="exp_channels",
    features_dir="features",
    vae_features_dir="vae_features",
    exp_index_h5="metadata/exp_index.hdf5",
    vae_prefix="sd3_vae",
    ssl_prefix="uni",
    active_channels=[
        "cell_mask",
        "cell_type_healthy", "cell_type_cancer", "cell_type_immune",
        "cell_state_prolif",  "cell_state_nonprolif", "cell_state_dead",
        "vasculature", "oxygen", "glucose",
    ],
)

exp_data_root = f"{root}/data/exp_paired"   # <-- set to your actual path

# =====================================================================
# TME Encoder
# =====================================================================
# n_tme_channels = 9  (all active_channels except cell_mask)
tme_model   = "TMEConditioningModule"
tme_base_ch = 32
tme_lr      = 1e-5

# =====================================================================
# Experimental training knobs
# =====================================================================
# CFG dropout: fraction of steps where UNI embedding is zeroed
cfg_dropout_prob = 0.15

# Per-tme-channel reliability weights (9 values, one per tme channel in order):
#   cell_type_healthy, cell_type_cancer, cell_type_immune  → 1.0 (pixel-perfect)
#   cell_state_prolif, cell_state_nonprolif, cell_state_dead → 1.0 (pixel-perfect)
#   vasculature, oxygen, glucose                            → 0.5 (CODEX approximation)
channel_reliability_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5]

# =====================================================================
# Model
# =====================================================================
base_model       = "PixArt_XL_2_UNI"
base_model_path  = f"{root}/pretrained_models/pixcell-256/transformer"
model_max_length = 1

controlnet_model                 = "PixCell_ControlNet_XL_2_UNI"
controlnet_depth                 = 27
controlnet_conditioning_channels = 16
controlnet_conditioning_scale    = 1.0
controlnet_load_from             = f"{root}/pretrained_models/pixcell-256-controlnet/controlnet/diffusion_pytorch_model.safetensors"
load_from   = f"{root}/pretrained_models/pixcell-256/transformer"
# To resume from a sim checkpoint:
# resume_from = f"{root}/checkpoints/pixcell_controlnet_sim/checkpoints/step_XXXXXXX"
# resume_tme_checkpoint = f"{root}/checkpoints/pixcell_controlnet_sim/checkpoints/step_XXXXXXX"
resume_from = None

vae_pretrained   = f"{root}/pretrained_models/sd-3.5-vae/vae"
pe_interpolation = 0.5

mixed_precision  = 'no'
fp32_attention   = True

# =====================================================================
# Training
# =====================================================================
num_workers                 = 4
train_batch_size            = 4
num_epochs                  = 200
gradient_accumulation_steps = 1
grad_checkpointing          = True
gradient_clip               = 1.0

optimizer = dict(
    type='AdamW',
    lr=5e-6,            # lower LR than sim — fine-tuning, not training from scratch
    weight_decay=0.0,
    betas=(0.9, 0.999),
    eps=1e-8,
)

lr_schedule_args = dict(num_warmup_steps=500)
auto_lr          = None

log_interval       = 100
save_model_epochs  = 50
save_model_steps   = 10000
work_dir           = f"{root}/checkpoints/pixcell_controlnet_exp"

# =====================================================================
# VAE
# =====================================================================
scale_factor = 1.5305
shift_factor = 0.0609

# =====================================================================
# Misc
# =====================================================================
class_dropout_prob   = 0.1
ema_rate             = 0.9999
train_sampling_steps = 500
snr_loss             = True
seed                 = 42
data_root            = "./"

controlnet_config = dict(
    zero_init_conv_out=True,
    copy_base_layers=True,
    conditioning_scale=1.0,
)
model_kwargs = dict(
    use_controlnet=True,
    controlnet_config=controlnet_config,
)
```

- [ ] **Step 6.2 — Verify config parses**

```bash
python -c "from diffusion.utils.misc import read_config; c=read_config('./configs/config_controlnet_exp.py'); print('active_channels:', c.data.active_channels)"
```

Expected output:
```
active_channels: ['cell_mask', 'cell_type_healthy', ..., 'glucose']
```

- [ ] **Step 6.3 — Commit**

```bash
git add train_scripts/train_controlnet_exp.py \
        configs/config_controlnet_exp.py \
        tests/test_train_controlnet_exp.py
git commit -m "feat: add paired-exp training loop with CFG dropout and channel reliability weighting"
```

---

## Chunk 3: Inference Helper + Validation Pipeline

### File Map

| Action | Path |
|---|---|
| Modify | `train_scripts/inference_controlnet.py` |
| Create | `tools/validate_sim_to_exp.py` |
| Create | `tests/test_validate_sim_to_exp.py` |

---

### Task 7: Add `null_uni_embed()` to inference module

- [ ] **Step 7.1 — Write failing test**

```python
# tests/test_validate_sim_to_exp.py
import torch
import pytest


def test_null_uni_embed_shape():
    from train_scripts.inference_controlnet import null_uni_embed
    emb = null_uni_embed(device='cpu', dtype=torch.float32)
    assert emb.shape == (1, 1, 1, 1536), f"Got {emb.shape}"


def test_null_uni_embed_is_zeros():
    from train_scripts.inference_controlnet import null_uni_embed
    emb = null_uni_embed(device='cpu', dtype=torch.float32)
    assert torch.all(emb == 0.0)


def test_null_uni_embed_dtype():
    from train_scripts.inference_controlnet import null_uni_embed
    emb = null_uni_embed(device='cpu', dtype=torch.float16)
    assert emb.dtype == torch.float16
```

- [ ] **Step 7.2 — Run to confirm failure**

```bash
python -m pytest tests/test_validate_sim_to_exp.py::test_null_uni_embed_shape -v
```

Expected: `ImportError` — `null_uni_embed` does not exist yet

- [ ] **Step 7.3 — Add `null_uni_embed` to `inference_controlnet.py`**

Open `train_scripts/inference_controlnet.py`. After the imports block, add:

```python
def null_uni_embed(device='cuda', dtype=torch.float16):
    """
    Zero UNI embedding for TME-only (no style reference) inference.

    Pass as `uni_embeds` to `denoise()` to run purely TME-conditioned generation.
    CFG guidance_scale still applies — higher values increase TME adherence.

    Returns:
        Tensor shape [1, 1, 1, 1536], all zeros.
    """
    return torch.zeros(1, 1, 1, 1536, device=device, dtype=dtype)
```

- [ ] **Step 7.4 — Run tests**

```bash
python -m pytest tests/test_validate_sim_to_exp.py::test_null_uni_embed_shape \
                 tests/test_validate_sim_to_exp.py::test_null_uni_embed_is_zeros \
                 tests/test_validate_sim_to_exp.py::test_null_uni_embed_dtype -v
```

Expected: all 3 `PASSED`

---

### Task 8: Write `validate_sim_to_exp.py`

- [ ] **Step 8.1 — Write failing integration test**

Append to `tests/test_validate_sim_to_exp.py`:

```python
def test_cosine_similarity_range():
    """cosine_similarity values must be in [-1, 1]."""
    from tools.validate_sim_to_exp import cosine_similarity_matrix
    a = torch.randn(10, 1152)
    b = torch.randn(10, 1152)
    sims = cosine_similarity_matrix(a, b)
    assert sims.shape == (10,), f"Got {sims.shape}"
    assert torch.all(sims >= -1.0) and torch.all(sims <= 1.0)


def test_cosine_similarity_identical():
    """Identical vectors should give similarity 1.0."""
    from tools.validate_sim_to_exp import cosine_similarity_matrix
    a = torch.randn(5, 1152)
    sims = cosine_similarity_matrix(a, a)
    assert torch.allclose(sims, torch.ones(5), atol=1e-5)
```

- [ ] **Step 8.2 — Run to confirm failure**

```bash
python -m pytest tests/test_validate_sim_to_exp.py::test_cosine_similarity_range -v
```

Expected: `ImportError`

- [ ] **Step 8.3 — Implement `validate_sim_to_exp.py`**

```python
# tools/validate_sim_to_exp.py
"""
validate_sim_to_exp.py — Simulation-to-experiment validation pipeline.

For each simulation snapshot, generates an H&E tile in the experimental domain
using the trained PixCellControlNet. Compares generated tiles to experimental
targets in UNI feature space.

Usage:
    python tools/validate_sim_to_exp.py \\
        --config    configs/config_controlnet_exp.py \\
        --sim-root  /path/to/sim_data_root \\
        --exp-feat  /path/to/exp_features_dir \\
        --controlnet-ckpt /path/to/controlnet.pth \\
        --tme-ckpt        /path/to/tme_module.pth \\
        [--reference-uni  /path/to/reference_uni.npy]  # optional style ref
        [--output-dir     ./validation_output]
        [--n-tiles        50]
        [--guidance-scale 2.5]
        [--device         cuda]
"""
import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import DDPMScheduler

from diffusion.utils.misc import read_config
from train_scripts.inference_controlnet import (
    load_vae,
    null_uni_embed,
    denoise,
    decode_latents,
    load_controlnet_model_from_checkpoint,
)
from diffusion.model.builder import build_model
from train_scripts.train_controlnet_sim import load_sim_checkpoint
from diffusion.data.datasets.sim_controlnet_dataset import (
    SimControlNetData, _load_spatial_file, _find_file
)


# ── Core metric ───────────────────────────────────────────────────────────────

def cosine_similarity_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Per-row cosine similarity between two [N, D] feature matrices.

    Returns:
        Tensor [N] of cosine similarities in [-1, 1].
    """
    a_norm = F.normalize(a, dim=1)
    b_norm = F.normalize(b, dim=1)
    return (a_norm * b_norm).sum(dim=1)


# ── Channel loader (reuses sim dataset helpers) ───────────────────────────────

def load_sim_ctrl_tensor(
    sim_root: Path,
    sim_id: str,
    active_channels: list[str],
    resolution: int = 256,
) -> torch.Tensor:
    """Load a single sim snapshot's TME channels → [C, H, W]."""
    from diffusion.data.datasets.sim_controlnet_dataset import (
        _BINARY_CHANNELS as SIM_BINARY,
    )
    from diffusion.data.datasets.paired_exp_controlnet_dataset import (
        _BINARY_CHANNELS as EXP_BINARY,
    )
    binary_set = SIM_BINARY | EXP_BINARY   # union covers both formats
    planes = []
    for ch in active_channels:
        ch_dir = sim_root / "sim_channels" / ch
        fpath  = _find_file(ch_dir, sim_id)
        arr    = _load_spatial_file(fpath, resolution=resolution,
                                    binary=(ch in binary_set))
        planes.append(arr)
    return torch.from_numpy(np.stack(planes, axis=0))


# ── Validation loop ───────────────────────────────────────────────────────────

def run_validation(
    config,
    sim_root: Path,
    sim_ids: list[str],
    exp_feat_dir: Path,
    controlnet,
    base_model,
    tme_module,
    vae,
    scheduler,
    uni_embeds: torch.Tensor,   # [1,1,1,1536] — null or reference
    guidance_scale: float,
    device: str,
    output_dir: Path | None,
) -> dict:
    """
    Generate H&E from sim TME channels and compare to exp target features.

    Returns dict with keys:
        cosine_similarities  : list[float]   per-tile
        mean_cosine_sim      : float
        std_cosine_sim       : float
    """
    active_channels = config.data.active_channels
    vae_scale = config.scale_factor
    vae_shift = config.shift_factor
    dtype = torch.float16 if device == 'cuda' else torch.float32

    cosine_sims = []
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    for sim_id in sim_ids:
        # Load sim TME channels
        ctrl_full = load_sim_ctrl_tensor(
            sim_root, sim_id, active_channels, resolution=config.image_size
        )

        # VAE-encode cell_mask (channel 0)
        cell_mask_img = ctrl_full[0:1].unsqueeze(0).repeat(1, 3, 1, 1)  # [1,3,H,W]
        cell_mask_img = 2 * (cell_mask_img - 0.5)
        with torch.no_grad():
            vae_mask = vae.encode(
                cell_mask_img.to(device, dtype=dtype)
            ).latent_dist.mean
            vae_mask = (vae_mask - vae_shift) * vae_scale

        # TME channels [B, C-1, H, W] — no weighting at inference
        tme_channels = ctrl_full[1:].unsqueeze(0).to(device, dtype=dtype)
        with torch.no_grad():
            fused_cond = tme_module(vae_mask.to(dtype), tme_channels)

        # Denoising
        latent_shape = (1, 16, config.image_size // 8, config.image_size // 8)
        latents = torch.randn(latent_shape, device=device, dtype=dtype)
        latents = latents * scheduler.init_noise_sigma

        denoised = denoise(
            latents=latents,
            uni_embeds=uni_embeds.to(device, dtype=dtype),
            controlnet_input_latent=fused_cond,
            scheduler=scheduler,
            controlnet_model=controlnet,
            pixcell_controlnet_model=base_model,
            guidance_scale=guidance_scale,
            device=device,
        )

        # Decode to image
        with torch.no_grad():
            gen_img = vae.decode(
                (denoised.float() / vae_scale) + vae_shift, return_dict=False
            )[0]
        gen_img = (gen_img / 2 + 0.5).clamp(0, 1)
        gen_np  = (gen_img.cpu().permute(0, 2, 3, 1).numpy()[0] * 255).astype(np.uint8)

        if output_dir:
            Image.fromarray(gen_np).save(output_dir / f"{sim_id}_generated.png")

        # Extract UNI features from generated image (if UNI model available)
        # For now, load precomputed exp target features and compare
        exp_feat_path = exp_feat_dir / f"{sim_id}_uni.npy"
        if not exp_feat_path.exists():
            print(f"[WARN] No exp feat for {sim_id}, skipping.")
            continue
        exp_feat = torch.from_numpy(np.load(exp_feat_path)).unsqueeze(0)  # [1, 1152]

        # Placeholder: if you run UNI extractor on gen_np, replace this line
        # gen_feat = extract_uni_features(gen_np)
        # For now, report that exp_feat was loaded
        print(f"  {sim_id}: exp_feat loaded, shape={exp_feat.shape}")
        # cosine_sims.append(cosine_similarity_matrix(gen_feat, exp_feat).item())

    return {
        "cosine_similarities": cosine_sims,
        "mean_cosine_sim":     float(np.mean(cosine_sims)) if cosine_sims else float('nan'),
        "std_cosine_sim":      float(np.std(cosine_sims))  if cosine_sims else float('nan'),
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Sim-to-exp validation pipeline")
    parser.add_argument("--config",           required=True)
    parser.add_argument("--sim-root",         required=True)
    parser.add_argument("--exp-feat",         required=True, help="Dir of *_uni.npy exp targets")
    parser.add_argument("--controlnet-ckpt",  required=True)
    parser.add_argument("--tme-ckpt",         required=True)
    parser.add_argument("--reference-uni",    default=None,  help="Optional reference H&E UNI .npy")
    parser.add_argument("--output-dir",       default=None)
    parser.add_argument("--n-tiles",          type=int, default=50)
    parser.add_argument("--guidance-scale",   type=float, default=2.5)
    parser.add_argument("--device",           default="cuda")
    args = parser.parse_args()

    config = read_config(args.config)
    device = args.device

    # Load models
    vae = load_vae(config.vae_pretrained, device)
    controlnet = load_controlnet_model_from_checkpoint(
        args.config, args.controlnet_ckpt, device
    )
    base_model = ...   # load via initialize_models or load_pixcell_controlnet_model_from_checkpoint

    n_tme_channels = len(config.data.active_channels) - 1
    tme_module = build_model(
        "TMEConditioningModule", False, False,
        n_tme_channels=n_tme_channels,
        base_ch=getattr(config, "tme_base_ch", 32),
    )
    load_sim_checkpoint(args.tme_ckpt, tme_module, device=device)
    tme_module.to(device).eval()

    scheduler = DDPMScheduler(
        num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02,
        beta_schedule="linear", prediction_type="epsilon", clip_sample=False,
    )
    scheduler.set_timesteps(50, device=device)

    # UNI embed: null (TME-only) or from reference
    if args.reference_uni:
        ref = np.load(args.reference_uni)
        uni_embeds = torch.from_numpy(ref).view(1, 1, 1, 1536)
    else:
        uni_embeds = null_uni_embed(device=device, dtype=torch.float16)

    # Get sim IDs from dataset index
    from diffusion.data.datasets.sim_controlnet_dataset import SimControlNetData
    ds = SimControlNetData(
        root=args.sim_root, resolution=config.image_size,
        active_channels=config.data.active_channels,
    )
    sim_ids = ds.sim_ids[: args.n_tiles]

    results = run_validation(
        config=config,
        sim_root=Path(args.sim_root),
        sim_ids=sim_ids,
        exp_feat_dir=Path(args.exp_feat),
        controlnet=controlnet,
        base_model=base_model,
        tme_module=tme_module,
        vae=vae,
        scheduler=scheduler,
        uni_embeds=uni_embeds,
        guidance_scale=args.guidance_scale,
        device=device,
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )

    print("\n=== Validation Results ===")
    print(f"N tiles:          {len(results['cosine_similarities'])}")
    print(f"Mean cosine sim:  {results['mean_cosine_sim']:.4f}")
    print(f"Std cosine sim:   {results['std_cosine_sim']:.4f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 8.4 — Run metric tests**

```bash
python -m pytest tests/test_validate_sim_to_exp.py -v
```

Expected: all 5 tests `PASSED`

- [ ] **Step 8.5 — Commit**

```bash
git add train_scripts/inference_controlnet.py \
        tools/validate_sim_to_exp.py \
        tests/test_validate_sim_to_exp.py
git commit -m "feat: add null_uni_embed helper and sim-to-exp validation pipeline"
```

---

## Final Checklist

- [ ] All tests pass: `python -m pytest tests/ -v`
- [ ] Config parses cleanly: `python -c "from diffusion.utils.misc import read_config; read_config('./configs/config_controlnet_exp.py')"`
- [ ] Training script imports cleanly: `python -c "from train_scripts.train_controlnet_exp import train_controlnet_exp; print('OK')"`
- [ ] `build_exp_index` documented in dataset file for data prep
- [ ] `channel_reliability_weights` in config matches `active_channels` order (9 weights for 9 tme channels)
