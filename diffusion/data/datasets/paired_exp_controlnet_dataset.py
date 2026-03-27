"""
paired_exp_controlnet_dataset.py

PairedExpControlNetData — paired experimental H&E + CODEX-derived TME channels.

Unlike SimControlNetData (unpaired), each tile provides:
    - H&E VAE latent     (paired — same tile_id)
    - UNI embedding      (paired — same tile_id)
    - CODEX TME channels (paired — same tile_id)
    - cell_masks VAE latent (paired — same tile_id)

Directory layout under `root`:
    exp_data_root/
    ├── metadata/
    │   └── exp_index.hdf5          # flat list of paired tile IDs
    ├── exp_channels/               # one sub-folder per channel (same format as sim_channels)
    │   ├── cell_masks/              # binary PNG (required)
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
    │   └── {tile_id}_uni.npy       # shape [1536]  paired H&E UNI-2h embedding
    └── vae_features/
        ├── {tile_id}_sd3_vae.npy           # shape [32, lt_sz, lt_sz]  (mean+std)
        └── {tile_id}_mask_sd3_vae.npy      # shape [32, lt_sz, lt_sz]  (mean+std)
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from diffusers.utils.torch_utils import randn_tensor
from torch.utils.data import Dataset

# Re-use file I/O helpers from sim dataset
from diffusion.data.datasets.sim_controlnet_dataset import (
    _find_file,
    _load_spatial_file,
    _write_h5_index,
)

# ── Channel registry ──────────────────────────────────────────────────────────

REQUIRED_CHANNELS: list[str] = [
    "cell_masks",
    "cell_type_healthy", "cell_type_cancer", "cell_type_immune",
    "cell_state_prolif", "cell_state_nonprolif", "cell_state_dead",
]
OPTIONAL_CHANNELS: list[str] = ["vasculature", "oxygen", "glucose"]
ALL_CHANNELS: list[str] = REQUIRED_CHANNELS + OPTIONAL_CHANNELS

_BINARY_CHANNELS: frozenset[str] = frozenset({
    "cell_masks",
    "cell_type_healthy", "cell_type_cancer", "cell_type_immune",
    "cell_state_prolif", "cell_state_nonprolif", "cell_state_dead",
})
_CHANNEL_ALIASES: dict[str, str] = {
    "cell_mask": "cell_masks",
}

# Reflect-pad non-binary channels by this many pixels before resize to suppress
# simulation boundary artifacts (Dirichlet BCs create sharp border gradients that
# the CNN encoder mistakes for signal).
_MIRROR_BORDER_PX: int = 8


def _normalize_channels(active: list[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for channel in active:
        name = _CHANNEL_ALIASES.get(channel, channel)
        if name not in seen:
            ordered.append(name)
            seen.add(name)
    return ordered


def _validate_channels(active: list[str]) -> list[str]:
    active = _normalize_channels(active)
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
        ssl_feat    : [1, 1, 1536]                     paired H&E UNI-2h embedding
        ctrl_tensor : [C, resolution, resolution]     CODEX TME channels
        vae_mask    : [16, lt_sz, lt_sz]               paired cell_mask VAE latent
        data_info   : dict (includes 'tile_id' for debugging)

    Args:
        root (str):            exp_data_root/ directory.
        resolution (int):      Spatial resolution (256).
        active_channels (list[str]):
            Which channels to include. Required: all 7 binary channels listed above.
            Optional: ["vasculature", "oxygen", "glucose"].
        exp_channels_dir (str):  sub-folder for channel PNGs. Default "exp_channels".
        features_dir (str):      sub-folder for UNI .npy files. Default "features".
        vae_features_dir (str):  sub-folder for VAE .npy files. Default "vae_features".
        exp_index_h5 (str):      HDF5 index path. Default "metadata/exp_index.hdf5".
        vae_prefix (str):        VAE latent filename suffix. Default "sd3_vae".
        ssl_prefix (str):        UNI embedding filename suffix. Default "uni".
        train_subset_keys (list[str] | None): restrict to specific HDF5 keys.
        max_train_samples (int | None): cap dataset size (useful for debug overfitting runs).
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

        max_samples = kwargs.get("max_train_samples", None)
        if max_samples is not None:
            self.tile_ids = self.tile_ids[:max_samples]

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
            is_binary = ch in _BINARY_CHANNELS
            arr    = _load_spatial_file(
                fpath,
                resolution=self.resolution,
                binary=is_binary,
                mirror_border_px=0 if is_binary else _MIRROR_BORDER_PX,
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
        return feat.view(1, 1, -1)   # [1, 1, D] where D=1536 for UNI-2h

    def __len__(self) -> int:
        return len(self.tile_ids)

    def __getitem__(self, idx: int):
        tile_id     = self.tile_ids[idx]
        ctrl_tensor = self._build_ctrl_tensor(tile_id)
        vae_feat    = self._load_vae_feat(tile_id)
        vae_mask    = self._load_vae_mask(tile_id)
        ssl_feat    = self._load_ssl_feat(tile_id)

        # Accelerate's dataloader concatenates all dict values across workers
        # and will crash on str/list types — only tensors allowed here.
        data_info = {
            "img_hw":       torch.tensor([self.resolution] * 2, dtype=torch.float32),
            "aspect_ratio": torch.tensor(1.0),
            "tile_idx":     torch.tensor(idx, dtype=torch.int64),
        }

        return vae_feat, ssl_feat, ctrl_tensor, vae_mask, data_info

    def get_ids(self, idx: int) -> dict[str, str]:
        """Debug helper: return IDs for a given dataset index (no tensors)."""
        return {"tile_id": self.tile_ids[idx]}


def build_exp_index(
    exp_channels_dir: str | Path,
    output_path: str | Path,
    resolution: int = 256,
    key_name: str | None = None,
):
    """
    Build HDF5 index for paired exp tiles by scanning exp_channels/cell_masks/.

    Run once before training to create metadata/exp_index.hdf5.

    Example:
        build_exp_index(
            "exp_data_root/exp_channels",
            "exp_data_root/metadata/exp_index.hdf5",
        )
    """
    mask_dir = Path(exp_channels_dir) / "cell_masks"
    if not mask_dir.exists():
        raise RuntimeError(f"cell_masks directory not found: {mask_dir}")
    files = sorted(p for ext in ("*.png", "*.npy") for p in mask_dir.glob(ext))
    if not files:
        raise RuntimeError(f"No PNG/NPY files found in {mask_dir}")
    tile_ids = [f.stem for f in files]
    _write_h5_index(Path(output_path), key_name or f"exp_{resolution}", tile_ids)
    print(f"[build_exp_index] Wrote {len(tile_ids)} IDs → {output_path}")
