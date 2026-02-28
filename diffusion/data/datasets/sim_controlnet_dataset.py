"""
SimControlNetData - Dataset for unpaired simulation-to-histology ControlNet training.

Loads simulation channels DIRECTLY from PNG or NPY files — no pre-rendering needed.

─────────────────────────────────────────────────────────────────────────────────
Directory layout expected under `root`:
─────────────────────────────────────────────────────────────────────────────────
    sim_data_root/
    ├── metadata/
    │   ├── sim_index.hdf5          # sim snapshot IDs  (built by build_sim_index)
    │   └── real_index.hdf5         # real H&E tile IDs (built by build_real_index)
    │
    ├── sim_channels/               # One sub-folder per channel
    │   ├── cell_mask/
    │   │   ├── {sim_id}.png        # binary or grayscale PNG  (REQUIRED)
    │   │   └── ...
    │   ├── oxygen/
    │   │   ├── {sim_id}.png/.npy   # grayscale PNG or .npy    (REQUIRED)
    │   │   └── ...
    │   ├── glucose/
    │   │   └── {sim_id}.png/.npy   (OPTIONAL)
    │   └── tgf/
    │       └── {sim_id}.png/.npy   (OPTIONAL)
    │
    ├── features/                   # UNI-2h embeddings from REAL unpaired H&E tiles
    │   └── {tile_id}_uni.npy       # shape [1152]
    │
    └── vae_features/               # SD3 VAE latents from REAL unpaired H&E tiles
        ├── {tile_id}_sd3_vae.npy           # shape [32, lt_sz, lt_sz] (mean+std cat on ch)
        └── {tile_id}_mask_sd3_vae.npy      # (optional) VAE-encoded cell mask

─────────────────────────────────────────────────────────────────────────────────
Unpaired training design:
─────────────────────────────────────────────────────────────────────────────────
    Sim channels  → tells model WHAT layout/TME to generate
    UNI embedding → tells model WHAT H&E style to use
    These are randomly paired at training time — no matched pairs needed.

─────────────────────────────────────────────────────────────────────────────────
Channel config:
─────────────────────────────────────────────────────────────────────────────────
    REQUIRED : ["cell_mask", "oxygen"]
    OPTIONAL : ["glucose", "tgf"]
    Set active_channels at init. Required channels are always placed first.
    The control tensor shape is [C, H, W] where C = len(active_channels).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from diffusers.utils.torch_utils import randn_tensor
from torch.utils.data import Dataset

# ── Channel registry ──────────────────────────────────────────────────────────

REQUIRED_CHANNELS: list[str] = ["cell_mask", "oxygen"]
OPTIONAL_CHANNELS: list[str] = ["glucose", "tgf"]
ALL_CHANNELS:      list[str] = REQUIRED_CHANNELS + OPTIONAL_CHANNELS

# Channels that should be thresholded to {0, 1} rather than normalized
_BINARY_CHANNELS: frozenset[str] = frozenset({"cell_mask"})

# Supported file extensions in search priority order
_EXTS = (".png", ".npy", ".jpg", ".tif")


# ── Validation ────────────────────────────────────────────────────────────────

def _validate_channels(active: list[str]) -> list[str]:
    """Return ordered channel list: required first, then optional in user order."""
    unknown = set(active) - set(ALL_CHANNELS)
    if unknown:
        raise ValueError(f"Unknown channels: {unknown}. Available: {ALL_CHANNELS}")
    for ch in REQUIRED_CHANNELS:
        if ch not in active:
            raise ValueError(
                f"Channel '{ch}' is required but missing from "
                f"active_channels={active}. Required: {REQUIRED_CHANNELS}"
            )
    # Required channels always first, then optional in the order the user specified
    ordered  = [ch for ch in REQUIRED_CHANNELS if ch in active]
    ordered += [ch for ch in active if ch not in REQUIRED_CHANNELS]
    return ordered


# ── File I/O ──────────────────────────────────────────────────────────────────

def _find_file(directory: Path, stem: str) -> Path:
    """
    Find a file in `directory` named `{stem}{ext}` for ext in _EXTS.
    Raises FileNotFoundError if none found.
    """
    for ext in _EXTS:
        p = directory / f"{stem}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(
        f"No file found for '{stem}' in {directory}. "
        f"Tried: {[stem + e for e in _EXTS]}"
    )


def _load_spatial_file(path: Path, resolution: int, binary: bool = False) -> np.ndarray:
    """
    Load a 2D spatial field from PNG or NPY and return a float32 [H, W] array.

    Handles:
        - PNG (binary, grayscale, or RGB)  → grayscale float32, normalized to [0,1]
        - NPY of shape (H,W), (1,H,W), (1,1,H,W), or (C,H,W) → first channel used
        - Any input resolution             → bilinear resize to (resolution, resolution)
        - binary=True                      → threshold at 0.5 → {0.0, 1.0}
        - binary=False                     → min-max normalize to [0, 1]

    Args:
        path:       File to load (.png or .npy).
        resolution: Target spatial resolution.
        binary:     Whether to binarize (True for cell_mask).

    Returns:
        np.ndarray, shape (resolution, resolution), dtype float32.
    """
    suffix = path.suffix.lower()

    # ── Load raw array ────────────────────────────────────────────────────────
    if suffix == ".npy":
        arr = np.load(path).astype(np.float32)
        # Squeeze leading size-1 dims: (1,H,W) or (1,1,H,W) → (H,W)
        while arr.ndim > 2 and arr.shape[0] == 1:
            arr = arr.squeeze(0)
        if arr.ndim == 3:
            arr = arr[0]   # (C,H,W) multi-channel → take first channel
        if arr.ndim != 2:
            raise ValueError(
                f"Cannot reduce array from {path} to 2D. "
                f"Got shape {arr.shape} after squeezing."
            )

    elif suffix in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
        raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise FileNotFoundError(f"cv2 could not read: {path}")
        if raw.ndim == 3:
            raw = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        arr = raw.astype(np.float32)
        # Normalize integer types to [0, 1]
        if arr.max() > 1.0:
            max_val = float(np.iinfo(raw.dtype).max) if raw.dtype.kind == "u" else 255.0
            arr /= max_val

    else:
        raise ValueError(
            f"Unsupported file extension '{suffix}'. "
            f"Supported: {list(_EXTS)}"
        )

    # ── Resize ────────────────────────────────────────────────────────────────
    if arr.shape != (resolution, resolution):
        arr = cv2.resize(arr, (resolution, resolution), interpolation=cv2.INTER_LINEAR)

    # ── Normalize or binarize ─────────────────────────────────────────────────
    if binary:
        arr = (arr > 0.5).astype(np.float32)
    else:
        vmin, vmax = arr.min(), arr.max()
        if vmax > vmin:
            arr = (arr - vmin) / (vmax - vmin)
        # If constant field: leave as-is (already 0.0 everywhere)

    return arr


# ── Dataset ───────────────────────────────────────────────────────────────────

class SimControlNetData(Dataset):
    """
    Dataset for PixCell-256 ControlNet training with simulation inputs (unpaired).

    Returns the same tuple as PanCancerControlNetData:
        (vae_feat, ssl_feat, ctrl_tensor, vae_mask, data_info)

    where:
        vae_feat    : [16, lt_sz, lt_sz]              real H&E VAE latent (unpaired)
        ssl_feat    : [1, 1, 1152]                     real H&E UNI-2h embedding (unpaired)
        ctrl_tensor : [C, resolution, resolution]     sim control signal
        vae_mask    : [16, lt_sz, lt_sz]               VAE-encoded mask (zeros if absent)
        data_info   : dict

    Args:
        root (str):
            Root of the dataset directory (sim_data_root/).
        resolution (int):
            Spatial resolution. Must match your PixCell model (256).
        active_channels (list[str]):
            Which channels to include.
            Required (always): ["cell_mask", "oxygen"].
            Optional additions: ["glucose", "tgf"].
            Required channels are always placed first in the tensor.
        sim_channels_dir (str):
            Path relative to root for sim channel sub-folders. Default: "sim_channels".
        features_dir (str):
            Path relative to root for UNI .npy files. Default: "features".
        vae_features_dir (str):
            Path relative to root for VAE latent .npy files. Default: "vae_features".
        sim_index_h5 (str):
            HDF5 index for sim IDs. Default: "metadata/sim_index.hdf5".
        real_index_h5 (str):
            HDF5 index for real tile IDs. Default: "metadata/real_index.hdf5".
        vae_prefix (str):
            Filename suffix for VAE latent files. Default: "sd3_vae".
        ssl_prefix (str):
            Filename suffix for UNI embedding files. Default: "uni".
        train_subset_keys (list[str] | None):
            Restrict to specific HDF5 keys (e.g. for held-out splits). Default: None.
        seed (int):
            RNG seed for random real-tile sampling. Default: 42.
    """

    def __init__(self, root: str, resolution: int, **kwargs):
        self.root       = Path(root)
        self.resolution = resolution
        self.lt_sz      = resolution // 8   # VAE latent spatial dim (32 for res=256)

        # ── Channels ───────────────────────────────────────────────────────────
        raw_channels         = kwargs.get("active_channels", list(REQUIRED_CHANNELS))
        self.active_channels = _validate_channels(list(raw_channels))
        self.n_channels      = len(self.active_channels)

        # ── Paths ──────────────────────────────────────────────────────────────
        self.sim_channels_dir = self.root / kwargs.get("sim_channels_dir", "sim_channels")
        self.features_dir     = self.root / kwargs.get("features_dir",     "features")
        self.vae_features_dir = self.root / kwargs.get("vae_features_dir", "vae_features")
        self.vae_prefix       = kwargs.get("vae_prefix", "sd3_vae")
        self.ssl_prefix       = kwargs.get("ssl_prefix", "uni")

        # ── Indices ────────────────────────────────────────────────────────────
        sim_h5  = self.root / kwargs.get("sim_index_h5",  "metadata/sim_index.hdf5")
        real_h5 = self.root / kwargs.get("real_index_h5", "metadata/real_index.hdf5")
        subset  = kwargs.get("train_subset_keys", None)

        self.sim_ids  = self._load_index(sim_h5,  resolution, subset)
        self.real_ids = self._load_index(real_h5, resolution, subset)

        if not self.sim_ids:
            raise RuntimeError(f"No simulation IDs found in {sim_h5}")
        if not self.real_ids:
            raise RuntimeError(f"No real tile IDs found in {real_h5}")

        self.load_vae_feat = True   # batch[0] is a VAE latent, not a raw image
        self._rng = np.random.default_rng(seed=kwargs.get("seed", 42))

        print(
            f"[SimControlNetData] "
            f"{len(self.sim_ids)} sim snapshots | "
            f"{len(self.real_ids)} real tiles | "
            f"channels={self.active_channels} | "
            f"resolution={resolution}"
        )

    # ── Index helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _load_index(
        h5_path: Path,
        resolution: int,
        subset_keys: list[str] | None,
    ) -> list[str]:
        """
        Load a flat list of IDs from an HDF5 index file.
        Keys must contain f'_{resolution}' to be included (e.g. 'sim_256').
        Multiple matching keys are concatenated.
        """
        ids: list[str] = []
        with h5py.File(h5_path, "r") as h5:
            keys = [k for k in h5.keys() if f"_{resolution}" in k]
            if subset_keys is not None:
                keys = [k for k in keys if k in subset_keys]
            for key in keys:
                ids.extend(v.decode("utf-8") for v in h5[key])
        return ids

    # ── Control tensor builder ─────────────────────────────────────────────────

    def _build_ctrl_tensor(self, sim_id: str) -> torch.Tensor:
        """
        Load all active channels for sim_id and stack → [C, H, W] float32 tensor.
        Files are found in sim_channels/{channel_name}/{sim_id}.{ext}.
        """
        planes: list[np.ndarray] = []
        for ch in self.active_channels:
            ch_dir = self.sim_channels_dir / ch
            fpath  = _find_file(ch_dir, sim_id)
            arr    = _load_spatial_file(
                fpath,
                resolution=self.resolution,
                binary=(ch in _BINARY_CHANNELS),
            )
            planes.append(arr)

        ctrl = torch.from_numpy(np.stack(planes, axis=0))  # [C, H, W]
        assert ctrl.shape == (self.n_channels, self.resolution, self.resolution), \
            f"ctrl_tensor shape error: got {ctrl.shape}, " \
            f"expected ({self.n_channels}, {self.resolution}, {self.resolution})"
        return ctrl

    # ── VAE / SSL loaders ──────────────────────────────────────────────────────

    def _load_vae_feat(self, tile_id: str) -> torch.Tensor:
        """Sample from VAE latent distribution → [16, lt_sz, lt_sz]."""
        path       = self.vae_features_dir / f"{tile_id}_{self.vae_prefix}.npy"
        arr        = torch.from_numpy(np.load(path))
        mean, std  = arr.chunk(2, dim=0)
        if mean.ndim == 4 and mean.shape[0] == 1:
            mean, std = mean.squeeze(0), std.squeeze(0)
        sample = randn_tensor(mean.shape, device=mean.device, dtype=mean.dtype)
        feat   = mean + std * sample
        assert feat.shape == (16, self.lt_sz, self.lt_sz), \
            f"vae_feat shape error: {feat.shape}"
        return feat

    def _load_vae_mask(self, tile_id: str) -> torch.Tensor:
        """Load VAE-encoded mask → [16, lt_sz, lt_sz]. Zeros if file absent."""
        path = self.vae_features_dir / f"{tile_id}_mask_{self.vae_prefix}.npy"
        if not path.exists():
            return torch.zeros(16, self.lt_sz, self.lt_sz)
        arr      = torch.from_numpy(np.load(path))
        mean, _  = arr.chunk(2, dim=0)
        if mean.ndim == 4 and mean.shape[0] == 1:
            mean = mean.squeeze(0)
        assert mean.shape == (16, self.lt_sz, self.lt_sz), \
            f"vae_mask shape error: {mean.shape}"
        return mean

    def _load_ssl_feat(self, tile_id: str) -> torch.Tensor:
        """Load UNI-2h embedding → [1, 1, 1152] (required by CaptionEmbedder)."""
        path = self.features_dir / f"{tile_id}_{self.ssl_prefix}.npy"
        feat = torch.from_numpy(np.load(path))   # [1152] or [1,1152]
        # CaptionEmbedder asserts caption.shape[2:] == (token_num, in_channels)
        # so per-sample shape must be [1, 1, 1152] → batched: [B, 1, 1, 1152]
        feat = feat.view(1, 1, -1)   # [1, 1, 1152]
        return feat

    # ── Dataset interface ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.sim_ids)

    def __getitem__(self, idx: int):
        """
        Returns:
            vae_feat    : Tensor [16, lt_sz, lt_sz]           unpaired real H&E latent
            ssl_feat    : Tensor [1, 1, 1152]                  unpaired UNI embedding
            ctrl_tensor : Tensor [C, resolution, resolution]  sim control signal
            vae_mask    : Tensor [16, lt_sz, lt_sz]            VAE mask / zeros
            data_info   : dict
        """
        # Simulation control (indexed deterministically)
        sim_id      = self.sim_ids[idx]
        ctrl_tensor = self._build_ctrl_tensor(sim_id)

        # Real H&E features (randomly sampled — intentionally unpaired)
        real_idx = int(self._rng.integers(0, len(self.real_ids)))
        tile_id  = self.real_ids[real_idx]
        vae_feat = self._load_vae_feat(tile_id)
        vae_mask = self._load_vae_mask(tile_id)
        ssl_feat = self._load_ssl_feat(tile_id)

        # Only tensor values — accelerate tries to concatenate every field in
        # the returned dict across workers and will crash on str/list types.
        # img_hw and aspect_ratio are the only fields the PixArt transformer
        # uses internally (positional embedding interpolation).
        data_info = {
            "img_hw":       torch.tensor([self.resolution] * 2, dtype=torch.float32),
            "aspect_ratio": torch.tensor(1.0),
            "sim_idx": torch.tensor(idx, dtype=torch.int64),
        }

        return vae_feat, ssl_feat, ctrl_tensor, vae_mask, data_info
    
    def get_ids(self, idx: int) -> dict[str, str]:
        """Debug helper: return IDs for a given dataset index (no tensors)."""
        sim_id = self.sim_ids[idx]
        # reproduce the same real_idx draw as __getitem__ would do (deterministic if you want)
        # simplest: just return sim_id and let caller sample a real_id if needed
        return {"sim_id": sim_id}

# ── Index builders (run once before training) ──────────────────────────────────

def build_sim_index(
    sim_channels_dir: str | Path,
    output_path: str | Path,
    resolution: int = 256,
    key_name: str | None = None,
):
    """
    Build HDF5 index for simulation snapshots by scanning sim_channels/cell_mask/.
    Sim IDs are the file stems (e.g. "combined_grid_0000_010080" from the .png).

    Example:
        build_sim_index(
            "sim_data_root/sim_channels",
            "sim_data_root/metadata/sim_index.hdf5",
        )
    """
    mask_dir = Path(sim_channels_dir) / "cell_mask"
    if not mask_dir.exists():
        raise RuntimeError(
            f"cell_mask directory not found: {mask_dir}\n"
            f"Expected: sim_channels/cell_mask/{{sim_id}}.png"
        )
    files = sorted(p for ext in ("*.png", "*.npy") for p in mask_dir.glob(ext))
    if not files:
        raise RuntimeError(f"No PNG/NPY files found in {mask_dir}")

    sim_ids = [f.stem for f in files]
    _write_h5_index(Path(output_path), key_name or f"sim_{resolution}", sim_ids)
    print(f"[build_sim_index] Wrote {len(sim_ids)} IDs → {output_path}")


def build_real_index(
    features_dir: str | Path,
    output_path: str | Path,
    ssl_prefix: str = "uni",
    resolution: int = 256,
    key_name: str | None = None,
):
    """
    Build HDF5 index for real H&E tiles by scanning features/ for UNI .npy files.

    Example:
        build_real_index(
            "sim_data_root/features",
            "sim_data_root/metadata/real_index.hdf5",
        )
    """
    features_dir = Path(features_dir)
    files        = sorted(features_dir.glob(f"*_{ssl_prefix}.npy"))
    if not files:
        raise RuntimeError(f"No *_{ssl_prefix}.npy files found in {features_dir}")

    tile_ids = [f.name.replace(f"_{ssl_prefix}.npy", "") for f in files]
    _write_h5_index(Path(output_path), key_name or f"real_{resolution}", tile_ids)
    print(f"[build_real_index] Wrote {len(tile_ids)} IDs → {output_path}")


def _write_h5_index(path: Path, key: str, ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        dt   = h5py.special_dtype(vlen=str)
        dset = h5.create_dataset(key, shape=(len(ids),), dtype=dt)
        for i, sid in enumerate(ids):
            dset[i] = sid


# ── Channel index utility ──────────────────────────────────────────────────────

def get_channel_index(active_channels: list[str], channel_name: str) -> int:
    """
    Get the tensor index of a named channel.

    Example (in model forward or loss function):
        idx  = get_channel_index(active_channels, "cell_mask")
        mask = ctrl_tensor[:, idx : idx + 1, :, :]   # [B, 1, H, W]
    """
    ordered = _validate_channels(active_channels)
    if channel_name not in ordered:
        raise ValueError(f"'{channel_name}' not in active_channels={ordered}")
    return ordered.index(channel_name)