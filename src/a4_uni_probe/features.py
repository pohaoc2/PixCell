"""Feature builders for UNI and pooled TME baselines."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.a1_mask_targets.main import (
    _find_file,
    _load_spatial_file,
    get_channel_load_config,
    resolve_channel_dir,
)


TME_CHANNELS = (
    "oxygen",
    "glucose",
)

# 10-channel order matching the a1_concat training dataset (ALL_CHANNELS ordering)
COND_CHANNELS = (
    "cell_masks",
    "cell_type_healthy", "cell_type_cancer", "cell_type_immune",
    "cell_state_prolif", "cell_state_nonprolif", "cell_state_dead",
    "vasculature", "oxygen", "glucose",
)
TME_FEATURE_NAMES = tuple(
    feature_name
    for channel_name in TME_CHANNELS
    for feature_name in (f"{channel_name}_mean", f"{channel_name}_std")
)
TME_FEATURE_DIM = len(TME_FEATURE_NAMES)


def build_uni_features(features_dir: str | Path, tile_ids: list[str]) -> np.ndarray:
    feature_root = Path(features_dir)
    rows: list[np.ndarray] = []
    for tile_id in tile_ids:
        row = np.asarray(np.load(feature_root / f"{tile_id}_uni.npy"), dtype=np.float32)
        rows.append(row.reshape(-1))
    return np.stack(rows, axis=0)


def _load_channel_array(
    exp_channels_root: Path,
    tile_id: str,
    channel_name: str,
    *,
    resolution: int = 256,
) -> np.ndarray | None:
    load_cfg = get_channel_load_config(channel_name)
    try:
        path = _find_file(resolve_channel_dir(exp_channels_root, channel_name), tile_id, tuple(load_cfg["preferred_exts"]))
    except FileNotFoundError:
        return None
    return _load_spatial_file(
        path,
        resolution=resolution,
        binary=bool(load_cfg["binary"]),
        normalization=str(load_cfg["normalization"]),
    )


def build_tme_baseline_features(
    exp_channels_dir: str | Path,
    tile_ids: list[str],
    *,
    resolution: int = 256,
) -> np.ndarray:
    exp_root = Path(exp_channels_dir)
    rows: list[list[float]] = []
    for tile_id in tile_ids:
        feature_row: list[float] = []
        for channel_name in TME_CHANNELS:
            array = _load_channel_array(exp_root, tile_id, channel_name, resolution=resolution)
            if array is None:
                feature_row.extend((0.0, 0.0))
                continue
            feature_row.extend((float(np.mean(array)), float(np.std(array))))
        rows.append(feature_row)
    return np.asarray(rows, dtype=np.float32)


def build_controlnet_encoder_features(
    exp_channels_dir: str | Path,
    tile_ids: list[str],
    checkpoint_path: str | Path,
    *,
    resolution: int = 256,
    use_ema: bool = True,
    cache_path: str | Path | None = None,
    batch_size: int = 256,
) -> np.ndarray:
    """Build per-tile features from the trained a1_concat cond_embedder.

    Loads the PatchEmbed (Conv2d 10→1152, kernel=16) from the ControlNet
    checkpoint, runs it over each tile's 10-channel TME stack, and
    global-average-pools the (res/16)^2 patch tokens to yield [N, 1152] floats.
    Results are cached to *cache_path* if provided.
    """
    import torch
    import torch.nn as nn

    cache = Path(cache_path) if cache_path else None
    if cache is not None and cache.is_file():
        return np.load(cache)

    ck = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    sd_key = "state_dict_ema" if use_ema else "state_dict"
    sd = ck[sd_key]

    w = sd["cond_embedder.proj.weight"]
    b = sd["cond_embedder.proj.bias"]
    hidden_size, in_channels, patch_size = w.shape[0], w.shape[1], w.shape[2]

    proj = nn.Conv2d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size, bias=True)
    proj.weight.data.copy_(w)
    proj.bias.data.copy_(b)
    proj.eval()

    exp_root = Path(exp_channels_dir)
    n = len(tile_ids)
    features = np.zeros((n, hidden_size), dtype=np.float32)

    with torch.no_grad():
        for start in range(0, n, batch_size):
            batch = tile_ids[start : start + batch_size]
            tensors: list[np.ndarray] = []
            for tile_id in batch:
                channels = []
                for ch in COND_CHANNELS:
                    arr = _load_channel_array(exp_root, tile_id, ch, resolution=resolution)
                    channels.append(
                        arr.astype(np.float32) if arr is not None
                        else np.zeros((resolution, resolution), dtype=np.float32)
                    )
                tensors.append(np.stack(channels, axis=0))
            x = torch.from_numpy(np.stack(tensors, axis=0))  # [B, 10, res, res]
            pooled = proj(x).mean(dim=[2, 3])                # [B, hidden_size]
            features[start : start + len(batch)] = pooled.numpy()

    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, features)

    return features


def save_feature_bundle(
    out_dir: str | Path,
    *,
    tile_ids: list[str],
    uni_features: np.ndarray,
    tme_features: np.ndarray,
    tme_label: str = "O2/Glc",
) -> Path:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    bundle_path = out_path / "features.npz"
    np.savez_compressed(
        bundle_path,
        tile_ids=np.asarray(tile_ids, dtype=str),
        uni=uni_features.astype(np.float32, copy=False),
        tme=tme_features.astype(np.float32, copy=False),
        tme_feature_names=np.asarray(TME_FEATURE_NAMES, dtype=str),
        tme_label=np.asarray(tme_label, dtype=str),
    )
    return bundle_path
