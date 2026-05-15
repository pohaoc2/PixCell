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
    "cell_type_cancer",
    "cell_type_healthy",
    "cell_type_immune",
    "cell_state_prolif",
    "cell_state_nonprolif",
    "cell_state_dead",
    "vasculature",
    "oxygen",
    "glucose",
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


def save_feature_bundle(
    out_dir: str | Path,
    *,
    tile_ids: list[str],
    uni_features: np.ndarray,
    tme_features: np.ndarray,
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
    )
    return bundle_path
