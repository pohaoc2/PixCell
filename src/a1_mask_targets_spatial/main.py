"""Build per-patch T1 targets aligned to UNI patch-token grid.

Inputs are 256x256 experimental channel maps (cell_masks, cell_type_*,
cell_state_*, vasculature, oxygen, glucose). Each is block-mean-pooled to a
PATCH_GRID x PATCH_GRID array (default 16x16). Cell-type / cell-state targets
are reported as per-patch fractions of the local density (binary channel mean /
cell_masks mean, with epsilon). Oxygen / glucose / vasculature stay as raw
per-patch means.

Output bundle mirrors src.a1_mask_targets but the matrix has an added patch
axis: targets_T1_spatial.npy with shape (N_tiles, PATCH_GRID**2, n_targets).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from src._tasklib.io import ensure_directory, write_json
from src._tasklib.tile_ids import list_feature_tile_ids, tile_ids_sha1, write_tile_ids
from src.a1_mask_targets.main import (
    _load_spatial_file,
    get_channel_load_config,
    resolve_channel_dir,
)

TARGET_NAMES = [
    "cell_density",
    "cancer_frac",
    "healthy_frac",
    "immune_frac",
    "prolif_frac",
    "nonprolif_frac",
    "dead_frac",
    "vasculature_frac",
    "oxygen_mean",
    "glucose_mean",
]

_DENSITY_CHANNEL = "cell_masks"
_CELL_TYPE_CHANNELS = ["cell_type_cancer", "cell_type_healthy", "cell_type_immune"]
_CELL_STATE_CHANNELS = ["cell_state_prolif", "cell_state_nonprolif", "cell_state_dead"]
_OPTIONAL_CHANNELS = ["vasculature", "oxygen", "glucose"]
_FRAC_CHANNELS = set(_CELL_TYPE_CHANNELS) | set(_CELL_STATE_CHANNELS) | {"vasculature"}


def block_mean_pool(array: np.ndarray, grid: int) -> np.ndarray:
    """Average-pool a 2D array into (grid, grid) blocks.

    Uses exact block means when the array side is divisible by ``grid``.
    Otherwise falls back to area interpolation, which preserves mean behavior
    for arbitrary output grids.
    """
    h, w = array.shape
    if h % grid == 0 and w % grid == 0:
        bh, bw = h // grid, w // grid
        return array.reshape(grid, bh, grid, bw).mean(axis=(1, 3))
    return cv2.resize(array, (grid, grid), interpolation=cv2.INTER_AREA)


def _find_file(directory: Path, stem: str, exts: tuple[str, ...]) -> Path:
    for ext in exts:
        path = directory / f"{stem}{ext}"
        if path.exists():
            return path
    raise FileNotFoundError(f"no file found for {stem!r} under {directory}")


def _load_channel_patches(
    exp_channels_dir: Path,
    tile_id: str,
    channel_name: str,
    *,
    resolution: int,
    grid: int,
) -> np.ndarray:
    cfg = get_channel_load_config(channel_name)
    channel_dir = resolve_channel_dir(exp_channels_dir, channel_name)
    path = _find_file(channel_dir, tile_id, exts=tuple(cfg["preferred_exts"]))
    array = _load_spatial_file(
        path,
        resolution=resolution,
        binary=bool(cfg["binary"]),
        normalization=str(cfg["normalization"]),
    )
    return block_mean_pool(array, grid)


def compute_tile_patch_targets(
    exp_channels_dir: str | Path,
    tile_id: str,
    *,
    resolution: int = 256,
    grid: int = 16,
    eps: float = 1e-6,
) -> np.ndarray:
    """Return per-patch T1 target tensor of shape (grid*grid, n_targets)."""
    exp_dir = Path(exp_channels_dir)
    density = _load_channel_patches(exp_dir, tile_id, _DENSITY_CHANNEL, resolution=resolution, grid=grid)
    denom = density + eps
    columns: list[np.ndarray] = [density]
    for channel in _CELL_TYPE_CHANNELS:
        columns.append(_load_channel_patches(exp_dir, tile_id, channel, resolution=resolution, grid=grid) / denom)
    for channel in _CELL_STATE_CHANNELS:
        columns.append(_load_channel_patches(exp_dir, tile_id, channel, resolution=resolution, grid=grid) / denom)
    for channel in _OPTIONAL_CHANNELS:
        columns.append(_load_channel_patches(exp_dir, tile_id, channel, resolution=resolution, grid=grid))
    matrix = np.stack(columns, axis=-1).reshape(grid * grid, len(TARGET_NAMES))
    return matrix.astype(np.float32, copy=False)


def build_t1_patch_targets(
    features_dir: str | Path,
    exp_channels_dir: str | Path,
    *,
    resolution: int = 256,
    grid: int = 16,
    feature_suffix: str = "_uni_tokens.npy",
) -> tuple[list[str], np.ndarray]:
    """Build the full (N, grid*grid, n_targets) T1 patch-target tensor."""
    tile_ids = list_feature_tile_ids(features_dir, suffix=feature_suffix)
    if not tile_ids:
        # Fall back to CLS-feature listing if patch tokens have not been cached
        # yet; helpful for unit tests that use synthetic fixtures.
        tile_ids = list_feature_tile_ids(features_dir)
    tensors = [
        compute_tile_patch_targets(exp_channels_dir, tile_id, resolution=resolution, grid=grid)
        for tile_id in tile_ids
    ]
    matrix = np.stack(tensors, axis=0).astype(np.float32, copy=False)
    return tile_ids, matrix


def summarize_patch_targets(matrix: np.ndarray) -> list[dict[str, float | str | int]]:
    """Per-target summary over all (tile, patch) entries."""
    rows: list[dict[str, float | str | int]] = []
    flat = matrix.reshape(-1, matrix.shape[-1])
    for index, name in enumerate(TARGET_NAMES):
        column = flat[:, index]
        finite = np.isfinite(column)
        rows.append(
            {
                "target": name,
                "mean": float(np.nanmean(column)),
                "std": float(np.nanstd(column)),
                "min": float(np.nanmin(column)),
                "max": float(np.nanmax(column)),
                "n_non_nan": int(finite.sum()),
            }
        )
    return rows


def save_patch_target_bundle(
    tile_ids: list[str],
    matrix: np.ndarray,
    out_dir: str | Path,
    *,
    grid: int,
) -> dict[str, Path]:
    """Persist the per-patch T1 target tensor."""
    output_dir = ensure_directory(out_dir)
    matrix_path = output_dir / "mask_targets_T1_spatial.npy"
    np.save(matrix_path, matrix.astype(np.float32))
    tile_ids_path = write_tile_ids(tile_ids, output_dir / "tile_ids.txt")
    names_path = write_json(TARGET_NAMES, output_dir / "target_names_T1_spatial.json")

    stats_rows = summarize_patch_targets(matrix)
    stats_path = output_dir / "target_stats.csv"
    with stats_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["target", "mean", "std", "min", "max", "n_non_nan"])
        writer.writeheader()
        for row in stats_rows:
            writer.writerow(row)

    manifest_path = write_json(
        {
            "version": 1,
            "tile_count": len(tile_ids),
            "tile_ids_sha1": tile_ids_sha1(tile_ids),
            "grid": grid,
            "n_patches": grid * grid,
            "target_names": TARGET_NAMES,
        },
        output_dir / "manifest.json",
    )
    return {
        "matrix": matrix_path,
        "tile_ids": tile_ids_path,
        "target_names": names_path,
        "stats": stats_path,
        "manifest": manifest_path,
    }


def run_task(
    features_dir: str | Path,
    exp_channels_dir: str | Path,
    out_dir: str | Path,
    *,
    resolution: int = 256,
    grid: int = 16,
    feature_suffix: str = "_uni_tokens.npy",
) -> dict[str, Path]:
    """Build and save the full per-patch T1 target bundle."""
    tile_ids, matrix = build_t1_patch_targets(
        features_dir,
        exp_channels_dir,
        resolution=resolution,
        grid=grid,
        feature_suffix=feature_suffix,
    )
    return save_patch_target_bundle(tile_ids, matrix, out_dir, grid=grid)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Build per-patch T1 mask targets from exp channels")
    parser.add_argument("--features-dir", required=True)
    parser.add_argument("--exp-channels-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--grid", type=int, default=16)
    parser.add_argument("--feature-suffix", type=str, default="_uni_tokens.npy")
    args = parser.parse_args(argv)

    run_task(
        args.features_dir,
        args.exp_channels_dir,
        args.out_dir,
        resolution=args.resolution,
        grid=args.grid,
        feature_suffix=args.feature_suffix,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
