"""Build T1 tile-level mask targets from experimental channels."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from src._tasklib.io import ensure_directory, write_json
from src._tasklib.tile_ids import list_feature_tile_ids, tile_ids_sha1, write_tile_ids


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
_EXTS = (".png", ".npy", ".jpg", ".jpeg", ".tif", ".tiff")
_PREFERRED_EXTS = {
    "oxygen": (".npy", ".png", ".jpg", ".jpeg", ".tif", ".tiff"),
    "glucose": (".npy", ".png", ".jpg", ".jpeg", ".tif", ".tiff"),
    "vasculature": (".npy", ".png", ".jpg", ".jpeg", ".tif", ".tiff"),
}
_BINARY_CHANNELS = {
    "cell_masks",
    "cell_type_cancer",
    "cell_type_healthy",
    "cell_type_immune",
    "cell_state_prolif",
    "cell_state_nonprolif",
    "cell_state_dead",
    "vasculature",
}
_CHANNEL_DIR_ALIASES = {
    "cell_masks": ("cell_mask",),
    "cell_mask": ("cell_masks",),
    "oxygen": ("oxygen_npy",),
    "glucose": ("glucose_npy",),
    "vasculature": ("vasculature_npy",),
}


def get_channel_load_config(channel_name: str) -> dict[str, object]:
    """Return the lightweight load policy for a channel name."""
    binary = channel_name in _BINARY_CHANNELS
    normalization = "binary" if binary else ("clip01" if channel_name in {"oxygen", "glucose"} else "minmax")
    return {
        "binary": binary,
        "normalization": normalization,
        "preferred_exts": _PREFERRED_EXTS.get(channel_name, _EXTS),
    }


def resolve_channel_dir(root_dir: Path, channel_name: str) -> Path:
    """Resolve the canonical or aliased channel directory."""
    for candidate in (channel_name, *_CHANNEL_DIR_ALIASES.get(channel_name, ())):
        candidate_dir = root_dir / candidate
        if candidate_dir.exists():
            return candidate_dir
    return root_dir / channel_name


def _find_file(directory: Path, stem: str, exts: tuple[str, ...]) -> Path:
    for ext in exts:
        path = directory / f"{stem}{ext}"
        if path.exists():
            return path
    raise FileNotFoundError(f"no file found for {stem!r} under {directory}")


def _load_spatial_file(
    path: Path,
    *,
    resolution: int,
    binary: bool,
    normalization: str,
) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        array = np.load(path).astype(np.float32)
        while array.ndim > 2 and array.shape[0] == 1:
            array = array.squeeze(0)
        if array.ndim == 3:
            array = array[0]
        if array.ndim != 2:
            raise ValueError(f"expected 2D array from {path}; got {array.shape}")
    else:
        raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise FileNotFoundError(f"cv2 could not read {path}")
        if raw.ndim == 3:
            raw = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        array = raw.astype(np.float32)
        if array.max() > 1.0:
            array /= 255.0

    if array.shape != (resolution, resolution):
        array = cv2.resize(array, (resolution, resolution), interpolation=cv2.INTER_LINEAR)

    if binary:
        return (array > 0.5).astype(np.float32)
    if normalization == "clip01":
        return np.clip(array, 0.0, 1.0).astype(np.float32)
    vmin = float(array.min())
    vmax = float(array.max())
    if vmax > vmin:
        array = (array - vmin) / (vmax - vmin)
    return array.astype(np.float32)


def _load_channel_mean(
    exp_channels_dir: Path,
    tile_id: str,
    channel_name: str,
    *,
    resolution: int = 256,
) -> float:
    load_cfg = get_channel_load_config(channel_name)
    channel_dir = resolve_channel_dir(exp_channels_dir, channel_name)
    path = _find_file(channel_dir, tile_id, exts=tuple(load_cfg["preferred_exts"]))
    array = _load_spatial_file(
        path,
        resolution=resolution,
        binary=bool(load_cfg["binary"]),
        normalization=str(load_cfg["normalization"]),
    )
    return float(array.mean())


def compute_tile_targets(
    exp_channels_dir: str | Path,
    tile_id: str,
    *,
    resolution: int = 256,
    eps: float = 1e-6,
) -> np.ndarray:
    """Compute the canonical 10-element T1 target vector for one tile."""
    exp_dir = Path(exp_channels_dir)
    cell_density = _load_channel_mean(exp_dir, tile_id, _DENSITY_CHANNEL, resolution=resolution)
    denom = cell_density + eps

    values = [cell_density]
    for channel_name in _CELL_TYPE_CHANNELS:
        values.append(_load_channel_mean(exp_dir, tile_id, channel_name, resolution=resolution) / denom)
    for channel_name in _CELL_STATE_CHANNELS:
        values.append(_load_channel_mean(exp_dir, tile_id, channel_name, resolution=resolution) / denom)
    for channel_name in _OPTIONAL_CHANNELS:
        values.append(_load_channel_mean(exp_dir, tile_id, channel_name, resolution=resolution))
    return np.asarray(values, dtype=np.float32)


def build_t1_targets(
    features_dir: str | Path,
    exp_channels_dir: str | Path,
    *,
    resolution: int = 256,
) -> tuple[list[str], np.ndarray]:
    """Build the full T1 target matrix aligned to cached UNI feature files."""
    tile_ids = list_feature_tile_ids(features_dir)
    matrix = np.stack(
        [compute_tile_targets(exp_channels_dir, tile_id, resolution=resolution) for tile_id in tile_ids],
        axis=0,
    )
    return tile_ids, matrix.astype(np.float32, copy=False)


def summarize_targets(target_matrix: np.ndarray) -> list[dict[str, float | str | int]]:
    """Summarize each target column for quick sanity checks."""
    rows: list[dict[str, float | str | int]] = []
    for index, name in enumerate(TARGET_NAMES):
        column = target_matrix[:, index]
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


def save_target_bundle(
    tile_ids: list[str],
    target_matrix: np.ndarray,
    out_dir: str | Path,
) -> dict[str, Path]:
    """Persist the T1 target matrix and its manifest."""
    output_dir = ensure_directory(out_dir)
    matrix_path = output_dir / "mask_targets_T1.npy"
    np.save(matrix_path, target_matrix.astype(np.float32))
    tile_ids_path = write_tile_ids(tile_ids, output_dir / "tile_ids.txt")
    names_path = write_json(TARGET_NAMES, output_dir / "target_names_T1.json")

    stats_rows = summarize_targets(target_matrix)
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
) -> dict[str, Path]:
    """Build and save the full T1 target bundle."""
    tile_ids, target_matrix = build_t1_targets(features_dir, exp_channels_dir, resolution=resolution)
    return save_target_bundle(tile_ids, target_matrix, out_dir)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Build T1 mask targets from exp channels")
    parser.add_argument("--features-dir", required=True)
    parser.add_argument("--exp-channels-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--resolution", type=int, default=256)
    args = parser.parse_args(argv)

    run_task(
        args.features_dir,
        args.exp_channels_dir,
        args.out_dir,
        resolution=args.resolution,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
