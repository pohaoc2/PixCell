"""Build per-patch T2 CODEX marker targets from per-cell features.

Mirrors src.a1_codex_targets.build but assigns each cell to a (tile, patch)
bucket based on centroid coordinates within the tile and aggregates marker
intensities into a (N_tiles, GRID*GRID, n_markers) tensor. Patches with zero
cells receive NaN; downstream the spatial probe masks these.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from src._tasklib.io import ensure_directory, write_json
from src._tasklib.tile_ids import tile_ids_sha1, write_tile_ids
from src.a1_codex_targets.build import load_marker_names


def centroid_to_patch_index(
    x_centroid: float,
    y_centroid: float,
    *,
    tile_size: int = 256,
    grid: int = 16,
) -> tuple[str, int]:
    """Map a centroid to (tile_id, patch_index in 0..grid*grid-1)."""
    if tile_size % grid:
        raise ValueError(f"tile_size={tile_size} not divisible by grid={grid}")
    patch_px = tile_size // grid
    tile_row_px = int(y_centroid // tile_size) * tile_size
    tile_col_px = int(x_centroid // tile_size) * tile_size
    patch_row = int((y_centroid - tile_row_px) // patch_px)
    patch_col = int((x_centroid - tile_col_px) // patch_px)
    patch_row = max(0, min(grid - 1, patch_row))
    patch_col = max(0, min(grid - 1, patch_col))
    tile_id = f"{tile_row_px}_{tile_col_px}"
    return tile_id, patch_row * grid + patch_col


def build_codex_patch_targets(
    features_csv: str | Path,
    markers_csv: str | Path,
    tile_ids: list[str],
    *,
    tile_size: int = 256,
    grid: int = 16,
    min_cells_per_patch: int = 1,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Aggregate per-cell CODEX marker intensities into a (N, P, M) patch tensor.

    Returns (marker_names, target_tensor, cell_counts) where:
        target_tensor : (N_tiles, grid*grid, n_markers) float32, NaN where empty
        cell_counts   : (N_tiles, grid*grid)            int32
    """
    marker_names = load_marker_names(markers_csv)
    n_markers = len(marker_names)
    n_patches = grid * grid
    tile_index = {tile_id: idx for idx, tile_id in enumerate(tile_ids)}

    sums = np.zeros((len(tile_ids), n_patches, n_markers), dtype=np.float64)
    counts = np.zeros((len(tile_ids), n_patches), dtype=np.int32)

    with Path(features_csv).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            tile_id, patch_idx = centroid_to_patch_index(
                float(row["X_centroid"]),
                float(row["Y_centroid"]),
                tile_size=tile_size,
                grid=grid,
            )
            if tile_id not in tile_index:
                continue
            tile_row = tile_index[tile_id]
            values = np.asarray(
                [float(row[marker]) for marker in marker_names],
                dtype=np.float64,
            )
            sums[tile_row, patch_idx] += values
            counts[tile_row, patch_idx] += 1

    with np.errstate(invalid="ignore"):
        means = sums / counts[..., None]
    means[counts < min_cells_per_patch] = np.nan
    return marker_names, means.astype(np.float32), counts


def save_codex_patch_bundle(
    tile_ids: list[str],
    marker_names: list[str],
    target_tensor: np.ndarray,
    cell_counts: np.ndarray,
    out_dir: str | Path,
    *,
    grid: int,
) -> dict[str, Path]:
    """Persist the per-patch CODEX target bundle."""
    output_dir = ensure_directory(out_dir)
    tensor_path = output_dir / "codex_T2_spatial_mean.npy"
    counts_path = output_dir / "codex_cell_counts_per_patch.npy"
    np.save(tensor_path, target_tensor)
    np.save(counts_path, cell_counts)
    tile_ids_path = write_tile_ids(tile_ids, output_dir / "tile_ids.txt")
    markers_path = write_json(marker_names, output_dir / "codex_marker_names.json")
    manifest_path = write_json(
        {
            "version": 1,
            "tile_count": len(tile_ids),
            "tile_ids_sha1": tile_ids_sha1(tile_ids),
            "grid": grid,
            "n_patches": grid * grid,
            "n_markers": len(marker_names),
        },
        output_dir / "manifest.json",
    )
    return {
        "tensor": tensor_path,
        "counts": counts_path,
        "tile_ids": tile_ids_path,
        "marker_names": markers_path,
        "manifest": manifest_path,
    }


def run_build_task(
    features_csv: str | Path,
    markers_csv: str | Path,
    tile_ids_path: str | Path,
    out_dir: str | Path,
    *,
    tile_size: int = 256,
    grid: int = 16,
    min_cells_per_patch: int = 1,
) -> dict[str, Path]:
    """Build and save the full per-patch CODEX target bundle."""
    tile_ids = [
        line.strip()
        for line in Path(tile_ids_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    marker_names, target_tensor, cell_counts = build_codex_patch_targets(
        features_csv,
        markers_csv,
        tile_ids,
        tile_size=tile_size,
        grid=grid,
        min_cells_per_patch=min_cells_per_patch,
    )
    return save_codex_patch_bundle(
        tile_ids,
        marker_names,
        target_tensor,
        cell_counts,
        out_dir,
        grid=grid,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Build per-patch T2 CODEX targets from per-cell features"
    )
    parser.add_argument("--features-csv", required=True)
    parser.add_argument("--markers-csv", required=True)
    parser.add_argument("--tile-ids-path", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--grid", type=int, default=16)
    parser.add_argument("--min-cells-per-patch", type=int, default=1)
    args = parser.parse_args(argv)

    run_build_task(
        args.features_csv,
        args.markers_csv,
        args.tile_ids_path,
        args.out_dir,
        tile_size=args.tile_size,
        grid=args.grid,
        min_cells_per_patch=args.min_cells_per_patch,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
