"""Build T2 and T3 tile-level CODEX targets from per-cell features."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from src._tasklib.io import ensure_directory, write_json
from src._tasklib.tile_ids import tile_ids_sha1, write_tile_ids


def load_marker_names(markers_csv: str | Path) -> list[str]:
    """Load marker names in channel-number order."""
    with Path(markers_csv).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = sorted(reader, key=lambda row: int(row["Channel_Number"]))
    return [row["Marker_Name"] for row in rows]


def centroid_to_tile_id(x_centroid: float, y_centroid: float, tile_size: int = 256) -> str:
    """Map full-resolution centroid coordinates into the canonical tile grid."""
    row_px = int(y_centroid // tile_size) * tile_size
    col_px = int(x_centroid // tile_size) * tile_size
    return f"{row_px}_{col_px}"


def build_codex_targets(
    features_csv: str | Path,
    markers_csv: str | Path,
    tile_ids: list[str],
    *,
    min_cells: int = 1,
    quantiles: tuple[float, float, float, float] = (0.10, 0.25, 0.75, 0.90),
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Build tile-aligned T2 means and T3 quantiles from per-cell CODEX features."""
    marker_names = load_marker_names(markers_csv)
    quantile_names = [
        f"{marker}_{suffix}"
        for marker in marker_names
        for suffix in ("q10", "q25", "q75", "q90")
    ]

    tile_set = set(tile_ids)
    sums: dict[str, np.ndarray] = {}
    values: dict[str, list[list[float]]] = {}
    counts: dict[str, int] = {}

    with Path(features_csv).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            tile_id = centroid_to_tile_id(float(row["X_centroid"]), float(row["Y_centroid"]))
            if tile_id not in tile_set:
                continue
            if tile_id not in sums:
                sums[tile_id] = np.zeros(len(marker_names), dtype=np.float64)
                values[tile_id] = [[] for _ in marker_names]
                counts[tile_id] = 0
            marker_values = np.asarray([float(row[marker]) for marker in marker_names], dtype=np.float64)
            sums[tile_id] += marker_values
            counts[tile_id] += 1
            for index, value in enumerate(marker_values.tolist()):
                values[tile_id][index].append(value)

    t2 = np.full((len(tile_ids), len(marker_names)), np.nan, dtype=np.float32)
    t3 = np.full((len(tile_ids), len(quantile_names)), np.nan, dtype=np.float32)
    cell_counts = np.zeros(len(tile_ids), dtype=np.int32)

    for row_index, tile_id in enumerate(tile_ids):
        count = counts.get(tile_id, 0)
        cell_counts[row_index] = count
        if count < min_cells:
            continue
        t2[row_index] = (sums[tile_id] / count).astype(np.float32)
        quantile_values: list[float] = []
        for marker_values in values[tile_id]:
            quantile_values.extend(np.quantile(np.asarray(marker_values, dtype=np.float32), quantiles).tolist())
        t3[row_index] = np.asarray(quantile_values, dtype=np.float32)

    return marker_names, t2, t3, cell_counts, quantile_names


def save_codex_bundle(
    tile_ids: list[str],
    marker_names: list[str],
    quantile_names: list[str],
    t2: np.ndarray,
    t3: np.ndarray,
    cell_counts: np.ndarray,
    out_dir: str | Path,
) -> dict[str, Path]:
    """Persist the CODEX target bundle."""
    output_dir = ensure_directory(out_dir)
    t2_path = output_dir / "codex_T2_mean.npy"
    t3_path = output_dir / "codex_T3_quantiles.npy"
    counts_path = output_dir / "codex_cell_counts.npy"
    np.save(t2_path, t2)
    np.save(t3_path, t3)
    np.save(counts_path, cell_counts)

    tile_ids_path = write_tile_ids(tile_ids, output_dir / "tile_ids.txt")
    markers_path = write_json(marker_names, output_dir / "codex_marker_names.json")
    quantile_names_path = write_json(quantile_names, output_dir / "codex_T3_feature_names.json")
    manifest_path = write_json(
        {
            "version": 1,
            "tile_count": len(tile_ids),
            "tile_ids_sha1": tile_ids_sha1(tile_ids),
            "n_markers": len(marker_names),
            "n_quantile_features": len(quantile_names),
        },
        output_dir / "manifest.json",
    )
    return {
        "t2": t2_path,
        "t3": t3_path,
        "counts": counts_path,
        "tile_ids": tile_ids_path,
        "marker_names": markers_path,
        "quantile_names": quantile_names_path,
        "manifest": manifest_path,
    }


def run_build_task(
    features_csv: str | Path,
    markers_csv: str | Path,
    tile_ids_path: str | Path,
    out_dir: str | Path,
    *,
    min_cells: int = 1,
) -> dict[str, Path]:
    """Build and save the full CODEX target bundle."""
    tile_ids = [line.strip() for line in Path(tile_ids_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    marker_names, t2, t3, cell_counts, quantile_names = build_codex_targets(
        features_csv,
        markers_csv,
        tile_ids,
        min_cells=min_cells,
    )
    return save_codex_bundle(tile_ids, marker_names, quantile_names, t2, t3, cell_counts, out_dir)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Build T2/T3 CODEX targets from per-cell features")
    parser.add_argument("--features-csv", required=True)
    parser.add_argument("--markers-csv", required=True)
    parser.add_argument("--tile-ids-path", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--min-cells", type=int, default=1)
    args = parser.parse_args(argv)

    run_build_task(
        args.features_csv,
        args.markers_csv,
        args.tile_ids_path,
        args.out_dir,
        min_cells=args.min_cells,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
