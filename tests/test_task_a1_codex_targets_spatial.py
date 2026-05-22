from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_centroid_to_patch_index_corner_cases():
    from src.a1_codex_targets_spatial.build import centroid_to_patch_index

    # Top-left of tile 256_512 (row=256, col=512) -> patch (0,0) -> index 0.
    tile_id, idx = centroid_to_patch_index(512.0, 256.0, tile_size=256, grid=16)
    assert tile_id == "256_512"
    assert idx == 0

    # Just inside the bottom-right patch of the same tile.
    tile_id, idx = centroid_to_patch_index(512.0 + 255.9, 256.0 + 255.9, tile_size=256, grid=16)
    assert tile_id == "256_512"
    assert idx == 16 * 16 - 1

    # Mid-tile.
    tile_id, idx = centroid_to_patch_index(128.0, 128.0, tile_size=256, grid=16)
    assert tile_id == "0_0"
    # patch row = 128 // 16 = 8, patch col = 8 -> idx = 8*16 + 8 = 136
    assert idx == 136


def _write_features_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_markers_csv(path: Path, marker_names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Channel_Number", "Marker_Name"])
        writer.writeheader()
        for index, name in enumerate(marker_names):
            writer.writerow({"Channel_Number": index, "Marker_Name": name})


def test_build_codex_patch_targets_aggregates_per_patch(tmp_path: Path):
    from src.a1_codex_targets_spatial.build import build_codex_patch_targets

    markers_csv = tmp_path / "markers.csv"
    _write_markers_csv(markers_csv, ["CD45", "PD-1"])

    features_csv = tmp_path / "features.csv"
    rows = [
        # Two cells in tile 0_0 patch (0,0)
        {"X_centroid": 4.0, "Y_centroid": 4.0, "CD45": 1.0, "PD-1": 0.0},
        {"X_centroid": 6.0, "Y_centroid": 6.0, "CD45": 3.0, "PD-1": 0.4},
        # One cell in tile 0_0 patch (1,1) -> idx grid+1
        {"X_centroid": 20.0, "Y_centroid": 20.0, "CD45": 2.0, "PD-1": 0.2},
        # One cell in tile that is not in tile_ids (drop)
        {"X_centroid": 1024.0, "Y_centroid": 1024.0, "CD45": 10.0, "PD-1": 1.0},
    ]
    _write_features_csv(features_csv, rows)

    tile_ids = ["0_0"]
    marker_names, tensor, counts = build_codex_patch_targets(
        features_csv,
        markers_csv,
        tile_ids,
        tile_size=256,
        grid=16,
    )
    assert marker_names == ["CD45", "PD-1"]
    assert tensor.shape == (1, 256, 2)
    # Patch (0,0) -> idx 0, mean CD45 = 2.0, mean PD-1 = 0.2
    assert tensor[0, 0, 0] == 2.0
    np.testing.assert_allclose(tensor[0, 0, 1], 0.2)
    assert counts[0, 0] == 2
    # Patch (1,1) -> idx grid+1 = 17, single-cell mean
    assert tensor[0, 17, 0] == 2.0
    assert counts[0, 17] == 1
    # Empty patches -> NaN
    assert np.isnan(tensor[0, 1, 0])
    assert counts[0, 1] == 0
