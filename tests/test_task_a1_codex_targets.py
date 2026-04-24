from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_build_codex_targets_assigns_cells_to_tiles(tmp_path: Path):
    from src.a1_codex_targets.build import run_build_task

    tile_ids_path = tmp_path / "tile_ids.txt"
    tile_ids_path.write_text("0_0\n0_256\n", encoding="utf-8")

    markers_csv = tmp_path / "markers.csv"
    markers_csv.write_text(
        "Channel_Number,Marker_Name\n1,MarkerA\n2,MarkerB\n",
        encoding="utf-8",
    )

    features_csv = tmp_path / "features.csv"
    with features_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["CellID", "MarkerA", "MarkerB", "X_centroid", "Y_centroid"])
        writer.writeheader()
        writer.writerow({"CellID": 1, "MarkerA": 1.0, "MarkerB": 2.0, "X_centroid": 10.0, "Y_centroid": 10.0})
        writer.writerow({"CellID": 2, "MarkerA": 3.0, "MarkerB": 4.0, "X_centroid": 100.0, "Y_centroid": 50.0})
        writer.writerow({"CellID": 3, "MarkerA": 5.0, "MarkerB": 6.0, "X_centroid": 300.0, "Y_centroid": 20.0})

    outputs = run_build_task(features_csv, markers_csv, tile_ids_path, tmp_path / "out", min_cells=1)
    t2 = np.load(outputs["t2"])
    t3 = np.load(outputs["t3"])
    counts = np.load(outputs["counts"])

    assert t2.shape == (2, 2)
    assert t3.shape == (2, 8)
    assert counts.tolist() == [2, 1]
    assert np.allclose(t2[0], [2.0, 3.0])
    assert np.allclose(t2[1], [5.0, 6.0])


def test_run_probe_tasks_uses_supplied_runners(tmp_path: Path):
    from src.a1_codex_targets.probe import run_probe_tasks

    calls: list[tuple[str, Path, int | None]] = []

    def fake_runner(*args, **kwargs):
        out_dir = Path(args[3])
        out_dir.mkdir(parents=True, exist_ok=True)
        result_path = out_dir / "result.json"
        result_path.write_text("{}", encoding="utf-8")
        calls.append((str(args[1]), out_dir, kwargs.get("n_jobs")))
        return {"json": result_path}

    outputs = run_probe_tasks(
        features_dir=tmp_path / "features",
        tile_ids_path=tmp_path / "tile_ids.txt",
        cv_splits_path=tmp_path / "cv_splits.json",
        t2_targets_path=tmp_path / "t2.npy",
        marker_names_path=tmp_path / "markers.json",
        out_dir=tmp_path / "out",
        linear_runner=fake_runner,
        mlp_runner=fake_runner,
        n_jobs=4,
        preloaded_X=np.zeros((1, 4), dtype=np.float32),
    )

    assert set(outputs.keys()) == {"t2_linear", "t2_mlp"}
    assert len(calls) == 2
    assert all(call[2] == 4 for call in calls)
