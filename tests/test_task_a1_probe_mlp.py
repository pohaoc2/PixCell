from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_mlp_probe_reuses_linear_splits_and_writes_comparison(tmp_path: Path):
    from src.a1_probe_linear.main import run_task as run_linear_task
    from src.a1_probe_mlp.main import run_task as run_mlp_task

    rng = np.random.default_rng(42)
    features_dir = tmp_path / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    tile_ids = []
    X_rows = []
    for index in range(20):
        tile_id = f"{index * 2048}_0"
        tile_ids.append(tile_id)
        vector = rng.normal(size=8).astype(np.float32)
        X_rows.append(vector)
        np.save(features_dir / f"{tile_id}_uni.npy", vector)

    X = np.stack(X_rows, axis=0)
    W = rng.normal(size=(8, 2)).astype(np.float32)
    Y = X @ W

    targets_path = tmp_path / "targets.npy"
    np.save(targets_path, Y.astype(np.float32))
    tile_ids_path = tmp_path / "tile_ids.txt"
    tile_ids_path.write_text("\n".join(tile_ids) + "\n", encoding="utf-8")
    target_names_path = tmp_path / "target_names.json"
    target_names_path.write_text(json.dumps(["t0", "t1"]), encoding="utf-8")

    linear_outputs = run_linear_task(
        features_dir,
        targets_path,
        tile_ids_path,
        tmp_path / "linear",
        target_names_path=target_names_path,
        n_splits=5,
        block_size_px=2048,
        alpha=1e-3,
    )
    mlp_outputs = run_mlp_task(
        features_dir,
        targets_path,
        tile_ids_path,
        tmp_path / "mlp",
        target_names_path=target_names_path,
        cv_splits_path=linear_outputs["splits"],
        linear_results_json=linear_outputs["json"],
        random_state=42,
    )

    assert mlp_outputs["json"].is_file()
    assert mlp_outputs["csv"].is_file()
    assert mlp_outputs["comparison"].is_file()


def test_run_cv_regression_parallel_matches_serial() -> None:
    from src.a1_probe_linear.main import run_cv_regression

    rng = np.random.default_rng(0)
    X = rng.normal(size=(24, 6)).astype(np.float32)
    weights = rng.normal(size=(6, 3)).astype(np.float32)
    Y = (X @ weights).astype(np.float32)
    splits = [
        {"train_idx": list(range(0, 12)), "test_idx": list(range(12, 24))},
        {"train_idx": list(range(12, 24)), "test_idx": list(range(0, 12))},
    ]

    serial = run_cv_regression(X, Y, splits, n_jobs=1)
    parallel = run_cv_regression(X, Y, splits, n_jobs=2)

    for serial_array, parallel_array in zip(serial, parallel):
        assert np.allclose(serial_array, parallel_array, equal_nan=True)
