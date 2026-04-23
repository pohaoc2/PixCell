from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_build_spatial_group_splits_keeps_same_block_grouped():
    from src.a1_probe_linear.main import build_spatial_group_splits

    tile_ids = [
        "0_0",
        "0_1024",
        "2048_0",
        "4096_0",
        "6144_0",
        "8192_0",
        "10240_0",
    ]
    splits = build_spatial_group_splits(tile_ids, n_splits=5, block_size_px=2048)

    same_block = {0, 1}
    for split in splits:
        test_set = set(split["test_idx"])
        assert not (same_block & test_set and same_block - test_set)


def test_linear_probe_recovers_synthetic_targets(tmp_path: Path):
    from src.a1_probe_linear.main import run_task

    features_dir = tmp_path / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    tile_ids = []
    X_rows = []
    for index in range(10):
        tile_id = f"{index * 2048}_0"
        tile_ids.append(tile_id)
        vector = np.array([1.0 + index, 2.0 * index, 3.0 - index, 0.5 * index], dtype=np.float32)
        X_rows.append(vector)
        np.save(features_dir / f"{tile_id}_uni.npy", vector)

    X = np.stack(X_rows, axis=0)
    W = np.array(
        [
            [0.4, 0.2],
            [0.1, -0.3],
            [0.7, 0.5],
            [-0.2, 0.1],
        ],
        dtype=np.float32,
    )
    Y = X @ W

    targets_path = tmp_path / "targets.npy"
    np.save(targets_path, Y.astype(np.float32))
    tile_ids_path = tmp_path / "tile_ids.txt"
    tile_ids_path.write_text("\n".join(tile_ids) + "\n", encoding="utf-8")
    target_names_path = tmp_path / "target_names.json"
    target_names_path.write_text(json.dumps(["t0", "t1"]), encoding="utf-8")

    outputs = run_task(
        features_dir,
        targets_path,
        tile_ids_path,
        tmp_path / "out",
        target_names_path=target_names_path,
        n_splits=5,
        block_size_px=2048,
        alpha=1e-6,
    )

    rows = json.loads(outputs["json"].read_text(encoding="utf-8"))["results"]
    assert rows[0]["r2_mean"] > 0.999
    assert rows[1]["r2_mean"] > 0.999
    assert outputs["splits"].is_file()
    assert outputs["coef_mean"].is_file()