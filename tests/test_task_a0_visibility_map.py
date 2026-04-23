from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _write_diff_tile(tile_dir: Path, payload: dict[str, dict[str, float]]) -> None:
    tile_dir.mkdir(parents=True, exist_ok=True)
    (tile_dir / "leave_one_out_diff_stats.json").write_text(json.dumps(payload), encoding="utf-8")
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(tile_dir / "leave_one_out_diff.png")


def test_visibility_map_run_task_writes_outputs(tmp_path: Path):
    from src.a0_visibility_map.run import VisibilityMapConfig, run_task

    paired_root = tmp_path / "paired_ablation" / "ablation_results"
    unpaired_root = tmp_path / "unpaired_ablation" / "leave_one_out"
    payload_a = {
        "cell_types": {"mean_diff": 2.0, "max_diff": 100.0, "pct_pixels_above_10": 5.0},
        "cell_state": {"mean_diff": 10.0, "max_diff": 255.0, "pct_pixels_above_10": 20.0},
        "vasculature": {"mean_diff": 1.0, "max_diff": 50.0, "pct_pixels_above_10": 3.0},
        "microenv": {"mean_diff": 6.0, "max_diff": 140.0, "pct_pixels_above_10": 15.0},
    }
    payload_b = {
        "cell_types": {"mean_diff": 4.0, "max_diff": 120.0, "pct_pixels_above_10": 7.0},
        "cell_state": {"mean_diff": 12.0, "max_diff": 255.0, "pct_pixels_above_10": 24.0},
        "vasculature": {"mean_diff": 2.0, "max_diff": 60.0, "pct_pixels_above_10": 4.0},
        "microenv": {"mean_diff": 8.0, "max_diff": 160.0, "pct_pixels_above_10": 18.0},
    }
    _write_diff_tile(paired_root / "10240_11008", payload_a)
    _write_diff_tile(paired_root / "10496_11776", payload_b)
    _write_diff_tile(unpaired_root / "10240_11008", payload_b)
    _write_diff_tile(unpaired_root / "10496_11776", payload_a)

    outputs = run_task(
        VisibilityMapConfig(
            paired_stats_root=tmp_path / "paired_ablation",
            unpaired_stats_root=tmp_path / "unpaired_ablation",
            out_dir=tmp_path / "out",
        )
    )

    assert outputs["csv"].is_file()
    assert outputs["chart"].is_file()
    inset_files = sorted((tmp_path / "out" / "inset_tiles").glob("*.png"))
    assert len(inset_files) == 4

    with outputs["csv"].open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["group"] for row in rows] == ["cell_state", "microenv", "cell_types", "vasculature"]
    assert float(rows[0]["paired_mean_diff"]) == 11.0
    assert float(rows[0]["unpaired_mean_diff"]) == 11.0
