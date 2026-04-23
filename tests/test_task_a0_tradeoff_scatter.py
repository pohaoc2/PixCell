from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _write_metrics(tile_dir: Path, per_condition: dict[str, dict[str, float]]) -> None:
    tile_dir.mkdir(parents=True, exist_ok=True)
    (tile_dir / "metrics.json").write_text(
        json.dumps({"version": 2, "tile_id": tile_dir.name, "per_condition": per_condition}),
        encoding="utf-8",
    )


def test_tradeoff_run_task_writes_csv_and_panels(tmp_path: Path):
    from src.a0_tradeoff_scatter.run import TradeoffScatterConfig, run_task

    paired_root = tmp_path / "paired_ablation" / "ablation_results"
    unpaired_root = tmp_path / "unpaired_ablation" / "ablation_results"
    paired_per_condition = {
        "cell_state": {"aji": 0.30, "pq": 0.20, "fid": 45.0},
        "cell_state+cell_types": {"aji": 0.35, "pq": 0.25, "fid": 43.0},
        "cell_state+cell_types+microenv": {"aji": 0.50, "pq": 0.42, "fid": 50.0},
        "cell_state+cell_types+microenv+vasculature": {"aji": 0.55, "pq": 0.46, "fid": 52.0},
    }
    unpaired_per_condition = {
        "cell_state": {"aji": 0.22, "pq": 0.16, "fud": 90.0},
        "cell_state+cell_types": {"aji": 0.28, "pq": 0.20, "fud": 85.0},
        "cell_state+cell_types+microenv": {"aji": 0.31, "pq": 0.24, "fud": 82.0},
        "cell_state+cell_types+microenv+vasculature": {"aji": 0.33, "pq": 0.26, "fud": 88.0},
    }
    _write_metrics(paired_root / "10240_11008", paired_per_condition)
    _write_metrics(paired_root / "10496_11776", paired_per_condition)
    _write_metrics(unpaired_root / "10240_11008", unpaired_per_condition)
    _write_metrics(unpaired_root / "10496_11776", unpaired_per_condition)

    outputs = run_task(
        TradeoffScatterConfig(
            paired_metric_dir=tmp_path / "paired_ablation",
            unpaired_metric_dir=tmp_path / "unpaired_ablation",
            out_dir=tmp_path / "out",
        )
    )

    assert outputs["csv"].is_file()
    assert outputs["paired_png"].is_file()
    assert outputs["unpaired_png"].is_file()

    with outputs["csv"].open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 8
    paired_rows = [row for row in rows if row["split"] == "paired"]
    unpaired_rows = [row for row in rows if row["split"] == "unpaired"]
    assert all(row["realism_key"] == "fid" for row in paired_rows)
    assert all(row["realism_key"] == "fud" for row in unpaired_rows)
    assert any(row["is_pareto"] == "True" for row in paired_rows)
