from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.render_dataset_metrics import load_combinations


def _write_metrics(tile_dir: Path, payload: dict) -> None:
    tile_dir.mkdir(parents=True, exist_ok=True)
    (tile_dir / "metrics.json").write_text(json.dumps(payload), encoding="utf-8")


def test_load_combinations_aggregates_metrics_from_metric_dir(tmp_path: Path):
    metric_dir = tmp_path / "full_ablation"
    _write_metrics(
        metric_dir / "tile_a",
        {
            "version": 2,
            "tile_id": "tile_a",
            "per_condition": {
                "cell_types": {"cosine": 0.70, "lpips": 0.22, "aji": 0.31, "pq": 0.28},
                "cell_types+cell_state": {"cosine": 0.81, "lpips": 0.18, "aji": 0.42, "pq": 0.39},
            },
        },
    )
    _write_metrics(
        metric_dir / "tile_b",
        {
            "version": 2,
            "tile_id": "tile_b",
            "per_condition": {
                "cell_types": {"cosine": 0.90, "lpips": 0.12, "aji": 0.51, "pq": 0.48},
                "cell_types+cell_state": {"cosine": 0.85, "lpips": 0.16, "aji": 0.46, "pq": 0.43},
            },
        },
    )

    combos, tile_count = load_combinations(metric_dir)

    assert tile_count == 2

    singles = next(combo for combo in combos if combo.mask == 1)
    assert singles.metrics["cosine"] is not None
    assert singles.metrics["cosine"]["mean"] == pytest.approx(0.80)
    assert singles.metrics["cosine"]["std"] == pytest.approx(0.10)
    assert singles.metrics["lpips"]["mean"] == pytest.approx(0.17)
    assert singles.metrics["aji"]["mean"] == pytest.approx(0.41)
    assert singles.metrics["pq"]["mean"] == pytest.approx(0.38)
    assert singles.metrics["fid"] is None


def test_load_combinations_requires_metrics_files(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_combinations(tmp_path / "missing_metrics")
