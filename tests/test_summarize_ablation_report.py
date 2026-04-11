from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.stage3.ablation_vis_utils import FOUR_GROUP_ORDER, condition_metric_key
from tools.summarize_ablation_report import (
    load_condition_means,
    load_leave_one_out_summary,
    summarize_added_group_effects,
    summarize_best_worst,
    summarize_by_cardinality,
    summarize_presence_absence,
)


def _write_metrics(tile_dir: Path, payload: dict) -> None:
    tile_dir.mkdir(parents=True, exist_ok=True)
    (tile_dir / "metrics.json").write_text(json.dumps(payload), encoding="utf-8")


def test_load_condition_means_discovers_available_metrics(tmp_path: Path) -> None:
    metrics_root = tmp_path / "ablation_results"
    _write_metrics(
        metrics_root / "tile_a",
        {
            "version": 2,
            "tile_id": "tile_a",
            "per_condition": {
                "cell_types": {"aji": 0.10, "pq": 0.05, "style_hed": 0.08},
                "cell_state+cell_types": {"aji": 0.22, "pq": 0.12, "style_hed": 0.07},
            },
        },
    )
    _write_metrics(
        metrics_root / "tile_b",
        {
            "version": 2,
            "tile_id": "tile_b",
            "per_condition": {
                "cell_types": {"aji": 0.14, "pq": 0.07, "style_hed": 0.09},
                "cell_state+cell_types": {"aji": 0.26, "pq": 0.14, "style_hed": 0.06},
            },
        },
    )

    condition_means, tile_count, metric_keys = load_condition_means(metrics_root)
    by_cardinality = summarize_by_cardinality(condition_means, metric_keys)

    assert tile_count == 2
    assert metric_keys == ["aji", "pq", "style_hed"]
    assert condition_means["cell_types"]["aji"] == pytest.approx(0.12)
    assert by_cardinality[1]["pq"] == pytest.approx(0.06)
    assert by_cardinality[2]["style_hed"] == pytest.approx(0.065)


def test_summary_helpers_capture_expected_metric_directions() -> None:
    condition_means: dict[str, dict[str, float]] = {}
    for size in range(1, len(FOUR_GROUP_ORDER) + 1):
        for subset in combinations(FOUR_GROUP_ORDER, size):
            groups = set(subset)
            key = condition_metric_key(subset)
            aji = 0.05 * size
            pq = 0.04 * size
            fud = 70.0 - (1.5 * size)
            if "cell_state" in groups:
                aji += 0.08
                pq += 0.07
                fud += 2.0
            if "microenv" in groups:
                aji += 0.06
                pq += 0.05
                fud -= 3.0
            if "cell_types" in groups:
                fud += 0.5
            if "vasculature" in groups:
                aji += 0.01
                pq += 0.01
                fud -= 0.2
            condition_means[key] = {
                "aji": aji,
                "pq": pq,
                "fud": fud,
                "style_hed": 0.090 - (0.004 * size) + (0.003 if "cell_state" in groups else 0.0) - (0.005 if "microenv" in groups else 0.0),
            }

    best_worst = summarize_best_worst(condition_means, ["aji", "pq", "fud", "style_hed"])
    added = summarize_added_group_effects(condition_means, ["aji", "pq", "fud", "style_hed"])
    presence = summarize_presence_absence(condition_means, ["aji", "pq", "fud", "style_hed"])

    full_key = condition_metric_key(FOUR_GROUP_ORDER)
    assert best_worst["aji"]["best"][0][0] == full_key
    assert best_worst["pq"]["best"][0][0] == full_key
    assert "microenv" in best_worst["fud"]["best"][0][0]
    assert "cell_state" not in best_worst["fud"]["best"][0][0]

    assert added["cell_state"]["aji"] > added["cell_types"]["aji"]
    assert added["microenv"]["fud"] > 0
    assert presence["cell_state"]["aji"] > 0
    assert presence["cell_state"]["fud"] < 0


def test_summarize_best_worst_top3() -> None:
    condition_means = {
        "cell_types+cell_state+microenv": {"aji": 0.654},
        "cell_types+cell_state+vasculature+microenv": {"aji": 0.650},
        "cell_types+cell_state+vasculature": {"aji": 0.648},
        "cell_types+microenv": {"aji": 0.600},
        "cell_types": {"aji": 0.500},
    }
    condition_stats = {
        "cell_types+cell_state+microenv": {"aji": (0.654, 0.010)},
        "cell_types+cell_state+vasculature+microenv": {"aji": (0.650, 0.008)},
        "cell_types+cell_state+vasculature": {"aji": (0.648, 0.009)},
        "cell_types+microenv": {"aji": (0.600, 0.007)},
        "cell_types": {"aji": (0.500, 0.020)},
    }

    result = summarize_best_worst(condition_means, ["aji"], condition_stats, n=3)

    assert result["aji"]["total"] == 5
    assert len(result["aji"]["best"]) == 3
    assert result["aji"]["best"] == [
        ("cell_types+cell_state+microenv", 0.654, 0.010),
        ("cell_types+cell_state+vasculature+microenv", 0.650, 0.008),
        ("cell_types+cell_state+vasculature", 0.648, 0.009),
    ]
    assert result["aji"]["best"][0][1] == pytest.approx(0.654)
    assert result["aji"]["best"][0][2] == pytest.approx(0.010)


def test_summarize_best_worst_worst3() -> None:
    condition_means = {
        "cell_types+cell_state+microenv": {"aji": 0.900},
        "cell_types+cell_state+vasculature+microenv": {"aji": 0.800},
        "cell_types+cell_state+vasculature": {"aji": 0.700},
        "cell_types+microenv": {"aji": 0.600},
        "cell_types": {"aji": 0.500},
    }
    condition_stats = {
        "cell_types+cell_state+microenv": {"aji": (0.900, 0.010)},
        "cell_types+cell_state+vasculature+microenv": {"aji": (0.800, 0.008)},
        "cell_types+cell_state+vasculature": {"aji": (0.700, 0.009)},
        "cell_types+microenv": {"aji": (0.600, 0.007)},
        "cell_types": {"aji": (0.500, 0.020)},
    }

    result = summarize_best_worst(condition_means, ["aji"], condition_stats, n=3)

    assert result["aji"]["total"] == 5
    assert result["aji"]["best"][0] == ("cell_types+cell_state+microenv", 0.900, 0.010)
    worst_keys = [item[0] for item in result["aji"]["worst"]]
    assert "cell_types+cell_state+microenv" not in worst_keys
    assert "cell_types+cell_state+vasculature+microenv" not in worst_keys
    assert "cell_types+cell_state+vasculature" not in worst_keys
    assert result["aji"]["worst"] == [
        ("cell_types+microenv", 0.600, 0.007),
        ("cell_types", 0.500, 0.020),
    ]


def test_load_leave_one_out_summary_selects_representative_tile(tmp_path: Path) -> None:
    loo_root = tmp_path / "leave_one_out"
    tiles = {
        "tile_a": {
            "cell_types": {"mean_diff": 2.0, "max_diff": 10.0, "pct_pixels_above_10": 5.0},
            "cell_state": {"mean_diff": 8.0, "max_diff": 30.0, "pct_pixels_above_10": 20.0},
            "vasculature": {"mean_diff": 3.0, "max_diff": 12.0, "pct_pixels_above_10": 6.0},
            "microenv": {"mean_diff": 7.0, "max_diff": 28.0, "pct_pixels_above_10": 18.0},
        },
        "tile_b": {
            "cell_types": {"mean_diff": 4.0, "max_diff": 14.0, "pct_pixels_above_10": 10.0},
            "cell_state": {"mean_diff": 16.0, "max_diff": 42.0, "pct_pixels_above_10": 38.0},
            "vasculature": {"mean_diff": 5.0, "max_diff": 16.0, "pct_pixels_above_10": 12.0},
            "microenv": {"mean_diff": 15.0, "max_diff": 40.0, "pct_pixels_above_10": 34.0},
        },
        "tile_c": {
            "cell_types": {"mean_diff": 6.0, "max_diff": 18.0, "pct_pixels_above_10": 15.0},
            "cell_state": {"mean_diff": 24.0, "max_diff": 54.0, "pct_pixels_above_10": 56.0},
            "vasculature": {"mean_diff": 7.0, "max_diff": 20.0, "pct_pixels_above_10": 16.0},
            "microenv": {"mean_diff": 23.0, "max_diff": 52.0, "pct_pixels_above_10": 50.0},
        },
    }

    for tile_id, payload in tiles.items():
        tile_dir = loo_root / tile_id
        tile_dir.mkdir(parents=True, exist_ok=True)
        (tile_dir / "leave_one_out_diff_stats.json").write_text(json.dumps(payload), encoding="utf-8")

    summary, representative_tile = load_leave_one_out_summary(loo_root)

    assert representative_tile == "tile_b"
    assert round(summary["cell_state"]["mean_diff"], 3) == 16.0
    assert round(summary["microenv"]["pct_pixels_above_10"], 3) == 34.0
