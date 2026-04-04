"""Tests for classify_tiles core logic."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _write_png(path: Path, value: float, size: int = 8) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.full((size, size), int(value * 255), dtype=np.uint8)
    Image.fromarray(arr, mode="L").save(path)


def _write_npy(path: Path, value: float, size: int = 8) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.full((size, size), value, dtype=np.float32)
    np.save(path, arr)


def _make_exp_channels(tmp_path: Path) -> Path:
    """Build a minimal exp_channels directory with a few synthetic tiles."""
    ec = tmp_path / "exp_channels"
    tiles = {
        "tile_cancer": {
            "cell_masks": 0.10,
            "cell_type_cancer": 0.09,
            "cell_type_healthy": 0.005,
            "cell_type_immune": 0.005,
            "cell_state_prolif": 0.08,
            "cell_state_nonprolif": 0.01,
            "cell_state_dead": 0.01,
        },
        "tile_immune": {
            "cell_masks": 0.08,
            "cell_type_cancer": 0.01,
            "cell_type_healthy": 0.01,
            "cell_type_immune": 0.06,
            "cell_state_prolif": 0.02,
            "cell_state_nonprolif": 0.05,
            "cell_state_dead": 0.01,
        },
        "tile_blank": {
            "cell_masks": 0.0,
            "cell_type_cancer": 0.0,
            "cell_type_healthy": 0.0,
            "cell_type_immune": 0.0,
            "cell_state_prolif": 0.0,
            "cell_state_nonprolif": 0.0,
            "cell_state_dead": 0.0,
        },
        "tile_pure_healthy": {
            "cell_masks": 0.10,
            "cell_type_cancer": 0.01,
            "cell_type_healthy": 0.09,
            "cell_type_immune": 0.0,
            "cell_state_prolif": 0.01,
            "cell_state_nonprolif": 0.08,
            "cell_state_dead": 0.01,
        },
        "tile_pure_prolif": {
            "cell_masks": 0.10,
            "cell_type_cancer": 0.08,
            "cell_type_healthy": 0.01,
            "cell_type_immune": 0.01,
            "cell_state_prolif": 0.09,
            "cell_state_nonprolif": 0.005,
            "cell_state_dead": 0.005,
        },
    }
    png_channels = [
        "cell_masks",
        "cell_type_cancer",
        "cell_type_healthy",
        "cell_type_immune",
        "cell_state_prolif",
        "cell_state_nonprolif",
        "cell_state_dead",
    ]
    for tile_id, vals in tiles.items():
        for ch in png_channels:
            _write_png(ec / ch / f"{tile_id}.png", vals[ch])
        _write_npy(ec / "oxygen" / f"{tile_id}.npy", 0.9 if tile_id != "tile_blank" else 0.95)
        _write_npy(ec / "glucose" / f"{tile_id}.npy", 0.85 if tile_id != "tile_blank" else 0.95)
    return ec


def test_compute_tile_stats_cancer_frac(tmp_path):
    from tools.stage3.classify_tiles import compute_tile_stats

    ec = _make_exp_channels(tmp_path)
    stats = compute_tile_stats("tile_cancer", ec)
    assert stats["cell_density"] == pytest.approx(0.10, abs=0.02)
    assert stats["cancer_frac"] == pytest.approx(0.9, abs=0.1)
    assert stats["immune_frac"] < 0.2


def test_compute_tile_stats_blank(tmp_path):
    from tools.stage3.classify_tiles import compute_tile_stats

    ec = _make_exp_channels(tmp_path)
    stats = compute_tile_stats("tile_blank", ec)
    assert stats["cell_density"] == pytest.approx(0.0, abs=1e-6)


def test_filter_blank_tiles(tmp_path):
    from tools.stage3.classify_tiles import compute_tile_stats, filter_blank_tiles

    ec = _make_exp_channels(tmp_path)
    all_stats = {
        tid: compute_tile_stats(tid, ec)
        for tid in ["tile_cancer", "tile_immune", "tile_blank"]
    }
    kept = filter_blank_tiles(all_stats, min_density=0.005)
    assert "tile_blank" not in kept
    assert "tile_cancer" in kept
    assert "tile_immune" in kept


def test_axis1_assignment():
    from tools.stage3.classify_tiles import assign_axis1

    thresholds = {
        "cancer_frac_p75": 0.5,
        "immune_frac_p75": 0.4,
        "healthy_frac_p75": 0.6,
        "cancer_frac_p25": 0.1,
    }
    assert assign_axis1({"cancer_frac": 0.8, "immune_frac": 0.05, "healthy_frac": 0.1}, thresholds) == "cancer"
    assert assign_axis1({"cancer_frac": 0.2, "immune_frac": 0.6, "healthy_frac": 0.1}, thresholds) == "immune"
    assert assign_axis1({"cancer_frac": 0.05, "immune_frac": 0.05, "healthy_frac": 0.8}, thresholds) == "healthy"
    assert assign_axis1({"cancer_frac": 0.3, "immune_frac": 0.3, "healthy_frac": 0.3}, thresholds) is None


def test_axis2_assignment():
    from tools.stage3.classify_tiles import assign_axis2

    thresholds = {"oxygen_p25": 0.5, "glucose_p25": 0.6}
    assert assign_axis2({"mean_oxygen": 0.3, "mean_glucose": 0.8}, thresholds) == "hypoxic"
    assert assign_axis2({"mean_oxygen": 0.8, "mean_glucose": 0.4}, thresholds) == "glucose_low"
    assert assign_axis2({"mean_oxygen": 0.8, "mean_glucose": 0.8}, thresholds) == "neutral"


def test_compute_percentile_thresholds_and_selection():
    from tools.stage3.classify_tiles import compute_percentile_thresholds, select_exp_tiles, select_representatives

    classified = {
        "t1": {
            "cancer_frac": 0.9,
            "immune_frac": 0.05,
            "healthy_frac": 0.05,
            "mean_oxygen": 0.2,
            "mean_glucose": 0.9,
            "axis1": "cancer",
            "axis2": "hypoxic",
        },
        "t2": {
            "cancer_frac": 0.2,
            "immune_frac": 0.7,
            "healthy_frac": 0.1,
            "mean_oxygen": 0.9,
            "mean_glucose": 0.1,
            "axis1": "immune",
            "axis2": "glucose_low",
        },
        "t3": {
            "cancer_frac": 0.1,
            "immune_frac": 0.1,
            "healthy_frac": 0.8,
            "mean_oxygen": 0.8,
            "mean_glucose": 0.8,
            "axis1": "healthy",
            "axis2": "neutral",
        },
    }
    thresholds = compute_percentile_thresholds(
        {
            tid: {
                k: v
                for k, v in stats.items()
                if k in {"cancer_frac", "immune_frac", "healthy_frac", "mean_oxygen", "mean_glucose"}
            }
            for tid, stats in classified.items()
        }
    )

    reps = select_representatives(classified, thresholds)
    assert reps["cancer+hypoxic"]["tile_id"] == "t1"
    assert reps["immune+glucose_low"]["tile_id"] == "t2"
    assert reps["healthy+neutral"]["tile_id"] == "t3"

    exp2, exp3 = select_exp_tiles(
        {
            "t1": {
                "cancer_frac": 0.91,
                "immune_frac": 0.1,
                "healthy_frac": 0.1,
                "prolif_frac": 0.85,
                "nonprolif_frac": 0.05,
                "dead_frac": 0.02,
            },
            "t2": {
                "cancer_frac": 0.1,
                "immune_frac": 0.92,
                "healthy_frac": 0.1,
                "prolif_frac": 0.05,
                "nonprolif_frac": 0.83,
                "dead_frac": 0.02,
            },
            "t3": {
                "cancer_frac": 0.1,
                "immune_frac": 0.1,
                "healthy_frac": 0.94,
                "prolif_frac": 0.05,
                "nonprolif_frac": 0.05,
                "dead_frac": 0.86,
            },
        },
        threshold=0.8,
    )
    assert exp2["cancer"]["tile_id"] == "t1"
    assert exp2["immune"]["tile_id"] == "t2"
    assert exp2["healthy"]["tile_id"] == "t3"
    assert exp3["prolif"]["tile_id"] == "t1"
    assert exp3["nonprolif"]["tile_id"] == "t2"
    assert exp3["dead"]["tile_id"] == "t3"


def test_main_writes_expected_json(tmp_path, monkeypatch):
    from tools.stage3 import classify_tiles as module

    exp_root = tmp_path / "data"
    ec = _make_exp_channels(exp_root)

    monkeypatch.setattr(
        module,
        "_discover_tile_ids",
        lambda _exp_channels_dir: ["tile_cancer", "tile_immune", "tile_pure_healthy", "tile_pure_prolif"],
    )
    monkeypatch.setattr(
        module,
        "compute_tile_stats",
        lambda tile_id, _exp_channels_dir: {
            "tile_cancer": {
                "cell_density": 0.1,
                "cancer_frac": 0.9,
                "immune_frac": 0.05,
                "healthy_frac": 0.05,
                "prolif_frac": 0.85,
                "nonprolif_frac": 0.05,
                "dead_frac": 0.02,
                "mean_oxygen": 0.2,
                "mean_glucose": 0.9,
            },
            "tile_immune": {
                "cell_density": 0.1,
                "cancer_frac": 0.2,
                "immune_frac": 0.88,
                "healthy_frac": 0.1,
                "prolif_frac": 0.1,
                "nonprolif_frac": 0.82,
                "dead_frac": 0.02,
                "mean_oxygen": 0.9,
                "mean_glucose": 0.1,
            },
            "tile_pure_healthy": {
                "cell_density": 0.1,
                "cancer_frac": 0.05,
                "immune_frac": 0.05,
                "healthy_frac": 0.95,
                "prolif_frac": 0.1,
                "nonprolif_frac": 0.1,
                "dead_frac": 0.86,
                "mean_oxygen": 0.8,
                "mean_glucose": 0.8,
            },
            "tile_pure_prolif": {
                "cell_density": 0.1,
                "cancer_frac": 0.8,
                "immune_frac": 0.05,
                "healthy_frac": 0.15,
                "prolif_frac": 0.91,
                "nonprolif_frac": 0.03,
                "dead_frac": 0.03,
                "mean_oxygen": 0.85,
                "mean_glucose": 0.85,
            },
        }[tile_id],
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "classify_tiles.py",
            "--exp-root",
            str(exp_root),
            "--out",
            str(tmp_path / "tile_classes.json"),
            "--exp-threshold",
            "0.8",
        ],
    )

    module.main()

    out = json.loads((tmp_path / "tile_classes.json").read_text(encoding="utf-8"))
    assert out["thresholds"]["cell_density_p5"] == pytest.approx(0.1)
    assert "cancer+hypoxic" in out["representatives"]
    assert out["exp2_tiles"]["cancer"]["tile_id"] == "tile_cancer"
    assert out["exp3_tiles"]["prolif"]["tile_id"] == "tile_pure_prolif"
