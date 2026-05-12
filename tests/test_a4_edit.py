"""Unit tests for a4_uni_probe.edit."""

from __future__ import annotations

import csv
import json

import numpy as np

from src.a4_uni_probe.edit import _summarize_slopes, null_uni, random_unit_direction, sweep_uni


def test_sweep_uni_is_linear_and_preserves_zero_alpha():
    uni = np.array([3.0, 4.0], dtype=np.float32)
    direction = np.array([1.0, 0.0], dtype=np.float32)
    edits = sweep_uni(uni, direction, [-1.0, 0.0, 1.0])
    np.testing.assert_allclose(edits[1], uni)
    np.testing.assert_allclose(edits[2] - edits[1], edits[1] - edits[0])


def test_null_uni_removes_projection():
    uni = np.array([2.0, 1.0], dtype=np.float32)
    direction = np.array([1.0, 0.0], dtype=np.float32)
    nulled = null_uni(uni, direction)
    assert abs(float(np.dot(nulled, direction))) < 1e-6


def test_random_unit_direction_is_seeded_and_normalized():
    left = random_unit_direction(5, seed=123)
    right = random_unit_direction(5, seed=123)
    np.testing.assert_allclose(left, right)
    assert np.isclose(np.linalg.norm(left), 1.0)


def test_summarize_slopes_detects_monotonic(tmp_path):
    csv_path = tmp_path / "metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["alpha", "target_value", "direction"])
        writer.writeheader()
        for direction, slope in (("targeted", 1.0), ("random", 0.0)):
            for alpha in (-2, -1, 0, 1, 2):
                writer.writerow({"alpha": alpha, "target_value": slope * alpha + 0.01 * alpha, "direction": direction})
    out_path = tmp_path / "slope_summary.json"
    _summarize_slopes(csv_path, out_path, "test_attr")
    summary = json.loads(out_path.read_text(encoding="utf-8"))
    assert summary["targeted"]["slope_mean"] > 0.5
    assert summary["pass_criterion_met"] is True