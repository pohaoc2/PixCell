"""Tests for the specificity-matrix module."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pytest

from src.a4_uni_probe.slope_stats import bootstrap_slope_summary, linear_slope
from src.a4_uni_probe.specificity import _baseline_std_per_metric, run_specificity


def _verbatim_old_bootstrap(alphas: np.ndarray, values: np.ndarray, *, n_boot: int, seed: int = 0) -> dict[str, float]:
    """Recreate the pre-refactor bootstrap loop exactly for regression-pinning."""
    alphas = np.asarray(alphas, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    valid = np.isfinite(alphas) & np.isfinite(values)
    n_valid = int(valid.sum())
    if n_valid < 2:
        return {"slope_mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": n_valid}
    x = alphas[valid]
    y = values[valid]
    rng = np.random.default_rng(seed)
    slopes: list[float] = []
    for _ in range(n_boot):
        choice = rng.integers(0, len(x), size=len(x))
        slopes.append(float(linear_slope(x[choice], y[choice])))
    return {
        "slope_mean": float(np.mean(slopes)),
        "ci_low": float(np.quantile(slopes, 0.025)),
        "ci_high": float(np.quantile(slopes, 0.975)),
        "n": n_valid,
    }


def test_bootstrap_slope_summary_matches_verbatim_old_loop():
    rng = np.random.default_rng(123)
    alphas = np.tile([-1.0, 0.0, 1.0], 30).astype(np.float32)
    values = (alphas * 2.5 + rng.normal(scale=0.4, size=alphas.size)).astype(np.float32)

    new = bootstrap_slope_summary(alphas, values, n_boot=400, seed=0)
    old = _verbatim_old_bootstrap(alphas, values, n_boot=400, seed=0)

    assert new["slope_mean"] == old["slope_mean"]
    assert new["slope_ci95"] == (old["ci_low"], old["ci_high"])
    assert new["n"] == old["n"]


def test_bootstrap_slope_summary_handles_too_few_finite():
    alphas = np.asarray([0.0, np.nan, np.nan], dtype=np.float32)
    values = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
    stats = bootstrap_slope_summary(alphas, values, n_boot=10, seed=0)
    assert stats["n"] == 1
    assert not np.isfinite(stats["slope_mean"])
    low, high = stats["slope_ci95"]
    assert not np.isfinite(low) and not np.isfinite(high)


def _write_synthetic_sweep(out_dir: Path, attr: str, *, slope_targeted: float, slope_random: float, n_tiles: int = 20, seed: int = 0) -> None:
    attr_dir = out_dir / "sweep" / attr
    attr_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for tile in range(n_tiles):
        for direction, slope in (("targeted", slope_targeted), ("random", slope_random)):
            for alpha in (-1.0, 0.0, 1.0):
                target_value = slope * alpha + float(rng.normal(scale=0.05))
                rows.append({
                    "tile_id": f"tile_{tile}",
                    "direction": direction,
                    "alpha": alpha,
                    "target_attr": attr,
                    "image_path": "",
                    "target_value": target_value,
                    "morpho.eccentricity_mean": target_value if attr == "eccentricity_mean" else float(rng.normal(scale=0.05)),
                    "morpho.nuclear_area_mean": float(rng.normal(scale=10.0)),
                    "appearance.h_mean": float(rng.normal(scale=0.01)),
                    "appearance.texture_h_contrast": float(rng.normal(scale=0.5)),
                })
    with (attr_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_run_specificity_emits_expected_csvs(tmp_path: Path):
    out_dir = tmp_path / "run"
    _write_synthetic_sweep(out_dir, "eccentricity_mean", slope_targeted=0.06, slope_random=-0.002, seed=1)
    _write_synthetic_sweep(out_dir, "nuclear_area_mean", slope_targeted=80.0, slope_random=-20.0, seed=2)

    args = argparse.Namespace(out_dir=out_dir)
    paths = run_specificity(args)

    full_rows = list(csv.DictReader(paths["full"].open(encoding="utf-8")))
    morpho_rows = list(csv.DictReader(paths["morphology_summary"].open(encoding="utf-8")))

    n_edited = 2
    n_morpho_metrics = 2
    n_appearance_metrics = 2
    assert len(full_rows) == n_edited * (n_morpho_metrics + n_appearance_metrics)
    assert len(morpho_rows) == n_edited * n_morpho_metrics

    for row in full_rows:
        family = row["family"]
        assert family in {"morpho", "appearance"}
        assert row["measured_metric"].startswith(f"{family}.")


def test_baseline_std_is_nan_for_constant_metric(tmp_path: Path):
    attr_dir = tmp_path / "sweep" / "eccentricity_mean"
    attr_dir.mkdir(parents=True)
    rows = [
        {"tile_id": f"t{i}", "direction": "targeted", "alpha": 0.0, "target_attr": "x",
         "image_path": "", "target_value": 0.0, "morpho.constant": 0.5, "appearance.varying": float(i)}
        for i in range(5)
    ]
    with (attr_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    stds = _baseline_std_per_metric(tmp_path, ["morpho.constant", "appearance.varying"])
    assert stds["morpho.constant"] == 0.0
    assert np.isfinite(stds["appearance.varying"]) and stds["appearance.varying"] > 0.0


def test_panel_f_renders(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    _write_synthetic_sweep(tmp_path, "eccentricity_mean", slope_targeted=0.06, slope_random=-0.002, seed=1)
    _write_synthetic_sweep(tmp_path, "nuclear_area_mean", slope_targeted=80.0, slope_random=-20.0, seed=2)
    args = argparse.Namespace(out_dir=tmp_path)
    run_specificity(args)

    from src.a4_uni_probe.figures import render_panel_f

    panel_path = render_panel_f(tmp_path)
    assert panel_path.is_file()
    assert panel_path.stat().st_size > 1000
