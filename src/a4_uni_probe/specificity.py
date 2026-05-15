"""Specificity matrix: per-(edited_attr, measured_metric) slope summaries.

For every sweep/<attr>/metrics.csv produced by run_sweep, fit targeted and random
slopes for every column starting with `morpho.` or `appearance.`. Emit a long-form
specificity_full.csv plus a morphology-side summary mirroring appearance_sweep_summary.csv.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from src._tasklib.io import ensure_directory
from src.a4_uni_probe.slope_stats import bootstrap_slope_summary


def _safe_float(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _list_measured_metrics(rows: list[dict[str, str]]) -> tuple[list[str], list[str]]:
    if not rows:
        return [], []
    morpho = [k for k in rows[0].keys() if k.startswith("morpho.")]
    appearance = [k for k in rows[0].keys() if k.startswith("appearance.")]
    return morpho, appearance


def _summarize_for_metric(rows: list[dict[str, str]], metric_name: str, n_boot: int = 400) -> dict[str, object]:
    out: dict[str, object] = {"metric": metric_name}
    for direction_name in ("targeted", "random"):
        direction_rows = [r for r in rows if r.get("direction") == direction_name]
        alphas = np.asarray([_safe_float(r.get("alpha")) for r in direction_rows], dtype=np.float32)
        values = np.asarray([_safe_float(r.get(metric_name)) for r in direction_rows], dtype=np.float32)
        stats = bootstrap_slope_summary(alphas, values, n_boot=n_boot, seed=0)
        ci_low, ci_high = stats["slope_ci95"]
        out[f"{direction_name}_slope_mean"] = stats["slope_mean"]
        out[f"{direction_name}_slope_ci95_low"] = ci_low
        out[f"{direction_name}_slope_ci95_high"] = ci_high
        out[f"{direction_name}_n"] = stats["n"]
    return out


def _baseline_std_per_metric(out_dir: Path, metric_names: list[str]) -> dict[str, float]:
    """Standard deviation of each metric across all sweep alpha=0 rows.

    Used to z-score targeted slopes so the specificity heatmap is comparable
    across metrics with very different units.
    """
    pooled: dict[str, list[float]] = {name: [] for name in metric_names}
    for metrics_path in sorted(out_dir.glob("sweep/*/metrics.csv")):
        for row in csv.DictReader(metrics_path.open(encoding="utf-8")):
            if row.get("direction") != "targeted":
                continue
            try:
                if float(row.get("alpha", "nan")) != 0.0:
                    continue
            except ValueError:
                continue
            for name in metric_names:
                val = _safe_float(row.get(name))
                if np.isfinite(val):
                    pooled[name].append(val)
    stds: dict[str, float] = {}
    for name, vals in pooled.items():
        if len(vals) < 2:
            stds[name] = float("nan")
        else:
            stds[name] = float(np.std(np.asarray(vals, dtype=np.float64), ddof=1))
    return stds


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_specificity(args: argparse.Namespace) -> dict[str, Path]:
    out_dir = ensure_directory(args.out_dir)
    sweep_root = out_dir / "sweep"
    if not sweep_root.is_dir():
        raise FileNotFoundError(f"no sweep/ directory under {out_dir}; run `sweep` first")

    attr_dirs = sorted(p for p in sweep_root.iterdir() if p.is_dir() and (p / "metrics.csv").is_file())
    if not attr_dirs:
        raise FileNotFoundError(f"no sweep/<attr>/metrics.csv under {sweep_root}")

    first_rows = list(csv.DictReader((attr_dirs[0] / "metrics.csv").open(encoding="utf-8")))
    morpho_names, appearance_names = _list_measured_metrics(first_rows)
    all_metric_names = morpho_names + appearance_names
    baseline_stds = _baseline_std_per_metric(out_dir, all_metric_names)

    morpho_summary_rows: list[dict[str, object]] = []
    full_rows: list[dict[str, object]] = []

    for attr_dir in attr_dirs:
        attr = attr_dir.name
        rows = list(csv.DictReader((attr_dir / "metrics.csv").open(encoding="utf-8")))
        for metric_name in all_metric_names:
            stats = _summarize_for_metric(rows, metric_name)
            family = "morpho" if metric_name.startswith("morpho.") else "appearance"

            if family == "morpho":
                morpho_summary_rows.append({
                    "attr": attr,
                    "metric": metric_name,
                    **{k: v for k, v in stats.items() if k != "metric"},
                })

            targeted = float(stats["targeted_slope_mean"])
            random_ = float(stats["random_slope_mean"])
            std = baseline_stds.get(metric_name, float("nan"))
            if np.isfinite(targeted) and np.isfinite(random_) and abs(random_) > 0.0:
                abs_ratio = abs(targeted) / abs(random_)
            else:
                abs_ratio = float("nan")
            if np.isfinite(std) and std > 0.0 and np.isfinite(targeted):
                normalized = targeted / std
            else:
                normalized = float("nan")
            full_rows.append({
                "edited_attr": attr,
                "measured_metric": metric_name,
                "family": family,
                "targeted_slope": targeted,
                "random_slope": random_,
                "abs_ratio": abs_ratio,
                "baseline_std": std,
                "normalized_targeted_slope": normalized,
            })

    morpho_csv = out_dir / "specificity_morphology_summary.csv"
    _write_csv(morpho_csv, morpho_summary_rows)
    full_csv = out_dir / "specificity_full.csv"
    _write_csv(full_csv, full_rows)
    return {"morphology_summary": morpho_csv, "full": full_csv}
