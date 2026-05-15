"""Walk sweep PNGs, recompute appearance metrics per compartment, fit slopes.

Outputs (alongside the existing run dir):
- sweep/<attr>/metrics_regional.csv  — per-PNG regional row keyed by (tile, direction, alpha)
- appearance_regional_sweep_summary.csv  — per (attr, regional metric) slope summary
- appearance_global_vs_regional.csv  — global/nuc/stroma targeted-slope side-by-side
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from src._tasklib.io import ensure_directory
from src.a4_uni_probe.appearance_metrics_regional import (
    appearance_row_regional,
    regional_metric_keys,
)
from src.a4_uni_probe.slope_stats import bootstrap_slope_summary


def _safe_float(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _compute_attr_regional_metrics(attr_dir: Path) -> Path | None:
    """Augment each sweep row with regional metrics and write metrics_regional.csv."""
    metrics_path = attr_dir / "metrics.csv"
    if not metrics_path.is_file():
        return None
    rows = list(csv.DictReader(metrics_path.open(encoding="utf-8")))
    if not rows:
        return None

    augmented: list[dict[str, object]] = []
    for row in rows:
        image_path = row.get("image_path", "")
        if image_path and Path(image_path).is_file():
            try:
                regional = appearance_row_regional(image_path)
            except Exception:
                regional = {key: float("nan") for key in regional_metric_keys()}
        else:
            regional = {key: float("nan") for key in regional_metric_keys()}
        merged = dict(row)
        merged.update({k: float(v) for k, v in regional.items()})
        augmented.append(merged)

    out_path = attr_dir / "metrics_regional.csv"
    _write_csv(out_path, augmented)
    return out_path


def _summarize_attr(rows: list[dict[str, str]], attr: str, *, n_boot: int = 400) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    for metric_name in regional_metric_keys():
        row: dict[str, object] = {"attr": attr, "metric": metric_name}
        for direction_name in ("targeted", "random"):
            direction_rows = [r for r in rows if r.get("direction") == direction_name]
            alphas = np.asarray([_safe_float(r.get("alpha")) for r in direction_rows], dtype=np.float32)
            values = np.asarray([_safe_float(r.get(metric_name)) for r in direction_rows], dtype=np.float32)
            stats = bootstrap_slope_summary(alphas, values, n_boot=n_boot, seed=0)
            ci_low, ci_high = stats["slope_ci95"]
            row[f"{direction_name}_slope_mean"] = stats["slope_mean"]
            row[f"{direction_name}_slope_ci95_low"] = ci_low
            row[f"{direction_name}_slope_ci95_high"] = ci_high
            row[f"{direction_name}_n"] = stats["n"]
        summary.append(row)
    return summary


def _load_global_appearance_summary(out_dir: Path) -> dict[tuple[str, str], float]:
    """Map (attr, base_metric_key without prefix) -> global targeted slope from appearance_sweep_summary.csv."""
    path = out_dir / "appearance_sweep_summary.csv"
    if not path.is_file():
        return {}
    out: dict[tuple[str, str], float] = {}
    for row in csv.DictReader(path.open(encoding="utf-8")):
        attr = row["attr"]
        metric = row["metric"]
        base = metric.removeprefix("appearance.")
        out[(attr, base)] = _safe_float(row.get("targeted_slope_mean"))
    return out


def _build_global_vs_regional(regional_rows: list[dict[str, object]], global_index: dict[tuple[str, str], float]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    by_attr_metric: dict[tuple[str, str], dict[str, float]] = {}
    for row in regional_rows:
        attr = str(row["attr"])
        metric = str(row["metric"])
        targeted = _safe_float(row.get("targeted_slope_mean"))
        # metric is like appearance.nuc.h_mean or appearance.stroma.texture_h_contrast
        parts = metric.split(".", 2)
        if len(parts) != 3:
            continue
        compartment, base = parts[1], parts[2]
        entry = by_attr_metric.setdefault((attr, base), {})
        entry[f"{compartment}_targeted_slope"] = targeted

    for (attr, base), entry in by_attr_metric.items():
        global_slope = global_index.get((attr, base), float("nan"))
        nuc = entry.get("nuc_targeted_slope", float("nan"))
        stroma = entry.get("stroma_targeted_slope", float("nan"))
        if np.isfinite(global_slope) and abs(global_slope) > 0.0 and np.isfinite(nuc):
            nuc_frac = float(nuc / global_slope)
        else:
            nuc_frac = float("nan")
        out.append({
            "attr": attr,
            "metric": f"appearance.{base}",
            "global_targeted_slope": global_slope,
            "nuc_targeted_slope": nuc,
            "stroma_targeted_slope": stroma,
            "nuc_fraction_of_global": nuc_frac,
        })
    out.sort(key=lambda r: (r["attr"], r["metric"]))
    return out


def run_regional(args: argparse.Namespace) -> dict[str, Path]:
    out_dir = ensure_directory(args.out_dir)
    sweep_root = out_dir / "sweep"
    if not sweep_root.is_dir():
        raise FileNotFoundError(f"no sweep/ directory under {out_dir}; run `sweep` first")

    attr_dirs = sorted(p for p in sweep_root.iterdir() if p.is_dir() and (p / "metrics.csv").is_file())
    if not attr_dirs:
        raise FileNotFoundError(f"no sweep/<attr>/metrics.csv under {sweep_root}")

    summary_rows: list[dict[str, object]] = []
    for attr_dir in attr_dirs:
        regional_metrics_path = _compute_attr_regional_metrics(attr_dir)
        if regional_metrics_path is None:
            continue
        regional_rows = list(csv.DictReader(regional_metrics_path.open(encoding="utf-8")))
        summary_rows.extend(_summarize_attr(regional_rows, attr=attr_dir.name))

    summary_path = out_dir / "appearance_regional_sweep_summary.csv"
    _write_csv(summary_path, summary_rows)

    global_index = _load_global_appearance_summary(out_dir)
    side_by_side = _build_global_vs_regional(summary_rows, global_index)
    comparison_path = out_dir / "appearance_global_vs_regional.csv"
    _write_csv(comparison_path, side_by_side)

    return {"regional_summary": summary_path, "global_vs_regional": comparison_path}
