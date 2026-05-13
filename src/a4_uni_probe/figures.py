"""Render summary figures for probe, sweep, and null results."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src._tasklib.io import ensure_directory


def _appearance_metric_title(metric_name: str) -> str:
    title = metric_name.removeprefix("appearance.")
    title = title.replace("texture_h_", "H texture ")
    title = title.replace("texture_e_", "E texture ")
    title = title.replace("stain_vector_angle_deg", "stain angle (deg)")
    title = title.replace("_", " ")
    return title.title()


def _appearance_attr_order(rows: list[dict[str, str]]) -> list[str]:
    preferred = ["eccentricity_mean", "nuclear_area_mean", "nuclei_density"]
    seen = {row["attr"] for row in rows}
    ordered = [attr for attr in preferred if attr in seen]
    ordered.extend(sorted(seen - set(ordered)))
    return ordered


def _render_grouped_metric_grid(
    rows: list[dict[str, str]],
    *,
    metric_key_specs: list[tuple[str, str, str]],
    title: str,
    y_label: str,
    panel_path: Path,
) -> Path:
    metric_names = sorted({row["metric"] for row in rows})
    attrs = _appearance_attr_order(rows)
    index = {(row["metric"], row["attr"]): row for row in rows}

    ncols = 3
    nrows = int(np.ceil(len(metric_names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15.5, max(6.5, nrows * 3.4)), squeeze=False)
    x = np.arange(len(attrs))
    width = 0.24 if len(metric_key_specs) == 3 else 0.32
    offsets = np.linspace(-(len(metric_key_specs) - 1) / 2, (len(metric_key_specs) - 1) / 2, len(metric_key_specs)) * width

    for axis, metric_name in zip(axes.flat, metric_names):
        for offset, (value_key, label, color) in zip(offsets, metric_key_specs):
            values = []
            for attr in attrs:
                row = index.get((metric_name, attr))
                values.append(float(row[value_key]) if row and row.get(value_key) not in (None, "") else float("nan"))
            axis.bar(x + offset, values, width=width, label=label, color=color)
        axis.axhline(0.0, color="#4a5568", linewidth=0.8)
        axis.set_title(_appearance_metric_title(metric_name), fontsize=10)
        axis.set_xticks(x)
        axis.set_xticklabels(attrs, rotation=35, ha="right", fontsize=8)
        axis.tick_params(axis="y", labelsize=8)

    for axis in axes.flat[len(metric_names) :]:
        axis.axis("off")

    axes[0, 0].legend(frameon=False, fontsize=9, loc="best")
    fig.suptitle(title, fontsize=14)
    fig.supylabel(y_label)
    fig.tight_layout()
    fig.savefig(panel_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return panel_path


def render_panel_a(out_dir: str | Path) -> Path:
    out_path = Path(out_dir)
    figure_dir = ensure_directory(out_path / "figures")
    csv_path = out_path / "probe_results.csv"
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    attrs = [row["attr"] for row in rows]
    uni = np.asarray([float(row["uni_r2_mean"]) for row in rows], dtype=np.float32)
    tme = np.asarray([float(row["tme_r2_mean"]) for row in rows], dtype=np.float32)
    positions = np.arange(len(attrs))

    fig, ax = plt.subplots(figsize=(max(8.0, len(attrs) * 0.5), 4.5))
    ax.bar(positions - 0.18, uni, width=0.36, label="UNI", color="#2b6cb0")
    ax.bar(positions + 0.18, tme, width=0.36, label="TME", color="#dd6b20")
    ax.set_ylabel("CV R^2")
    ax.set_xticks(positions)
    ax.set_xticklabels(attrs, rotation=45, ha="right")
    ax.legend(frameon=False)
    ax.set_title("Panel A: UNI vs TME Probe Performance")
    fig.tight_layout()
    panel_path = figure_dir / "panel_a_probe_R2.png"
    fig.savefig(panel_path, dpi=200)
    plt.close(fig)
    return panel_path


def render_panel_b(out_dir: str | Path) -> Path:
    out_path = Path(out_dir)
    figure_dir = ensure_directory(out_path / "figures")
    sweep_root = out_path / "sweep"
    attrs: list[str] = []
    targeted_slopes: list[float] = []
    random_slopes: list[float] = []
    for summary_path in sorted(sweep_root.glob("*/slope_summary.json")):
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        attrs.append(str(payload["attr"]))
        targeted = payload.get("targeted", {})
        random = payload.get("random", {})
        targeted_slopes.append(float(targeted.get("slope_mean", float("nan"))))
        random_slopes.append(float(random.get("slope_mean", float("nan"))))

    positions = np.arange(len(attrs))
    fig, ax = plt.subplots(figsize=(max(6.0, len(attrs) * 0.75), 4.5))
    ax.bar(positions - 0.18, targeted_slopes, width=0.36, color="#2f855a", label="Targeted")
    ax.bar(positions + 0.18, random_slopes, width=0.36, color="#a0aec0", label="Random")
    ax.axhline(0.0, color="#4a5568", linewidth=1.0)
    ax.set_ylabel("Slope")
    ax.set_xticks(positions)
    ax.set_xticklabels(attrs, rotation=45, ha="right")
    ax.legend(frameon=False)
    ax.set_title("Panel B: Sweep Slopes")
    fig.tight_layout()
    panel_path = figure_dir / "panel_b_sweep_slope.png"
    fig.savefig(panel_path, dpi=200)
    plt.close(fig)
    return panel_path


def render_panel_c(out_dir: str | Path) -> Path:
    out_path = Path(out_dir)
    figure_dir = ensure_directory(out_path / "figures")
    null_root = out_path / "null"
    attrs: list[str] = []
    targeted_means: list[float] = []
    random_means: list[float] = []
    full_null_means: list[float] = []
    for summary_path in sorted(null_root.glob("*/null_comparison.json")):
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        attrs.append(str(payload["attr"]))
        targeted_means.append(float(payload.get("targeted", {}).get("metric_mean", float("nan"))))
        random_means.append(float(payload.get("random", {}).get("metric_mean", float("nan"))))
        full_null_means.append(float(payload.get("full_uni_null", {}).get("metric_mean", float("nan"))))

    positions = np.arange(len(attrs))
    fig, ax = plt.subplots(figsize=(max(6.0, len(attrs) * 0.8), 4.5))
    ax.plot(positions, targeted_means, marker="o", color="#c53030", label="Targeted null")
    ax.plot(positions, random_means, marker="o", color="#718096", label="Random null")
    if any(np.isfinite(full_null_means)):
        ax.plot(positions, full_null_means, marker="o", color="#2b6cb0", label="Full UNI null")
    ax.set_ylabel("Target metric")
    ax.set_xticks(positions)
    ax.set_xticklabels(attrs, rotation=45, ha="right")
    ax.legend(frameon=False)
    ax.set_title("Panel C: Null Comparison")
    fig.tight_layout()
    panel_path = figure_dir / "panel_c_null_drop.png"
    fig.savefig(panel_path, dpi=200)
    plt.close(fig)
    return panel_path


def render_panel_d(out_dir: str | Path) -> Path:
    out_path = Path(out_dir)
    figure_dir = ensure_directory(out_path / "figures")
    csv_path = out_path / "appearance_sweep_summary.csv"
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    return _render_grouped_metric_grid(
        rows,
        metric_key_specs=[
            ("targeted_slope_mean", "Targeted", "#2f855a"),
            ("random_slope_mean", "Random", "#a0aec0"),
        ],
        title="Panel D: Appearance Sweep Slopes Across All Metrics",
        y_label="Slope",
        panel_path=figure_dir / "panel_d_appearance_sweep_all_metrics.png",
    )


def render_panel_e(out_dir: str | Path) -> Path:
    out_path = Path(out_dir)
    figure_dir = ensure_directory(out_path / "figures")
    csv_path = out_path / "appearance_null_summary.csv"
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    return _render_grouped_metric_grid(
        rows,
        metric_key_specs=[
            ("targeted_mean", "Targeted null", "#c53030"),
            ("random_mean", "Random null", "#718096"),
            ("full_uni_null_mean", "Full UNI null", "#2b6cb0"),
        ],
        title="Panel E: Appearance Null Readouts Across All Metrics",
        y_label="Metric value",
        panel_path=figure_dir / "panel_e_appearance_null_all_metrics.png",
    )


def render_all(out_dir: str | Path) -> dict[str, Path]:
    out_path = Path(out_dir)
    outputs: dict[str, Path] = {}
    if (out_path / "probe_results.csv").is_file():
        outputs["panel_a"] = render_panel_a(out_path)
    if any((out_path / "sweep").glob("*/slope_summary.json")):
        outputs["panel_b"] = render_panel_b(out_path)
    if any((out_path / "null").glob("*/null_comparison.json")):
        outputs["panel_c"] = render_panel_c(out_path)
    if (out_path / "appearance_sweep_summary.csv").is_file():
        outputs["panel_d"] = render_panel_d(out_path)
    if (out_path / "appearance_null_summary.csv").is_file():
        outputs["panel_e"] = render_panel_e(out_path)
    return outputs
