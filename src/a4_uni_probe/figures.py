"""Render summary figures for probe, sweep, and null results."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src._tasklib.io import ensure_directory


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


def render_all(out_dir: str | Path) -> dict[str, Path]:
    out_path = Path(out_dir)
    outputs: dict[str, Path] = {}
    if (out_path / "probe_results.csv").is_file():
        outputs["panel_a"] = render_panel_a(out_path)
    if any((out_path / "sweep").glob("*/slope_summary.json")):
        outputs["panel_b"] = render_panel_b(out_path)
    if any((out_path / "null").glob("*/null_comparison.json")):
        outputs["panel_c"] = render_panel_c(out_path)
    return outputs
