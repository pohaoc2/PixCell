"""Build SI_A3_zero_init.png: loss curves, divergence bar, and metric table."""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.paper_figures.style import (
    FONT_SIZE_ANNOTATION,
    FONT_SIZE_LABEL,
    FONT_SIZE_TICK,
    FONT_SIZE_TITLE,
)
from tools.ablation_a3.aggregate_stability import _read_log


def _load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_loss_curves(seed_log_paths: list[Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not seed_log_paths:
        return np.array([], dtype=int), np.array([], dtype=float), np.array([], dtype=float)
    step_lists: list[list[int]] = []
    loss_by_step: list[dict[int, float]] = []
    for path in seed_log_paths:
        entries = _read_log(Path(path))
        values: dict[int, float] = {}
        for entry in entries:
            loss = float(entry.get("loss", float("nan")))
            if not math.isnan(loss):
                values[int(entry["step"])] = loss
        if values:
            step_lists.append(sorted(values))
            loss_by_step.append(values)
    if not step_lists:
        return np.array([], dtype=int), np.array([], dtype=float), np.array([], dtype=float)
    common = sorted(set(step_lists[0]).intersection(*step_lists[1:]))
    if not common:
        common = sorted(set().union(*[set(steps) for steps in step_lists]))
    arr = np.full((len(loss_by_step), len(common)), np.nan, dtype=float)
    for row_idx, values in enumerate(loss_by_step):
        for col_idx, step in enumerate(common):
            if step in values:
                arr[row_idx, col_idx] = values[step]
    return np.asarray(common), np.nanmean(arr, axis=0), np.nanstd(arr, axis=0)


def _draw_loss_panel(ax: plt.Axes, seeds_true: list[Path], seeds_false: list[Path]) -> None:
    for paths, label, color in (
        (seeds_true, "zero_init=True", "#2b6cb0"),
        (seeds_false, "zero_init=False", "#c53030"),
    ):
        steps, mean, std = _load_loss_curves(paths)
        if len(steps) == 0:
            continue
        ax.plot(steps, mean, label=label, color=color, linewidth=1.8)
        ax.fill_between(steps, mean - std, mean + std, color=color, alpha=0.18, linewidth=0)
    ax.set_xscale("log")
    ax.set_xlabel("Training step", fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel("Loss, mean +/- SD", fontsize=FONT_SIZE_LABEL)
    ax.tick_params(labelsize=FONT_SIZE_TICK)
    ax.legend(fontsize=FONT_SIZE_ANNOTATION, frameon=False)
    ax.set_title("Training-loss stability", fontsize=FONT_SIZE_TITLE, loc="left")


def _draw_divergence_bar(ax: plt.Axes, summary_true: dict, summary_false: dict) -> None:
    summaries = [summary_true, summary_false]
    labels = ["zero_init=True", "zero_init=False"]
    colors = ["#2b6cb0", "#c53030"]
    counts = [int(s.get("divergence_count", 0)) for s in summaries]
    totals = [max(1, len(s.get("per_seed", []))) for s in summaries]
    fractions = [count / total for count, total in zip(counts, totals, strict=True)]
    ax.bar(labels, fractions, color=colors, edgecolor="black", linewidth=0.8)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Diverged seeds", fontsize=FONT_SIZE_LABEL)
    ax.tick_params(labelsize=FONT_SIZE_TICK)
    ax.set_title("Divergence rate", fontsize=FONT_SIZE_TITLE, loc="left")
    for idx, (count, total, frac) in enumerate(zip(counts, totals, fractions, strict=True)):
        ax.text(idx, min(0.98, frac + 0.04), f"{count}/{total}", ha="center", fontsize=FONT_SIZE_ANNOTATION)


def _format_float(value, fmt: str) -> str:
    if value is None:
        return "-"
    try:
        return fmt.format(float(value))
    except (TypeError, ValueError):
        return str(value)


def _draw_summary_table(ax: plt.Axes, metrics_summary: dict) -> None:
    ax.axis("off")
    cols = ("Variant", "Loss@step", "Diverged", "FID", "UNI-cos", "Cell r", "Type KL", "Nuc KS")
    x_positions = [0.01, 0.24, 0.40, 0.52, 0.62, 0.73, 0.84, 0.94]
    for x, label in zip(x_positions, cols, strict=True):
        ax.text(x, 0.95, label, fontsize=FONT_SIZE_LABEL, fontweight="bold", va="top", ha="center" if x > 0.2 else "left")
    for row_idx, row in enumerate(metrics_summary.get("rows", [])):
        y = 0.78 - row_idx * 0.20
        loss_text = "-"
        if row.get("loss_mean") is not None:
            loss_text = f"{float(row['loss_mean']):.3f} +/- {float(row.get('loss_std', 0.0)):.3f}"
        cells = [
            str(row.get("variant", row.get("key", "?"))),
            loss_text,
            f"{row.get('divergence_count', 0)}/{row.get('n_seeds', 0)}",
            _format_float(row.get("fid"), "{:.2f}"),
            _format_float(row.get("uni_cos"), "{:.3f}"),
            _format_float(row.get("cellvit_count_r"), "{:.3f}"),
            _format_float(row.get("cellvit_type_kl"), "{:.3f}"),
            _format_float(row.get("cellvit_nuc_ks"), "{:.3f}"),
        ]
        for x, text in zip(x_positions, cells, strict=True):
            ax.text(x, y, text, fontsize=FONT_SIZE_ANNOTATION, va="top", ha="center" if x > 0.2 else "left")
    ax.set_title("A3 summary metrics", fontsize=FONT_SIZE_TITLE, loc="left", pad=2)


def build_si_a3_zero_init_figure(
    *,
    seeds_true_logs: list[Path],
    seeds_false_logs: list[Path],
    stability_summary_true_path: Path,
    stability_summary_false_path: Path,
    metrics_summary_path: Path,
) -> plt.Figure:
    """Build the A3 SI figure from training logs and metric summaries."""
    summary_true = _load_json(stability_summary_true_path)
    summary_false = _load_json(stability_summary_false_path)
    metrics_summary = _load_json(metrics_summary_path)

    fig = plt.figure(figsize=(15.8, 11.0))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.6, 1.0, 1.35], hspace=0.45)
    _draw_loss_panel(fig.add_subplot(gs[0]), seeds_true_logs, seeds_false_logs)
    _draw_divergence_bar(fig.add_subplot(gs[1]), summary_true, summary_false)
    _draw_summary_table(fig.add_subplot(gs[2]), metrics_summary)
    fig.suptitle("SI A3: Zero-init residual gating stability", fontsize=FONT_SIZE_TITLE, y=0.995)
    return fig
