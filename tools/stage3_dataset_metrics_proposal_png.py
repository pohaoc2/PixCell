#!/usr/bin/env python3
"""Render static PNG mockups for dataset-level ablation metric proposals."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.stage3_dataset_metrics_proposal import (
    CARD_COLORS,
    METRIC_HIGHER_IS_BETTER,
    METRIC_LABELS,
    METRICS,
    _build_payload,
)

METRICS_NO_FID = tuple(m for m in METRICS if m != "fid")


def _rows(payload: dict) -> list[dict]:
    return list(payload["rows"])


def _metric_limits(metric: str) -> tuple[float, float]:
    if metric == "cosine":
        return -1.0, 1.0
    return 0.0, 1.0


def _cardinality_dot(ax, x: float, y: float, card: int) -> None:
    ax.scatter([x], [y], s=22, color=CARD_COLORS[card], edgecolor="white", linewidth=0.5, zorder=4)


def _style_figure(fig, title: str, subtitle: str) -> None:
    fig.patch.set_facecolor("#f8f5ee")
    fig.suptitle(title, x=0.03, y=0.985, ha="left", va="top", fontsize=20, fontweight="bold")
    fig.text(0.03, 0.955, subtitle, ha="left", va="top", fontsize=10, color="#555555")


def render_option_a(payload: dict, out_path: Path) -> Path:
    rows = _rows(payload)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    axes = axes.flatten()
    _style_figure(
        fig,
        "Option A: Small-Multiple Bar Charts",
        "Mean ± std across all tiles for each condition. FID remains a reserved dataset-level panel.",
    )

    for ax, metric in zip(axes, METRICS):
        ax.set_facecolor("white")
        for spine in ax.spines.values():
            spine.set_color("#d7d0c3")
        ax.grid(axis="y", color="#ece7dc", linewidth=0.8)
        ax.set_axisbelow(True)

        if metric == "fid":
            ax.text(
                0.5,
                0.55,
                "FID Placeholder",
                ha="center",
                va="center",
                fontsize=16,
                fontweight="bold",
                transform=ax.transAxes,
            )
            ax.text(
                0.5,
                0.42,
                "Use once dataset-level FID is computed\nfor each of the 15 combinations.",
                ha="center",
                va="center",
                fontsize=11,
                color="#666666",
                transform=ax.transAxes,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        means = [row["metrics"][metric]["mean"] for row in rows]
        stds = [row["metrics"][metric]["std"] for row in rows]
        labels = [row["condition_label"] for row in rows]
        cards = [row["cardinality"] for row in rows]
        x = np.arange(len(rows))

        ax.bar(x, means, color="#111111", width=0.72, zorder=3)
        ax.errorbar(x, means, yerr=stds, fmt="none", ecolor="#111111", elinewidth=1.1, capsize=2.5, zorder=4)
        for xi, yi, card in zip(x, means, cards, strict=True):
            _cardinality_dot(ax, float(xi), float(yi), int(card))

        ymin, ymax = _metric_limits(metric)
        ax.set_ylim(ymin, ymax)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=8)
        ax.set_title(
            f"{METRIC_LABELS[metric]} ({'higher' if METRIC_HIGHER_IS_BETTER[metric] else 'lower'} is better)",
            fontsize=12,
            loc="left",
        )
        ax.tick_params(axis="y", labelsize=9)

    for ax in axes[len(METRICS):]:
        ax.axis("off")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def render_option_b(payload: dict, out_path: Path) -> Path:
    rows = _rows(payload)
    fig, ax = plt.subplots(figsize=(14.5, 9.5), constrained_layout=True)
    _style_figure(
        fig,
        "Option B: Heatmap Matrix",
        "Dense overview of all 15 combinations. Each cell shows mean ± std; color encodes the mean.",
    )

    metric_order = list(METRICS)
    n_rows = len(rows)
    n_cols = len(metric_order)
    image = np.full((n_rows, n_cols), np.nan, dtype=np.float64)

    for r, row in enumerate(rows):
        for c, metric in enumerate(metric_order):
            stat = row["metrics"][metric]
            if stat is not None:
                image[r, c] = stat["mean"]

    display = np.where(np.isnan(image), 0.0, image)
    im = ax.imshow(display, cmap="cividis", aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_facecolor("white")
    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels([METRIC_LABELS[m] for m in metric_order], fontsize=10)
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels([row["condition_label"] for row in rows], fontsize=9)

    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="#ffffff", linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)

    for r, row in enumerate(rows):
        for c, metric in enumerate(metric_order):
            stat = row["metrics"][metric]
            if stat is None:
                ax.text(c, r, "Pending", ha="center", va="center", fontsize=8, color="#111111", fontweight="bold")
            else:
                mean = stat["mean"]
                std = stat["std"]
                text_color = "white" if mean < 0.42 else "black"
                ax.text(
                    c,
                    r,
                    f"{mean:.3f}\n± {std:.3f}",
                    ha="center",
                    va="center",
                    fontsize=7.6,
                    color=text_color,
                    fontweight="bold",
                )

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.ax.set_ylabel("Mean metric value", rotation=90)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def render_option_c(payload: dict, out_path: Path) -> Path:
    rows = _rows(payload)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
    axes = axes.flatten()
    _style_figure(
        fig,
        "Option C: Ranked Explorer",
        "Static ranking mockup. Each panel sorts combinations by one metric and shows std as an error bar.",
    )

    for ax, metric in zip(axes, METRICS_NO_FID, strict=True):
        ordered = sorted(
            rows,
            key=lambda row: row["metrics"][metric]["mean"],
            reverse=METRIC_HIGHER_IS_BETTER[metric],
        )
        labels = [row["condition_label"] for row in ordered]
        means = [row["metrics"][metric]["mean"] for row in ordered]
        stds = [row["metrics"][metric]["std"] for row in ordered]
        cards = [row["cardinality"] for row in ordered]

        y = np.arange(len(ordered))
        ax.barh(y, means, color="#111111", zorder=3)
        ax.errorbar(means, y, xerr=stds, fmt="none", ecolor="#111111", elinewidth=1.1, capsize=2.5, zorder=4)
        for yi, xi, card in zip(y, means, cards, strict=True):
            _cardinality_dot(ax, float(xi), float(yi), int(card))

        xmin, xmax = _metric_limits(metric)
        ax.set_xlim(xmin, xmax)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8.5)
        ax.invert_yaxis()
        ax.grid(axis="x", color="#ece7dc", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_title(
            f"{METRIC_LABELS[metric]} ranking ({'higher' if METRIC_HIGHER_IS_BETTER[metric] else 'lower'} is better)",
            fontsize=12,
            loc="left",
        )
        for spine in ax.spines.values():
            spine.set_color("#d7d0c3")
        ax.set_facecolor("white")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def render_all(cache_root: Path, output_dir: Path) -> list[Path]:
    payload = _build_payload(cache_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        render_option_a(payload, output_dir / "ablation_dataset_metrics_option_a_small_multiples.png"),
        render_option_b(payload, output_dir / "ablation_dataset_metrics_option_b_heatmap.png"),
        render_option_c(payload, output_dir / "ablation_dataset_metrics_option_c_ranking.png"),
    ]
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render dataset-level ablation metric proposal mockups as PNGs.",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=ROOT / "inference_output/cache",
        help="Parent directory containing per-tile cache folders with metrics.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "inference_output/dataset_metrics_pngs",
        help="Output directory for rendered PNG proposal boards",
    )
    args = parser.parse_args()

    outputs = render_all(args.cache_root.resolve(), args.output_dir.resolve())
    for path in outputs:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
