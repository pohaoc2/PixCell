#!/usr/bin/env python3
"""Render a static dataset-level metrics PNG inspired by Option A."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.stage3.ablation_vis_utils import FOUR_GROUP_ORDER

matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]

GROUP_SHORT = {
    "cell_types": "CT",
    "cell_state": "CS",
    "vasculature": "Vas",
    "microenv": "Env",
}
GROUP_COLORS = {
    "cell_types": "#E69F00",
    "cell_state": "#CC79A7",
    "vasculature": "#56B4E9",
    "microenv": "#c8b400",
}
CARD_COLORS = {
    1: "#009E73",
    2: "#0072B2",
    3: "#D55E00",
    4: "#9B59B6",
}
METRIC_SPECS = [
    {"key": "fid", "label": "FID", "higher_is_better": False, "placeholder": True},
    {"key": "cosine", "label": "Cosine", "higher_is_better": True, "placeholder": False},
    {"key": "lpips", "label": "LPIPS", "higher_is_better": False, "placeholder": False},
    {"key": "aji", "label": "AJI", "higher_is_better": True, "placeholder": False},
    {"key": "pq", "label": "PQ", "higher_is_better": True, "placeholder": False},
]


def _condition_label(cond_key: str) -> str:
    groups = set(cond_key.split("+")) if cond_key else set()
    return "+".join(GROUP_SHORT[g] for g in FOUR_GROUP_ORDER if g in groups)


def _cardinality(cond_key: str) -> int:
    return len(cond_key.split("+")) if cond_key else 0


def _condition_bits(cond_key: str) -> list[bool]:
    groups = set(cond_key.split("+")) if cond_key else set()
    return [g in groups for g in FOUR_GROUP_ORDER]


def _aggregate(cache_root: Path) -> tuple[list[dict], int]:
    grouped: dict[str, dict[str, list[float]]] = {}
    tile_count = 0
    for metrics_path in sorted(cache_root.glob("*/metrics.json")):
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        per_condition = payload.get("per_condition", {})
        if not isinstance(per_condition, dict):
            continue
        tile_count += 1
        for cond_key, record in per_condition.items():
            if not isinstance(record, dict):
                continue
            bucket = grouped.setdefault(cond_key, {})
            for metric in ("cosine", "lpips", "aji", "pq"):
                value = record.get(metric)
                if value is None:
                    continue
                bucket.setdefault(metric, []).append(float(value))

    rows: list[dict] = []
    for cond_key in sorted(grouped.keys(), key=lambda k: (_cardinality(k), k)):
        metrics: dict[str, dict | None] = {}
        for spec in METRIC_SPECS:
            metric = spec["key"]
            vals = grouped[cond_key].get(metric, [])
            if vals:
                metrics[metric] = {
                    "mean": statistics.mean(vals),
                    "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
                    "n": len(vals),
                }
            else:
                metrics[metric] = None
        rows.append(
            {
                "condition_key": cond_key,
                "condition_label": _condition_label(cond_key),
                "cardinality": _cardinality(cond_key),
                "bits": _condition_bits(cond_key),
                "metrics": metrics,
            }
        )
    return rows, tile_count


def _draw_dot_strip(ax, rows: list[dict], *, show_group_labels: bool) -> None:
    ax.set_xlim(-0.5, len(rows) - 0.5)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    y_dot = 0.34
    y_label = 0.82
    offsets = np.array([-0.24, -0.08, 0.08, 0.24])

    for idx, row in enumerate(rows):
        for j, active in enumerate(row["bits"]):
            x = idx + offsets[j]
            face = GROUP_COLORS[FOUR_GROUP_ORDER[j]] if active else "white"
            edge = GROUP_COLORS[FOUR_GROUP_ORDER[j]] if active else "#bdbdbd"
            ax.scatter([x], [y_dot], s=42, c=[face], edgecolors=[edge], linewidths=1.2, zorder=3)
            if show_group_labels:
                ax.text(
                    x,
                    y_label,
                    GROUP_SHORT[FOUR_GROUP_ORDER[j]],
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    color="#4d4d4d",
                )


def _draw_group_separators(ax, rows: list[dict], y0: float, y1: float) -> None:
    cards = [row["cardinality"] for row in rows]
    for idx in range(1, len(rows)):
        if cards[idx] != cards[idx - 1]:
            ax.vlines(idx - 0.5, y0, y1, color="#d9d2c4", linewidth=1.2, zorder=1)


def _render(cache_root: Path, out_path: Path) -> Path:
    rows, tile_count = _aggregate(cache_root)
    n = len(rows)

    fig = plt.figure(figsize=(18, 13), facecolor="#f7f4ed")
    gs = gridspec.GridSpec(
        8,
        1,
        height_ratios=[0.9, 0.95, 1.45, 1.45, 1.45, 1.45, 1.45, 1.15],
        hspace=0.22,
    )

    ax_header = fig.add_subplot(gs[0])
    ax_header.axis("off")
    ax_header.text(
        0.0, 0.88,
        "Dataset-Level Ablation Metrics",
        fontsize=23,
        fontweight="bold",
        ha="left",
        va="top",
        transform=ax_header.transAxes,
        color="#1a1a1a",
    )
    ax_header.text(
        0.0, 0.48,
        "Static adaptation of Option A: all metrics merged into one figure with one shared combination order and one shared dot-strip key.",
        fontsize=11,
        ha="left",
        va="top",
        transform=ax_header.transAxes,
        color="#666666",
    )
    ax_header.text(
        0.0, 0.10,
        f"n = {tile_count} tiles · 15 combinations · mean ± σ across tiles · FID reserved as dataset-level placeholder",
        fontsize=10,
        ha="left",
        va="bottom",
        transform=ax_header.transAxes,
        color="#555555",
    )

    ax_legend = fig.add_subplot(gs[1])
    ax_legend.axis("off")
    ax_legend.set_xlim(0, 1)
    ax_legend.set_ylim(0, 1)
    ax_legend.text(
        0.0, 0.82, "Channel identities", transform=ax_legend.transAxes,
        ha="left", va="center", fontsize=9.5, fontweight="bold", color="#444444",
    )
    ax_legend.text(
        0.58, 0.82, "Bar fill = combination size", transform=ax_legend.transAxes,
        ha="left", va="center", fontsize=9.5, fontweight="bold", color="#444444",
    )
    for xpos, group in zip([0.01, 0.17, 0.33, 0.49], FOUR_GROUP_ORDER, strict=True):
        ax_legend.scatter(
            [xpos], [0.44], s=58, transform=ax_legend.transAxes,
            c=[GROUP_COLORS[group]], edgecolors=[GROUP_COLORS[group]], linewidths=1.0, clip_on=False,
        )
        ax_legend.text(
            xpos + 0.02, 0.44, GROUP_SHORT[group],
            transform=ax_legend.transAxes, ha="left", va="center",
            fontsize=9.5, color="#333333",
        )
    ax_legend.scatter(
        [0.49 + 0.18], [0.44], s=58, transform=ax_legend.transAxes,
        c=["white"], edgecolors=["#bdbdbd"], linewidths=1.0, clip_on=False,
    )
    ax_legend.text(
        0.49 + 0.20, 0.44, "inactive",
        transform=ax_legend.transAxes, ha="left", va="center",
        fontsize=9.5, color="#666666",
    )
    for xpos, (card, color) in zip([0.58, 0.71, 0.84, 0.94], CARD_COLORS.items(), strict=True):
        width = 0.022 if card < 4 else 0.018
        ax_legend.add_patch(
            patches.Rectangle(
                (xpos, 0.34), width, 0.20,
                transform=ax_legend.transAxes,
                facecolor=color,
                edgecolor="none",
                clip_on=False,
            )
        )
        ax_legend.text(
            xpos + width + 0.008, 0.44, f"{card}",
            transform=ax_legend.transAxes, ha="left", va="center",
            fontsize=9.5, color="#333333",
        )

    bar_axes = [fig.add_subplot(gs[i]) for i in range(2, 7)]
    dot_ax = fig.add_subplot(gs[7])

    x = np.arange(n)
    for ax, spec in zip(bar_axes, METRIC_SPECS, strict=True):
        ax.set_facecolor("white")
        for spine in ax.spines.values():
            spine.set_color("#e0dbd0")
        ax.grid(axis="y", color="#ece7dd", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_xlim(-0.6, n - 0.4)
        _draw_group_separators(ax, rows, 0, 1)
        ax.tick_params(axis="x", length=0, labelbottom=False)

        if spec["placeholder"]:
            ax.set_ylim(0, 1)
            ax.set_yticks([])
            ax.text(
                0.5, 0.55, "FID Placeholder",
                transform=ax.transAxes,
                ha="center", va="center",
                fontsize=16, fontweight="bold", color="#444444",
            )
            ax.text(
                0.5, 0.34, "Reserve this row for dataset-level FID\nonce computed per combination.",
                transform=ax.transAxes,
                ha="center", va="center",
                fontsize=10, color="#777777",
            )
            ax.text(0.01, 0.92, "FID", transform=ax.transAxes, ha="left", va="top", fontsize=12, fontweight="bold")
            continue

        means = np.array([row["metrics"][spec["key"]]["mean"] for row in rows], dtype=float)
        stds = np.array([row["metrics"][spec["key"]]["std"] for row in rows], dtype=float)
        colors = [CARD_COLORS[row["cardinality"]] for row in rows]

        if spec["key"] == "cosine":
            y_min, y_max = -0.05, 1.0
        else:
            y_min, y_max = 0.0, 1.0
        ax.set_ylim(y_min, y_max)
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])

        ax.bar(x, means, color=colors, width=0.74, edgecolor="none", zorder=3)
        ax.errorbar(x, means, yerr=stds, fmt="none", ecolor="#202020", elinewidth=1.0, capsize=2.0, zorder=4)
        ax.text(
            0.01, 0.92,
            f"{spec['label']} {'↑' if spec['higher_is_better'] else '↓'}",
            transform=ax.transAxes,
            ha="left", va="top", fontsize=12, fontweight="bold", color="#1a1a1a",
        )
        ax.tick_params(axis="y", labelsize=9, colors="#555555")

    _draw_dot_strip(dot_ax, rows, show_group_labels=False)
    dot_ax.set_xlim(-0.6, n - 0.4)
    dot_ax.set_facecolor("#f7f4ed")
    _draw_group_separators(dot_ax, rows, 0.04, 0.92)
    dot_ax.text(
        0.0, 0.98,
        "Combination key: filled dot = active channel",
        transform=dot_ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.5,
        color="#555555",
    )
    boundaries = [0, 4, 10, 14, 15]
    labels = ["1 group", "2 groups", "3 groups", "4 groups"]
    for start, end, label in zip(boundaries[:-1], boundaries[1:], labels, strict=True):
        center = (start + end - 1) / 2
        dot_ax.text(
            center,
            0.92,
            label,
            ha="center",
            va="top",
            fontsize=9,
            color="#666666",
        )
    dot_ax.text(
        0.0, 0.06,
        "Dot order: CT · CS · Vas · Env",
        transform=dot_ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        color="#666666",
    )

    fig.subplots_adjust(left=0.055, right=0.99, top=0.985, bottom=0.085)
    fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a static Option-A-inspired dataset metrics PNG.",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=ROOT / "inference_output/cache",
        help="Parent directory containing per-tile caches with metrics.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "inference_output/dataset_metrics_option_a_static.png",
        help="Output PNG path",
    )
    args = parser.parse_args()

    out_path = _render(args.cache_root.resolve(), args.output.resolve())
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
