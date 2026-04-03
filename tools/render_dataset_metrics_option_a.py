#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mpl-cache"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

INK = "#000000"
MUTED = "#000000"
SOFT = "#aaaaaa"
GRID = "#ece8e0"
AXIS = "#000000"
DOT_ACTIVE = "#4b5563"
DOT_INACTIVE = "#cfc7ba"
CARD_COLORS = ["#009E73", "#0072B2", "#D55E00", "#9B59B6"]

GROUPS = ["CT", "CS", "Vas", "Env"]


@dataclass(frozen=True)
class Metric:
    key: str
    label: str
    hib: bool
    value_range: tuple[float, float]


@dataclass
class Combination:
    mask: int
    bits: list[bool]
    n: int
    metrics: dict[str, dict[str, float]]


METRICS = [
    Metric("fid", "FID", False, (55, 200)),
    Metric("cosine", "Cosine", True, (0.34, 0.63)),
    Metric("lpips", "LPIPS", False, (0.38, 0.52)),
    Metric("aji", "AJI", True, (0.0, 0.38)),
    Metric("pq", "PQ", True, (0.0, 0.32)),
]

BASE = {"fid": 158, "cosine": 0.410, "lpips": 0.478, "aji": 0.018, "pq": 0.014}
BONUS = {
    "fid": [-24, -10, -17, -7],
    "cosine": [0.052, 0.020, 0.030, 0.014],
    "lpips": [-0.011, -0.005, -0.009, -0.003],
    "aji": [0.108, 0.065, 0.050, 0.024],
    "pq": [0.095, 0.055, 0.048, 0.020],
}
JITTER_SCALE = {"fid": 0.28, "cosine": 0.22, "lpips": 0.18, "aji": 0.25, "pq": 0.20}
STD_BASE = {"fid": 9.5, "cosine": 0.019, "lpips": 0.007, "aji": 0.016, "pq": 0.013}


def lcg(seed: int) -> float:
    return ((seed * 1664525 + 1013904223) & 0xFFFFFFFF) / 4294967296


def build_combinations() -> list[Combination]:
    combinations: list[Combination] = []
    for mask in range(1, 16):
        bits = [bool(mask & (1 << index)) for index in range(4)]
        combinations.append(Combination(mask=mask, bits=bits, n=sum(bits), metrics={}))

    combinations.sort(key=lambda combo: (combo.n, combo.mask))

    for combo in combinations:
        for metric in METRICS:
            mean = BASE[metric.key]
            for group_index, active in enumerate(combo.bits):
                if active:
                    mean += BONUS[metric.key][group_index]

            seed = combo.mask * 31 + ord(metric.key[0])
            jitter = (lcg(seed) - 0.5) * 2 * STD_BASE[metric.key] * JITTER_SCALE[metric.key]
            std = STD_BASE[metric.key] * (0.5 + 0.5 * combo.n / 4)
            combo.metrics[metric.key] = {"mean": mean + jitter, "std": std}

    return combinations


def cardinality_spans(combinations: list[Combination]) -> list[tuple[int, int, int]]:
    spans: list[tuple[int, int, int]] = []
    start = 0
    for index in range(1, len(combinations)):
        if combinations[index].n != combinations[index - 1].n:
            spans.append((combinations[index - 1].n, start, index - 1))
            start = index
    spans.append((combinations[-1].n, start, len(combinations) - 1))
    return spans


def fmt_value(metric: Metric, value: float) -> str:
    if metric.key == "fid":
        return str(round(value))
    return f"{value:.3f}"

def render(output_path: Path, dpi: int) -> None:
    combos = build_combinations()
    spans = cardinality_spans(combos)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "axes.titleweight": "bold",
            "axes.labelcolor": INK,
            "xtick.color": SOFT,
            "ytick.color": INK,
        }
    )

    fig = plt.figure(figsize=(17.8, 4.6), facecolor="none")

    grid = fig.add_gridspec(
        2,
        len(METRICS),
        left=0.05,
        right=0.995,
        top=0.96,
        bottom=0.12,
        height_ratios=[7.0, 2.0],
        hspace=0.0,
        wspace=0.40,
    )

    x_positions = list(range(len(combos)))

    for column, metric in enumerate(METRICS):
        ax = fig.add_subplot(grid[0, column])
        dot_ax = fig.add_subplot(grid[1, column])

        lo, hi = metric.value_range
        means = [combo.metrics[metric.key]["mean"] for combo in combos]
        stds = [combo.metrics[metric.key]["std"] for combo in combos]
        fills = [CARD_COLORS[combo.n - 1] for combo in combos]

        ax.set_facecolor("none")
        dot_ax.set_facecolor("none")

        for tick_index in range(6):
            y_value = lo + (hi - lo) * tick_index / 5
            ax.axhline(y_value, color=GRID, linewidth=0.85, zorder=0)

        for _, start, end in spans[:-1]:
            ax.axvline(end + 0.5, color="#bcbcbc", linewidth=1.0, linestyle=(0, (3, 2.5)), zorder=1)
            dot_ax.axvline(end + 0.5, color="#d7d0c4", linewidth=0.9, linestyle=(0, (3, 2.5)), zorder=0)

        ax.bar(
            x_positions,
            means,
            width=0.64,
            color=fills,
            edgecolor=INK,
            linewidth=0.8,
            alpha=0.9,
            zorder=3,
        )
        ax.errorbar(
            x_positions,
            means,
            yerr=stds,
            fmt="none",
            ecolor="#222222",
            elinewidth=1.15,
            capsize=2.8,
            zorder=4,
        )

        for cardinality, start, end in spans:
            center = (start + end) / 2
            ax.text(
                center,
                1.008,
                f"{cardinality}g",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=7.5,
                color=SOFT,
                family="DejaVu Sans Mono",
            )

        ax.set_xlim(-0.55, len(combos) - 0.45)
        ax.set_ylim(lo, hi)
        ax.set_xticks([])
        ax.set_ylabel(
            f"{metric.label} ({'↑' if metric.hib else '↓'})",
            fontsize=9.0,
            color=INK,
            labelpad=10,
        )
        ax.set_yticks([lo + (hi - lo) * tick_index / 5 for tick_index in range(6)])
        ax.set_yticklabels([fmt_value(metric, lo + (hi - lo) * tick_index / 5) for tick_index in range(6)], fontsize=7.5)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(AXIS)
        ax.spines["bottom"].set_color(AXIS)
        ax.spines["left"].set_linewidth(1.0)
        ax.spines["bottom"].set_linewidth(1.0)
        ax.tick_params(axis="y", labelsize=7.5, colors=INK, width=0.8)

        for x_value, combo in enumerate(combos):
            for row_index, active in enumerate(combo.bits):
                y_value = 3 - row_index
                if active:
                    dot_ax.scatter(
                        x_value,
                        y_value,
                        s=36,
                        facecolors=DOT_ACTIVE,
                        edgecolors="none",
                        zorder=3,
                    )
                else:
                    dot_ax.scatter(
                        x_value,
                        y_value,
                        s=28,
                        facecolors="white",
                        edgecolors=DOT_INACTIVE,
                        linewidths=1.35,
                        zorder=2,
                    )

        dot_ax.set_xlim(-0.55, len(combos) - 0.45)
        dot_ax.set_ylim(-0.6, 3.6)
        dot_ax.axis("off")

        if column == 0:
            ymin, ymax = dot_ax.get_ylim()
            for row_index, label in enumerate(GROUPS):
                y_value = 3 - row_index
                y_axes = (y_value - ymin) / (ymax - ymin)
                dot_ax.text(
                    -0.055,
                    y_axes,
                    label,
                    transform=dot_ax.transAxes,
                    ha="right",
                    va="center",
                    fontsize=7.6,
                    color=INK,
                    family="sans-serif",
                )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, transparent=True)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render dataset_metrics_option_a.png.")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "dataset_metrics_option_a.png",
        help="PNG output path.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="PNG DPI.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    render(args.output.resolve(), dpi=args.dpi)


if __name__ == "__main__":
    main()
