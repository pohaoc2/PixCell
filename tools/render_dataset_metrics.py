#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import statistics
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mpl-cache"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.stage3.ablation_vis_utils import FOUR_GROUP_ORDER, condition_metric_key

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
    metrics: dict[str, dict[str, float] | None]


METRICS = [
    Metric("fud", "FUD", False, (55, 200)),
    Metric("cosine", "Cosine", True, (0.34, 0.63)),
    Metric("lpips", "LPIPS", False, (0.38, 0.52)),
    Metric("aji", "AJI", True, (0.0, 0.38)),
    Metric("pq", "PQ", True, (0.0, 0.32)),
    Metric("style_hed", "HED", False, (0.04, 0.10)),
]
METRIC_BY_KEY = {metric.key: metric for metric in METRICS}
METRIC_SETS: dict[str, tuple[str, ...]] = {
    "paired": ("fud", "cosine", "lpips", "aji", "pq"),
    "unpaired": ("fud", "aji", "pq", "style_hed"),
    "all": tuple(metric.key for metric in METRICS),
}


def _metric_value_from_record(record: dict[str, object], metric_key: str) -> object:
    if metric_key == "fud":
        value = record.get("fud")
        if value is None:
            return record.get("fid")
        return value
    return record.get(metric_key)

def _ordered_condition_tuples() -> list[tuple[str, ...]]:
    return [
        tuple(cond)
        for size in range(1, len(FOUR_GROUP_ORDER) + 1)
        for cond in combinations(FOUR_GROUP_ORDER, size)
    ]


def _condition_mask(cond_tuple: tuple[str, ...]) -> int:
    groups = set(cond_tuple)
    mask = 0
    for idx, group in enumerate(FOUR_GROUP_ORDER):
        if group in groups:
            mask |= 1 << idx
    return mask


def load_combinations(
    metric_dir: Path,
    min_gt_cells: int = 0,
    orion_root: Path | None = None,
) -> tuple[list[Combination], int]:
    grouped: dict[str, dict[str, list[float]]] = {}
    metrics_paths = sorted(Path(metric_dir).glob("*/metrics.json"))
    if not metrics_paths:
        raise FileNotFoundError(
            f"no per-tile metrics.json files found under {Path(metric_dir).resolve()}"
        )
    tile_count = 0
    filtered = 0

    for metrics_path in metrics_paths:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        tile_id = str(payload.get("tile_id", "") or metrics_path.parent.name).strip()
        if min_gt_cells > 0 and orion_root is not None and tile_id:
            from tools.compute_ablation_metrics import _load_gt_instance_mask, _instance_ids
            try:
                gt = _load_gt_instance_mask(orion_root, tile_id)
                if _instance_ids(gt).size < min_gt_cells:
                    filtered += 1
                    continue
            except FileNotFoundError:
                filtered += 1
                continue
        per_condition = payload.get("per_condition", {})
        if not isinstance(per_condition, dict):
            continue
        tile_count += 1
        for cond_key, record in per_condition.items():
            if not isinstance(record, dict):
                continue
            bucket = grouped.setdefault(str(cond_key), {})
            for metric in METRICS:
                value = _metric_value_from_record(record, metric.key)
                if value is None:
                    continue
                bucket.setdefault(metric.key, []).append(float(value))

    combinations: list[Combination] = []
    for cond_tuple in _ordered_condition_tuples():
        cond_key = condition_metric_key(cond_tuple)
        metric_stats: dict[str, dict[str, float] | None] = {}
        for metric in METRICS:
            values = grouped.get(cond_key, {}).get(metric.key, [])
            metric_stats[metric.key] = (
                {
                    "mean": float(statistics.mean(values)),
                    "std": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
                }
                if values
                else None
            )
        combinations.append(
            Combination(
                mask=_condition_mask(cond_tuple),
                bits=[group in set(cond_tuple) for group in FOUR_GROUP_ORDER],
                n=len(cond_tuple),
                metrics=metric_stats,
            )
        )

    if min_gt_cells > 0:
        print(f"Filtered {filtered} tiles with < {min_gt_cells} GT cells ({tile_count} kept)")
    return combinations, tile_count


def resolve_metric_set(metric_set: str) -> list[Metric]:
    try:
        metric_keys = METRIC_SETS[metric_set]
    except KeyError as exc:
        raise ValueError(f"unsupported metric set: {metric_set}") from exc
    return [METRIC_BY_KEY[key] for key in metric_keys]


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
    if metric.key == "fud":
        return str(round(value))
    return f"{value:.3f}"


def _metric_range(metric: Metric, combos: list[Combination]) -> tuple[float, float]:
    lo, hi = metric.value_range
    observed: list[float] = []
    for combo in combos:
        stats = combo.metrics.get(metric.key)
        if not stats:
            continue
        observed.append(float(stats["mean"] - stats["std"]))
        observed.append(float(stats["mean"] + stats["std"]))
    if not observed:
        return lo, hi

    obs_lo = min(observed)
    obs_hi = max(observed)
    lo = min(lo, obs_lo)
    hi = max(hi, obs_hi)
    if hi <= lo:
        pad = 1.0 if metric.key == "fud" else 0.05
        return lo - pad, hi + pad
    pad = 0.04 * (hi - lo)
    return lo - pad, hi + pad


def render(
    metric_dir: Path,
    output_path: Path,
    dpi: int,
    min_gt_cells: int = 0,
    orion_root: Path | None = None,
    metric_set: str = "paired",
) -> None:
    metrics = resolve_metric_set(metric_set)
    combos, tile_count = load_combinations(metric_dir, min_gt_cells=min_gt_cells, orion_root=orion_root)
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
        len(metrics),
        left=0.05,
        right=0.995,
        top=0.96,
        bottom=0.12,
        height_ratios=[7.0, 2.0],
        hspace=0.0,
        wspace=0.40,
    )

    x_positions = list(range(len(combos)))

    for column, metric in enumerate(metrics):
        ax = fig.add_subplot(grid[0, column])
        dot_ax = fig.add_subplot(grid[1, column])

        lo, hi = _metric_range(metric, combos)
        stats_list = [combo.metrics.get(metric.key) for combo in combos]
        means = [stats["mean"] if stats else None for stats in stats_list]
        stds = [stats["std"] if stats else 0.0 for stats in stats_list]
        fills = [CARD_COLORS[combo.n - 1] for combo in combos]

        ax.set_facecolor("none")
        dot_ax.set_facecolor("none")

        for tick_index in range(6):
            y_value = lo + (hi - lo) * tick_index / 5
            ax.axhline(y_value, color=GRID, linewidth=0.85, zorder=0)

        for _, start, end in spans[:-1]:
            ax.axvline(end + 0.5, color="#bcbcbc", linewidth=1.0, linestyle=(0, (3, 2.5)), zorder=1)
            dot_ax.axvline(end + 0.5, color="#d7d0c4", linewidth=0.9, linestyle=(0, (3, 2.5)), zorder=0)

        if any(mean is not None for mean in means):
            valid_x = [x for x, mean in zip(x_positions, means, strict=True) if mean is not None]
            valid_means = [float(mean) for mean in means if mean is not None]
            valid_stds = [std for mean, std in zip(means, stds, strict=True) if mean is not None]
            valid_fills = [fill for mean, fill in zip(means, fills, strict=True) if mean is not None]
            ax.bar(
                valid_x,
                valid_means,
                width=0.64,
                color=valid_fills,
                edgecolor=INK,
                linewidth=0.8,
                alpha=0.9,
                zorder=3,
            )
            ax.errorbar(
                valid_x,
                valid_means,
                yerr=valid_stds,
                fmt="none",
                ecolor="#222222",
                elinewidth=1.15,
                capsize=2.8,
                zorder=4,
            )
        else:
            ax.text(
                0.5,
                0.56,
                f"{metric.label}\nnot available",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=12,
                color=SOFT,
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
        ax.set_yticklabels(
            [fmt_value(metric, lo + (hi - lo) * tick_index / 5) for tick_index in range(6)],
            fontsize=7.5,
        )

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
    print(f"Rendered dataset metrics from {tile_count} tiles -> {output_path}")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render dataset_metrics.png from cached metrics.")
    parser.add_argument(
        "--metric-dir",
        type=Path,
        default=ROOT / "inference_output" / "full_ablation",
        help="Directory containing per-tile metrics.json files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "dataset_metrics.png",
        help="PNG output path.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="PNG DPI.")
    parser.add_argument(
        "--metric-set",
        choices=sorted(METRIC_SETS),
        default="paired",
        help="Metrics to show: paired=FUD/Cosine/LPIPS/AJI/PQ, unpaired=FUD/AJI/PQ/HED, all=all metrics.",
    )
    parser.add_argument("--min-gt-cells", type=int, default=0,
        help="Skip tiles with fewer than this many GT cell instances (default: 0 = no filter).")
    parser.add_argument("--orion-root", type=Path,
        default=ROOT / "data/orion-crc33",
        help="Paired dataset root for GT cell mask lookup (default: data/orion-crc33).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metric_dir = args.metric_dir.resolve()
    if not metric_dir.exists():
        raise SystemExit(f"metric dir not found: {metric_dir}")
    render(metric_dir, args.output.resolve(), dpi=args.dpi,
           min_gt_cells=args.min_gt_cells, orion_root=args.orion_root.resolve(),
           metric_set=args.metric_set)


if __name__ == "__main__":
    main()
