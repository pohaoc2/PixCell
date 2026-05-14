from __future__ import annotations

import os
import statistics
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mpl-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from tools.stage3.ablation_vis_utils import FOUR_GROUP_ORDER, GROUP_SHORT_LABELS, normalize_group_name

OKABE_BLUE = "#4C78A8"
OKABE_ORANGE = "#E28E2B"
OKABE_GREEN = "#5C8F5B"
OKABE_PURPLE = "#8D6A9F"
OKABE_RED = "#B22222"
OKABE_TEAL = "#5B8F96"
OKABE_GRAY = "#565656"
INK = "#000000"
SOFT_GRID = "#D7D6D2"
METRIC_COLORS = {
    "lpips": OKABE_ORANGE,
    "pq": OKABE_PURPLE,
    "dice": OKABE_GREEN,
    "fud": OKABE_RED,
    "style_hed": OKABE_TEAL,
}
GROUP_COLORS = {
    "cell_types": "#7C8AA5",
    "cell_state": "#B9795F",
    "vasculature": "#6E9C92",
    "microenv": "#9175A6",
}
GROUP_LABELS = {
    "cell_types": "Cell types",
    "cell_state": "Cell state",
    "vasculature": "Vasculature",
    "microenv": "Nutrient",
}
METRIC_LABELS = {
    "lpips": "LPIPS",
    "pq": "PQ",
    "dice": "DICE",
    "fud": "FUD",
    "style_hed": "HED",
}
TRADEOFF_METRIC_ORDER = ("fud", "lpips", "pq", "dice", "style_hed")
TRADEOFF_HIGHER_IS_BETTER = {"dice": True}
TRADEOFF_REFERENCE_BANDS = {
    "fud": {"label": "PixCell Cond FUD", "mean": 95.57},
    "pq": {"label": "CellViT proposed PQ", "mean": 0.6696, "std": 0.034},
    "dice": {"label": "PixCell ControlNet DICE", "mean": 0.653},
}
CHANNEL_EFFECT_CMAP = plt.get_cmap("RdBu")


@dataclass(frozen=True)
class MetricSpec:
    key: str
    label: str
    higher_is_better: bool


METRIC_SPECS: tuple[MetricSpec, ...] = (
    MetricSpec("lpips", "lpips", False),
    MetricSpec("aji", "aji", True),
    MetricSpec("pq", "pq", True),
    MetricSpec("dice", "dice", True),
    MetricSpec("fud", "fud", False),
    MetricSpec("style_hed", "style_hed", False),
)
METRIC_SPEC_BY_KEY = {spec.key: spec for spec in METRIC_SPECS}
DEFAULT_METRIC_ORDER = tuple(spec.key for spec in METRIC_SPECS)


@dataclass(frozen=True)
class DatasetSummary:
    slug: str
    title: str
    metrics_root: Path
    dataset_root: Path
    tile_count: int
    metric_keys: list[str]
    condition_stats: dict[str, dict[str, tuple[float, float]]]
    by_cardinality: dict[int, dict[str, float]]
    by_cardinality_stats: dict[int, dict[str, tuple[float, float]]]
    best_worst: dict[str, dict[str, list[tuple[str, float, float]] | int]]
    added_effects: dict[str, dict[str, float]]
    presence_absence: dict[str, dict[str, float]]
    added_effect_stats: dict[str, dict[str, tuple[float, float]]]
    presence_absence_stats: dict[str, dict[str, tuple[float, float]]]
    loo_summary: dict[str, dict[str, float]]
    loo_stats: dict[str, dict[str, tuple[float, float]]]
    representative_tile: str | None
    key_takeaways: list[str]
    ablation_grid_path: Path | None
    loo_diff_path: Path | None


def _mean_std(values: list[float]) -> tuple[float, float] | None:
    if not values:
        return None
    return float(statistics.mean(values)), float(statistics.pstdev(values)) if len(values) > 1 else 0.0


def metric_union(summaries: list[DatasetSummary]) -> list[str]:
    present = {metric for summary in summaries for metric in summary.metric_keys}
    return [metric for metric in DEFAULT_METRIC_ORDER if metric in present]


def condition_groups(value: object) -> set[str]:
    groups: set[str] = set()
    for token in str(value).split("+"):
        normalized = normalize_group_name(token.strip())
        if normalized:
            groups.add(normalized)
    return groups


def condition_order_label() -> str:
    return " / ".join(GROUP_SHORT_LABELS[group] for group in FOUR_GROUP_ORDER)


def compact_condition_order_label() -> str:
    return "CT/CS/VAS/NUT"


def condition_indicator_text(condition: object) -> str:
    active_groups = condition_groups(condition)
    return " ".join("●" if group in active_groups else "○" for group in FOUR_GROUP_ORDER)


def comparison_metric_keys(summary: DatasetSummary) -> list[str]:
    present = set(summary.metric_keys)
    return [metric for metric in TRADEOFF_METRIC_ORDER if metric in present]


def metric_tradeoff_keys(summaries: list[DatasetSummary]) -> list[str]:
    present = {
        metric
        for summary in summaries
        for condition_stats in summary.condition_stats.values()
        for metric in condition_stats
    }
    return [metric for metric in TRADEOFF_METRIC_ORDER if metric in present]


def _reference_band_limits(metric_key: str) -> tuple[float, float] | None:
    reference = TRADEOFF_REFERENCE_BANDS.get(metric_key)
    if reference is None:
        return None
    mean = float(reference["mean"])
    std = reference.get("std")
    if std is not None:
        return mean - float(std), mean + float(std)
    return None


def humanize_token(value: object) -> str:
    raw = str(value).strip()
    if not raw:
        return raw
    if raw in METRIC_LABELS:
        return METRIC_LABELS[raw]
    if raw in GROUP_LABELS:
        return GROUP_LABELS[raw]
    if "+" in raw:
        return " + ".join(humanize_token(part) for part in raw.split("+"))
    return raw.replace("_", " ")


def format_condition(value: object) -> str:
    return humanize_token(value)


def metric_name_cell(metric_key: str) -> str:
    return humanize_token(metric_key)


def metric_plain_text(metric_key: str) -> str:
    return humanize_token(metric_key)


def _format_mean_sd(mean: float, sd: float) -> str:
    return f"{float(mean):.3f} ± {float(sd):.3f}"


def _metric_caption(metric_key: str) -> str:
    spec = METRIC_SPEC_BY_KEY[metric_key]
    arrow = "↑" if spec.higher_is_better else "↓"
    return f"{humanize_token(metric_key)} ({arrow})"


def _ranked_labels(total: int, count: int, *, tail: bool = False) -> list[str]:
    if count <= 0:
        return []
    if tail:
        start = max(1, total - count + 1)
        return [str(rank) for rank in range(start, start + count)]
    return [str(rank) for rank in range(1, count + 1)]


def _coerce_best_worst_entry(entry: object) -> tuple[str, float, float] | None:
    if isinstance(entry, dict):
        condition = entry.get("condition_key") or entry.get("condition") or entry.get("best_condition")
        mean = entry.get("mean")
        sd = entry.get("sd", entry.get("std"))
        if condition is None or mean is None:
            return None
        try:
            return str(condition), float(mean), float(0.0 if sd is None else sd)
        except (TypeError, ValueError):
            return None
    if isinstance(entry, (list, tuple)) and len(entry) >= 3:
        condition, mean, sd = entry[:3]
        try:
            return str(condition), float(mean), float(sd)
        except (TypeError, ValueError):
            return None
    return None


def _normalize_best_worst_record(
    record: object,
) -> tuple[list[tuple[str, float, float]], list[tuple[str, float, float]], int | None]:
    if not isinstance(record, dict):
        return [], [], None
    best = record.get("best")
    worst = record.get("worst")
    if isinstance(best, list) and isinstance(worst, list):
        best_entries = [entry for entry in (_coerce_best_worst_entry(item) for item in best) if entry is not None]
        worst_entries = [entry for entry in (_coerce_best_worst_entry(item) for item in worst) if entry is not None]
        total = record.get("total")
        try:
            total_count = int(total) if total is not None else None
        except (TypeError, ValueError):
            total_count = None
        return best_entries, worst_entries, total_count
    best_condition = record.get("best_condition")
    best_value = record.get("best_value")
    worst_condition = record.get("worst_condition")
    worst_value = record.get("worst_value")
    if None in {best_condition, best_value, worst_condition, worst_value}:
        return [], [], None
    try:
        return (
            [(str(best_condition), float(best_value), 0.0)],
            [(str(worst_condition), float(worst_value), 0.0)],
            None,
        )
    except (TypeError, ValueError):
        return [], [], None


def _best_worst_total(
    record_total: int | None,
    best_entries: list[tuple[str, float, float]],
    worst_entries: list[tuple[str, float, float]],
) -> int:
    if record_total is not None and record_total > 0:
        return record_total
    return max(len(best_entries), len(worst_entries))


def _ranked_best_worst_selection(
    record: object,
) -> tuple[list[tuple[str, float, float]], list[tuple[str, float, float]], int]:
    best_entries, worst_entries, record_total = _normalize_best_worst_record(record)
    total = _best_worst_total(record_total, best_entries, worst_entries)
    if total <= 0:
        return [], [], 0

    top_count = min(3, total, len(best_entries))
    top_entries = best_entries[:top_count]

    bottom_start = max(top_count + 1, total - 2)
    bottom_count = min(3, len(worst_entries), max(0, total - bottom_start + 1))
    bottom_entries = worst_entries[-bottom_count:] if bottom_count > 0 else []
    return top_entries, bottom_entries, total


def plain_takeaway_text(text: str) -> str:
    parts = text.split("`")
    plain: list[str] = []
    for index, part in enumerate(parts):
        plain.append(humanize_token(part) if index % 2 == 1 else part)
    return "".join(plain)


def _tight_range(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 1.0
    lo = min(values)
    hi = max(values)
    if hi <= lo:
        pad = 0.1 if lo == 0 else abs(lo) * 0.08
        return lo - pad, hi + pad
    pad = 0.12 * (hi - lo)
    return lo - pad, hi + pad
