#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.stage3.ablation_vis_utils import FOUR_GROUP_ORDER, condition_metric_key


@dataclass(frozen=True)
class MetricSpec:
    key: str
    label: str
    higher_is_better: bool


METRIC_SPECS: tuple[MetricSpec, ...] = (
    MetricSpec("cosine", "cosine", True),
    MetricSpec("lpips", "lpips", False),
    MetricSpec("aji", "aji", True),
    MetricSpec("pq", "pq", True),
    MetricSpec("fud", "fud", False),
    MetricSpec("style_hed", "style_hed", False),
)
METRIC_SPEC_BY_KEY = {spec.key: spec for spec in METRIC_SPECS}
DEFAULT_METRIC_ORDER = tuple(spec.key for spec in METRIC_SPECS)
LOO_DISTANCE_KEYS: tuple[str, ...] = ("mean_diff", "pct_pixels_above_10")


def _mean(values: list[float]) -> float | None:
    return float(statistics.mean(values)) if values else None


def _fmt(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.3f}"


def _format_condition_key(cond_key: str) -> str:
    return f"`{cond_key}`"


def _ordered_metrics(metric_keys: set[str]) -> list[str]:
    return [key for key in DEFAULT_METRIC_ORDER if key in metric_keys]


def _metric_value_from_record(record: dict[str, object], metric_key: str) -> object:
    if metric_key == "fud":
        value = record.get("fud")
        if value is None:
            return record.get("fid")
        return value
    return record.get(metric_key)


def _condition_sd(
    condition_stats: dict[str, dict[str, tuple[float, float]]] | None,
    cond_key: str,
    metric_key: str,
) -> float:
    if condition_stats is None:
        return 0.0
    return condition_stats.get(cond_key, {}).get(metric_key, (0.0, 0.0))[1]


def load_condition_means(metrics_root: Path) -> tuple[dict[str, dict[str, float]], int, list[str]]:
    grouped: dict[str, dict[str, list[float]]] = {}
    metric_keys: set[str] = set()
    metrics_paths = sorted(Path(metrics_root).glob("*/metrics.json"))
    if not metrics_paths:
        raise FileNotFoundError(f"no per-tile metrics.json files found under {Path(metrics_root).resolve()}")

    tile_count = 0
    for metrics_path in metrics_paths:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        per_condition = payload.get("per_condition")
        if not isinstance(per_condition, dict):
            continue
        tile_count += 1
        for cond_key, record in per_condition.items():
            if not isinstance(record, dict):
                continue
            bucket = grouped.setdefault(str(cond_key), {})
            for metric in METRIC_SPECS:
                value = _metric_value_from_record(record, metric.key)
                if value is None:
                    continue
                bucket.setdefault(metric.key, []).append(float(value))
                metric_keys.add(metric.key)

    condition_means: dict[str, dict[str, float]] = {}
    for cond_key, record in grouped.items():
        condition_means[cond_key] = {
            metric_key: float(statistics.mean(values))
            for metric_key, values in record.items()
            if values
        }
    return condition_means, tile_count, _ordered_metrics(metric_keys)


def summarize_by_cardinality(
    condition_means: dict[str, dict[str, float]],
    metric_keys: list[str],
) -> dict[int, dict[str, float]]:
    grouped: dict[int, dict[str, list[float]]] = {}
    for cond_key, metrics in condition_means.items():
        bucket = grouped.setdefault(len(cond_key.split("+")), {})
        for metric_key in metric_keys:
            value = metrics.get(metric_key)
            if value is None:
                continue
            bucket.setdefault(metric_key, []).append(value)

    return {
        group_count: {
            metric_key: float(statistics.mean(values))
            for metric_key, values in metric_lists.items()
            if values
        }
        for group_count, metric_lists in grouped.items()
    }


def summarize_best_worst(
    condition_means: dict[str, dict[str, float]],
    metric_keys: list[str],
    condition_stats: dict[str, dict[str, tuple[float, float]]] | None = None,
    n: int = 3,
) -> dict[str, dict[str, list[tuple[str, float, float]] | int]]:
    summary: dict[str, dict[str, list[tuple[str, float, float]] | int]] = {}
    for metric_key in metric_keys:
        items = [
            (cond_key, metrics[metric_key])
            for cond_key, metrics in condition_means.items()
            if metric_key in metrics
        ]
        if not items:
            continue
        spec = METRIC_SPEC_BY_KEY[metric_key]
        ranked = sorted(items, key=lambda item: item[1], reverse=spec.higher_is_better)
        best_items = [
            (cond_key, value, _condition_sd(condition_stats, cond_key, metric_key))
            for cond_key, value in ranked[:n]
        ]
        worst_start = max(n, len(ranked) - n)
        worst_items = [
            (cond_key, value, _condition_sd(condition_stats, cond_key, metric_key))
            for cond_key, value in ranked[worst_start:]
        ]
        summary[metric_key] = {"best": best_items, "worst": worst_items, "total": len(items)}
    return summary


def summarize_added_group_effects(
    condition_means: dict[str, dict[str, float]],
    metric_keys: list[str],
) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for group in FOUR_GROUP_ORDER:
        deltas: dict[str, list[float]] = {metric_key: [] for metric_key in metric_keys}
        others = [candidate for candidate in FOUR_GROUP_ORDER if candidate != group]
        for subset_size in range(1, len(others) + 1):
            for subset in combinations(others, subset_size):
                without_key = condition_metric_key(subset)
                with_key = condition_metric_key((*subset, group))
                without_metrics = condition_means.get(without_key)
                with_metrics = condition_means.get(with_key)
                if without_metrics is None or with_metrics is None:
                    continue
                for metric_key in metric_keys:
                    before = without_metrics.get(metric_key)
                    after = with_metrics.get(metric_key)
                    if before is None or after is None:
                        continue
                    spec = METRIC_SPEC_BY_KEY[metric_key]
                    delta = after - before if spec.higher_is_better else before - after
                    deltas[metric_key].append(delta)
        summary[group] = {
            metric_key: float(statistics.mean(values))
            for metric_key, values in deltas.items()
            if values
        }
    return summary


def summarize_presence_absence(
    condition_means: dict[str, dict[str, float]],
    metric_keys: list[str],
) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for group in FOUR_GROUP_ORDER:
        present: dict[str, list[float]] = {metric_key: [] for metric_key in metric_keys}
        absent: dict[str, list[float]] = {metric_key: [] for metric_key in metric_keys}
        for cond_key, metrics in condition_means.items():
            target = present if group in cond_key.split("+") else absent
            for metric_key in metric_keys:
                value = metrics.get(metric_key)
                if value is None:
                    continue
                target[metric_key].append(value)

        summary[group] = {}
        for metric_key in metric_keys:
            present_mean = _mean(present[metric_key])
            absent_mean = _mean(absent[metric_key])
            if present_mean is None or absent_mean is None:
                continue
            spec = METRIC_SPEC_BY_KEY[metric_key]
            summary[group][metric_key] = (
                present_mean - absent_mean if spec.higher_is_better else absent_mean - present_mean
            )
    return summary


def load_leave_one_out_summary(loo_root: Path) -> tuple[dict[str, dict[str, float]], str | None]:
    stats_paths = sorted(Path(loo_root).glob("*/leave_one_out_diff_stats.json"))
    if not stats_paths:
        return {}, None

    grouped: dict[str, dict[str, list[float]]] = {
        group: {"mean_diff": [], "max_diff": [], "pct_pixels_above_10": []}
        for group in FOUR_GROUP_ORDER
    }
    tile_vectors: dict[str, dict[str, float]] = {}

    for stats_path in stats_paths:
        payload = json.loads(stats_path.read_text(encoding="utf-8"))
        vector: dict[str, float] = {}
        for group in FOUR_GROUP_ORDER:
            record = payload.get(group)
            if not isinstance(record, dict):
                continue
            for key in ("mean_diff", "max_diff", "pct_pixels_above_10"):
                value = record.get(key)
                if value is None:
                    continue
                grouped[group][key].append(float(value))
                if key in LOO_DISTANCE_KEYS:
                    vector[f"{group}:{key}"] = float(value)
        if vector:
            tile_vectors[stats_path.parent.name] = vector

    means = {
        group: {
            key: float(statistics.mean(values))
            for key, values in stats.items()
            if values
        }
        for group, stats in grouped.items()
    }
    representative_tile = select_representative_tile(tile_vectors)
    return means, representative_tile


def select_representative_tile(tile_vectors: dict[str, dict[str, float]]) -> str | None:
    if not tile_vectors:
        return None

    axes = sorted({axis for vector in tile_vectors.values() for axis in vector})
    axis_means = {
        axis: statistics.mean([vector[axis] for vector in tile_vectors.values() if axis in vector])
        for axis in axes
    }
    axis_stds = {}
    for axis in axes:
        values = [vector[axis] for vector in tile_vectors.values() if axis in vector]
        axis_stds[axis] = statistics.pstdev(values) if len(values) > 1 else 1.0
        if axis_stds[axis] == 0:
            axis_stds[axis] = 1.0

    best_tile = None
    best_distance = None
    for tile_id, vector in tile_vectors.items():
        distance = 0.0
        for axis in axes:
            if axis not in vector:
                continue
            z = (vector[axis] - axis_means[axis]) / axis_stds[axis]
            distance += z * z
        if best_distance is None or distance < best_distance:
            best_tile = tile_id
            best_distance = distance
    return best_tile


def _trend_phrase(metric_key: str, values: list[float], group_counts: list[int]) -> str | None:
    spec = METRIC_SPEC_BY_KEY[metric_key]
    oriented = values if spec.higher_is_better else [-value for value in values]
    nondecreasing = all(curr >= prev - 1e-9 for prev, curr in zip(oriented, oriented[1:]))
    nonincreasing = all(curr <= prev + 1e-9 for prev, curr in zip(oriented, oriented[1:]))
    if nondecreasing:
        return f"- Adding more groups improves `{metric_key}` almost monotonically."
    if nonincreasing:
        return f"- Adding more groups worsens `{metric_key}` almost monotonically."

    best_index = max(range(len(oriented)), key=lambda index: oriented[index])
    worst_index = min(range(len(oriented)), key=lambda index: oriented[index])
    best_group = group_counts[best_index]
    worst_group = group_counts[worst_index]
    if best_group == worst_group:
        return None
    return f"- `{metric_key}` is best at `{best_group}g` and weakest at `{worst_group}g`."


def cardinality_notes(
    by_cardinality: dict[int, dict[str, float]],
    metric_keys: list[str],
) -> list[str]:
    group_counts = sorted(by_cardinality)
    notes: list[str] = []
    for metric_key in metric_keys:
        values = [by_cardinality[group_count].get(metric_key) for group_count in group_counts]
        if any(value is None for value in values):
            continue
        note = _trend_phrase(metric_key, [float(value) for value in values if value is not None], group_counts)
        if note:
            notes.append(note)
    return notes


def best_worst_notes(best_worst: dict[str, dict[str, list[tuple[str, float, float]] | int]]) -> list[str]:
    full_condition = condition_metric_key(FOUR_GROUP_ORDER)
    full_best = [
        metric_key
        for metric_key, summary in best_worst.items()
        if any(cond == full_condition for cond, _, _ in summary["best"])
    ]
    full_worst = [
        metric_key
        for metric_key, summary in best_worst.items()
        if any(cond == full_condition for cond, _, _ in summary["worst"])
    ]

    worst_counts: dict[str, int] = {}
    for summary in best_worst.values():
        for worst_condition, _, _ in summary["worst"]:
            worst_counts[worst_condition] = worst_counts.get(worst_condition, 0) + 1

    notes: list[str] = []
    if full_best:
        metrics_text = ", ".join(f"`{metric_key}`" for metric_key in full_best)
        notes.append(f"- The full `4g` model is strongest for {metrics_text}.")
    if full_worst:
        metrics_text = ", ".join(f"`{metric_key}`" for metric_key in full_worst)
        notes.append(f"- The full `4g` model is weakest for {metrics_text}.")

    if worst_counts:
        repeated_worst = max(worst_counts.items(), key=lambda item: (item[1], item[0]))
        if repeated_worst[1] > 1:
            notes.append(f"- {_format_condition_key(repeated_worst[0])} is weak across multiple metrics.")
    return notes


def effect_notes(
    added_effects: dict[str, dict[str, float]],
    presence_absence: dict[str, dict[str, float]],
    metric_keys: list[str],
) -> list[str]:
    notes: list[str] = []

    structure_metrics = [metric_key for metric_key in ("aji", "pq") if metric_key in metric_keys]
    realism_metrics = [metric_key for metric_key in ("fud", "style_hed", "lpips") if metric_key in metric_keys]

    if structure_metrics:
        group = max(
            FOUR_GROUP_ORDER,
            key=lambda candidate: statistics.mean(
                [added_effects.get(candidate, {}).get(metric_key, float("-inf")) for metric_key in structure_metrics]
            ),
        )
        notes.append(
            f"- `{group}` is the strongest positive contributor to {', '.join(f'`{metric_key}`' for metric_key in structure_metrics)}."
        )

    if realism_metrics:
        group = max(
            FOUR_GROUP_ORDER,
            key=lambda candidate: statistics.mean(
                [added_effects.get(candidate, {}).get(metric_key, float('-inf')) for metric_key in realism_metrics]
            ),
        )
        positive = [
            metric_key
            for metric_key in realism_metrics
            if added_effects.get(group, {}).get(metric_key, float("-inf")) > 0
        ]
        negative = [
            metric_key
            for metric_key in realism_metrics
            if added_effects.get(group, {}).get(metric_key, float("inf")) < 0
        ]
        if positive and negative:
            notes.append(
                f"- `{group}` helps {', '.join(f'`{metric_key}`' for metric_key in positive)} most on average, "
                f"but slightly hurts {', '.join(f'`{metric_key}`' for metric_key in negative)}."
            )
        elif positive:
            notes.append(
                f"- `{group}` helps the realism/style metrics most on average "
                f"({', '.join(f'`{metric_key}`' for metric_key in positive)})."
            )

    if presence_absence:
        structure_group = max(
            FOUR_GROUP_ORDER,
            key=lambda candidate: statistics.mean(
                [presence_absence.get(candidate, {}).get(metric_key, float("-inf")) for metric_key in structure_metrics]
            ) if structure_metrics else float("-inf"),
        )
        if structure_metrics:
            notes.append(
                f"- Conditions containing `{structure_group}` have the largest positive shift in {', '.join(f'`{metric_key}`' for metric_key in structure_metrics)}."
            )
    return notes


def loo_notes(loo_summary: dict[str, dict[str, float]]) -> list[str]:
    if not loo_summary:
        return ["- No leave-one-out stats were found under the requested dataset root."]

    ranked = sorted(
        FOUR_GROUP_ORDER,
        key=lambda group: loo_summary.get(group, {}).get("pct_pixels_above_10", float("-inf")),
        reverse=True,
    )
    dominant = ranked[:2]
    if len(dominant) < 2:
        return []
    return [
        f"- `{dominant[0]}` and `{dominant[1]}` create the largest localized pixel changes in leave-one-out diffs.",
        "- The global stain family can still look similar even when those local deltas are large.",
    ]


def fud_answer_notes(
    by_cardinality: dict[int, dict[str, float]],
    metric_keys: list[str],
) -> list[str]:
    notes: list[str] = []
    group_counts = sorted(by_cardinality)
    structural_metrics = [metric_key for metric_key in ("aji", "pq") if metric_key in metric_keys]
    realism_metrics = [metric_key for metric_key in ("fud", "style_hed", "lpips") if metric_key in metric_keys]

    if structural_metrics and group_counts:
        structural_improved = all(
            by_cardinality[group_counts[-1]].get(metric_key, float("-inf"))
            > by_cardinality[group_counts[0]].get(metric_key, float("-inf"))
            for metric_key in structural_metrics
        )
        if structural_improved:
            notes.append(
                f"- If 'better' means stronger structure-aware metrics, then more groups help: "
                + ", ".join(f"`{metric_key}`" for metric_key in structural_metrics)
                + f" rise from `{group_counts[0]}g -> {group_counts[-1]}g`."
            )

    for metric_key in realism_metrics:
        values = [by_cardinality[group_count].get(metric_key) for group_count in group_counts]
        if any(value is None for value in values):
            continue
        spec = METRIC_SPEC_BY_KEY[metric_key]
        oriented = values if spec.higher_is_better else [-float(value) for value in values]
        best_index = max(range(len(oriented)), key=lambda index: oriented[index])
        best_group = group_counts[best_index]
        if best_group != group_counts[-1]:
            notes.append(f"- `{metric_key}` is not best at `4g`; its best average is `{best_group}g`.")
    return notes


def render_markdown(
    title: str,
    metrics_root: Path,
    dataset_root: Path,
    tile_count: int,
    metric_keys: list[str],
    by_cardinality: dict[int, dict[str, float]],
    best_worst: dict[str, dict[str, list[tuple[str, float, float]] | int]],
    added_effects: dict[str, dict[str, float]],
    presence_absence: dict[str, dict[str, float]],
    loo_summary: dict[str, dict[str, float]],
    representative_tile: str | None,
) -> str:
    lines: list[str] = [f"# {title}", ""]
    lines.append(
        "This note summarizes the cached Stage 3 ablation results across "
        f"`{tile_count}` tile-level `metrics.json` files under `{metrics_root}`."
    )

    mapping_candidates = [
        dataset_root / "metadata" / "unpaired_mapping.json",
        dataset_root / "data" / "orion-crc33-unpaired" / "metadata" / "unpaired_mapping.json",
    ]
    mapping_path = next((path for path in mapping_candidates if path.is_file()), None)
    if mapping_path is not None:
        lines.append("")
        lines.append(
            "For this unpaired setup, `exp_channels/` stay attached to the layout tile while "
            f"`he/` and `features/` are remapped to a different style tile via "
            f"[unpaired_mapping.json]({mapping_path})."
        )

    evidence_links: list[tuple[str, Path]] = []
    dataset_metrics_path = dataset_root / "dataset_metrics_filtered.png"
    if not dataset_metrics_path.is_file():
        dataset_metrics_path = dataset_root / "dataset_metrics.png"
    if dataset_metrics_path.is_file():
        evidence_links.append((dataset_metrics_path.name, dataset_metrics_path))

    if representative_tile:
        rep_grid = dataset_root / "ablation_results" / representative_tile / "ablation_grid.png"
        rep_loo_png = dataset_root / "leave_one_out" / representative_tile / "leave_one_out_diff.png"
        rep_loo_stats = dataset_root / "leave_one_out" / representative_tile / "leave_one_out_diff_stats.json"
        if rep_grid.is_file():
            evidence_links.append((f"{representative_tile} ablation grid", rep_grid))
        if rep_loo_png.is_file():
            evidence_links.append((f"{representative_tile} leave-one-out diff", rep_loo_png))
        if rep_loo_stats.is_file():
            evidence_links.append((f"{representative_tile} leave-one-out stats", rep_loo_stats))

    channel_sweep_pngs = sorted((dataset_root / "channel_sweep").glob("**/*.png"))
    for path in channel_sweep_pngs[:3]:
        evidence_links.append((str(path.relative_to(dataset_root)), path))

    if evidence_links:
        lines.append("")
        lines.append("Evidence files:")
        lines.append("")
        for label, path in evidence_links:
            lines.append(f"- [{label}]({path})")

    lines.extend(["", "## 1. Average performance by number of active groups", ""])
    header = "| groups | " + " | ".join(metric_keys) + " |"
    divider = "| " + " | ".join(["------"] * (len(metric_keys) + 1)) + " |"
    lines.extend([header, divider])
    for group_count in sorted(by_cardinality):
        row = [f"{group_count}g"] + [_fmt(by_cardinality[group_count].get(metric_key)) for metric_key in metric_keys]
        lines.append("| " + " | ".join(row) + " |")

    lines.extend(["", "Interpretation:", ""])
    lines.extend(cardinality_notes(by_cardinality, metric_keys))

    lines.extend(["", "## 2. Best and worst conditions", ""])
    header = "| metric | best condition | value | worst condition | value |"
    divider = "| ------ | -------------- | ----- | --------------- | ----- |"
    lines.extend([header, divider])
    for metric_key in metric_keys:
        summary = best_worst.get(metric_key)
        if not summary:
            continue
        best_cond, best_value, _ = summary["best"][0]
        worst_cond, worst_value, _ = summary["worst"][0]
        row = [
            metric_key,
            _format_condition_key(best_cond),
            _fmt(best_value),
            _format_condition_key(worst_cond),
            _fmt(worst_value),
        ]
        lines.append("| " + " | ".join(row) + " |")

    lines.extend(["", "Interpretation:", ""])
    lines.extend(best_worst_notes(best_worst))

    lines.extend(["", "## 3. Average effect of adding each group", "", "Positive deltas mean \"adding this group tends to help.\"", ""])
    header = "| added group | " + " | ".join(
        f"delta {metric_key} (better +)" if not METRIC_SPEC_BY_KEY[metric_key].higher_is_better else f"delta {metric_key}"
        for metric_key in metric_keys
    ) + " |"
    divider = "| " + " | ".join(["-----------"] * (len(metric_keys) + 1)) + " |"
    lines.extend([header, divider])
    for group in FOUR_GROUP_ORDER:
        row = [f"`{group}`"] + [_fmt(added_effects.get(group, {}).get(metric_key)) for metric_key in metric_keys]
        lines.append("| " + " | ".join(row) + " |")

    lines.extend(["", "Interpretation:", ""])
    lines.extend(effect_notes(added_effects, presence_absence, metric_keys))

    lines.extend(["", "## 4. Presence-vs-absence summary", "", "This table compares all conditions where a group is present versus absent.", ""])
    header = "| group present? | " + " | ".join(f"{metric_key} delta" for metric_key in metric_keys) + " |"
    divider = "| " + " | ".join(["--------------"] * (len(metric_keys) + 1)) + " |"
    lines.extend([header, divider])
    for group in FOUR_GROUP_ORDER:
        row = [f"`{group}`"] + [_fmt(presence_absence.get(group, {}).get(metric_key)) for metric_key in metric_keys]
        lines.append("| " + " | ".join(row) + " |")

    lines.extend(["", "Notes:", ""])
    for metric_key in metric_keys:
        spec = METRIC_SPEC_BY_KEY[metric_key]
        if spec.higher_is_better:
            lines.append(f"- For `{metric_key}`, higher is better, so a positive delta is an improvement.")
        else:
            lines.append(f"- For `{metric_key}`, lower is better, so a positive delta means the group helps.")

    lines.extend(["", "## 5. Figure-based evidence", ""])
    if representative_tile:
        lines.append(
            f"Representative leave-one-out tile: `{representative_tile}` "
            f"(chosen as the closest tile to the dataset-average leave-one-out profile)."
        )
        lines.append("")
    header = "| removed group | mean diff | max diff | pct pixels > 10 |"
    divider = "| ----------- | --------- | -------- | --------------- |"
    lines.extend([header, divider])
    for group in FOUR_GROUP_ORDER:
        stats = loo_summary.get(group)
        if not stats:
            continue
        row = [
            f"`{group}`",
            _fmt(stats.get("mean_diff")),
            _fmt(stats.get("max_diff")),
            _fmt(stats.get("pct_pixels_above_10")),
        ]
        lines.append("| " + " | ".join(row) + " |")

    lines.extend(["", "Interpretation:", ""])
    lines.extend(loo_notes(loo_summary))

    lines.extend(["", "## 6. Answer to the FUD question", "", "Does adding more groups generate \"worse\" results?", "", "Not in a blanket sense.", ""])
    lines.extend(fud_answer_notes(by_cardinality, metric_keys))

    return "\n".join(lines).rstrip() + "\n"


def build_report(title: str, metrics_root: Path, dataset_root: Path) -> str:
    condition_means, tile_count, metric_keys = load_condition_means(metrics_root)
    by_cardinality = summarize_by_cardinality(condition_means, metric_keys)
    best_worst = summarize_best_worst(condition_means, metric_keys)
    added_effects = summarize_added_group_effects(condition_means, metric_keys)
    presence_absence = summarize_presence_absence(condition_means, metric_keys)
    loo_summary, representative_tile = load_leave_one_out_summary(dataset_root / "leave_one_out")
    return render_markdown(
        title=title,
        metrics_root=metrics_root,
        dataset_root=dataset_root,
        tile_count=tile_count,
        metric_keys=metric_keys,
        by_cardinality=by_cardinality,
        best_worst=best_worst,
        added_effects=added_effects,
        presence_absence=presence_absence,
        loo_summary=loo_summary,
        representative_tile=representative_tile,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize ablation metrics into a markdown report.")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Dataset root containing ablation_results/ and leave_one_out/")
    parser.add_argument("--metrics-root", type=Path, default=None, help="Explicit metrics root (default: <dataset-root>/ablation_results)")
    parser.add_argument("--title", type=str, default="Channel Ablation Summary", help="Markdown heading for the report")
    parser.add_argument("--output", type=Path, required=True, help="Where to write the markdown report")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    metrics_root = args.metrics_root.resolve() if args.metrics_root is not None else dataset_root / "ablation_results"
    report = build_report(args.title, metrics_root, dataset_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"Wrote summary → {args.output}")


fid_answer_notes = fud_answer_notes


if __name__ == "__main__":
    main()
