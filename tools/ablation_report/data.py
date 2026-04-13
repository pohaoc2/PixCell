from __future__ import annotations

import json
import statistics
import sys
from itertools import combinations
from pathlib import Path

from tools.compute_ablation_metrics import (
    _load_rgb_pil as load_rgb_pil_local,
    compute_style_hed_for_pair as compute_style_hed_scores_local,
)
from tools.stage3.ablation_vis_utils import (
    FOUR_GROUP_ORDER,
    condition_metric_key,
    ordered_subset_condition_tuples,
)
from tools.summarize_ablation_report import (
    best_worst_notes,
    cardinality_notes,
    effect_notes,
    fud_answer_notes,
    loo_notes,
    summarize_added_group_effects,
    summarize_best_worst,
    summarize_by_cardinality,
    summarize_presence_absence,
)

from .shared import (
    ROOT,
    _mean_std,
    DEFAULT_METRIC_ORDER,
    DatasetSummary,
    METRIC_SPEC_BY_KEY,
    TRADEOFF_METRIC_ORDER,
)


def load_fud_scores(metrics_root: Path) -> dict[str, float]:
    root = Path(metrics_root)
    path = None
    # Prefer explicit FUD outputs when present; older FVD/FID files are fallback-only.
    for candidate_name in ("fud_scores.json", "fvd_scores.json", "fid_scores.json"):
        candidate = root / candidate_name
        if candidate.is_file():
            path = candidate
            break
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}
    out: dict[str, float] = {}
    for cond_key, value in payload.items():
        try:
            out[str(cond_key)] = float(value)
        except (TypeError, ValueError):
            continue
    return out


def _has_gt_cell_masks(root: Path) -> bool:
    exp_dir = Path(root) / "exp_channels"
    return (exp_dir / "cell_masks").is_dir() or (exp_dir / "cell_mask").is_dir()


def resolve_gt_root(
    *,
    dataset_root: Path,
    slug: str,
    reference_root: Path | None = None,
) -> Path | None:
    candidates: list[Path] = []
    if reference_root is not None:
        candidates.append(reference_root)
    if slug == "unpaired":
        candidates.extend(
            [
                dataset_root / "data" / "orion-crc33-unpaired",
                ROOT / "inference_output" / "unpaired_ablation" / "data" / "orion-crc33-unpaired",
            ]
        )
    candidates.extend(
        [
            dataset_root / "data" / "orion-crc33",
            ROOT / "data" / "orion-crc33",
        ]
    )
    for candidate in candidates:
        if _has_gt_cell_masks(candidate):
            return candidate
    return None


def filter_metrics_paths_by_gt_cells(
    metrics_root: Path,
    *,
    gt_root: Path | None,
    min_gt_cells: int = 0,
) -> tuple[list[Path], int]:
    metrics_paths = sorted(Path(metrics_root).glob("*/metrics.json"))
    if not metrics_paths:
        raise FileNotFoundError(f"no per-tile metrics.json files found under {Path(metrics_root).resolve()}")
    if min_gt_cells <= 0 or gt_root is None:
        return metrics_paths, 0

    from tools.compute_ablation_metrics import _instance_ids, _load_gt_instance_mask

    kept: list[Path] = []
    filtered = 0
    for metrics_path in metrics_paths:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        tile_id = str(payload.get("tile_id", "") or metrics_path.parent.name).strip()
        if not tile_id:
            filtered += 1
            continue
        try:
            gt = _load_gt_instance_mask(gt_root, tile_id)
        except FileNotFoundError:
            filtered += 1
            continue
        if _instance_ids(gt).size < min_gt_cells:
            filtered += 1
            continue
        kept.append(metrics_path)
    return kept, filtered


def load_condition_means_from_paths(metrics_paths: list[Path]) -> tuple[dict[str, dict[str, float]], int, list[str]]:
    grouped: dict[str, dict[str, list[float]]] = {}
    metric_keys: set[str] = set()
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
            for metric in DEFAULT_METRIC_ORDER:
                value = record.get(metric)
                if value is None:
                    continue
                bucket.setdefault(metric, []).append(float(value))
                metric_keys.add(metric)

    condition_means: dict[str, dict[str, float]] = {}
    for cond_key, record in grouped.items():
        condition_means[cond_key] = {
            metric_key: float(statistics.mean(values))
            for metric_key, values in record.items()
            if values
        }
    return condition_means, tile_count, [key for key in DEFAULT_METRIC_ORDER if key in metric_keys]


def load_tile_condition_metrics(
    metrics_root: Path,
    *,
    metrics_paths: list[Path] | None = None,
    reference_root: Path | None = None,
    style_mapping: dict[str, str] | None = None,
) -> tuple[list[dict[str, dict[str, float]]], list[str]]:
    metrics_paths = sorted(metrics_paths) if metrics_paths is not None else sorted(Path(metrics_root).glob("*/metrics.json"))
    if not metrics_paths:
        raise FileNotFoundError(f"no per-tile metrics.json files found under {Path(metrics_root).resolve()}")

    metric_keys: set[str] = set()
    tile_records: list[tuple[Path, dict[str, dict[str, float]]]] = []
    for metrics_path in metrics_paths:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        per_condition = payload.get("per_condition")
        if not isinstance(per_condition, dict):
            continue

        tile_record: dict[str, dict[str, float]] = {}
        for cond_key, record in per_condition.items():
            if not isinstance(record, dict):
                continue
            normalized: dict[str, float] = {}
            for metric in TRADEOFF_METRIC_ORDER:
                value = record.get(metric)
                if value is None and metric == "fud":
                    value = record.get("fid")
                if value is None:
                    continue
                normalized[metric] = float(value)
                metric_keys.add(metric)
            if normalized:
                tile_record[str(cond_key)] = normalized
        tile_records.append((metrics_path.parent, tile_record))

    fud_scores = load_fud_scores(metrics_root)
    if fud_scores:
        for _, tile_record in tile_records:
            for cond_key, fud_value in fud_scores.items():
                tile_record.setdefault(cond_key, {})["fud"] = float(fud_value)
                metric_keys.add("fud")

    if reference_root is not None:
        for cache_dir, tile_record in tile_records:
            needs_style_hed = any("style_hed" not in metrics for metrics in tile_record.values())
            if not needs_style_hed:
                continue
            try:
                scores = compute_style_hed_scores_local(cache_dir, reference_root, style_mapping=style_mapping)
            except (FileNotFoundError, ValueError):
                continue
            for cond_key, value in scores.items():
                tile_record.setdefault(cond_key, {})["style_hed"] = float(value)
                metric_keys.add("style_hed")

    return [tile_record for _, tile_record in tile_records], [metric for metric in TRADEOFF_METRIC_ORDER if metric in metric_keys]


def summarize_by_cardinality_tile_stats(
    tile_records: list[dict[str, dict[str, float]]],
    metric_keys: list[str],
) -> tuple[dict[int, dict[str, float]], dict[int, dict[str, tuple[float, float]]]]:
    grouped: dict[int, dict[str, list[float]]] = {}
    for tile_record in tile_records:
        per_tile: dict[int, dict[str, list[float]]] = {}
        for cond_key, metrics in tile_record.items():
            bucket = per_tile.setdefault(len(cond_key.split("+")), {})
            for metric_key in metric_keys:
                value = metrics.get(metric_key)
                if value is None:
                    continue
                bucket.setdefault(metric_key, []).append(float(value))

        for group_count, metric_lists in per_tile.items():
            dest = grouped.setdefault(group_count, {})
            for metric_key, values in metric_lists.items():
                if values:
                    dest.setdefault(metric_key, []).append(float(statistics.mean(values)))

    means: dict[int, dict[str, float]] = {}
    stats_summary: dict[int, dict[str, tuple[float, float]]] = {}
    for group_count, metric_lists in grouped.items():
        means[group_count] = {}
        stats_summary[group_count] = {}
        for metric_key, values in metric_lists.items():
            stats = _mean_std(values)
            if stats is None:
                continue
            means[group_count][metric_key] = float(stats[0])
            stats_summary[group_count][metric_key] = stats
    return means, stats_summary


def summarize_condition_stats(
    tile_records: list[dict[str, dict[str, float]]],
    metric_keys: list[str],
) -> dict[str, dict[str, tuple[float, float]]]:
    summary: dict[str, dict[str, tuple[float, float]]] = {}
    for cond in ordered_subset_condition_tuples():
        cond_key = condition_metric_key(cond)
        summary[cond_key] = {}
        for metric_key in metric_keys:
            values = [
                float(tile_record[cond_key][metric_key])
                for tile_record in tile_records
                if cond_key in tile_record and metric_key in tile_record[cond_key]
            ]
            stats = _mean_std(values)
            if stats is not None:
                summary[cond_key][metric_key] = stats
    return summary


def summarize_added_group_effect_stats(
    condition_means: dict[str, dict[str, float]],
    metric_keys: list[str],
) -> dict[str, dict[str, tuple[float, float]]]:
    summary: dict[str, dict[str, tuple[float, float]]] = {}
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
        summary[group] = {}
        for metric_key, values in deltas.items():
            stats = _mean_std(values)
            if stats is not None:
                summary[group][metric_key] = stats
    return summary


def summarize_by_cardinality_stats(
    condition_means: dict[str, dict[str, float]],
    metric_keys: list[str],
) -> dict[int, dict[str, tuple[float, float]]]:
    grouped: dict[int, dict[str, list[float]]] = {}
    for cond_key, metrics in condition_means.items():
        bucket = grouped.setdefault(len(cond_key.split("+")), {})
        for metric_key in metric_keys:
            value = metrics.get(metric_key)
            if value is None:
                continue
            bucket.setdefault(metric_key, []).append(float(value))

    summary: dict[int, dict[str, tuple[float, float]]] = {}
    for group_count, metric_lists in grouped.items():
        summary[group_count] = {}
        for metric_key, values in metric_lists.items():
            stats = _mean_std(values)
            if stats is not None:
                summary[group_count][metric_key] = stats
    return summary


def summarize_presence_absence_stats(
    condition_means: dict[str, dict[str, float]],
    metric_keys: list[str],
) -> dict[str, dict[str, tuple[float, float]]]:
    summary: dict[str, dict[str, tuple[float, float]]] = {}
    for group in FOUR_GROUP_ORDER:
        present: dict[str, list[float]] = {metric_key: [] for metric_key in metric_keys}
        absent: dict[str, list[float]] = {metric_key: [] for metric_key in metric_keys}
        for cond_key, metrics in condition_means.items():
            target = present if group in cond_key.split("+") else absent
            for metric_key in metric_keys:
                value = metrics.get(metric_key)
                if value is not None:
                    target[metric_key].append(float(value))

        summary[group] = {}
        for metric_key in metric_keys:
            present_values = present[metric_key]
            absent_values = absent[metric_key]
            if not present_values or not absent_values:
                continue
            spec = METRIC_SPEC_BY_KEY[metric_key]
            pairwise_deltas = [
                (present_value - absent_value) if spec.higher_is_better else (absent_value - present_value)
                for present_value in present_values
                for absent_value in absent_values
            ]
            stats = _mean_std(pairwise_deltas)
            if stats is not None:
                summary[group][metric_key] = stats
    return summary


def load_leave_one_out_summary_stats(
    loo_root: Path,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, tuple[float, float]]], str | None]:
    stats_paths = sorted(Path(loo_root).glob("*/leave_one_out_diff_stats.json"))
    if not stats_paths:
        return {}, {}, None

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
                if key in {"mean_diff", "pct_pixels_above_10"}:
                    vector[f"{group}:{key}"] = float(value)
        if vector:
            tile_vectors[stats_path.parent.name] = vector

    means: dict[str, dict[str, float]] = {}
    stats_summary: dict[str, dict[str, tuple[float, float]]] = {}
    for group, stats in grouped.items():
        means[group] = {}
        stats_summary[group] = {}
        for key, values in stats.items():
            stats_pair = _mean_std(values)
            if stats_pair is None:
                continue
            means[group][key] = stats_pair[0]
            stats_summary[group][key] = stats_pair

    representative_tile = None
    if tile_vectors:
        axes = sorted({axis for vector in tile_vectors.values() for axis in vector})
        axis_means = {
            axis: statistics.mean([vector[axis] for vector in tile_vectors.values() if axis in vector])
            for axis in axes
        }
        axis_stds = {}
        for axis in axes:
            values = [vector[axis] for vector in tile_vectors.values() if axis in vector]
            axis_std = statistics.pstdev(values) if len(values) > 1 else 1.0
            axis_stds[axis] = axis_std if axis_std > 0 else 1.0

        best_tile = None
        best_distance = None
        for tile_id, vector in tile_vectors.items():
            distance = 0.0
            for axis in axes:
                if axis not in vector:
                    continue
                z_score = (vector[axis] - axis_means[axis]) / axis_stds[axis]
                distance += z_score * z_score
            if best_distance is None or distance < best_distance:
                best_tile = tile_id
                best_distance = distance
        representative_tile = best_tile

    return means, stats_summary, representative_tile


def resolve_first_existing(candidates: list[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def resolve_loo_root(metrics_root: Path, dataset_root: Path) -> Path:
    for candidate in (dataset_root / "leave_one_out", metrics_root):
        if candidate.is_dir() and next(candidate.glob("*/leave_one_out_diff_stats.json"), None) is not None:
            return candidate
    return metrics_root


def resolve_representative_paths(
    metrics_root: Path,
    dataset_root: Path,
    representative_tile: str | None,
) -> tuple[Path | None, Path | None]:
    if not representative_tile:
        return None, None
    ablation_grid = resolve_first_existing(
        [
            dataset_root / "ablation_results" / representative_tile / "ablation_grid.png",
            metrics_root / representative_tile / "ablation_grid.png",
        ]
    )
    loo_diff = resolve_first_existing(
        [
            dataset_root / "leave_one_out" / representative_tile / "leave_one_out_diff.png",
            metrics_root / representative_tile / "leave_one_out_diff.png",
        ]
    )
    return ablation_grid, loo_diff


def resolve_reference_root(
    *,
    dataset_root: Path,
    slug: str,
    reference_root: Path | None = None,
) -> Path | None:
    candidates: list[Path] = []
    if reference_root is not None:
        candidates.append(reference_root)
    if slug == "unpaired":
        candidates.extend(
            [
                dataset_root / "data" / "orion-crc33-unpaired",
                ROOT / "inference_output" / "unpaired_ablation" / "data" / "orion-crc33-unpaired",
            ]
        )
    candidates.extend(
        [
            dataset_root / "data" / "orion-crc33",
            ROOT / "data" / "orion-crc33",
        ]
    )
    for candidate in candidates:
        if (candidate / "he").is_dir():
            return candidate
    return None


def add_missing_style_hed_means(
    condition_means: dict[str, dict[str, float]],
    metrics_root: Path,
    metric_keys: list[str],
    *,
    reference_root: Path | None,
    style_mapping: dict[str, str] | None = None,
) -> tuple[dict[str, dict[str, float]], list[str]]:
    if "style_hed" in metric_keys or reference_root is None:
        return condition_means, metric_keys

    grouped: dict[str, list[float]] = {}
    for metrics_path in sorted(metrics_root.glob("*/metrics.json")):
        cache_dir = metrics_path.parent
        try:
            scores = compute_style_hed_scores_local(cache_dir, reference_root, style_mapping=style_mapping)
        except (FileNotFoundError, ValueError):
            continue
        for cond_key, value in scores.items():
            grouped.setdefault(cond_key, []).append(float(value))

    if not grouped:
        return condition_means, metric_keys

    augmented = {
        cond_key: dict(metrics)
        for cond_key, metrics in condition_means.items()
    }
    for cond_key, values in grouped.items():
        if not values:
            continue
        augmented.setdefault(cond_key, {})["style_hed"] = float(statistics.mean(values))

    present = set(metric_keys)
    present.add("style_hed")
    ordered = [metric for metric in DEFAULT_METRIC_ORDER if metric in present]
    return augmented, ordered


def add_missing_fud_means(
    condition_means: dict[str, dict[str, float]],
    metrics_root: Path,
    metric_keys: list[str],
) -> tuple[dict[str, dict[str, float]], list[str]]:
    if "fud" in metric_keys:
        return condition_means, metric_keys

    fud_scores = load_fud_scores(metrics_root)
    if not fud_scores:
        return condition_means, metric_keys

    augmented = {
        cond_key: dict(metrics)
        for cond_key, metrics in condition_means.items()
    }
    for cond_key, value in fud_scores.items():
        augmented.setdefault(cond_key, {})["fud"] = float(value)

    present = set(metric_keys)
    present.add("fud")
    ordered = [metric for metric in DEFAULT_METRIC_ORDER if metric in present]
    return augmented, ordered


def load_dataset_summary(
    *,
    slug: str,
    title: str,
    metrics_root: Path,
    dataset_root: Path,
    reference_root: Path | None = None,
    style_mapping: dict[str, str] | None = None,
    enable_style_hed_backfill: bool = True,
    min_gt_cells: int = 0,
) -> DatasetSummary:
    resolved_gt_root = resolve_gt_root(
        dataset_root=dataset_root,
        slug=slug,
        reference_root=reference_root,
    )
    resolved_reference_root = (
        resolve_reference_root(
            dataset_root=dataset_root,
            slug=slug,
            reference_root=reference_root,
        )
        if enable_style_hed_backfill
        else None
    )
    filtered_metrics_paths, filtered_count = filter_metrics_paths_by_gt_cells(
        metrics_root,
        gt_root=resolved_gt_root if min_gt_cells > 0 else None,
        min_gt_cells=min_gt_cells,
    )
    if min_gt_cells > 0 and filtered_count:
        print(
            f"[{slug}] Filtered {filtered_count} tiles with < {min_gt_cells} GT cells "
            f"({len(filtered_metrics_paths)} kept)",
            file=sys.stderr,
        )
    if not filtered_metrics_paths:
        raise ValueError(
            f"no tiles remain under {metrics_root} after applying min_gt_cells={min_gt_cells}"
        )

    condition_means, tile_count, metric_keys = load_condition_means_from_paths(filtered_metrics_paths)
    condition_means, metric_keys = add_missing_fud_means(
        condition_means,
        metrics_root,
        metric_keys,
    )
    condition_means, metric_keys = add_missing_style_hed_means(
        condition_means,
        metrics_root,
        metric_keys,
        reference_root=resolved_reference_root,
        style_mapping=style_mapping,
    )
    tile_records, tile_metric_keys = load_tile_condition_metrics(
        metrics_root,
        metrics_paths=filtered_metrics_paths,
        reference_root=resolved_reference_root,
        style_mapping=style_mapping,
    )
    condition_stat_metric_keys = list(metric_keys)
    if tile_metric_keys:
        present_metrics = set(metric_keys) | set(tile_metric_keys)
        metric_keys = [metric for metric in DEFAULT_METRIC_ORDER if metric in present_metrics]
        condition_stat_metric_keys = [metric for metric in TRADEOFF_METRIC_ORDER if metric in present_metrics]
    condition_stats = summarize_condition_stats(tile_records, condition_stat_metric_keys)
    by_cardinality, by_cardinality_stats = summarize_by_cardinality_tile_stats(tile_records, metric_keys)
    if not by_cardinality:
        by_cardinality = summarize_by_cardinality(condition_means, metric_keys)
    if not by_cardinality_stats:
        by_cardinality_stats = summarize_by_cardinality_stats(condition_means, metric_keys)
    best_worst = summarize_best_worst(condition_means, metric_keys, condition_stats)
    added_effects = summarize_added_group_effects(condition_means, metric_keys)
    presence_absence = summarize_presence_absence(condition_means, metric_keys)
    added_effect_stats = summarize_added_group_effect_stats(condition_means, metric_keys)
    presence_absence_stats = summarize_presence_absence_stats(condition_means, metric_keys)
    loo_summary, loo_stats, representative_tile = load_leave_one_out_summary_stats(
        resolve_loo_root(metrics_root, dataset_root)
    )
    ablation_grid_path, loo_diff_path = resolve_representative_paths(metrics_root, dataset_root, representative_tile)

    notes: list[str] = []
    notes.extend(cardinality_notes(by_cardinality, metric_keys))
    notes.extend(best_worst_notes(best_worst))
    notes.extend(effect_notes(added_effects, presence_absence, metric_keys))
    notes.extend(loo_notes(loo_summary))
    notes.extend(fud_answer_notes(by_cardinality, metric_keys))

    seen: set[str] = set()
    deduped_notes: list[str] = []
    for note in notes:
        stripped = note.strip()
        if stripped and stripped not in seen:
            deduped_notes.append(stripped)
            seen.add(stripped)

    return DatasetSummary(
        slug=slug,
        title=title,
        metrics_root=metrics_root,
        dataset_root=dataset_root,
        tile_count=tile_count,
        metric_keys=metric_keys,
        condition_stats=condition_stats,
        by_cardinality=by_cardinality,
        by_cardinality_stats=by_cardinality_stats,
        best_worst=best_worst,
        added_effects=added_effects,
        presence_absence=presence_absence,
        added_effect_stats=added_effect_stats,
        presence_absence_stats=presence_absence_stats,
        loo_summary=loo_summary,
        loo_stats=loo_stats,
        representative_tile=representative_tile,
        key_takeaways=deduped_notes[:8],
        ablation_grid_path=ablation_grid_path,
        loo_diff_path=loo_diff_path,
    )
