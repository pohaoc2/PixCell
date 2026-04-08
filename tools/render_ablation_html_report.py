#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import html
import io
import json
import os
import statistics
import sys
import textwrap
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mpl-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from PIL import Image

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.stage3.ablation_vis_utils import (
    FOUR_GROUP_ORDER,
    condition_metric_key,
    normalize_group_name,
    ordered_subset_condition_tuples,
)
from tools.summarize_ablation_report import (
    DEFAULT_METRIC_ORDER,
    METRIC_SPEC_BY_KEY,
    best_worst_notes,
    cardinality_notes,
    effect_notes,
    fud_answer_notes,
    load_condition_means,
    loo_notes,
    summarize_added_group_effects,
    summarize_best_worst,
    summarize_by_cardinality,
    summarize_presence_absence,
)

OKABE_BLUE = "#4C78A8"
OKABE_ORANGE = "#E28E2B"
OKABE_GREEN = "#5C8F5B"
OKABE_PURPLE = "#8D6A9F"
OKABE_RED = "#B22222"
OKABE_TEAL = "#5B8F96"
OKABE_GRAY = "#565656"
INK = "#000000"
SOFT_GRID = "#D7D6D2"
_RGB_FROM_HED = np.array(
    [
        [0.65, 0.70, 0.29],
        [0.07, 0.99, 0.11],
        [0.27, 0.57, 0.78],
    ],
    dtype=np.float64,
)
_HED_FROM_RGB = np.linalg.inv(_RGB_FROM_HED)
METRIC_COLORS = {
    "cosine": OKABE_BLUE,
    "lpips": OKABE_ORANGE,
    "aji": OKABE_GREEN,
    "pq": OKABE_PURPLE,
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
    "microenv": "Microenv",
}
GROUP_SHORT_LABELS = {
    "cell_types": "CT",
    "cell_state": "CS",
    "vasculature": "Vas",
    "microenv": "Env",
}
METRIC_LABELS = {
    "cosine": "Cosine",
    "lpips": "LPIPS",
    "aji": "AJI",
    "pq": "PQ",
    "fud": "FUD",
    "style_hed": "HED",
}
CHANNEL_EFFECT_CMAP = plt.get_cmap("RdBu")


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
    best_worst: dict[str, dict[str, str | float]]
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


def load_fud_scores(metrics_root: Path) -> dict[str, float]:
    root = Path(metrics_root)
    path = root / "fud_scores.json"
    if not path.is_file():
        path = root / "fid_scores.json"
    if not path.is_file():
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
            for metric in DEFAULT_METRIC_ORDER:
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
                scores = compute_style_hed_scores_local(cache_dir, reference_root)
            except (FileNotFoundError, ValueError):
                continue
            for cond_key, value in scores.items():
                tile_record.setdefault(cond_key, {})["style_hed"] = float(value)
                metric_keys.add("style_hed")

    return [tile_record for _, tile_record in tile_records], [metric for metric in DEFAULT_METRIC_ORDER if metric in metric_keys]


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


def load_manifest_local(cache_dir: Path) -> dict:
    manifest_path = cache_dir / "manifest.json"
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def tile_id_from_manifest_local(cache_dir: Path) -> str:
    manifest = load_manifest_local(cache_dir)
    tile_id = str(manifest.get("tile_id", "")).strip()
    if not tile_id:
        raise ValueError(f"manifest.json must contain tile_id: {cache_dir / 'manifest.json'}")
    return tile_id


def iter_condition_images_local(cache_dir: Path) -> dict[str, Path]:
    manifest = load_manifest_local(cache_dir)
    per_condition: dict[str, Path] = {}
    for section in manifest.get("sections", []):
        for entry in section.get("entries", []):
            key = condition_metric_key(tuple(entry["active_groups"]))
            per_condition[key] = cache_dir / entry["image_path"]
    all4_path = cache_dir / "all" / "generated_he.png"
    if all4_path.is_file():
        per_condition[condition_metric_key(FOUR_GROUP_ORDER)] = all4_path
    return per_condition


def load_rgb_pil_local(path: Path, *, size: tuple[int, int] | None = None) -> Image.Image:
    image = Image.open(path).convert("RGB")
    if size is not None and image.size != size:
        image = image.resize(size, Image.BILINEAR)
    return image


def rgb_to_hed_local(image: Image.Image) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float64) / 255.0
    arr = np.clip(arr, 1e-6, 1.0)
    optical_density = -np.log(arr)
    return optical_density @ _HED_FROM_RGB.T


def tissue_mask_from_rgb_local(image: Image.Image) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return np.mean(arr, axis=2) < 0.95


def masked_mean_std_local(values: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    data = values[np.asarray(mask, dtype=bool)]
    if data.size == 0:
        data = values.reshape(-1)
    return float(np.mean(data)), float(np.std(data))


def compute_style_hed_scores_local(cache_dir: Path, reference_root: Path) -> dict[str, float]:
    tile_id = tile_id_from_manifest_local(cache_dir)
    ref_path = resolve_first_existing(
        [reference_root / "he" / f"{tile_id}{ext}" for ext in (".png", ".jpg", ".jpeg", ".tif")]
    )
    if ref_path is None:
        raise FileNotFoundError(f"reference H&E not found for tile {tile_id!r}")

    ref_img = load_rgb_pil_local(ref_path)
    ref_hed = rgb_to_hed_local(ref_img)
    ref_mask = tissue_mask_from_rgb_local(ref_img)
    scores: dict[str, float] = {}

    for cond_key, img_path in iter_condition_images_local(cache_dir).items():
        if not img_path.is_file():
            raise FileNotFoundError(f"generated image not found: {img_path}")
        gen_img = load_rgb_pil_local(img_path, size=ref_img.size)
        gen_hed = rgb_to_hed_local(gen_img)
        gen_mask = tissue_mask_from_rgb_local(gen_img)
        tissue_mask = ref_mask | gen_mask

        score = 0.0
        for stain_channel in (0, 1):
            ref_mean, ref_std = masked_mean_std_local(ref_hed[..., stain_channel], tissue_mask)
            gen_mean, gen_std = masked_mean_std_local(gen_hed[..., stain_channel], tissue_mask)
            score += abs(gen_mean - ref_mean) + abs(gen_std - ref_std)
        scores[cond_key] = float(score)
    return scores


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
) -> tuple[dict[str, dict[str, float]], list[str]]:
    if "style_hed" in metric_keys or reference_root is None:
        return condition_means, metric_keys

    grouped: dict[str, list[float]] = {}
    for metrics_path in sorted(metrics_root.glob("*/metrics.json")):
        cache_dir = metrics_path.parent
        try:
            scores = compute_style_hed_scores_local(cache_dir, reference_root)
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
    )
    tile_records, tile_metric_keys = load_tile_condition_metrics(
        metrics_root,
        metrics_paths=filtered_metrics_paths,
        reference_root=resolved_reference_root,
    )
    if tile_metric_keys:
        present_metrics = set(metric_keys) | set(tile_metric_keys)
        metric_keys = [metric for metric in DEFAULT_METRIC_ORDER if metric in present_metrics]
    condition_stats = summarize_condition_stats(tile_records, metric_keys)
    by_cardinality, by_cardinality_stats = summarize_by_cardinality_tile_stats(tile_records, metric_keys)
    if not by_cardinality:
        by_cardinality = summarize_by_cardinality(condition_means, metric_keys)
    if not by_cardinality_stats:
        by_cardinality_stats = summarize_by_cardinality_stats(condition_means, metric_keys)
    best_worst = summarize_best_worst(condition_means, metric_keys)
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


def figure_to_data_uri(fig: plt.Figure, *, dpi: int = 180) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def save_figure_png(fig: plt.Figure, output_path: Path, *, dpi: int = 220) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def image_file_to_data_uri(path: Path) -> str:
    mime = "image/png"
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        mime = "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


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


def render_condition_glyph(condition: object) -> str:
    active_groups = condition_groups(condition)
    label = format_condition(condition)
    dots = "".join(
        (
            f"<span class='condition-dot{' is-active' if group in active_groups else ''}' "
            f"title='{html.escape(GROUP_SHORT_LABELS[group], quote=True)}'></span>"
        )
        for group in FOUR_GROUP_ORDER
    )
    return (
        f"<span class='condition-glyph' aria-label='{html.escape(label, quote=True)}' "
        f"title='{html.escape(label, quote=True)}'>{dots}</span>"
    )


def condition_indicator_text(condition: object) -> str:
    active_groups = condition_groups(condition)
    return " ".join("●" if group in active_groups else "○" for group in FOUR_GROUP_ORDER)


def comparison_metric_keys(summary: DatasetSummary) -> list[str]:
    return list(summary.metric_keys)


def metric_tradeoff_keys(summaries: list[DatasetSummary]) -> list[str]:
    present = {metric for summary in summaries for metric in summary.metric_keys}
    return [metric for metric in DEFAULT_METRIC_ORDER if metric in present]


def build_metric_trends_figure(summaries: list[DatasetSummary]) -> plt.Figure:
    metrics = metric_tradeoff_keys(summaries)
    fig = plt.figure(figsize=(15.6, 7.7))
    outer = fig.add_gridspec(2, 3, wspace=0.22, hspace=0.28)
    condition_tuples = ordered_subset_condition_tuples()
    condition_keys = [condition_metric_key(cond) for cond in condition_tuples]
    x_positions = list(range(len(condition_keys)))
    spans: list[tuple[int, int, int]] = []
    start = 0
    for index in range(1, len(condition_tuples)):
        if len(condition_tuples[index]) != len(condition_tuples[index - 1]):
            spans.append((len(condition_tuples[index - 1]), start, index - 1))
            start = index
    spans.append((len(condition_tuples[-1]), start, len(condition_tuples) - 1))
    dataset_styles = {
        "paired": {"linestyle": "-", "markerfacecolor": INK, "markeredgecolor": INK, "error_linestyle": "solid"},
        "unpaired": {"linestyle": ":", "markerfacecolor": "white", "markeredgecolor": INK, "error_linestyle": (0, (4, 2))},
    }

    for index, metric_key in enumerate(metrics):
        row, col = divmod(index, 3)
        subgrid = outer[row, col].subgridspec(2, 1, height_ratios=[7.0, 2.0], hspace=0.02)
        ax = fig.add_subplot(subgrid[0, 0])
        dot_ax = fig.add_subplot(subgrid[1, 0], sharex=ax)
        all_values: list[float] = []
        for summary in summaries:
            valid: list[tuple[int, float, float]] = []
            for x_value, cond_key in zip(x_positions, condition_keys, strict=True):
                stats = summary.condition_stats.get(cond_key, {}).get(metric_key)
                if stats is None:
                    continue
                valid.append((x_value, float(stats[0]), float(stats[1])))
            if not valid:
                continue
            xs, ys, stds = zip(*valid)
            for value, std_value in zip(ys, stds, strict=True):
                all_values.extend([value - std_value, value + std_value])
            style = dataset_styles[summary.slug]
            container = ax.errorbar(
                xs,
                ys,
                yerr=stds,
                color=INK,
                linestyle=style["linestyle"],
                marker="o",
                markerfacecolor=style["markerfacecolor"],
                markeredgecolor=style["markeredgecolor"],
                markeredgewidth=1.1,
                linewidth=1.6,
                markersize=4.8,
                capsize=2.0,
                elinewidth=0.9,
                zorder=4,
            )
            _, _, barlinecols = container.lines
            error_linestyle = style["error_linestyle"]
            for barlinecol in barlinecols:
                barlinecol.set_linestyle(error_linestyle)

        if not all_values:
            ax.axis("off")
            dot_ax.axis("off")
            continue

        lo, hi = _tight_range(all_values)
        spec = METRIC_SPEC_BY_KEY[metric_key]
        arrow = "↑" if spec.higher_is_better else "↓"
        ax.set_title(
            f"{METRIC_LABELS.get(metric_key, metric_key)} {arrow}",
            color=INK,
            fontweight="bold",
            pad=16,
        )
        for _, start_idx, end_idx in spans[:-1]:
            ax.axvline(end_idx + 0.5, color="#BEBEBE", linewidth=0.9, linestyle=(0, (3, 2.5)), zorder=1)
            dot_ax.axvline(end_idx + 0.5, color="#D0D0D0", linewidth=0.9, linestyle=(0, (3, 2.5)), zorder=1)
        for cardinality, start_idx, end_idx in spans:
            center = (start_idx + end_idx) / 2
            ax.text(
                center,
                1.01,
                f"{cardinality}g",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=7.2,
                color="#666666",
            )
        ax.set_xlim(-0.55, len(condition_keys) - 0.45)
        ax.set_xticks([])
        ax.set_ylim(lo, hi)
        ax.grid(True, axis="y", color=SOFT_GRID, linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_color(INK)
        ax.spines["left"].set_color(INK)
        ax.tick_params(axis="x", length=0)
        ax.tick_params(axis="y", colors=INK, labelsize=8.6)
        ax.set_axisbelow(True)
        dot_ax.set_facecolor("white")
        for x_value, cond in zip(x_positions, condition_tuples, strict=True):
            active_groups = set(cond)
            for row_index, group in enumerate(FOUR_GROUP_ORDER):
                y_value = 3 - row_index
                is_active = group in active_groups
                dot_ax.scatter(
                    x_value,
                    y_value,
                    s=24 if is_active else 21,
                    facecolors=INK if is_active else "white",
                    edgecolors=INK if is_active else "#A0A0A0",
                    linewidths=0.9 if is_active else 1.0,
                    zorder=3,
                )
        dot_ax.set_xlim(-0.55, len(condition_keys) - 0.45)
        dot_ax.set_ylim(-0.6, 3.6)
        dot_ax.axis("off")
        if col == 0:
            ymin, ymax = dot_ax.get_ylim()
            for row_index, group in enumerate(FOUR_GROUP_ORDER):
                y_value = 3 - row_index
                y_axes = (y_value - ymin) / (ymax - ymin)
                dot_ax.text(
                    -0.05,
                    y_axes,
                    GROUP_SHORT_LABELS[group],
                    transform=dot_ax.transAxes,
                    ha="right",
                    va="center",
                    fontsize=7.0,
                    color=INK,
                )

    for index in range(len(metrics), 6):
        row, col = divmod(index, 3)
        subgrid = outer[row, col].subgridspec(2, 1, height_ratios=[7.0, 2.0], hspace=0.02)
        fig.add_subplot(subgrid[0, 0]).axis("off")
        fig.add_subplot(subgrid[1, 0]).axis("off")

    handles = [
        Line2D([0], [0], color="#444444", linestyle="-", marker="o", markerfacecolor=INK, markeredgecolor=INK, markersize=5.0, linewidth=1.6, label="Paired"),
        Line2D(
            [0],
            [0],
            color="#444444",
            linestyle=":",
            marker="o",
            markerfacecolor="white",
            markeredgecolor="#444444",
            markersize=5.0,
            linewidth=1.6,
            label="Unpaired",
        ),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.962), fontsize=9.6)
    fig.suptitle("Metric tradeoffs across all 15 channel-group combinations", y=0.992, fontsize=14.0, fontweight="bold")
    fig.subplots_adjust(left=0.055, right=0.99, bottom=0.075, top=0.885)
    return fig


def heatmap_metric_keys(summary: DatasetSummary) -> list[str]:
    if summary.slug != "unpaired":
        return list(summary.metric_keys)
    filtered = [metric for metric in summary.metric_keys if metric not in {"cosine", "lpips"}]
    return filtered or list(summary.metric_keys)


def _heatmap_matrix(
    summary: DatasetSummary,
    source: dict[str, dict[str, tuple[float, float]]],
    metric_keys: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.full((len(FOUR_GROUP_ORDER), len(metric_keys)), np.nan, dtype=float)
    stds = np.full((len(FOUR_GROUP_ORDER), len(metric_keys)), np.nan, dtype=float)
    for row, group in enumerate(FOUR_GROUP_ORDER):
        for col, metric_key in enumerate(metric_keys):
            value = source.get(group, {}).get(metric_key)
            if value is not None:
                matrix[row, col] = float(value[0])
                stds[row, col] = float(value[1])
    return matrix, stds


def _heatmap_metric_ranges(summaries: list[DatasetSummary]) -> dict[str, tuple[float, float]]:
    per_metric_values: dict[str, list[float]] = {}
    for summary in summaries:
        metric_keys = heatmap_metric_keys(summary)
        for source in (summary.added_effect_stats, summary.presence_absence_stats):
            matrix, _ = _heatmap_matrix(summary, source, metric_keys)
            for col, metric_key in enumerate(metric_keys):
                values = matrix[:, col]
                finite_values = [float(value) for value in values[np.isfinite(values)]]
                if finite_values:
                    per_metric_values.setdefault(metric_key, []).extend(finite_values)

    ranges: dict[str, tuple[float, float]] = {}
    for metric_key, values in per_metric_values.items():
        ranges[metric_key] = (min(values), max(values))
    return ranges


def _normalize_heatmap_matrix(
    matrix: np.ndarray,
    metric_keys: list[str],
    metric_ranges: dict[str, tuple[float, float]],
) -> np.ndarray:
    normalized = np.full(matrix.shape, np.nan, dtype=float)
    for col, metric_key in enumerate(metric_keys):
        lo, hi = metric_ranges.get(metric_key, (0.0, 1.0))
        column = matrix[:, col]
        finite_mask = np.isfinite(column)
        if not np.any(finite_mask):
            continue
        if hi > lo:
            normalized[finite_mask, col] = (column[finite_mask] - lo) / (hi - lo)
        else:
            normalized[finite_mask, col] = 0.5
    return normalized


def _normalize_heatmap_stds(
    stds: np.ndarray,
    metric_keys: list[str],
    metric_ranges: dict[str, tuple[float, float]],
) -> np.ndarray:
    normalized = np.full(stds.shape, np.nan, dtype=float)
    for col, metric_key in enumerate(metric_keys):
        lo, hi = metric_ranges.get(metric_key, (0.0, 1.0))
        column = stds[:, col]
        finite_mask = np.isfinite(column)
        if not np.any(finite_mask):
            continue
        if hi > lo:
            normalized[finite_mask, col] = column[finite_mask] / (hi - lo)
        else:
            normalized[finite_mask, col] = 0.0
    return normalized


def build_channel_effect_heatmaps_figure(summaries: list[DatasetSummary]) -> plt.Figure:
    fig = plt.figure(figsize=(12.2, 3.9 * len(summaries)))
    grid = fig.add_gridspec(len(summaries), 3, width_ratios=[1, 1, 0.045], wspace=0.20, hspace=0.34)
    metric_ranges = _heatmap_metric_ranges(summaries)
    im = None

    for row, summary in enumerate(summaries):
        panels = [
            ("Average improvement from adding a group", summary.added_effect_stats),
            ("Present vs absent delta", summary.presence_absence_stats),
        ]
        for col, (panel_title, source) in enumerate(panels):
            ax = fig.add_subplot(grid[row, col])
            metric_keys = heatmap_metric_keys(summary)
            raw_matrix, stds = _heatmap_matrix(summary, source, metric_keys)
            normalized_matrix = _normalize_heatmap_matrix(raw_matrix, metric_keys, metric_ranges)
            normalized_stds = _normalize_heatmap_stds(stds, metric_keys, metric_ranges)
            masked = np.ma.masked_invalid(normalized_matrix)
            im = ax.imshow(masked, cmap=CHANNEL_EFFECT_CMAP, vmin=0.0, vmax=1.0, aspect="auto")
            ax.set_title(f"{summary.title}: {panel_title}", fontsize=11.0, fontweight="bold", color=INK)
            ax.set_yticks(range(len(FOUR_GROUP_ORDER)))
            if col == 0:
                ax.set_yticklabels([GROUP_LABELS[group] for group in FOUR_GROUP_ORDER], fontsize=10.0, color=INK)
            else:
                ax.set_yticklabels([])
            ax.set_xticks(range(len(metric_keys)))
            ax.set_xticklabels(
                [METRIC_LABELS.get(metric, metric) for metric in metric_keys],
                rotation=0,
                ha="center",
                fontsize=10.0,
                color=INK,
            )
            for r in range(raw_matrix.shape[0]):
                for c in range(raw_matrix.shape[1]):
                    value = raw_matrix[r, c]
                    if not np.isfinite(value):
                        continue
                    norm_value = normalized_matrix[r, c]
                    norm_std_value = normalized_stds[r, c]
                    ax.text(
                        c,
                        r,
                        f"{norm_value:.3f}\n±{norm_std_value:.3f}",
                        ha="center",
                        va="center",
                        fontsize=8.6,
                        fontweight="semibold",
                        color="white" if norm_value <= 0.20 or norm_value >= 0.70 else "#111111",
                    )
            ax.set_xticks(np.arange(-0.5, len(metric_keys), 1), minor=True)
            ax.set_yticks(np.arange(-0.5, len(FOUR_GROUP_ORDER), 1), minor=True)
            ax.grid(which="minor", color=INK, linewidth=0.8)
            ax.tick_params(which="minor", bottom=False, left=False)
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.tick_params(axis="x", length=0, pad=8)
            ax.tick_params(axis="y", length=0)

    cax = fig.add_subplot(grid[:, 2])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Per-metric min-max normalized improvement (0 = worst, 1 = best)", color=INK)
    cbar.ax.yaxis.set_tick_params(color=INK, labelcolor=INK)
    fig.suptitle("Channel effect matrices", y=0.98, fontsize=15, fontweight="bold")
    fig.subplots_adjust(left=0.08, right=0.94, top=0.89, bottom=0.11)
    return fig


def build_leave_one_out_figure(summaries: list[DatasetSummary]) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.5), gridspec_kw={"wspace": 0.16})
    x = np.arange(len(FOUR_GROUP_ORDER))
    width = 0.28
    styles = {
        "paired": {"offset": -width / 2, "facecolor": "white", "alpha": 1.0, "hatch": None},
        "unpaired": {"offset": width / 2, "facecolor": "white", "alpha": 1.0, "hatch": "//"},
    }
    x_tick_labels = ["Cell\ntypes", "Cell\nstate", "Vasc.", "Env."]
    panel_keys = [
        ("mean_diff", r"Mean normalized $|\Delta \mathrm{pixel}|$"),
        ("pct_pixels_above_10", r"Pixels with normalized $|\Delta \mathrm{pixel}| > 10$ (%)"),
    ]
    for ax, (metric_key, title) in zip(axes, panel_keys, strict=True):
        values_for_range: list[float] = []
        for group_index, group in enumerate(FOUR_GROUP_ORDER):
            for summary in summaries:
                style = styles[summary.slug]
                value = float(summary.loo_summary.get(group, {}).get(metric_key, 0.0))
                std_value = float(summary.loo_stats.get(group, {}).get(metric_key, (value, 0.0))[1])
                values_for_range.extend([value - std_value, value + std_value])
                ax.bar(
                    group_index + style["offset"],
                    value,
                    width=width,
                    color=style["facecolor"],
                    alpha=style["alpha"],
                    edgecolor=INK,
                    hatch=style["hatch"],
                    linewidth=1.0,
                    yerr=std_value,
                    capsize=3,
                    error_kw={"ecolor": INK, "elinewidth": 1.0, "capthick": 1.0},
                )
        ax.set_title(title, fontweight="bold", fontsize=11.0, pad=7)
        ax.set_xticks(x)
        ax.set_xticklabels(x_tick_labels)
        for tick in ax.get_xticklabels():
            tick.set_color(INK)
            tick.set_fontsize(9.5)
        lo, hi = _tight_range(values_for_range)
        ax.set_ylim(max(0.0, lo), hi)
        ax.grid(True, axis="y", color=SOFT_GRID, linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="y", labelsize=9.5)
        ax.set_axisbelow(True)
    handles = [
        Patch(facecolor="white", edgecolor=INK, label="Paired"),
        Patch(facecolor="white", edgecolor=INK, hatch="//", label="Unpaired"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.01), fontsize=10.0)
    fig.suptitle("Representative leave-one-out channel impact", y=1.03, fontsize=13.5, fontweight="bold")
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.19, top=0.77, wspace=0.16)
    return fig


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
    return f"<span class='metric-name'>{html.escape(humanize_token(metric_key))}</span>"


def metric_plain_text(metric_key: str) -> str:
    return html.escape(humanize_token(metric_key))


def render_best_worst_table(summary: DatasetSummary) -> str:
    rows: list[str] = []
    for metric_key in comparison_metric_keys(summary):
        record = summary.best_worst.get(metric_key)
        if not record:
            continue
        rows.append(
            "<tr>"
            f"<td>{metric_name_cell(metric_key)}</td>"
            f"<td class='condition-cell'>{render_condition_glyph(record['best_condition'])}</td>"
            f"<td>{float(record['best_value']):.3f}</td>"
            f"<td class='condition-cell'>{render_condition_glyph(record['worst_condition'])}</td>"
            f"<td>{float(record['worst_value']):.3f}</td>"
            "</tr>"
        )
    return (
        "<table class='metric-table'>"
        "<thead><tr>"
        "<th>Metric</th>"
        f"<th>Best condition <span class='condition-order'>{html.escape(condition_order_label())}</span></th>"
        "<th>Value</th>"
        f"<th>Worst condition <span class='condition-order'>{html.escape(condition_order_label())}</span></th>"
        "<th>Value</th>"
        "</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def render_comparison_table(summaries: list[DatasetSummary]) -> str:
    rows: list[str] = []
    for summary in summaries:
        records: list[str] = []
        for metric_key in comparison_metric_keys(summary):
            record = summary.best_worst.get(metric_key)
            if not record:
                continue
            records.append(
                "<tr>"
                f"<td class='metric-text'>{metric_plain_text(metric_key)}</td>"
                f"<td class='condition-cell'>{render_condition_glyph(record['best_condition'])}</td>"
                f"<td>{float(record['best_value']):.3f}</td>"
                f"<td class='condition-cell'>{render_condition_glyph(record['worst_condition'])}</td>"
                f"<td>{float(record['worst_value']):.3f}</td>"
                "</tr>"
            )
        if not records:
            continue
        records[0] = records[0].replace(
            "<tr>",
            f"<tr><td class='dataset-cell' rowspan='{len(records)}'>{html.escape(summary.title)}</td>",
            1,
        )
        rows.extend(records)
    return (
        "<table class='metric-table comparison-table'>"
        "<thead><tr>"
        "<th>Dataset</th>"
        "<th>Metric</th>"
        f"<th>Best condition <span class='condition-order'>{html.escape(condition_order_label())}</span></th>"
        "<th>Best</th>"
        f"<th>Worst condition <span class='condition-order'>{html.escape(condition_order_label())}</span></th>"
        "<th>Worst</th>"
        "</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def render_notes_list(notes: list[str]) -> str:
    items = "".join(f"<li>{render_takeaway_text(note.lstrip('- ').strip())}</li>" for note in notes)
    return f"<ul class='takeaways'>{items}</ul>"


def path_to_html_src(path: Path, output_dir: Path) -> str:
    rel_path = os.path.relpath(path.resolve(), start=output_dir.resolve())
    return html.escape(rel_path.replace(os.sep, "/"), quote=True)


def render_evidence_block(
    summary: DatasetSummary,
    output_dir: Path,
    *,
    self_contained: bool = False,
) -> str:
    cards: list[str] = []
    for label, path in (
        ("Representative ablation grid", summary.ablation_grid_path),
        ("Representative leave-one-out diff", summary.loo_diff_path),
    ):
        if path is None:
            continue
        src = image_file_to_data_uri(path) if self_contained else path_to_html_src(path, output_dir)
        cards.append(
            "<figure class='evidence-card'>"
            f"<img src='{src}' alt='{html.escape(label)}' loading='lazy' />"
            f"<figcaption>{html.escape(label)}"
            f"<span>{html.escape(str(path))}</span></figcaption>"
            "</figure>"
        )
    if not cards:
        return "<p class='muted'>Representative figures were not found for this dataset.</p>"
    return "<div class='evidence-grid'>" + "".join(cards) + "</div>"


def metric_direction_badge(metric_key: str) -> str:
    spec = METRIC_SPEC_BY_KEY[metric_key]
    arrow = "↑" if spec.higher_is_better else "↓"
    return f"<span class='metric-chip'>{html.escape(humanize_token(metric_key))} {arrow}</span>"


def render_takeaway_text(text: str) -> str:
    parts = text.split("`")
    rendered: list[str] = []
    for index, part in enumerate(parts):
        if index % 2 == 1:
            rendered.append(f"<strong>{html.escape(humanize_token(part))}</strong>")
        else:
            rendered.append(html.escape(part))
    return "".join(rendered)


def render_dataset_section(
    summary: DatasetSummary,
    output_dir: Path,
    *,
    self_contained: bool = False,
) -> str:
    chips = "".join(metric_direction_badge(metric_key) for metric_key in summary.metric_keys)
    representative = summary.representative_tile or "not found"
    return (
        "<section class='dataset-section'>"
        f"<div class='section-header'><h2>{html.escape(summary.title)}</h2>"
        f"<p>{summary.tile_count} tiles · representative tile {html.escape(representative)}</p></div>"
        f"<div class='chips'>{chips}</div>"
        "<div class='card'>"
        "<h3>Key takeaways</h3>"
        f"{render_notes_list(summary.key_takeaways[:6])}"
        "</div>"
        "<div class='card'>"
        "<h3>Representative evidence</h3>"
        f"{render_evidence_block(summary, output_dir, self_contained=self_contained)}"
        "</div>"
        "</section>"
    )


def render_report_html(
    title: str,
    summaries: list[DatasetSummary],
    output_path: Path,
    *,
    self_contained: bool = False,
) -> str:
    trend_uri = figure_to_data_uri(build_metric_trends_figure(summaries))
    heatmap_uri = figure_to_data_uri(build_channel_effect_heatmaps_figure(summaries))
    loo_uri = figure_to_data_uri(build_leave_one_out_figure(summaries))
    comparison_table = render_comparison_table(summaries)

    best_structure = []
    for summary in summaries:
        group_scores = {
            group: statistics.mean(
                [
                    summary.added_effects.get(group, {}).get(metric_key, 0.0)
                    for metric_key in ("aji", "pq")
                    if metric_key in summary.metric_keys
                ]
            )
            for group in FOUR_GROUP_ORDER
            if any(metric_key in summary.metric_keys for metric_key in ("aji", "pq"))
        }
        if group_scores:
            winner = max(group_scores, key=group_scores.get)
            best_structure.append(f"{summary.title}: {GROUP_LABELS[winner]}")

    overview_line = " | ".join(best_structure)
    dataset_sections = "".join(
        render_dataset_section(summary, output_path.parent, self_contained=self_contained)
        for summary in summaries
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      --bg: #f7f7f5;
      --paper: #ffffff;
      --ink: #1b1b1b;
      --muted: #686760;
      --line: #dddcd6;
      --blue: {OKABE_BLUE};
      --orange: {OKABE_ORANGE};
      --green: {OKABE_GREEN};
      --purple: {OKABE_PURPLE};
      --serif: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Georgia, serif;
      --sans: "Helvetica Neue", Helvetica, Arial, sans-serif;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: var(--sans);
      line-height: 1.45;
    }}
    .page {{
      max-width: 1420px;
      margin: 0 auto;
      padding: 28px 22px 44px;
    }}
    .hero {{
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 24px 26px 20px;
      box-shadow: 0 8px 24px rgba(28, 28, 28, 0.04);
      margin-bottom: 22px;
    }}
    .hero h1 {{
      margin: 0 0 10px;
      font-size: 2rem;
      line-height: 1.1;
      font-family: var(--serif);
      letter-spacing: -0.02em;
    }}
    .hero p {{
      margin: 0;
      color: var(--muted);
      max-width: 1000px;
    }}
    .overview-bar {{
      margin-top: 14px;
      padding: 10px 12px;
      border-radius: 8px;
      background: #f3f3ef;
      color: #3f3c35;
      font-size: 0.95rem;
    }}
    .report-grid {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 18px;
    }}
    .figure-card, .card, .dataset-section {{
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: 10px;
      box-shadow: 0 6px 20px rgba(28, 28, 28, 0.03);
    }}
    .figure-card {{
      padding: 18px 18px 14px;
    }}
    .figure-card h2, .dataset-section h2, .card h3 {{
      margin: 0 0 10px;
      font-family: var(--serif);
      letter-spacing: -0.01em;
    }}
    .figure-card p {{
      margin: 0 0 14px;
      color: var(--muted);
    }}
    .figure-card img {{
      width: 100%;
      border-radius: 6px;
      display: block;
      border: 1px solid #e6e4dd;
      background: white;
    }}
    .dataset-section {{
      padding: 18px;
    }}
    .section-header {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: baseline;
      flex-wrap: wrap;
      margin-bottom: 10px;
    }}
    .section-header p {{
      margin: 0;
      color: var(--muted);
    }}
    .chips {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 14px;
    }}
    .metric-chip {{
      display: inline-block;
      padding: 0;
      background: transparent;
      color: var(--ink);
      border: none;
      box-shadow: none;
      font-size: 0.9rem;
      font-weight: 600;
      letter-spacing: 0.01em;
    }}
    .card {{
      padding: 16px;
      margin-top: 14px;
    }}
    .takeaways {{
      margin: 0;
      padding-left: 18px;
    }}
    .takeaways li {{
      margin-bottom: 8px;
    }}
    .takeaways strong {{
      font-weight: 700;
    }}
    .metric-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.92rem;
    }}
    .metric-table th, .metric-table td {{
      text-align: left;
      padding: 8px 10px;
      border-bottom: 1px solid #ece5d9;
      vertical-align: top;
    }}
    .metric-table thead th {{
      font-size: 0.82rem;
      text-transform: uppercase;
      letter-spacing: 0.03em;
      color: var(--muted);
    }}
    .metric-name {{
      display: inline-block;
      padding: 4px 8px;
      border-radius: 999px;
      background: #f5f5f2;
      border: 1px solid #e2e1db;
      color: var(--ink);
      font-weight: 500;
    }}
    .comparison-table td:first-child {{
      font-weight: 600;
    }}
    .comparison-table {{
      font-family: "Times New Roman", Times, serif;
      font-size: 1rem;
      border-top: 2px solid #000;
      border-bottom: 2px solid #000;
    }}
    .comparison-table th, .comparison-table td {{
      border-bottom: 1px solid #c9c3b9;
      padding: 7px 10px;
    }}
    .comparison-table thead th {{
      color: #000;
      text-transform: none;
      letter-spacing: 0;
      font-size: 0.95rem;
      border-bottom: 1.5px solid #000;
    }}
    .comparison-table tbody tr:last-child td {{
      border-bottom: none;
    }}
    .comparison-table .dataset-cell {{
      font-weight: 700;
      vertical-align: middle;
      white-space: nowrap;
      padding-right: 14px;
    }}
    .comparison-table .metric-text {{
      font-weight: 700;
    }}
    .comparison-table .condition-cell {{
      white-space: nowrap;
    }}
    .condition-order {{
      display: block;
      margin-top: 2px;
      color: var(--muted);
      font-size: 0.74rem;
      font-weight: 400;
      line-height: 1.25;
    }}
    .condition-glyph {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      white-space: nowrap;
    }}
    .condition-dot {{
      width: 12px;
      height: 12px;
      border-radius: 999px;
      border: 1.6px solid #000;
      background: #fff;
      display: inline-block;
      flex: 0 0 auto;
    }}
    .condition-dot.is-active {{
      background: #000;
    }}
    .evidence-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
    }}
    .evidence-card {{
      margin: 0;
      border: 1px solid #e6e4dd;
      border-radius: 8px;
      overflow: hidden;
      background: #fff;
    }}
    .evidence-card img {{
      width: 100%;
      display: block;
      background: #fff;
    }}
    .evidence-card figcaption {{
      padding: 10px 12px 12px;
      font-size: 0.92rem;
    }}
    .evidence-card span {{
      display: block;
      color: var(--muted);
      font-size: 0.8rem;
      margin-top: 4px;
      word-break: break-all;
    }}
    .muted {{
      color: var(--muted);
    }}
    @media (max-width: 960px) {{
      .evidence-grid {{
        grid-template-columns: 1fr;
      }}
      .page {{
        padding: 18px 14px 28px;
      }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <h1>{html.escape(title)}</h1>
      <p>
        This report reframes the ablation summaries in a journal-style layout:
        metric trajectories, oriented channel-effect matrices with uncertainty, representative leave-one-out evidence,
        and a direct paired versus unpaired comparison table.
      </p>
      <div class="overview-bar">{html.escape(overview_line)}</div>
    </section>
    <div class="report-grid">
      <section class="figure-card">
        <h2>Metric Tradeoffs</h2>
        <img src="{trend_uri}" alt="Metric trends by active group count" />
      </section>
      <section class="figure-card">
        <h2>Paired vs Unpaired Best/Worst Conditions</h2>
        <p>A compact paper-style table compares each metric’s best and worst ablation condition across paired and unpaired settings. Filled circles indicate included groups in {html.escape(condition_order_label())} order.</p>
        {comparison_table}
      </section>
      <section class="figure-card">
        <h2>Channel Effect Sizes</h2>
        <p>Each cell shows per-metric min-max normalized improvement ± normalized SD, so both the numbers and colors run from 0 for worst to 1 for best across the combined plot.</p>
        <img src="{heatmap_uri}" alt="Channel effect heatmaps" />
      </section>
      <section class="figure-card">
        <h2>Leave-One-Out Impact</h2>
        <p>Grouped bars show mean ± SD across tiles, with paired bars on white fill and unpaired bars hatched for contrast.</p>
        <img src="{loo_uri}" alt="Leave one out summary bars" />
      </section>
      {dataset_sections}
    </div>
  </div>
</body>
</html>
"""


def plain_takeaway_text(text: str) -> str:
    parts = text.split("`")
    plain: list[str] = []
    for index, part in enumerate(parts):
        plain.append(humanize_token(part) if index % 2 == 1 else part)
    return "".join(plain)


def build_comparison_table_figure(summaries: list[DatasetSummary]) -> plt.Figure:
    rows: list[list[str]] = []
    for summary in summaries:
        for metric_key in comparison_metric_keys(summary):
            record = summary.best_worst.get(metric_key)
            if not record:
                continue
            rows.append(
                [
                    summary.title,
                    humanize_token(metric_key),
                    condition_indicator_text(record["best_condition"]),
                    f"{float(record['best_value']):.3f}",
                    condition_indicator_text(record["worst_condition"]),
                    f"{float(record['worst_value']):.3f}",
                ]
            )

    fig_height = max(3.8, 1.7 + 0.42 * max(len(rows), 1))
    fig, ax = plt.subplots(figsize=(15.5, fig_height))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=[
            "Dataset",
            "Metric",
            f"Best ({condition_order_label()})",
            "Best",
            f"Worst ({condition_order_label()})",
            "Worst",
        ],
        cellLoc="left",
        colLoc="left",
        loc="center",
        bbox=[0.0, 0.02, 1.0, 0.92],
        colWidths=[0.12, 0.12, 0.19, 0.08, 0.19, 0.08],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.0)
    table.scale(1.0, 1.35)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#C9C3B9")
        cell.set_linewidth(0.8)
        if row == 0:
            cell.set_facecolor("#F3F3EF")
            cell.set_text_props(weight="bold", color=INK)
        else:
            cell.set_facecolor("white")
            cell.set_text_props(color=INK, weight="normal")
            if col in {0, 1}:
                cell.set_text_props(weight="bold", color=INK)

    ax.set_title("Paired vs Unpaired Best/Worst Conditions", fontsize=15, fontweight="bold", pad=12)
    return fig


def draw_evidence_panel(ax: plt.Axes, label: str, path: Path | None) -> None:
    ax.set_title(label, fontsize=11.5, fontweight="bold", pad=10, color=INK)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#D9D5CA")
        spine.set_linewidth(1.0)
    if path is None or not path.is_file():
        ax.set_facecolor("#F6F4EF")
        ax.text(0.5, 0.5, "Figure not found", ha="center", va="center", color=OKABE_GRAY, fontsize=11.0)
        return
    ax.imshow(np.asarray(load_rgb_pil_local(path)))


def build_dataset_summary_figure(summary: DatasetSummary) -> plt.Figure:
    fig = plt.figure(figsize=(15.0, 7.2))
    grid = fig.add_gridspec(1, 3, width_ratios=[0.96, 1.0, 1.0], wspace=0.08)
    text_ax = fig.add_subplot(grid[0, 0])
    text_ax.axis("off")

    representative = summary.representative_tile or "not found"
    metrics_text = ", ".join(
        f"{humanize_token(metric_key)} {'↑' if METRIC_SPEC_BY_KEY[metric_key].higher_is_better else '↓'}"
        for metric_key in summary.metric_keys
    )
    metrics_text = textwrap.fill(metrics_text, width=34)
    takeaway_blocks = [
        textwrap.fill(
            plain_takeaway_text(note.lstrip("- ").strip()),
            width=40,
            initial_indent="- ",
            subsequent_indent="  ",
        )
        for note in summary.key_takeaways[:6]
    ]
    takeaways_text = "\n\n".join(takeaway_blocks) if takeaway_blocks else "Representative findings unavailable."

    text_ax.text(0.0, 0.99, summary.title, fontsize=19, fontweight="bold", color=INK, va="top")
    text_ax.text(
        0.0,
        0.91,
        f"{summary.tile_count} tiles\nRepresentative tile: {representative}",
        fontsize=11.0,
        color=OKABE_GRAY,
        va="top",
        linespacing=1.5,
    )
    text_ax.text(0.0, 0.77, "Metrics", fontsize=12.5, fontweight="bold", color=INK, va="top")
    text_ax.text(0.0, 0.72, metrics_text, fontsize=10.8, color=INK, va="top", linespacing=1.4)
    text_ax.text(0.0, 0.58, "Key takeaways", fontsize=12.5, fontweight="bold", color=INK, va="top")
    text_ax.text(0.0, 0.53, takeaways_text, fontsize=10.7, color=INK, va="top", linespacing=1.45)

    draw_evidence_panel(fig.add_subplot(grid[0, 1]), "Representative ablation grid", summary.ablation_grid_path)
    draw_evidence_panel(fig.add_subplot(grid[0, 2]), "Representative leave-one-out diff", summary.loo_diff_path)
    fig.subplots_adjust(left=0.04, right=0.98, top=0.96, bottom=0.06)
    return fig


def export_report_png_pages(summaries: list[DatasetSummary], png_dir: Path) -> list[Path]:
    page_builders: list[tuple[str, plt.Figure]] = [
        ("01_metric_tradeoffs.png", build_metric_trends_figure(summaries)),
        ("02_paired_vs_unpaired.png", build_comparison_table_figure(summaries)),
        ("03_channel_effect_sizes.png", build_channel_effect_heatmaps_figure(summaries)),
        ("04_leave_one_out_impact.png", build_leave_one_out_figure(summaries)),
    ]
    for index, summary in enumerate(summaries, start=5):
        page_builders.append((f"{index:02d}_{summary.slug}_summary.png", build_dataset_summary_figure(summary)))

    output_paths: list[Path] = []
    for filename, fig in page_builders:
        output_path = png_dir / filename
        save_figure_png(fig, output_path)
        output_paths.append(output_path)
    return output_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a paired vs unpaired ablation HTML report.")
    parser.add_argument(
        "--paired-metrics-root",
        type=Path,
        default=ROOT / "inference_output" / "paired_ablation" / "ablation_results",
        help="Directory containing paired per-tile metrics.json files.",
    )
    parser.add_argument(
        "--paired-dataset-root",
        type=Path,
        default=ROOT / "inference_output" / "paired_ablation",
        help="Paired dataset root used for representative figure lookup.",
    )
    parser.add_argument(
        "--paired-reference-root",
        type=Path,
        default=ROOT / "data" / "orion-crc33",
        help="Reference H&E root used to compute missing paired HED metrics.",
    )
    parser.add_argument(
        "--unpaired-metrics-root",
        type=Path,
        default=ROOT / "inference_output" / "unpaired_ablation" / "ablation_results",
        help="Directory containing unpaired per-tile metrics.json files.",
    )
    parser.add_argument(
        "--unpaired-dataset-root",
        type=Path,
        default=ROOT / "inference_output" / "unpaired_ablation",
        help="Unpaired dataset root used for representative figure lookup.",
    )
    parser.add_argument(
        "--unpaired-reference-root",
        type=Path,
        default=ROOT / "inference_output" / "unpaired_ablation" / "data" / "orion-crc33-unpaired",
        help="Reference H&E root used to compute missing unpaired HED metrics when absent.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "docs" / "ablation_scientific_report.html",
        help="HTML output path.",
    )
    parser.add_argument(
        "--min-gt-cells",
        type=int,
        default=0,
        help="Skip tiles with fewer than this many GT cell instances in each dataset (default: 0 = no filter).",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Channel Ablation Scientific Report",
        help="Title shown at the top of the HTML report.",
    )
    parser.add_argument(
        "--self-contained",
        action="store_true",
        help="Embed representative evidence images so the HTML can be opened as a standalone file.",
    )
    parser.add_argument(
        "--png-dir",
        type=Path,
        default=ROOT / "figures" / "pngs",
        help="Directory where standalone PNG pages will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = [
        load_dataset_summary(
            slug="paired",
            title="Paired",
            metrics_root=args.paired_metrics_root.resolve(),
            dataset_root=args.paired_dataset_root.resolve(),
            reference_root=args.paired_reference_root.resolve(),
            min_gt_cells=args.min_gt_cells,
        ),
        load_dataset_summary(
            slug="unpaired",
            title="Unpaired",
            metrics_root=args.unpaired_metrics_root.resolve(),
            dataset_root=args.unpaired_dataset_root.resolve(),
            reference_root=args.unpaired_reference_root.resolve(),
            min_gt_cells=args.min_gt_cells,
        ),
    ]
    report = render_report_html(
        args.title,
        summaries,
        args.output.resolve(),
        self_contained=args.self_contained,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    png_paths = export_report_png_pages(summaries, args.png_dir.resolve())
    print(f"Rendered ablation HTML report -> {args.output}")
    print(f"Saved {len(png_paths)} PNG pages -> {args.png_dir}")


if __name__ == "__main__":
    main()
