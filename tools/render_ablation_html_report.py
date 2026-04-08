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

from tools.stage3.ablation_vis_utils import FOUR_GROUP_ORDER, condition_metric_key
from tools.summarize_ablation_report import (
    DEFAULT_METRIC_ORDER,
    METRIC_SPEC_BY_KEY,
    best_worst_notes,
    cardinality_notes,
    effect_notes,
    fid_answer_notes,
    load_condition_means,
    load_leave_one_out_summary,
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
    "fid": OKABE_RED,
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
METRIC_LABELS = {
    "cosine": "Cosine",
    "lpips": "LPIPS",
    "aji": "AJI",
    "pq": "PQ",
    "fid": "FID",
    "style_hed": "HED",
}


@dataclass(frozen=True)
class DatasetSummary:
    slug: str
    title: str
    metrics_root: Path
    dataset_root: Path
    tile_count: int
    metric_keys: list[str]
    by_cardinality: dict[int, dict[str, float]]
    best_worst: dict[str, dict[str, str | float]]
    added_effects: dict[str, dict[str, float]]
    presence_absence: dict[str, dict[str, float]]
    added_effect_stats: dict[str, dict[str, tuple[float, float]]]
    presence_absence_stats: dict[str, dict[str, tuple[float, float]]]
    loo_summary: dict[str, dict[str, float]]
    representative_tile: str | None
    key_takeaways: list[str]
    ablation_grid_path: Path | None
    loo_diff_path: Path | None


def _mean_std(values: list[float]) -> tuple[float, float] | None:
    if not values:
        return None
    return float(statistics.mean(values)), float(statistics.pstdev(values)) if len(values) > 1 else 0.0


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


def load_dataset_summary(
    *,
    slug: str,
    title: str,
    metrics_root: Path,
    dataset_root: Path,
    reference_root: Path | None = None,
) -> DatasetSummary:
    condition_means, tile_count, metric_keys = load_condition_means(metrics_root)
    condition_means, metric_keys = add_missing_style_hed_means(
        condition_means,
        metrics_root,
        metric_keys,
        reference_root=resolve_reference_root(
            dataset_root=dataset_root,
            slug=slug,
            reference_root=reference_root,
        ),
    )
    by_cardinality = summarize_by_cardinality(condition_means, metric_keys)
    best_worst = summarize_best_worst(condition_means, metric_keys)
    added_effects = summarize_added_group_effects(condition_means, metric_keys)
    presence_absence = summarize_presence_absence(condition_means, metric_keys)
    added_effect_stats = summarize_added_group_effect_stats(condition_means, metric_keys)
    presence_absence_stats = summarize_presence_absence_stats(condition_means, metric_keys)
    loo_summary, representative_tile = load_leave_one_out_summary(resolve_loo_root(metrics_root, dataset_root))
    ablation_grid_path, loo_diff_path = resolve_representative_paths(metrics_root, dataset_root, representative_tile)

    notes: list[str] = []
    notes.extend(cardinality_notes(by_cardinality, metric_keys))
    notes.extend(best_worst_notes(best_worst))
    notes.extend(effect_notes(added_effects, presence_absence, metric_keys))
    notes.extend(loo_notes(loo_summary))
    notes.extend(fid_answer_notes(by_cardinality, metric_keys))

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
        by_cardinality=by_cardinality,
        best_worst=best_worst,
        added_effects=added_effects,
        presence_absence=presence_absence,
        added_effect_stats=added_effect_stats,
        presence_absence_stats=presence_absence_stats,
        loo_summary=loo_summary,
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


def image_file_to_data_uri(path: Path) -> str:
    mime = "image/png"
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        mime = "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def metric_union(summaries: list[DatasetSummary]) -> list[str]:
    present = {metric for summary in summaries for metric in summary.metric_keys}
    return [metric for metric in DEFAULT_METRIC_ORDER if metric in present]


def render_metric_trends_figure(summaries: list[DatasetSummary]) -> str:
    metrics = metric_union(summaries)
    fig, axes = plt.subplots(2, 3, figsize=(13.6, 7.2))
    axes = axes.ravel()
    dataset_styles = {
        "paired": {"linestyle": "-", "markerfacecolor": None},
        "unpaired": {"linestyle": ":", "markerfacecolor": "white"},
    }

    for ax, metric_key in zip(axes, metrics, strict=False):
        all_values: list[float] = []
        color = METRIC_COLORS.get(metric_key, OKABE_GRAY)
        for summary in summaries:
            x_values = sorted(summary.by_cardinality)
            y_values = [summary.by_cardinality[group_count].get(metric_key) for group_count in x_values]
            valid = [(x, float(y)) for x, y in zip(x_values, y_values, strict=True) if y is not None]
            if not valid:
                continue
            xs, ys = zip(*valid)
            all_values.extend(ys)
            style = dataset_styles[summary.slug]
            ax.plot(
                xs,
                ys,
                color=color,
                linestyle=style["linestyle"],
                marker="o",
                markerfacecolor=color if style["markerfacecolor"] is None else style["markerfacecolor"],
                markeredgecolor=color,
                markeredgewidth=1.1,
                linewidth=2.1,
                markersize=5.4,
            )

        if not all_values:
            ax.axis("off")
            continue

        lo, hi = _tight_range(all_values)
        spec = METRIC_SPEC_BY_KEY[metric_key]
        arrow = "↑" if spec.higher_is_better else "↓"
        ax.set_title(f"{METRIC_LABELS.get(metric_key, metric_key)} {arrow}", color=INK, fontweight="bold")
        ax.set_xlim(0.85, 4.15)
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xticklabels(["1g", "2g", "3g", "4g"])
        ax.set_ylim(lo, hi)
        ax.grid(True, axis="y", color=SOFT_GRID, linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_color(INK)
        ax.spines["left"].set_color(INK)
        ax.tick_params(axis="x", colors=INK)
        ax.tick_params(axis="y", colors=INK)
        ax.set_axisbelow(True)

    for ax in axes[len(metrics):]:
        ax.axis("off")

    handles = [
        Line2D([0], [0], color="#444444", linestyle="-", marker="o", markersize=5.4, linewidth=2.0, label="Paired"),
        Line2D(
            [0],
            [0],
            color="#444444",
            linestyle=":",
            marker="o",
            markerfacecolor="white",
            markeredgecolor="#444444",
            markersize=5.4,
            linewidth=2.0,
            label="Unpaired",
        ),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle("Metric trends by active channel-group count", y=1.03, fontsize=15, fontweight="bold")
    fig.tight_layout()
    return figure_to_data_uri(fig)


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


def render_channel_effect_heatmaps(summaries: list[DatasetSummary]) -> str:
    fig = plt.figure(figsize=(14.8, 5.8 * len(summaries)))
    grid = fig.add_gridspec(len(summaries), 3, width_ratios=[1, 1, 0.055], wspace=0.34, hspace=0.52)
    all_values: list[float] = []
    im = None
    for summary in summaries:
        metric_keys = heatmap_metric_keys(summary)
        for source in (summary.added_effect_stats, summary.presence_absence_stats):
            matrix, _ = _heatmap_matrix(summary, source, metric_keys)
            all_values.extend([float(value) for value in matrix[np.isfinite(matrix)]])
    vmax = max((abs(value) for value in all_values), default=1.0)
    vmax = max(vmax, 1e-6)

    for row, summary in enumerate(summaries):
        panels = [
            ("Average gain from adding a group", summary.added_effect_stats),
            ("Present vs absent delta", summary.presence_absence_stats),
        ]
        for col, (panel_title, source) in enumerate(panels):
            ax = fig.add_subplot(grid[row, col])
            metric_keys = heatmap_metric_keys(summary)
            matrix, stds = _heatmap_matrix(summary, source, metric_keys)
            masked = np.ma.masked_invalid(matrix)
            im = ax.imshow(masked, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")
            ax.set_title(f"{summary.title}: {panel_title}", fontsize=12.5, fontweight="bold", color=INK)
            ax.set_yticks(range(len(FOUR_GROUP_ORDER)))
            if col == 0:
                ax.set_yticklabels([GROUP_LABELS[group] for group in FOUR_GROUP_ORDER], fontsize=11.5, color=INK)
            else:
                ax.set_yticklabels([])
            ax.set_xticks(range(len(metric_keys)))
            ax.set_xticklabels(
                [METRIC_LABELS.get(metric, metric) for metric in metric_keys],
                rotation=0,
                ha="center",
                fontsize=11.5,
                color=INK,
                fontweight="bold",
            )
            for r in range(matrix.shape[0]):
                for c in range(matrix.shape[1]):
                    value = matrix[r, c]
                    if not np.isfinite(value):
                        continue
                    std_value = stds[r, c]
                    ax.text(
                        c,
                        r,
                        f"{value:+.3f}\n±{std_value:.3f}",
                        ha="center",
                        va="center",
                        fontsize=10.0,
                        fontweight="semibold",
                        color="white" if abs(value) > vmax * 0.45 else "#111111",
                    )
            ax.set_xticks(np.arange(-0.5, len(metric_keys), 1), minor=True)
            ax.set_yticks(np.arange(-0.5, len(FOUR_GROUP_ORDER), 1), minor=True)
            ax.grid(which="minor", color=INK, linewidth=0.9)
            ax.tick_params(which="minor", bottom=False, left=False)
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.tick_params(axis="x", length=0, pad=10)
            ax.tick_params(axis="y", length=0)

    cax = fig.add_subplot(grid[:, 2])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Oriented delta (positive = helps)", color=INK)
    cbar.ax.yaxis.set_tick_params(color=INK, labelcolor=INK)
    fig.suptitle("Channel effect matrices", y=0.98, fontsize=15, fontweight="bold")
    fig.subplots_adjust(left=0.10, right=0.93, top=0.92, bottom=0.12)
    return figure_to_data_uri(fig)


def render_leave_one_out_figure(summaries: list[DatasetSummary]) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.4))
    x = np.arange(len(FOUR_GROUP_ORDER))
    width = 0.34
    styles = {
        "paired": {"offset": -width / 2, "alpha": 0.92, "hatch": None, "edgecolor": "#3B3B3B"},
        "unpaired": {"offset": width / 2, "alpha": 0.55, "hatch": "//", "edgecolor": "#3B3B3B"},
    }
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
                values_for_range.append(value)
                ax.bar(
                    group_index + style["offset"],
                    value,
                    width=width,
                    color=GROUP_COLORS[group],
                    alpha=style["alpha"],
                    edgecolor=style["edgecolor"],
                    hatch=style["hatch"],
                    linewidth=1.0,
                )
        ax.set_title(title, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([GROUP_LABELS[group] for group in FOUR_GROUP_ORDER], rotation=15, ha="right")
        for tick, group in zip(ax.get_xticklabels(), FOUR_GROUP_ORDER, strict=True):
            tick.set_color(GROUP_COLORS[group])
            tick.set_fontweight("bold")
        lo, hi = _tight_range(values_for_range)
        ax.set_ylim(max(0.0, lo), hi)
        ax.grid(True, axis="y", color=SOFT_GRID, linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_axisbelow(True)
    handles = [
        Patch(facecolor="#8E8E8E", edgecolor="#3B3B3B", alpha=0.92, label="Paired"),
        Patch(facecolor="#8E8E8E", edgecolor="#3B3B3B", alpha=0.55, hatch="//", label="Unpaired"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.05))
    fig.suptitle("Representative leave-one-out channel impact", y=1.08, fontsize=15, fontweight="bold")
    fig.tight_layout()
    return figure_to_data_uri(fig)


def format_condition(value: object) -> str:
    return str(value).replace("+", " + ")


def metric_name_cell(metric_key: str) -> str:
    color = METRIC_COLORS.get(metric_key, OKABE_GRAY)
    return (
        f"<span class='metric-name' style='--metric-color:{color}'>"
        f"{html.escape(METRIC_LABELS.get(metric_key, metric_key))}"
        "</span>"
    )


def metric_plain_text(metric_key: str) -> str:
    return html.escape(METRIC_LABELS.get(metric_key, metric_key))


def render_best_worst_table(summary: DatasetSummary) -> str:
    rows: list[str] = []
    for metric_key in summary.metric_keys:
        record = summary.best_worst.get(metric_key)
        if not record:
            continue
        rows.append(
            "<tr>"
            f"<td>{metric_name_cell(metric_key)}</td>"
            f"<td>{html.escape(format_condition(record['best_condition']))}</td>"
            f"<td>{float(record['best_value']):.3f}</td>"
            f"<td>{html.escape(format_condition(record['worst_condition']))}</td>"
            f"<td>{float(record['worst_value']):.3f}</td>"
            "</tr>"
        )
    return (
        "<table class='metric-table'>"
        "<thead><tr><th>Metric</th><th>Best condition</th><th>Value</th><th>Worst condition</th><th>Value</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def render_comparison_table(summaries: list[DatasetSummary]) -> str:
    rows: list[str] = []
    for summary in summaries:
        records: list[str] = []
        for metric_key in summary.metric_keys:
            record = summary.best_worst.get(metric_key)
            if not record:
                continue
            records.append(
                "<tr>"
                f"<td class='metric-text'>{metric_plain_text(metric_key)}</td>"
                f"<td>{html.escape(format_condition(record['best_condition']))}</td>"
                f"<td>{float(record['best_value']):.3f}</td>"
                f"<td>{html.escape(format_condition(record['worst_condition']))}</td>"
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
        "<thead><tr><th>Dataset</th><th>Metric</th><th>Best condition</th><th>Best</th><th>Worst condition</th><th>Worst</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def render_notes_list(notes: list[str]) -> str:
    items = "".join(f"<li>{html.escape(note.lstrip('- ').strip())}</li>" for note in notes)
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
    color = METRIC_COLORS.get(metric_key, OKABE_GRAY)
    return (
        f"<span class='metric-chip' style='--metric-color:{color}'>"
        f"{html.escape(METRIC_LABELS.get(metric_key, metric_key))} {arrow}"
        "</span>"
    )


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
    trend_uri = render_metric_trends_figure(summaries)
    heatmap_uri = render_channel_effect_heatmaps(summaries)
    loo_uri = render_leave_one_out_figure(summaries)
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
      color: var(--metric-color);
      border: none;
      box-shadow: none;
      font-size: 0.9rem;
      font-weight: 700;
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
      box-shadow: inset 3px 0 0 var(--metric-color);
      color: var(--metric-color);
      font-weight: 600;
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
      font-weight: 400;
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
        <p>Metric identity is encoded by color across the report; paired and unpaired are separated by solid versus dotted styling.</p>
        <img src="{trend_uri}" alt="Metric trends by active group count" />
      </section>
      <section class="figure-card">
        <h2>Channel Effect Sizes</h2>
        <p>Each cell shows mean oriented effect ± SD. Positive values help the metric after orientation by metric direction.</p>
        <img src="{heatmap_uri}" alt="Channel effect heatmaps" />
      </section>
      <section class="figure-card">
        <h2>Leave-One-Out Impact</h2>
        <p>Grouped bars summarize representative leave-one-out changes, with channel identity carried by color and paired versus unpaired by fill pattern.</p>
        <img src="{loo_uri}" alt="Leave one out summary bars" />
      </section>
      <section class="figure-card">
        <h2>Paired vs Unpaired Best/Worst Conditions</h2>
        <p>A compact paper-style table compares each metric’s best and worst ablation condition across paired and unpaired settings.</p>
        {comparison_table}
      </section>
      {dataset_sections}
    </div>
  </div>
</body>
</html>
"""


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
        ),
        load_dataset_summary(
            slug="unpaired",
            title="Unpaired",
            metrics_root=args.unpaired_metrics_root.resolve(),
            dataset_root=args.unpaired_dataset_root.resolve(),
            reference_root=args.unpaired_reference_root.resolve(),
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
    print(f"Rendered ablation HTML report -> {args.output}")


if __name__ == "__main__":
    main()
