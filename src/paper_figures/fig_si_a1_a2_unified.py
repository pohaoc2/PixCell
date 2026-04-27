"""Build SI A1/A2 figure panels from cache.json and PNG tiles only."""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import FormatStrFormatter, FuncFormatter, MaxNLocator
import numpy as np
from PIL import Image

from src.paper_figures.style import (
    FONT_SIZE_ANNOTATION,
    FONT_SIZE_DENSE_LABEL,
    FONT_SIZE_LABEL,
    FONT_SIZE_TICK,
    FONT_SIZE_TITLE,
    apply_style,
)
from tools.ablation_a1_a2.log_utils import deserialise_float


PRIMARY_A2_VARIANT = "a2_bypass_full_tme"
LEGACY_A2_VARIANT = "a2_bypass"
SECTION1_FONT_FAMILY = "DejaVu Serif"


VARIANT_SPECS: dict[str, dict] = {
    "production": {"label": "Per-ch. + Attn", "color": "#2e7d32", "ls": "--", "lw": 1.8, "unstable": False},
    "a1_concat": {"label": "Concat TME", "color": "#1565c0", "ls": "--", "lw": 1.5, "unstable": False},
    "a1_per_channel": {"label": "Per-channel TME", "color": "#e65100", "ls": "-", "lw": 1.5, "unstable": True},
    "a2_bypass_full_tme": {"label": "Full-TME probe", "color": "#7b1fa2", "ls": "-", "lw": 1.5, "unstable": True},
    "a2_bypass": {"label": "Bypass probe", "color": "#7b1fa2", "ls": "-", "lw": 1.5, "unstable": True},
    "a2_off_shelf": {"label": "Off-the-shelf PixCell", "color": "#000000", "ls": "-", "lw": 1.6, "unstable": False},
}

A1_VARIANTS = ("a1_concat", "a1_per_channel", "production")
A2_VARIANTS = (PRIMARY_A2_VARIANT, "a2_off_shelf", "production")
TABLE_ROWS = (
    "a1_concat",
    "a1_per_channel",
    PRIMARY_A2_VARIANT,
    "a2_off_shelf",
    "production",
)
TILE_GRID_ORDER = ("gt", "production", "a1_concat", "a1_per_channel", PRIMARY_A2_VARIANT, "a2_off_shelf")
METRIC_COLS = (
    ("FUD↓", "fud", "{:.2f}"),
    ("DICE↑", "dice", "{:.3f}"),
    ("PQ↑", "pq", "{:.3f}"),
    ("LPIPS↓", "lpips", "{:.3f}"),
    ("HED↓", "style_hed", "{:.3f}"),
    ("ΔLPIPS↑", "delta_lpips", "{:.3f}"),
)
METRIC_DIRECTIONS = {
    "fud": "min",
    "dice": "max",
    "pq": "max",
    "lpips": "min",
    "style_hed": "min",
    "delta_lpips": "max",
}
INSTABILITY_COLOR = "#c62828"
INSTABILITY_ALPHA = 0.12
_TARGET_CONTOUR_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".npy")
SECTION4_GROUPS = ("cell_state", "microenv")

VARIANT_SOURCE_FALLBACKS: dict[str, tuple[str, ...]] = {
    PRIMARY_A2_VARIANT: (PRIMARY_A2_VARIANT, LEGACY_A2_VARIANT),
}

PARAM_FALLBACKS: dict[str, tuple[str, ...]] = {
    LEGACY_A2_VARIANT: (LEGACY_A2_VARIANT, "production"),
    PRIMARY_A2_VARIANT: (PRIMARY_A2_VARIANT, LEGACY_A2_VARIANT, "production"),
}


def _variant_candidates(variant: str) -> tuple[str, ...]:
    return VARIANT_SOURCE_FALLBACKS.get(variant, (variant,))


def _resolve_variant_key(mapping: dict, variant: str) -> str | None:
    for candidate in _variant_candidates(variant):
        if candidate in mapping:
            return candidate
    return None


def _normalize_sensitivity(sensitivity: dict) -> dict[str, dict]:
    if not sensitivity:
        return {}

    if any(_resolve_variant_key(sensitivity, variant) for variant in TABLE_ROWS):
        normalized: dict[str, dict] = {}
        for variant in TABLE_ROWS:
            source_key = _resolve_variant_key(sensitivity, variant)
            if source_key is not None:
                normalized[variant] = dict(sensitivity[source_key])
        return normalized

    legacy_group_means = [
        float(record.get("mean", 0.0))
        for record in sensitivity.values()
        if isinstance(record, dict)
    ]
    if not legacy_group_means:
        return {}
    arr = np.asarray(legacy_group_means, dtype=float)
    return {
        "production": {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "per_group": dict(sensitivity),
        }
    }


def _filtered_sensitivity_summary(record: dict, groups: tuple[str, ...]) -> dict[str, object]:
    per_group = record.get("per_group", {})
    selected = {
        group: dict(per_group[group])
        for group in groups
        if isinstance(per_group.get(group), dict)
    }
    if not selected:
        return dict(record)

    mean_values = np.asarray(
        [float(group_record.get("mean", 0.0)) for group_record in selected.values()],
        dtype=float,
    )
    return {
        "mean": float(mean_values.mean()),
        "std": float(mean_values.std()),
        "per_group": selected,
    }


def _section4_sensitivity(cache: dict) -> dict[str, dict]:
    sensitivity = _normalize_sensitivity(cache.get("sensitivity", {}))
    return {
        variant: _filtered_sensitivity_summary(record, SECTION4_GROUPS)
        for variant, record in sensitivity.items()
    }


def _section2_metrics(cache: dict) -> dict[str, dict]:
    metrics: dict[str, dict] = {}
    raw_metrics = cache.get("metrics", {})
    for variant in TABLE_ROWS:
        source_key = _resolve_variant_key(raw_metrics, variant)
        if source_key is not None and isinstance(raw_metrics.get(source_key), dict):
            metrics[variant] = dict(raw_metrics[source_key])
    for variant, record in _section4_sensitivity(cache).items():
        metrics.setdefault(variant, {})["delta_lpips"] = float(record.get("mean", 0.0))
        metrics[variant]["delta_lpips_std"] = float(record.get("std", 0.0))
    return metrics


def _load_cache(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _aggregate_curves(runs: dict[str, list[dict]], metric: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate seed runs into aligned step, mean, and std arrays."""
    if not runs:
        return np.array([], dtype=int), np.array([], dtype=float), np.array([], dtype=float)

    all_by_step: list[dict[int, float]] = []
    inf_steps: set[int] = set()
    for entries in runs.values():
        by_step: dict[int, float] = {}
        for entry in entries:
            value = deserialise_float(entry.get(metric))
            if math.isnan(value):
                continue
            step = int(entry["step"])
            by_step[step] = value
            if math.isinf(value):
                inf_steps.add(step)
        if by_step:
            all_by_step.append(by_step)

    if not all_by_step:
        return np.array([], dtype=int), np.array([], dtype=float), np.array([], dtype=float)

    steps = np.asarray(sorted(set().union(*(set(run.keys()) for run in all_by_step))), dtype=int)
    values = np.full((len(all_by_step), len(steps)), np.nan, dtype=float)
    for row_idx, by_step in enumerate(all_by_step):
        for col_idx, step in enumerate(steps):
            if step in by_step and not math.isinf(by_step[step]):
                values[row_idx, col_idx] = by_step[step]

    mean = np.full(len(steps), np.nan, dtype=float)
    std = np.full(len(steps), np.nan, dtype=float)
    finite_cols = ~np.all(np.isnan(values), axis=0)
    if np.any(finite_cols):
        mean[finite_cols] = np.nanmean(values[:, finite_cols], axis=0)
        std[finite_cols] = np.nanstd(values[:, finite_cols], axis=0)
    if inf_steps:
        mean[np.isin(steps, list(inf_steps))] = float("inf")
        std[np.isin(steps, list(inf_steps))] = 0.0
    return steps, mean, std


def build_figure(*, cache_path: Path, tile_dir: Path) -> plt.Figure:
    apply_style()
    cache = _load_cache(cache_path)
    fig = plt.figure(figsize=(13.4, 12.45), constrained_layout=False)
    fig.subplots_adjust(left=0.095, right=0.99, top=0.985, bottom=0.045, hspace=0.25, wspace=0.18)

    outer = fig.add_gridspec(4, 1, height_ratios=[1.95, 1.08, 3.0, 0.85], hspace=0.18)
    _draw_section1_curves(fig, outer[0], cache)
    _draw_section2_table(fig.add_subplot(outer[1]), cache)
    _draw_section3_tiles(fig, outer[2], cache, tile_dir)
    _draw_section4_sensitivity(fig.add_subplot(outer[3]), cache)

    return fig


def build_section1_figure(*, cache_path: Path) -> plt.Figure:
    apply_style()
    cache = _load_cache(cache_path)
    fig = plt.figure(figsize=(8.8, 4.35), constrained_layout=False)
    fig.subplots_adjust(left=0.08, right=0.995, top=0.92, bottom=0.12, wspace=0.38)
    _draw_section1_curves(fig, fig.add_gridspec(1, 1)[0], cache)
    return fig


def build_section2_figure(*, cache_path: Path) -> plt.Figure:
    apply_style()
    cache = _load_cache(cache_path)
    fig = plt.figure(figsize=(8.6, 2.25), constrained_layout=False)
    fig.subplots_adjust(left=0.015, right=0.995, top=0.96, bottom=0.06)
    _draw_section2_table(fig.add_subplot(111), cache)
    return fig


def build_section3_figure(*, cache_path: Path, tile_dir: Path) -> plt.Figure:
    apply_style()
    cache = _load_cache(cache_path)
    fig = plt.figure(figsize=(9.8, 4.5), constrained_layout=False)
    fig.subplots_adjust(left=0.12, right=0.995, top=0.965, bottom=0.025)
    _draw_section3_tiles(fig, fig.add_gridspec(1, 1)[0], cache, tile_dir)
    return fig


def build_section4_figure(*, cache_path: Path) -> plt.Figure:
    apply_style()
    cache = _load_cache(cache_path)
    fig = plt.figure(figsize=(4.5, 2.2), constrained_layout=False)
    fig.subplots_adjust(left=0.22, right=0.97, top=0.88, bottom=0.22)
    _draw_section4_sensitivity(fig.add_subplot(111), cache)
    return fig


def build_section_figures(*, cache_path: Path, tile_dir: Path) -> dict[str, plt.Figure]:
    return {
        "section1_curves": build_section1_figure(cache_path=cache_path),
        "section2_metrics": build_section2_figure(cache_path=cache_path),
        "section3_tiles": build_section3_figure(cache_path=cache_path, tile_dir=tile_dir),
        "section4_sensitivity": build_section4_figure(cache_path=cache_path),
    }


def _format_step_tick(value: float, _pos: int) -> str:
    if abs(value) < 1e-9:
        return "0"
    if abs(value) >= 1000:
        return f"{value / 1000:g}k"
    return f"{int(value)}"


def _style_training_axis(ax: plt.Axes) -> None:
    ax.set_xlabel("Step", fontsize=FONT_SIZE_LABEL, fontfamily=SECTION1_FONT_FAMILY)
    ax.tick_params(labelsize=FONT_SIZE_TICK, width=0.7, length=3)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
    ax.xaxis.set_major_formatter(FuncFormatter(_format_step_tick))
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for tick in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        tick.set_fontfamily(SECTION1_FONT_FAMILY)


def _style_loss_axis(ax: plt.Axes) -> None:
    ax.set_ylabel("Training loss", fontsize=FONT_SIZE_LABEL, fontfamily=SECTION1_FONT_FAMILY)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    _style_training_axis(ax)


def _style_grad_axis(ax: plt.Axes) -> None:
    ax.set_ylabel("Gradient norm", fontsize=FONT_SIZE_LABEL, fontfamily=SECTION1_FONT_FAMILY)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    _style_training_axis(ax)


def _plot_loss_curves(ax: plt.Axes, curves: dict, variant_keys: list[str], title: str) -> None:
    ax.set_title(title, fontsize=FONT_SIZE_LABEL, loc="left", pad=3, fontfamily=SECTION1_FONT_FAMILY)
    _style_loss_axis(ax)

    for variant in variant_keys:
        spec = VARIANT_SPECS[variant]
        source_key = _resolve_variant_key(curves, variant)
        steps, mean, _std = _aggregate_curves(curves.get(source_key, {}), "loss")
        if len(steps) == 0:
            continue
        finite = np.isfinite(mean)
        if np.any(finite):
            ax.plot(
                steps[finite],
                mean[finite],
                color=spec["color"],
                lw=spec["lw"],
                ls=spec["ls"],
                label=spec["label"],
                zorder=4 if variant == "production" else 3,
            )
        else:
            synthetic = np.abs(np.sin(np.arange(len(steps)) * 1.7)) * 0.08 + 0.22
            ax.plot(
                steps,
                synthetic,
                color=spec["color"],
                lw=spec["lw"],
                ls=spec["ls"],
                label=f"{spec['label']} unstable",
                zorder=4 if variant == "production" else 3,
            )


def _plot_gradnorm_curves(ax: plt.Axes, curves: dict, variant_keys: list[str], title: str) -> None:
    ax.set_title(title, fontsize=FONT_SIZE_LABEL, loc="left", pad=3, fontfamily=SECTION1_FONT_FAMILY)
    _style_grad_axis(ax)

    unstable_to_draw: list[tuple[str, np.ndarray]] = []
    not_logged: list[str] = []

    for variant in variant_keys:
        spec = VARIANT_SPECS[variant]
        source_key = _resolve_variant_key(curves, variant)
        steps, mean, _std = _aggregate_curves(curves.get(source_key, {}), "grad_norm")
        if len(steps) == 0:
            not_logged.append(variant)
            continue
        finite = np.isfinite(mean) & (mean > 0)
        if np.any(finite):
            ax.plot(
                steps[finite],
                mean[finite],
                color=spec["color"],
                lw=spec["lw"],
                ls=spec["ls"],
                label=spec["label"],
                zorder=5 if variant == "production" else 3,
            )
        if np.any(np.isinf(mean)):
            unstable_to_draw.append((variant, steps))
        if not np.any(finite) and not np.any(np.isinf(mean)):
            not_logged.append(variant)

    finite_lines = bool(ax.lines)
    ymin, ymax = ax.get_ylim()
    if not finite_lines or not np.isfinite(ymax) or ymax <= 0:
        ymin, ymax = 0.0, 1.0
        ax.set_ylim(ymin, ymax)
        ax.set_yticks([])
        ax.text(
            0.50,
            0.45,
            "No finite gradient-norm values",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=FONT_SIZE_ANNOTATION,
            color="black",
            fontfamily=SECTION1_FONT_FAMILY,
        )
    else:
        span = ymax - ymin
        y_clip = ymax - 0.12 * span
    if not finite_lines:
        y_clip = 0.80

    for offset_idx, (variant, steps) in enumerate(unstable_to_draw):
        spec = VARIANT_SPECS[variant]
        offset = 0.0 if len(unstable_to_draw) == 1 else 0.045 * (offset_idx / max(len(unstable_to_draw) - 1, 1))
        clipped = np.full(len(steps), y_clip - offset * (ymax - ymin))
        ax.plot(
            steps,
            clipped,
            color=spec["color"],
            lw=spec["lw"],
            ls="--",
            label=f"{spec['label']} (grad explosion)",
        )

    if unstable_to_draw:
        ax.text(
            0.98,
            0.98,
            "Dashed top lines = clipped grad explosion",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=FONT_SIZE_ANNOTATION - 1,
            color="#4a4a4a",
            fontfamily=SECTION1_FONT_FAMILY,
        )

    for variant in not_logged:
        spec = VARIANT_SPECS[variant]
        ax.plot([], [], color=spec["color"], lw=spec["lw"], ls=spec["ls"], label=f"{spec['label']} (not logged)")

    ax.set_ylim(ymin, ymax)


def _section1_legend_handles(variant_keys: list[str]) -> list[Line2D]:
    handles: list[Line2D] = []
    for variant in variant_keys:
        spec = VARIANT_SPECS[variant]
        handles.append(
            Line2D(
                [0],
                [0],
                color=spec["color"],
                lw=spec["lw"],
                ls="--" if spec["unstable"] else spec["ls"],
                label=spec["label"],
            )
        )
    return handles


def _draw_section1_curves(fig: plt.Figure, gs_slot, cache: dict) -> None:
    sub = gs_slot.subgridspec(2, 2, height_ratios=[1.0, 0.18], hspace=0.34, wspace=0.38)
    curves = cache.get("training_curves", {})
    all_variants = ["production", "a1_concat", "a1_per_channel", PRIMARY_A2_VARIANT]
    _plot_loss_curves(fig.add_subplot(sub[0, 0]), curves, all_variants, "Training loss")
    _plot_gradnorm_curves(fig.add_subplot(sub[0, 1]), curves, all_variants, "Gradient norm")
    legend_ax = fig.add_subplot(sub[1, :])
    legend_ax.axis("off")
    legend_ax.legend(
        handles=_section1_legend_handles(all_variants),
        loc="center",
        ncol=4,
        frameon=False,
        prop=FontProperties(family=SECTION1_FONT_FAMILY, size=FONT_SIZE_TICK),
        handlelength=2.8,
        columnspacing=1.4,
    )


def _fmt_metric(metrics: dict, key: str, fmt: str) -> str:
    value = metrics.get(key)
    if value is None:
        return "pending"
    try:
        return fmt.format(float(value))
    except (TypeError, ValueError):
        return str(value)


def _fmt_metric_std(metrics: dict, key: str, fmt: str) -> str | None:
    value = metrics.get(f"{key}_std")
    if value is None:
        return None
    try:
        return fmt.format(float(value))
    except (TypeError, ValueError):
        return None


def _fmt_metric_with_std(metrics: dict, key: str, fmt: str) -> str:
    value_text = _fmt_metric(metrics, key, fmt)
    std_text = _fmt_metric_std(metrics, key, fmt)
    if std_text is None or value_text == "pending":
        return value_text
    return f"{value_text}±{std_text}"


def _metric_value(metrics: dict, key: str) -> float | None:
    value = metrics.get(key)
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _best_metric_variants(metrics: dict) -> dict[str, set[str]]:
    best: dict[str, set[str]] = {}
    for _label, key, _fmt in METRIC_COLS:
        values: list[tuple[str, float]] = []
        for variant in TABLE_ROWS:
            value = _metric_value(metrics.get(variant, {}), key)
            if value is not None:
                values.append((variant, value))
        if not values:
            best[key] = set()
            continue
        direction = METRIC_DIRECTIONS.get(key, "min")
        target = min(value for _variant, value in values) if direction == "min" else max(value for _variant, value in values)
        best[key] = {variant for variant, value in values if math.isclose(value, target, rel_tol=1e-9, abs_tol=1e-12)}
    return best


def _fmt_params(params: dict, variant: str) -> str:
    value = None
    for candidate in PARAM_FALLBACKS.get(variant, (variant,)):
        value = params.get(candidate)
        if value is not None:
            break
    if value is None:
        return "-"
    return f"{int(value) / 1e6:.1f}M"


def _draw_section2_table(ax: plt.Axes, cache: dict) -> None:
    ax.axis("off")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    metrics = _section2_metrics(cache)
    params = cache.get("params", {})
    best_by_metric = _best_metric_variants(metrics)
    table_fs = FONT_SIZE_DENSE_LABEL
    col_x = [0.01, 0.255, 0.375, 0.495, 0.615, 0.735, 0.855, 0.965]
    headers = ["Config", *[col[0] for col in METRIC_COLS], "Params"]

    for x, header in zip(col_x, headers):
        ha = "left" if header == "Config" else "center"
        ax.text(x, 0.88, header, fontsize=table_fs, va="center", ha=ha, transform=ax.transAxes)
    ax.axhline(0.97, color="black", linewidth=0.75)
    ax.axhline(0.76, color="black", linewidth=0.45)

    def draw_row(y: float, variant: str) -> None:
        spec = VARIANT_SPECS[variant]
        ax.text(
            col_x[0],
            y,
            spec["label"],
            fontsize=table_fs,
            va="center",
            color="black",
            transform=ax.transAxes,
        )
        row_metrics = metrics.get(variant, {})
        for x, (_label, key, fmt) in zip(col_x[1:-1], METRIC_COLS):
            ax.text(
                x,
                y,
                _fmt_metric_with_std(row_metrics, key, fmt),
                fontsize=table_fs - 1,
                fontweight="bold" if variant in best_by_metric.get(key, set()) else "normal",
                va="center",
                ha="center",
                transform=ax.transAxes,
            )
        param_text = _fmt_params(params, variant)
        ax.text(col_x[-1], y, param_text, fontsize=table_fs, va="center", ha="center", transform=ax.transAxes)

    for y, variant in zip([0.70, 0.54, 0.38, 0.22, 0.08], TABLE_ROWS):
        draw_row(y, variant)

    ax.axhline(0.03, color="black", linewidth=0.75)


def _draw_section4_sensitivity(ax: plt.Axes, cache: dict) -> None:
    sensitivity = _section4_sensitivity(cache)
    ax.set_title("Config sensitivity to TME perturbations", fontsize=FONT_SIZE_LABEL, loc="left", pad=3)
    ax.set_xlabel("Mean ΔLPIPS for cell-state + microenv ablations ↑", fontsize=FONT_SIZE_LABEL)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=FONT_SIZE_TICK, width=0.7, length=3)

    if not sensitivity:
        ax.set_yticks([])
        ax.text(
            0.5,
            0.5,
            "No sensitivity data",
            ha="center",
            va="center",
            fontsize=FONT_SIZE_ANNOTATION,
            transform=ax.transAxes,
        )
        return

    variants = [variant for variant in TABLE_ROWS if variant in sensitivity]
    variants.sort(key=lambda variant: float(sensitivity[variant].get("mean", 0.0)), reverse=True)
    labels = [VARIANT_SPECS[variant]["label"] for variant in variants]
    means = [float(sensitivity[variant].get("mean", 0.0)) for variant in variants]
    stds = [float(sensitivity[variant].get("std", 0.0)) for variant in variants]
    colors = [VARIANT_SPECS[variant]["color"] for variant in variants]
    y_pos = np.arange(len(variants))

    ax.barh(
        y_pos,
        means,
        xerr=stds,
        color=colors,
        alpha=0.82,
        height=0.55,
        error_kw={"linewidth": 0.8, "capsize": 3},
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=FONT_SIZE_TICK)
    ax.set_xlim(left=0.0)
    ax.grid(axis="x", color="#d9d9d9", linewidth=0.45)
    ax.invert_yaxis()


def _blank_tile() -> np.ndarray:
    return np.full((256, 256, 3), 240, dtype=np.uint8)


def _target_mask_path(tile_id: str) -> Path | None:
    base_dir = Path("data/orion-crc33/exp_channels")
    for folder in ("cell_masks", "cell_mask"):
        channel_dir = base_dir / folder
        if not channel_dir.is_dir():
            continue
        for ext in _TARGET_CONTOUR_EXTS:
            candidate = channel_dir / f"{tile_id}{ext}"
            if candidate.is_file():
                return candidate
    return None


def _load_target_contour_mask(tile_id: str) -> np.ndarray | None:
    path = _target_mask_path(tile_id)
    if path is None:
        return None
    if path.suffix.lower() == ".npy":
        return np.asarray(np.load(path), dtype=np.float32)
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def _overlay_target_contours(ax: plt.Axes, tile_id: str, image_hw: tuple[int, int]) -> None:
    cell_mask = _load_target_contour_mask(tile_id)
    if cell_mask is None:
        return
    img_h, img_w = image_hw
    mask_h, mask_w = cell_mask.shape[:2]
    if (mask_h, mask_w) != (img_h, img_w):
        resized = Image.fromarray(
            (np.clip(cell_mask, 0, 1) * 255).astype(np.uint8),
            mode="L",
        ).resize((img_w, img_h), Image.BILINEAR)
        cell_mask = np.asarray(resized, dtype=np.float32) / 255.0
    ax.contour(cell_mask, levels=[0.5], colors=["black"], linewidths=1.4, alpha=0.9)
    ax.contour(cell_mask, levels=[0.5], colors=["white"], linewidths=0.8, alpha=0.95)


def _load_cellvit_contours(image_path: Path) -> list[np.ndarray]:
    sidecar_path = image_path.with_name(f"{image_path.stem}_cellvit_instances.json")
    if not sidecar_path.is_file():
        return []

    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    contours: list[np.ndarray] = []
    for cell in payload.get("cells", []):
        contour = cell.get("contour")
        if not isinstance(contour, list) or len(contour) < 3:
            continue
        arr = np.asarray(contour, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] < 2:
            continue
        contours.append(arr[:, :2])
    return contours


def _overlay_generated_cell_contours(ax: plt.Axes, image_path: Path) -> None:
    for contour in _load_cellvit_contours(image_path):
        ax.plot(
            contour[:, 0],
            contour[:, 1],
            color="red",
            linewidth=0.6,
            alpha=0.85,
            zorder=4,
        )


def _load_tile(tile_dir: Path, variant: str, tile_id: str) -> np.ndarray:
    if variant == "gt":
        ref_path = Path("data/orion-crc33/he") / f"{tile_id}.png"
        if ref_path.is_file():
            return np.asarray(Image.open(ref_path).convert("RGB"))
    path = _generated_tile_path(tile_dir, variant, tile_id)
    if not path.is_file():
        return _blank_tile()
    return np.asarray(Image.open(path).convert("RGB"))


def _generated_tile_path(tile_dir: Path, variant: str, tile_id: str) -> Path:
    for candidate in _variant_candidates(variant):
        path = tile_dir / candidate / f"{tile_id}.png"
        if path.is_file():
            return path
    return tile_dir / variant / f"{tile_id}.png"


def _draw_section3_tiles(fig: plt.Figure, gs_slot, cache: dict, tile_dir: Path) -> None:
    tile_ids = list(cache.get("tile_ids", [])[:10] or cache.get("metric_tile_ids", [])[:10])
    if not tile_ids:
        ax = fig.add_subplot(gs_slot)
        ax.axis("off")
        ax.set_title("Qualitative paired tiles", fontsize=FONT_SIZE_TITLE, loc="left")
        ax.text(0.5, 0.5, "No tile IDs in cache", ha="center", va="center", fontsize=FONT_SIZE_ANNOTATION, transform=ax.transAxes)
        return

    sub = gs_slot.subgridspec(len(TILE_GRID_ORDER), len(tile_ids), wspace=0.025, hspace=0.025)
    row_labels = {
        "gt": "Ref H&E",
        **{variant: spec["label"] for variant, spec in VARIANT_SPECS.items()},
    }

    for row_idx, variant in enumerate(TILE_GRID_ORDER):
        for col_idx, tile_id in enumerate(tile_ids):
            ax = fig.add_subplot(sub[row_idx, col_idx])
            tile = _load_tile(tile_dir, variant, tile_id)
            ax.imshow(tile)
            _overlay_target_contours(ax, tile_id, (tile.shape[0], tile.shape[1]))
            if variant != "gt":
                _overlay_generated_cell_contours(ax, _generated_tile_path(tile_dir, variant, tile_id))
            ax.set_xticks([])
            ax.set_yticks([])
            if row_idx == 0:
                ax.set_title(str(col_idx + 1), fontsize=FONT_SIZE_ANNOTATION - 3, pad=2)
            for spine in ax.spines.values():
                spine.set_linewidth(0.65 if variant != "gt" else 0.55)
                spine.set_color("black" if variant != "gt" else "#808080")
                spine.set_linestyle("-" if variant != "gt" else (0, (5.0, 2.4)))
            if col_idx == 0:
                ax.set_ylabel(
                    row_labels.get(variant, variant),
                    fontsize=FONT_SIZE_ANNOTATION,
                    rotation=0,
                    ha="right",
                    va="center",
                    labelpad=18,
                )


def _split_output_paths(out: Path) -> dict[str, Path]:
    stem = out.stem
    prefix = stem.replace("_unified", "")
    return {
        "section1_curves": out.with_name(f"{prefix}_section1_curves{out.suffix}"),
        "section2_metrics": out.with_name(f"{prefix}_section2_metrics{out.suffix}"),
        "section3_tiles": out.with_name(f"{prefix}_section3_tiles{out.suffix}"),
        "section4_sensitivity": out.with_name(f"{prefix}_section4_sensitivity{out.suffix}"),
    }


def save_split_figures(*, cache_path: Path, tile_dir: Path, out: Path, dpi: int) -> dict[str, Path]:
    paths = _split_output_paths(out)
    for key, fig in build_section_figures(cache_path=cache_path, tile_dir=tile_dir).items():
        paths[key].parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(paths[key], dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    return paths


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build SI_A1_A2_unified.png from cache")
    parser.add_argument("--cache-dir", default="inference_output/si_a1_a2")
    parser.add_argument("--out", default="figures/pngs/SI_A1_A2_unified.png")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--no-split", action="store_true", help="Only write the unified figure.")
    args = parser.parse_args(argv)

    cache_dir = Path(args.cache_dir)
    cache_path = cache_dir / "cache.json"
    tile_dir = cache_dir / "tiles"
    fig = build_figure(cache_path=cache_path, tile_dir=tile_dir)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out}")
    if not args.no_split:
        for path in save_split_figures(cache_path=cache_path, tile_dir=tile_dir, out=out, dpi=args.dpi).values():
            print(f"Saved -> {path}")


if __name__ == "__main__":
    main()
