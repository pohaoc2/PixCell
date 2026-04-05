"""Figure rendering helpers for stage 3 channel sweep experiments."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from tools.color_constants import CELL_STATE_COLORS, CELL_TYPE_COLORS
from tools.stage3.ablation_vis_utils import draw_image_border, load_exp_channel_plane
from tools.stage3.channel_sweep_cache import (
    load_channel_sweep_manifest,
    load_exp1_microenv_cache,
    load_relabeling_cache,
    load_rgb_png,
    source_labels_from_results,
    target_labels_from_results,
)


SWEEP_SCALES: list[float] = [0.0, 0.25, 0.5, 0.75, 1.0]
_HOT4_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "hot4",
    ["#000000", "#ff4400", "#ffff00", "#ffffff"],
)
_BASELINE_BORDER = "#9B59B6"


def _rgb(rgba: tuple[int, int, int, int]) -> tuple[int, int, int]:
    return rgba[:3]


CELL_TYPE_THUMB_SPECS = {
    "cancer": ("cell_type_cancer", _rgb(CELL_TYPE_COLORS["cancer"])),
    "immune": ("cell_type_immune", _rgb(CELL_TYPE_COLORS["immune"])),
    "healthy": ("cell_type_healthy", _rgb(CELL_TYPE_COLORS["healthy"])),
}

CELL_STATE_THUMB_SPECS = {
    "prolif": ("cell_state_prolif", _rgb(CELL_STATE_COLORS["proliferative"])),
    "nonprolif": ("cell_state_nonprolif", _rgb(CELL_STATE_COLORS["nonprolif"])),
    "dead": ("cell_state_dead", _rgb(CELL_STATE_COLORS["dead"])),
}


def _render_label_badge(label: str, rgb: tuple[int, int, int], *, resolution: int = 96) -> np.ndarray:
    badge = np.zeros((resolution, resolution, 3), dtype=np.uint8)
    badge[...] = np.array(rgb, dtype=np.uint8)
    yy, xx = np.ogrid[:resolution, :resolution]
    dist = np.sqrt((xx - resolution / 2) ** 2 + (yy - resolution / 2) ** 2)
    vignette = np.clip(1.0 - dist / (resolution * 0.9), 0.35, 1.0)
    return np.clip(badge.astype(np.float32) * vignette[..., None], 0, 255).astype(np.uint8)


def _render_group_thumbnail(
    *,
    exp_channels_dir: Path,
    tile_id: str,
    label: str,
    thumb_specs: dict[str, tuple[str, tuple[int, int, int]]],
    resolution: int = 96,
) -> np.ndarray:
    channel_name, rgb = thumb_specs[label]
    try:
        plane = load_exp_channel_plane(
            exp_channels_dir,
            channel_name,
            tile_id,
            resolution=resolution,
        )
    except (FileNotFoundError, OSError, ValueError, KeyError):
        return _render_label_badge(label, rgb, resolution=resolution)

    out = np.zeros((resolution, resolution, 3), dtype=np.float32)
    color = np.array(rgb, dtype=np.float32) / 255.0
    out += plane[..., None] * color
    return (np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)


def _render_relabel_input_thumbnail(
    *,
    exp_channels_dir: Path,
    tile_id: str,
    source_label: str,
    target_label: str,
    thumb_specs: dict[str, tuple[str, tuple[int, int, int]]],
    resolution: int = 96,
) -> np.ndarray:
    src_channel, _ = thumb_specs[source_label]
    _, target_rgb = thumb_specs[target_label]
    try:
        plane = load_exp_channel_plane(
            exp_channels_dir,
            src_channel,
            tile_id,
            resolution=resolution,
        )
    except (FileNotFoundError, OSError, ValueError, KeyError):
        return _render_label_badge(target_label, target_rgb, resolution=resolution)

    out = np.zeros((resolution, resolution, 3), dtype=np.float32)
    out += plane[..., None] * (np.array(target_rgb, dtype=np.float32) / 255.0)
    return (np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)


def render_exp1_figure(
    images_grid: dict[tuple[float, float], np.ndarray],
    *,
    tile_id: str,
    tile_class_label: str,
    out_path: Path,
) -> None:
    """Render the 5x5 microenvironment sweep as separate generation and diff chunks."""
    n = len(SWEEP_SCALES)
    baseline = images_grid[(1.0, 1.0)].astype(np.float32)
    fig = plt.figure(figsize=(n * 2.25 + 1.6, n * 4.1 + 1.6))
    fig.suptitle(f"Microenv sweep - {tile_id} ({tile_class_label})", fontsize=11)
    gs = fig.add_gridspec(
        (n * 2) + 2,
        n + 1,
        hspace=0.12,
        wspace=0.04,
        height_ratios=[1] * n + [0.22] + [1] * n + [0.16],
        width_ratios=[0.2] + [1] * n,
    )

    ax_label_gen = fig.add_subplot(gs[:n, 0])
    ax_label_gen.axis("off")
    for i, scale in enumerate(SWEEP_SCALES):
        ax_label_gen.text(
            0.95,
            1 - (i + 0.5) / n,
            f"O2={scale:.2f}",
            ha="right",
            va="center",
            fontsize=7,
            transform=ax_label_gen.transAxes,
        )
    ax_label_gen.text(
        0.12,
        1.02,
        "Generated H&E",
        fontsize=9,
        fontweight="bold",
        transform=ax_label_gen.transAxes,
    )

    sep_ax = fig.add_subplot(gs[n, :])
    sep_ax.axis("off")

    ax_label_diff = fig.add_subplot(gs[n + 1 : (n * 2) + 1, 0])
    ax_label_diff.axis("off")
    for i, scale in enumerate(SWEEP_SCALES):
        ax_label_diff.text(
            0.95,
            1 - (i + 0.5) / n,
            f"O2={scale:.2f}",
            ha="right",
            va="center",
            fontsize=7,
            transform=ax_label_diff.transAxes,
        )
    ax_label_diff.text(
        0.12,
        1.02,
        "Pixel diff vs baseline",
        fontsize=9,
        fontweight="bold",
        transform=ax_label_diff.transAxes,
    )

    diff_max = 0.0
    diff_maps: dict[tuple[float, float], np.ndarray] = {}
    for o2_scale in SWEEP_SCALES:
        for glucose_scale in SWEEP_SCALES:
            img = images_grid[(o2_scale, glucose_scale)]
            diff = np.abs(img.astype(np.float32) - baseline).mean(axis=2)
            if o2_scale == 1.0 and glucose_scale == 1.0:
                diff = np.zeros_like(diff)
            diff_maps[(o2_scale, glucose_scale)] = diff
            diff_max = max(diff_max, float(diff.max()))
    diff_vmax = max(diff_max, 1.0)

    for i, o2_scale in enumerate(SWEEP_SCALES):
        for j, glucose_scale in enumerate(SWEEP_SCALES):
            ax = fig.add_subplot(gs[i, j + 1])
            img = images_grid[(o2_scale, glucose_scale)]
            ax.imshow(img)
            ax.axis("off")
            if i == 0:
                ax.set_title(f"Glucose={glucose_scale:.2f}", fontsize=7, pad=4)

            if o2_scale == 1.0 and glucose_scale == 1.0:
                draw_image_border(ax, _BASELINE_BORDER, linewidth=3.0)

            ax_diff = fig.add_subplot(gs[n + 1 + i, j + 1])
            diff = diff_maps[(o2_scale, glucose_scale)]
            im = ax_diff.imshow(diff, cmap=_HOT4_CMAP, vmin=0, vmax=diff_vmax)
            ax_diff.axis("off")
            if i == 0:
                ax_diff.set_title(f"Glucose={glucose_scale:.2f}", fontsize=7, pad=4)
            if o2_scale == 1.0 and glucose_scale == 1.0:
                draw_image_border(ax_diff, _BASELINE_BORDER, linewidth=3.0)

    cbar_ax = fig.add_subplot(gs[(n * 2) + 1, 1:3])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
    cbar.ax.tick_params(labelsize=7, pad=1)
    cbar.set_label("Mean absolute pixel diff", fontsize=8, labelpad=2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.07, right=0.995, top=0.95, bottom=0.05)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def render_relabeling_figure(
    results: dict[str, dict[str, np.ndarray]],
    *,
    tiles: dict[str, str],
    thumb_specs: dict[str, tuple[str, tuple[int, int, int]]],
    exp_channels_dir: Path | None = None,
    baseline_group_thumbs: dict[str, np.ndarray] | None = None,
    input_thumbs: dict[str, dict[str, np.ndarray]] | None = None,
    labels: list[str] | None = None,
    source_labels: list[str] | None = None,
    target_labels: list[str] | None = None,
    exp_title: str,
    out_path: Path,
) -> None:
    """Render relabeling results as generated-H&E and diff chunks."""
    del tiles, thumb_specs, exp_channels_dir, baseline_group_thumbs, input_thumbs
    source_labels = (
        list(source_labels)
        if source_labels is not None
        else list(labels)
        if labels is not None
        else source_labels_from_results(results)
    )
    target_labels = (
        list(target_labels)
        if target_labels is not None
        else list(labels)
        if labels is not None
        else target_labels_from_results(results)
    )
    if not source_labels or not target_labels:
        raise ValueError("results must not be empty")

    n_source = len(source_labels)
    n_target = len(target_labels)
    n_rows = n_source * 2
    fig, axes = plt.subplots(n_rows, n_target, figsize=(3.1 * n_target, 2.55 * n_rows))
    if n_rows == 1 and n_target == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[None, :]
    elif n_target == 1:
        axes = axes[:, None]
    fig.suptitle(exp_title, fontsize=11)

    diff_maps: dict[tuple[str, str], np.ndarray] = {}
    diff_max = 0.0
    for src_label in source_labels:
        baseline = results[src_label][src_label].astype(np.float32)
        for tgt_label in target_labels:
            diff = np.abs(results[src_label][tgt_label].astype(np.float32) - baseline).mean(axis=2)
            if tgt_label == src_label:
                diff = np.zeros_like(diff)
            diff_maps[(src_label, tgt_label)] = diff
            diff_max = max(diff_max, float(diff.max()))
    diff_vmax = max(diff_max, 1.0)

    for i, src_label in enumerate(source_labels):
        row_he = i
        row_diff = i + n_source
        for j, tgt_label in enumerate(target_labels):
            ax_he = axes[row_he, j]
            img = results[src_label][tgt_label]
            ax_he.imshow(img)
            title = f"{src_label}->{tgt_label}"
            if tgt_label == src_label:
                title = f"all {src_label} (baseline)"
            elif src_label.startswith("non") and tgt_label.startswith("non"):
                title = f"all {tgt_label}"
            else:
                title = f"replace all {src_label} with {tgt_label}"
            ax_he.set_title(title, fontsize=8, pad=4)
            ax_he.axis("off")
            if tgt_label == src_label:
                draw_image_border(ax_he, _BASELINE_BORDER)

            ax_diff = axes[row_diff, j]
            diff = diff_maps[(src_label, tgt_label)]
            ax_diff.imshow(diff, cmap=_HOT4_CMAP, vmin=0, vmax=diff_vmax)
            ax_diff.axis("off")

        axes[row_he, 0].set_ylabel(
            f"Generated H&E\nall {src_label}",
            fontsize=8,
            rotation=90,
            labelpad=18,
        )
        axes[row_diff, 0].set_ylabel(
            f"Pixel diff\nvs all {src_label}",
            fontsize=8,
            rotation=90,
            labelpad=18,
        )

    fig.subplots_adjust(left=0.11, right=0.99, top=0.95, bottom=0.04, wspace=0.05, hspace=0.16)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def build_baseline_group_thumbs(
    *,
    tiles: dict[str, str],
    exp_channels_dir: Path,
    thumb_specs: dict[str, tuple[str, tuple[int, int, int]]],
    resolution: int = 72,
) -> dict[str, np.ndarray]:
    return {
        label: _render_group_thumbnail(
            exp_channels_dir=exp_channels_dir,
            tile_id=tile_id,
            label=label,
            thumb_specs=thumb_specs,
            resolution=resolution,
        )
        for label, tile_id in tiles.items()
    }


def build_relabel_input_thumbs(
    *,
    tiles: dict[str, str],
    labels: list[str],
    exp_channels_dir: Path,
    thumb_specs: dict[str, tuple[str, tuple[int, int, int]]],
    resolution: int = 96,
) -> dict[str, dict[str, np.ndarray]]:
    out: dict[str, dict[str, np.ndarray]] = {}
    for src_label, tile_id in tiles.items():
        out[src_label] = {
            tgt_label: _render_relabel_input_thumbnail(
                exp_channels_dir=exp_channels_dir,
                tile_id=tile_id,
                source_label=src_label,
                target_label=tgt_label,
                thumb_specs=thumb_specs,
                resolution=resolution,
            )
            for tgt_label in labels
        }
    return out


def render_figures_from_cache(
    *,
    cache_dir: Path,
    out_dir: Path,
    experiments: list[str] | None = None,
) -> None:
    manifest = load_channel_sweep_manifest(cache_dir)
    selected = set(experiments or ["1", "2", "3"])
    exp_data = manifest.get("experiments", {})

    if "1" in selected and "exp1_microenv" in exp_data:
        for record in exp_data["exp1_microenv"].get("tiles", []):
            grid = load_exp1_microenv_cache(cache_dir, record)
            render_exp1_figure(
                grid,
                tile_id=str(record["tile_id"]),
                tile_class_label=str(record["tile_class_label"]),
                out_path=Path(out_dir) / "exp1_microenv" / f"{record['tile_class_label']}_{record['tile_id']}.png",
            )

    if "2" in selected and "exp2_cell_type_relabeling" in exp_data:
        loaded = load_relabeling_cache(cache_dir, exp_data["exp2_cell_type_relabeling"])
        render_relabeling_figure(
            loaded["results"],
            tiles=loaded["tiles"],
            thumb_specs=CELL_TYPE_THUMB_SPECS,
            baseline_group_thumbs=loaded["baseline_group_thumbs"],
            input_thumbs=loaded.get("input_thumbs"),
            labels=loaded["labels"],
            source_labels=loaded.get("source_labels"),
            target_labels=loaded.get("target_labels"),
            exp_title="Exp 2: Cell type relabeling (given cell states + microenv)",
            out_path=Path(out_dir) / "exp2_cell_type_relabeling.png",
        )

    if "3" in selected and "exp3_cell_state_relabeling" in exp_data:
        loaded = load_relabeling_cache(cache_dir, exp_data["exp3_cell_state_relabeling"])
        render_relabeling_figure(
            loaded["results"],
            tiles=loaded["tiles"],
            thumb_specs=CELL_STATE_THUMB_SPECS,
            baseline_group_thumbs=loaded["baseline_group_thumbs"],
            input_thumbs=loaded.get("input_thumbs"),
            labels=loaded["labels"],
            source_labels=loaded.get("source_labels"),
            target_labels=loaded.get("target_labels"),
            exp_title="Exp 3: Cell state relabeling (given cell types + microenv)",
            out_path=Path(out_dir) / "exp3_cell_state_relabeling.png",
        )

    rendered_known = {
        "exp1_microenv",
        "exp2_cell_type_relabeling",
        "exp3_cell_state_relabeling",
    }
    for exp_name, record in exp_data.items():
        if exp_name in rendered_known:
            continue
        entries = record.get("entries")
        if not isinstance(entries, list) or not entries:
            continue
        images: list[tuple[str, np.ndarray]] = []
        for entry in entries:
            rel = entry.get("image_path")
            if not rel:
                continue
            img_path = Path(cache_dir) / rel
            if not img_path.is_file():
                continue
            label = f"{entry.get('source_label', '')} -> {entry.get('target_label', '')}".strip()
            images.append((label, load_rgb_png(img_path)))
        if not images:
            continue
        n = len(images)
        fig, axes = plt.subplots(1, n, figsize=(2.6 * n, 2.8))
        if n == 1:
            axes = np.array([axes])
        for ax, (label, image) in zip(axes, images, strict=True):
            ax.imshow(image)
            ax.set_title(label, fontsize=8)
            ax.axis("off")
        fig.suptitle(exp_name, fontsize=10)
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(out_dir) / f"{exp_name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def render_channel_sweep_figures(
    *,
    cache_dir: Path,
    out_dir: Path,
    output_dir: Path | None = None,
    cache_path: Path | None = None,
    cache_manifest: Path | None = None,
    experiments: list[str] | None = None,
) -> None:
    del cache_manifest
    render_figures_from_cache(
        cache_dir=Path(cache_path or cache_dir),
        out_dir=Path(output_dir or out_dir),
        experiments=experiments,
    )


__all__ = [
    "CELL_STATE_THUMB_SPECS",
    "CELL_TYPE_THUMB_SPECS",
    "SWEEP_SCALES",
    "build_baseline_group_thumbs",
    "build_relabel_input_thumbs",
    "render_channel_sweep_figures",
    "render_exp1_figure",
    "render_figures_from_cache",
    "render_relabeling_figure",
]
