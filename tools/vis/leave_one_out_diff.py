"""Leave-one-out group pixel diff from cached ablation PNGs.

Usage:
    python tools/vis/leave_one_out_diff.py \
        --cache-dir inference_output/cache/512_9728 \
        --orion-root data/orion-crc33 \
        --out inference_output/cache/512_9728/leave_one_out_diff.png

    python tools/vis/leave_one_out_diff.py \
        --cache-root inference_output/cache \
        --orion-root data/orion-crc33

    python tools/vis/leave_one_out_diff.py \
        --cache-dir inference_output/cache/512_9728 \
        --figure ssim \
        --out inference_output/cache/512_9728/leave_one_out_ssim.png
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from tools.stage3.ablation_cache import load_manifest
from tools.stage3.ablation_vis_utils import (
    FOUR_GROUP_ORDER,
    default_orion_he_png_path,
    draw_image_border,
)
from tools.stage3.style_mapping import load_style_mapping

COLOR_REF = "#000000"
COLOR_BASELINE = "#9B59B6"

# Per-group highlight colours for the SSIM figure
_LOO_SSIM_HIGHLIGHT: dict[str, str] = {
    "cell_state": "#ff6644",
    "microenv": "#ddaa00",
}
_LOO_SSIM_INSET_TEAL = "#00ccaa"
_LOO_SSIM_NEUTRAL = "#555555"
_FIGURE_MODES = ("diff", "ssim", "both")


def _load_rgb_float32(path: Path) -> np.ndarray:
    """Load a PNG as float32 H×W×3 in [0, 255]."""
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)


def _resize_rgb_uint8(image: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    """Resize an RGB uint8 image to (height, width) when needed."""
    target_h, target_w = target_hw
    if image.shape[:2] == (target_h, target_w):
        return image
    return np.asarray(
        Image.fromarray(image.astype(np.uint8), mode="RGB").resize((target_w, target_h), Image.BILINEAR),
        dtype=np.uint8,
    )


def _section_by_subset_size(sections: list[dict], subset_size: int) -> dict:
    for section in sections:
        if section.get("subset_size") == subset_size:
            return section
    raise KeyError(f"No manifest section found for subset_size={subset_size}")


def find_loo_entry(sections: list[dict], omit_group: str) -> dict:
    """Return the triples manifest entry whose active_groups excludes `omit_group`."""
    if omit_group not in FOUR_GROUP_ORDER:
        raise KeyError(f"Unknown group: {omit_group}")

    triples = _section_by_subset_size(sections, 3)
    for entry in triples.get("entries", []):
        if omit_group not in entry.get("active_groups", []):
            return entry
    raise KeyError(f"No triples entry found omitting {omit_group!r}")


def _find_all_entry(sections: list[dict], n_groups: int) -> dict:
    all_section = _section_by_subset_size(sections, n_groups)
    entries = all_section.get("entries", [])
    if not entries:
        raise KeyError("All-groups section has no entries")
    return entries[0]


def compute_loo_diffs(cache_dir: Path) -> dict[str, np.ndarray]:
    """Compute globally-normalized per-group leave-one-out absolute pixel diffs."""
    cache_dir = Path(cache_dir)
    manifest = load_manifest(cache_dir)
    sections = manifest["sections"]
    group_names = tuple(manifest["group_names"])

    if tuple(group_names) != FOUR_GROUP_ORDER:
        raise ValueError(
            f"Expected group_names={FOUR_GROUP_ORDER}, got {tuple(group_names)}",
        )

    all_entry = _find_all_entry(sections, len(group_names))
    img_all = _load_rgb_float32(cache_dir / all_entry["image_path"])

    raw_diffs: dict[str, np.ndarray] = {}
    for group in group_names:
        entry = find_loo_entry(sections, group)
        img_loo = _load_rgb_float32(cache_dir / entry["image_path"])
        diff = np.abs(img_all - img_loo).mean(axis=2).astype(np.float32)
        raw_diffs[group] = diff

    global_max = max(float(diff.max()) for diff in raw_diffs.values())
    if global_max <= 0.0:
        return {group: np.zeros_like(diff, dtype=np.float32) for group, diff in raw_diffs.items()}

    return {
        group: (diff / global_max).astype(np.float32)
        for group, diff in raw_diffs.items()
    }


def save_loo_stats(diffs: dict[str, np.ndarray], out_path: Path) -> None:
    """Write per-group summary stats to JSON."""
    stats = {}
    for group in FOUR_GROUP_ORDER:
        diff = diffs[group]
        diff_255 = diff * 255.0
        stats[group] = {
            "mean_diff": round(float(diff_255.mean()), 4),
            "max_diff": round(float(diff_255.max()), 4),
            "pct_pixels_above_10": round(float((diff_255 > 10).mean() * 100.0), 2),
        }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")


def _display_title(group: str | None) -> str:
    if group is None:
        return "All four channels"
    return f"Drop {group.replace('_', ' ').title()}"


def _resolve_figure_mode(figure_mode: str, legacy_ssim: bool = False) -> str:
    """Normalize CLI figure-mode flags while preserving legacy --ssim behavior."""
    if legacy_ssim:
        return "both"
    if figure_mode not in _FIGURE_MODES:
        raise ValueError(f"Unsupported figure_mode={figure_mode!r}; expected one of {_FIGURE_MODES}")
    return figure_mode


def _resolve_render_paths(
    cache_dir: Path,
    *,
    figure_mode: str,
    out_path: Path | None = None,
    stats_path: Path | None = None,
    ssim_out_path: Path | None = None,
) -> tuple[Path | None, Path | None, Path]:
    """Resolve diff, SSIM, and stats output paths for one cache directory."""
    cache_dir = Path(cache_dir)
    figure_mode = _resolve_figure_mode(figure_mode)

    resolved_stats = (
        Path(stats_path) if stats_path is not None
        else cache_dir / "leave_one_out_diff_stats.json"
    )

    diff_path: Path | None = None
    ssim_path: Path | None = None
    if figure_mode == "diff":
        diff_path = Path(out_path) if out_path is not None else cache_dir / "leave_one_out_diff.png"
    elif figure_mode == "ssim":
        ssim_path = Path(out_path) if out_path is not None else cache_dir / "leave_one_out_ssim.png"
    else:
        diff_path = Path(out_path) if out_path is not None else cache_dir / "leave_one_out_diff.png"
        ssim_path = (
            Path(ssim_out_path) if ssim_out_path is not None
            else diff_path.with_name("leave_one_out_ssim.png")
        )

    return diff_path, ssim_path, resolved_stats


def _compute_relative_diff_maps(
    images: list[np.ndarray],
    baseline: np.ndarray | None,
    per_map: bool = False,
) -> list[np.ndarray]:
    """Return normalized absolute pixel diffs from generated H&E to one baseline image.

    When ``per_map=False`` (default) all maps are normalized by the single global max
    so values remain quantitatively comparable across conditions.  When ``per_map=True``
    each diff map is normalized independently by its own 99th-percentile value, which
    stretches sparse diffs across the full colormap range for visualization.
    """
    if not images:
        return []
    if baseline is None:
        return [np.zeros(images[0].shape[:2], dtype=np.float32) for _ in images]

    baseline_image = _resize_rgb_uint8(baseline, images[0].shape[:2]).astype(np.float32)
    raw_diffs = [
        np.abs(image.astype(np.float32) - baseline_image).mean(axis=2).astype(np.float32)
        for image in images
    ]

    if per_map:
        normalized: list[np.ndarray] = []
        for diff in raw_diffs:
            p99 = float(np.percentile(diff, 99))
            if p99 <= 0.0:
                normalized.append(np.zeros_like(diff, dtype=np.float32))
            else:
                normalized.append(np.clip(diff / p99, 0.0, 1.0).astype(np.float32))
        return normalized

    global_max = max(float(diff.max()) for diff in raw_diffs)
    if global_max <= 0.0:
        return [np.zeros_like(diff, dtype=np.float32) for diff in raw_diffs]
    return [(diff / global_max).astype(np.float32) for diff in raw_diffs]


def _normalize_cell_masked_diff(
    diff: np.ndarray,
    cell_mask: np.ndarray,
) -> np.ndarray:
    """Normalize diff by 99th percentile of cell-region pixels; zero out background.

    Args:
        diff: H×W float32 absolute pixel diff (values in [0, 255]).
        cell_mask: H×W float32 in [0, 1]; pixels > 0.5 are treated as cells.

    Returns:
        H×W float32 in [0, 1]. Non-cell pixels are 0. Returns all-zeros when
        there are no cell pixels or when the 99th-percentile diff is zero.
    """
    cell_pixels = diff[cell_mask > 0.5]
    if len(cell_pixels) == 0 or float(cell_pixels.max()) <= 0.0:
        return np.zeros_like(diff, dtype=np.float32)
    p99 = float(np.percentile(cell_pixels, 99))
    if p99 <= 0.0:
        return np.zeros_like(diff, dtype=np.float32)
    masked = diff * (cell_mask > 0.5).astype(np.float32)
    return np.clip(masked / p99, 0.0, 1.0).astype(np.float32)


def _render_cell_masked_overlay(
    ax,
    raw_diff: np.ndarray,
    cell_mask: np.ndarray,
    baseline_he: np.ndarray,
    cmap,
    *,
    bg_brightness: float = 0.5,
) -> matplotlib.cm.ScalarMappable:
    """Render a cell-masked diff overlay onto `ax`.

    Cell-region pixels: hot-colormap colour keyed to the per-condition
    99th-percentile-normalised diff.  Background pixels: dimmed greyscale
    of `baseline_he`.

    Args:
        ax: matplotlib Axes to draw on.
        raw_diff: H×W float32 absolute pixel diff in [0, 255].
        cell_mask: H×W float32 in [0, 1]; pixels > 0.5 are cells.
        baseline_he: H×W×3 uint8 baseline H&E image.
        cmap: Matplotlib colormap applied to the normalised diff.
        bg_brightness: Multiplier for the greyscale background (default 0.5).

    Returns:
        ScalarMappable suitable for passing to fig.colorbar().
    """
    diff_norm = _normalize_cell_masked_diff(raw_diff, cell_mask)

    bg = baseline_he.astype(np.float32).mean(axis=2) / 255.0 * bg_brightness

    heatmap_rgba = cmap(diff_norm)  # H×W×4

    alpha = (cell_mask > 0.5).astype(np.float32)
    composite = np.stack(
        [alpha * heatmap_rgba[:, :, c] + (1.0 - alpha) * bg for c in range(3)],
        axis=2,
    )
    composite = np.clip(composite, 0.0, 1.0).astype(np.float32)

    ax.imshow(composite, vmin=0.0, vmax=1.0)

    sm = matplotlib.cm.ScalarMappable(
        cmap=cmap,
        norm=mcolors.Normalize(vmin=0.0, vmax=1.0),
    )
    sm.set_array([])
    return sm


def _load_cell_mask_array(cache_dir: Path, manifest: dict) -> np.ndarray | None:
    """Load cached reference cell mask for contour overlay when available."""
    rel = manifest.get("cell_mask_path")
    if not rel:
        return None
    path = cache_dir / str(rel)
    if not path.is_file():
        return None
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def _box_filter_mean(image: np.ndarray, window: int) -> np.ndarray:
    """Return reflect-padded local mean over a square window."""
    pad = window // 2
    padded = np.pad(image, ((pad, pad), (pad, pad)), mode="reflect")
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)
    summed = (
        integral[window:, window:]
        - integral[:-window, window:]
        - integral[window:, :-window]
        + integral[:-window, :-window]
    )
    return summed / float(window * window)


def ssim_loss_map(img_all: np.ndarray, img_drop: np.ndarray, *, win_size: int = 11) -> np.ndarray:
    """Return H×W float32 SSIM structural loss in [0, 1].

    Args:
        img_all: H×W×3 uint8 baseline (all channels) image.
        img_drop: H×W×3 uint8 leave-one-out image.
        win_size: SSIM window size; auto-clamped to image size (must be odd).

    Returns:
        H×W float32 array where 0 = identical structure, 1 = maximum loss.
    """
    gray_all = img_all.mean(axis=2).astype(np.float64)
    gray_drop = img_drop.mean(axis=2).astype(np.float64)
    H, W = gray_all.shape
    actual_win = min(win_size, H, W)
    if actual_win % 2 == 0:
        actual_win -= 1
    actual_win = max(actual_win, 3)
    try:
        from skimage.metrics import structural_similarity as _ssim

        _, ssim_full = _ssim(gray_all, gray_drop, full=True, win_size=actual_win, data_range=255)
        return np.clip(1.0 - ssim_full, 0.0, 1.0).astype(np.float32)
    except ImportError:
        c1 = (0.01 * 255.0) ** 2
        c2 = (0.03 * 255.0) ** 2
        mu_all = _box_filter_mean(gray_all, actual_win)
        mu_drop = _box_filter_mean(gray_drop, actual_win)
        mu_all_sq = mu_all * mu_all
        mu_drop_sq = mu_drop * mu_drop
        mu_all_drop = mu_all * mu_drop

        sigma_all_sq = _box_filter_mean(gray_all * gray_all, actual_win) - mu_all_sq
        sigma_drop_sq = _box_filter_mean(gray_drop * gray_drop, actual_win) - mu_drop_sq
        sigma_all_drop = _box_filter_mean(gray_all * gray_drop, actual_win) - mu_all_drop

        numerator = (2.0 * mu_all_drop + c1) * (2.0 * sigma_all_drop + c2)
        denominator = (mu_all_sq + mu_drop_sq + c1) * (sigma_all_sq + sigma_drop_sq + c2)
        ssim_full = np.ones_like(gray_all, dtype=np.float64)
        valid = denominator > 0.0
        ssim_full[valid] = numerator[valid] / denominator[valid]
        return np.clip(1.0 - ssim_full, 0.0, 1.0).astype(np.float32)


def _select_inset_region(
    loss_mean: np.ndarray,
    crop: int = 64,
    stride: int = 8,
) -> tuple[int, int]:
    """Return (y, x) top-left of the crop with highest mean SSIM loss.

    Slides a ``crop × crop`` window (step ``stride``) over ``loss_mean`` and
    picks the position whose window mean is highest.  The first window wins on
    ties (top-left bias).

    Args:
        loss_mean: H×W float32 mean SSIM loss map (average across conditions).
        crop: Crop side length in pixels.
        stride: Sliding-window stride in pixels.

    Returns:
        ``(y, x)`` top-left corner of the selected crop (both ≥ 0).
    """
    H, W = loss_mean.shape
    best_score: float = -1.0
    best_yx: tuple[int, int] = (0, 0)
    for y in range(0, H - crop + 1, stride):
        for x in range(0, W - crop + 1, stride):
            score = float(loss_mean[y : y + crop, x : x + crop].mean())
            if score > best_score:
                best_score = score
                best_yx = (y, x)
    return best_yx



def _maybe_contour_cell_mask(
    ax,
    cell_mask: np.ndarray | None,
    image_hw: tuple[int, int],
) -> None:
    """Overlay the reference cell-mask contour on an H&E axes."""
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
    ax.contour(cell_mask, levels=[0.5], colors=["lime"], linewidths=0.7, alpha=0.85)


def render_loo_diff_figure(
    diffs: dict[str, np.ndarray],
    cache_dir: Path,
    *,
    orion_root: Path | None = None,
    style_mapping: dict[str, str] | None = None,
    out_path: Path,
) -> None:
    """Save the leave-one-out diff figure."""
    del diffs  # Stats JSON still uses leave-one-out diffs; the figure is reference-vs-generated.
    cache_dir = Path(cache_dir)
    manifest = load_manifest(cache_dir)
    sections = manifest["sections"]
    group_names = tuple(manifest["group_names"])
    tile_id = str(manifest["tile_id"])
    cell_mask = _load_cell_mask_array(cache_dir, manifest)

    all_entry = _find_all_entry(sections, len(group_names))
    img_all = _load_rgb_float32(cache_dir / all_entry["image_path"]).astype(np.uint8)

    ref_he = None
    if orion_root is not None:
        he_path = default_orion_he_png_path(Path(orion_root), tile_id, style_mapping=style_mapping)
        if he_path is not None:
            ref_he = np.asarray(Image.open(he_path).convert("RGB"), dtype=np.uint8)

    hot_cmap = mcolors.LinearSegmentedColormap.from_list(
        "hot4",
        ["#000000", "#ff4400", "#ffff00", "#ffffff"],
    )

    def _blank_rgb(size: int = 64) -> np.ndarray:
        return np.full((size, size, 3), 45, dtype=np.uint8)

    display_labels = [_display_title(None)] + [_display_title(group) for group in group_names]
    display_images = [img_all]
    for group in group_names:
        entry = find_loo_entry(sections, group)
        display_images.append(_load_rgb_float32(cache_dir / entry["image_path"]).astype(np.uint8))

    baseline_float = img_all.astype(np.float32)
    raw_diffs = [
        np.abs(img.astype(np.float32) - baseline_float).mean(axis=2).astype(np.float32)
        for img in display_images
    ]
    fallback_diffs = (
        None if cell_mask is not None
        else _compute_relative_diff_maps(display_images, img_all, per_map=True)
    )

    fig_width = 15.0
    fig_height = 4.45
    fig = plt.figure(figsize=(fig_width, fig_height))
    fig.suptitle(f"Leave-one-out group diff - tile {tile_id}", fontsize=12, y=0.985)

    left_margin = 0.035
    right_margin = 0.985
    top = 0.80
    bottom = 0.14
    row_gap = 0.01
    col_gap = 0.001
    ref_gap = 0.012

    content_height = top - bottom
    row_height = (content_height - row_gap) / 2.0
    ref_width = content_height * (fig_height / fig_width)
    x_right_start = left_margin + ref_width + ref_gap
    available_width = right_margin - x_right_start
    panel_width = (available_width - col_gap * 4.0) / 7.4

    reference_ax = fig.add_axes([left_margin, bottom, ref_width, content_height])
    reference_image = ref_he if ref_he is not None else _blank_rgb(128)
    reference_ax.imshow(reference_image)
    _maybe_contour_cell_mask(reference_ax, cell_mask, reference_image.shape[:2])
    reference_ax.set_title("Reference H&E (style)", fontsize=10)
    draw_image_border(reference_ax, COLOR_REF, dashed=True)
    reference_ax.set_xticks([])
    reference_ax.set_yticks([])

    last_im = None
    top_row_y = bottom + row_height + row_gap
    bottom_row_y = bottom

    for index, (label, image, raw_diff) in enumerate(zip(display_labels, display_images, raw_diffs, strict=True)):
        x0 = x_right_start + index * (panel_width + col_gap)

        image_ax = fig.add_axes([x0, top_row_y, panel_width, row_height])
        image_ax.imshow(image)
        _maybe_contour_cell_mask(image_ax, cell_mask, image.shape[:2])
        image_ax.set_title(label, fontsize=9)
        image_ax.set_xticks([])
        image_ax.set_yticks([])
        if index == 0:
            image_ax.set_ylabel("Generated H&E", fontsize=10, rotation=90, labelpad=2)

        diff_ax = fig.add_axes([x0, bottom_row_y, panel_width, row_height])
        if cell_mask is not None:
            last_im = _render_cell_masked_overlay(diff_ax, raw_diff, cell_mask, img_all, hot_cmap)
        else:
            last_im = diff_ax.imshow(
                fallback_diffs[index], cmap=hot_cmap, vmin=0.0, vmax=1.0
            )
        diff_ax.set_xticks([])
        diff_ax.set_yticks([])
        if index == 0:
            diff_ax.set_ylabel("Pixel Diff", fontsize=10, rotation=90, labelpad=2)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if last_im is not None:
        cbar_width = panel_width
        cbar_x = x_right_start + 4.0 * (panel_width + col_gap)
        cax = fig.add_axes([cbar_x, 0.06, cbar_width, 0.02])
        cbar = fig.colorbar(last_im, cax=cax, orientation="horizontal")
        cbar.ax.tick_params(labelsize=8, pad=1)
        cbar_label = (
            "Cell-masked pixel diff (per-condition, 99th-pct norm.)"
            if cell_mask is not None
            else "Pixel diff (per-condition, 99th-pct norm.)"
        )
        cbar.set_label(cbar_label, fontsize=8, labelpad=3)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def render_loo_ssim_figure(
    cache_dir: Path,
    *,
    orion_root: Path | None = None,
    style_mapping: dict[str, str] | None = None,
    out_path: Path,
    crop_size: int = 64,
) -> None:
    """Save the LOO SSIM publication figure (3 rows x 5 columns).

    Row 0 -- Generated H&E (full tile) with teal inset-region marker on the
             all-channels panel.
    Row 1 -- Cell inset: 64x64 crop (same region every column), upsampled 4x
             with nearest-neighbour. Region auto-selected from max mean SSIM loss.
    Row 2 -- SSIM structural loss map (cell-masked, globally normalised).
             All-channels column shown as black "0 (baseline)" panel.

    Args:
        cache_dir: Ablation cache directory containing manifest.json.
        orion_root: Optional ORION dataset root (reserved, unused in output).
        style_mapping: Optional tile-id remapping (reserved, unused in output).
        out_path: Destination PNG path.
        crop_size: Side length of the inset crop in pixels (default 64).
    """
    from matplotlib.patches import Rectangle
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    cache_dir = Path(cache_dir)
    manifest = load_manifest(cache_dir)
    sections = manifest["sections"]
    group_names = tuple(manifest["group_names"])
    tile_id = str(manifest["tile_id"])
    cell_mask = _load_cell_mask_array(cache_dir, manifest)

    # Load images
    all_entry = _find_all_entry(sections, len(group_names))
    img_all = _load_rgb_float32(cache_dir / all_entry["image_path"]).astype(np.uint8)

    loo_images: list[np.ndarray] = []
    for group in FOUR_GROUP_ORDER:
        entry = find_loo_entry(sections, group)
        loo_images.append(
            _load_rgb_float32(cache_dir / entry["image_path"]).astype(np.uint8)
        )

    # SSIM loss maps
    raw_ssim: list[np.ndarray] = [ssim_loss_map(img_all, img) for img in loo_images]

    if cell_mask is not None:
        H, W = img_all.shape[:2]
        if cell_mask.shape != (H, W):
            cell_mask = np.asarray(
                Image.fromarray((np.clip(cell_mask, 0, 1) * 255).astype(np.uint8)).resize(
                    (W, H), Image.BILINEAR
                ),
                dtype=np.float32,
            ) / 255.0
        binary = (cell_mask > 0.5).astype(np.float32)
        raw_ssim = [m * binary for m in raw_ssim]

    global_max = max(float(m.max()) for m in raw_ssim)
    if global_max > 0.0:
        ssim_norm: list[np.ndarray] = [
            np.clip(m / global_max, 0.0, 1.0).astype(np.float32) for m in raw_ssim
        ]
    else:
        ssim_norm = [np.zeros_like(m) for m in raw_ssim]

    # Inset region selection
    loss_mean = np.stack(ssim_norm).mean(axis=0).astype(np.float32)
    iy, ix = _select_inset_region(loss_mean, crop=crop_size, stride=8)

    def _crop_upsample(img: np.ndarray) -> np.ndarray:
        H_out, W_out = img.shape[:2]
        crop = img[iy : iy + crop_size, ix : ix + crop_size]
        return np.asarray(
            Image.fromarray(crop).resize((W_out, H_out), Image.NEAREST),
            dtype=np.uint8,
        )

    he_images = [img_all] + loo_images
    inset_images = [_crop_upsample(img) for img in he_images]
    ssim_display = [np.zeros(img_all.shape[:2], dtype=np.float32)] + ssim_norm

    col_labels = ["All channels"] + [
        f"Drop {g.replace('_', ' ').title()}" for g in FOUR_GROUP_ORDER
    ]
    col_title_colors = ["#cccccc"] + [
        _LOO_SSIM_HIGHLIGHT.get(g, "#888888") for g in FOUR_GROUP_ORDER
    ]
    col_title_weights = ["normal"] + [
        "bold" if g in _LOO_SSIM_HIGHLIGHT else "normal" for g in FOUR_GROUP_ORDER
    ]
    inset_colors = [_LOO_SSIM_INSET_TEAL] + [
        _LOO_SSIM_HIGHLIGHT.get(g, _LOO_SSIM_NEUTRAL) for g in FOUR_GROUP_ORDER
    ]
    inset_lw = [2.0] + [2.0 if g in _LOO_SSIM_HIGHLIGHT else 1.0 for g in FOUR_GROUP_ORDER]

    hot_cmap = mcolors.LinearSegmentedColormap.from_list(
        "hot4", ["#000000", "#ff4400", "#ffff00", "#ffffff"]
    )

    fig, axes = plt.subplots(3, 5, figsize=(15, 9))
    fig.suptitle(
        f"Leave-one-out group diff (SSIM) -- tile {tile_id}", fontsize=12, y=0.985
    )

    for col in range(5):
        ax0 = axes[0, col]
        ax0.imshow(he_images[col])
        if cell_mask is not None:
            _maybe_contour_cell_mask(ax0, cell_mask, he_images[col].shape[:2])
        if col == 0:
            ax0.add_patch(
                Rectangle(
                    (ix, iy),
                    crop_size,
                    crop_size,
                    linewidth=2,
                    edgecolor=_LOO_SSIM_INSET_TEAL,
                    facecolor="none",
                )
            )
        ax0.set_title(
            col_labels[col],
            fontsize=9,
            color=col_title_colors[col],
            fontweight=col_title_weights[col],
        )
        ax0.set_xticks([])
        ax0.set_yticks([])
        if col == 0:
            ax0.set_ylabel("Generated H&E", fontsize=10)

        ax1 = axes[1, col]
        ax1.imshow(inset_images[col])
        for spine in ax1.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(inset_colors[col])
            spine.set_linewidth(inset_lw[col])
        ax1.set_xticks([])
        ax1.set_yticks([])
        if col == 0:
            ax1.set_ylabel("Cell inset\n(auto-selected)", fontsize=10)

        ax2 = axes[2, col]
        ax2.imshow(ssim_display[col], cmap=hot_cmap, vmin=0.0, vmax=1.0)
        if col == 0:
            ax2.text(
                0.5, 0.5, "0\n(baseline)",
                transform=ax2.transAxes,
                ha="center", va="center",
                fontsize=8, color="#777777",
            )
        ax2.set_xticks([])
        ax2.set_yticks([])
        if col == 0:
            ax2.set_ylabel("SSIM loss\n(cell-masked)", fontsize=10)

    divider = make_axes_locatable(axes[2, 4])
    cbar_ax = divider.append_axes("bottom", size="8%", pad=0.20)
    sm = matplotlib.cm.ScalarMappable(
        cmap=hot_cmap, norm=mcolors.Normalize(vmin=0.0, vmax=1.0)
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_ticks([0.0, 0.5, 1.0])
    cbar.ax.tick_params(labelsize=8, pad=1)
    cbar.set_label("SSIM structural loss (globally normalized)", fontsize=8, labelpad=3)

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def render_loo_cache(
    cache_dir: Path,
    *,
    orion_root: Path | None = None,
    style_mapping: dict[str, str] | None = None,
    out_path: Path | None = None,
    stats_path: Path | None = None,
    ssim_out_path: Path | None = None,
    figure_mode: str = "diff",
    ssim: bool = False,
    crop_size: int = 64,
) -> tuple[Path, Path]:
    """Render one cache dir and return figure/stats paths.

    ``figure_mode`` controls which figure(s) are written:
    ``"diff"`` (default), ``"ssim"``, or ``"both"``. The legacy ``ssim=True``
    flag is preserved as an alias for ``figure_mode="both"``.
    """
    cache_dir = Path(cache_dir)
    figure_mode = _resolve_figure_mode(figure_mode, legacy_ssim=ssim)
    diff_out_path, ssim_render_path, stats_path = _resolve_render_paths(
        cache_dir,
        figure_mode=figure_mode,
        out_path=out_path,
        stats_path=stats_path,
        ssim_out_path=ssim_out_path,
    )

    diffs = compute_loo_diffs(cache_dir)
    save_loo_stats(diffs, stats_path)
    if diff_out_path is not None:
        render_loo_diff_figure(
            diffs,
            cache_dir,
            orion_root=orion_root,
            style_mapping=style_mapping,
            out_path=diff_out_path,
        )

    if ssim_render_path is not None:
        render_loo_ssim_figure(
            cache_dir,
            orion_root=orion_root,
            style_mapping=style_mapping,
            out_path=ssim_render_path,
            crop_size=crop_size,
        )

    primary_figure_path = diff_out_path if diff_out_path is not None else ssim_render_path
    if primary_figure_path is None:
        raise RuntimeError("No figure output path resolved")
    return primary_figure_path, stats_path


def _find_cache_dirs(cache_root: Path) -> list[Path]:
    """Return all cache directories under cache_root that contain a manifest."""
    cache_root = Path(cache_root)
    return sorted(path.parent for path in cache_root.rglob("manifest.json"))


def _progress(iterable, *, total: int, desc: str, disable: bool):
    """Return a tqdm progress bar when available, else the raw iterable."""
    if disable:
        return iterable
    try:
        from tqdm.auto import tqdm

        return tqdm(iterable, total=total, desc=desc)
    except Exception:
        return iterable


def render_loo_cache_root(
    cache_root: Path,
    *,
    orion_root: Path | None = None,
    style_mapping: dict[str, str] | None = None,
    out_root: Path | None = None,
    workers: int = 1,
    show_progress: bool = True,
    figure_mode: str = "diff",
    ssim: bool = False,
    crop_size: int = 64,
) -> list[tuple[Path, Path]]:
    """Render leave-one-out figures for every cache under cache_root."""
    cache_root = Path(cache_root)
    cache_dirs = _find_cache_dirs(cache_root)
    if not cache_dirs:
        raise FileNotFoundError(f"No manifest.json files found under {cache_root}")

    worker_count = max(1, int(workers))
    figure_mode = _resolve_figure_mode(figure_mode, legacy_ssim=ssim)

    def _resolve_outputs(cache_dir: Path) -> tuple[Path | None, Path | None, Path]:
        if out_root is None:
            return _resolve_render_paths(cache_dir, figure_mode=figure_mode)

        rel = cache_dir.relative_to(cache_root)
        base_dir = Path(out_root) / rel
        out_path = None
        ssim_out_path = None
        if figure_mode == "diff":
            out_path = base_dir / "leave_one_out_diff.png"
        elif figure_mode == "ssim":
            out_path = base_dir / "leave_one_out_ssim.png"
        else:
            out_path = base_dir / "leave_one_out_diff.png"
            ssim_out_path = base_dir / "leave_one_out_ssim.png"
        stats_path = base_dir / "leave_one_out_diff_stats.json"
        return _resolve_render_paths(
            cache_dir,
            figure_mode=figure_mode,
            out_path=out_path,
            ssim_out_path=ssim_out_path,
            stats_path=stats_path,
        )

    if worker_count == 1:
        rendered: list[tuple[Path, Path]] = []
        iterator = _progress(
            cache_dirs,
            total=len(cache_dirs),
            desc="Rendering LOO",
            disable=not show_progress,
        )
        for cache_dir in iterator:
            diff_out_path, ssim_out_path, stats_path = _resolve_outputs(cache_dir)
            primary_out = diff_out_path if diff_out_path is not None else ssim_out_path
            if primary_out is None:
                raise RuntimeError("No primary figure output path resolved")
            rendered.append(
                render_loo_cache(
                    cache_dir,
                    orion_root=orion_root,
                    style_mapping=style_mapping,
                    out_path=primary_out,
                    stats_path=stats_path,
                    ssim_out_path=ssim_out_path,
                    figure_mode=figure_mode,
                    crop_size=crop_size,
                )
            )
        return rendered

    future_to_cache: dict[Any, Path] = {}
    rendered: list[tuple[Path, Path]] = []
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        for cache_dir in cache_dirs:
            diff_out_path, ssim_out_path, stats_path = _resolve_outputs(cache_dir)
            primary_out = diff_out_path if diff_out_path is not None else ssim_out_path
            if primary_out is None:
                raise RuntimeError("No primary figure output path resolved")
            future = executor.submit(
                render_loo_cache,
                cache_dir,
                orion_root=orion_root,
                style_mapping=style_mapping,
                out_path=primary_out,
                stats_path=stats_path,
                ssim_out_path=ssim_out_path,
                figure_mode=figure_mode,
                crop_size=crop_size,
            )
            future_to_cache[future] = cache_dir

        iterator = _progress(
            as_completed(future_to_cache),
            total=len(future_to_cache),
            desc=f"Rendering LOO ({worker_count} workers)",
            disable=not show_progress,
        )
        completed: dict[Path, tuple[Path, Path]] = {}
        for future in iterator:
            cache_dir = future_to_cache[future]
            completed[cache_dir] = future.result()

    for cache_dir in cache_dirs:
        rendered.append(completed[cache_dir])
    return rendered


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Leave-one-out group pixel diff from ablation cache")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--cache-dir", help="Path to one tile cache dir containing manifest.json")
    input_group.add_argument("--cache-root", help="Root directory containing many tile cache dirs")
    parser.add_argument("--orion-root", default=None, help="Optional ORION dataset root for channel thumbnails")
    parser.add_argument(
        "--style-mapping-json",
        default=None,
        help="Optional layout->style mapping JSON for unpaired reference H&E lookup.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output PNG path for the requested figure mode in --cache-dir mode.",
    )
    parser.add_argument(
        "--out-root",
        default=None,
        help="Batch output root for --cache-root mode; mirrors cache-root subdirs",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, max(1, os.cpu_count() or 1)),
        help="Worker count for --cache-root mode (default: min(8, cpu_count))",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the batch progress bar in --cache-root mode",
    )
    parser.add_argument(
        "--figure",
        choices=_FIGURE_MODES,
        default="diff",
        help="Which figure to render: diff, ssim, or both (default: diff).",
    )
    parser.add_argument(
        "--ssim",
        action="store_true",
        help="Legacy alias for --figure both; preserved for backwards compatibility.",
    )
    parser.add_argument(
        "--ssim-out",
        default=None,
        help="Optional SSIM PNG path in --cache-dir mode when --figure both is used.",
    )
    parser.add_argument(
        "--crop-size",
        type=int,
        default=64,
        help="Inset crop side length in pixels for the SSIM figure (default: 64).",
    )
    args = parser.parse_args(argv)

    orion_root = Path(args.orion_root) if args.orion_root else None
    style_mapping = load_style_mapping(args.style_mapping_json)
    out_root = Path(args.out_root) if args.out_root else None
    figure_mode = _resolve_figure_mode(args.figure, legacy_ssim=args.ssim)

    if args.crop_size <= 0:
        parser.error("--crop-size must be a positive integer")
    if args.ssim_out and not args.cache_dir:
        parser.error("--ssim-out is only valid with --cache-dir")
    if args.ssim_out and figure_mode != "both":
        parser.error("--ssim-out is only valid when --figure both is used")

    if args.cache_dir:
        cache_dir = Path(args.cache_dir)
        out_path = Path(args.out) if args.out else None
        ssim_out_path = Path(args.ssim_out) if args.ssim_out else None
        diff_render_path, ssim_render_path, stats_path = _resolve_render_paths(
            cache_dir,
            figure_mode=figure_mode,
            out_path=out_path,
            ssim_out_path=ssim_out_path,
        )
        fig_path, stats_path = render_loo_cache(
            cache_dir,
            orion_root=orion_root,
            style_mapping=style_mapping,
            out_path=out_path,
            ssim_out_path=ssim_out_path,
            figure_mode=figure_mode,
            crop_size=args.crop_size,
        )
        print(f"Saved stats -> {stats_path}")
        if diff_render_path is not None:
            print(f"Saved diff figure -> {diff_render_path}")
        if ssim_render_path is not None:
            print(f"Saved SSIM figure -> {ssim_render_path}")
        return

    if args.out:
        parser.error("--out is only valid with --cache-dir")
    if args.ssim_out:
        parser.error("--ssim-out is only valid with --cache-dir")

    rendered = render_loo_cache_root(
        Path(args.cache_root),
        orion_root=orion_root,
        style_mapping=style_mapping,
        out_root=out_root,
        workers=args.workers,
        show_progress=not args.no_progress,
        figure_mode=figure_mode,
        crop_size=args.crop_size,
    )
    print(f"Rendered {figure_mode} figure(s) for {len(rendered)} cache dirs under {args.cache_root}")
    for fig_path, stats_path in rendered:
        print(f"Saved stats -> {stats_path}")
        print(f"Saved figure -> {fig_path}")


if __name__ == "__main__":
    main()
