"""Leave-one-out group pixel diff from cached ablation PNGs.

Usage:
    python tools/vis/leave_one_out_diff.py \
        --cache-dir inference_output/cache/512_9728 \
        --orion-root data/orion-crc33 \
        --out inference_output/cache/512_9728/leave_one_out_diff.png

    python tools/vis/leave_one_out_diff.py \
        --cache-root inference_output/cache \
        --orion-root data/orion-crc33
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


def _load_cell_mask_array(cache_dir: Path, manifest: dict) -> np.ndarray | None:
    """Load cached reference cell mask for contour overlay when available."""
    rel = manifest.get("cell_mask_path")
    if not rel:
        return None
    path = cache_dir / str(rel)
    if not path.is_file():
        return None
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0


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
    display_diffs = _compute_relative_diff_maps(display_images, img_all, per_map=True)

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

    for index, (label, image, diff_map) in enumerate(zip(display_labels, display_images, display_diffs, strict=True)):
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
        last_im = diff_ax.imshow(diff_map, cmap=hot_cmap, vmin=0.0, vmax=1.0)
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
        cbar.set_label("Pixel diff (per-condition, 99th-pct norm.)", fontsize=8, labelpad=3)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def render_loo_cache(
    cache_dir: Path,
    *,
    orion_root: Path | None = None,
    style_mapping: dict[str, str] | None = None,
    out_path: Path | None = None,
    stats_path: Path | None = None,
) -> tuple[Path, Path]:
    """Render one cache dir and return figure/stats paths."""
    cache_dir = Path(cache_dir)
    out_path = Path(out_path) if out_path is not None else cache_dir / "leave_one_out_diff.png"
    stats_path = Path(stats_path) if stats_path is not None else out_path.with_name("leave_one_out_diff_stats.json")

    diffs = compute_loo_diffs(cache_dir)
    save_loo_stats(diffs, stats_path)
    render_loo_diff_figure(
        diffs,
        cache_dir,
        orion_root=orion_root,
        style_mapping=style_mapping,
        out_path=out_path,
    )
    return out_path, stats_path


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
) -> list[tuple[Path, Path]]:
    """Render leave-one-out figures for every cache under cache_root."""
    cache_root = Path(cache_root)
    cache_dirs = _find_cache_dirs(cache_root)
    if not cache_dirs:
        raise FileNotFoundError(f"No manifest.json files found under {cache_root}")

    worker_count = max(1, int(workers))

    def _resolve_outputs(cache_dir: Path) -> tuple[Path, Path]:
        if out_root is None:
            out_path = cache_dir / "leave_one_out_diff.png"
            stats_path = cache_dir / "leave_one_out_diff_stats.json"
        else:
            rel = cache_dir.relative_to(cache_root)
            out_path = Path(out_root) / rel / "leave_one_out_diff.png"
            stats_path = Path(out_root) / rel / "leave_one_out_diff_stats.json"
        return out_path, stats_path

    if worker_count == 1:
        rendered: list[tuple[Path, Path]] = []
        iterator = _progress(
            cache_dirs,
            total=len(cache_dirs),
            desc="Rendering LOO",
            disable=not show_progress,
        )
        for cache_dir in iterator:
            out_path, stats_path = _resolve_outputs(cache_dir)
            rendered.append(
                render_loo_cache(
                    cache_dir,
                    orion_root=orion_root,
                    style_mapping=style_mapping,
                    out_path=out_path,
                    stats_path=stats_path,
                )
            )
        return rendered

    future_to_cache: dict[Any, Path] = {}
    rendered: list[tuple[Path, Path]] = []
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        for cache_dir in cache_dirs:
            out_path, stats_path = _resolve_outputs(cache_dir)
            future = executor.submit(
                render_loo_cache,
                cache_dir,
                orion_root=orion_root,
                style_mapping=style_mapping,
                out_path=out_path,
                stats_path=stats_path,
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


def main() -> None:
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
        help="Output PNG path (default: <cache-dir>/leave_one_out_diff.png)",
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
    args = parser.parse_args()

    orion_root = Path(args.orion_root) if args.orion_root else None
    style_mapping = load_style_mapping(args.style_mapping_json)
    out_root = Path(args.out_root) if args.out_root else None

    if args.cache_dir:
        cache_dir = Path(args.cache_dir)
        out_path = Path(args.out) if args.out else cache_dir / "leave_one_out_diff.png"
        fig_path, stats_path = render_loo_cache(
            cache_dir,
            orion_root=orion_root,
            style_mapping=style_mapping,
            out_path=out_path,
        )
        print(f"Saved stats -> {stats_path}")
        print(f"Saved figure -> {fig_path}")
        return

    if args.out:
        parser.error("--out is only valid with --cache-dir")

    rendered = render_loo_cache_root(
        Path(args.cache_root),
        orion_root=orion_root,
        style_mapping=style_mapping,
        out_root=out_root,
        workers=args.workers,
        show_progress=not args.no_progress,
    )
    print(f"Rendered {len(rendered)} cache dirs under {args.cache_root}")
    for fig_path, stats_path in rendered:
        print(f"Saved stats -> {stats_path}")
        print(f"Saved figure -> {fig_path}")


if __name__ == "__main__":
    main()
