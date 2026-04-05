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
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image

from tools.stage3.ablation_vis_utils import (
    FOUR_GROUP_ORDER,
    build_exp_channel_header_rgb,
    default_orion_he_png_path,
)

COLOR_REF = "#999999"
COLOR_BASELINE = "#9B59B6"


def _load_rgb_float32(path: Path) -> np.ndarray:
    """Load a PNG as float32 H×W×3 in [0, 255]."""
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)


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
    manifest_path = cache_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
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


def render_loo_diff_figure(
    diffs: dict[str, np.ndarray],
    cache_dir: Path,
    *,
    orion_root: Path | None = None,
    out_path: Path,
) -> None:
    """Save the leave-one-out diff figure."""
    cache_dir = Path(cache_dir)
    manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    sections = manifest["sections"]
    group_names = tuple(manifest["group_names"])
    tile_id = str(manifest["tile_id"])

    all_entry = _find_all_entry(sections, len(group_names))
    img_all = _load_rgb_float32(cache_dir / all_entry["image_path"]).astype(np.uint8)

    ref_he = None
    if orion_root is not None:
        he_path = default_orion_he_png_path(Path(orion_root), tile_id)
        if he_path is not None:
            ref_he = np.asarray(Image.open(he_path).convert("RGB"), dtype=np.uint8)

    group_thumbs = None
    if orion_root is not None:
        try:
            group_thumbs = build_exp_channel_header_rgb(Path(orion_root) / "exp_channels", tile_id)
        except (FileNotFoundError, OSError, ValueError, KeyError):
            group_thumbs = None

    hot_cmap = mcolors.LinearSegmentedColormap.from_list(
        "hot4",
        ["#000000", "#ff4400", "#ffff00", "#ffffff"],
    )

    n_rows = 4
    n_cols = len(group_names) + 1  # combined reference/baseline column + four groups
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.45 * n_cols, 8.5))
    fig.suptitle(f"Leave-one-out group diff - tile {tile_id}", fontsize=12, y=0.995)

    def _blank_rgb(size: int = 64) -> np.ndarray:
        return np.full((size, size, 3), 45, dtype=np.uint8)

    def _draw_cell_border(ax, color: str, *, dashed: bool = False) -> None:
        patch = Rectangle(
            (0.0, 0.0),
            1.0,
            1.0,
            transform=ax.transAxes,
            fill=False,
            edgecolor=color,
            linewidth=2.5,
            linestyle="--" if dashed else "-",
            zorder=10,
            clip_on=False,
        )
        ax.add_patch(patch)

    row_labels = ("Inputs", "Leave-one-out H&E", "Diff heatmap", "Diff stats")
    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(label, fontsize=10, rotation=90, labelpad=22)

    # Combined left column: reference on the first row, baseline on the second row.
    if ref_he is not None:
        axes[0, 0].imshow(ref_he)
    else:
        axes[0, 0].imshow(_blank_rgb())
    axes[0, 0].set_title("reference", fontsize=9)
    _draw_cell_border(axes[0, 0], COLOR_REF, dashed=True)
    axes[0, 0].set_xticks([])
    axes[0, 0].set_yticks([])
    for row in range(1, 4):
        axes[row, 0].axis("off")

    axes[1, 0].imshow(img_all)
    axes[1, 0].set_title("baseline", fontsize=9)
    _draw_cell_border(axes[1, 0], COLOR_BASELINE, dashed=False)
    axes[1, 0].set_xticks([])
    axes[1, 0].set_yticks([])
    axes[2, 0].axis("off")
    axes[3, 0].axis("off")

    # Per-group columns.
    last_im = None
    for col, group in enumerate(group_names, start=1):
        entry = find_loo_entry(sections, group)
        img_loo = _load_rgb_float32(cache_dir / entry["image_path"]).astype(np.uint8)
        diff = diffs[group]
        stats = diff * 255.0

        if group_thumbs is not None:
            axes[0, col].imshow(group_thumbs.get(group, _blank_rgb()))
        else:
            axes[0, col].imshow(_blank_rgb())
        axes[0, col].set_title(group.replace("_", "\n"), fontsize=9)
        axes[0, col].axis("off")

        axes[1, col].imshow(img_loo)
        axes[1, col].set_title(f"w/o {group}", fontsize=9)
        axes[1, col].axis("off")

        last_im = axes[2, col].imshow(diff, cmap=hot_cmap, vmin=0.0, vmax=1.0)
        axes[2, col].axis("off")

        mean = float(stats.mean())
        std = float(stats.std())
        max_ = float(stats.max())
        pct10 = float((stats > 10.0).mean() * 100.0)
        axes[3, col].text(
            0.5,
            0.55,
            f"mean±std: {mean:.1f} ± {std:.1f}\nmax: {max_:.1f}\npct>10: {pct10:.1f}%",
            ha="center",
            va="center",
            fontsize=9,
        )
        axes[3, col].set_xlim(0, 1)
        axes[3, col].set_ylim(0, 1)
        axes[3, col].axis("off")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.07, right=0.998, top=0.93, bottom=0.1, wspace=0.004, hspace=0.14)
    if last_im is not None:
        ref_ax = axes[2, 1] if n_cols > 1 else axes[2, 0]
        stats_ax = axes[3, 1] if n_cols > 1 else axes[3, 0]
        ref_pos = ref_ax.get_position()
        stats_pos = stats_ax.get_position()
        gap = max(ref_pos.y0 - stats_pos.y1, 0.02)
        cbar_height = min(0.01, gap * 0.16)
        cbar_y = stats_pos.y1 + gap * 0.08
        cax = fig.add_axes([ref_pos.x0, cbar_y, ref_pos.width, cbar_height])
        cbar = fig.colorbar(last_im, cax=cax, orientation="horizontal")
        cbar.ax.tick_params(labelsize=8, pad=1)
        fig.text(
            ref_pos.x0 + ref_pos.width * 0.5,
            ref_pos.y0 - gap * 0.18,
            "Normalized absolute pixel diff",
            ha="center",
            va="bottom",
            fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.4},
        )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def render_loo_cache(
    cache_dir: Path,
    *,
    orion_root: Path | None = None,
    out_path: Path | None = None,
    stats_path: Path | None = None,
) -> tuple[Path, Path]:
    """Render one cache dir and return figure/stats paths."""
    cache_dir = Path(cache_dir)
    out_path = Path(out_path) if out_path is not None else cache_dir / "leave_one_out_diff.png"
    stats_path = Path(stats_path) if stats_path is not None else out_path.with_name("leave_one_out_diff_stats.json")

    diffs = compute_loo_diffs(cache_dir)
    save_loo_stats(diffs, stats_path)
    render_loo_diff_figure(diffs, cache_dir, orion_root=orion_root, out_path=out_path)
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
    out_root = Path(args.out_root) if args.out_root else None

    if args.cache_dir:
        cache_dir = Path(args.cache_dir)
        out_path = Path(args.out) if args.out else cache_dir / "leave_one_out_diff.png"
        fig_path, stats_path = render_loo_cache(
            cache_dir,
            orion_root=orion_root,
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
