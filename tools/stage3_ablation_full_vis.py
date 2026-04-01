"""
Standalone matrix-plus-grid visualization for the full Stage 3 group ablation suite.

This keeps the combined single/pair/triple layout separate from ``stage3_figures.py``
while we iterate on the publication-style design.
"""
from __future__ import annotations

import argparse
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
import sys
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.color_constants import SECTION_BG, SECTION_TEXT
from tools.stage3_ablation import AblationCondition, build_subset_conditions, group_display_name
from tools.stage3_ablation_cache import is_per_tile_cache_manifest_dir, list_cached_tile_ids

_ACTIVE_DOT = "#2b6f52"
_INACTIVE_EDGE = "#c3ccd3"
_GRID_LINE = "#e1e6ea"


@dataclass(frozen=True)
class AblationVisSection:
    """One grouped set of ablation conditions to render in the combined figure."""

    title: str
    conditions: tuple[AblationCondition, ...]
    images: tuple[tuple[str, np.ndarray], ...]


def build_subset_ablation_sections(
    group_names: Sequence[str],
    *,
    single_images: Sequence[tuple[str, np.ndarray]],
    pair_images: Sequence[tuple[str, np.ndarray]],
    triple_images: Sequence[tuple[str, np.ndarray]],
) -> list[AblationVisSection]:
    """Build the standard 1/2/3-group subset sections for the combined figure."""
    return [
        AblationVisSection(
            title="1 active group",
            conditions=tuple(build_subset_conditions(group_names, subset_size=1)),
            images=tuple(single_images),
        ),
        AblationVisSection(
            title="2 active groups",
            conditions=tuple(build_subset_conditions(group_names, subset_size=2)),
            images=tuple(pair_images),
        ),
        AblationVisSection(
            title="3 active groups",
            conditions=tuple(build_subset_conditions(group_names, subset_size=3)),
            images=tuple(triple_images),
        ),
    ]


def _extract_cell_mask(
    ctrl_full: np.ndarray | None,
    active_channels: Sequence[str] | None,
) -> np.ndarray | None:
    if ctrl_full is None or active_channels is None or "cell_masks" not in active_channels:
        return None
    return ctrl_full[active_channels.index("cell_masks")]


def _validate_sections(sections: Sequence[AblationVisSection]) -> None:
    if not sections:
        raise ValueError("sections must not be empty")

    for section in sections:
        if not section.conditions:
            raise ValueError(f"section {section.title!r} has no conditions")
        if len(section.conditions) != len(section.images):
            raise ValueError(
                f"section {section.title!r} has {len(section.conditions)} conditions "
                f"but {len(section.images)} images"
            )


def _condition_label(condition: AblationCondition) -> str:
    names = [group_display_name(name) for name in condition.active_groups]
    return "\n".join(textwrap.wrap(" + ".join(names), width=24, break_long_words=False))


def _draw_condition_matrix(
    ax,
    *,
    conditions: Sequence[AblationCondition],
    group_names: Sequence[str],
) -> None:
    y_positions = np.arange(len(group_names) - 1, -1, -1)
    active_lookup = [set(condition.active_groups) for condition in conditions]

    for x in range(len(conditions) + 1):
        ax.axvline(x - 0.5, color=_GRID_LINE, linewidth=0.8, zorder=0)
    for y in range(len(group_names) + 1):
        ax.axhline(y - 0.5, color=_GRID_LINE, linewidth=0.8, zorder=0)

    for row_idx, group_name in enumerate(group_names):
        y = y_positions[row_idx]
        for col_idx, active_groups in enumerate(active_lookup):
            is_active = group_name in active_groups
            ax.scatter(
                col_idx,
                y,
                s=130,
                facecolors=_ACTIVE_DOT if is_active else "white",
                edgecolors=_ACTIVE_DOT if is_active else _INACTIVE_EDGE,
                linewidths=1.3,
                zorder=3,
            )

    ax.set_xlim(-0.5, len(conditions) - 0.5)
    ax.set_ylim(-0.5, len(group_names) - 0.5)
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels([])
    ax.set_yticks(y_positions)
    ax.set_yticklabels([group_display_name(name) for name in group_names], fontsize=8)
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _draw_section_header(ax, title: str, *, is_first_section: bool) -> None:
    ax.set_facecolor(SECTION_BG["analysis"])
    ax.text(
        0.015,
        0.5,
        title,
        ha="left",
        va="center",
        fontsize=10,
        fontweight="bold",
        color=SECTION_TEXT["analysis"],
        transform=ax.transAxes,
    )
    if is_first_section:
        ax.text(
            0.985,
            0.5,
            "filled dots = active groups",
            ha="right",
            va="center",
            fontsize=8,
            color="#5a5a5a",
            transform=ax.transAxes,
        )
    ax.axis("off")


def save_condition_matrix_ablation_grid(
    sections: Sequence[AblationVisSection],
    save_path: str | Path,
    *,
    group_names: Sequence[str] | None = None,
    ctrl_full: np.ndarray | None = None,
    active_channels: Sequence[str] | None = None,
) -> None:
    """
    Save a combined single/pair/triple ablation figure with a condition matrix above each band.

    The figure centers sections with fewer columns so all image panels keep the same size.
    """
    _validate_sections(sections)

    if group_names is None:
        group_names = tuple(
            dict.fromkeys(
                group
                for section in sections
                for cond in section.conditions
                for group in cond.active_groups
            )
        )
    else:
        group_names = tuple(group_names)
    if not group_names:
        raise ValueError("could not infer any group names from sections")

    max_cols = max(len(section.conditions) for section in sections)
    n_section_rows = len(sections) * 3
    height_ratios = []
    for _ in sections:
        height_ratios.extend([0.16, 0.23, 0.61])

    fig = plt.figure(
        figsize=(max_cols * 2.25 + 0.6, len(sections) * 3.35),
        facecolor="white",
    )
    gs = gridspec.GridSpec(
        n_section_rows,
        max_cols,
        figure=fig,
        height_ratios=height_ratios,
        wspace=0.05,
        hspace=0.18,
        left=0.11,
        right=0.99,
        top=0.97,
        bottom=0.03,
    )

    cell_mask = _extract_cell_mask(ctrl_full, active_channels)

    for section_idx, section in enumerate(sections):
        row_offset = section_idx * 3
        n_conditions = len(section.conditions)
        start_col = (max_cols - n_conditions) // 2
        end_col = start_col + n_conditions

        header_ax = fig.add_subplot(gs[row_offset, start_col:end_col])
        _draw_section_header(
            header_ax,
            f"{section.title} ({n_conditions} conditions)",
            is_first_section=section_idx == 0,
        )

        matrix_ax = fig.add_subplot(gs[row_offset + 1, start_col:end_col])
        _draw_condition_matrix(
            matrix_ax,
            conditions=section.conditions,
            group_names=group_names,
        )

        for img_idx, (condition, (_, image)) in enumerate(zip(section.conditions, section.images, strict=True)):
            ax = fig.add_subplot(gs[row_offset + 2, start_col + img_idx])
            ax.set_facecolor(SECTION_BG["output"])
            vmax = None if image.dtype == np.uint8 else 1.0
            ax.imshow(image, vmin=0, vmax=vmax)
            if cell_mask is not None:
                ax.contour(cell_mask, levels=[0.5], colors=["lime"], linewidths=0.7, alpha=0.85)
            ax.set_title(
                _condition_label(condition),
                fontsize=7,
                fontweight="bold",
                color=SECTION_TEXT["output"],
                pad=4,
            )
            ax.axis("off")

    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Combined ablation matrix grid saved → {save_path}")


def load_cached_subset_ablation_sections(
    cache_dir: str | Path,
) -> tuple[tuple[str, ...], list[AblationVisSection], np.ndarray | None, list[str] | None]:
    """Load cached subset PNGs and reconstruct visualization sections."""
    from tools.stage3_ablation_cache import load_subset_condition_cache

    cache = load_subset_condition_cache(cache_dir)
    sections = [
        AblationVisSection(
            title=section["title"],
            conditions=tuple(section["conditions"]),
            images=tuple(section["images"]),
        )
        for section in cache["sections"]
    ]

    ctrl_full = None
    active_channels = None
    if cache["cell_mask"] is not None:
        ctrl_full = np.expand_dims(cache["cell_mask"], axis=0)
        active_channels = ["cell_masks"]

    return cache["group_names"], sections, ctrl_full, active_channels


def render_cached_subset_ablation_figure(
    cache_dir: str | Path,
    *,
    save_path: str | Path | None = None,
) -> Path:
    """Render the combined subset figure directly from cached PNGs."""
    cache_dir = Path(cache_dir)
    if save_path is None:
        save_path = cache_dir / "ablation_group_combinations.png"
    else:
        save_path = Path(save_path)

    group_names, sections, ctrl_full, active_channels = load_cached_subset_ablation_sections(cache_dir)
    save_condition_matrix_ablation_grid(
        sections,
        save_path,
        group_names=group_names,
        ctrl_full=ctrl_full,
        active_channels=active_channels,
    )
    return save_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render the combined Stage 3 subset ablation figure from cached PNGs",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        required=True,
        help="Single-tile cache (contains manifest.json) or parent of per-tile subdirs "
        "(each subdir with manifest.json)",
    )
    parser.add_argument(
        "--n-tiles",
        "--n-tile",
        type=int,
        default=None,
        metavar="N",
        dest="n_tiles",
        help="When --cache-dir is a parent: randomly sample N cached tiles (omit to render all)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for --n-tiles sampling (default: 42)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output PNG path (single-tile mode only; default: {cache-dir}/ablation_group_combinations.png)",
    )
    args = parser.parse_args()

    cache_path = Path(args.cache_dir)

    if is_per_tile_cache_manifest_dir(cache_path):
        if args.n_tiles is not None:
            parser.error("--n-tiles only applies when --cache-dir is a parent of per-tile caches")
        save_path = render_cached_subset_ablation_figure(
            cache_path,
            save_path=args.output,
        )
        print(f"Rendered combined subset figure → {save_path}")
        return

    if args.output is not None:
        parser.error("--output only applies in single-tile mode (manifest directly under --cache-dir)")

    try:
        cached_ids = list_cached_tile_ids(cache_path)
    except FileNotFoundError as exc:
        parser.error(str(exc))
    if not cached_ids:
        parser.error(
            f"no per-tile caches with manifest.json under {cache_path} "
            f"(expected subdirs like {cache_path}/<tile_id>/manifest.json)"
        )

    if args.n_tiles is not None:
        if args.n_tiles < 1:
            parser.error("--n-tiles must be >= 1")
        if len(cached_ids) < args.n_tiles:
            parser.error(
                f"need at least {args.n_tiles} cached tiles under {cache_path}, found {len(cached_ids)}"
            )
        random.seed(args.seed)
        to_render = random.sample(cached_ids, args.n_tiles)
    else:
        to_render = cached_ids

    for tile_id in to_render:
        tile_cache = cache_path / tile_id
        save_path = render_cached_subset_ablation_figure(tile_cache)
        print(f"Rendered combined subset figure → {save_path}")
    suffix = (
        f" (sampled {args.n_tiles} of {len(cached_ids)} cached, seed={args.seed})"
        if args.n_tiles is not None
        else ""
    )
    print(f"Done: {len(to_render)} tiles rendered{suffix}.")


if __name__ == "__main__":
    main()
