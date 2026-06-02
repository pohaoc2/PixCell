"""Compact channel-ablation strip: paired | unpaired, single-channel columns.

Two randomly chosen tiles (one per row) are shown across both halves. Each half has
seven columns: Real H&E (reference), Layout (input cell mask), generated H&E with all
four channels, then the four single-channel ablations (CT, CS, Vas, NUT). Channel usage
is shown with the four-circle indicator used by the ablation grid; the reference H&E
carries a gray cell-layout contour and each generated H&E a red CellViT contour. A dashed
vertical line splits paired (left) from unpaired (right). Width spans the full page.

Regenerable from ``build_channel_ablation_strip``.
"""
from __future__ import annotations

import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.paper_figures.style import FONT_SIZE_DENSE_TITLE, apply_style
from tools.stage3.ablation_vis_utils import (
    FOUR_GROUP_ORDER,
    condition_metric_key,
    default_orion_he_png_path,
    draw_image_border,
)
from tools.stage3.ablation_cache import load_manifest
from tools.cellvit.contours import overlay_cellvit_contours
from tools.stage3.style_mapping import load_style_mapping

# --- Column definition (per half) -------------------------------------------
# kind: "ref" (real H&E), "layout" (input cell mask), "all"/<group> (generated).
COLUMNS: list[tuple[str, str]] = [
    ("ref", "Real H&E"),
    ("layout", "Layout"),
    ("all", "All"),
    *[(g, lbl) for g, lbl in zip(FOUR_GROUP_ORDER, ("CT", "CS", "Vas", "NUT"))],
]
N_COLS_PER_HALF = len(COLUMNS)
ALL4_KEY = condition_metric_key(tuple(FOUR_GROUP_ORDER))

_DIVIDER_COLOR = "#000000"
_GRAY_CONTOUR = "#666666"
_REF_BORDER_GRAY = "#888888"


def _manifest_image_lookup(cache_dir: Path) -> dict[str, Path]:
    """Map ``condition_metric_key`` -> absolute generated-image path for a tile cache."""
    manifest = load_manifest(cache_dir)
    lookup: dict[str, Path] = {}
    for section in manifest["sections"]:
        for entry in section["entries"]:
            key = condition_metric_key(tuple(entry["active_groups"]))
            lookup[key] = cache_dir / entry["image_path"]
    return lookup


def _layout_mask_path(layout_root: Path, tile_id: str) -> Path | None:
    """Input cell-mask path under ``exp_channels/cell_masks/<tile>.png`` (shared input)."""
    p = layout_root / "exp_channels" / "cell_masks" / f"{tile_id}.png"
    return p if p.is_file() else None


def _half_image_paths(
    cache_root: Path,
    orion_root: Path,
    layout_root: Path,
    tile_id: str,
    *,
    style_mapping: dict[str, str] | None,
) -> list[Path] | None:
    """Seven image paths for one tile in one half, or None if anything is missing."""
    cache_dir = cache_root / tile_id
    if not (cache_dir / "manifest.json").is_file():
        return None
    he_path = default_orion_he_png_path(orion_root, tile_id, style_mapping=style_mapping)
    mask_path = _layout_mask_path(layout_root, tile_id)
    if he_path is None or mask_path is None:
        return None
    lookup = _manifest_image_lookup(cache_dir)
    paths: list[Path | None] = []
    for kind, _ in COLUMNS:
        if kind == "ref":
            paths.append(he_path)
        elif kind == "layout":
            paths.append(mask_path)
        elif kind == "all":
            paths.append(lookup.get(ALL4_KEY))
        else:
            paths.append(lookup.get(condition_metric_key((kind,))))
    if any(p is None or not Path(p).is_file() for p in paths):
        return None
    return [Path(p) for p in paths]


def _pick_tiles(
    paired_root: Path,
    unpaired_root: Path,
    paired_orion: Path,
    unpaired_orion: Path,
    layout_root: Path,
    unpaired_mapping: dict[str, str],
    *,
    n_tiles: int = 3,
    seed: int = 1,
    min_fg: float = 0.06,
) -> list[str]:
    """Random tile ids resolvable (all images) in BOTH halves, reproducible by seed.

    Tiles whose input cell-layout foreground fraction is below *min_fg* are skipped so
    every row shows visible cellular structure rather than a near-empty layout.
    """
    candidates = sorted(
        d.name for d in paired_root.iterdir()
        if (d / "manifest.json").is_file() and (unpaired_root / d.name / "manifest.json").is_file()
    )
    rng = random.Random(seed)
    rng.shuffle(candidates)
    picked: list[str] = []
    for tile_id in candidates:
        mask_path = _layout_mask_path(layout_root, tile_id)
        if mask_path is None:
            continue
        fg = float((np.asarray(Image.open(mask_path).convert("L"), dtype=np.float32) / 255.0 > 0.5).mean())
        if fg < min_fg:
            continue
        paired = _half_image_paths(
            paired_root, paired_orion, layout_root, tile_id, style_mapping=None
        )
        unpaired = _half_image_paths(
            unpaired_root, unpaired_orion, layout_root, tile_id, style_mapping=unpaired_mapping
        )
        if paired is not None and unpaired is not None:
            picked.append(tile_id)
        if len(picked) == n_tiles:
            break
    if len(picked) < n_tiles:
        raise RuntimeError(f"Only found {len(picked)} fully-resolvable tiles; need {n_tiles}.")
    return picked


def _load_mask01(path: Path, hw: tuple[int, int]) -> np.ndarray:
    """Load a cell-mask PNG as float [0,1], resized to (H, W)."""
    m = Image.open(path).convert("L").resize((hw[1], hw[0]), Image.BILINEAR)
    return np.asarray(m, dtype=np.float32) / 255.0


def _gray_layout_contour(ax, layout_mask: Path, hw: tuple[int, int]) -> None:
    """Overlay the input cell-layout (target) contour in gray."""
    ax.contour(
        _load_mask01(layout_mask, hw), levels=[0.5],
        colors=[_GRAY_CONTOUR], linewidths=0.7, alpha=0.9,
    )


def _draw_cell(ax, kind: str, path: Path, *, layout_mask: Path, is_unpaired: bool) -> None:
    """Render one grid cell with the appropriate contour overlay and border style."""
    if kind == "layout":
        # White = cells, black = background.
        mask = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
        ax.imshow(mask, cmap="gray", interpolation="none", vmin=0.0, vmax=1.0)
    else:
        img = np.asarray(Image.open(path).convert("RGB"))
        ax.imshow(img, interpolation="none")
        h, w = img.shape[:2]
        if kind == "ref":
            # Gray target-layout contour on the paired GT H&E only (the unpaired
            # reference is a style tile, so its layout would not align).
            if not is_unpaired:
                _gray_layout_contour(ax, layout_mask, (h, w))
        else:
            # Every generated H&E: thin red CellViT detection, then the gray target
            # layout on top so the target stays visible where the two overlap.
            overlay_cellvit_contours(ax, path, color="red", linewidth=0.35, alpha=0.9)
            _gray_layout_contour(ax, layout_mask, (h, w))

    ax.set_xticks([])
    ax.set_yticks([])
    # Reference H&E: gray dashed border. Generated H&E: thin black border. Layout: none.
    for spine in ax.spines.values():
        spine.set_visible(False)
    if kind == "ref":
        draw_image_border(ax, _REF_BORDER_GRAY, dashed=True, linewidth=0.9)
    elif kind != "layout":
        draw_image_border(ax, "#000000", linewidth=0.5)


_DOT_LABELS = ("CT", "CS", "Vas", "NUT")


def _draw_header(
    overlay,
    kind: str,
    label: str,
    *,
    x0_in: float,
    cell_in: float,
    dots_y_in: float,
    names_y_in: float,
    show_group_labels: bool,
) -> None:
    """Draw one column header on an inch-coordinate overlay axis.

    Vertical positions are absolute inch offsets (``dots_y_in`` above the first image
    row, ``names_y_in`` above the dots), so moving labels closer is a literal offset
    and does not drift with figure height.
    """
    cx = x0_in + cell_in / 2.0
    if kind in ("ref", "layout"):
        overlay.text(cx, dots_y_in, label, ha="center", va="center", fontsize=FONT_SIZE_DENSE_TITLE)
        return
    active = set(FOUR_GROUP_ORDER) if kind == "all" else {kind}
    xs = x0_in + cell_in * np.linspace(0.16, 0.84, 4)
    for x, g, name in zip(xs, FOUR_GROUP_ORDER, _DOT_LABELS):
        overlay.scatter(
            [x], [dots_y_in], s=36, marker="o",
            c=["black" if g in active else "white"], edgecolors=["black"],
            linewidths=0.8, zorder=3, clip_on=False,
        )
        if show_group_labels:
            overlay.text(x, names_y_in, name, ha="center", va="bottom", fontsize=6.0, color="black")


def build_channel_ablation_strip(
    *,
    paired_root: Path,
    unpaired_root: Path,
    paired_orion: Path,
    unpaired_orion: Path,
    layout_root: Path,
    unpaired_mapping_json: Path,
    seed: int = 1,
) -> plt.Figure:
    """Build the 2-row paired|unpaired single-channel ablation strip figure."""
    apply_style()
    unpaired_mapping = load_style_mapping(unpaired_mapping_json)
    tiles = _pick_tiles(
        paired_root, unpaired_root, paired_orion, unpaired_orion, layout_root,
        unpaired_mapping, seed=seed,
    )

    # grid[row] = [paired_paths, unpaired_paths]; layout mask path per (row, half).
    grid: list[list[list[Path]]] = []
    layout_masks: list[list[Path]] = []
    for tile_id in tiles:
        paired = _half_image_paths(
            paired_root, paired_orion, layout_root, tile_id, style_mapping=None
        )
        unpaired = _half_image_paths(
            unpaired_root, unpaired_orion, layout_root, tile_id, style_mapping=unpaired_mapping
        )
        grid.append([paired, unpaired])
        lm = _layout_mask_path(layout_root, tile_id)
        layout_masks.append([lm, lm])

    n_rows = len(tiles)

    # --- Absolute inch layout so the H&E gap is identical horizontally and vertically
    # (wspace == hspace == GAP_IN) by construction, regardless of figure size. ---
    CELL_IN = 0.95          # square H&E / layout cell
    GAP_IN = 0.05           # equal gap between adjacent cells (rows AND columns)
    DIVIDER_GAP_IN = 0.22   # central gap that holds the dashed divider
    # Header spacing as absolute inch offsets above the first image row (NOT fractions,
    # so they don't drift with figure height): dots sit DOT_OFF_IN above the top image
    # row, channel names NAME_OFF_IN above the dots.
    DOT_OFF_IN = 0.07
    NAME_OFF_IN = 0.07
    HEADER_IN = DOT_OFF_IN + NAME_OFF_IN + 0.08  # reserve room for the name text
    M_L, M_R, M_T, M_B = 0.05, 0.05, 0.03, 0.05  # figure margins

    half_w = N_COLS_PER_HALF * CELL_IN + (N_COLS_PER_HALF - 1) * GAP_IN
    fig_w = M_L + 2 * half_w + DIVIDER_GAP_IN + M_R
    fig_h = M_T + HEADER_IN + n_rows * CELL_IN + (n_rows - 1) * GAP_IN + M_B

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")

    def _col_x0(half: int, c: int) -> float:
        base = M_L + half * (half_w + DIVIDER_GAP_IN)
        return base + c * (CELL_IN + GAP_IN)

    images_top = fig_h - M_T - HEADER_IN  # top edge of the first image row

    def _row_y0(r: int) -> float:
        return images_top - r * (CELL_IN + GAP_IN) - CELL_IN

    def _add_ax(x0: float, y0: float, w: float, h: float) -> plt.Axes:
        return fig.add_axes([x0 / fig_w, y0 / fig_h, w / fig_w, h / fig_h])

    # Inch-coordinate overlay axis for the header (dots + labels), so vertical offsets
    # from the image row are exact inches.
    overlay = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    overlay.set_xlim(0.0, fig_w)
    overlay.set_ylim(0.0, fig_h)
    overlay.axis("off")
    overlay.set_facecolor("none")
    overlay.patch.set_alpha(0.0)
    overlay.set_zorder(5)

    dots_y_in = images_top + DOT_OFF_IN
    names_y_in = dots_y_in + NAME_OFF_IN
    for half in range(2):
        for c, (kind, label) in enumerate(COLUMNS):
            _draw_header(
                overlay, kind, label,
                x0_in=_col_x0(half, c), cell_in=CELL_IN,
                dots_y_in=dots_y_in, names_y_in=names_y_in,
                show_group_labels=(half == 0 and kind == "all"),
            )

    # Image rows.
    for r in range(n_rows):
        for half in range(2):
            paths = grid[r][half]
            for c, (kind, _) in enumerate(COLUMNS):
                ax = _add_ax(_col_x0(half, c), _row_y0(r), CELL_IN, CELL_IN)
                _draw_cell(
                    ax, kind, paths[c],
                    layout_mask=layout_masks[r][half], is_unpaired=(half == 1),
                )

    # Dashed vertical divider centred in the central gap.
    x_split = (M_L + half_w + DIVIDER_GAP_IN / 2.0) / fig_w
    fig.add_artist(
        plt.Line2D(
            [x_split, x_split], [M_B / fig_h, 1.0 - M_T / fig_h],
            transform=fig.transFigure,
            color=_DIVIDER_COLOR, linewidth=1.1, linestyle=(0, (6, 4)),
        )
    )

    return fig
