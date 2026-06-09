"""Combined Fig 2: training architecture + conditioning ablation + performance.

Five panels, four rows, full-width composite consistent with the other
`concat/` figures (220 DPI raster stack, ~15.8 in target width):

    Row 1:  A = training/architecture schematic (stage_2_svg.svg)
            B = conditioning-encoder ablation (SI ΔLPIPS bars)
    Row 2:  C = metric trade-offs across channel conditions (performance A)
    Row 3:  D = paired channel-group effect sizes + qualitative strip
    Row 4:  E = unpaired channel-group effect sizes + qualitative strip

Text-size consistency (vis_guidance: never rescale a rendered panel) is held by
rendering every matplotlib panel directly at its final pixel size and only ever
*resizing image rasters* (the SVG schematic and the tile strip), never the
text-bearing plots.
"""
import io
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from tools.ablation_report.data import DatasetSummary
from tools.ablation_report.figures import (
    build_channel_effect_heatmaps_figure,
    build_metric_trends_figure,
)
from matplotlib.font_manager import FontProperties

from src.paper_figures.fig_si_a1_a2_unified import (
    PRIMARY_A2_VARIANT,
    SECTION1_FONT_FAMILY,
    _draw_section4_sensitivity,
    _load_cache,
    _plot_loss_curves,
    _section1_legend_handles,
)
from src.paper_figures.fig_channel_ablation_strip import build_channel_ablation_strip_split
from src.paper_figures.style import FONT_SIZE_TICK

_RENDER_DPI = 220
_LABEL_STRIP_PX = 56
COMPOSITE_WIDTH_IN = 15.8
COMPOSITE_HEATMAP_FONT_SCALE = 1.28

_SCHEMATIC_ASPECT = 2084 / 1762          # native stage_2_svg.svg width/height
_ROW1_GAP_PX = 50                        # white gutter between panels A and B
_PANELB_TARGET_ASPECT = 1.15             # B = loss + ΔLPIPS + legend, ~square to match A
_PANELB_VARIANTS = ["production", "a1_concat", "a1_per_channel", PRIMARY_A2_VARIANT]
_PANEL_LETTER_FS = 18                    # matches FONT_PANEL_LETTER across paper figs
_DE_ROW_GAP_IN = 0.05
_DE_HEATMAP_W_IN = 6.55
_DE_QUAL_W_IN = COMPOSITE_WIDTH_IN - _DE_HEATMAP_W_IN - _DE_ROW_GAP_IN


def _fig_to_rgb(fig: plt.Figure, *, tight: bool = True) -> np.ndarray:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=_RENDER_DPI, bbox_inches="tight" if tight else None)
    buf.seek(0)
    arr = np.array(Image.open(buf).convert("RGB"))
    plt.close(fig)
    return arr


def _load_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def _svg_to_rgb_on_white(svg_path: Path, height_px: int) -> np.ndarray:
    import cairosvg

    png = cairosvg.svg2png(url=str(svg_path), output_height=height_px)
    im = Image.open(io.BytesIO(png)).convert("RGBA")
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
    bg.alpha_composite(im)
    return np.array(bg.convert("RGB"))


def _resize_to_width(img: np.ndarray, width_px: int) -> np.ndarray:
    if img.shape[1] == width_px:
        return img
    h = round(img.shape[0] * width_px / img.shape[1])
    return np.array(Image.fromarray(img).resize((width_px, h), Image.LANCZOS))


def _pad_to_width(img: np.ndarray, width_px: int) -> np.ndarray:
    """Centre a panel on a white canvas of the target width (no rescaling)."""
    if img.shape[1] >= width_px:
        return img
    pad = width_px - img.shape[1]
    left = pad // 2
    right = pad - left
    return np.pad(img, ((0, 0), (left, right), (0, 0)), constant_values=255)


def _add_label_strip(img: np.ndarray, px: int = _LABEL_STRIP_PX) -> np.ndarray:
    strip = np.full((px, img.shape[1], 3), 255, dtype=np.uint8)
    return np.concatenate([strip, img], axis=0)


def _detect_gutter_frac(img_d: np.ndarray) -> float:
    """Fraction-of-width of the white gutter between D's two heatmaps."""
    h = img_d.shape[0]
    band = img_d[int(h * 0.25):int(h * 0.75)]
    dens = (band < 245).any(axis=2).mean(axis=0)   # colored-pixel fraction per column
    w = len(dens)
    filled = dens > 0.5
    runs: list[tuple[int, int]] = []
    start = None
    for i, v in enumerate(filled):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(filled)))
    runs = [r for r in runs if r[1] - r[0] > w * 0.05]
    two = sorted(sorted(runs, key=lambda r: r[1] - r[0], reverse=True)[:2])
    return (two[0][1] + two[1][0]) / 2 / w


def _detect_strip_divider_frac(img_e: np.ndarray, lo: float = 0.46, hi: float = 0.54) -> float:
    """Fraction-of-width of the dashed paired|unpaired divider, searched only in the
    central band so the black cell-mask columns can't be mistaken for it."""
    h = img_e.shape[0]
    band = img_e[int(h * 0.15):int(h * 0.95)]
    dashes = (band < 160).all(axis=2).mean(axis=0)
    w = img_e.shape[1]
    a, b = int(w * lo), int(w * hi)
    return (a + int(np.argmax(dashes[a:b]))) / w


def _place_strip_aligned(img_e: np.ndarray, full_width: int, gutter_frac: float) -> np.ndarray:
    """Uniformly scale + right/left-pad the tile strip so its dashed divider lands on
    `gutter_frac` (D's gutter). No cropping, no aspect distortion."""
    f_e = _detect_strip_divider_frac(img_e)
    s_max = min(gutter_frac * full_width / f_e, full_width * (1 - gutter_frac) / (1 - f_e))
    s = int(min(full_width, s_max))
    scaled = _resize_to_width(img_e, s)
    off = max(0, int(round(gutter_frac * full_width - f_e * s)))
    off = min(off, full_width - s)
    canvas = np.full((scaled.shape[0], full_width, 3), 255, dtype=np.uint8)
    canvas[:, off:off + s] = scaled
    return canvas


def _build_panelB_raster(cache: dict, *, width_px: int, height_px: int) -> np.ndarray:
    """Panel B, rendered at its final pixel size: training loss (top), ΔLPIPS
    encoder-ablation bars (middle), shared variant color legend (bottom)."""
    fig = plt.figure(figsize=(width_px / _RENDER_DPI, height_px / _RENDER_DPI))
    gs = fig.add_gridspec(
        3, 1, height_ratios=[1.0, 0.92, 0.20], hspace=0.55,
        left=0.15, right=0.965, top=0.955, bottom=0.075,
    )
    curves = cache.get("training_curves", {})
    _plot_loss_curves(fig.add_subplot(gs[0]), curves, _PANELB_VARIANTS, "Training loss")
    _draw_section4_sensitivity(fig.add_subplot(gs[1]), cache)
    legend_ax = fig.add_subplot(gs[2])
    legend_ax.axis("off")
    legend_ax.legend(
        handles=_section1_legend_handles(_PANELB_VARIANTS),
        loc="center",
        ncol=2,
        frameon=False,
        prop=FontProperties(family=SECTION1_FONT_FAMILY, size=FONT_SIZE_TICK),
        handlelength=2.0,
        columnspacing=1.2,
        handletextpad=0.5,
    )
    arr = _fig_to_rgb(fig, tight=False)
    return _resize_to_width(arr, width_px)  # guard against rounding drift only


def _build_row1(svg_path: Path, cache: dict, full_width: int) -> np.ndarray:
    """Panels A (schematic) + B (loss + ΔLPIPS + legend), sharing height, summing to full_width."""
    h1 = round((full_width - _ROW1_GAP_PX) / (_PANELB_TARGET_ASPECT + _SCHEMATIC_ASPECT))
    w_a = round(_SCHEMATIC_ASPECT * h1)
    w_b = full_width - w_a - _ROW1_GAP_PX

    img_a = _svg_to_rgb_on_white(svg_path, height_px=h1)
    img_a = np.array(Image.fromarray(img_a).resize((w_a, h1), Image.LANCZOS))
    img_b = _build_panelB_raster(cache, width_px=w_b, height_px=h1)
    img_b = img_b[:h1] if img_b.shape[0] >= h1 else _pad_row_height(img_b, h1)

    gutter = np.full((h1, _ROW1_GAP_PX, 3), 255, dtype=np.uint8)
    return np.concatenate([img_a, gutter, img_b], axis=1), w_a


def _pad_row_height(img: np.ndarray, h: int) -> np.ndarray:
    pad = h - img.shape[0]
    return np.pad(img, ((0, pad), (0, 0), (0, 0)), constant_values=255)


def _pad_to_height(img: np.ndarray, height_px: int, *, valign: str = "center") -> np.ndarray:
    if img.shape[0] >= height_px:
        return img
    pad = height_px - img.shape[0]
    if valign == "top":
        top = 0
    elif valign == "bottom":
        top = pad
    else:
        top = pad // 2
    bottom = pad - top
    return np.pad(img, ((top, bottom), (0, 0), (0, 0)), constant_values=255)


def _build_dataset_de_row(
    summary: DatasetSummary,
    *,
    paired_root: Path,
    unpaired_root: Path,
    paired_orion: Path,
    unpaired_orion: Path,
    layout_root: Path,
    unpaired_mapping_json: Path,
) -> tuple[np.ndarray, int]:
    """One Fig. 2 D/E row: heatmap at left, matching split qualitative at right.

    Returns the composited row and the pixel x of the heatmap|qualitative
    boundary (left edge of the qualitative strip), used to place the (i)/(ii)
    sub-panel labels.
    """
    img_heat = _fig_to_rgb(
        build_channel_effect_heatmaps_figure(
            [summary],
            width_inches=_DE_HEATMAP_W_IN,
            font_scale=COMPOSITE_HEATMAP_FONT_SCALE,
        )
    )
    img_qual = _fig_to_rgb(
        build_channel_ablation_strip_split(
            split=summary.slug,
            paired_root=paired_root,
            unpaired_root=unpaired_root,
            paired_orion=paired_orion,
            unpaired_orion=unpaired_orion,
            layout_root=layout_root,
            unpaired_mapping_json=unpaired_mapping_json,
            target_height_in=img_heat.shape[0] / _RENDER_DPI,
            n_tiles=3,
        ),
        tight=False,
    )

    target_h = img_heat.shape[0]
    if img_qual.shape[0] != target_h:
        img_qual = _pad_to_height(img_qual, target_h, valign="center")
    gap = np.full((target_h, round(_DE_ROW_GAP_IN * _RENDER_DPI), 3), 255, dtype=np.uint8)
    divider_px = img_heat.shape[1] + gap.shape[1]
    return np.concatenate([img_heat, gap, img_qual], axis=1), divider_px


def build_combined_method_perf_figure(
    summaries: list[DatasetSummary],
    *,
    svg_path: Path,
    sensitivity_cache_path: Path,
    channel_ablation_png: Path,
    paired_root: Path | None = None,
    unpaired_root: Path | None = None,
    paired_orion: Path | None = None,
    unpaired_orion: Path | None = None,
    layout_root: Path | None = None,
    unpaired_mapping_json: Path | None = None,
) -> plt.Figure:
    # Full-width text-bearing panels rendered at native point sizes.
    img_c = _fig_to_rgb(build_metric_trends_figure(summaries, width_inches=COMPOSITE_WIDTH_IN))
    if paired_root is None:
        paired_root = channel_ablation_png.parents[3] / "inference_output" / "concat_ablation_1000" / "paired_ablation" / "ablation_results"
    if unpaired_root is None:
        unpaired_root = channel_ablation_png.parents[3] / "inference_output" / "concat_ablation_1000" / "unpaired_ablation" / "ablation_results"
    if paired_orion is None:
        paired_orion = channel_ablation_png.parents[3] / "data" / "orion-crc33"
    if unpaired_orion is None:
        unpaired_orion = paired_orion
    if layout_root is None:
        layout_root = paired_orion
    if unpaired_mapping_json is None:
        unpaired_mapping_json = channel_ablation_png.parents[3] / "inference_output" / "concat_ablation_1000" / "unpaired_ablation" / "metadata" / "unpaired_mapping.json"

    summary_by_slug = {summary.slug: summary for summary in summaries}
    img_d, div_d = _build_dataset_de_row(
        summary_by_slug["paired"],
        paired_root=paired_root,
        unpaired_root=unpaired_root,
        paired_orion=paired_orion,
        unpaired_orion=unpaired_orion,
        layout_root=layout_root,
        unpaired_mapping_json=unpaired_mapping_json,
    )
    img_e, div_e = _build_dataset_de_row(
        summary_by_slug["unpaired"],
        paired_root=paired_root,
        unpaired_root=unpaired_root,
        paired_orion=paired_orion,
        unpaired_orion=unpaired_orion,
        layout_root=layout_root,
        unpaired_mapping_json=unpaired_mapping_json,
    )

    # Common full width = widest text panel; narrower text panels are padded
    # (centred on white), never stretched, so their fonts stay consistent.
    full_width = max(img_c.shape[1], img_d.shape[1], img_e.shape[1])
    d_left = max(0, (full_width - img_d.shape[1]) // 2)
    e_left = max(0, (full_width - img_e.shape[1]) // 2)
    img_c = _pad_to_width(img_c, full_width)
    img_d = _pad_to_width(img_d, full_width)
    img_e = _pad_to_width(img_e, full_width)

    cache = _load_cache(sensitivity_cache_path)
    row1, w_a = _build_row1(svg_path, cache, full_width)

    # --- stack rows, each preceded by a white label strip ---
    rows = [row1, img_c, img_d, img_e]
    rows = [_add_label_strip(r) for r in rows]
    heights = [r.shape[0] for r in rows]

    fig_w = full_width / _RENDER_DPI
    fig_h = sum(heights) / _RENDER_DPI
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(len(rows), 1, height_ratios=heights, hspace=0)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    b_x = (w_a + _ROW1_GAP_PX) / full_width
    # D/E each split into a heatmap (i) and a qualitative-H&E (ii) sub-panel;
    # label both at their respective left edges.
    row_label_specs = [
        [("A", 0.006), ("B", b_x + 0.006)],
        [("C", 0.006)],
        [("D (i)", d_left / full_width + 0.006), ("D (ii)", (d_left + div_d) / full_width + 0.006)],
        [("E (i)", e_left / full_width + 0.006), ("E (ii)", (e_left + div_e) / full_width + 0.006)],
    ]
    for i, (img, specs) in enumerate(zip(rows, row_label_specs)):
        ax = fig.add_subplot(gs[i, 0])
        ax.imshow(img, interpolation="none")
        ax.axis("off")
        for letter, x in specs:
            ax.text(
                x,
                0.992,
                letter,
                transform=ax.transAxes,
                fontsize=_PANEL_LETTER_FS,
                fontweight="bold",
                va="top",
                ha="left",
                color="black",
            )
    return fig


def _load_summaries():
    from tools.ablation_report.data import load_dataset_summary
    from tools.ablation_report.shared import ROOT
    from tools.stage3.style_mapping import load_style_mapping

    base = ROOT / "inference_output" / "concat_ablation_1000"
    ref = ROOT / "data" / "orion-crc33"
    style = load_style_mapping(None)
    paired = load_dataset_summary(
        slug="paired", title="Paired",
        metrics_root=base / "paired_ablation" / "ablation_results",
        dataset_root=base / "paired_ablation",
        reference_root=ref, style_mapping=style,
    )
    unpaired = load_dataset_summary(
        slug="unpaired", title="Unpaired",
        metrics_root=base / "unpaired_ablation" / "ablation_results",
        dataset_root=base / "unpaired_ablation",
        reference_root=ref, style_mapping=style,
    )
    return [paired, unpaired], ROOT


if __name__ == "__main__":
    from tools.ablation_report.figures import save_figure_png
    from src.paper_figures.style import apply_style

    apply_style()
    summaries, ROOT = _load_summaries()
    fig = build_combined_method_perf_figure(
        summaries,
        svg_path=ROOT / "figures" / "pngs_updated" / "methods" / "stage_2_svg.svg",
        sensitivity_cache_path=ROOT / "inference_output" / "si_a1_a2" / "cache.json",
        channel_ablation_png=ROOT / "figures" / "pngs_updated" / "individual"
        / "channel_ablation_paired_unpaired.png",
        paired_root=ROOT / "inference_output" / "concat_ablation_1000" / "paired_ablation" / "ablation_results",
        unpaired_root=ROOT / "inference_output" / "concat_ablation_1000" / "unpaired_ablation" / "ablation_results",
        paired_orion=ROOT / "data" / "orion-crc33",
        unpaired_orion=ROOT / "data" / "orion-crc33",
        layout_root=ROOT / "data" / "orion-crc33",
        unpaired_mapping_json=ROOT / "inference_output" / "concat_ablation_1000" / "unpaired_ablation" / "metadata" / "unpaired_mapping.json",
    )
    out = ROOT / "figures" / "pngs_updated" / "concat" / "fig2_architecture_performance.png"
    save_figure_png(fig, out)
    print("wrote", out)
