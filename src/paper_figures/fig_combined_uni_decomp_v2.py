"""Fig 3 v2 (exploratory side-by-side layout).

Left column  = UNI/TME decomposition panels stacked vertically:
                 A (image grid) / B (metric trends) / C (effect heatmap).
Right column = UNI linear-probe overview: D (probe ΔR²) | E (specificity) on
                 top, F (alpha sweep grid, reduced to 3 attributes) below.

Both columns are rendered independently and then scaled *proportionally* to a
common height before being placed side by side (vis_guidance: proportional
scaling only, never single-axis distortion). The left column is rendered at a
higher dpi so the proportional match to the taller right column needs almost no
upscaling and a given point size stays close in physical size across columns.

Run:  python -m src.paper_figures.fig_combined_uni_decomp_v2
"""
from __future__ import annotations

import io
import tempfile
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from src.paper_figures.style import apply_style
import src.paper_figures.fig_uni_tme_decomposition as uni_decomp_fig
from src.paper_figures.fig_combined_method_perf import _load_rgb
from src.paper_figures.fig_uni_tme_decomposition import (
    DEFAULT_ORION_ROOT,
    _render_panel_a,
    _render_panel_b,
    _render_panel_c,
    _resolve_representative_tile,
)
from src.a2_decomposition.metrics import (
    DEFAULT_GENERATED_ROOT,
    DEFAULT_METRICS_ROOT,
    DEFAULT_REPRESENTATIVE_JSON,
    DEFAULT_SUMMARY_CSV,
    load_summary_csv,
)
from src.a4_uni_probe import figures as a4_figures
from src.a4_uni_probe.figures import COMBINED_DPI
from tools.ablation_report.shared import ROOT
from tools.ablation_report.figures import save_figure_png

PROBE_OUT_DIR = ROOT / "inference_output" / "a1_concat" / "a4_uni_probe"
OUT_PNG = ROOT / "figures" / "pngs_updated" / "concat" / "fig3_uni_decomposition_v2.png"

F_SWEEP_ATTRS = [
    "eccentricity_mean",
    "nuclear_area_mean",
    "nuclei_density",
    "texture_e_contrast",
    "texture_h_contrast",
    "texture_h_energy",
]

_LEFT_DPI = COMBINED_DPI
_COL_GAP_IN = 0.05        # white gutter between the two columns
_RIGHT_ROW_MIN_GAP_IN = 0.035
_PANEL_LETTER_FS = 18


def _fig_to_rgb(fig: plt.Figure, *, dpi: int) -> np.ndarray:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor="white")
    buf.seek(0)
    arr = np.array(Image.open(buf).convert("RGB"))
    plt.close(fig)
    return arr


def _resize_to_height(img: np.ndarray, height_px: int) -> np.ndarray:
    if img.shape[0] == height_px:
        return img
    w = round(img.shape[1] * height_px / img.shape[0])
    return np.array(Image.fromarray(img).resize((w, height_px), Image.LANCZOS))


def _resize_to_width(img: np.ndarray, width_px: int) -> np.ndarray:
    if img.shape[1] == width_px:
        return img
    h = round(img.shape[0] * width_px / img.shape[1])
    return np.array(Image.fromarray(img).resize((width_px, h), Image.LANCZOS))


def _first_tile_row_bottom_px(img: np.ndarray) -> int:
    """Bottom y of the first dense image-tile row in an F sweep block."""
    nonwhite = (np.abs(img.astype(np.int16) - 255) > 8).any(axis=2)
    row_density = nonwhite.mean(axis=1)
    dense = row_density > 0.08
    runs: list[tuple[int, int]] = []
    start = None
    for i, is_dense in enumerate(dense):
        if is_dense and start is None:
            start = i
        elif not is_dense and start is not None:
            if i - start > 20:
                runs.append((start, i))
            start = None
    if start is not None and len(dense) - start > 20:
        runs.append((start, len(dense)))
    # The first long dense run after the title/header text is the Ref H&E tile row.
    for start, end in runs:
        if end - start > 60:
            return end
    return max(1, round(0.32 * img.shape[0]))


def _panel_letter(fig: plt.Figure, *, fig_w: float, fig_h: float, x_in: float, y_in: float, letter: str) -> None:
    fig.text(
        x_in / fig_w,
        y_in / fig_h,
        letter,
        ha="left",
        va="top",
        fontsize=_PANEL_LETTER_FS,
        fontweight="bold",
        color="black",
    )


def build_left_column_figure(
    *,
    generated_root: Path = DEFAULT_GENERATED_ROOT,
    metrics_root: Path = DEFAULT_METRICS_ROOT,
    summary_csv: Path = DEFAULT_SUMMARY_CSV,
    representative_json: Path = DEFAULT_REPRESENTATIVE_JSON,
    orion_root: Path = DEFAULT_ORION_ROOT,
) -> plt.Figure:
    """A / B / C stacked vertically, absolute-inch layout. A's image grid, B's
    plot band and C's heatmap all share the same x-extent so width(A) ==
    width(B) == width(C)."""
    summary_csv = Path(summary_csv)
    if not summary_csv.is_file():
        raise FileNotFoundError(f"missing decomposition summary: {summary_csv}")
    summary = load_summary_csv(summary_csv)
    tile_id = _resolve_representative_tile(
        generated_root=Path(generated_root),
        metrics_root=Path(metrics_root),
        representative_json=Path(representative_json),
    )

    M_L, M_R, M_T, M_B = 0.05, 0.06, 0.09, 0.08
    # Shared left margin must fit C's longest y-label ("Interact."), not just
    # B's numeric ticks, since A/B/C share this band in the stacked layout.
    YLABEL_IN = 0.62
    HEADER_IN = 0.18
    GAP_A = 0.05
    CELL_A = 1.14
    GAP_AB = 0.38
    B_H = 1.42
    GAP_BC = 0.32
    C_XLAB = 0.46
    CBAR_GAP, CBAR_W, CBAR_LAB = 0.07, 0.11, 0.66

    PLOT_L = M_L + YLABEL_IN
    W_AB = 3 * CELL_A + 2 * GAP_A
    A_H = 2 * CELL_A + GAP_A
    CELL_C = W_AB / 5.0          # heatmap spans the shared band; square cells
    C_H = 3 * CELL_C

    fig_w = PLOT_L + W_AB + CBAR_GAP + CBAR_W + CBAR_LAB + M_R
    fig_h = M_T + HEADER_IN + A_H + GAP_AB + B_H + GAP_BC + C_H + C_XLAB + M_B

    fig = plt.figure(figsize=(fig_w, fig_h))

    a_top = fig_h - M_T - HEADER_IN
    a_bot = a_top - A_H
    b_top = a_bot - GAP_AB
    b_bot = b_top - B_H
    c_top = b_bot - GAP_BC
    c_bot = c_top - C_H

    old_font_values = {
        "FONT_SIZE_DENSE_LABEL": uni_decomp_fig.FONT_SIZE_DENSE_LABEL,
        "FONT_SIZE_DENSE_TITLE": uni_decomp_fig.FONT_SIZE_DENSE_TITLE,
        "FONT_SIZE_CELL_TEXT": uni_decomp_fig.FONT_SIZE_CELL_TEXT,
        "FONT_SIZE_LABEL": uni_decomp_fig.FONT_SIZE_LABEL,
    }
    try:
        uni_decomp_fig.FONT_SIZE_DENSE_LABEL = 9
        uni_decomp_fig.FONT_SIZE_DENSE_TITLE = 9
        uni_decomp_fig.FONT_SIZE_CELL_TEXT = 10
        uni_decomp_fig.FONT_SIZE_LABEL = 12

        _render_panel_a(
            fig, fig_w=fig_w, fig_h=fig_h, x0=PLOT_L, y_top=a_top, cell=CELL_A, gap=GAP_A,
            generated_root=Path(generated_root), orion_root=Path(orion_root), tile_id=tile_id,
        )

        b_gs = fig.add_gridspec(
            1, 1, left=PLOT_L / fig_w, right=(PLOT_L + W_AB) / fig_w,
            bottom=b_bot / fig_h, top=b_top / fig_h,
        )
        _render_panel_b(fig, b_gs[0, 0], summary, draw_label=False)

        ax_c = fig.add_axes([PLOT_L / fig_w, c_bot / fig_h, W_AB / fig_w, C_H / fig_h])
        cax_c = fig.add_axes(
            [(PLOT_L + W_AB + CBAR_GAP) / fig_w, c_bot / fig_h, CBAR_W / fig_w, C_H / fig_h]
        )
        _render_panel_c(fig, ax_c, cax_c, summary)

        # Panel letters share the left edge (same column) and match D/E/F size.
        _panel_letter(fig, fig_w=fig_w, fig_h=fig_h, x_in=M_L, y_in=a_top + HEADER_IN, letter="A")
        _panel_letter(fig, fig_w=fig_w, fig_h=fig_h, x_in=M_L, y_in=b_top + 0.16, letter="B")
        _panel_letter(fig, fig_w=fig_w, fig_h=fig_h, x_in=M_L, y_in=c_top + 0.16, letter="C")
    finally:
        for name, value in old_font_values.items():
            setattr(uni_decomp_fig, name, value)
    fig._pixcell_c_heatmap_bottom_frac = 1.0 - (c_bot / fig_h)  # type: ignore[attr-defined]
    return fig


def _render_right_column_DEF() -> tuple[np.ndarray, int]:
    """D (probe ΔR²) | E (specificity) over F (six-attribute sweep grid)."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        old_values = {
            "PANEL_SQUARE_IN": a4_figures.PANEL_SQUARE_IN,
            "COMBINED_GAP_IN": a4_figures.COMBINED_GAP_IN,
            "FONT_AXIS_LABEL": a4_figures.FONT_AXIS_LABEL,
            "FONT_TICK": a4_figures.FONT_TICK,
            "FONT_INAXES": a4_figures.FONT_INAXES,
            "FONT_HEATMAP_ANNOT": a4_figures.FONT_HEATMAP_ANNOT,
            "FONT_LEGEND": a4_figures.FONT_LEGEND,
            "FONT_PANEL_LETTER": a4_figures.FONT_PANEL_LETTER,
        }
        try:
            a4_figures.PANEL_SQUARE_IN = 3.0
            a4_figures.COMBINED_GAP_IN = 0.30
            a4_figures.FONT_AXIS_LABEL = 12
            a4_figures.FONT_TICK = 10
            a4_figures.FONT_INAXES = 9
            a4_figures.FONT_HEATMAP_ANNOT = 10
            a4_figures.FONT_LEGEND = 10
            a4_figures.FONT_PANEL_LETTER = 18
            a4_figures.render_pngs_updated_probe_delta(PROBE_OUT_DIR, tmp_path)
            a4_figures.render_pngs_updated_specificity_heatmap(PROBE_OUT_DIR, tmp_path)
            row_gap_in = 0.42
            img_d = np.array(Image.open(tmp_path / "probe_delta_r2.png").convert("RGBA"))
            img_e = np.array(Image.open(tmp_path / "specificity_heatmap.png").convert("RGBA"))
            f_start_y = max(img_d.shape[0], img_e.shape[0]) + round(row_gap_in * COMBINED_DPI)
            out = a4_figures.render_pngs_updated_combined_abc(
                PROBE_OUT_DIR,
                tmp_path,
                concat_dir=tmp,
                panel_letters=("D", "E", "F"),
                sweep_attrs=F_SWEEP_ATTRS,
                out_name="uni_probe_overview_def_v2.png",
                row_gap_in=row_gap_in,
            )
        finally:
            for name, value in old_values.items():
                setattr(a4_figures, name, value)
        return _load_rgb(Path(out)), f_start_y


def build_uni_decomp_v2_figure() -> plt.Figure:
    left_fig = build_left_column_figure()
    left = _fig_to_rgb(left_fig, dpi=_LEFT_DPI)
    right, f_start_y = _render_right_column_DEF()

    # Split between D/E and F, keeping the right column at native dpi so text is
    # not raster-shrunk relative to A/B/C.
    height = left.shape[0]
    if right.shape[0] > height:
        old_h = right.shape[0]
        right = _resize_to_height(right, height)
        f_start_y = round(f_start_y * right.shape[0] / old_h)

    min_right_gap_px = round(_RIGHT_ROW_MIN_GAP_IN * COMBINED_DPI)
    right_canvas_h = height
    if 0 < f_start_y < right.shape[0]:
        top = right[:f_start_y]
        bottom = right[f_start_y:]
        # Align A/D/E by pinning the right top block to the same y-origin as A.
        # F is then placed close underneath D/E; do not move F to satisfy C.
        top_y = 0
        y0 = top.shape[0] + min_right_gap_px
        right_canvas_h = max(height, y0 + bottom.shape[0])
        right_canvas = np.full((right_canvas_h, right.shape[1], 3), 255, dtype=np.uint8)
        # Draw F first, then D/E on top. If the E x-label slightly overlaps F,
        # the label remains visible instead of being covered by the sweep tiles.
        right_canvas[y0:y0 + bottom.shape[0], :bottom.shape[1]] = bottom
        right_canvas[top_y:top_y + top.shape[0], :top.shape[1]] = top
    else:
        y0 = height - right.shape[0]
        right_canvas = np.full((right_canvas_h, right.shape[1], 3), 255, dtype=np.uint8)
        right_canvas[y0:y0 + right.shape[0], :right.shape[1]] = right
    right = right_canvas

    gap_px = round(_COL_GAP_IN * COMBINED_DPI)
    if left.shape[0] < right.shape[0]:
        left = np.pad(left, ((0, right.shape[0] - left.shape[0]), (0, 0), (0, 0)), constant_values=255)
    elif right.shape[0] < left.shape[0]:
        right = np.pad(right, ((0, left.shape[0] - right.shape[0]), (0, 0), (0, 0)), constant_values=255)

    gutter = np.full((left.shape[0], gap_px, 3), 255, dtype=np.uint8)
    composite = np.concatenate([left, gutter, right], axis=1)

    fig = plt.figure(figsize=(composite.shape[1] / COMBINED_DPI, composite.shape[0] / COMBINED_DPI))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(composite, interpolation="none")
    ax.axis("off")
    return fig


def main() -> None:
    apply_style()
    fig = build_uni_decomp_v2_figure()
    save_figure_png(fig, OUT_PNG)
    print("wrote", OUT_PNG)


if __name__ == "__main__":
    main()
