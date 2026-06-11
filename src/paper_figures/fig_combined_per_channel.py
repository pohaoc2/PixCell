"""Fig 4 + SI: per-channel impact — channel-selection guide (Stage 3).

Restructured 2026-06-11: the main figure carries only the schematic and the
decodability-vs-impact guide; the raw decodability and ranked-impact panels move
to the SI.

Main ``fig4_per_channel_impact.png`` — two panels, side by side:

    ┌───────────┬──────────────────────────┐
    │ A         │ B  decodability vs impact │
    │ Stage-3   │  ┌─────────┐┌──────────┐  │
    │ schematic │  │ ΔE color││ ΔPQ layout│  │
    │           │  │  vs R²  ││   vs R²   │  │
    └───────────┴──────────────────────────┘

B is today's compact color / layout quadrant pair, stripped of its internal A/B
sub-labels (the composite supplies the single "B") with its group legend inside;
A is scaled (aspect preserved) so its height equals B's, so the two read as one
band.

SI ``si_channel_decodability_impact.png`` — a 2×2 grid: decodability on the left
(A: UNI→channel R²; B: per-protein R²), ranked per-channel generative impact on
the right (C: color ΔE; D: layout ΔPQ), real LOO metrics from the 300-tile
sub-channel ablation, ranked descending, bars colored by group.

Every panel is rendered at one shared DPI (`RENDER_DPI`) with the project's
shared point sizes, then placed at its native pixel size — never resized — so one
nominal font size renders at one physical size across panels (see
`vis_guidance.md`, Cross-panel consistency).

Run:  python -m src.paper_figures.fig_combined_per_channel
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image

from src.paper_figures.style import FONT_FAMILY, apply_style
from src.paper_figures.fig_t1_spatial_multi_encoder import (
    _R2_CAP,
    _draw_panel,
    _draw_raw_mx_panel,
    _load_encoder,
    _load_raw_mx_spatial,
)
from src.paper_figures.fig_channel_utility_spatial import (
    build_channel_utility_spatial_figure,
    PANEL_SQ_IN as F_PANEL_SQ_IN,
    MARGIN_LEFT_IN as F_M_LEFT_IN,
    MARGIN_GAP_IN as F_M_GAP_IN,
    MARGIN_RIGHT_IN as F_M_RIGHT_IN,
    MARGIN_TOP_IN as F_M_TOP_IN,
    MARGIN_XLABEL_IN as F_M_XLABEL_IN,
    MARGIN_LEGEND_IN as F_M_LEGEND_IN,
)
from src.paper_figures.fig_channel_utility import (
    GROUP_COLORS,
    GROUP_LABELS,
    PRETTY_SUB,
    SUB_GROUP,
)
from tools.ablation_report.shared import ROOT

# F's native (scale 1.0) figure size — used to pick the scale that makes F's
# width equal the shared right-column width (B/C/F share one column), and, in the
# radar variant, to size F's height to the radar.
_F_NATIVE_W_IN = F_M_LEFT_IN + F_PANEL_SQ_IN + F_M_GAP_IN + F_PANEL_SQ_IN + F_M_RIGHT_IN
_F_NATIVE_H_IN = F_M_TOP_IN + F_PANEL_SQ_IN + F_M_XLABEL_IN + F_M_LEGEND_IN

# ── Shared resolution ─────────────────────────────────────────────────────────
# 220 dpi == save_figure_png default == the dpi the cached 04 LOO panel was
# written at, so that panel can be reused at native size (no rescale) and still
# match the freshly rendered B/C/F panels point-for-point.
RENDER_DPI = 220

# ── Source paths ─────────────────────────────────────────────────────────────
_SVG_PATH = ROOT / "figures" / "pngs_updated" / "methods" / "stage_3_simple_svg.svg"
_CONCAT_DIR = ROOT / "figures" / "pngs_updated" / "concat"
_SI_DIR = ROOT / "figures" / "pngs_updated" / "si"
OUT_PNG = _CONCAT_DIR / "fig4_per_channel_impact.png"
OUT_SI_COMBINED = _SI_DIR / "si_channel_decodability_impact.png"

# Real LOO metric tables (300-tile sub-channel ablation).
_LOO_COLOR_CSV = ROOT / "inference_output" / "subchannel_loo_n300" / "per_subchannel_summary.csv"
_LOO_LAYOUT_CSV = ROOT / "inference_output" / "subchannel_loo_n300" / "per_subchannel_layout_summary.csv"

_ENCODER_CSVS: dict[str, Path] = {
    "UNI-2h": ROOT / "src/a1_probe_mlp_spatial/out/uni_16/mlp_spatial_probe_results.csv",
    "Virchow2": ROOT / "src/a1_probe_mlp_spatial/out/virchow2_16/mlp_spatial_probe_results.csv",
    "CTransPath": ROOT / "src/a1_probe_mlp_spatial/out/ctranspath_07/mlp_spatial_probe_results.csv",
    "ResNet-50": ROOT / "src/a1_probe_mlp_spatial/out/resnet50_07/mlp_spatial_probe_results.csv",
}
_T2_SPATIAL_CSV = ROOT / "src/a1_probe_mlp_spatial/out/t2_spatial/mlp_spatial_probe_results.csv"

# ── Layout (absolute inches; px derived via RENDER_DPI) ───────────────────────
# B/C/F share one right column (same vertical lines). A and D/E share the left
# column. Alignment is by the *data region* (bars / box / plot square), not the
# figure box: A's green box spans B-bars + gap + C-bars; F's plot squares span
# D-bars + gap + E-bars. The left column width is derived so this holds.
RIGHT_COL_IN = 7.2     # shared width of B, C and F
COL_GAP_IN = 0.40      # horizontal gap between the left and right columns
ROW_GAP_IN = 0.05      # vertical gap between the top and bottom blocks (tight)
LABEL_STRIP_IN = 0.34  # white strip above each block for the bold panel letters

# Top-right stack: B over C.  Figure heights and the axes (data-region) insets
# the render functions use; the gap is widened so the B+C *bar* band is tall
# enough to match the schematic's green box at a sensible A width.
B_HEIGHT_IN = 2.55
C_HEIGHT_IN = 2.70
BC_GAP_IN = 1.05       # figure gap between B and C (the tunable "hspace")
B_TOP_M_IN, B_BOT_M_IN = 0.10, 0.96   # B axes top / bottom margins
C_TOP_M_IN, C_BOT_M_IN = 0.10, 1.12   # C axes top / bottom margins

# Bottom-left stack: D over E wide-bar placeholders, their axes insets, and the
# figure gap between them (the tunable "hspace" for D/E).
PH_TOP_M_IN, PH_BOT_M_IN = 0.16, 0.52
DE_GAP_IN = 0.55

_PANEL_LETTER_FS = 18  # pt, matches the other paper composites


def _px(inches: float) -> int:
    return round(inches * RENDER_DPI)


# ── Native-render helpers ─────────────────────────────────────────────────────

def _fig_to_rgb(fig: plt.Figure, *, tight: bool = False) -> np.ndarray:
    """Rasterise a figure at RENDER_DPI. ``tight`` trims margins but does not
    rescale, so glyph pixel sizes stay identical across panels."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=RENDER_DPI, facecolor="white",
                bbox_inches="tight" if tight else None)
    buf.seek(0)
    arr = np.array(Image.open(buf).convert("RGB"))
    plt.close(fig)
    return arr


def _abs_axes(fig: plt.Figure, fig_w: float, fig_h: float,
              left: float, right: float, top: float, bottom: float) -> plt.Axes:
    """Add an axes whose margins are specified in absolute inches."""
    return fig.add_axes([
        left / fig_w,
        bottom / fig_h,
        (fig_w - left - right) / fig_w,
        (fig_h - top - bottom) / fig_h,
    ])


def _svg_to_rgb(svg_path: Path, width_px: int) -> np.ndarray:
    import cairosvg

    png = cairosvg.svg2png(url=str(svg_path), output_width=width_px)
    im = Image.open(io.BytesIO(png)).convert("RGBA")
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
    bg.alpha_composite(im)
    return np.array(bg.convert("RGB"))


def _svg_aspect(svg_path: Path) -> float:
    """Root width / height of the SVG (w:h aspect), read from its header."""
    import re

    head = svg_path.read_text(encoding="utf-8")[:2000]
    w = float(re.search(r'width="([0-9.]+)"', head).group(1))
    h = float(re.search(r'height="([0-9.]+)"', head).group(1))
    return w / h


def _green_box_vspan_frac(img: np.ndarray) -> tuple[float, float]:
    """Vertical span of the schematic's green box, as (top, bottom) fractions of
    the image height. The green rounded rectangle is the outermost green element,
    so the bounding box of all green pixels gives the box extent. Scale-invariant,
    so it can be measured on a reference render and reused at the final size."""
    r = img[..., 0].astype(int)
    g = img[..., 1].astype(int)
    b = img[..., 2].astype(int)
    mask = (g > 90) & (g > r + 25) & (g > b + 25)
    rows = np.where(mask.any(axis=1))[0]
    if rows.size == 0:
        return 0.0, 1.0
    h = img.shape[0]
    return rows.min() / h, (rows.max() + 1) / h


def _render_panel_b(width_in: float, height_in: float) -> np.ndarray:
    """B: UNI/encoder within-tile R² per TME channel (07d panel A, standalone)."""
    apply_style()
    encoder_rows = {
        name: _load_encoder(path)
        for name, path in _ENCODER_CSVS.items()
        if path.is_file()
    }
    encoder_order = [name for name in _ENCODER_CSVS if name in encoder_rows]
    uni_rows = encoder_rows["UNI-2h"]
    targets = sorted(uni_rows, key=lambda t: uni_rows[t]["r2_within_mean"], reverse=True)

    fig = plt.figure(figsize=(width_in, height_in), facecolor="white")
    ax = _abs_axes(fig, width_in, height_in, left=0.66, right=0.10,
                   top=B_TOP_M_IN, bottom=B_BOT_M_IN)
    _draw_panel(
        ax,
        targets=targets,
        encoder_rows=encoder_rows,
        encoder_order=encoder_order,
        folds_key="r2_within_folds",
        mean_key="r2_within_mean",
        ylabel="Within-tile R²",
        ylim=(_R2_CAP, 0.5),
        clip_floor=_R2_CAP,
        show_legend=True,
    )
    return _fig_to_rgb(fig)


def _render_panel_c(width_in: float, height_in: float) -> np.ndarray:
    """C: per-protein raw-MX within-tile R² (07d panel B, standalone)."""
    apply_style()
    markers = _load_raw_mx_spatial(_T2_SPATIAL_CSV)
    fig = plt.figure(figsize=(width_in, height_in), facecolor="white")
    ax = _abs_axes(fig, width_in, height_in, left=0.66, right=0.10,
                   top=C_TOP_M_IN, bottom=C_BOT_M_IN)
    _draw_raw_mx_panel(ax, markers)
    return _fig_to_rgb(fig)


def _read_loo_csv(csv_path: Path, value_col: str, sem_col: str) -> list[tuple[str, float, float]]:
    """Return [(sub_channel, value, sem), …] for the real LOO metric table,
    sorted by descending value (ranked impact)."""
    import csv as _csv

    rows: list[tuple[str, float, float]] = []
    with csv_path.open(newline="") as fh:
        for row in _csv.DictReader(fh):
            sub = row["sub_channel"]
            if sub not in SUB_GROUP:
                continue
            try:
                val = float(row[value_col])
                sem = float(row.get(sem_col, 0.0) or 0.0)
            except (KeyError, TypeError, ValueError):
                continue
            rows.append((sub, val, sem))
    rows.sort(key=lambda r: r[1], reverse=True)
    return rows


def _draw_impact_bar(ax: plt.Axes, csv_path: Path, *, value_col: str, sem_col: str,
                     ylabel: str, legend: bool) -> None:
    """Ranked per-channel impact bars, colored by group (real LOO metrics)."""
    data = _read_loo_csv(csv_path, value_col, sem_col)
    x = np.arange(len(data))
    vals = [v for _s, v, _e in data]
    sems = [e for _s, _v, e in data]
    colors = [GROUP_COLORS[SUB_GROUP[s]] for s, _v, _e in data]
    ax.bar(x, vals, width=0.72, color=colors, edgecolor="black", linewidth=0.8, zorder=2)
    ax.errorbar(x, vals, yerr=sems, fmt="none", ecolor="black", elinewidth=0.7,
                capsize=2.0, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([PRETTY_SUB[s] for s, _v, _e in data], fontfamily=FONT_FAMILY,
                       rotation=35, ha="right")
    ax.set_xlim(-0.7, len(data) - 0.3)
    ax.set_ylim(0, max(v + e for _s, v, e in data) * 1.12)
    ax.set_ylabel(ylabel, fontfamily=FONT_FAMILY)
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(0.8)
    ax.set_axisbelow(True)
    if legend:
        handles = [Line2D([0], [0], marker="s", linestyle="none", markersize=7,
                          markerfacecolor=GROUP_COLORS[g], markeredgecolor="black",
                          markeredgewidth=0.6, label=GROUP_LABELS[g])
                   for g in GROUP_COLORS]
        ax.legend(handles=handles, loc="upper right", frameon=False,
                  prop={"family": FONT_FAMILY, "size": 9}, handletextpad=0.3,
                  labelspacing=0.3)


def _render_panel_f(scale: float, *, legend: str = "bottom") -> np.ndarray:
    """F: color (ΔE) and layout (ΔPQ) impact quadrants (09b), enlarged, no A/B."""
    fig = build_channel_utility_spatial_figure(scale=scale, panel_letters=False,
                                               legend=legend)
    return _fig_to_rgb(fig)


def _render_impact_panel(csv_path: Path, *, value_col: str, sem_col: str,
                         ylabel: str, legend: bool,
                         width_in: float = RIGHT_COL_IN, height_in: float = 2.7) -> np.ndarray:
    """One ranked per-channel impact bar panel, rendered at the shared SI width so
    it stacks flush with the decodability panels (same y-axis left margin)."""
    apply_style()
    fig = plt.figure(figsize=(width_in, height_in), facecolor="white")
    ax = _abs_axes(fig, width_in, height_in, left=0.66, right=0.10, top=0.10, bottom=1.05)
    _draw_impact_bar(ax, csv_path, value_col=value_col, sem_col=sem_col,
                     ylabel=ylabel, legend=legend)
    return _fig_to_rgb(fig)


# ── Compositing ────────────────────────────────────────────────────────────────

def _letter(ax: plt.Axes, x_px: float, y_px: float, letter: str,
            full_w: int, total_h: int) -> None:
    ax.text(x_px / full_w, 1.0 - y_px / total_h, letter, transform=ax.transAxes,
            fontsize=_PANEL_LETTER_FS, fontweight="bold", fontfamily=FONT_FAMILY,
            va="center", ha="left", color="black")


def _array_figure(composite: np.ndarray) -> tuple[plt.Figure, plt.Axes]:
    """Wrap a raster composite in a full-bleed figure at RENDER_DPI."""
    h, w = composite.shape[:2]
    fig = plt.figure(figsize=(w / RENDER_DPI, h / RENDER_DPI))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(composite, interpolation="none", aspect="auto")
    ax.axis("off")
    return fig, ax


def _stack_column(panels: list[tuple[str, np.ndarray]], *, gap_in: float = 0.20
                  ) -> tuple[np.ndarray, list[tuple[str, float]]]:
    """Vertical stack of (letter, img) into one column raster (letters not drawn);
    returns the column array and each letter's (label, y_px)."""
    label_strip_px = _px(LABEL_STRIP_IN)
    gap_px = _px(gap_in)
    w = max(img.shape[1] for _l, img in panels)
    rows: list[np.ndarray] = []
    letters: list[tuple[str, float]] = []
    y = 0
    for i, (letter, img) in enumerate(panels):
        if i > 0:
            rows.append(np.full((gap_px, w, 3), 255, dtype=np.uint8)); y += gap_px
        rows.append(np.full((label_strip_px, w, 3), 255, dtype=np.uint8))
        letters.append((letter, y + label_strip_px / 2)); y += label_strip_px
        if img.shape[1] < w:
            padded = np.full((img.shape[0], w, 3), 255, dtype=np.uint8)
            padded[:, :img.shape[1]] = img; img = padded
        rows.append(img); y += img.shape[0]
    return np.concatenate(rows, axis=0), letters


def _compose_two_columns(left: list[tuple[str, np.ndarray]],
                         right: list[tuple[str, np.ndarray]], *,
                         col_gap_in: float = 0.5) -> plt.Figure:
    """Two stacked columns side by side (left A/B, right C/D), top-aligned."""
    col_l, letters_l = _stack_column(left)
    col_r, letters_r = _stack_column(right)
    gap_px = _px(col_gap_in)
    h = max(col_l.shape[0], col_r.shape[0])
    rx = col_l.shape[1] + gap_px
    full_w = rx + col_r.shape[1]
    canvas = np.full((h, full_w, 3), 255, dtype=np.uint8)
    canvas[:col_l.shape[0], :col_l.shape[1]] = col_l
    canvas[:col_r.shape[0], rx:rx + col_r.shape[1]] = col_r
    fig, ax = _array_figure(canvas)
    lm = _px(0.04)
    for letter, y_px in letters_l:
        _letter(ax, lm, y_px, letter, full_w, h)
    for letter, y_px in letters_r:
        _letter(ax, rx + lm, y_px, letter, full_w, h)
    return fig


def build_fig4_main() -> plt.Figure:
    """Fig 4: A (Stage-3 schematic) | B (compact decodability-vs-impact guide),
    side by side.  A is scaled (aspect preserved) so its height equals B's, so the
    two panels read as one band."""
    apply_style()
    aspect = _svg_aspect(_SVG_PATH)
    img_B = _render_panel_f(1.0)                         # native, compact; legend inside
    h = img_B.shape[0]
    img_A = _svg_to_rgb(_SVG_PATH, max(1, round(aspect * h)))
    if img_A.shape[0] != h:                              # absorb rounding to exact height
        fixed = np.full((h, img_A.shape[1], 3), 255, dtype=np.uint8)
        k = min(h, img_A.shape[0]); fixed[:k] = img_A[:k]; img_A = fixed

    gap_px = _px(COL_GAP_IN)
    label_strip_px = _px(LABEL_STRIP_IN)
    full_w = img_A.shape[1] + gap_px + img_B.shape[1]
    body = np.full((h, full_w, 3), 255, dtype=np.uint8)
    body[:, :img_A.shape[1]] = img_A
    bx = img_A.shape[1] + gap_px
    body[:, bx:bx + img_B.shape[1]] = img_B
    strip = np.full((label_strip_px, full_w, 3), 255, dtype=np.uint8)
    composite = np.concatenate([strip, body], axis=0)

    fig, ax = _array_figure(composite)
    lm = _px(0.04)
    _letter(ax, lm, label_strip_px / 2, "A", full_w, composite.shape[0])
    _letter(ax, bx + lm, label_strip_px / 2, "B", full_w, composite.shape[0])
    return fig


def build_si_combined() -> plt.Figure:
    """SI 2×2: decodability on the left (A: UNI→channel R²; B: per-protein R²),
    ranked per-channel generative impact on the right (C: color ΔE; D: layout ΔPQ).
    All four panels share one width and height so rows/columns align."""
    apply_style()
    panel_h = 2.7
    img_A = _render_panel_b(RIGHT_COL_IN, panel_h)
    img_B = _render_panel_c(RIGHT_COL_IN, panel_h)
    img_C = _render_impact_panel(_LOO_COLOR_CSV, value_col="delta_e_mean_mean",
                                 sem_col="delta_e_mean_sem", ylabel="Color impact  ΔE",
                                 legend=True, height_in=panel_h)
    img_D = _render_impact_panel(_LOO_LAYOUT_CSV, value_col="pq_drop_mean",
                                 sem_col="pq_drop_sem", ylabel="Layout impact  ΔPQ",
                                 legend=False, height_in=panel_h)
    return _compose_two_columns([("A", img_A), ("B", img_B)], [("C", img_C), ("D", img_D)])


def main() -> None:
    apply_style()
    for builder, out in (
        (build_fig4_main, OUT_PNG),
        (build_si_combined, OUT_SI_COMBINED),
    ):
        out.parent.mkdir(parents=True, exist_ok=True)
        fig = builder()
        fig.savefig(out, dpi=RENDER_DPI, facecolor="white")
        plt.close(fig)
        print("wrote", out)


if __name__ == "__main__":
    main()
