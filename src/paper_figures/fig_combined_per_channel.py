"""Combined Fig 4: Per-channel impact — channel-selection guide (Stage 3).

Layout (two blocks, schematic on the left):

    ┌───────────────────────────┬──────────────────────────┐
    │  A  stage_3 schematic     │  B  UNI→channel decode    │
    │  (raster, tall left col)  ├──────────────────────────┤
    │                           │  C  per-protein decode    │
    ├──────────────────────────┬┴──────────────────────────┤
    │  D  per-channel impact    │                           │
    ├──────────────────────────┤   F  color / layout        │
    │  E  per-channel impact    │      quadrants             │
    └──────────────────────────┴───────────────────────────┘

A is scaled (preserving its aspect) so its height equals the B+C stack, so the
top block reads as one band.  Because that makes A wider, B/C are *re-rendered*
at a smaller width (never horizontally squashed — see `vis_guidance.md`, "never
distort a rendered image").  D/E are wide-bar placeholders stacked in the left
column; F is the color/layout quadrant pair, enlarged slightly and stripped of
its internal A/B sub-labels (the composite supplies the single "F").

Panels are edge-aligned to the figure margins (A/D/E to the left, B/C/F to the
right) so F can be wider than B/C while every column still lines up on a figure
edge.  Every data panel is rendered **natively at one shared DPI**
(`RENDER_DPI`) with the project's shared point sizes, then placed at its native
pixel size — never resized — so one nominal font size renders at one physical
size across panels (see `vis_guidance.md`, Cross-panel consistency).

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
OUT_PNG = ROOT / "figures" / "pngs_updated" / "concat" / "fig4_per_channel_impact.png"
OUT_PNG_V2 = ROOT / "figures" / "pngs_updated" / "concat" / "fig4_per_channel_impact_v2.png"

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


def _render_placeholder_bar(width_in: float, height_in: float, *, ylabel: str,
                            seed: int) -> np.ndarray:
    """A wide-bar placeholder panel (D / E), styled like the real data panels."""
    apply_style()
    rng = np.random.default_rng(seed)
    n = 12
    vals = np.sort(rng.uniform(0.15, 1.0, n))[::-1]
    fig = plt.figure(figsize=(width_in, height_in), facecolor="white")
    ax = _abs_axes(fig, width_in, height_in, left=0.74, right=0.12,
                   top=PH_TOP_M_IN, bottom=PH_BOT_M_IN)
    x = np.arange(n)
    ax.bar(x, vals, width=0.72, color="#d7dde6", edgecolor="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"ch{i + 1}" for i in range(n)], fontfamily=FONT_FAMILY)
    ax.set_xlim(-0.7, n - 0.3)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel(ylabel, fontfamily=FONT_FAMILY)
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(0.8)
    ax.set_axisbelow(True)
    ax.text(0.5, 0.90, "Placeholder", transform=ax.transAxes, ha="center", va="top",
            fontsize=13, fontstyle="italic", color="#9aa3ad", fontfamily=FONT_FAMILY)
    return _fig_to_rgb(fig)


def _render_panel_f(scale: float) -> np.ndarray:
    """F: color (ΔE) and layout (ΔPQ) impact quadrants (09b), enlarged, no A/B."""
    fig = build_channel_utility_spatial_figure(scale=scale, panel_letters=False)
    return _fig_to_rgb(fig)


# 10 TME channels (matches panel B's x-axis order) and the two impact overlays.
_RADAR_CHANNELS = ["Prolif", "Nonprolif", "Density", "Cancer", "Healthy",
                   "Immune", "Vasc", "Glucose", "O$_2$", "Dead"]
_RADAR_COLOR_D = "#3b6fb6"   # impact D  (colorblind-safe blue)
_RADAR_COLOR_E = "#d55e00"   # impact E  (Okabe-Ito vermillion)


def _render_radar(diameter_in: float) -> np.ndarray:
    """D (v2): one radar, 10 channel spokes, both impacts overlaid as two
    translucent polygons.  Square figure so the web stays circular."""
    apply_style()
    n = len(_RADAR_CHANNELS)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    closed = angles + angles[:1]

    rng = np.random.default_rng(3)
    vals_d = rng.uniform(0.35, 1.0, n)
    vals_e = rng.uniform(0.35, 1.0, n)
    dc = np.concatenate([vals_d, vals_d[:1]])
    ec = np.concatenate([vals_e, vals_e[:1]])

    fig = plt.figure(figsize=(diameter_in, diameter_in), facecolor="white")
    # Big web: keep just enough margin for the spoke labels + the legend strip.
    ax = fig.add_axes([0.11, 0.13, 0.78, 0.78], polar=True)  # square → circular web
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles)
    ax.set_xticklabels(_RADAR_CHANNELS, fontfamily=FONT_FAMILY, fontsize=11)
    ax.tick_params(axis="x", pad=2)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=8,
                       fontfamily=FONT_FAMILY, color="#555")
    ax.set_rlabel_position(180.0 / n)
    ax.grid(color="#cccccc", linewidth=0.6)
    for spine in ax.spines.values():
        spine.set_color("#888888")
        spine.set_linewidth(0.8)

    for series, color in ((dc, _RADAR_COLOR_D), (ec, _RADAR_COLOR_E)):
        ax.plot(closed, series, color=color, linewidth=1.6, zorder=4)
        ax.fill(closed, series, color=color, alpha=0.22, zorder=3)

    handles = [
        Line2D([0], [0], color=_RADAR_COLOR_D, lw=2.4, label="Impact D"),
        Line2D([0], [0], color=_RADAR_COLOR_E, lw=2.4, label="Impact E"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.03),
              ncol=2, frameon=False, prop={"family": FONT_FAMILY, "size": 11},
              handlelength=1.4, columnspacing=1.2, handletextpad=0.4)
    fig.text(0.5, 0.5, "Placeholder", ha="center", va="center", fontsize=11,
             fontstyle="italic", color="#9aa3ad", fontfamily=FONT_FAMILY, alpha=0.7)
    return _fig_to_rgb(fig)


# ── Compositing ────────────────────────────────────────────────────────────────

def build_per_channel_figure(*, bottom_left: str = "bars") -> plt.Figure:
    """``bottom_left="bars"`` → D and E wide-bar placeholders (default fig 4).
    ``bottom_left="radar"`` → a single radar (the v2 variant): 10 channel spokes
    with both impacts overlaid as two translucent polygons."""
    aspect = _svg_aspect(_SVG_PATH)
    col_gap_px = _px(COL_GAP_IN)
    label_strip_px = _px(LABEL_STRIP_IN)
    row_gap_px = _px(ROW_GAP_IN)
    right_col_px = _px(RIGHT_COL_IN)

    # ── TOP block: schematic green box spans B-bars + gap + C-bars ────────────
    # Data-region band (from B's bar-area top to C's bar-area bottom), measured
    # in B/C figure-stack inches. The wide BC_GAP_IN is what makes this band
    # tall enough that A renders at a sensible width.
    band_top_in = B_TOP_M_IN
    band_bot_in = B_HEIGHT_IN + BC_GAP_IN + (C_HEIGHT_IN - C_BOT_M_IN)
    band_h_in = band_bot_in - band_top_in

    # Green box vspan (scale-invariant), measured on a small reference render.
    g_top_f, g_bot_f = _green_box_vspan_frac(_svg_to_rgb(_SVG_PATH, _px(6.0)))
    a_total_h_in = band_h_in / (g_bot_f - g_top_f)   # A scaled so green box == band
    left_col_in = a_total_h_in * aspect
    left_col_px = _px(left_col_in)

    img_A = _svg_to_rgb(_SVG_PATH, left_col_px)
    img_B = _render_panel_b(RIGHT_COL_IN, B_HEIGHT_IN)[:, :right_col_px]
    img_C = _render_panel_c(RIGHT_COL_IN, C_HEIGHT_IN)[:, :right_col_px]

    full_w = left_col_px + col_gap_px + right_col_px
    right_x = left_col_px + col_gap_px                 # shared left edge of B / C / F

    # Place A so its green-box top aligns with B's bar top; everything shifted so
    # the topmost element sits at the block origin.
    a_top0 = band_top_in - g_top_f * a_total_h_in
    b_top0, c_top0 = 0.0, B_HEIGHT_IN + BC_GAP_IN
    shift = -min(a_top0, b_top0, c_top0)
    a_top_px, b_top_px, c_top_px = _px(a_top0 + shift), _px(b_top0 + shift), _px(c_top0 + shift)

    top_h = max(a_top_px + img_A.shape[0], b_top_px + img_B.shape[0], c_top_px + img_C.shape[0])
    top = np.full((top_h, full_w, 3), 255, dtype=np.uint8)
    top[a_top_px:a_top_px + img_A.shape[0], :img_A.shape[1]] = img_A
    top[b_top_px:b_top_px + img_B.shape[0], right_x:right_x + img_B.shape[1]] = img_B
    top[c_top_px:c_top_px + img_C.shape[0], right_x:right_x + img_C.shape[1]] = img_C

    # ── BOTTOM block: F (right column), D/E (left column) ─────────────────────
    radar = bottom_left == "radar"

    if radar:
        # One radar replacing D+E. To fill the row with no large blank, the radar
        # (square) and F (landscape) are made the **same height** and sized so
        # radar + gap + F span the full width: a circle can't fill the wide left
        # column on its own, so F is re-rendered wider and right-aligned to soak up
        # the remaining width. Radar left-aligned, F right-aligned, both top-flush.
        full_w_in = left_col_in + COL_GAP_IN + RIGHT_COL_IN
        f_aspect = _F_NATIVE_W_IN / _F_NATIVE_H_IN
        row_h_in = (full_w_in - COL_GAP_IN) / (1.0 + f_aspect)   # radar diam == F height
        img_R = _render_radar(row_h_in)
        img_F = _render_panel_f(row_h_in * f_aspect / _F_NATIVE_W_IN)

        bot_h = max(img_R.shape[0], img_F.shape[0])
        bot = np.full((bot_h, full_w, 3), 255, dtype=np.uint8)
        bot[:img_R.shape[0], :img_R.shape[1]] = img_R                     # radar, left
        fx = full_w - img_F.shape[1]                                      # F, right-aligned
        bot[:img_F.shape[0], fx:fx + img_F.shape[1]] = img_F
        radar_right_x = fx                                                # F letter x
        vE_px = 0                                                         # unused
    else:
        s_F = RIGHT_COL_IN / _F_NATIVE_W_IN            # F width == shared column
        img_F = _render_panel_f(s_F)[:, :right_col_px]
        bot_h = img_F.shape[0]
        bot = np.full((bot_h, full_w, 3), 255, dtype=np.uint8)
        bot[:img_F.shape[0], right_x:right_x + img_F.shape[1]] = img_F
        radar_right_x = right_x                         # F letter x (shared column)

        # D/E wide-bar placeholders: their stacked bar-areas span F's plot square.
        band_top_b = F_M_TOP_IN * s_F                  # F plot-square top inset
        band_h_b = F_PANEL_SQ_IN * s_F                 # F plot-square height
        p = (band_h_b - PH_TOP_M_IN - PH_BOT_M_IN - DE_GAP_IN) / 2.0
        de_fig_h_in = p + PH_TOP_M_IN + PH_BOT_M_IN

        img_D = _render_placeholder_bar(left_col_in, de_fig_h_in,
                                        ylabel="Impact", seed=0)[:, :left_col_px]
        img_E = _render_placeholder_bar(left_col_in, de_fig_h_in,
                                        ylabel="Impact", seed=7)[:, :left_col_px]

        v_D0 = band_top_b - PH_TOP_M_IN                # align D bar top to F plot top
        v_E0 = v_D0 + de_fig_h_in + DE_GAP_IN
        shift_b = -min(0.0, v_D0, v_E0)
        vD_px, vE_px = _px(v_D0 + shift_b), _px(v_E0 + shift_b)
        bot_h = max(bot_h, vD_px + img_D.shape[0], vE_px + img_E.shape[0])
        if bot.shape[0] < bot_h:                       # grow if D/E stack is taller (rare)
            grown = np.full((bot_h, full_w, 3), 255, dtype=np.uint8)
            grown[:bot.shape[0]] = bot
            bot = grown
        bot[vD_px:vD_px + img_D.shape[0], :img_D.shape[1]] = img_D
        bot[vE_px:vE_px + img_E.shape[0], :img_E.shape[1]] = img_E

    # ── Stack blocks with label strips and a row gap ──────────────────────────
    strip = lambda: np.full((label_strip_px, full_w, 3), 255, dtype=np.uint8)
    row_gap = np.full((row_gap_px, full_w, 3), 255, dtype=np.uint8)
    composite = np.concatenate([strip(), top, row_gap, strip(), bot], axis=0)
    total_h = composite.shape[0]

    y_top_strip = label_strip_px / 2
    y_C_abs = label_strip_px + c_top_px                          # C letter at C figure top
    bot_origin = label_strip_px + top_h + row_gap_px + label_strip_px
    y_bot_strip = bot_origin - label_strip_px / 2

    # ── Render composite + bold panel letters ─────────────────────────────────
    fig = plt.figure(figsize=(full_w / RENDER_DPI, total_h / RENDER_DPI))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(composite, interpolation="none", aspect="auto")
    ax.axis("off")

    def _frac(x_px: float, y_px: float) -> tuple[float, float]:
        return x_px / full_w, 1.0 - y_px / total_h

    lm = _px(0.04)
    positions = [
        ("A", lm, y_top_strip),
        ("B", right_x + lm, y_top_strip),
        ("C", right_x + lm, y_C_abs),
        ("D", lm, y_bot_strip),
        ("F", radar_right_x + lm, y_bot_strip),
    ]
    if not radar:
        positions.append(("E", lm, bot_origin + vE_px))         # E letter at E figure top
    for letter, x_px, y_px in positions:
        xf, yf = _frac(x_px, y_px)
        ax.text(xf, yf, letter, transform=ax.transAxes,
                fontsize=_PANEL_LETTER_FS, fontweight="bold",
                fontfamily=FONT_FAMILY, va="center", ha="left", color="black")
    return fig


def main() -> None:
    apply_style()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    for out, mode in ((OUT_PNG, "bars"), (OUT_PNG_V2, "radar")):
        fig = build_per_channel_figure(bottom_left=mode)
        fig.savefig(out, dpi=RENDER_DPI, facecolor="white")
        plt.close(fig)
        print("wrote", out)


if __name__ == "__main__":
    main()
