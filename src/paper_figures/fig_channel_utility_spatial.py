"""SI Figure 09b — Spatial decodability vs color and layout effects.

Panel A shows per-subchannel color impact under leave-one-out. Panel B reuses
the same x-axis and point identities, but swaps the y-axis to per-subchannel
layout impact measured as PQ drop. No probe refit is required; panel B is a
metric pass over the existing cached LOO generations.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
from adjustText import adjust_text

from src.paper_figures.fig_channel_utility import (
    DECODE_TARGET,
    DELTA_E_THRESHOLD,
    FONT_NAME,
    GROUP_COLORS,
    GROUP_LABELS,
    GROUP_MARKERS,
    PRETTY_SUB,
    SUB_GROUP,
    _loo_lookup,
    _save_figure_png,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPATIAL_CSV = ROOT / "src" / "a1_probe_mlp_spatial" / "out" / "uni_16" / "mlp_spatial_probe_results.csv"
DEFAULT_LOO_CSV = ROOT / "inference_output" / "subchannel_loo_n300" / "per_subchannel_summary.csv"
DEFAULT_LAYOUT_CSV = ROOT / "inference_output" / "subchannel_loo_n300" / "per_subchannel_layout_summary.csv"
DEFAULT_OUT_PNG = ROOT / "figures" / "pngs_updated" / "concat" / "09b_channel_color_layout_impact.png"

TITLE_SIZE = 10.5
AXIS_LABEL_SIZE = 9.5
TICK_SIZE = 8.5
POINT_LABEL_SIZE = 7.0
QUADRANT_LABEL_SIZE = 7.5
LEGEND_SIZE = 7.5
# Vertical half-spacing of the circled-number cluster, as a fraction of the
# panel's y-range (keeps circles from overlapping at any y-scale).
QUADRANT_CLUSTER_DY_FRAC = 0.055

R2_THRESHOLD = 0.0
# Boundary between the main right pane and the compressed left pane.
BREAK_AT = -1.0
LEFT_XLIM = (-8.5, -1.5)
RIGHT_XLIM = (-1.0, 1.0)
COLOR_Y_LIM = (-0.3, 6.0)
LAYOUT_THRESHOLD = 0.05
LAYOUT_Y_MIN = -0.02

# --- Compact, square panel geometry (absolute inches) ---------------------
# Each panel's data box (compressed left pane + main right pane) is a
# PANEL_SQ_IN square; both panels share the same height. Margins are tight.
PANEL_SQ_IN = 2.05
PANE_LEFT_RATIO = 1.0      # width share of the compressed (R² < -1) pane
PANE_RIGHT_RATIO = 3.6     # width share of the main pane
MARGIN_LEFT_IN = 0.56      # y-axis label + tick labels (panel A)
MARGIN_GAP_IN = 0.66       # between panel A and panel B (room for B's y-label)
MARGIN_RIGHT_IN = 0.08
MARGIN_TOP_IN = 0.22       # panel letters
MARGIN_XLABEL_IN = 0.44    # x tick labels + "Within-tile R²"
MARGIN_LEGEND_IN = 0.32    # legend row
# One decimal everywhere (x and y, both panels). ΔPQ is small, so 1-dp y-ticks
# are necessarily coarse (0.0, 0.1); the 0.05 threshold is shown by the dotted line.
LAYOUT_Y_TICKS = [0.0, 0.1]

# Manual label anchors for panel B (absolute data coords + ha), tuned to the
# real marker positions. Healthy is kept in its own (bottom-left) quadrant per
# request; the rest are routed to clear space with leader lines. Left-pane
# labels (compressed axis) are stacked above/below/left of their markers.
LAYOUT_LABEL_XY = {
    # left pane (R² < -1)
    "cell_state_dead": (-7.48, -0.013, "center"),
    "oxygen": (-4.94, 0.030, "center"),
    "glucose": (-3.25, 0.066, "right"),
    # right pane
    "cell_state_nonprolif": (0.34, 0.095, "left"),
    "cell_type_cancer": (0.36, 0.026, "left"),
    "cell_state_prolif": (0.46, 0.002, "left"),
    "cell_type_healthy": (-0.35, 0.041, "right"),
    "cell_type_immune": (-0.30, -0.007, "center"),
    "vasculature": (-0.55, 0.022, "center"),
}

def _spatial_lookup(spatial_csv: Path) -> dict[str, tuple[float, float]]:
    """target -> (r2_within_mean, r2_within_sd)."""
    out: dict[str, tuple[float, float]] = {}
    with Path(spatial_csv).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                mean_v = float(row["r2_within_mean"])
                sd_v = float(row.get("r2_within_sd", 0.0) or 0.0)
            except (KeyError, TypeError, ValueError):
                continue
            out[row["target"]] = (mean_v, sd_v)
    return out


def _fill_quadrants(ax_left: plt.Axes, ax_right: plt.Axes, *, y_mid: float, y_lim: tuple[float, float]) -> None:
    """Draw quadrant crosshair lines (no background tints)."""

    x_mid = R2_THRESHOLD
    ax_right.axvline(x_mid, color="#888", linestyle=":", linewidth=0.6, zorder=1)
    for axis in (ax_left, ax_right):
        axis.axhline(y_mid, color="#888", linestyle=":", linewidth=0.6, zorder=1)


def _place_quadrant_labels(
    ax_right: plt.Axes,
    *,
    y_mid: float,
    y_lim: tuple[float, float],
) -> list:
    """Mark the four quadrants with circled numbers (1-4, reading order), in
    BOTH panels, clustered tightly around the crosshair (x=0, y=y_mid).

    Names (Critical / Redundant / Skip / MX optional) move to the figure caption.
    1 = top-left, 2 = top-right, 3 = bottom-left, 4 = bottom-right. Offsets are
    the smallest that keep the four circles from overlapping each other.
    Returns the circle Text artists so label placement can route around them.
    """
    x_mid = R2_THRESHOLD
    dx = 0.13                                   # x is shared across panels
    dy = QUADRANT_CLUSTER_DY_FRAC * (y_lim[1] - y_lim[0])
    spec = [
        ("1", x_mid - dx, y_mid + dy),
        ("2", x_mid + dx, y_mid + dy),
        ("3", x_mid - dx, y_mid - dy),
        ("4", x_mid + dx, y_mid - dy),
    ]
    artists = []
    for label, x, y in spec:
        artists.append(ax_right.text(
            x, y, label, ha="center", va="center",
            fontsize=QUADRANT_LABEL_SIZE, color="black",
            fontfamily=FONT_NAME, zorder=6,
            bbox=dict(boxstyle="circle,pad=0.20", facecolor="white",
                      edgecolor="black", linewidth=0.8),
        ))
    return artists


def _layout_lookup(layout_csv: Path) -> dict[str, tuple[float, float]]:
    out: dict[str, tuple[float, float]] = {}
    if not Path(layout_csv).is_file():
        return out
    with Path(layout_csv).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                mean_v = float(row["pq_drop_mean"])
                sem_v = float(row.get("pq_drop_sem", 0.0) or 0.0)
            except (KeyError, TypeError, ValueError):
                continue
            out[row["sub_channel"]] = (mean_v, sem_v)
    return out


def _draw_break_marks(fig: plt.Figure, ax_left: plt.Axes, ax_right: plt.Axes) -> None:
    """Draw explicit `//` break marks anchored on the broken left spine.

    Using the left axis transform keeps both marks on the actual axis line
    instead of drifting into the seam or outer margin after export.
    """
    x_anchor = 0.985
    dx = 0.028
    dy = 0.030
    separation = 0.040
    style = dict(
        transform=ax_left.transAxes,
        color="black",
        linewidth=0.9,
        solid_capstyle="round",
        clip_on=False,
        zorder=7,
    )

    for y_center in (0.0, 1.0):
        for offset in (-separation / 2.0, separation / 2.0):
            ax_left.add_line(
                Line2D(
                    [x_anchor + offset - dx, x_anchor + offset + dx],
                    [y_center - dy, y_center + dy],
                    **style,
                )
            )



def _plot_point(ax: plt.Axes, *, x: float, y: float, r2_sd: float, y_sem: float,
                color: str, marker: str) -> None:
    # Clamp the horizontal (R²) whisker to the visible pane so a huge std (e.g.
    # Dead: ±9.8, far wider than the axis) is capped at the border instead of
    # running off the figure. The cap is drawn at the clamped end, signalling the
    # whisker continues beyond the axis.
    x_lo, x_hi = ax.get_xlim()
    margin = 0.04 * (x_hi - x_lo)
    left_err = x - max(x - r2_sd, x_lo + margin)
    right_err = min(x + r2_sd, x_hi - margin) - x
    container = ax.errorbar(
        x, y, xerr=[[left_err], [right_err]], yerr=y_sem,
        fmt=marker, markersize=5,
        color=color, ecolor=color, elinewidth=0.7, capsize=2.0,
        markerfacecolor="white", markeredgecolor=color, markeredgewidth=1.2,
        zorder=3,
    )
    # Don't let the axes clip a marker whose vertex pokes a hair past the pane
    # edge (e.g. Glucose sits at R²≈-2.02, right against the left pane's -1.5 edge,
    # so its right diamond corner was being clipped). The whisker is already
    # clamped above, so nothing escapes the figure.
    for artist in container.get_children():
        artist.set_clip_on(False)


def _draw_panel(
    fig: plt.Figure,
    *,
    left_rect: tuple[float, float, float, float],
    right_rect: tuple[float, float, float, float],
    points: list[tuple[str, float, float, float, float, str]],
    y_label: str,
    y_mid: float,
    y_lim: tuple[float, float],
    panel_kind: str,
    tick_decimals: int,
    y_ticks: list[float] | None = None,
) -> tuple[plt.Axes, plt.Axes, set[str]]:
    ax_left = fig.add_axes(left_rect)
    ax_right = fig.add_axes(right_rect, sharey=ax_left)

    ax_left.set_xlim(*LEFT_XLIM)
    ax_right.set_xlim(*RIGHT_XLIM)
    ax_left.set_ylim(*y_lim)
    ax_left.set_facecolor("none")
    ax_right.set_facecolor("none")

    _fill_quadrants(ax_left, ax_right, y_mid=y_mid, y_lim=y_lim)
    circle_artists = _place_quadrant_labels(ax_right, y_mid=y_mid, y_lim=y_lim)

    # shrinkB keeps the leader from terminating on the marker itself (it was
    # covering the diamond/triangle vertices).
    _leader = dict(arrowstyle="-", color="#aaaaaa", lw=0.5, shrinkB=5)
    texts_left: list = []
    texts_right: list = []
    plotted_groups: set[str] = set()
    for sub, r2, y_value, r2_sd, y_sem, group in points:
        color = GROUP_COLORS[group]
        marker = GROUP_MARKERS[group]
        plotted_groups.add(group)
        ax = ax_left if r2 < BREAK_AT else ax_right
        _plot_point(ax, x=r2, y=y_value, r2_sd=r2_sd, y_sem=y_sem, color=color, marker=marker)

        if panel_kind == "layout":
            # Manual placement with curated anchors (the compressed left pane and
            # the central circled numbers leave no room for automatic layout).
            # Left-pane labels sit adjacent to their marker, so no leader (a short
            # leader would just clip the marker's vertices).
            lx, ly, lha = LAYOUT_LABEL_XY.get(sub, (r2 + 0.04, y_value, "left"))
            ax.annotate(PRETTY_SUB.get(sub, sub), xy=(r2, y_value),
                        xytext=(lx, ly), textcoords="data",
                        ha=lha, va="center", fontsize=POINT_LABEL_SIZE,
                        color="black", fontfamily=FONT_NAME, zorder=4,
                        arrowprops=None if r2 < BREAK_AT else _leader)
        else:
            bucket = texts_left if r2 < BREAK_AT else texts_right
            bucket.append(ax.text(r2, y_value, PRETTY_SUB.get(sub, sub),
                                  fontsize=POINT_LABEL_SIZE, color="black",
                                  fontfamily=FONT_NAME, zorder=4))

    # Panel A (color): adjust_text routes labels clear of markers AND circles.
    if texts_left:
        adjust_text(texts_left, ax=ax_left, expand=(1.1, 1.3), arrowprops=_leader)
    if texts_right:
        adjust_text(texts_right, ax=ax_right, objects=circle_artists,
                    force_text=(0.5, 1.0), force_static=(0.6, 1.1),
                    force_explode=(0.3, 0.6), expand=(1.5, 2.0),
                    max_move=40, iter_lim=600, arrowprops=_leader)

    ax_left.spines["right"].set_visible(False)
    ax_right.spines["left"].set_visible(False)
    ax_right.tick_params(axis="y", which="both", left=False, labelleft=False)
    for axis in (ax_left, ax_right):
        for spine in ("top", "bottom", "left", "right"):
            if axis.spines[spine].get_visible():
                axis.spines[spine].set_color("black")
                axis.spines[spine].set_linewidth(0.8)
        axis.tick_params(labelsize=TICK_SIZE)
        for lbl in axis.get_xticklabels() + axis.get_yticklabels():
            lbl.set_fontfamily(FONT_NAME)
        axis.set_axisbelow(True)

    _draw_break_marks(fig, ax_left, ax_right)
    ax_left.set_xticks([-5])
    # Omit the tick at the break (-1.0): it would collide with the left pane's
    # tick across the narrow seam.
    ax_right.set_xticks([-0.5, 0.0, 0.5, 1.0])
    if y_ticks is not None:
        ax_left.set_yticks(y_ticks)

    # X and Y tick labels use the SAME number of decimal places (see vis_guidance.md).
    fmt = FormatStrFormatter(f"%.{tick_decimals}f")
    for axis in (ax_left, ax_right):
        axis.xaxis.set_major_formatter(fmt)
    ax_left.yaxis.set_major_formatter(fmt)

    ax_left.set_ylabel(y_label, fontsize=AXIS_LABEL_SIZE, fontfamily=FONT_NAME, labelpad=4)
    return ax_left, ax_right, plotted_groups


def draw_channel_utility_spatial(
    fig: plt.Figure,
    *,
    spatial_csv: Path,
    loo_csv: Path,
    left_rect: tuple[float, float, float, float],
    right_rect: tuple[float, float, float, float],
) -> tuple[plt.Axes, plt.Axes, set[str]]:
    spatial = _spatial_lookup(spatial_csv)
    loo = _loo_lookup(loo_csv)

    points = []
    for sub, target in DECODE_TARGET.items():
        if target not in spatial or sub not in loo:
            continue
        r2, r2_sd = spatial[target]
        impact, impact_sem = loo[sub]
        points.append((sub, r2, impact, r2_sd, impact_sem, SUB_GROUP[sub]))

    return _draw_panel(
        fig,
        left_rect=left_rect,
        right_rect=right_rect,
        points=points,
        y_label=r"Color impact ($\Delta E$)",
        y_mid=DELTA_E_THRESHOLD,
        y_lim=COLOR_Y_LIM,
        panel_kind="color",
        tick_decimals=1,
    )


def draw_channel_utility_layout(
    fig: plt.Figure,
    *,
    spatial_csv: Path,
    layout_csv: Path,
    left_rect: tuple[float, float, float, float],
    right_rect: tuple[float, float, float, float],
) -> tuple[plt.Axes, plt.Axes, set[str]] | None:
    spatial = _spatial_lookup(spatial_csv)
    layout = _layout_lookup(layout_csv)
    if not layout:
        return None

    points = []
    max_y = 0.0
    for sub, target in DECODE_TARGET.items():
        if target not in spatial or sub not in layout:
            continue
        r2, r2_sd = spatial[target]
        pq_drop, pq_sem = layout[sub]
        pq_drop = abs(pq_drop)
        max_y = max(max_y, pq_drop + pq_sem)
        points.append((sub, r2, pq_drop, r2_sd, pq_sem, SUB_GROUP[sub]))

    y_lim = (LAYOUT_Y_MIN, max(0.17, max_y + 0.02))
    return _draw_panel(
        fig,
        left_rect=left_rect,
        right_rect=right_rect,
        points=points,
        y_label=r"Layout impact ($\Delta$PQ)",
        y_mid=LAYOUT_THRESHOLD,
        y_lim=y_lim,
        panel_kind="layout",
        tick_decimals=1,
        y_ticks=LAYOUT_Y_TICKS,
    )


def build_channel_utility_spatial_figure(
    *,
    spatial_csv: Path = DEFAULT_SPATIAL_CSV,
    loo_csv: Path = DEFAULT_LOO_CSV,
    layout_csv: Path = DEFAULT_LAYOUT_CSV,
) -> plt.Figure:
    # Absolute-inch geometry → square, compact panels (see PANEL_* constants).
    bottom_margin_in = MARGIN_XLABEL_IN + MARGIN_LEGEND_IN
    fig_w = MARGIN_LEFT_IN + PANEL_SQ_IN + MARGIN_GAP_IN + PANEL_SQ_IN + MARGIN_RIGHT_IN
    fig_h = MARGIN_TOP_IN + PANEL_SQ_IN + bottom_margin_in
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")

    pane_tot = PANE_LEFT_RATIO + PANE_RIGHT_RATIO
    left_w_in = PANEL_SQ_IN * PANE_LEFT_RATIO / pane_tot
    right_w_in = PANEL_SQ_IN * PANE_RIGHT_RATIO / pane_tot
    y0_in = bottom_margin_in

    def _rects(x0_in: float):
        left = (x0_in / fig_w, y0_in / fig_h, left_w_in / fig_w, PANEL_SQ_IN / fig_h)
        right = ((x0_in + left_w_in) / fig_w, y0_in / fig_h, right_w_in / fig_w, PANEL_SQ_IN / fig_h)
        return left, right

    a_x0_in = MARGIN_LEFT_IN
    b_x0_in = MARGIN_LEFT_IN + PANEL_SQ_IN + MARGIN_GAP_IN
    lr_a, rr_a = _rects(a_x0_in)
    lr_b, rr_b = _rects(b_x0_in)

    ax_left_a, ax_right_a, plotted_groups = draw_channel_utility_spatial(
        fig,
        spatial_csv=Path(spatial_csv),
        loo_csv=Path(loo_csv),
        left_rect=lr_a,
        right_rect=rr_a,
    )

    panel_b = draw_channel_utility_layout(
        fig,
        spatial_csv=Path(spatial_csv),
        layout_csv=Path(layout_csv),
        left_rect=lr_b,
        right_rect=rr_b,
    )
    if panel_b is None:
        placeholder = fig.add_axes([lr_b[0], lr_b[1], (rr_b[0] + rr_b[2]) - lr_b[0], lr_b[3]])
        placeholder.axis("off")
        placeholder.text(
            0.5, 0.55,
            "PQ-drop summary not computed yet.\nCompute CellViT sidecars +\nper_subchannel_layout_summary.csv.",
            ha="center", va="center", fontsize=TICK_SIZE, fontfamily=FONT_NAME,
        )
    else:
        _ax_left_b, _ax_right_b, plotted_groups_b = panel_b
        plotted_groups |= plotted_groups_b

    # Panel letters (figure coords) and x-axis label, centered under each panel.
    letter_y = (y0_in + PANEL_SQ_IN + 0.03) / fig_h
    xlabel_y = (MARGIN_LEGEND_IN + 0.10) / fig_h
    for x0_in, letter in ((a_x0_in, "A"), (b_x0_in, "B")):
        fig.text((x0_in - 0.48) / fig_w, letter_y, letter, ha="left", va="bottom",
                 fontsize=TITLE_SIZE, fontweight="bold", fontfamily=FONT_NAME)
        fig.text((x0_in + PANEL_SQ_IN / 2.0) / fig_w, xlabel_y, "Within-tile R²",
                 ha="center", va="center", fontsize=AXIS_LABEL_SIZE, fontfamily=FONT_NAME)

    handles = [
        plt.Line2D([0], [0], marker=GROUP_MARKERS[g], linestyle="",
                   color=GROUP_COLORS[g], markerfacecolor="white",
                   markeredgecolor=GROUP_COLORS[g], markeredgewidth=1.2,
                   markersize=6, label=GROUP_LABELS[g])
        for g in GROUP_COLORS.keys()
        if g in plotted_groups
    ]
    if handles:
        fig.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.005),
            ncol=len(handles),
            frameon=False,
            prop={"family": FONT_NAME, "size": LEGEND_SIZE},
            handlelength=1.4,
            columnspacing=0.6,
            handletextpad=0.3,
            borderaxespad=0.0,
        )
    return fig


def save_channel_utility_spatial_figure(
    *,
    out_png: Path = DEFAULT_OUT_PNG,
    spatial_csv: Path = DEFAULT_SPATIAL_CSV,
    loo_csv: Path = DEFAULT_LOO_CSV,
    layout_csv: Path = DEFAULT_LAYOUT_CSV,
    dpi: int = 300,
) -> Path:
    fig = build_channel_utility_spatial_figure(spatial_csv=spatial_csv, loo_csv=loo_csv, layout_csv=layout_csv)
    return _save_figure_png(fig, Path(out_png), dpi=dpi)


if __name__ == "__main__":
    save_channel_utility_spatial_figure()
