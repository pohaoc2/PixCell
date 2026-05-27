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
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
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

R2_THRESHOLD = 0.0
# Boundary between the main right pane and the compressed left pane.
BREAK_AT = -1.0
LEFT_XLIM = (-8.5, -1.5)
RIGHT_XLIM = (-1.0, 1.0)
COLOR_Y_LIM = (-0.3, 6.0)
LAYOUT_THRESHOLD = 0.05
LAYOUT_Y_MIN = -0.02

LAYOUT_LABEL_OFFSETS = {
    "cell_type_healthy": (0.015, 0.008),
    "cell_type_cancer": (0.015, 0.003),
    "cell_type_immune": (0.015, -0.003),
    "cell_state_prolif": (0.020, 0.003),
    "cell_state_nonprolif": (0.020, 0.002),
    "cell_state_dead": (0.020, -0.003),
    "vasculature": (0.020, -0.004),
    "oxygen": (-0.040, 0.005),
    "glucose": (-0.040, 0.005),
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
    panel_kind: str,
) -> None:
    """Restore the original quadrant naming at the split crosshair."""
    if panel_kind == "layout":
        return

    x_mid = R2_THRESHOLD
    dx = 0.014
    dy = 0.16
    spec = [
        ("Critical", x_mid - dx, y_mid + dy, "right", "bottom"),
        ("Redundant", x_mid + dx, y_mid + dy, "left", "bottom"),
        ("Skip", x_mid - dx, y_mid - dy, "right", "top"),
        ("MX optional", x_mid + dx, y_mid - dy, "left", "top"),
    ]
    for label, x, y, ha, va in spec:
        ax_right.text(x, y, label, ha=ha, va=va, fontsize=QUADRANT_LABEL_SIZE, color="black",
                      alpha=0.95, fontweight="bold", fontfamily=FONT_NAME, zorder=5,
                      linespacing=0.95)


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


def _fill_break_seam(
    fig: plt.Figure,
    ax_left: plt.Axes,
    ax_right: plt.Axes,
    *,
    y_mid: float,
    y_lim: tuple[float, float],
) -> None:
    """No-op: quadrant tinting removed; seam stays figure-background."""
    return


def _plot_point(ax: plt.Axes, *, x: float, y: float, r2_sd: float, y_sem: float,
                color: str, marker: str) -> None:
    ax.errorbar(
        x, y, xerr=r2_sd, yerr=y_sem,
        fmt=marker, markersize=5,
        color=color, ecolor=color, elinewidth=0.7, capsize=2.0,
        markerfacecolor="white", markeredgecolor=color, markeredgewidth=1.2,
        zorder=3,
    )


def _annotate_layout_label(ax: plt.Axes, *, sub: str, x: float, y: float) -> None:
    dx, dy = LAYOUT_LABEL_OFFSETS.get(sub, (0.015, 0.004))
    ha = "left" if dx >= 0 else "right"
    ax.annotate(
        PRETTY_SUB.get(sub, sub),
        xy=(x, y),
        xytext=(x + dx, y + dy),
        textcoords="data",
        ha=ha,
        va="center",
        fontsize=POINT_LABEL_SIZE,
        color="black",
        fontfamily=FONT_NAME,
        arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.5),
        zorder=4,
    )


def _draw_panel(
    fig: plt.Figure,
    *,
    subplot_spec,
    points: list[tuple[str, float, float, float, float, str]],
    y_label: str,
    y_mid: float,
    y_lim: tuple[float, float],
    panel_kind: str,
) -> tuple[plt.Axes, plt.Axes, set[str]]:
    gs = GridSpecFromSubplotSpec(1, 2, subplot_spec=subplot_spec, width_ratios=[1.0, 3.6], wspace=0.0)
    ax_left = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[0, 1], sharey=ax_left)

    ax_left.set_xlim(*LEFT_XLIM)
    ax_right.set_xlim(*RIGHT_XLIM)
    ax_left.set_ylim(*y_lim)
    ax_left.set_facecolor("none")
    ax_right.set_facecolor("none")

    _fill_quadrants(ax_left, ax_right, y_mid=y_mid, y_lim=y_lim)
    _place_quadrant_labels(ax_right, y_mid=y_mid, panel_kind=panel_kind)

    texts_left: list = []
    texts_right: list = []
    plotted_groups: set[str] = set()
    for sub, r2, y_value, r2_sd, y_sem, group in points:
        color = GROUP_COLORS[group]
        marker = GROUP_MARKERS[group]
        plotted_groups.add(group)
        if r2 < BREAK_AT:
            _plot_point(ax_left, x=r2, y=y_value, r2_sd=r2_sd, y_sem=y_sem, color=color, marker=marker)
            if panel_kind == "layout":
                _annotate_layout_label(ax_left, sub=sub, x=r2, y=y_value)
            else:
                txt = ax_left.text(r2, y_value, PRETTY_SUB.get(sub, sub), fontsize=POINT_LABEL_SIZE, color="black", fontfamily=FONT_NAME, zorder=4)
                texts_left.append(txt)
        else:
            _plot_point(ax_right, x=r2, y=y_value, r2_sd=r2_sd, y_sem=y_sem, color=color, marker=marker)
            if panel_kind == "layout":
                _annotate_layout_label(ax_right, sub=sub, x=r2, y=y_value)
            else:
                txt = ax_right.text(r2, y_value, PRETTY_SUB.get(sub, sub), fontsize=POINT_LABEL_SIZE, color="black", fontfamily=FONT_NAME, zorder=4)
                texts_right.append(txt)

    if panel_kind != "layout" and texts_left:
        adjust_text(texts_left, ax=ax_left, expand=(1.1, 1.3), arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.5))
    if panel_kind != "layout" and texts_right:
        adjust_text(texts_right, ax=ax_right, expand=(1.15, 1.3), arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.5))

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

    _fill_break_seam(fig, ax_left, ax_right, y_mid=y_mid, y_lim=y_lim)
    _draw_break_marks(fig, ax_left, ax_right)
    ax_left.set_xticks([-5])
    ax_right.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax_left.set_ylabel(y_label, fontsize=AXIS_LABEL_SIZE, fontfamily=FONT_NAME, labelpad=4)
    return ax_left, ax_right, plotted_groups


def draw_channel_utility_spatial(
    fig: plt.Figure,
    *,
    spatial_csv: Path,
    loo_csv: Path,
    subplot_spec,
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
        subplot_spec=subplot_spec,
        points=points,
        y_label=r"Color impact ($\Delta E$)",
        y_mid=DELTA_E_THRESHOLD,
        y_lim=COLOR_Y_LIM,
        panel_kind="color",
    )


def draw_channel_utility_layout(
    fig: plt.Figure,
    *,
    spatial_csv: Path,
    layout_csv: Path,
    subplot_spec,
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

    y_lim = (LAYOUT_Y_MIN, max(0.18, max_y + 0.02))
    return _draw_panel(
        fig,
        subplot_spec=subplot_spec,
        points=points,
        y_label=r"Layout impact ($\Delta$PQ)",
        y_mid=LAYOUT_THRESHOLD,
        y_lim=y_lim,
        panel_kind="layout",
    )


def build_channel_utility_spatial_figure(
    *,
    spatial_csv: Path = DEFAULT_SPATIAL_CSV,
    loo_csv: Path = DEFAULT_LOO_CSV,
    layout_csv: Path = DEFAULT_LAYOUT_CSV,
) -> plt.Figure:
    fig = plt.figure(figsize=(6.7, 3.55), facecolor="white")
    outer = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.0, 1.0],
        left=0.07,
        right=0.98,
        bottom=0.28,
        top=0.93,
        wspace=0.36,
    )
    ax_left_a, ax_right_a, plotted_groups = draw_channel_utility_spatial(
        fig,
        spatial_csv=Path(spatial_csv),
        loo_csv=Path(loo_csv),
        subplot_spec=outer[0, 0],
    )
    ax_left_a.text(-0.26, 1.03, "A", transform=ax_left_a.transAxes,
                   ha="left", va="bottom", fontsize=TITLE_SIZE, fontweight="bold",
                   fontfamily=FONT_NAME)

    panel_b = draw_channel_utility_layout(
        fig,
        spatial_csv=Path(spatial_csv),
        layout_csv=Path(layout_csv),
        subplot_spec=outer[0, 1],
    )
    if panel_b is None:
        placeholder = fig.add_subplot(outer[0, 1])
        placeholder.axis("off")
        placeholder.text(
            0.5,
            0.55,
            "PQ-drop summary not computed yet.\nCompute CellViT sidecars +\nper_subchannel_layout_summary.csv.",
            ha="center",
            va="center",
            fontsize=TICK_SIZE,
            fontfamily=FONT_NAME,
        )
        panel_axes = [(ax_left_a, ax_right_a)]
    else:
        ax_left_b, ax_right_b, plotted_groups_b = panel_b
        plotted_groups |= plotted_groups_b
        panel_axes = [(ax_left_a, ax_right_a), (ax_left_b, ax_right_b)]
        ax_left_b.text(-0.26, 1.03, "B", transform=ax_left_b.transAxes,
                       ha="left", va="bottom", fontsize=TITLE_SIZE, fontweight="bold",
                       fontfamily=FONT_NAME)

    for left_ax, right_ax in panel_axes:
        panel_left = left_ax.get_position().x0
        panel_right = right_ax.get_position().x1
        panel_bottom = min(left_ax.get_position().y0, right_ax.get_position().y0)
        panel_center = 0.5 * (panel_left + panel_right)
        fig.text(
            panel_center,
            max(0.01, panel_bottom - 0.095),
            "Patch-level R²",
            ha="center",
            va="bottom",
            fontsize=AXIS_LABEL_SIZE,
            fontfamily=FONT_NAME,
        )

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
            loc="lower right",
            bbox_to_anchor=(0.985, 0.095),
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
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_png


if __name__ == "__main__":
    save_channel_utility_spatial_figure()
