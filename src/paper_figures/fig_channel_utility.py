"""Figure 5 — Per-channel utility scatter for MX conditioning.

Crosses H&E -> MX decodability (ridge probe R²) against per-sub-channel
generative impact (ΔE under leave-one-out on the trained ControlNet) for the
9 conditioning sub-channels of the a1_concat run. Quadrant labels classify
each channel as Critical / Redundant / Skip / MX-optional for downstream
MX panel design.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DECODE_CSV = ROOT / "src" / "a1_probe_linear" / "out" / "linear_probe_results.csv"
DEFAULT_LOO_CSV = ROOT / "inference_output" / "subchannel_loo_n300" / "per_subchannel_summary.csv"
DEFAULT_OUT_PNG = ROOT / "figures" / "pngs_updated" / "09_channel_utility.png"

# Sub-channel name -> matching ridge-probe target name in linear_probe_results.csv.
DECODE_TARGET = {
    "cell_type_healthy":    "healthy_frac",
    "cell_type_cancer":     "cancer_frac",
    "cell_type_immune":     "immune_frac",
    "cell_state_prolif":    "prolif_frac",
    "cell_state_nonprolif": "nonprolif_frac",
    "cell_state_dead":      "dead_frac",
    "vasculature":          "vasculature_frac",
    "oxygen":               "oxygen_mean",
    "glucose":              "glucose_mean",
}

# Sub-channel -> ControlNet conditioning group (for color).
SUB_GROUP = {
    "cell_type_healthy": "cell_types", "cell_type_cancer": "cell_types", "cell_type_immune": "cell_types",
    "cell_state_prolif": "cell_state", "cell_state_nonprolif": "cell_state", "cell_state_dead": "cell_state",
    "vasculature": "vasculature",
    "oxygen": "microenv", "glucose": "microenv",
}

GROUP_COLORS = {
    "cell_types":  "#2a5db0",
    "cell_state":  "#b04a2a",
    "vasculature": "#2a8a4a",
    "microenv":    "#c2a83e",
}

GROUP_MARKERS = {
    "cell_types":  "o",
    "cell_state":  "s",
    "vasculature": "^",
    "microenv":    "D",
}

GROUP_LABELS = {
    "cell_types": "cell types",
    "cell_state": "cell state",
    "vasculature": "vasculature",
    "microenv": "microenv",
}

FONT_NAME = "Nimbus Sans"

# Pretty display: drop redundant prefixes, replace underscores.
PRETTY_SUB = {
    "cell_type_healthy": "Healthy",
    "cell_type_cancer": "Cancer",
    "cell_type_immune": "Immune",
    "cell_state_prolif": "Prolif",
    "cell_state_nonprolif": "NonProlif",
    "cell_state_dead": "Dead",
    "vasculature": "Vasculature",
    "oxygen": "Oxygen",
    "glucose": "Glucose",
}

R2_THRESHOLD = 0.5
DELTA_E_THRESHOLD = 2.0


def _read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8") as h:
        return list(csv.DictReader(h))


def _decode_lookup(decode_csv: Path) -> dict[str, tuple[float, float]]:
    """Return target -> (r2_mean, r2_sd) from the ridge probe CSV."""
    rows = _read_csv(decode_csv)
    out: dict[str, tuple[float, float]] = {}
    for row in rows:
        try:
            r2 = float(row["r2_mean"])
            sd = float(row.get("r2_sd", 0.0) or 0.0)
        except (KeyError, TypeError, ValueError):
            continue
        out[row["target"]] = (r2, sd)
    return out


def _loo_lookup(loo_csv: Path) -> dict[str, tuple[float, float]]:
    """Return sub_channel -> (delta_e_mean_mean, delta_e_mean_sem)."""
    rows = _read_csv(loo_csv)
    out: dict[str, tuple[float, float]] = {}
    for row in rows:
        try:
            mean_v = float(row["delta_e_mean_mean"])
            sem_v = float(row.get("delta_e_mean_sem", 0.0) or 0.0)
        except (KeyError, TypeError, ValueError):
            continue
        out[row["sub_channel"]] = (mean_v, sem_v)
    return out


def _draw_quadrants(ax: plt.Axes) -> None:
    fills = {
        ("right", "top"):   ("#f0c060", "Redundant"),    # high R², high ΔE
        ("left",  "top"):   ("#d05050", "Critical"),     # low R², high ΔE
        ("left",  "bottom"):("#a0a0a0", "Skip"),         # low R², low ΔE
        ("right", "bottom"):("#70b070", "MX optional"),  # high R², low ΔE
    }
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_mid = R2_THRESHOLD
    y_mid = DELTA_E_THRESHOLD
    rects = {
        ("right", "top"):    (x_mid, y_mid, xlim[1] - x_mid, ylim[1] - y_mid),
        ("left",  "top"):    (xlim[0], y_mid, x_mid - xlim[0], ylim[1] - y_mid),
        ("left",  "bottom"): (xlim[0], ylim[0], x_mid - xlim[0], y_mid - ylim[0]),
        ("right", "bottom"): (x_mid, ylim[0], xlim[1] - x_mid, y_mid - ylim[0]),
    }
    for key, (x, y, w, h) in rects.items():
        color, label = fills[key]
        ax.add_patch(plt.Rectangle((x, y), w, h, color=color, alpha=0.06, zorder=0))
        # Label in the far corner of each quadrant.
        if key[0] == "right":
            tx, ha = xlim[1] - 0.01, "right"
        else:
            tx, ha = xlim[0] + 0.01, "left"
        if key[1] == "top":
            ty, va = ylim[1] - 0.15, "top"
        else:
            ty, va = ylim[0] + 0.15, "bottom"
        ax.text(tx, ty, label, ha=ha, va=va, fontsize=7, color="black",
                alpha=0.95, fontweight="bold", fontfamily=FONT_NAME)

    ax.axvline(x_mid, color="#888", linestyle=":", linewidth=0.6, zorder=1)
    ax.axhline(y_mid, color="#888", linestyle=":", linewidth=0.6, zorder=1)


def draw_channel_utility(
    ax: plt.Axes,
    *,
    decode_csv: Path,
    loo_csv: Path,
) -> None:
    decode = _decode_lookup(decode_csv)
    loo = _loo_lookup(loo_csv)

    points: list[tuple[str, float, float, float, float, str]] = []
    for sub, target in DECODE_TARGET.items():
        if target not in decode or sub not in loo:
            continue
        r2, r2_sd = decode[target]
        de, de_sem = loo[sub]
        group = SUB_GROUP[sub]
        points.append((sub, r2, de, r2_sd, de_sem, group))

    if not points:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    ax.set_xlim(-0.25, 1.0)
    de_max = max(de for _, _, de, *_ in points) + 0.6
    ax.set_ylim(-0.3, max(6.0, de_max))

    _draw_quadrants(ax)

    texts: list = []
    plotted_groups: set[str] = set()
    for sub, r2, de, r2_sd, de_sem, group in points:
        color = GROUP_COLORS[group]
        marker = GROUP_MARKERS[group]
        plotted_groups.add(group)
        ax.errorbar(
            r2, de,
            xerr=r2_sd, yerr=de_sem,
            fmt=marker, markersize=5,
            color=color, ecolor=color, elinewidth=0.7, capsize=2.0,
            markerfacecolor="white", markeredgecolor=color, markeredgewidth=1.2,
            zorder=3,
        )
        txt = ax.text(
            r2, de, PRETTY_SUB.get(sub, sub),
            fontsize=6.5, color="black",
            fontfamily=FONT_NAME, zorder=4,
        )
        texts.append(txt)

    adjust_text(
        texts, ax=ax,
        expand=(1.15, 1.3),
        arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.5),
    )

    ax.set_xlabel("H&E → MX decodability (R²)", fontsize=9, fontfamily=FONT_NAME)
    ax.set_ylabel("Generative impact ΔE (LOO)", fontsize=9, fontfamily=FONT_NAME)
    ax.tick_params(labelsize=8)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontfamily(FONT_NAME)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.8)
    ax.set_axisbelow(True)
    ax.grid(True, alpha=0.2, linewidth=0.4, zorder=0)

    # Group legend below plot, hollow markers with shape per group.
    handles = [
        plt.Line2D([0], [0], marker=GROUP_MARKERS[g], linestyle="",
                   color=GROUP_COLORS[g], markerfacecolor="white",
                   markeredgecolor=GROUP_COLORS[g], markeredgewidth=1.2,
                   markersize=6, label=GROUP_LABELS[g])
        for g in GROUP_COLORS.keys()
        if g in plotted_groups
    ]
    if handles:
        ax.legend(
            handles=handles,
            loc="upper center", bbox_to_anchor=(0.5, -0.18),
            ncol=len(handles), frameon=False,
            prop={"family": FONT_NAME, "size": 7.0},
            handlelength=1.4, columnspacing=0.6, handletextpad=0.3,
            borderaxespad=0.0,
        )


def build_channel_utility_figure(
    *,
    decode_csv: Path = DEFAULT_DECODE_CSV,
    loo_csv: Path = DEFAULT_LOO_CSV,
) -> plt.Figure:
    fig = plt.figure(figsize=(3.5, 3.5), facecolor="white", constrained_layout=True)
    ax = fig.add_subplot(1, 1, 1)
    draw_channel_utility(ax, decode_csv=Path(decode_csv), loo_csv=Path(loo_csv))
    return fig


def save_channel_utility_figure(
    *,
    out_png: Path = DEFAULT_OUT_PNG,
    decode_csv: Path = DEFAULT_DECODE_CSV,
    loo_csv: Path = DEFAULT_LOO_CSV,
    dpi: int = 300,
) -> Path:
    fig = build_channel_utility_figure(decode_csv=decode_csv, loo_csv=loo_csv)
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_png


if __name__ == "__main__":
    save_channel_utility_figure()
