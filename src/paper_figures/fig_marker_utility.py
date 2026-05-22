"""Figure 09b — Per-marker utility scatter.

x: H&E → marker decodability R² (ridge probe, tile-level T2 mean intensity).
y: predicted per-marker generative impact ΔE (kmeans-leverage × per-channel LOO ΔE).

Quadrants split on median R² and median predicted ΔE (data-driven, not heuristic).
Compared to fig 09 (derived sub-channels), this resolves into ~14 raw markers — the
13 cell-type kmeans markers + Ki67 (via threshold pathway).
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from adjustText import adjust_text


ROOT = Path(__file__).resolve().parents[2]
PROBE_CSV = ROOT / "src" / "a1_codex_targets" / "probe_out" / "t2_linear" / "linear_probe_results.csv"
PREDICTED_PER_TILE_CSV = ROOT / "src" / "a5_marker_leverage" / "out" / "predicted_delta_e_per_tile.csv"
DEFAULT_OUT_PNG = ROOT / "figures" / "pngs_updated" / "09b_marker_utility.png"

# Marker → group + role. Group drives shape/color (matches vis_guidance.md / fig 09).
MARKER_GROUP = {
    # Cell-type kmeans markers
    "Pan-CK":     "cell_types",
    "E-cadherin": "cell_types",
    "CD45":       "cell_types",
    "CD3e":       "cell_types",
    "CD4":        "cell_types",
    "CD45RO":     "cell_types",
    "CD8a":       "cell_types",
    "FOXP3":      "cell_types",
    "CD20":       "cell_types",
    "CD68":       "cell_types",
    "CD163":      "cell_types",
    "CD31":       "vasculature",   # endothelial cluster maps to healthy, but biology is vasc
    "SMA":        "vasculature",
    # Ki67 threshold pathway
    "Ki67":       "cell_state",
}

GROUP_COLORS = {
    "cell_types":  "#2a5db0",
    "cell_state":  "#b04a2a",
    "vasculature": "#2a8a4a",
}
GROUP_MARKERS = {
    "cell_types":  "o",
    "cell_state":  "s",
    "vasculature": "^",
}
FONT_NAME = "Nimbus Sans"


def _load_probe_r2(path: Path) -> dict[str, tuple[float, float]]:
    out: dict[str, tuple[float, float]] = {}
    with path.open() as h:
        for row in csv.DictReader(h):
            try:
                r2 = float(row["r2_mean"])
                sd = float(row.get("r2_sd", 0.0) or 0.0)
            except (KeyError, TypeError, ValueError):
                continue
            out[row["target"]] = (r2, sd)
    return out


def _load_predicted(path: Path) -> pd.DataFrame:
    """Return marker × (mean, sem) aggregated across tiles."""
    df = pd.read_csv(path)
    return df.groupby("marker")["predicted_delta_e"].agg(["mean", "sem"]).reset_index()


def build_figure(*, probe_csv: Path = PROBE_CSV, predicted_csv: Path = PREDICTED_PER_TILE_CSV) -> plt.Figure:
    probe = _load_probe_r2(probe_csv)
    pred = _load_predicted(predicted_csv)

    rows: list[dict] = []
    for marker, group in MARKER_GROUP.items():
        if marker not in probe:
            continue
        r2, r2_sd = probe[marker]
        row = pred[pred["marker"] == marker]
        if row.empty:
            continue
        y_mean = float(row["mean"].iloc[0])
        y_sem = float(row["sem"].iloc[0]) if pd.notna(row["sem"].iloc[0]) else 0.0
        rows.append(dict(marker=marker, group=group, r2=r2, r2_sd=r2_sd, y=y_mean, y_sem=y_sem))

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("no marker points to plot")

    # Clip very-negative R²s for visual readability; mark them as truncated.
    R2_MIN = -2.5
    df["r2_clip"] = df["r2"].clip(lower=R2_MIN)
    df["truncated"] = df["r2"] < R2_MIN

    # Median splits for quadrant lines (computed on clipped values so cuts match plot).
    r2_mid = float(df["r2_clip"].median())
    y_mid = float(df["y"].median())

    fig = plt.figure(figsize=(3.6, 3.6), facecolor="white", constrained_layout=True)
    ax = fig.add_subplot(1, 1, 1)

    ax.axvline(r2_mid, color="#888", linestyle=":", linewidth=0.6, zorder=1)
    ax.axhline(y_mid, color="#888", linestyle=":", linewidth=0.6, zorder=1)

    texts = []
    plotted_groups: set[str] = set()
    for _, r in df.iterrows():
        g = r["group"]
        color = GROUP_COLORS[g]
        marker = GROUP_MARKERS[g]
        plotted_groups.add(g)
        # If truncated, clip xerr so it doesn't extend off-axis to the left.
        xerr_use = min(r["r2_sd"], (r["r2_clip"] - R2_MIN)) if r["truncated"] else r["r2_sd"]
        ax.errorbar(
            r["r2_clip"], r["y"],
            xerr=xerr_use, yerr=r["y_sem"],
            fmt=marker, markersize=5,
            color=color, ecolor=color, elinewidth=0.7, capsize=2.0,
            markerfacecolor="white", markeredgecolor=color, markeredgewidth=1.2,
            zorder=3,
        )
        # Truncated markers get a leftward arrow hint.
        if r["truncated"]:
            ax.annotate("", xy=(R2_MIN - 0.05, r["y"]), xytext=(R2_MIN + 0.15, r["y"]),
                        arrowprops=dict(arrowstyle="->", color=color, lw=0.8), zorder=3)
        label = r["marker"] + (f" (R²={r['r2']:.1f})" if r["truncated"] else "")
        texts.append(ax.text(
            r["r2_clip"], r["y"], label,
            fontsize=6.5, color="black", fontfamily=FONT_NAME, zorder=4,
        ))

    adjust_text(
        texts, ax=ax,
        expand=(1.15, 1.3),
        arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.5),
    )

    # Quadrant text in corners, black.
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    labels = {
        ("right", "top"):    ("Redundant", xlim[1], ylim[1]),
        ("left",  "top"):    ("Critical",  xlim[0], ylim[1]),
        ("left",  "bottom"): ("Skip",      xlim[0], ylim[0]),
        ("right", "bottom"): ("MX optional", xlim[1], ylim[0]),
    }
    for (hx, hy), (lbl, x, y) in labels.items():
        ha = "right" if hx == "right" else "left"
        va = "top" if hy == "top" else "bottom"
        dx = -0.005 * (xlim[1] - xlim[0]) if hx == "right" else 0.005 * (xlim[1] - xlim[0])
        dy = -0.02 * (ylim[1] - ylim[0]) if hy == "top" else 0.02 * (ylim[1] - ylim[0])
        ax.text(x + dx, y + dy, lbl, ha=ha, va=va, fontsize=7,
                color="black", fontweight="bold", fontfamily=FONT_NAME, zorder=2)

    # Axes/style.
    ax.set_xlabel("H&E → marker decodability (R²)", fontsize=9, fontfamily=FONT_NAME)
    ax.set_ylabel("Predicted generative impact ΔE", fontsize=9, fontfamily=FONT_NAME)
    ax.tick_params(labelsize=8)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontfamily(FONT_NAME)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.8)
    ax.set_axisbelow(True)
    ax.grid(True, alpha=0.2, linewidth=0.4, zorder=0)

    # Legend below.
    handles = [
        plt.Line2D([0], [0], marker=GROUP_MARKERS[g], linestyle="",
                   color=GROUP_COLORS[g], markerfacecolor="white",
                   markeredgecolor=GROUP_COLORS[g], markeredgewidth=1.2,
                   markersize=6, label=g)
        for g in GROUP_COLORS if g in plotted_groups
    ]
    ax.legend(
        handles=handles,
        loc="upper center", bbox_to_anchor=(0.5, -0.18),
        ncol=len(handles), frameon=False,
        prop={"family": FONT_NAME, "size": 7.0},
        handlelength=1.4, columnspacing=0.6, handletextpad=0.3,
        borderaxespad=0.0,
    )
    return fig


def save(out_png: Path = DEFAULT_OUT_PNG, dpi: int = 300) -> Path:
    fig = build_figure()
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_png


if __name__ == "__main__":
    save()
