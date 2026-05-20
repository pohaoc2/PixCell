"""Panel A: stacked variance-partition bars, one per metric."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.paper_figures.fig_combinatorial_grammar_panels._shared import load_variance_partition

_SEG_ORDER = ("anchor", "state", "o2", "gluc", "interactions", "resid")
_SEG_COLORS = {
    "anchor":       "#6b7280",
    "state":        "#2a5db0",
    "o2":           "#2a8a4a",
    "gluc":         "#c2a83e",
    "interactions": "#b04a2a",
    "resid":        "#cccccc",
}
_INTERACTION_KEYS = ("s_x_o", "s_x_g", "o_x_g", "s_x_o_x_g")


def _collapse(row: dict[str, float]) -> dict[str, float]:
    interactions = sum(float(row[key]) for key in _INTERACTION_KEYS)
    return {
        "anchor":       float(row["anchor"]),
        "state":        float(row["state"]),
        "o2":           float(row["o2"]),
        "gluc":         float(row["gluc"]),
        "interactions": interactions,
        "resid":        float(row["resid"]),
    }


def draw_variance_bars(ax: plt.Axes, variance_csv: Path) -> None:
    rows = load_variance_partition(variance_csv)
    if not rows:
        ax.text(0.5, 0.5, "no variance data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    collapsed = [(row["metric"], _collapse(row)) for row in rows]
    collapsed.sort(key=lambda pair: pair[1]["interactions"], reverse=True)

    metrics = [name for name, _ in collapsed]
    bar_data = np.array([[shares[k] for k in _SEG_ORDER] for _, shares in collapsed])
    cumulative = np.zeros(len(metrics), dtype=np.float64)
    y_positions = np.arange(len(metrics))[::-1]

    for col_index, seg_name in enumerate(_SEG_ORDER):
        widths = bar_data[:, col_index]
        ax.barh(
            y_positions, widths, left=cumulative,
            color=_SEG_COLORS[seg_name], edgecolor="white", linewidth=0.3,
            label=seg_name,
        )
        cumulative = cumulative + widths

    ax.set_yticks(y_positions)
    ax.set_yticklabels(metrics, fontsize=8)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("variance share")
    ax.set_title("Variance partition by metric (sorted by interaction share)", fontsize=10)
    ax.legend(loc="lower right", fontsize=7, frameon=False, ncol=3)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
