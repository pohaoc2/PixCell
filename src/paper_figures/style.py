"""Shared Nature-Communications-compatible figure style for paper figures."""
from __future__ import annotations

import matplotlib.pyplot as plt

FONT_FAMILY = "DejaVu Sans"
FONT_SIZE_BASE = 12
FONT_SIZE_TITLE = 13
FONT_SIZE_LABEL = 12
FONT_SIZE_TICK = 11
FONT_SIZE_LEGEND = 11
FONT_SIZE_ANNOTATION = 10
FONT_SIZE_PANEL_LABEL = 11   # bold A/B/C corner labels
FONT_SIZE_CELL_TEXT = 9      # text inside heatmap/matrix cells
FONT_SIZE_DENSE_LABEL = 8    # axis labels in dense small-tile grids
FONT_SIZE_DENSE_TITLE = 8    # subplot titles in dense grids
FONT_SIZE_INLINE = 10        # inline figure annotations

RC_PARAMS: dict[str, object] = {
    "font.family": FONT_FAMILY,
    "font.size": FONT_SIZE_BASE,
    "axes.titlesize": FONT_SIZE_TITLE,
    "axes.titleweight": "normal",
    "axes.labelsize": FONT_SIZE_LABEL,
    "xtick.labelsize": FONT_SIZE_TICK,
    "ytick.labelsize": FONT_SIZE_TICK,
    "legend.fontsize": FONT_SIZE_LEGEND,
    "figure.titlesize": FONT_SIZE_TITLE,
    "figure.titleweight": "normal",
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def apply_style() -> None:
    """Apply Nature-Communications-compatible rcParams. Call once before building figures."""
    plt.rcParams.update(RC_PARAMS)
