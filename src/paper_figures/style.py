"""Shared Nature-Communications-compatible figure style for paper figures."""
from __future__ import annotations

import matplotlib.pyplot as plt

FONT_FAMILY = "DejaVu Sans"
FONT_SIZE_BASE = 10
FONT_SIZE_TITLE = 11
FONT_SIZE_LABEL = 10
FONT_SIZE_TICK = 9
FONT_SIZE_LEGEND = 9
FONT_SIZE_ANNOTATION = 8

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
