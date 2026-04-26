"""Panel B: 3x9 residual-L2 heatmap with numeric cell labels."""
from __future__ import annotations

import numpy as np
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable

from src.paper_figures.style import (
    FONT_SIZE_CELL_TEXT,
    FONT_SIZE_DENSE_LABEL,
    FONT_SIZE_DENSE_TITLE,
    FONT_SIZE_LABEL,
)
from tools.ablation_report.shared import INK, plt

from . import _shared


STATES = _shared.STATES
LEVELS = _shared.LEVELS


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.05,
        1.02,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=FONT_SIZE_LABEL,
        fontweight="bold",
        color=INK,
    )


def _draw_dashed_border(ax: plt.Axes) -> None:
    ax.add_patch(
        Rectangle(
            (0.0, 0.0),
            1.0,
            1.0,
            transform=ax.transAxes,
            fill=False,
            linestyle="--",
            linewidth=0.8,
            edgecolor="#9A9A9A",
        )
    )


def render_panel_b(fig: plt.Figure, subgrid, *, residual_rows: list[dict[str, str]]) -> None:
    """Render the 3x9 residual L2 heatmap."""
    outer_ax = fig.add_subplot(subgrid)
    outer_ax.axis("off")
    _panel_label(outer_ax, "B")
    _draw_dashed_border(outer_ax)

    ax = fig.add_subplot(subgrid.subgridspec(1, 1)[0, 0])
    lookup = _shared.residual_lookup(residual_rows)
    matrix = np.zeros((len(STATES), len(LEVELS) * len(LEVELS)), dtype=np.float64)
    for state_idx, state in enumerate(STATES):
        for oxygen_idx, oxygen_label in enumerate(LEVELS):
            for glucose_idx, glucose_label in enumerate(LEVELS):
                col = oxygen_idx * len(LEVELS) + glucose_idx
                residuals = lookup.get((state, oxygen_label, glucose_label), {})
                matrix[state_idx, col] = float(residuals.get("residual_l2_norm", 0.0))

    vmax = max(float(matrix.max()) if matrix.size else 0.0, 1e-6)
    im = ax.imshow(matrix, cmap="magma", vmin=0.0, vmax=vmax, aspect="auto")
    ax.set_yticks(range(len(STATES)))
    ax.set_yticklabels(STATES, fontsize=FONT_SIZE_DENSE_LABEL, color=INK)
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_xticklabels(
        [f"{o}/{g}" for o in LEVELS for g in LEVELS],
        rotation=30,
        ha="right",
        fontsize=FONT_SIZE_DENSE_LABEL,
        color=INK,
    )
    ax.set_title(
        "interaction magnitude: residual L2 norm (low=0.50, mid=0.75, high=1.00)",
        fontsize=FONT_SIZE_DENSE_TITLE,
        loc="left",
        color=INK,
        pad=4.0,
    )
    ax.grid(False)

    threshold = 0.5 * vmax
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            text_color = "white" if value < threshold else "black"
            ax.text(j, i, f"{value:.2g}", ha="center", va="center", fontsize=FONT_SIZE_CELL_TEXT, color=text_color)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.08)
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=FONT_SIZE_DENSE_LABEL, colors=INK)
