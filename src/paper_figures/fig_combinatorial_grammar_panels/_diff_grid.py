"""Panel A: reference H&E inset plus 3x9 pixel-diff grid."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.patches import Rectangle

from src.paper_figures.style import (
    FONT_SIZE_DENSE_LABEL,
    FONT_SIZE_DENSE_TITLE,
    FONT_SIZE_INLINE,
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


def render_panel_a(
    fig: plt.Figure,
    subgrid,
    *,
    anchor_id: str,
    generated_root: Path,
    reference_path: Path,
) -> None:
    """Render reference inset above a 3x9 diff heatmap grid."""
    outer_ax = fig.add_subplot(subgrid)
    outer_ax.axis("off")
    _panel_label(outer_ax, "A")
    _draw_dashed_border(outer_ax)
    outer_ax.text(
        0.02,
        0.985,
        f"Representative anchor: {anchor_id}",
        transform=outer_ax.transAxes,
        fontsize=FONT_SIZE_INLINE,
        color=INK,
        ha="left",
        va="top",
    )

    reference_rgb = _shared.load_rgb(reference_path)
    diffs: dict[tuple[int, int], np.ndarray] = {}
    vmax = 0.0
    for state_idx, state in enumerate(STATES):
        for oxygen_idx, oxygen_label in enumerate(LEVELS):
            for glucose_idx, glucose_label in enumerate(LEVELS):
                col = oxygen_idx * len(LEVELS) + glucose_idx
                cond_path = generated_root / anchor_id / f"{_shared.condition_id(state, oxygen_label, glucose_label)}.png"
                cond_rgb = _shared.load_rgb(cond_path)
                diff = _shared.compute_pixel_diff(cond_rgb, reference_rgb)
                diffs[(state_idx, col)] = diff
                vmax = max(vmax, float(diff.max()))

    inner = subgrid.subgridspec(
        4,
        10,
        height_ratios=[1.15, 1.0, 1.0, 1.0],
        width_ratios=[1.0] * 9 + [0.08],
        hspace=0.08,
        wspace=0.04,
    )

    ref_ax = fig.add_subplot(inner[0, 0])
    ref_ax.imshow(reference_rgb)
    ref_ax.set_xticks([])
    ref_ax.set_yticks([])
    ref_ax.set_title("reference (original TME)", fontsize=FONT_SIZE_DENSE_TITLE, color=INK, pad=2.0)
    for spine in ref_ax.spines.values():
        spine.set_linewidth(0.4)
        spine.set_edgecolor("#6A6A6A")

    last_im = None
    for state_idx, state in enumerate(STATES):
        for col in range(9):
            ax = fig.add_subplot(inner[state_idx + 1, col])
            last_im = ax.imshow(
                diffs[(state_idx, col)],
                cmap="magma",
                vmin=0.0,
                vmax=max(vmax, 1e-6),
                aspect="auto",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            if state_idx == 0:
                oxygen_label = LEVELS[col // len(LEVELS)]
                glucose_label = LEVELS[col % len(LEVELS)]
                ax.set_title(f"{oxygen_label}/{glucose_label}", fontsize=FONT_SIZE_DENSE_TITLE, pad=1.2, color=INK)
            if col == 0:
                ax.set_ylabel(state, fontsize=FONT_SIZE_DENSE_LABEL, color=INK)
            for spine in ax.spines.values():
                spine.set_linewidth(0.25)
                spine.set_edgecolor("#8A8A8A")

    cbar_ax = fig.add_subplot(inner[1:4, 9])
    cbar = fig.colorbar(last_im, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=FONT_SIZE_DENSE_LABEL, colors=INK)
    cbar.set_label("|cond - ref| (mean abs RGB)", fontsize=FONT_SIZE_DENSE_LABEL, color=INK)
