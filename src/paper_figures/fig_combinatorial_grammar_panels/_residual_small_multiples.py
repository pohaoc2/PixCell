"""SI panel: per-metric residual heatmap small-multiples (state x (O2,gluc))."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.a3_combinatorial_sweep.main import MORPHOLOGY_METRICS
from src.paper_figures.fig_combinatorial_grammar_panels._shared import (
    LEVELS, STATES, read_csv,
)


def _residual_grid(rows: list[dict[str, str]], metric: str) -> np.ndarray:
    """Return 3 x 9 array of (actual - additive expected) for one metric."""
    by_key: dict[tuple[str, str, str], float] = {}
    for row in rows:
        key = (str(row["cell_state"]), str(row["oxygen_label"]), str(row["glucose_label"]))
        actual = float(row[f"actual_{metric}"])
        expected = float(row[f"expected_{metric}"])
        by_key[key] = actual - expected
    grid = np.zeros((len(STATES), len(LEVELS) * len(LEVELS)), dtype=np.float64)
    for s_idx, state in enumerate(STATES):
        for ox_idx, ox in enumerate(LEVELS):
            for g_idx, gluc in enumerate(LEVELS):
                col = ox_idx * len(LEVELS) + g_idx
                grid[s_idx, col] = by_key.get((state, ox, gluc), 0.0)
    return grid


def _metric_has_signal(rows: list[dict[str, str]], metric: str) -> bool:
    actual_col = f"actual_{metric}"
    expected_col = f"expected_{metric}"
    if not rows or actual_col not in rows[0] or expected_col not in rows[0]:
        return False
    for row in rows:
        try:
            diff = float(row[actual_col]) - float(row[expected_col])
        except (TypeError, ValueError):
            continue
        if np.isfinite(diff) and abs(diff) > 1e-12:
            return True
    return False


def draw_residual_small_multiples(fig: plt.Figure, subgrid, *, residuals_csv: Path) -> None:
    rows = read_csv(residuals_csv)
    visible_metrics = tuple(m for m in MORPHOLOGY_METRICS if _metric_has_signal(rows, m))
    n_metrics = len(visible_metrics)
    ncols = 3
    nrows = (n_metrics + ncols - 1) // ncols
    inner = subgrid.subgridspec(nrows, ncols, hspace=0.55, wspace=0.25)

    for idx, metric in enumerate(visible_metrics):
        row, col = divmod(idx, ncols)
        ax = fig.add_subplot(inner[row, col])
        grid = _residual_grid(rows, metric)
        max_abs = float(np.max(np.abs(grid))) if grid.size else 1.0
        max_abs = max(max_abs, 1e-9)
        im = ax.imshow(grid, cmap="RdBu_r", vmin=-max_abs, vmax=max_abs, aspect="auto")
        ax.set_title(metric, fontsize=7)
        ax.set_xticks(range(len(LEVELS) * len(LEVELS)))
        ax.set_xticklabels([f"{ox[0]}/{g[0]}" for ox in LEVELS for g in LEVELS], fontsize=5, rotation=45)
        ax.set_yticks(range(len(STATES)))
        ax.set_yticklabels(STATES, fontsize=6)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
