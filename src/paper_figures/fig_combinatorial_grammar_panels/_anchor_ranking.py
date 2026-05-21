"""SI panel: horizontal bar chart of per-anchor sweep magnitude (||Delta metric||_2)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from src.paper_figures.fig_combinatorial_grammar_panels._shared import (
    compute_anchor_sweep_magnitude, read_csv,
)


def draw_anchor_ranking(fig: plt.Figure, subgrid, *, signatures_csv: Path) -> None:
    rows = read_csv(signatures_csv)
    magnitudes = compute_anchor_sweep_magnitude(rows)
    ordered = sorted(magnitudes.items(), key=lambda pair: pair[1])

    ax = fig.add_subplot(subgrid)
    anchor_ids = [pair[0] for pair in ordered]
    values = [pair[1] for pair in ordered]
    ax.barh(anchor_ids, values, color="#2a5db0")
    ax.set_xlabel("sweep magnitude (sum-var across metrics)")
    ax.set_title("Anchor sweep responsiveness", fontsize=9)
    ax.tick_params(axis="y", labelsize=6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
