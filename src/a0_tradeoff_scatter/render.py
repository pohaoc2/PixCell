"""Render the specificity-realism tradeoff plots."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mpl-cache"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from .collect import TradeoffAggregate


def render_tradeoff_panel(
    rows: list[TradeoffAggregate],
    output_path: str | Path,
    *,
    split: str,
    dpi: int = 300,
) -> Path:
    """Render one split's AJI/PQ vs realism panel."""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    split_rows = sorted([row for row in rows if row.split == split], key=lambda row: row.n_groups)

    x = [row.n_groups for row in split_rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax2 = ax.twinx()

    ax.errorbar(
        x,
        [row.aji_mean for row in split_rows],
        yerr=[row.aji_sd for row in split_rows],
        color="#0072B2",
        marker="o",
        linewidth=2,
        capsize=4,
        label="AJI",
    )
    ax.errorbar(
        x,
        [row.pq_mean for row in split_rows],
        yerr=[row.pq_sd for row in split_rows],
        color="#009E73",
        marker="s",
        linewidth=2,
        capsize=4,
        label="PQ",
    )
    ax2.errorbar(
        x,
        [row.realism_mean for row in split_rows],
        yerr=[row.realism_sd for row in split_rows],
        color="#D55E00",
        marker="^",
        linewidth=2,
        capsize=4,
        label=split_rows[0].realism_key.upper() if split_rows else "Realism",
    )

    for row in split_rows:
        if row.is_pareto:
            ax.annotate(
                row.condition,
                xy=(row.n_groups, row.aji_mean),
                xytext=(4, 6),
                textcoords="offset points",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xlabel("Active groups")
    ax.set_ylabel("Structural fidelity (AJI / PQ)")
    ax2.set_ylabel(f"Realism ({split_rows[0].realism_key.upper()})" if split_rows else "Realism")
    ax.set_title(f"Specificity-realism tradeoff ({split})")
    ax.grid(True, axis="y", alpha=0.25)

    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, loc="lower right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path
