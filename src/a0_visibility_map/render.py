"""Render the visibility map figure."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mpl-cache"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .collect import VisibilityRow


def render_visibility_chart(
    rows: list[VisibilityRow],
    output_path: str | Path,
    *,
    dpi: int = 300,
) -> Path:
    """Render the paired vs unpaired visibility figure."""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    x = np.arange(len(rows), dtype=float)
    width = 0.36
    paired_mean = [row.paired.mean_diff for row in rows]
    unpaired_mean = [row.unpaired.mean_diff for row in rows]
    paired_sd = [row.paired.mean_diff_sd for row in rows]
    unpaired_sd = [row.unpaired.mean_diff_sd for row in rows]
    paired_pct = [row.paired.pct_pixels_above_10 for row in rows]
    unpaired_pct = [row.unpaired.pct_pixels_above_10 for row in rows]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax2 = ax.twinx()

    ax.bar(
        x - width / 2,
        paired_mean,
        width,
        yerr=paired_sd,
        capsize=4,
        label="Paired mean diff",
        color="#0072B2",
        alpha=0.9,
    )
    ax.bar(
        x + width / 2,
        unpaired_mean,
        width,
        yerr=unpaired_sd,
        capsize=4,
        label="Unpaired mean diff",
        color="#D55E00",
        alpha=0.8,
    )

    ax2.plot(
        x - width / 2,
        paired_pct,
        color="#004C7F",
        marker="o",
        linewidth=1.8,
        label="Paired % pixels > 10",
    )
    ax2.plot(
        x + width / 2,
        unpaired_pct,
        color="#8C3A00",
        marker="s",
        linewidth=1.8,
        label="Unpaired % pixels > 10",
    )

    ax.set_xticks(x)
    ax.set_xticklabels([row.group_label for row in rows])
    ax.set_ylabel("Mean pixel difference")
    ax2.set_ylabel("% pixels > 10")
    ax.set_xlabel("Channel group")
    ax.set_title("Visibility map: paired vs unpaired leave-one-out impact")
    ax.grid(True, axis="y", alpha=0.25)

    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, loc="upper left", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path
