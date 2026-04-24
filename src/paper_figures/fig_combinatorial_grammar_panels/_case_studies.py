"""Panel C: three horizontal-bar case studies."""
from __future__ import annotations

import numpy as np
from matplotlib.patches import Rectangle

from tools.ablation_report.shared import INK, SOFT_GRID, plt


_METRIC_LABELS: dict[str, str] = {
    "residual_nuclear_density": "nuclear density",
    "residual_mean_cell_size": "mean cell size",
    "residual_nucleus_area_median": "nucleus area median",
    "residual_nucleus_area_iqr": "nucleus area IQR",
    "residual_hematoxylin_burden": "hematoxylin burden",
    "residual_hematoxylin_ratio": "hematoxylin ratio",
    "residual_eosin_ratio": "eosin ratio",
    "residual_glcm_contrast": "GLCM contrast",
    "residual_glcm_homogeneity": "GLCM homogeneity",
}


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.05,
        1.02,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=13,
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


def select_case_rows(residual_rows: list[dict[str, str]]) -> list[tuple[str, dict[str, str]]]:
    """Return lowest, median, and highest rows by residual_l2_norm."""
    if not residual_rows:
        return []
    sorted_rows = sorted(
        residual_rows,
        key=lambda row: float(row.get("residual_l2_norm", 0.0) or 0.0),
    )
    labels = ["lowest", "median", "highest"]
    indices = [0, len(sorted_rows) // 2, len(sorted_rows) - 1]
    out: list[tuple[str, dict[str, str]]] = []
    used: set[int] = set()
    for label, idx in zip(labels, indices, strict=True):
        if idx in used:
            continue
        used.add(idx)
        out.append((label, sorted_rows[idx]))
    return out


def render_panel_c(fig: plt.Figure, subgrid, *, residual_rows: list[dict[str, str]]) -> None:
    """Render three stacked horizontal-bar case-study subplots."""
    outer_ax = fig.add_subplot(subgrid)
    outer_ax.axis("off")
    _panel_label(outer_ax, "C")
    _draw_dashed_border(outer_ax)
    outer_ax.text(
        0.0,
        1.01,
        "Signed residuals: lowest / median / highest L2",
        transform=outer_ax.transAxes,
        fontsize=7.5,
        ha="left",
        va="bottom",
        color=INK,
    )

    cases = select_case_rows(residual_rows)
    if not cases:
        return

    inner = subgrid.subgridspec(len(cases), 1, hspace=0.55)
    for row_idx, (case_label, case_row) in enumerate(cases):
        ax = fig.add_subplot(inner[row_idx, 0])
        state = str(case_row["cell_state"])
        oxygen_label = str(case_row["oxygen_label"])
        glucose_label = str(case_row["glucose_label"])
        l2_value = float(case_row.get("residual_l2_norm", 0.0) or 0.0)

        ranked: list[tuple[float, str, float]] = []
        for key, label in _METRIC_LABELS.items():
            value = case_row.get(key)
            if value in ("", None):
                continue
            numeric = float(value)
            ranked.append((abs(numeric), label, numeric))
        ranked.sort(reverse=True)

        labels = [label for _, label, _ in ranked]
        values = [value for _, _, value in ranked]
        y = np.arange(len(labels), dtype=np.float64)

        ax.barh(y, values, height=0.66, color="#4C78A8", edgecolor="black", linewidth=0.5)
        ax.axvline(0.0, color="black", linewidth=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=6.5, color=INK)
        ax.invert_yaxis()
        ax.tick_params(axis="x", labelsize=6.5, colors=INK)
        ax.set_title(
            f"{case_label}: {state}, O2={oxygen_label}, glucose={glucose_label}, L2={l2_value:.3g}",
            fontsize=7,
            loc="left",
            color=INK,
        )
        ax.grid(axis="x", color=SOFT_GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

