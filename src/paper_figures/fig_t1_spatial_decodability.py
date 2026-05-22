"""SI Figure 07c — per-patch UNI-2h spatial decodability for T1 targets.

Companion to figures/pngs_updated/07_inverse_decoding.png. The original panel A
ranks T1 targets by tile-mean R² (`r2_mean`), which inflates targets that are
nearly constant within a tile (oxygen, glucose). This figure instead ranks by
`r2_within`, which subtracts each tile's own mean before scoring — i.e. asks
whether UNI-2h patch tokens explain *within-tile* spatial variation, not just
slide-level drift.

Source: `src/a1_probe_mlp_spatial/out/t1_spatial/mlp_spatial_probe_results.csv`.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from src.paper_figures.style import (
    FONT_FAMILY,
    FONT_SIZE_ANNOTATION,
    FONT_SIZE_LABEL,
    FONT_SIZE_TICK,
    apply_style,
)


_T1_DISPLAY_LABELS: dict[str, str] = {
    "cell_density": "Density",
    "prolif_frac": "Prolif.",
    "nonprolif_frac": "Non-prolif.",
    "glucose_mean": "Glucose",
    "oxygen_mean": r"O$_2$",
    "healthy_frac": "Healthy",
    "cancer_frac": "Cancer",
    "vasculature_frac": "Vasculature",
    "immune_frac": "Immune",
    "dead_frac": "Dead",
}

_BAR_FACE_COLOR = "#f98866"
_BAR_EDGE_COLOR = "black"
_R2_WITHIN_FLOOR = -5.0  # clip extreme negatives so the bar plot stays readable


def _read_spatial_csv(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "target": row["target"],
                    "r2_global_mean": float(row["r2_mean"]),
                    "r2_global_sd": float(row["r2_sd"]),
                    "r2_within_mean": float(row["r2_within_mean"]),
                    "r2_within_sd": float(row["r2_within_sd"]),
                    "pearson_r_mean": float(row["pearson_r_mean"]),
                    "pearson_r_sd": float(row["pearson_r_sd"]),
                    "n_valid_folds": int(row["n_valid_folds"]),
                }
            )
    return rows


def _sorted_for_panel(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda r: r[key], reverse=True)


def _draw_panel(
    ax: plt.Axes,
    rows: list[dict[str, Any]],
    *,
    metric: str,
    sd_key: str,
    ylabel: str,
    clip_floor: float | None = None,
) -> None:
    labels = [_T1_DISPLAY_LABELS.get(r["target"], r["target"]) for r in rows]
    values = np.asarray([r[metric] for r in rows], dtype=np.float64)
    sds = np.asarray([r[sd_key] for r in rows], dtype=np.float64)

    truncated = np.zeros_like(values, dtype=bool)
    if clip_floor is not None:
        truncated = values < clip_floor
        values_plot = np.where(truncated, clip_floor, values)
        sds_plot = np.where(truncated, 0.0, sds)
    else:
        values_plot = values
        sds_plot = sds

    x = np.arange(len(rows), dtype=np.float64)
    ax.bar(
        x,
        values_plot,
        width=0.7,
        color=_BAR_FACE_COLOR,
        edgecolor=_BAR_EDGE_COLOR,
        linewidth=0.7,
        yerr=sds_plot,
        ecolor="#333333",
        error_kw={"elinewidth": 0.7, "capsize": 2.5, "capthick": 0.7},
        zorder=2,
    )
    ax.axhline(0.0, color="#555555", linewidth=0.6, zorder=1)

    for i, (val, trunc) in enumerate(zip(values, truncated)):
        if trunc:
            ax.annotate(
                f"{val:.1f}",
                xy=(x[i], clip_floor),
                xytext=(0, -10),
                textcoords="offset points",
                ha="center",
                va="top",
                fontsize=FONT_SIZE_ANNOTATION,
                fontfamily=FONT_FAMILY,
                color="#b22222",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontfamily=FONT_FAMILY, fontsize=FONT_SIZE_TICK)
    ax.set_ylabel(ylabel, fontfamily=FONT_FAMILY, fontsize=FONT_SIZE_LABEL)
    for spine in ("top", "right", "bottom", "left"):
        ax.spines[spine].set_color("black")
        ax.spines[spine].set_linewidth(0.8)
    ax.tick_params(axis="both", which="both", length=3, width=0.7, color="black")


def build_figure(*, spatial_csv: Path) -> plt.Figure:
    apply_style()
    rows = _read_spatial_csv(Path(spatial_csv))

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.8))

    # Panel A: r2_global (matches the original tile-mean convention).
    rows_a = _sorted_for_panel(rows, "r2_global_mean")
    _draw_panel(
        axes[0],
        rows_a,
        metric="r2_global_mean",
        sd_key="r2_global_sd",
        ylabel=r"Per-patch R² (global)",
        clip_floor=_R2_WITHIN_FLOOR,
    )
    axes[0].text(
        -0.10, 1.06, "A", transform=axes[0].transAxes,
        fontsize=12, fontweight="bold", fontfamily=FONT_FAMILY,
    )

    # Panel B: r2_within — the honest within-tile metric.
    rows_b = _sorted_for_panel(rows, "r2_within_mean")
    _draw_panel(
        axes[1],
        rows_b,
        metric="r2_within_mean",
        sd_key="r2_within_sd",
        ylabel=r"Per-patch R² (within-tile)",
        clip_floor=_R2_WITHIN_FLOOR,
    )
    axes[1].text(
        -0.10, 1.06, "B", transform=axes[1].transAxes,
        fontsize=12, fontweight="bold", fontfamily=FONT_FAMILY,
    )

    # Panel C: Pearson r — scale/offset-invariant correlation.
    rows_c = _sorted_for_panel(rows, "pearson_r_mean")
    _draw_panel(
        axes[2],
        rows_c,
        metric="pearson_r_mean",
        sd_key="pearson_r_sd",
        ylabel="Per-patch Pearson r",
    )
    axes[2].text(
        -0.10, 1.06, "C", transform=axes[2].transAxes,
        fontsize=12, fontweight="bold", fontfamily=FONT_FAMILY,
    )

    fig.tight_layout(rect=[0.0, 0.0, 1.0, 1.0])
    return fig


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    spatial_csv = root / "src" / "a1_probe_mlp_spatial" / "out" / "t1_spatial" / "mlp_spatial_probe_results.csv"
    out_dir = root / "figures" / "pngs_updated"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig = build_figure(spatial_csv=spatial_csv)
    out_path = out_dir / "07c_t1_spatial_decodability.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"wrote {out_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
