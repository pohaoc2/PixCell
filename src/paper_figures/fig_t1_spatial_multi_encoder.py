"""SI Figure 07d: multi-encoder spatial decodability for T1 targets."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from src.paper_figures.fig_inverse_decoding import _ENCODER_COLORS
from src.paper_figures.style import (
    FONT_SIZE_LABEL,
    FONT_SIZE_LEGEND,
    FONT_SIZE_TICK,
    FONT_SIZE_TITLE,
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

_NON_TYPING_MARKERS: frozenset[str] = frozenset({"Hoechst", "AF1", "Argo550", "PD-L1"})
_R2_CAP = -1.0
_X_TICK_LABEL_PAD = 8


def _load_encoder(path: Path) -> dict[str, dict[str, Any]]:
    """Load per-target summary + fold arrays from CSV (means/SD) + JSON (folds)."""
    rows: dict[str, dict[str, Any]] = {}
    with Path(path).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows[str(row["target"])] = {
                "r2_within_mean": float(row["r2_within_mean"]),
                "r2_within_sd": float(row["r2_within_sd"]),
                "pearson_r_mean": float(row["pearson_r_mean"]),
                "pearson_r_sd": float(row["pearson_r_sd"]),
                "n_valid_folds": float(row["n_valid_folds"]),
                "r2_within_folds": [],
                "pearson_r_folds": [],
            }
    json_path = Path(path).with_suffix(".json")
    if json_path.is_file():
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        for result in payload.get("results", []):
            target = str(result.get("target", ""))
            if target not in rows:
                continue
            for fold_key in ("r2_within_folds", "pearson_r_folds"):
                folds = result.get(fold_key, [])
                if isinstance(folds, list):
                    rows[target][fold_key] = [float(v) for v in folds]
    return rows


def _load_raw_mx_spatial(path: Path) -> list[dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with Path(path).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            marker = str(row["target"])
            if marker in _NON_TYPING_MARKERS:
                continue
            r2_within_mean = float(row["r2_within_mean"])
            r2_within_sd = float(row.get("r2_within_sd", "nan"))
            n_valid_folds = float(row.get("n_valid_folds", "nan"))
            rows[marker] = {
                "marker": marker,
                "r2_within_mean": r2_within_mean,
                "r2_within_sd": r2_within_sd,
                "r2_within_sem": (
                    r2_within_sd / np.sqrt(n_valid_folds)
                    if n_valid_folds > 0 and np.isfinite(r2_within_sd)
                    else float("nan")
                ),
                "r2_within_folds": [],
            }
    json_path = Path(path).with_suffix(".json")
    if json_path.is_file():
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        for result in payload.get("results", []):
            marker = str(result.get("target", ""))
            if marker not in rows:
                continue
            folds = result.get("r2_within_folds", [])
            if isinstance(folds, list):
                rows[marker]["r2_within_folds"] = [float(v) for v in folds if np.isfinite(v)]
    return sorted(rows.values(), key=lambda row: row["r2_within_mean"], reverse=True)


def _draw_panel(
    ax: plt.Axes,
    *,
    targets: list[str],
    encoder_rows: dict[str, dict[str, dict[str, Any]]],
    encoder_order: list[str],
    folds_key: str,
    mean_key: str,
    ylabel: str,
    ylim: tuple[float, float],
    clip_floor: float | None = None,
    show_legend: bool = False,
) -> list[Any]:
    n_encoders = max(1, len(encoder_order))
    step = min(0.20, 0.96 / n_encoders)
    box_width = step * 0.68
    x_positions = np.arange(len(targets), dtype=np.float64)
    rng = np.random.default_rng(seed=42)

    legend_handles: list[Any] = []

    for encoder_index, encoder_name in enumerate(encoder_order):
        offset = (encoder_index - (n_encoders - 1) / 2.0) * step
        color = _ENCODER_COLORS.get(encoder_name, "#888888")
        positions = [float(x) + offset for x in x_positions]
        target_map = encoder_rows.get(encoder_name, {})

        all_folds: list[list[float]] = []
        for target in targets:
            folds = target_map.get(target, {}).get(folds_key, [])
            valid = [v for v in folds if np.isfinite(v)]
            all_folds.append(valid)

        box_positions: list[float] = []
        box_data: list[list[float]] = []
        truncated_positions: list[float] = []

        for pos, folds in zip(positions, all_folds):
            if len(folds) < 2:
                continue
            if clip_floor is not None and float(np.min(folds)) < clip_floor:
                truncated_positions.append(pos)
            clipped = [max(v, clip_floor) if clip_floor is not None else v for v in folds]
            box_positions.append(pos)
            box_data.append(clipped)

        if box_data:
            bp = ax.boxplot(
                box_data,
                positions=box_positions,
                widths=box_width,
                patch_artist=True,
                manage_ticks=False,
                showfliers=False,
                zorder=2,
                medianprops={"color": "black", "linewidth": 1.35},
                whiskerprops={"linewidth": 0.95},
                capprops={"linewidth": 0.95},
                boxprops={"linewidth": 1.0},
            )
            for patch in bp["boxes"]:
                patch.set_facecolor(color)
                patch.set_alpha(0.58)
                patch.set_edgecolor("black")

        for pos, folds in zip(positions, all_folds):
            if len(folds) == 0:
                continue
            clipped = np.asarray(
                [max(v, clip_floor) if clip_floor is not None else v for v in folds],
                dtype=np.float64,
            )
            if len(folds) >= 2:
                jitter = rng.uniform(-box_width * 0.18, box_width * 0.18, len(folds))
            else:
                jitter = np.zeros(1)
            ax.scatter(
                np.array([pos] * len(folds)) + jitter,
                clipped,
                facecolors=color,
                edgecolors="black",
                linewidths=0.45,
                s=5.0,
                zorder=3,
                alpha=0.92,
            )

        for pos, folds, target in zip(positions, all_folds, targets):
            if len(folds) >= 2:
                continue
            r2_mean = target_map.get(target, {}).get(mean_key, float("nan"))
            if not np.isfinite(r2_mean):
                continue
            if clip_floor is not None and float(r2_mean) < clip_floor:
                truncated_positions.append(pos)
            value = max(r2_mean, clip_floor) if clip_floor is not None else r2_mean
            ax.scatter(
                [pos],
                [value],
                marker="D",
                facecolors=color,
                s=18,
                zorder=3,
                edgecolors="black",
                linewidths=0.45,
            )

        if clip_floor is not None:
            for pos in sorted(set(truncated_positions)):
                ax.scatter(
                    [pos],
                    [clip_floor],
                    marker="v",
                    facecolors=color,
                    s=22,
                    zorder=4,
                    edgecolors="black",
                    linewidths=0.5,
                    clip_on=False,
                )

        legend_handles.append(Patch(facecolor=color, edgecolor="black", linewidth=0.6, label=encoder_name))

    ax.axhline(0.0, color="black", linewidth=0.8, zorder=1)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        [_T1_DISPLAY_LABELS.get(target, target) for target in targets],
        rotation=45,
        ha="right",
        rotation_mode="anchor",
        fontsize=FONT_SIZE_TICK,
    )
    ax.tick_params(axis="x", pad=_X_TICK_LABEL_PAD, labelsize=FONT_SIZE_TICK)
    ax.tick_params(axis="y", labelsize=FONT_SIZE_TICK)
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel, fontsize=FONT_SIZE_LABEL)
    ax.grid(axis="y", linewidth=0.4, color="#E0E0E0", zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.8)

    if show_legend:
        ax.legend(
            handles=legend_handles,
            frameon=False,
            loc="upper right",
            bbox_to_anchor=(0.995, 0.985),
            ncol=2,
            fontsize=FONT_SIZE_LEGEND,
            handletextpad=0.5,
            columnspacing=0.9,
            borderaxespad=0.0,
        )
    return legend_handles


def _draw_raw_mx_panel(ax: plt.Axes, markers: list[dict[str, Any]]) -> None:
    x_positions = np.arange(len(markers), dtype=np.float64)
    means = np.asarray([row["r2_within_mean"] for row in markers], dtype=np.float64)
    sems = np.asarray([row["r2_within_sem"] for row in markers], dtype=np.float64)
    clipped_means = np.maximum(means, _R2_CAP)
    truncated_mask = means < _R2_CAP

    ax.bar(
        x_positions,
        clipped_means,
        width=0.65,
        facecolor=_ENCODER_COLORS["UNI-2h"],
        edgecolor="black",
        linewidth=0.9,
        zorder=2,
    )
    valid_errors = np.isfinite(sems) & (sems > 0.0)
    if np.any(valid_errors):
        lower = np.maximum(clipped_means[valid_errors] - sems[valid_errors], _R2_CAP)
        upper = clipped_means[valid_errors] + sems[valid_errors]
        ax.errorbar(
            x_positions[valid_errors],
            clipped_means[valid_errors],
            yerr=np.vstack([clipped_means[valid_errors] - lower, upper - clipped_means[valid_errors]]),
            fmt="none",
            ecolor="black",
            elinewidth=0.9,
            capsize=2.2,
            capthick=0.9,
            zorder=3,
        )
    if np.any(truncated_mask):
        ax.scatter(
            x_positions[truncated_mask],
            np.full(np.sum(truncated_mask), _R2_CAP),
            marker="v",
            facecolors=_ENCODER_COLORS["UNI-2h"],
            edgecolors="black",
            linewidths=0.5,
            s=24,
            zorder=4,
            clip_on=False,
        )

    ax.axhline(0.0, color="black", linewidth=0.8, zorder=1)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        [row["marker"] for row in markers],
        rotation=45,
        ha="right",
        rotation_mode="anchor",
        fontsize=FONT_SIZE_TICK,
    )
    ax.tick_params(axis="x", pad=_X_TICK_LABEL_PAD, labelsize=FONT_SIZE_TICK)
    ax.tick_params(axis="y", labelsize=FONT_SIZE_TICK)
    ax.set_ylim(_R2_CAP, 0.05)
    ax.set_ylabel("Within-tile R²", fontsize=FONT_SIZE_LABEL)
    ax.grid(axis="y", linewidth=0.4, color="#E0E0E0", zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.8)


def build_figure(*, encoder_csvs: dict[str, Path | None], t2_spatial_csv: Path) -> plt.Figure:
    apply_style()
    encoder_rows = {
        encoder_name: _load_encoder(Path(csv_path))
        for encoder_name, csv_path in encoder_csvs.items()
        if csv_path is not None and Path(csv_path).is_file()
    }
    if not encoder_rows:
        raise ValueError("no encoder CSVs found")
    if "UNI-2h" not in encoder_rows:
        raise ValueError("UNI-2h CSV is required to order targets")

    encoder_order = [name for name in encoder_csvs.keys() if name in encoder_rows]
    uni_rows = encoder_rows["UNI-2h"]
    targets = sorted(
        uni_rows.keys(),
        key=lambda target: uni_rows[target]["r2_within_mean"],
        reverse=True,
    )
    markers = _load_raw_mx_spatial(Path(t2_spatial_csv))

    # Side-by-side at the common figure width keeps the panel short (compact
    # height) while sharing the standard 12 pt text with the other figures.
    fig = plt.figure(figsize=(15.8, 4.6), facecolor="white")
    outer = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.16)
    ax_a = fig.add_subplot(outer[0, 0])
    ax_b = fig.add_subplot(outer[0, 1])

    _draw_panel(
        ax_a,
        targets=targets,
        encoder_rows=encoder_rows,
        encoder_order=encoder_order,
        folds_key="r2_within_folds",
        mean_key="r2_within_mean",
        ylabel="Within-tile R²",
        ylim=(_R2_CAP, 0.5),
        clip_floor=_R2_CAP,
        show_legend=True,
    )
    _draw_raw_mx_panel(ax_b, markers)

    ax_a.text(
        -0.03,
        1.03,
        "A",
        transform=ax_a.transAxes,
        fontsize=FONT_SIZE_TITLE,
        fontweight="bold",
        va="top",
        ha="left",
    )

    ax_b.text(
        -0.03,
        1.03,
        "B",
        transform=ax_b.transAxes,
        fontsize=FONT_SIZE_TITLE,
        fontweight="bold",
        va="top",
        ha="left",
    )

    fig.subplots_adjust(left=0.055, right=0.995, bottom=0.22, top=0.93)
    return fig


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    fig = build_figure(
        encoder_csvs={
            "UNI-2h": root / "src/a1_probe_mlp_spatial/out/uni_16/mlp_spatial_probe_results.csv",
            "Virchow2": root / "src/a1_probe_mlp_spatial/out/virchow2_16/mlp_spatial_probe_results.csv",
            "CTransPath": root / "src/a1_probe_mlp_spatial/out/ctranspath_07/mlp_spatial_probe_results.csv",
            "ResNet-50": root / "src/a1_probe_mlp_spatial/out/resnet50_07/mlp_spatial_probe_results.csv",
            "REMEDIS": root / "src/a1_probe_mlp_spatial/out/remedis_07/mlp_spatial_probe_results.csv",
        },
        t2_spatial_csv=root / "src/a1_probe_mlp_spatial/out/t2_spatial/mlp_spatial_probe_results.csv",
    )
    out_path = root / "figures" / "pngs_updated" / "concat" / "07d_t1_spatial_multi_encoder.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"wrote {out_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
