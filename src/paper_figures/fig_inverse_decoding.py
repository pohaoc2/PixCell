"""Figure 4 inverse-decoding panel builder."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from src.paper_figures.style import FONT_SIZE_LABEL, FONT_SIZE_TICK, FONT_SIZE_TITLE


_NON_TYPING_MARKERS: frozenset[str] = frozenset({"Hoechst", "AF1", "Argo550", "PD-L1"})

_T1_DISPLAY_LABELS: dict[str, str] = {
    "cell_density": "density",
    "prolif_frac": "prolif",
    "nonprolif_frac": "nonprolif",
    "glucose_mean": "glucose",
    "oxygen_mean": r"O$_2$",
    "healthy_frac": "healthy",
    "cancer_frac": "cancer",
    "vasculature_frac": "vasc",
    "immune_frac": "immune",
    "dead_frac": "dead",
}

_ENCODER_COLORS: dict[str, str] = {
    "UNI-2h": "#f98866",
    "Virchow2": "#9b59b6",
    "CTransPath": "#bed7d8",
    "REMEDIS": "#5A5A5A",
    "ResNet-50": "#A9A9A9",
}
_ENCODER_DASHED: frozenset[str] = frozenset({"REMEDIS", "ResNet-50"})
_BAR_FACE_COLOR = _ENCODER_COLORS["UNI-2h"]
_BAR_EDGE_COLOR = "black"
_T2_UPPER_YMIN = -0.55
_T2_LOWER_YMAX = -0.75
_X_TICK_LABEL_PAD = 8
_YLABEL_XPOS = 0.05


def _read_probe_csv(path: Path) -> dict[str, dict[str, Any]]:
    import json as _json
    rows: dict[str, dict[str, Any]] = {}
    with Path(path).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows[str(row["target"])] = {
                "r2_mean": float(row["r2_mean"]),
                "r2_sd": float(row.get("r2_sd", "nan")),
                "n_valid_folds": float(row.get("n_valid_folds", "nan")),
                "r2_folds": [],
            }
    json_path = Path(path).with_suffix(".json")
    if json_path.is_file():
        payload = _json.loads(json_path.read_text(encoding="utf-8"))
        for result in payload.get("results", []):
            target = str(result.get("target", ""))
            if target in rows and isinstance(result.get("r2_folds"), list):
                rows[target]["r2_folds"] = [float(v) for v in result["r2_folds"]]
    return rows


def load_t1_data(encoder_csvs: dict[str, Path | None]) -> list[dict[str, Any]]:
    """Return T1 targets sorted by UNI-2h probe performance."""
    all_data = {
        encoder: _read_probe_csv(Path(path))
        for encoder, path in encoder_csvs.items()
        if path is not None and Path(path).is_file()
    }
    if "UNI-2h" not in all_data:
        raise ValueError("UNI-2h CSV is required to build the T1 panel")

    uni_data = all_data["UNI-2h"]
    ordered_targets = sorted(uni_data, key=lambda target: uni_data[target]["r2_mean"], reverse=True)
    rows: list[dict[str, Any]] = []
    for target in ordered_targets:
        encoder_rows = {
            encoder: data.get(
                target,
                {"r2_mean": float("nan"), "r2_sd": float("nan"), "n_valid_folds": float("nan"), "r2_folds": []},
            )
            for encoder, data in all_data.items()
        }
        rows.append(
            {
                "target": target,
                "label": _T1_DISPLAY_LABELS.get(target, target),
                "encoders": encoder_rows,
            }
        )
    return rows


def load_t2_data(t2_mlp_csv: Path) -> list[dict[str, Any]]:
    """Return filtered T2 markers sorted by mean probe R2."""
    rows = _read_probe_csv(Path(t2_mlp_csv))
    markers: list[dict[str, Any]] = []
    for marker, values in rows.items():
        if marker in _NON_TYPING_MARKERS:
            continue
        r2_mean = float(values["r2_mean"])
        n_valid_folds = int(values["n_valid_folds"]) if np.isfinite(values["n_valid_folds"]) else 0
        r2_sd = float(values["r2_sd"])
        markers.append(
            {
                "marker": marker,
                "r2_mean": r2_mean,
                "r2_sd": r2_sd,
                "r2_sem": (r2_sd / np.sqrt(n_valid_folds)) if n_valid_folds > 0 and np.isfinite(r2_sd) else float("nan"),
                "n_valid_folds": n_valid_folds,
                "r2_folds": [float(v) for v in values["r2_folds"] if np.isfinite(v)],
            }
        )
    return sorted(markers, key=lambda row: row["r2_mean"], reverse=True)


def _draw_panel_a(ax: plt.Axes, targets: list[dict[str, Any]], encoder_order: list[str]) -> None:
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

        # Gather fold data per target
        all_folds: list[list[float]] = []
        for row in targets:
            folds = row["encoders"].get(encoder_name, {}).get("r2_folds", [])
            valid = [v for v in folds if np.isfinite(v)]
            all_folds.append(valid)

        # Draw box plot for targets that have >= 2 fold values
        box_data = [folds if len(folds) >= 2 else [float("nan")] for folds in all_folds]
        bp = ax.boxplot(
            box_data,
            positions=positions,
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

        # Scatter jittered fold points
        for pos, folds in zip(positions, all_folds):
            if len(folds) == 0:
                continue
            if len(folds) >= 2:
                jitter = rng.uniform(-box_width * 0.18, box_width * 0.18, len(folds))
            else:
                jitter = np.zeros(1)
            ax.scatter(
                np.array([pos] * len(folds)) + jitter,
                folds,
                facecolors=color,
                edgecolors="black",
                linewidths=0.45,
                s=5.0,
                zorder=3,
                alpha=0.92,
            )

        # For targets with no fold data, plot mean as a diamond marker
        for pos, folds, row in zip(positions, all_folds, targets):
            if len(folds) == 0:
                r2_mean = row["encoders"].get(encoder_name, {}).get("r2_mean", float("nan"))
                if np.isfinite(r2_mean):
                    ax.scatter(
                        [pos],
                        [r2_mean],
                        marker="D",
                        facecolors=color,
                        s=18,
                        zorder=3,
                        edgecolors="black",
                        linewidths=0.45,
                    )

        legend_handles.append(Patch(facecolor=color, edgecolor="black", linewidth=0.6, label=encoder_name))

    ax.axhline(0.0, color="black", linewidth=0.8, zorder=1)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        [row["label"] for row in targets],
        rotation=45,
        ha="right",
        rotation_mode="anchor",
        fontsize=FONT_SIZE_TICK,
    )
    ax.tick_params(axis="x", pad=_X_TICK_LABEL_PAD, labelsize=FONT_SIZE_TICK)
    ax.tick_params(axis="y", labelsize=FONT_SIZE_TICK)
    ax.set_ylim(-0.35, 1.00)
    ax.grid(axis="y", linewidth=0.4, color="#E0E0E0", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(
        handles=legend_handles,
        frameon=False,
        loc="lower left",
        bbox_to_anchor=(0.015, 0.02),
        ncol=2,
        fontsize=FONT_SIZE_TICK,
        handletextpad=0.5,
        columnspacing=0.9,
        borderaxespad=0.0,
    )
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def _draw_panel_b_bars(ax: plt.Axes, markers: list[dict[str, Any]], x_positions: np.ndarray) -> None:
    means = np.asarray([row["r2_mean"] for row in markers], dtype=np.float64)
    sems = np.asarray([row["r2_sem"] for row in markers], dtype=np.float64)
    ax.bar(
        x_positions,
        means,
        width=0.65,
        facecolor=_BAR_FACE_COLOR,
        edgecolor=_BAR_EDGE_COLOR,
        linewidth=0.9,
        zorder=2,
    )
    valid_errors = np.isfinite(sems) & (sems > 0.0)
    if np.any(valid_errors):
        ax.errorbar(
            x_positions[valid_errors],
            means[valid_errors],
            yerr=sems[valid_errors],
            fmt="none",
            ecolor="black",
            elinewidth=0.9,
            capsize=2.2,
            capthick=0.9,
            zorder=3,
        )


def _draw_y_break_marks(ax_top: plt.Axes, ax_bottom: plt.Axes) -> None:
    fig = ax_top.figure
    top_box = ax_top.get_position()
    bottom_box = ax_bottom.get_position()
    x_center = top_box.x0 - 0.010
    dx = 0.010
    dy = 0.0065
    for y_center in (top_box.y0, bottom_box.y1):
        fig.add_artist(
            Line2D(
                [x_center - dx, x_center + dx],
                [y_center - dy, y_center + dy],
                transform=fig.transFigure,
                color="black",
                linewidth=1.0,
                solid_capstyle="round",
                clip_on=False,
            )
        )


def _add_shared_broken_ylabel(fig: plt.Figure, ax_top: plt.Axes, ax_bottom: plt.Axes, label: str) -> None:
    top_box = ax_top.get_position()
    bottom_box = ax_bottom.get_position()
    y_center = (((top_box.y0 + top_box.y1) / 2.0) + ((bottom_box.y0 + bottom_box.y1) / 2.0)) / 2.0
    fig.text(_YLABEL_XPOS, y_center, label, rotation=90, va="center", ha="center", fontsize=FONT_SIZE_LABEL)


def _add_axis_ylabel(fig: plt.Figure, ax: plt.Axes, label: str) -> None:
    box = ax.get_position()
    y_center = (box.y0 + box.y1) / 2.0
    fig.text(_YLABEL_XPOS, y_center, label, rotation=90, va="center", ha="center", fontsize=FONT_SIZE_LABEL)


def _draw_panel_b(ax_top: plt.Axes, ax_bottom: plt.Axes, markers: list[dict[str, Any]]) -> None:
    x_positions = np.arange(len(markers), dtype=np.float32)
    lower_min = min(
        float(np.floor((min(row["r2_mean"] - row["r2_sem"] for row in markers) - 0.25) * 2.0) / 2.0),
        -1.5,
    )

    for axis in (ax_top, ax_bottom):
        _draw_panel_b_bars(axis, markers, x_positions)
        axis.grid(axis="y", linewidth=0.4, color="#E0E0E0", zorder=0)
        axis.set_axisbelow(True)
        axis.set_xlim(-0.6, len(markers) - 0.4)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)

    ax_top.axhline(0.0, color="black", linewidth=0.8, zorder=1)
    ax_top.set_ylim(_T2_UPPER_YMIN, 0.50)
    ax_bottom.set_ylim(lower_min, _T2_LOWER_YMAX)

    ax_top.spines["bottom"].set_visible(False)
    ax_bottom.spines["top"].set_visible(False)
    ax_top.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax_top.tick_params(axis="y", labelsize=FONT_SIZE_TICK)
    ax_bottom.set_xticks(x_positions)
    ax_bottom.set_xticklabels(
        [row["marker"] for row in markers],
        rotation=45,
        ha="right",
        rotation_mode="anchor",
        fontsize=FONT_SIZE_TICK,
    )
    ax_bottom.tick_params(axis="x", pad=_X_TICK_LABEL_PAD, labelsize=FONT_SIZE_TICK)
    ax_bottom.tick_params(axis="y", labelsize=FONT_SIZE_TICK)

    ax_top.legend(
        handles=[Patch(facecolor=_BAR_FACE_COLOR, edgecolor=_BAR_EDGE_COLOR, label="UNI-2h MLP probe")],
        frameon=False,
        loc="upper right",
        bbox_to_anchor=(0.995, 0.985),
        fontsize=FONT_SIZE_LABEL,
        handletextpad=0.55,
        borderaxespad=0.0,
    )
    _draw_y_break_marks(ax_top, ax_bottom)


def _add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.085,
        1.03,
        label,
        transform=ax.transAxes,
        fontsize=FONT_SIZE_TITLE,
        fontweight="bold",
        va="top",
        ha="left",
    )


def build_inverse_decoding_figure(
    *,
    uni_t1_csv: Path,
    virchow_t1_csv: Path | None = None,
    ctranspath_t1_csv: Path | None = None,
    resnet50_t1_csv: Path | None = None,
    remedis_t1_csv: Path | None = None,
    t2_mlp_csv: Path,
    figsize: tuple[float, float] = (10.8, 7.5),
) -> plt.Figure:
    """Build the two-panel inverse-decoding figure."""
    encoder_csvs: dict[str, Path | None] = {"UNI-2h": Path(uni_t1_csv)}
    encoder_order = ["UNI-2h"]
    for encoder_name, path in (
        ("Virchow2", virchow_t1_csv),
        ("CTransPath", ctranspath_t1_csv),
        ("REMEDIS", remedis_t1_csv),
        ("ResNet-50", resnet50_t1_csv),
    ):
        if path is not None and Path(path).is_file():
            encoder_csvs[encoder_name] = Path(path)
            encoder_order.append(encoder_name)

    targets = load_t1_data(encoder_csvs)
    markers = load_t2_data(Path(t2_mlp_csv))

    fig = plt.figure(figsize=figsize, facecolor="white")
    outer = fig.add_gridspec(2, 1, height_ratios=[0.74, 0.74], hspace=0.34)
    ax_a = fig.add_subplot(outer[0, 0])
    panel_b = outer[1, 0].subgridspec(2, 1, height_ratios=[2.7, 1.45], hspace=0.05)
    ax_b_top = fig.add_subplot(panel_b[0, 0])
    ax_b_bottom = fig.add_subplot(panel_b[1, 0], sharex=ax_b_top)

    _draw_panel_a(ax_a, targets, encoder_order)
    _draw_panel_b(ax_b_top, ax_b_bottom, markers)
    _add_panel_label(ax_a, "A")
    _add_panel_label(ax_b_top, "B")
    fig.subplots_adjust(left=0.11, right=0.99, bottom=0.15, top=0.96)
    _add_axis_ylabel(fig, ax_a, r"R$^2$")
    _add_shared_broken_ylabel(fig, ax_b_top, ax_b_bottom, r"R$^2$")
    return fig
