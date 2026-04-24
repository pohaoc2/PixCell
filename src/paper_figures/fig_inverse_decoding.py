"""Figure 4 inverse-decoding panel builder."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


_T_CRIT_95_BY_DF = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
}

_NON_TYPING_MARKERS: frozenset[str] = frozenset({"Hoechst", "AF1", "Argo550", "PD-L1"})

_T1_DISPLAY_LABELS: dict[str, str] = {
    "cell_density": "density",
    "prolif_frac": r"$f_{\mathrm{prolif}}$",
    "nonprolif_frac": r"$f_{\mathrm{nonprolif}}$",
    "glucose_mean": "glucose",
    "oxygen_mean": r"O$_2$",
    "healthy_frac": r"$f_{\mathrm{healthy}}$",
    "cancer_frac": r"$f_{\mathrm{cancer}}$",
    "vasculature_frac": r"$f_{\mathrm{vasc}}$",
    "immune_frac": r"$f_{\mathrm{immune}}$",
    "dead_frac": r"$f_{\mathrm{dead}}$",
}

_ENCODER_COLORS: dict[str, str] = {
    "UNI-2h": "#2C7BB6",
    "Virchow2": "#D7191C",
    "CTransPath": "#1B9E77",
    "REMEDIS": "#5A5A5A",
    "ResNet-50": "#A9A9A9",
}
_ENCODER_DASHED: frozenset[str] = frozenset({"REMEDIS", "ResNet-50"})

_T2_CATEGORY_COLORS: dict[str, str] = {
    "immune_signaling": "#8E44AD",
    "epithelial": "#E67E22",
    "proliferation": "#16A085",
    "immune_structural": "#7F8C8D",
}

_T2_MARKER_CATEGORIES: dict[str, str] = {
    "PD-1": "immune_signaling",
    "E-cadherin": "epithelial",
    "Pan-CK": "epithelial",
    "Ki67": "proliferation",
}

_T2_AXIS_CAP = -0.30


def _read_probe_csv(path: Path) -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    with Path(path).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows[str(row["target"])] = {
                "r2_mean": float(row["r2_mean"]),
                "r2_sd": float(row.get("r2_sd", "nan")),
                "n_valid_folds": float(row.get("n_valid_folds", "nan")),
            }
    return rows


def _ci_half_width(r2_sd: float, n_valid_folds: float) -> float:
    if not np.isfinite(r2_sd) or not np.isfinite(n_valid_folds) or n_valid_folds < 2:
        return float("nan")
    df = max(1, int(round(n_valid_folds)) - 1)
    t_crit = _T_CRIT_95_BY_DF.get(df, 1.96)
    return t_crit * r2_sd / math.sqrt(float(n_valid_folds))


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
                {"r2_mean": float("nan"), "r2_sd": float("nan"), "n_valid_folds": float("nan")},
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
        markers.append(
            {
                "marker": marker,
                "r2_mean": r2_mean,
                "r2_display": max(r2_mean, _T2_AXIS_CAP),
                "category": _T2_MARKER_CATEGORIES.get(marker, "immune_structural"),
                "capped": r2_mean < _T2_AXIS_CAP,
            }
        )
    return sorted(markers, key=lambda row: row["r2_mean"], reverse=True)


def _draw_panel_a(ax: plt.Axes, targets: list[dict[str, Any]], encoder_order: list[str]) -> None:
    n_encoders = max(1, len(encoder_order))
    width = min(0.18, 0.75 / n_encoders)
    x_positions = np.arange(len(targets), dtype=np.float32)

    for encoder_index, encoder_name in enumerate(encoder_order):
        offset = (encoder_index - (n_encoders - 1) / 2.0) * width
        heights = [row["encoders"].get(encoder_name, {}).get("r2_mean", float("nan")) for row in targets]
        bar_kwargs = {
            "width": width,
            "color": _ENCODER_COLORS.get(encoder_name, "#888888"),
            "label": encoder_name,
            "zorder": 2,
        }
        if encoder_name in _ENCODER_DASHED:
            bar_kwargs.update({"alpha": 0.85, "edgecolor": "black", "linewidth": 0.6, "linestyle": "--"})
        else:
            bar_kwargs.update({"edgecolor": "none"})
        ax.bar(x_positions + offset, heights, **bar_kwargs)

        if encoder_name == "UNI-2h":
            for x_value, row in zip(x_positions + offset, targets, strict=True):
                values = row["encoders"]["UNI-2h"]
                half_width = _ci_half_width(values["r2_sd"], values["n_valid_folds"])
                if np.isfinite(half_width) and np.isfinite(values["r2_mean"]):
                    ax.errorbar(
                        x_value,
                        values["r2_mean"],
                        yerr=half_width,
                        fmt="none",
                        ecolor="black",
                        elinewidth=0.9,
                        capsize=2.5,
                        capthick=0.9,
                        zorder=3,
                    )

    ax.axhline(0.0, color="black", linewidth=0.8, zorder=1)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([row["label"] for row in targets], rotation=45, ha="right")
    ax.set_ylabel("R$^2$")
    ax.set_ylim(-0.20, 1.05)
    ax.grid(axis="y", linewidth=0.4, color="#E0E0E0", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right", ncol=2)
    ax.set_title("(a) T1 targets · linear probe R$^2$", loc="left")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def _draw_panel_b(ax: plt.Axes, markers: list[dict[str, Any]]) -> None:
    x_positions = np.arange(len(markers), dtype=np.float32)
    for x_value, row in zip(x_positions, markers, strict=True):
        color = _T2_CATEGORY_COLORS[row["category"]]
        ax.bar(x_value, row["r2_display"], width=0.65, color=color, zorder=2)
        label = f"{row['r2_mean']:.3f}" + ("*" if row["capped"] else "")
        if row["r2_display"] >= 0:
            y_value = row["r2_display"] + 0.012
            va = "bottom"
        else:
            y_value = row["r2_display"] - 0.012
            va = "top"
        ax.text(x_value, y_value, label, ha="center", va=va, fontsize=6.5)

    ax.axhline(0.0, color="black", linewidth=0.8, zorder=1)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([row["marker"] for row in markers], rotation=45, ha="right")
    ax.set_ylabel("R$^2$")
    ax.set_ylim(_T2_AXIS_CAP - 0.10, 0.50)
    ax.grid(axis="y", linewidth=0.4, color="#E0E0E0", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(
        handles=[Patch(color=color, label=label.replace("_", " ")) for label, color in _T2_CATEGORY_COLORS.items()],
        frameon=False,
        loc="upper right",
    )
    if any(row["capped"] for row in markers):
        ax.text(0.02, 0.02, f"* axis capped at {_T2_AXIS_CAP}", transform=ax.transAxes, fontsize=7, va="bottom")
    ax.set_title("(b) T2 CODEX markers · MLP probe R$^2$", loc="left")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def build_inverse_decoding_figure(
    *,
    uni_t1_csv: Path,
    virchow_t1_csv: Path | None = None,
    ctranspath_t1_csv: Path | None = None,
    resnet50_t1_csv: Path | None = None,
    remedis_t1_csv: Path | None = None,
    t2_mlp_csv: Path,
    figsize: tuple[float, float] = (10.0, 4.2),
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

    fig, (ax_a, ax_b) = plt.subplots(
        1,
        2,
        figsize=figsize,
        facecolor="white",
        gridspec_kw={"width_ratios": [len(targets), len(markers)]},
    )
    _draw_panel_a(ax_a, targets, encoder_order)
    _draw_panel_b(ax_b, markers)
    fig.tight_layout(pad=1.2)
    return fig
