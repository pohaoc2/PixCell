"""Render summary figures for probe, sweep, and null results."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from adjustText import adjust_text
from PIL import Image

from src._tasklib.io import ensure_directory
from src.a4_uni_probe.labels import (
    _load_channel_array,
    CHANNEL_ATTR_NAMES,
    MORPHOLOGY_ATTR_NAMES,
    APPEARANCE_ATTR_NAMES,
)


PREFERRED_SWEEP_ATTRS = [
    "eccentricity_mean",
    "nuclear_area_mean",
    "nuclei_density",
    "texture_e_contrast",
    "texture_h_contrast",
    "texture_h_energy",
]
SWEEP_ALPHA_ROWS = [
    ("-1.00", "α = -1"),
    ("+0.00", "α = 0"),
    ("+1.00", "α = +1"),
]
SWEEP_ATTRS: tuple[str, ...] = (
    "nuclei_density",
    "nuclear_area_mean",
    "eccentricity_mean",
    "texture_e_contrast",
    "texture_h_contrast",
    "texture_h_energy",
)
SWEEP_ATTR_DISPLAY_NAMES: dict[str, str] = {
    "nuclei_density": "Nuclei density",
    "nuclear_area_mean": "Nuclear area mean",
    "eccentricity_mean": "Eccentricity mean",
    "texture_e_contrast": "Eosin contrast",
    "texture_h_contrast": "Hematoxylin contrast",
    "texture_h_energy": "Hematoxylin energy",
}
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_A4_DATA_ROOT = ROOT / "data" / "orion-crc33"
ATTR_TO_PRIMARY_METRIC = {
    "eccentricity_mean": "morpho.eccentricity_mean",
    "nuclear_area_mean": "morpho.nuclear_area_mean",
    "nuclei_density": "morpho.nuclei_density",
    "texture_e_contrast": "appearance.texture_e_contrast",
    "texture_h_contrast": "appearance.texture_h_contrast",
    "texture_h_energy": "appearance.texture_h_energy",
}


ATTR_DISPLAY_NAMES: dict[str, str] = {
    "eccentricity_mean": "Eccentricity",
    "nuclear_area_mean": "Nuclear area",
    "nuclei_density": "Nuclei density",
    "texture_h_contrast": "H contrast",
    "texture_e_contrast": "E contrast",
    "texture_h_energy": "H energy",
    "texture_e_energy": "E energy",
    "texture_h_homogeneity": "H homogeneity",
    "texture_e_homogeneity": "E homogeneity",
    "texture_h_correlation": "H correlation",
    "texture_e_correlation": "E correlation",
    "texture_h_dissimilarity": "H dissimilarity",
    "texture_e_dissimilarity": "E dissimilarity",
    "prolif_fraction": "Prolif fraction",
    "intensity_mean_h": "H intensity",
    "intensity_mean_e": "E intensity",
    "h_mean": "H mean",
    "e_mean": "E mean",
    "h_std": "H std",
    "e_std": "E std",
    "stain_vector_angle_deg": "Stain angle",
}


def _display_attr(attr: str) -> str:
    """Return a clean display label for an attribute name."""
    if attr in ATTR_DISPLAY_NAMES:
        return ATTR_DISPLAY_NAMES[attr]
    label = attr.replace("_", " ")
    label = re.sub(r"\btexture\s+", "", label, flags=re.IGNORECASE)
    label = re.sub(r"\b(h|e)\b", lambda m: m.group().upper(), label)
    return label


def _clean_attr_label(attr: str) -> str:
    """Backward-compat alias for _display_attr."""
    return _display_attr(attr)


def _read_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _ordered_attrs(attr_names: set[str], preferred: list[str]) -> list[str]:
    ordered = [attr for attr in preferred if attr in attr_names]
    ordered.extend(sorted(attr_names - set(ordered)))
    return ordered


def _appearance_metric_title(metric_name: str) -> str:
    title = metric_name.removeprefix("appearance.")
    title = title.replace("texture_h_", "H texture ")
    title = title.replace("texture_e_", "E texture ")
    title = title.replace("stain_vector_angle_deg", "stain angle (deg)")
    title = title.replace("_", " ")
    return title.title()


def _appearance_attr_order(rows: list[dict[str, str]]) -> list[str]:
    preferred = ["eccentricity_mean", "nuclear_area_mean", "nuclei_density"]
    seen = {row["attr"] for row in rows}
    ordered = [attr for attr in preferred if attr in seen]
    ordered.extend(sorted(seen - set(ordered)))
    return ordered


def _render_grouped_metric_grid(
    rows: list[dict[str, str]],
    *,
    metric_key_specs: list[tuple[str, str, str]],
    title: str,
    y_label: str,
    panel_path: Path,
) -> Path:
    metric_names = sorted({row["metric"] for row in rows})
    attrs = _appearance_attr_order(rows)
    index = {(row["metric"], row["attr"]): row for row in rows}

    ncols = 3
    nrows = int(np.ceil(len(metric_names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15.5, max(6.5, nrows * 3.4)), squeeze=False)
    x = np.arange(len(attrs))
    width = 0.24 if len(metric_key_specs) == 3 else 0.32
    offsets = np.linspace(-(len(metric_key_specs) - 1) / 2, (len(metric_key_specs) - 1) / 2, len(metric_key_specs)) * width

    for axis, metric_name in zip(axes.flat, metric_names):
        for offset, (value_key, label, color) in zip(offsets, metric_key_specs):
            values = []
            for attr in attrs:
                row = index.get((metric_name, attr))
                values.append(float(row[value_key]) if row and row.get(value_key) not in (None, "") else float("nan"))
            axis.bar(x + offset, values, width=width, label=label, color=color)
        axis.axhline(0.0, color="#4a5568", linewidth=0.8)
        axis.set_title(_appearance_metric_title(metric_name), fontsize=10)
        axis.set_xticks(x)
        axis.set_xticklabels(attrs, rotation=35, ha="right", fontsize=8)
        axis.tick_params(axis="y", labelsize=8)

    for axis in axes.flat[len(metric_names) :]:
        axis.axis("off")

    axes[0, 0].legend(frameon=False, fontsize=9, loc="best")
    fig.suptitle(title, fontsize=14)
    fig.supylabel(y_label)
    fig.tight_layout()
    fig.savefig(panel_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return panel_path


def render_panel_a(out_dir: str | Path) -> Path:
    out_path = Path(out_dir)
    figure_dir = ensure_directory(out_path / "figures")
    csv_path = out_path / "probe_results.csv"
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    attrs = [row["attr"] for row in rows]
    uni = np.asarray([float(row["uni_r2_mean"]) for row in rows], dtype=np.float32)
    tme = np.asarray([float(row["tme_r2_mean"]) for row in rows], dtype=np.float32)
    positions = np.arange(len(attrs))

    fig, ax = plt.subplots(figsize=(max(8.0, len(attrs) * 0.5), 4.5))
    ax.bar(positions - 0.18, uni, width=0.36, label="UNI", color="#2b6cb0")
    ax.bar(positions + 0.18, tme, width=0.36, label="O₂/Glc", color="#dd6b20")
    ax.set_ylabel("CV R^2")
    ax.set_xticks(positions)
    def _panel_a_label(attr: str) -> str:
        lbl = _display_attr(attr).lower().replace(" fraction", " frac")
        return re.sub(r"\b(h|e)\b", lambda m: m.group().upper(), lbl)
    ax.set_xticklabels([_panel_a_label(a) for a in attrs], rotation=45, ha="right")
    ax.legend(frameon=False)
    ax.set_title("Panel A: UNI vs TME Probe Performance")
    fig.tight_layout()
    panel_path = figure_dir / "panel_a_probe_R2.png"
    fig.savefig(panel_path, dpi=200)
    plt.close(fig)
    return panel_path


def render_panel_b(out_dir: str | Path) -> Path:
    out_path = Path(out_dir)
    figure_dir = ensure_directory(out_path / "figures")
    sweep_root = out_path / "sweep"
    attrs: list[str] = []
    targeted_slopes: list[float] = []
    random_slopes: list[float] = []
    for summary_path in sorted(sweep_root.glob("*/slope_summary.json")):
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        attrs.append(str(payload["attr"]))
        targeted = payload.get("targeted", {})
        random = payload.get("random", {})
        targeted_slopes.append(float(targeted.get("slope_mean", float("nan"))))
        random_slopes.append(float(random.get("slope_mean", float("nan"))))

    positions = np.arange(len(attrs))
    fig, ax = plt.subplots(figsize=(max(6.0, len(attrs) * 0.75), 4.5))
    ax.bar(positions - 0.18, targeted_slopes, width=0.36, color="#2f855a", label="Targeted")
    ax.bar(positions + 0.18, random_slopes, width=0.36, color="#a0aec0", label="Random")
    ax.axhline(0.0, color="#4a5568", linewidth=1.0)
    ax.set_ylabel("Slope")
    ax.set_xticks(positions)
    ax.set_xticklabels(attrs, rotation=45, ha="right")
    ax.legend(frameon=False)
    ax.set_title("Panel B: Sweep Slopes")
    fig.tight_layout()
    panel_path = figure_dir / "panel_b_sweep_slope.png"
    fig.savefig(panel_path, dpi=200)
    plt.close(fig)
    return panel_path


def render_panel_c(out_dir: str | Path) -> Path:
    out_path = Path(out_dir)
    figure_dir = ensure_directory(out_path / "figures")
    null_root = out_path / "null"
    attrs: list[str] = []
    targeted_means: list[float] = []
    random_means: list[float] = []
    full_null_means: list[float] = []
    for summary_path in sorted(null_root.glob("*/null_comparison.json")):
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        attrs.append(str(payload["attr"]))
        targeted_means.append(float(payload.get("targeted", {}).get("metric_mean", float("nan"))))
        random_means.append(float(payload.get("random", {}).get("metric_mean", float("nan"))))
        full_null_means.append(float(payload.get("full_uni_null", {}).get("metric_mean", float("nan"))))

    positions = np.arange(len(attrs))
    fig, ax = plt.subplots(figsize=(max(6.0, len(attrs) * 0.8), 4.5))
    ax.plot(positions, targeted_means, marker="o", color="#c53030", label="Targeted null")
    ax.plot(positions, random_means, marker="o", color="#718096", label="Random null")
    if any(np.isfinite(full_null_means)):
        ax.plot(positions, full_null_means, marker="o", color="#2b6cb0", label="Full UNI null")
    ax.set_ylabel("Target metric")
    ax.set_xticks(positions)
    ax.set_xticklabels(attrs, rotation=45, ha="right")
    ax.legend(frameon=False)
    ax.set_title("Panel C: Null Comparison")
    fig.tight_layout()
    panel_path = figure_dir / "panel_c_null_drop.png"
    fig.savefig(panel_path, dpi=200)
    plt.close(fig)
    return panel_path


def render_panel_d(out_dir: str | Path) -> Path:
    out_path = Path(out_dir)
    figure_dir = ensure_directory(out_path / "figures")
    csv_path = out_path / "appearance_sweep_summary.csv"
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    return _render_grouped_metric_grid(
        rows,
        metric_key_specs=[
            ("targeted_slope_mean", "Targeted", "#2f855a"),
            ("random_slope_mean", "Random", "#a0aec0"),
        ],
        title="Panel D: Appearance Sweep Slopes Across All Metrics",
        y_label="Slope",
        panel_path=figure_dir / "panel_d_appearance_sweep_all_metrics.png",
    )


def render_panel_e(out_dir: str | Path) -> Path:
    out_path = Path(out_dir)
    figure_dir = ensure_directory(out_path / "figures")
    csv_path = out_path / "appearance_null_summary.csv"
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    return _render_grouped_metric_grid(
        rows,
        metric_key_specs=[
            ("targeted_mean", "Targeted null", "#c53030"),
            ("random_mean", "Random null", "#718096"),
            ("full_uni_null_mean", "Full UNI null", "#2b6cb0"),
        ],
        title="Panel E: Appearance Null Readouts Across All Metrics",
        y_label="Metric value",
        panel_path=figure_dir / "panel_e_appearance_null_all_metrics.png",
    )


def _specificity_metric_title(metric_name: str) -> str:
    if metric_name.startswith("morpho."):
        return metric_name.removeprefix("morpho.").replace("_", " ")
    return _appearance_metric_title(metric_name)


def _specificity_plot_payload(rows: list[dict[str, str]]) -> tuple[list[str], list[str], np.ndarray, np.ndarray]:
    edited = _ordered_attrs({r["edited_attr"] for r in rows}, PREFERRED_SWEEP_ATTRS)

    morpho_metrics = sorted({r["measured_metric"] for r in rows if r["family"] == "morpho"})
    appearance_metrics = sorted({r["measured_metric"] for r in rows if r["family"] == "appearance"})
    metrics = morpho_metrics + appearance_metrics

    grid = np.full((len(edited), len(metrics)), np.nan, dtype=np.float32)
    annot = np.full((len(edited), len(metrics)), "", dtype=object)
    index = {(r["edited_attr"], r["measured_metric"]): r for r in rows}
    for i, attr in enumerate(edited):
        for j, metric in enumerate(metrics):
            row = index.get((attr, metric))
            if row is None:
                continue
            normalized = float(row.get("normalized_targeted_slope") or "nan")
            grid[i, j] = normalized
            ratio = row.get("abs_ratio")
            try:
                ratio_value = float(ratio)
                annot[i, j] = f"{ratio_value:.1f}x" if np.isfinite(ratio_value) else ""
            except (TypeError, ValueError):
                annot[i, j] = ""
    return edited, metrics, grid, annot


def _draw_specificity_diagonal_and_annotations(
    ax: plt.Axes,
    *,
    edited: list[str],
    metrics: list[str],
    annot: np.ndarray,
    fontsize: int,
    grid: np.ndarray | None = None,
    vlim: float | None = None,
) -> None:
    diag_metrics = {
        "eccentricity_mean": "morpho.eccentricity_mean",
        "nuclear_area_mean": "morpho.nuclear_area_mean",
        "nuclei_density": "morpho.nuclei_density",
        "texture_e_contrast": "appearance.texture_e_contrast",
        "texture_h_contrast": "appearance.texture_h_contrast",
        "texture_h_energy": "appearance.texture_h_energy",
    }
    for i, attr in enumerate(edited):
        diag_metric = diag_metrics.get(attr)
        for j, metric in enumerate(metrics):
            if annot[i, j]:
                text_color = "#1a202c"
                if grid is not None and vlim is not None and np.isfinite(grid[i, j]) and vlim > 0 and abs(float(grid[i, j])) / vlim > 0.5:
                    text_color = "white"
                ax.text(j, i, annot[i, j], ha="center", va="center", fontsize=fontsize, color=text_color)



def _specificity_square_payload(rows: list[dict[str, str]]) -> tuple[list[str], np.ndarray, np.ndarray]:
    edited, metrics, grid, annot = _specificity_plot_payload(rows)
    metric_index = {metric: index for index, metric in enumerate(metrics)}
    square_grid = np.full((len(edited), len(edited)), np.nan, dtype=np.float32)
    square_annot = np.full((len(edited), len(edited)), "", dtype=object)
    for row_index, _edited_attr in enumerate(edited):
        for column_index, metric_attr in enumerate(edited):
            primary_metric = ATTR_TO_PRIMARY_METRIC.get(metric_attr)
            if primary_metric is None:
                continue
            source_index = metric_index.get(primary_metric)
            if source_index is None:
                continue
            square_grid[row_index, column_index] = grid[row_index, source_index]
            square_annot[row_index, column_index] = annot[row_index, source_index]
    return edited, square_grid, square_annot


def _pick_sweep_tiles(attr_dir: Path, n_tiles: int = 6) -> list[Path]:
    valid_tiles: list[Path] = []
    for tile_dir in sorted(path for path in attr_dir.iterdir() if path.is_dir()):
        has_all_pngs = all(
            (tile_dir / "targeted" / f"alpha_{alpha}.png").is_file()
            for alpha, _label in SWEEP_ALPHA_ROWS
        )
        if has_all_pngs:
            valid_tiles.append(tile_dir)
        if len(valid_tiles) == n_tiles:
            break
    return valid_tiles


def _ordered_sweep_attrs(sweep_root: Path) -> list[str]:
    available = {path.name for path in sweep_root.iterdir() if path.is_dir()}
    return _ordered_attrs(available, PREFERRED_SWEEP_ATTRS)


def _resolve_reference_he_root(out_path: Path) -> Path:
    for candidate in (out_path / "he", DEFAULT_A4_DATA_ROOT / "he"):
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"could not resolve H&E root for {out_path}")


def _resolve_exp_channels_root(out_path: Path) -> Path:
    for candidate in (out_path / "exp_channels", DEFAULT_A4_DATA_ROOT / "exp_channels"):
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"could not resolve experimental channels root for {out_path}")


def _find_reference_he_path(he_root: Path, tile_id: str) -> Path:
    for suffix in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
        candidate = he_root / f"{tile_id}{suffix}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"reference H&E missing for tile {tile_id} under {he_root}")


def _load_cell_layout_image(exp_channels_root: Path, tile_id: str) -> np.ndarray:
    cell_layout = _load_channel_array(exp_channels_root, tile_id, "cell_masks", resolution=256, missing_ok=False)
    if cell_layout is None:
        raise FileNotFoundError(f"cell_masks missing for tile {tile_id} under {exp_channels_root}")
    return cell_layout


def render_panel_f(out_dir: str | Path) -> Path:
    """6 (edited attr) x N (measured metric) specificity heatmap.

    Cell color = normalized targeted slope (z-scored by metric baseline std).
    Cell annotation = abs_ratio (|targeted| / |random|). Diagonal cells get a
    bold border to highlight the edit-self alignment.
    """
    out_path = Path(out_dir)
    figure_dir = ensure_directory(out_path / "figures")
    csv_path = out_path / "specificity_full.csv"
    rows = _read_csv_rows(csv_path)
    edited, metrics, grid, annot = _specificity_plot_payload(rows)

    finite = grid[np.isfinite(grid)]
    vlim = float(np.quantile(np.abs(finite), 0.95)) if finite.size else 1.0
    if vlim == 0.0:
        vlim = 1.0

    fig, ax = plt.subplots(figsize=(max(10.0, 0.7 * len(metrics)), max(4.5, 0.7 * len(edited))))
    im = ax.imshow(grid, cmap="RdBu_r", vmin=-vlim, vmax=vlim, aspect="auto")
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([_specificity_metric_title(m) for m in metrics], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(edited)))
    ax.set_yticklabels(edited, fontsize=9)
    ax.set_xlabel("Measured metric", fontsize=10)
    ax.set_ylabel("Edited attribute", fontsize=10)
    ax.set_title("Panel F: Specificity matrix (z-scored targeted slope; cell label = |targeted|/|random| ratio)", fontsize=11)

    _draw_specificity_diagonal_and_annotations(ax, edited=edited, metrics=metrics, annot=annot, fontsize=7)

    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Normalized targeted slope (per metric std)", fontsize=9)
    fig.tight_layout()
    panel_path = figure_dir / "panel_f_specificity_matrix.png"
    fig.savefig(panel_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return panel_path


def render_panel_g(out_dir: str | Path) -> Path:
    """Per (edited appearance attr) compare global vs nucleus vs stroma targeted slopes.

    Focuses on the three appearance-edited attrs (texture_e_contrast,
    texture_h_contrast, texture_h_energy) because that is where the
    distributed-vs-localized question is most meaningful.
    """
    out_path = Path(out_dir)
    figure_dir = ensure_directory(out_path / "figures")
    csv_path = out_path / "appearance_global_vs_regional.csv"
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))

    focus_attrs = ["texture_e_contrast", "texture_h_contrast", "texture_h_energy"]
    metrics = sorted({row["metric"] for row in rows})
    index = {(row["attr"], row["metric"]): row for row in rows}

    fig, axes = plt.subplots(1, len(focus_attrs), figsize=(5.0 * len(focus_attrs), 6.5), squeeze=False)
    x = np.arange(len(metrics))
    width = 0.27
    bar_specs = [
        ("global_targeted_slope", "Global", "#1a202c"),
        ("nuc_targeted_slope", "Nucleus", "#c53030"),
        ("stroma_targeted_slope", "Stroma", "#2b6cb0"),
    ]
    offsets = (-width, 0.0, width)

    for axis, attr in zip(axes[0], focus_attrs):
        for offset, (key, label, color) in zip(offsets, bar_specs):
            vals = []
            for metric in metrics:
                row = index.get((attr, metric))
                vals.append(float(row[key]) if row and row.get(key) not in (None, "") else float("nan"))
            axis.bar(x + offset, vals, width=width, label=label, color=color)
        axis.axhline(0.0, color="#4a5568", linewidth=0.7)
        axis.set_title(f"Edit: {attr}", fontsize=11)
        axis.set_xticks(x)
        axis.set_xticklabels([m.removeprefix("appearance.") for m in metrics], rotation=45, ha="right", fontsize=8)
        axis.set_ylabel("Targeted slope (metric / alpha)", fontsize=9)
        axis.legend(fontsize=8)

    fig.suptitle("Panel G: Appearance compartment breakdown — global vs nucleus vs stroma targeted slope", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    panel_path = figure_dir / "panel_g_appearance_compartment.png"
    fig.savefig(panel_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return panel_path


def render_pngs_updated_probe_delta(out_dir: str | Path, dest_dir: str | Path) -> Path:
    out_path = Path(out_dir)
    rows = _read_csv_rows(out_path / "probe_results.csv")
    # Drop attrs with no valid folds
    rows = [row for row in rows if int(row.get("uni_n_valid_folds") or 0) > 0 or int(row.get("tme_n_valid_folds") or 0) > 0]

    _APPEARANCE_SET = set(APPEARANCE_ATTR_NAMES)
    _MORPHOLOGY_SET = set(MORPHOLOGY_ATTR_NAMES)
    # Exclude attrs trivially predicted by their own channel mean (TME R² ≈ 1 by construction)
    _EXCLUDE_ATTRS = {"vessel_area_pct", "mean_oxygen", "mean_glucose"}

    # category → (color, hollow marker shape)
    _CAT_STYLE: dict[str, tuple[str, str]] = {
        "appearance": ("#dd6b20", "o"),
        "morphology": ("#4a90d9", "s"),
        "cell composition": ("#52b788", "^"),
    }

    def _attr_category(attr: str) -> str:
        if attr in _APPEARANCE_SET:
            return "appearance"
        if attr in _MORPHOLOGY_SET:
            return "morphology"
        return "cell composition"

    # Collect valid points first so we can compute axis limits
    points: list[tuple[str, str, float, float, float, float]] = []
    for row in rows:
        attr = row["attr"]
        uni_r2 = float(row.get("uni_r2_mean") or "nan")
        tme_r2 = float(row.get("tme_r2_mean") or "nan")
        uni_std = float(row.get("uni_r2_std") or "nan")
        tme_std = float(row.get("tme_r2_std") or "nan")
        if not (np.isfinite(uni_r2) and np.isfinite(tme_r2)):
            continue
        if attr in _EXCLUDE_ATTRS:
            continue
        points.append((_attr_category(attr), attr, tme_r2, uni_r2,
                       tme_std if np.isfinite(tme_std) else 0.0,
                       uni_std if np.isfinite(uni_std) else 0.0))

    all_y = [p[3] for p in points]
    all_x = [p[2] for p in points]
    # Single shared range for both axes so the diagonal is exactly 45° and axes are square
    all_vals = all_x + all_y
    ax_lo = min(all_vals) - 0.1
    ax_hi = max(all_vals) + 0.05

    font_name = "Nimbus Sans"
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    ax.set_aspect("equal")
    ax.set_axisbelow(True)
    ax.grid(linewidth=0.4, color="#E0E0E0", zorder=0)
    ax.plot([ax_lo, ax_hi], [ax_lo, ax_hi],
            color="black", linewidth=0.8, linestyle="--", zorder=1)

    plotted_categories: set[str] = set()
    texts: list = []
    for cat, attr, tme_r2, uni_r2, tme_std, uni_std in points:
        color, marker = _CAT_STYLE[cat]
        legend_label = cat if cat not in plotted_categories else None
        plotted_categories.add(cat)
        ax.errorbar(
            tme_r2, uni_r2,
            xerr=tme_std, yerr=uni_std,
            fmt=marker, color=color, ecolor=color,
            markerfacecolor="white", markeredgecolor=color,
            markeredgewidth=1.2, elinewidth=0.7,
            capsize=2, markersize=5, label=legend_label, zorder=3,
        )
        _label = _display_attr(attr).lower().replace(" fraction", " frac")
        _label = re.sub(r"\b(h|e)\b", lambda m: m.group().upper(), _label)
        txt = ax.text(
            tme_r2, uni_r2, _label,
            fontsize=6.5, color="black",
            fontfamily=font_name, zorder=4,
        )
        texts.append(txt)

    adjust_text(
        texts, ax=ax,
        expand=(1.15, 1.3),
        arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.5),
    )

    ax.set_xlim(ax_lo, ax_hi)
    ax.set_ylim(ax_lo, ax_hi)
    _bundle_path = out_path / "features.npz"
    _tme_label = "TME"
    if _bundle_path.is_file():
        _bundle = np.load(_bundle_path, allow_pickle=True)
        if "tme_label" in _bundle:
            _tme_label = str(_bundle["tme_label"])
    ax.set_xlabel(f"{_tme_label} R²", fontsize=9, fontfamily=font_name)
    ax.set_ylabel("UNI R²", fontsize=9, fontfamily=font_name)
    ax.tick_params(labelsize=8)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontfamily(font_name)
    handles = [
        Line2D([0], [0], marker=_CAT_STYLE[cat][1], color=_CAT_STYLE[cat][0],
               markerfacecolor="white", markeredgecolor=_CAT_STYLE[cat][0],
               markeredgewidth=1.2, markersize=6, linestyle="none", label=cat)
        for cat in ("appearance", "morphology", "cell composition")
        if cat in plotted_categories
    ]
    if handles:
        ax.legend(handles=handles, fontsize=7.0,
                  loc="upper center", bbox_to_anchor=(0.5, -0.12),
                  ncol=len(handles),
                  prop={"family": font_name, "size": 7.0},
                  frameon=False,
                  handlelength=1.4, columnspacing=0.5, handletextpad=0.3)
    fig.tight_layout()

    output_path = ensure_directory(Path(dest_dir)) / "probe_delta_r2.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def render_pngs_updated_sweep_grid(out_dir: str | Path, dest_dir: str | Path, attr: str, n_tiles: int = 3) -> Path:
    """4-row × n_tiles grid: Row0=Ref H&E, Rows1-3=alpha=-1,0,+1 (targeted).
    No random section. No section headers. Attr display name as suptitle.
    """
    out_path = Path(out_dir)
    attr_dir = out_path / "sweep" / attr
    tile_dirs = _pick_sweep_tiles(attr_dir, n_tiles=n_tiles)
    if len(tile_dirs) < n_tiles:
        raise ValueError(f"expected at least {n_tiles} valid tiles for {attr}, found {len(tile_dirs)}")

    he_root = _resolve_reference_he_root(out_path)
    tile_ids = [tile_dir.name for tile_dir in tile_dirs]
    reference_images = {tile_id: mpimg.imread(_find_reference_he_path(he_root, tile_id)) for tile_id in tile_ids}

    n_rows = 4  # Row 0 = Ref H&E, rows 1-3 = alpha
    figsize = (n_tiles * 1.25 + 0.5, n_rows * 1.25 + 0.8)

    fig, axes = plt.subplots(
        n_rows, n_tiles,
        figsize=figsize,
        gridspec_kw={"wspace": 0.03, "hspace": 0.03},
        squeeze=False,
    )
    fig.subplots_adjust(left=0.19, right=0.99, top=0.91, bottom=0.01)

    # Row 0: Reference H&E
    for col, tile_id in enumerate(tile_ids):
        ax = axes[0, col]
        ax.imshow(reference_images[tile_id])
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Rows 1-3: alpha-directed (targeted only)
    for row, (alpha, _row_label) in enumerate(SWEEP_ALPHA_ROWS, start=1):
        for col, tile_dir in enumerate(tile_dirs):
            ax = axes[row, col]
            img_path = tile_dir / "targeted" / f"alpha_{alpha}.png"
            ax.imshow(mpimg.imread(img_path))
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

    # Row labels on left column
    row_labels = ["Ref H&E"] + [lbl for _, lbl in SWEEP_ALPHA_ROWS]
    for row_idx, lbl in enumerate(row_labels):
        axes[row_idx, 0].set_ylabel(
            lbl, fontsize=13, rotation=0, ha="right", va="center", labelpad=4
        )

    # Attribute name as suptitle
    fig.suptitle(
        SWEEP_ATTR_DISPLAY_NAMES.get(attr, attr),
        fontsize=14, fontweight="bold", y=0.995,
    )

    output_path = ensure_directory(Path(dest_dir)) / f"sweep_grid_{attr}.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def render_pngs_updated_combined_sweep_grid(out_dir: str | Path, dest_dir: str | Path) -> Path:
    """6 attribute sweep grids in one matplotlib figure (horizontal ribbon).

    - 4 rows × 3 cols per attribute block, all 6 blocks side by side.
    - Y-labels (row names) only on the leftmost block.
    - Attr display name centered over its grid, with a black horizontal line below
      the text and above the tile rows.
    - Tight wspace/hspace between tiles and between attribute blocks.
    """
    out_path = Path(out_dir)
    dest_path = ensure_directory(Path(dest_dir))

    n_attrs = len(SWEEP_ATTRS)  # 6
    n_tiles = 3
    n_rows = 4  # Ref H&E + 3 alpha rows

    he_root = _resolve_reference_he_root(out_path)

    # Pre-load all tile data
    attr_data: dict[str, dict] = {}
    for attr in SWEEP_ATTRS:
        attr_dir = out_path / "sweep" / attr
        tile_dirs = _pick_sweep_tiles(attr_dir, n_tiles=n_tiles)
        tile_ids = [td.name for td in tile_dirs]
        attr_data[attr] = {
            "tile_dirs": tile_dirs,
            "tile_ids": tile_ids,
            "ref_imgs": {
                tid: mpimg.imread(_find_reference_he_path(he_root, tid))
                for tid in tile_ids
            },
        }

    # Column layout: n_tiles tile cols per attr + thin spacer between attrs
    spacer_r = 0.06  # spacer width as fraction of one tile width
    col_ratios: list[float] = []
    for i in range(n_attrs):
        col_ratios.extend([1.0] * n_tiles)
        if i < n_attrs - 1:
            col_ratios.append(spacer_r)
    n_grid_cols = len(col_ratios)  # 6*3 + 5 = 23

    # Figure sizing (aim for ~0.88" square tiles)
    TILE_IN = 0.88
    LEFT_M = 1.10   # room for y-labels (increased for larger font)
    RIGHT_M = 0.04
    TOP_M = 0.55    # room for subtitle text + line (increased for larger font)
    BOT_M = 0.04
    SPACER_IN = TILE_IN * spacer_r

    fig_w = LEFT_M + n_attrs * (n_tiles * TILE_IN) + (n_attrs - 1) * SPACER_IN + RIGHT_M
    fig_h = TOP_M + n_rows * TILE_IN + BOT_M

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
    gs = fig.add_gridspec(
        n_rows, n_grid_cols,
        width_ratios=col_ratios,
        wspace=0.02,
        hspace=0.02,
        left=LEFT_M / fig_w,
        right=1.0 - RIGHT_M / fig_w,
        top=1.0 - TOP_M / fig_h,
        bottom=BOT_M / fig_h,
    )

    row_labels = ["Ref H&E"] + [lbl for _, lbl in SWEEP_ALPHA_ROWS]

    # Store top-left / top-right axes per attr block for label/line placement
    block_axes: dict[int, dict[str, plt.Axes]] = {}

    for attr_idx, attr in enumerate(SWEEP_ATTRS):
        data = attr_data[attr]
        tile_dirs = data["tile_dirs"]
        tile_ids = data["tile_ids"]
        ref_imgs = data["ref_imgs"]
        gs_col0 = attr_idx * (n_tiles + 1)  # tiles + 1 spacer step

        block_axes[attr_idx] = {}
        for row in range(n_rows):
            for col in range(n_tiles):
                ax = fig.add_subplot(gs[row, gs_col0 + col])
                if row == 0 and col == 0:
                    block_axes[attr_idx]["top_left"] = ax
                if row == 0 and col == n_tiles - 1:
                    block_axes[attr_idx]["top_right"] = ax

                if row == 0:
                    ax.imshow(ref_imgs[tile_ids[col]])
                else:
                    alpha, _ = SWEEP_ALPHA_ROWS[row - 1]
                    img_path = tile_dirs[col] / "targeted" / f"alpha_{alpha}.png"
                    ax.imshow(mpimg.imread(img_path))

                ax.set_xticks([])
                ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(False)

                # Y-labels only on leftmost column of first attr
                if attr_idx == 0 and col == 0:
                    ax.set_ylabel(
                        row_labels[row], fontsize=20, rotation=0,
                        ha="right", va="center", labelpad=4,
                    )

    # Add subtitle text and black line above each attr block
    fig.canvas.draw()

    for attr_idx, attr in enumerate(SWEEP_ATTRS):
        pos_tl = block_axes[attr_idx]["top_left"].get_position()
        pos_tr = block_axes[attr_idx]["top_right"].get_position()

        x_left = pos_tl.x0
        x_right = pos_tr.x1
        x_center = (x_left + x_right) / 2
        y_tile_top = pos_tl.y1

        line_y = y_tile_top + 0.012
        text_y = line_y + 0.008

        fig.add_artist(
            Line2D(
                [x_left, x_right], [line_y, line_y],
                transform=fig.transFigure,
                color="black", linewidth=1.2,
            )
        )
        fig.text(
            x_center, text_y,
            SWEEP_ATTR_DISPLAY_NAMES.get(attr, attr),
            ha="center", va="bottom",
            fontsize=20, fontweight="normal",
        )

    output_path = dest_path / "sweep_grid_combined.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def render_pngs_updated_combined_abc(out_dir: str | Path, dest_dir: str | Path) -> Path:
    """Combined figure: [A: probe_delta | B: specificity_heatmap] on top,
    [C: sweep_combined] below. Width(A) + Width(B) = Width(C).

    A is padded with white below to match B's height. C is scaled to match A+B total width.
    """
    out_path = Path(out_dir)
    dest_path = ensure_directory(Path(dest_dir))

    # Ensure C exists
    sweep_combined_path = dest_path / "sweep_grid_combined.png"
    if not sweep_combined_path.is_file():
        render_pngs_updated_combined_sweep_grid(out_path, dest_path)

    img_A = np.array(Image.open(dest_path / "probe_delta_r2.png").convert("RGB"))
    img_B = np.array(Image.open(dest_path / "specificity_heatmap.png").convert("RGB"))
    img_C = np.array(Image.open(sweep_combined_path).convert("RGB"))

    H_A, W_A = img_A.shape[:2]
    H_B, W_B = img_B.shape[:2]
    H_C, W_C = img_C.shape[:2]

    # Scale A up slightly and B down so both reach the geometric-mean height,
    # preserving each image's aspect ratio.
    H_target = round(float(np.sqrt(H_A * H_B)))
    W_A_new = round(W_A * H_target / H_A)
    W_B_new = round(W_B * H_target / H_B)

    img_A_r = np.array(Image.fromarray(img_A).resize((W_A_new, H_target), Image.LANCZOS))
    img_B_r = np.array(Image.fromarray(img_B).resize((W_B_new, H_target), Image.LANCZOS))

    img_AB = np.concatenate([img_A_r, img_B_r], axis=1)  # (H_target, W_A_new+W_B_new, 3)
    W_AB = img_AB.shape[1]

    # Scale C so its width matches W_AB
    H_C_scaled = max(1, round(H_C * W_AB / W_C))
    img_C_scaled = np.array(
        Image.fromarray(img_C).resize((W_AB, H_C_scaled), Image.LANCZOS)
    )

    # No separator — stack directly
    combined = np.concatenate([img_AB, img_C_scaled], axis=0)

    # Draw panel labels A, B, C using matplotlib (data coords)
    _DPI = 180
    fig, ax = plt.subplots(
        figsize=(combined.shape[1] / _DPI, combined.shape[0] / _DPI),
        facecolor="white",
    )
    ax.imshow(combined, interpolation="bilinear")
    ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1, 0, 0)

    label_pad = max(6, int(0.015 * combined.shape[1]))
    top_pad = max(6, int(0.015 * combined.shape[0]))

    # A: top-left of A panel
    ax.text(label_pad, top_pad, "A", fontsize=20, fontweight="bold",
            color="black", ha="left", va="top")
    # B: top-left of B panel
    ax.text(W_A_new + label_pad, top_pad, "B", fontsize=20, fontweight="bold",
            color="black", ha="left", va="top")
    # C: top-left of C panel (starts after H_target rows)
    ax.text(label_pad, H_target + top_pad, "C", fontsize=20, fontweight="bold",
            color="black", ha="left", va="top")

    output_path = dest_path / "combined_a_b_c.png"
    fig.savefig(output_path, dpi=_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def render_pngs_updated_specificity_heatmap(out_dir: str | Path, dest_dir: str | Path) -> Path:
    out_path = Path(out_dir)
    rows = _read_csv_rows(out_path / "specificity_full.csv")
    edited, grid, _abs_ratio_annot = _specificity_square_payload(rows)

    # Annotations show the same normalized slope that drives the color
    annot = np.full((len(edited), len(edited)), "", dtype=object)
    for i in range(len(edited)):
        for j in range(len(edited)):
            v = grid[i, j]
            if np.isfinite(v):
                annot[i, j] = f"{v:.2f}"

    finite = grid[np.isfinite(grid)]
    vlim = float(np.quantile(np.abs(finite), 0.95)) if finite.size else 1.0
    if vlim == 0.0:
        vlim = 1.0

    fig, ax = plt.subplots(figsize=(max(6.0, 0.9 * len(edited)), max(6.0, 0.9 * len(edited))))
    im = ax.imshow(grid, cmap="RdBu_r", vmin=-vlim, vmax=vlim, aspect="equal")
    ax.set_xticks(range(len(edited)))
    ax.set_xticklabels([_display_attr(e) for e in edited], rotation=35, ha="right", fontsize=12)
    ax.set_yticks(range(len(edited)))
    ax.set_yticklabels([_display_attr(e) for e in edited], fontsize=12)
    ax.set_xticks(np.arange(-0.5, len(edited), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(edited), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_xlabel("Measured metric", fontsize=13)
    ax.set_ylabel("Edited attribute", fontsize=13)

    _draw_specificity_diagonal_and_annotations(
        ax,
        edited=edited,
        metrics=[ATTR_TO_PRIMARY_METRIC[attr] for attr in edited],
        annot=annot,
        fontsize=11,
        grid=grid,
        vlim=vlim,
    )

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.12)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_ticks([-8, -4, 0, 4, 8])
    cbar.ax.tick_params(labelsize=11)
    fig.tight_layout()

    output_path = ensure_directory(Path(dest_dir)) / "specificity_heatmap.png"
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return output_path


def render_pngs_updated(out_dir: str | Path, dest_dir: str | Path) -> dict[str, Path]:
    out_path = Path(out_dir)
    destination = ensure_directory(Path(dest_dir))
    outputs = {
        "probe_delta_r2": render_pngs_updated_probe_delta(out_path, destination),
        "specificity_heatmap": render_pngs_updated_specificity_heatmap(out_path, destination),
    }
    for attr in _ordered_sweep_attrs(out_path / "sweep"):
        outputs[f"sweep_grid_{attr}"] = render_pngs_updated_sweep_grid(out_path, destination, attr)
    outputs["sweep_grid_combined"] = render_pngs_updated_combined_sweep_grid(out_path, destination)
    outputs["combined_a_b_c"] = render_pngs_updated_combined_abc(out_path, destination)
    return outputs


def render_all(out_dir: str | Path) -> dict[str, Path]:
    out_path = Path(out_dir)
    outputs: dict[str, Path] = {}
    if (out_path / "probe_results.csv").is_file():
        outputs["panel_a"] = render_panel_a(out_path)
    if any((out_path / "sweep").glob("*/slope_summary.json")):
        outputs["panel_b"] = render_panel_b(out_path)
    if any((out_path / "null").glob("*/null_comparison.json")):
        outputs["panel_c"] = render_panel_c(out_path)
    if (out_path / "appearance_sweep_summary.csv").is_file():
        outputs["panel_d"] = render_panel_d(out_path)
    if (out_path / "appearance_null_summary.csv").is_file():
        outputs["panel_e"] = render_panel_e(out_path)
    if (out_path / "specificity_full.csv").is_file():
        outputs["panel_f"] = render_panel_f(out_path)
    if (out_path / "appearance_global_vs_regional.csv").is_file():
        outputs["panel_g"] = render_panel_g(out_path)
    return outputs
