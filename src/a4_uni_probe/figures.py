"""Render summary figures for probe, sweep, and null results."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from src._tasklib.io import ensure_directory
from src.a4_uni_probe.labels import _load_channel_array


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
    ax.bar(positions + 0.18, tme, width=0.36, label="TME", color="#dd6b20")
    ax.set_ylabel("CV R^2")
    ax.set_xticks(positions)
    ax.set_xticklabels(attrs, rotation=45, ha="right")
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
                ax.text(j, i, annot[i, j], ha="center", va="center", fontsize=fontsize, color="#1a202c")
            if diag_metric == metric:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="#1a202c", linewidth=2.0))


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
            (tile_dir / direction / f"alpha_{alpha}.png").is_file()
            for direction in ("targeted", "random")
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
    sorted_rows = sorted(
        rows,
        key=lambda row: (
            not np.isfinite(float(row.get("delta_r2_uni_minus_tme") or "nan")),
            -float(row.get("delta_r2_uni_minus_tme") or "nan") if np.isfinite(float(row.get("delta_r2_uni_minus_tme") or "nan")) else 0.0,
        ),
    )

    attrs = [row["attr"] for row in sorted_rows]
    deltas = np.asarray([float(row.get("delta_r2_uni_minus_tme") or "nan") for row in sorted_rows], dtype=np.float32)
    errors = np.sqrt(
        np.square(np.asarray([float(row.get("uni_r2_std") or "nan") for row in sorted_rows], dtype=np.float32))
        + np.square(np.asarray([float(row.get("tme_r2_std") or "nan") for row in sorted_rows], dtype=np.float32))
    )
    positions = np.arange(len(attrs))

    fig, ax = plt.subplots(figsize=(max(8.0, len(attrs) * 0.5), 4.5))
    ax.bar(positions, deltas, yerr=errors, capsize=3, color="#dd6b20", ecolor="#4a5568")
    ax.axhline(0.0, color="#4a5568", linewidth=0.8)
    ax.set_ylabel("ΔR² (UNI − TME)", fontsize=10)
    ax.set_xticks(positions)
    ax.set_xticklabels(attrs, rotation=45, ha="right", fontsize=9)
    ax.tick_params(axis="y", labelsize=9)
    fig.tight_layout()

    output_path = ensure_directory(Path(dest_dir)) / "probe_delta_r2.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def render_pngs_updated_sweep_grid(out_dir: str | Path, dest_dir: str | Path, attr: str, n_tiles: int = 6) -> Path:
    out_path = Path(out_dir)
    attr_dir = out_path / "sweep" / attr
    tile_dirs = _pick_sweep_tiles(attr_dir, n_tiles=n_tiles)
    if len(tile_dirs) < n_tiles:
        raise ValueError(f"expected at least {n_tiles} valid tiles for {attr}, found {len(tile_dirs)}")

    he_root = _resolve_reference_he_root(out_path)
    exp_channels_root = _resolve_exp_channels_root(out_path)
    tile_ids = [tile_dir.name for tile_dir in tile_dirs]
    reference_images = {tile_id: mpimg.imread(_find_reference_he_path(he_root, tile_id)) for tile_id in tile_ids}
    cell_layouts = {tile_id: _load_cell_layout_image(exp_channels_root, tile_id) for tile_id in tile_ids}

    group_names = ["Reference H&E", "Cell layout", "Targeted", "Random"]
    spacer_width = 0.28
    width_ratios: list[float] = []
    group_offsets: list[int] = []
    column_index = 0
    for group_index, _group_name in enumerate(group_names):
        group_offsets.append(column_index)
        width_ratios.extend([1.0] * n_tiles)
        column_index += n_tiles
        if group_index != len(group_names) - 1:
            width_ratios.append(spacer_width)
            column_index += 1

    fig = plt.figure(figsize=(24.0 * 0.8 + 2.2, 3.0 * 1.0 + 0.8))
    grid = fig.add_gridspec(
        3,
        len(width_ratios),
        width_ratios=width_ratios,
        wspace=0.05,
        hspace=0.05,
    )
    grouped_axes: dict[str, list[list[plt.Axes]]] = {group_name: [] for group_name in group_names}

    for row_index, (alpha, _row_label) in enumerate(SWEEP_ALPHA_ROWS):
        reference_row_axes: list[plt.Axes] = []
        layout_row_axes: list[plt.Axes] = []
        targeted_row_axes: list[plt.Axes] = []
        random_row_axes: list[plt.Axes] = []
        for col_index, tile_dir in enumerate(tile_dirs):
            tile_id = tile_dir.name

            reference_ax = fig.add_subplot(grid[row_index, group_offsets[0] + col_index])
            reference_ax.imshow(reference_images[tile_id])
            reference_ax.set_xticks([])
            reference_ax.set_yticks([])
            for spine in reference_ax.spines.values():
                spine.set_visible(False)
            reference_row_axes.append(reference_ax)

            layout_ax = fig.add_subplot(grid[row_index, group_offsets[1] + col_index])
            layout_ax.imshow(cell_layouts[tile_id], cmap="gray", vmin=0.0, vmax=1.0)
            layout_ax.set_xticks([])
            layout_ax.set_yticks([])
            for spine in layout_ax.spines.values():
                spine.set_visible(False)
            layout_row_axes.append(layout_ax)

            targeted_ax = fig.add_subplot(grid[row_index, group_offsets[2] + col_index])
            targeted_ax.imshow(mpimg.imread(tile_dir / "targeted" / f"alpha_{alpha}.png"))
            targeted_ax.set_xticks([])
            targeted_ax.set_yticks([])
            for spine in targeted_ax.spines.values():
                spine.set_visible(False)
            targeted_row_axes.append(targeted_ax)

            random_ax = fig.add_subplot(grid[row_index, group_offsets[3] + col_index])
            random_ax.imshow(mpimg.imread(tile_dir / "random" / f"alpha_{alpha}.png"))
            random_ax.set_xticks([])
            random_ax.set_yticks([])
            for spine in random_ax.spines.values():
                spine.set_visible(False)
            random_row_axes.append(random_ax)

        grouped_axes["Reference H&E"].append(reference_row_axes)
        grouped_axes["Cell layout"].append(layout_row_axes)
        grouped_axes["Targeted"].append(targeted_row_axes)
        grouped_axes["Random"].append(random_row_axes)

    fig.subplots_adjust(left=0.055, right=0.995, bottom=0.03, top=0.83)
    fig.canvas.draw()

    for row_index, (_alpha, row_label) in enumerate(SWEEP_ALPHA_ROWS):
        row_ax = grouped_axes["Reference H&E"][row_index][0]
        position = row_ax.get_position()
        fig.text(position.x0 - 0.012, (position.y0 + position.y1) / 2, row_label, ha="right", va="center", fontsize=11)

    bracket_y = grouped_axes["Reference H&E"][0][0].get_position().y1 + 0.02
    text_y = bracket_y + 0.008
    for group_name in group_names:
        top_left = grouped_axes[group_name][0][0].get_position()
        top_right = grouped_axes[group_name][0][-1].get_position()
        fig.add_artist(
            Line2D(
                [top_left.x0, top_right.x1],
                [bracket_y, bracket_y],
                transform=fig.transFigure,
                color="#000000",
                linewidth=1.5,
            )
        )
        fig.text((top_left.x0 + top_right.x1) / 2, text_y, group_name, ha="center", va="bottom", fontsize=11, fontweight="bold")

    output_path = ensure_directory(Path(dest_dir)) / f"sweep_grid_{attr}.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def render_pngs_updated_specificity_heatmap(out_dir: str | Path, dest_dir: str | Path) -> Path:
    out_path = Path(out_dir)
    rows = _read_csv_rows(out_path / "specificity_full.csv")
    edited, grid, annot = _specificity_square_payload(rows)

    finite = grid[np.isfinite(grid)]
    vlim = float(np.quantile(np.abs(finite), 0.95)) if finite.size else 1.0
    if vlim == 0.0:
        vlim = 1.0

    fig, ax = plt.subplots(figsize=(max(7.0, 1.05 * len(edited)), max(6.5, 1.05 * len(edited))))
    im = ax.imshow(grid, cmap="RdBu_r", vmin=-vlim, vmax=vlim, aspect="equal")
    ax.set_xticks(range(len(edited)))
    ax.set_xticklabels(edited, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(len(edited)))
    ax.set_yticklabels(edited, fontsize=10)
    ax.set_xlabel("Edited attribute", fontsize=10)
    ax.set_ylabel("Edited attribute", fontsize=10)
    ax.set_title("Specificity matrix (normalized targeted slope)", fontsize=12)

    _draw_specificity_diagonal_and_annotations(
        ax,
        edited=edited,
        metrics=[ATTR_TO_PRIMARY_METRIC[attr] for attr in edited],
        annot=annot,
        fontsize=9,
    )

    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label("Normalized targeted slope (per metric std)", fontsize=9)
    fig.tight_layout()

    output_path = ensure_directory(Path(dest_dir)) / "specificity_heatmap.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
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
