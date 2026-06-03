"""Combined Fig 1: Stage-1 approach + paired data characterization.

Three panels, two rows, full-width composite consistent with the other
`concat/` figures (220 DPI raster stack, ~15.8 in target width):

    Row 1:  A  = Stage-1 sub-step schematic (stage_1_svg.svg)
            B  = paired-data cell characterization, two columns:
                   B(i)  stacked Area / Circularity bar charts (median ± IQR)
                   B(ii) marker z-score heatmap (markers x cell-type x state)
    Row 2:  C  = real H&E + CODEX + segmentation + O2/glucose montage
                 (stage_1_results.png)

Panel B is rendered here from the CRC33 cell tables (no dependency on the
he-feature-visualizer repo beyond the two CSVs), at its final pixel size so its
text stays the same physical size as the other paper panels (vis_guidance:
never rescale a rendered text panel). The schematic (A) and montage (C) are
image rasters and are only ever *resized*, never their text re-typeset.

Run:  python -m src.paper_figures.fig_combined_stage1
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats as _scipy_stats

from src.paper_figures.style import (
    apply_style,
    FONT_SIZE_LABEL,
    FONT_SIZE_TICK,
    FONT_SIZE_ANNOTATION,
    FONT_SIZE_CELL_TEXT,
)
from src.paper_figures.fig_combined_method_perf import (
    _RENDER_DPI,
    _add_label_strip,
    _load_rgb,
    _pad_row_height,
    _resize_to_width,
    _svg_to_rgb_on_white,
)
from tools.ablation_report.shared import ROOT
from tools.ablation_report.figures import save_figure_png
from tools.color_constants import CELL_TYPES, CELL_TYPE_COLORS_NORM

# ── Inputs / outputs ────────────────────────────────────────────────────────
DATA_DIR = Path.home() / "he-feature-visualizer" / "processed_crc33"
SVG_PATH = ROOT / "figures" / "pngs_updated" / "methods" / "stage_1_svg.svg"
RESULTS_PNG = ROOT / "figures" / "pngs_updated" / "methods" / "stage_1_results.png"
OUT_PNG = ROOT / "figures" / "pngs_updated" / "concat" / "fig1_approach_data.png"

# ── Composition geometry (mirrors fig_combined_method_perf) ─────────────────
_SCHEMATIC_ASPECT = 2217 / 1053          # native stage_1_svg.svg width / height
_ROW1_GAP_PX = 40                        # white gutter between panels A and B
_PANELB_TARGET_ASPECT = 2.55             # B = bars | heatmap, landscape to sit beside A
_PANEL_LETTER_FS = 18                    # matches FONT_PANEL_LETTER across paper figs
COMPOSITE_WIDTH_IN = 15.8

# ── Panel B internal layout (fractions of the B raster) ─────────────────────
# Left column: two stacked bar charts; right column: heatmap + colorbar.
# B(ii) heatmap shares the bar stack's vertical span [_B_BOTTOM, _B_TOP], so its
# height equals (area height + circ height + the hspace between them) by build.
_B_TOP = 0.875
_B_BOTTOM = 0.175
_BAR_HSPACE = 0.045                                # gap between the two stacked bars
_BAR_H = (_B_TOP - _B_BOTTOM - _BAR_HSPACE) / 2
_BAR_LEFT = 0.070
_BAR_W = 0.220
_AREA_BOX = (_BAR_LEFT, _B_TOP - _BAR_H, _BAR_W, _BAR_H)   # x, y, w, h
_CIRC_BOX = (_BAR_LEFT, _B_BOTTOM, _BAR_W, _BAR_H)
_HEAT_LABELS_LEFT = 0.330                          # B(ii) letter sits above marker labels
_HEAT_BOX = (0.410, _B_BOTTOM, 0.480, _B_TOP - _B_BOTTOM)
_CBAR_BOX = (0.900, _B_BOTTOM, 0.013, _B_TOP - _B_BOTTOM)
_GROUP_HEADER_Y = _B_TOP + 0.020                   # cell-type headers above the heatmap

# ── Cell characterization config ────────────────────────────────────────────
_HE_MPP = 0.325  # um/px for CRC33 H&E (OME-XML PhysicalSizeX)
_STATE_ORDER = ["proliferative", "nonproliferative", "dead"]
# "nonprolif." is too wide for a heatmap column under DejaVu; wrap it so the
# column labels stay horizontal (unrotated) without colliding.
_STATE_SHORT = {"proliferative": "prolif.", "nonproliferative": "non-\nprolif.", "dead": "dead"}
_MARKERS = [
    "Hoechst", "Pan-CK", "E-cadherin", "CD45", "CD3e",
    "CD4", "CD8a", "CD20", "CD68", "Ki67",
]
_HEAT_CMAP = LinearSegmentedColormap.from_list(
    "nat_rdbu", [(0.0, "#B2182B"), (0.5, "#F7F7F7"), (1.0, "#2166AC")],
)
_TYPE_COLOR = {ct: CELL_TYPE_COLORS_NORM[ct][:3] for ct in CELL_TYPES}


# ── Data ────────────────────────────────────────────────────────────────────
def _normalize_state(value: str) -> str:
    state = value.strip().lower()
    if state in {"nonprolif", "nonproliferative"}:
        return "nonproliferative"
    if state.startswith("q") and state.endswith("cent"):  # quiescent
        return "nonproliferative"
    if state == "apoptotic":
        return "dead"
    return value


def _to_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def load_cell_table(data_dir: Path = DATA_DIR) -> dict:
    """Merge cell assignments with cached shape features (CRC33 tables).

    Returns a dict of parallel numpy arrays (no pandas dependency):
    ``cell_type``, ``cell_state``, ``area_um2``, ``circularity`` and a
    ``markers`` sub-dict of per-marker float arrays.
    """
    circ_by_key: dict[tuple[str, str], float] = {}
    with open(data_dir / "cell_shape_features.csv", newline="") as fh:
        for row in csv.DictReader(fh):
            circ_by_key[(row["CellID"], row["PatchID"])] = _to_float(row.get("circularity", ""))

    cell_type: list[str] = []
    cell_state: list[str] = []
    area_um2: list[float] = []
    circularity: list[float] = []
    markers: dict[str, list[float]] = {m: [] for m in _MARKERS}
    with open(data_dir / "cell_assignments.csv", newline="") as fh:
        for row in csv.DictReader(fh):
            cell_type.append(row["cell_type"])
            cell_state.append(_normalize_state(row["cell_state"]))
            area_um2.append(_to_float(row.get("Area_cellvit_px", "")) * (_HE_MPP ** 2))
            circularity.append(circ_by_key.get((row["CellID"], row["PatchID"]), np.nan))
            for m in _MARKERS:
                markers[m].append(_to_float(row.get(m, "")))

    return {
        "cell_type": np.array(cell_type),
        "cell_state": np.array(cell_state),
        "area_um2": np.array(area_um2, dtype=float),
        "circularity": np.array(circularity, dtype=float),
        "markers": {m: np.array(v, dtype=float) for m, v in markers.items()},
    }


def _zscore_markers(data: dict) -> dict[str, np.ndarray]:
    """Per-marker: clip at the 99th pct, then standardize (NaN-aware)."""
    out: dict[str, np.ndarray] = {}
    for m, raw in data["markers"].items():
        s = raw.copy()
        finite = np.isfinite(s)
        if finite.any():
            s = np.clip(s, None, np.nanpercentile(s, 99.0))
            std = np.nanstd(s)
            s = np.zeros_like(s) if (not np.isfinite(std) or std == 0.0) else (s - np.nanmean(s)) / std
        out[m] = s
    return out


def _type_arrays(data: dict, key: str, clip_pct: float = 99.0) -> list[np.ndarray]:
    arrays = []
    for ct in CELL_TYPES:
        vals = data[key][data["cell_type"] == ct]
        vals = vals[np.isfinite(vals)]
        if vals.size:
            vals = vals[vals <= np.percentile(vals, clip_pct)]
        arrays.append(vals)
    return arrays


# ── Bar chart (median +/- IQR, Kruskal-Wallis + Mann-Whitney brackets) ──────
def _stat_brackets(ax, y_top: float, step: float, pairs, xpos) -> None:
    for rank, (i, j, sig) in enumerate(pairs):
        y = y_top + step * (rank + 1)
        tick = step * 0.25
        xi, xj = xpos[i], xpos[j]
        ax.plot([xi, xj], [y, y], color="#000000", lw=0.9, zorder=5, clip_on=False)
        ax.plot([xi, xi], [y - tick, y], color="#000000", lw=0.9, zorder=5, clip_on=False)
        ax.plot([xj, xj], [y - tick, y], color="#000000", lw=0.9, zorder=5, clip_on=False)
        ax.text((xi + xj) / 2, y + step * 0.1, sig, ha="center", va="bottom",
                fontsize=FONT_SIZE_ANNOTATION, color="#000000", clip_on=False)


def _fmt_k(n: int) -> str:
    """Compact count: 110491 -> '110.5k', 439 -> '439'."""
    return f"{n / 1000:.1f}k" if n >= 1000 else str(n)


def _black_spines(ax, lw: float = 1.0) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("#000000")
        spine.set_linewidth(lw)


def _bar_panel(ax, data, ylabel, counts, *, decimals=1, show_xlabels=True) -> None:
    xpos = [0.0, 0.82, 1.64]
    q3_tops: list[float] = []
    for pos, arr, ct in zip(xpos, data, CELL_TYPES):
        if arr.size < 2:
            q3_tops.append(0.0)
            continue
        med = float(np.median(arr))
        q1, q3 = float(np.percentile(arr, 25)), float(np.percentile(arr, 75))
        ax.bar(pos, med, width=0.52, color=_TYPE_COLOR[ct], alpha=0.85,
               zorder=3, linewidth=1.0, edgecolor="#000000")
        ax.errorbar(pos, med, yerr=[[med - q1], [q3 - med]], color="#000000",
                    linewidth=1.1, capsize=3.5, capthick=1.1, zorder=4, fmt="none")
        q3_tops.append(q3)
        ax.text(pos, med * 0.5, f"{med:.{decimals}f}", ha="center", va="center",
                fontsize=FONT_SIZE_ANNOTATION, color="#000000", zorder=6)

    valid = [(i, arr) for i, arr in enumerate(data) if arr.size >= 2]
    if len(valid) >= 2 and _scipy_stats.kruskal(*[a for _, a in valid]).pvalue < 0.05:
        idx = [(valid[a][0], valid[b][0]) for a in range(len(valid)) for b in range(a + 1, len(valid))]
        sig = []
        for i, j in idx:
            p = min(_scipy_stats.mannwhitneyu(data[i], data[j], alternative="two-sided").pvalue * len(idx), 1.0)
            mark = "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 5e-2 else None
            if mark:
                sig.append((i, j, mark))
        if sig:
            y_top = max(q3_tops)
            step = max(y_top * 0.10, 0.01)
            _stat_brackets(ax, y_top, step, sig, xpos)
            ax.set_ylim(0, y_top + step * (len(sig) + 1.6))

    ax.set_xlim(-0.45, xpos[-1] + 0.45)
    ax.set_xticks(xpos)
    if show_xlabels:
        ax.set_xticklabels([f"{ct.capitalize()}\n{_fmt_k(counts[ct])}" for ct in CELL_TYPES],
                           fontsize=FONT_SIZE_ANNOTATION, ha="center", va="top")
    else:
        ax.set_xticklabels([])
    ax.set_ylabel(ylabel, fontsize=FONT_SIZE_LABEL)
    ax.tick_params(labelsize=FONT_SIZE_TICK, width=0.8, length=3)
    ax.yaxis.grid(True, color="#E5E5E5", linewidth=0.6)
    ax.set_axisbelow(True)
    _black_spines(ax, lw=1.0)


# ── Heatmap ─────────────────────────────────────────────────────────────────
def _heatmap_panel(ax, matrix, row_labels, col_labels):
    vmax = max(float(np.abs(matrix).max()), 0.5)
    im = ax.imshow(matrix, aspect="auto", cmap=_HEAT_CMAP, vmin=-vmax, vmax=vmax,
                   interpolation="nearest")
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=FONT_SIZE_ANNOTATION, ha="center", va="top")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=FONT_SIZE_TICK)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            color = "#000000" if abs(val) < vmax * 0.6 else "#FFFFFF"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=FONT_SIZE_CELL_TEXT, color=color)
    ax.tick_params(length=0)
    _black_spines(ax, lw=0.9)
    # group separators
    for sep in (0.5, 1.5, 3.5, 4.5, 6.5, 7.5):
        ax.axvline(sep, color="#000000", linewidth=0.7, zorder=5)
    for sep in (2.5, 5.5):
        ax.axvline(sep, color="#000000", linewidth=2.6, zorder=6)
    return im


def _build_panelB_raster(data: dict, *, width_px: int, height_px: int) -> np.ndarray:
    """Panel B at its final pixel size: bar charts (left) + marker heatmap (right)."""
    cell_type = data["cell_type"]
    cell_state = data["cell_state"]
    counts = {ct: int((cell_type == ct).sum()) for ct in CELL_TYPES}
    area_data = _type_arrays(data, "area_um2")
    circ_data = _type_arrays(data, "circularity")

    zmarkers = _zscore_markers(data)
    columns = [(ct, st) for ct in CELL_TYPES for st in _STATE_ORDER]
    mat = np.zeros((len(_MARKERS), len(columns)))
    col_counts: dict[tuple[str, str], int] = {}
    for j, (ct, st) in enumerate(columns):
        mask = (cell_type == ct) & (cell_state == st)
        col_counts[(ct, st)] = int(mask.sum())
        if mask.any():
            for i, m in enumerate(_MARKERS):
                vals = zmarkers[m][mask]
                mat[i, j] = float(np.nanmedian(vals)) if np.isfinite(vals).any() else 0.0
    col_labels = [f"{_STATE_SHORT[st]}\n{_fmt_k(col_counts[(ct, st)])}" for ct, st in columns]

    fig = plt.figure(figsize=(width_px / _RENDER_DPI, height_px / _RENDER_DPI))
    ax_area = fig.add_axes(_AREA_BOX)
    ax_circ = fig.add_axes(_CIRC_BOX)
    ax_heat = fig.add_axes(_HEAT_BOX)
    ax_cbar = fig.add_axes(_CBAR_BOX)

    _bar_panel(ax_area, area_data, "Area (µm²)", counts, decimals=1, show_xlabels=False)
    _bar_panel(ax_circ, circ_data, "Circularity", counts, decimals=2, show_xlabels=True)

    im = _heatmap_panel(ax_heat, mat, _MARKERS, col_labels)
    for k, ct in enumerate(CELL_TYPES):
        ax_heat.text((k * 3 + 1.5) / len(columns), 1.012, ct.capitalize(),
                     transform=ax_heat.transAxes, ha="center", va="bottom",
                     fontsize=FONT_SIZE_LABEL, fontweight="bold", color=_TYPE_COLOR[ct],
                     clip_on=False)
    cbar = fig.colorbar(im, cax=ax_cbar)
    cbar.set_label("Median Z-score", fontsize=FONT_SIZE_TICK)
    cbar.ax.tick_params(labelsize=FONT_SIZE_TICK, length=3, width=0.7)
    cbar.outline.set_edgecolor("#000000")
    cbar.outline.set_linewidth(0.9)

    buf = _fig_to_array(fig)
    return _resize_to_width(buf, width_px)


def _fig_to_array(fig: plt.Figure) -> np.ndarray:
    import io
    from PIL import Image

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=_RENDER_DPI, facecolor="white")
    buf.seek(0)
    arr = np.array(Image.open(buf).convert("RGB"))
    plt.close(fig)
    return arr


# ── Composite ───────────────────────────────────────────────────────────────
def build_stage1_figure(data: dict, *, svg_path: Path, results_png: Path) -> plt.Figure:
    from PIL import Image

    full_width = round(COMPOSITE_WIDTH_IN * _RENDER_DPI)

    # Row 1: A (schematic) + B (cell characterization), sharing height h1.
    h1 = round((full_width - _ROW1_GAP_PX) / (_PANELB_TARGET_ASPECT + _SCHEMATIC_ASPECT))
    w_a = round(_SCHEMATIC_ASPECT * h1)
    w_b = full_width - w_a - _ROW1_GAP_PX

    img_a = _svg_to_rgb_on_white(svg_path, height_px=h1)
    img_a = np.array(Image.fromarray(img_a).resize((w_a, h1), Image.LANCZOS))
    img_b = _build_panelB_raster(data, width_px=w_b, height_px=h1)
    img_b = img_b[:h1] if img_b.shape[0] >= h1 else _pad_row_height(img_b, h1)
    gutter = np.full((h1, _ROW1_GAP_PX, 3), 255, dtype=np.uint8)
    row1 = np.concatenate([img_a, gutter, img_b], axis=1)

    # Row 2: C (montage) resized to full width.
    row2 = _resize_to_width(_load_rgb(results_png), full_width)

    rows = [_add_label_strip(r) for r in (row1, row2)]
    heights = [r.shape[0] for r in rows]

    fig = plt.figure(figsize=(full_width / _RENDER_DPI, sum(heights) / _RENDER_DPI))
    gs = fig.add_gridspec(len(rows), 1, height_ratios=heights, hspace=0)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    b_x0 = (w_a + _ROW1_GAP_PX) / full_width          # B raster left, as composite fraction
    bx = lambda frac: b_x0 + frac * w_b / full_width  # noqa: E731 - map B-internal frac -> composite
    row_letters = [
        [("A", 0.006), ("B(i)", bx(_BAR_LEFT - 0.02)), ("B(ii)", bx(_HEAT_LABELS_LEFT - 0.012))],
        [("C", 0.006)],
    ]
    for i, (img, letters) in enumerate(zip(rows, row_letters)):
        ax = fig.add_subplot(gs[i, 0])
        ax.imshow(img, interpolation="none")
        ax.axis("off")
        for letter, x in letters:
            ax.text(x, 0.992, letter, transform=ax.transAxes, fontsize=_PANEL_LETTER_FS,
                    fontweight="bold", va="top", ha="left", color="black")
    return fig


def main() -> None:
    apply_style()
    data = load_cell_table()
    fig = build_stage1_figure(data, svg_path=SVG_PATH, results_png=RESULTS_PNG)
    save_figure_png(fig, OUT_PNG)
    print("wrote", OUT_PNG)


if __name__ == "__main__":
    main()
