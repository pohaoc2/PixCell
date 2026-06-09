"""Figure 5: UNI/TME foundation-model decomposition."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from src.a2_decomposition.metrics import (
    DECOMPOSITION_METRICS,
    DEFAULT_GENERATED_ROOT,
    DEFAULT_METRICS_ROOT,
    DEFAULT_REPRESENTATIVE_JSON,
    DEFAULT_SUMMARY_CSV,
    MODE_KEYS,
    MODE_LABELS,
    complete_generated_tile_ids,
    effect_decomposition,
    load_summary_csv,
    select_representative_tile,
)
from tools.ablation_report.shared import INK, METRIC_LABELS, OKABE_GRAY, SOFT_GRID, plt
from tools.stage3.hed_utils import tissue_mask_from_rgb

from src.paper_figures.style import (
    FONT_SIZE_CELL_TEXT,
    FONT_SIZE_DENSE_LABEL,
    FONT_SIZE_DENSE_TITLE,
    FONT_SIZE_LABEL,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ORION_ROOT = ROOT / "data" / "orion-crc33"
DEFAULT_OUT_PNG = ROOT / "figures" / "pngs" / "08_uni_tme_decomposition.png"
MODE_USE_UNI = {"uni_plus_tme": True, "uni_only": True, "tme_only": False, "neither": False}
MODE_USE_TME = {"uni_plus_tme": True, "uni_only": False, "tme_only": True, "neither": False}
DISPLAY_METRICS = ("fud", "lpips", "pq", "dice", "style_hed")


def _load_rgb(path: Path, *, size: tuple[int, int] | None = None) -> Image.Image:
    image = Image.open(path).convert("RGB")
    if size is not None and image.size != size:
        image = image.resize(size, Image.BILINEAR)
    return image


def _blank_image(size: tuple[int, int] = (256, 256), value: int = 245) -> Image.Image:
    return Image.fromarray(np.full((size[1], size[0], 3), value, dtype=np.uint8), mode="RGB")


def _image_is_visually_blank(image: Image.Image) -> bool:
    arr = np.asarray(image, dtype=np.float32)
    if arr.size == 0:
        return True
    return float(arr.mean()) >= 245.0 and float(arr.std()) <= 2.0


def _resolve_representative_tile(
    *,
    generated_root: Path,
    metrics_root: Path,
    representative_json: Path,
) -> str:
    if representative_json.is_file():
        payload = json.loads(representative_json.read_text(encoding="utf-8"))
        tile_id = str(payload.get("tile_id", "")).strip()
        if tile_id and (generated_root / tile_id).is_dir():
            return tile_id

    tile_id, _ = select_representative_tile(metrics_root=metrics_root)
    if tile_id and (generated_root / tile_id).is_dir():
        return tile_id

    tile_ids = complete_generated_tile_ids(generated_root)
    if not tile_ids:
        raise FileNotFoundError(f"no complete decomposition tiles under {generated_root}")
    return tile_ids[0]


def _resolve_panel_a_tile(*, generated_root: Path, preferred_tile_id: str) -> str:
    preferred_dir = generated_root / preferred_tile_id
    if preferred_dir.is_dir():
        preferred_neither = _load_rgb(preferred_dir / "neither.png")
        if not _image_is_visually_blank(preferred_neither):
            return preferred_tile_id

    best_tile_id = preferred_tile_id
    best_score: tuple[float, float, float] | None = None
    for tile_id in complete_generated_tile_ids(generated_root):
        tile_dir = generated_root / tile_id
        neither_path = tile_dir / "neither.png"
        full_path = tile_dir / "uni_plus_tme.png"
        if not neither_path.is_file() or not full_path.is_file():
            continue
        neither = np.asarray(_load_rgb(neither_path), dtype=np.uint8)
        full = np.asarray(_load_rgb(full_path), dtype=np.uint8)
        neither_tissue = float(tissue_mask_from_rgb(neither).mean())
        neither_std = float(neither.std())
        full_tissue = float(tissue_mask_from_rgb(full).mean())
        score = (neither_tissue, neither_std, full_tissue)
        if best_score is None or score > best_score:
            best_score = score
            best_tile_id = tile_id
    return best_tile_id


def _load_tme_thumbnail(orion_root: Path, tile_id: str, *, size: tuple[int, int]) -> Image.Image:
    for folder in ("cell_masks", "cell_mask"):
        channel_dir = orion_root / "exp_channels" / folder
        if not channel_dir.is_dir():
            continue
        for ext in (".npy", ".png", ".jpg", ".jpeg", ".tif", ".tiff"):
            path = channel_dir / f"{tile_id}{ext}"
            if not path.is_file():
                continue
            if path.suffix.lower() == ".npy":
                arr = np.load(path)
            else:
                arr = np.asarray(Image.open(path))
            while arr.ndim > 2 and arr.shape[0] == 1:
                arr = arr.squeeze(0)
            if arr.ndim == 3:
                arr = arr[..., 0]
            arr = np.asarray(arr, dtype=np.float32)
            lo = float(np.nanmin(arr)) if arr.size else 0.0
            hi = float(np.nanmax(arr)) if arr.size else 1.0
            if hi > lo:
                arr = (arr - lo) / (hi - lo)
            arr = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
            rgb = np.stack([arr, arr, arr], axis=-1)
            return Image.fromarray(rgb, mode="RGB").resize(size, Image.NEAREST)
    return _blank_image(size=size, value=230)


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.08,
        1.05,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=FONT_SIZE_LABEL,
        fontweight="bold",
        color=INK,
    )


def _render_image_cell(
    ax: plt.Axes,
    image: Image.Image,
    title: str,
    *,
    border_color: str = "#333333",
    border_linestyle: str = "-",
) -> None:
    ax.imshow(image)
    if title:
        ax.set_title(title, fontsize=FONT_SIZE_DENSE_TITLE, pad=2)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color(border_color)
        spine.set_linestyle(border_linestyle)


def _draw_panel_a_headers(
    fig: plt.Figure, *, fig_w: float, fig_h: float, x0: float, y_top: float, cell: float, gap: float
) -> None:
    """Column-group headers above A's transposed 2x3 grid, in absolute-inch
    positions: 'Inputs' over the left column, 'Generated outputs' centred over
    the two generated columns."""
    y = (y_top + 0.03) / fig_h
    fig.text(
        (x0 + cell / 2) / fig_w, y, "Inputs",
        ha="center", va="bottom", fontsize=FONT_SIZE_DENSE_TITLE, fontweight="bold", color=INK,
    )
    fig.text(
        (x0 + 2 * cell + 1.5 * gap) / fig_w, y, "Generated outputs",
        ha="center", va="bottom", fontsize=FONT_SIZE_DENSE_TITLE, fontweight="bold", color=INK,
    )


def _fig_letter(fig: plt.Figure, *, fig_w: float, fig_h: float, x_in: float, y_in: float, letter: str) -> None:
    """Bold panel letter at an absolute-inch position (top-left anchored)."""
    fig.text(
        x_in / fig_w, y_in / fig_h, letter,
        ha="left", va="top", fontsize=FONT_SIZE_LABEL, fontweight="bold", color=INK,
    )


def _render_cell_caption(ax: plt.Axes, text: str) -> None:
    """Small in-cell identifier (white bbox) for the input tiles."""
    ax.text(
        0.04,
        0.96,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=FONT_SIZE_DENSE_LABEL,
        color=INK,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.0},
        zorder=5,
    )


def _render_mode_indicator(ax: plt.Axes, mode_key: str, *, show_labels: bool) -> None:
    """Draw UNI/TME ●/○ dots stacked at bottom-right corner of an image axes."""
    rows = [
        (0.84, 0.20, "UNI", MODE_USE_UNI[mode_key]),
        (0.84, 0.08, "TME", MODE_USE_TME[mode_key]),
    ]
    for x_dot, y_dot, label, active in rows:
        ax.scatter(
            [x_dot], [y_dot],
            s=16,
            facecolors=INK if active else "white",
            edgecolors=INK,
            linewidths=0.8,
            transform=ax.transAxes,
            clip_on=False,
            zorder=5,
        )
        if show_labels:
            ax.text(x_dot - 0.09, y_dot, label, transform=ax.transAxes,
                    ha="right", va="center", fontsize=FONT_SIZE_DENSE_LABEL, color=INK)


def _render_panel_a(
    fig: plt.Figure,
    *,
    fig_w: float,
    fig_h: float,
    x0: float,
    y_top: float,
    cell: float,
    gap: float,
    generated_root: Path,
    orion_root: Path,
    tile_id: str,
) -> None:
    """Transposed 2 rows x 3 cols image grid placed in absolute inches so the
    cell gap is equal horizontally and vertically (`wspace == hspace`) at any
    figure size. Col 0 = inputs (Real H&E over cell masks); cols 1-2 = generated
    2x2 (rows = UNI on/off, cols = TME on/off)."""
    tile_id = _resolve_panel_a_tile(generated_root=generated_root, preferred_tile_id=tile_id)
    sample = _load_rgb(generated_root / tile_id / "uni_plus_tme.png")
    size = sample.size

    def _cell_ax(col: int, row: int) -> plt.Axes:
        cx = x0 + col * (cell + gap)
        cy = y_top - cell - row * (cell + gap)
        return fig.add_axes([cx / fig_w, cy / fig_h, cell / fig_w, cell / fig_h])

    ref_ax = _cell_ax(0, 0)
    ref_path = orion_root / "he" / f"{tile_id}.png"
    ref_img = _load_rgb(ref_path, size=size) if ref_path.is_file() else _blank_image(size=size)
    _render_image_cell(ref_ax, ref_img, "", border_color="#999999", border_linestyle="--")
    _render_cell_caption(ref_ax, "Real H&E")

    tme_ax = _cell_ax(0, 1)
    _render_image_cell(
        tme_ax, _load_tme_thumbnail(orion_root, tile_id, size=size),
        "", border_color="#999999", border_linestyle="--",
    )
    _render_cell_caption(tme_ax, "cell masks")

    generated_layout = [
        ("uni_plus_tme", (1, 0)),  # (col, row)
        ("uni_only", (2, 0)),
        ("tme_only", (1, 1)),
        ("neither", (2, 1)),
    ]
    for mode_key, (col, row) in generated_layout:
        ax = _cell_ax(col, row)
        image = _load_rgb(generated_root / tile_id / f"{mode_key}.png", size=size)
        _render_image_cell(ax, image, "")
        _render_mode_indicator(ax, mode_key, show_labels=(mode_key == "uni_plus_tme"))

    _draw_panel_a_headers(fig, fig_w=fig_w, fig_h=fig_h, x0=x0, y_top=y_top, cell=cell, gap=gap)


def _values_for_metric(summary: dict[str, dict], metric_key: str) -> tuple[list[float], list[float | None]]:
    values: list[float] = []
    errors: list[float | None] = []
    for mode_key in MODE_KEYS:
        row = summary.get(mode_key, {}).get(metric_key)
        if row is None or row.mean is None:
            values.append(float("nan"))
            errors.append(None)
            continue
        values.append(float(row.mean))
        errors.append(float(row.sd) if row.sd is not None else None)
    return values, errors


def _tight_ylim(values: list[float], errors: list[float | None]) -> tuple[float, float]:
    finite: list[float] = []
    for value, err in zip(values, errors, strict=True):
        if not np.isfinite(value):
            continue
        if err is None:
            finite.append(float(value))
        else:
            finite.extend([float(value - err), float(value + err)])
    if not finite:
        return 0.0, 1.0
    lo = min(finite)
    hi = max(finite)
    if lo == hi:
        pad = max(abs(lo) * 0.1, 0.05)
    else:
        pad = (hi - lo) * 0.12
    return lo - pad, hi + pad


def _render_dot_key_single(key_ax: plt.Axes, *, show_labels: bool) -> None:
    """One column of the dot-key strip. show_labels=True only for the leftmost column."""
    key_ax.set_xlim(-1.25, len(MODE_KEYS) - 0.5)
    key_ax.set_ylim(-0.5, 1.5)
    for x, mode_key in enumerate(MODE_KEYS):
        key_ax.scatter(x, 1, s=20, facecolors=INK if MODE_USE_UNI[mode_key] else "white", edgecolors=INK, linewidths=0.8)
        key_ax.scatter(x, 0, s=20, facecolors=INK if MODE_USE_TME[mode_key] else "white", edgecolors=INK, linewidths=0.8)
    if show_labels:
        key_ax.text(-0.95, 1, "UNI", ha="right", va="center", fontsize=FONT_SIZE_DENSE_LABEL, color=INK)
        key_ax.text(-0.95, 0, "TME", ha="right", va="center", fontsize=FONT_SIZE_DENSE_LABEL, color=INK)
    key_ax.axis("off")


def _render_panel_b(fig: plt.Figure, subgrid, summary: dict[str, dict], *, label_subgrid=None, draw_label: bool = True) -> None:
    if draw_label:
        label_ax = fig.add_subplot(label_subgrid if label_subgrid is not None else subgrid)
        label_ax.axis("off")
        _panel_label(label_ax, "B")

    _SHARED_METRICS = {"lpips", "pq", "dice", "style_hed"}
    _SHARED_LIST = [m for m in DISPLAY_METRICS if m in _SHARED_METRICS]

    outer_grid = subgrid.subgridspec(2, 1, height_ratios=[4.8, 1.0], hspace=0.05)

    # FUD gets its own cell; LPIPS/PQ/DICE/HED share a tighter sub-grid
    m_outer = outer_grid[0, 0].subgridspec(1, 2, width_ratios=[1, len(_SHARED_LIST)], wspace=0.25)
    fud_m = m_outer[0, 0]
    shared_m = m_outer[0, 1].subgridspec(1, len(_SHARED_LIST), wspace=0.20)

    k_outer = outer_grid[1, 0].subgridspec(1, 2, width_ratios=[1, len(_SHARED_LIST)], wspace=0.25)
    fud_k = k_outer[0, 0]
    shared_k = k_outer[0, 1].subgridspec(1, len(_SHARED_LIST), wspace=0.20)

    x = np.arange(len(MODE_KEYS), dtype=float)
    for idx, metric_key in enumerate(DISPLAY_METRICS):
        if metric_key == "fud":
            ax = fig.add_subplot(fud_m)
            key_ax = fig.add_subplot(fud_k)
        else:
            s_idx = _SHARED_LIST.index(metric_key)
            ax = fig.add_subplot(shared_m[0, s_idx])
            key_ax = fig.add_subplot(shared_k[0, s_idx])

        values, errors = _values_for_metric(summary, metric_key)
        valid_x = [xv for xv, v in zip(x, values, strict=True) if np.isfinite(v)]
        valid_y = [v for v in values if np.isfinite(v)]
        valid_err = [e if e is not None else 0.0 for v, e in zip(values, errors, strict=True) if np.isfinite(v)]
        if valid_y:
            ax.errorbar(
                valid_x,
                valid_y,
                yerr=valid_err,
                color=INK,
                linestyle="none",
                marker="o",
                markerfacecolor="white",
                markeredgecolor=INK,
                markersize=4.5,
                capsize=2.0,
                elinewidth=0.9,
                markeredgewidth=0.9,
            )
        label = METRIC_LABELS.get(metric_key, metric_key)
        row = summary.get("uni_plus_tme", {}).get(metric_key)
        raw_dir = row.direction if row is not None else ""
        arrow = {"up": "↑", "down": "↓"}.get(raw_dir.lower(), raw_dir)
        ax.set_title(f"{label}\n({arrow})", fontsize=FONT_SIZE_DENSE_TITLE, pad=1, linespacing=0.9)
        ax.set_xlim(-0.5, len(MODE_KEYS) - 0.5)
        ax.set_xticks([])
        ax.grid(True, axis="y", color=SOFT_GRID, linewidth=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if metric_key in _SHARED_METRICS:
            ax.set_ylim(0.0, 1.0)
            if metric_key == "lpips":
                ax.tick_params(axis="y", labelsize=FONT_SIZE_DENSE_LABEL, colors=INK)
            else:
                ax.tick_params(axis="y", left=True, labelleft=False)
        else:
            ax.set_ylim(*_tight_ylim(values, errors))
            ax.tick_params(axis="y", labelsize=FONT_SIZE_DENSE_LABEL, colors=INK)

        _render_dot_key_single(key_ax, show_labels=(idx == 0))


_EFFECT_SHORT = {
    "UNI effect": "UNI eff.",
    "TME effect": "TME eff.",
    "Interaction": "Interact.",
}


def _fmt_val(v: float) -> str:
    a = abs(v)
    if a >= 10:
        return f"{v:.0f}"
    elif a >= 1:
        return f"{v:.1f}"
    return f"{v:.2f}"


def _render_panel_c(fig: plt.Figure, ax: plt.Axes, cax: plt.Axes, summary: dict[str, dict]) -> None:
    # Cells are square by construction: the host axes rect is sized 5*CELL x
    # 3*CELL (5 metric columns, 3 effect rows), so aspect="auto" fills it with
    # square cells.
    # Raw (non-oriented) differences so annotations match intuition per metric
    effect_names = ["UNI effect", "TME effect", "Interaction"]
    effects: dict[str, dict[str, float | None]] = {n: {} for n in effect_names}
    effect_sds: dict[str, dict[str, float | None]] = {n: {} for n in effect_names}
    for metric_key in DISPLAY_METRICS:
        raw: dict[str, float] = {}
        sds: dict[str, float] = {}
        for mode_key in MODE_KEYS:
            record = summary.get(mode_key, {}).get(metric_key)
            if record is not None and record.mean is not None:
                raw[mode_key] = float(record.mean)
            if record is not None and record.sd is not None:
                sds[mode_key] = float(record.sd)
        if set(MODE_KEYS).issubset(raw):
            effects["UNI effect"][metric_key] = raw["uni_only"] - raw["neither"]
            effects["TME effect"][metric_key] = raw["tme_only"] - raw["neither"]
            effects["Interaction"][metric_key] = (
                raw["uni_plus_tme"] - raw["uni_only"] - raw["tme_only"] + raw["neither"]
            )
        else:
            for n in effect_names:
                effects[n][metric_key] = None
        if set(MODE_KEYS).issubset(sds):
            effect_sds["UNI effect"][metric_key] = float(np.sqrt(sds["uni_only"] ** 2 + sds["neither"] ** 2))
            effect_sds["TME effect"][metric_key] = float(np.sqrt(sds["tme_only"] ** 2 + sds["neither"] ** 2))
            effect_sds["Interaction"][metric_key] = float(np.sqrt(
                sds["uni_plus_tme"] ** 2 + sds["uni_only"] ** 2 + sds["tme_only"] ** 2 + sds["neither"] ** 2
            ))
        else:
            for n in effect_names:
                effect_sds[n][metric_key] = None
    rows = list(effects)
    cols = list(DISPLAY_METRICS)
    matrix = np.full((len(rows), len(cols)), np.nan, dtype=float)
    for row_idx, row_name in enumerate(rows):
        for col_idx, metric_key in enumerate(cols):
            value = effects[row_name].get(metric_key)
            if value is not None:
                matrix[row_idx, col_idx] = float(value)

    finite = matrix[np.isfinite(matrix)]
    if finite.size == 0:
        ax.axis("off")
        cax.axis("off")
        ax.text(0.5, 0.5, "Effect metrics missing", ha="center", va="center", transform=ax.transAxes)
        return

    # Oriented color matrix: flip sign for lower-is-better metrics so blue=good, red=bad
    color_matrix = np.full_like(matrix, np.nan)
    for col_idx, metric_key in enumerate(cols):
        direction = "up"
        for mode_key in MODE_KEYS:
            rec = summary.get(mode_key, {}).get(metric_key)
            if rec is not None:
                direction = rec.direction
                break
        sign = 1.0 if direction == "up" else -1.0
        color_matrix[:, col_idx] = matrix[:, col_idx] * sign

    # Per-column normalise the oriented color matrix so each metric's extreme maps to ±1
    norm_matrix = np.full_like(color_matrix, np.nan)
    for col_idx in range(len(cols)):
        col_vals = color_matrix[:, col_idx]
        col_finite = col_vals[np.isfinite(col_vals)]
        col_max = float(np.max(np.abs(col_finite))) if col_finite.size else 1.0
        norm_matrix[:, col_idx] = col_vals / (col_max or 1.0)

    im = ax.imshow(norm_matrix, cmap="RdBu", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([METRIC_LABELS.get(m, m) for m in cols], rotation=35, ha="right", fontsize=FONT_SIZE_DENSE_LABEL)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([_EFFECT_SHORT.get(r, r) for r in rows], fontsize=FONT_SIZE_DENSE_LABEL)
    ax.set_xticks(np.arange(-0.5, len(cols), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
    ax.grid(which="minor", color="#F2F0EA", linestyle="-", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    for row_idx in range(len(rows)):
        for col_idx in range(len(cols)):
            value = matrix[row_idx, col_idx]
            if np.isfinite(value):
                text_color = "white" if abs(norm_matrix[row_idx, col_idx]) > 0.6 else INK
                sd = effect_sds[rows[row_idx]].get(cols[col_idx])
                label = _fmt_val(value) if sd is None else f"{_fmt_val(value)}\n±{_fmt_val(sd)}"
                ax.text(col_idx, row_idx, label, ha="center", va="center", fontsize=FONT_SIZE_CELL_TEXT, color=text_color)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_ticks([-1, 0, 1])
    cbar.set_ticklabels(["most\nneg.", "0", "most\npos."])
    cbar.ax.tick_params(labelsize=FONT_SIZE_DENSE_LABEL)



def build_uni_tme_decomposition_figure(
    *,
    generated_root: Path = DEFAULT_GENERATED_ROOT,
    metrics_root: Path = DEFAULT_METRICS_ROOT,
    summary_csv: Path = DEFAULT_SUMMARY_CSV,
    representative_json: Path = DEFAULT_REPRESENTATIVE_JSON,
    orion_root: Path = DEFAULT_ORION_ROOT,
) -> plt.Figure:
    """Build the full Figure 5 panel."""
    generated_root = Path(generated_root)
    metrics_root = Path(metrics_root)
    summary_csv = Path(summary_csv)
    if not summary_csv.is_file():
        raise FileNotFoundError(f"missing decomposition summary: {summary_csv}")

    summary = load_summary_csv(summary_csv)
    tile_id = _resolve_representative_tile(
        generated_root=generated_root,
        metrics_root=metrics_root,
        representative_json=Path(representative_json),
    )

    # ---- Absolute-inch layout (vis_guidance: equal cell gaps + data-region
    # alignment). A's image grid and B's plot band share the SAME x-extent
    # [PLOT_L, PLOT_L + W_AB], so width(A) == width(B) by construction. C is a
    # square-celled heatmap placed to the right of the A/B stack. ----
    M_L, M_R, M_T, M_B = 0.05, 0.06, 0.09, 0.08
    YLABEL_IN = 0.42          # room left of the plot band for B's y-tick labels
    HEADER_IN = 0.18          # room above A for the column-group headers + letters
    GAP_A = 0.05              # equal cell gap in A (rows AND columns)
    CELL_A = 0.92             # square image cell
    GAP_AB = 0.44             # vertical gap between A and B (B's metric titles live here)
    B_H = 1.20               # reduced B height
    GAP_LR = 0.40             # gap between the A/B block and C's y labels
    C_YLAB, C_XLAB = 0.58, 0.46
    CBAR_GAP, CBAR_W, CBAR_LAB = 0.07, 0.11, 0.66

    fig_w = 7.2               # held constant so the composite scales A-F letters consistently
    PLOT_L = M_L + YLABEL_IN
    W_AB = 3 * CELL_A + 2 * GAP_A
    A_H = 2 * CELL_A + GAP_A
    C_W = (fig_w - PLOT_L - W_AB - GAP_LR - M_R) - (C_YLAB + CBAR_GAP + CBAR_W + CBAR_LAB)
    CELL_C = C_W / 5.0        # 5 metric columns; square cells -> 3*CELL_C tall
    C_H = 3 * CELL_C
    fig_h = M_T + HEADER_IN + A_H + GAP_AB + B_H + M_B

    fig = plt.figure(figsize=(fig_w, fig_h))

    a_top = fig_h - M_T - HEADER_IN
    a_bot = a_top - A_H
    b_top = a_bot - GAP_AB
    b_bot = b_top - B_H

    _render_panel_a(
        fig, fig_w=fig_w, fig_h=fig_h, x0=PLOT_L, y_top=a_top, cell=CELL_A, gap=GAP_A,
        generated_root=generated_root, orion_root=Path(orion_root), tile_id=tile_id,
    )

    # B: gridspec constrained to the same horizontal band as A.
    b_gs = fig.add_gridspec(
        1, 1, left=PLOT_L / fig_w, right=(PLOT_L + W_AB) / fig_w,
        bottom=b_bot / fig_h, top=b_top / fig_h,
    )
    _render_panel_b(fig, b_gs[0, 0], summary, draw_label=False)

    # C: square-celled heatmap + colorbar, vertically centred against the A/B
    # stack (a 3x5 matrix is too short to span the stack without oversized cells).
    c_left = PLOT_L + W_AB + GAP_LR + C_YLAB
    c_top = (a_top + b_bot) / 2 + C_H / 2
    ax_c = fig.add_axes([c_left / fig_w, (c_top - C_H) / fig_h, C_W / fig_w, C_H / fig_h])
    cax_c = fig.add_axes(
        [(c_left + C_W + CBAR_GAP) / fig_w, (c_top - C_H) / fig_h, CBAR_W / fig_w, C_H / fig_h]
    )
    _render_panel_c(fig, ax_c, cax_c, summary)

    # Panel letters (A/B share the left edge; C sits above its centred heatmap).
    _fig_letter(fig, fig_w=fig_w, fig_h=fig_h, x_in=M_L, y_in=a_top + HEADER_IN, letter="A")
    _fig_letter(fig, fig_w=fig_w, fig_h=fig_h, x_in=M_L, y_in=b_top + 0.16, letter="B")
    _fig_letter(fig, fig_w=fig_w, fig_h=fig_h, x_in=c_left - C_YLAB, y_in=c_top + 0.16, letter="C")
    return fig


def save_uni_tme_decomposition_figure(
    *,
    out_png: Path = DEFAULT_OUT_PNG,
    generated_root: Path = DEFAULT_GENERATED_ROOT,
    metrics_root: Path = DEFAULT_METRICS_ROOT,
    summary_csv: Path = DEFAULT_SUMMARY_CSV,
    representative_json: Path = DEFAULT_REPRESENTATIVE_JSON,
    orion_root: Path = DEFAULT_ORION_ROOT,
    dpi: int = 300,
) -> Path:
    fig = build_uni_tme_decomposition_figure(
        generated_root=generated_root,
        metrics_root=metrics_root,
        summary_csv=summary_csv,
        representative_json=representative_json,
        orion_root=orion_root,
    )
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_png
