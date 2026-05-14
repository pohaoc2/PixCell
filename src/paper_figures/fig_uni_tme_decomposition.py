"""Figure 5: UNI/TME foundation-model decomposition."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
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
_GENERATED_GRID: list[tuple[str, bool]] = [
    ("uni_plus_tme", True),
    ("uni_only", False),
    ("tme_only", False),
    ("neither", False),
]
_GENERATED_POSITIONS: list[tuple[int, int]] = [(1, 0), (1, 1), (2, 0), (2, 1)]
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


def _render_panel_a_row_labels(ax: plt.Axes) -> None:
    ax.text(
        -0.055,
        0.835,
        "Inputs",
        transform=ax.transAxes,
        rotation=90,
        ha="center",
        va="center",
        fontsize=FONT_SIZE_DENSE_TITLE,
        fontweight="bold",
        color=INK,
        clip_on=False,
    )
    ax.text(
        -0.055,
        0.335,
        "Generated outputs",
        transform=ax.transAxes,
        rotation=90,
        ha="center",
        va="center",
        fontsize=FONT_SIZE_DENSE_TITLE,
        fontweight="bold",
        color=INK,
        clip_on=False,
    )


def _render_mode_indicator(ax: plt.Axes, mode_key: str, *, show_labels: bool) -> None:
    """Draw UNI/TME ●/○ dots stacked at bottom-right corner of an image axes."""
    rows = [
        (0.88, 0.15, "UNI", MODE_USE_UNI[mode_key]),
        (0.88, 0.05, "TME", MODE_USE_TME[mode_key]),
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
            ax.text(x_dot - 0.06, y_dot, label, transform=ax.transAxes,
                    ha="right", va="center", fontsize=FONT_SIZE_DENSE_LABEL, color=INK)


def _render_panel_a(
    fig: plt.Figure,
    subgrid,
    *,
    generated_root: Path,
    orion_root: Path,
    tile_id: str,
) -> None:
    outer_ax = fig.add_subplot(subgrid)
    outer_ax.axis("off")
    _panel_label(outer_ax, "A")
    _render_panel_a_row_labels(outer_ax)

    tile_id = _resolve_panel_a_tile(generated_root=generated_root, preferred_tile_id=tile_id)

    grid = subgrid.subgridspec(3, 2, wspace=0.03, hspace=0.03)
    sample = _load_rgb(generated_root / tile_id / "uni_plus_tme.png")
    size = sample.size

    ref_ax = fig.add_subplot(grid[0, 0])
    ref_path = orion_root / "he" / f"{tile_id}.png"
    ref_img = _load_rgb(ref_path, size=size) if ref_path.is_file() else _blank_image(size=size)
    _render_image_cell(ref_ax, ref_img, "Real H&E", border_color="#999999", border_linestyle="--")

    tme_ax = fig.add_subplot(grid[0, 1])
    _render_image_cell(
        tme_ax, _load_tme_thumbnail(orion_root, tile_id, size=size),
        "cell masks", border_color="#999999", border_linestyle="--",
    )

    for (mode_key, show_text), (row, col) in zip(_GENERATED_GRID, _GENERATED_POSITIONS, strict=True):
        ax = fig.add_subplot(grid[row, col])
        image = _load_rgb(generated_root / tile_id / f"{mode_key}.png", size=size)
        _render_image_cell(ax, image, "")
        _render_mode_indicator(ax, mode_key, show_labels=show_text)


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
    key_ax.set_xlim(-0.5, len(MODE_KEYS) - 0.5)
    key_ax.set_ylim(-0.5, 1.5)
    for x, mode_key in enumerate(MODE_KEYS):
        key_ax.scatter(x, 1, s=20, facecolors=INK if MODE_USE_UNI[mode_key] else "white", edgecolors=INK, linewidths=0.8)
        key_ax.scatter(x, 0, s=20, facecolors=INK if MODE_USE_TME[mode_key] else "white", edgecolors=INK, linewidths=0.8)
    if show_labels:
        key_ax.text(-0.8, 1, "UNI", ha="right", va="center", fontsize=FONT_SIZE_DENSE_LABEL, color=INK)
        key_ax.text(-0.8, 0, "TME", ha="right", va="center", fontsize=FONT_SIZE_DENSE_LABEL, color=INK)
    key_ax.axis("off")


def _render_panel_b(fig: plt.Figure, subgrid, summary: dict[str, dict], *, label_subgrid=None) -> None:
    label_ax = fig.add_subplot(label_subgrid if label_subgrid is not None else subgrid)
    label_ax.axis("off")
    _panel_label(label_ax, "B")

    _SHARED_METRICS = {"lpips", "pq", "dice", "style_hed"}
    _SHARED_LIST = [m for m in DISPLAY_METRICS if m in _SHARED_METRICS]

    outer_grid = subgrid.subgridspec(2, 1, height_ratios=[5.5, 0.85], hspace=0.05)

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
        ax.set_title(f"{label} ({arrow})", fontsize=FONT_SIZE_DENSE_TITLE, pad=2)
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


def _render_panel_c(fig: plt.Figure, subgrid, summary: dict[str, dict]) -> None:
    ax = fig.add_subplot(subgrid)
    _panel_label(ax, "C")
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
        ax.text(0.5, 0.5, "Effect metrics missing", ha="center", va="center", transform=ax.transAxes)
        return

    # Oriented color matrix: flip sign for lower-is-better metrics so red=good, blue=bad
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
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.06)
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

    fig = plt.figure(figsize=(7.2, 3.95))
    outer = fig.add_gridspec(1, 2, width_ratios=[0.78, 1.22], wspace=0.22)
    _render_panel_a(fig, outer[0, 0], generated_root=generated_root, orion_root=Path(orion_root), tile_id=tile_id)
    right = outer[0, 1].subgridspec(2, 1, height_ratios=[1.15, 1.0], hspace=0.38)
    _render_panel_b(fig, right[0, 0], summary, label_subgrid=outer[0, 1])
    _render_panel_c(fig, right[1, 0], summary)
    fig.subplots_adjust(left=0.02, right=0.96, bottom=0.08, top=0.97)
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
