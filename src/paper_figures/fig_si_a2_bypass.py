"""Build SI_A2_bypass_probe.png: metric table plus qualitative grid."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.paper_figures.style import (
    FONT_SIZE_ANNOTATION,
    FONT_SIZE_LABEL,
    FONT_SIZE_TITLE,
)


ROW_KEYS = ("production", "bypass_probe", "off_the_shelf")
ROW_LABELS = {
    "production": "Production\nzero_mask_latent=True, full TME",
    "bypass_probe": "Bypass probe\nzero_mask_latent=False, TME=0",
    "off_the_shelf": "Off-the-shelf PixCell\nmask-only, no fine-tune",
}
METRIC_COLUMNS = (
    ("FID", "fid", "{:.2f}"),
    ("UNI-cos", "uni_cos", "{:.3f}"),
    ("Cell-count r", "cellvit_count_r", "{:.3f}"),
    ("Type KL", "cellvit_type_kl", "{:.3f}"),
    ("Nuc KS", "cellvit_nuc_ks", "{:.3f}"),
)


def _load_metrics_summary(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _rows_by_key(summary: dict) -> dict[str, dict]:
    rows = summary.get("rows", [])
    by_key: dict[str, dict] = {}
    for idx, row in enumerate(rows):
        key = str(row.get("key") or row.get("variant") or "").lower().replace("-", "_").replace(" ", "_")
        if key in ROW_KEYS:
            by_key[key] = row
        elif idx < len(ROW_KEYS):
            by_key[ROW_KEYS[idx]] = row
    return by_key


def _format_metric(row: dict, key: str, fmt: str) -> str:
    value = row.get(key)
    if value is None:
        return "-"
    try:
        return fmt.format(float(value))
    except (TypeError, ValueError):
        return str(value)


def _draw_metric_table(ax: plt.Axes, summary: dict) -> None:
    ax.axis("off")
    by_key = _rows_by_key(summary)
    col_x = [0.02, 0.42, 0.54, 0.66, 0.78, 0.90]
    ax.text(col_x[0], 0.95, "Variant", fontsize=FONT_SIZE_LABEL, fontweight="bold", va="top")
    for x, (label, _, _) in zip(col_x[1:], METRIC_COLUMNS, strict=True):
        ax.text(x, 0.95, label, fontsize=FONT_SIZE_LABEL, fontweight="bold", va="top", ha="center")

    for row_idx, key in enumerate(ROW_KEYS):
        y = 0.75 - row_idx * 0.26
        row = by_key.get(key, {})
        ax.text(col_x[0], y, ROW_LABELS[key], fontsize=FONT_SIZE_ANNOTATION, va="top")
        for x, (_, metric_key, fmt) in zip(col_x[1:], METRIC_COLUMNS, strict=True):
            ax.text(
                x,
                y,
                _format_metric(row, metric_key, fmt),
                fontsize=FONT_SIZE_ANNOTATION,
                va="top",
                ha="center",
            )
    ax.set_title("A2 metrics on paired ORION test tiles", fontsize=FONT_SIZE_TITLE, loc="left", pad=2)


def _blank_tile() -> np.ndarray:
    return np.full((256, 256, 3), 245, dtype=np.uint8)


def _load_tile(path: Path | None) -> np.ndarray:
    if path is None or not Path(path).is_file():
        return _blank_tile()
    return np.asarray(Image.open(path).convert("RGB"))


def _draw_qual_grid(fig: plt.Figure, gs, tile_paths: dict[str, list[Path]]) -> None:
    n_rows = len(ROW_KEYS)
    n_cols = max(1, max((len(paths) for paths in tile_paths.values()), default=4))
    sub = gs.subgridspec(n_rows, n_cols, wspace=0.03, hspace=0.10)
    for row_idx, row_key in enumerate(ROW_KEYS):
        paths = tile_paths.get(row_key, [])
        for col_idx in range(n_cols):
            ax = fig.add_subplot(sub[row_idx, col_idx])
            ax.imshow(_load_tile(paths[col_idx] if col_idx < len(paths) else None))
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)
                spine.set_color("#cccccc")
            if col_idx == 0:
                ax.set_ylabel(
                    ROW_LABELS[row_key],
                    fontsize=FONT_SIZE_ANNOTATION,
                    rotation=0,
                    ha="right",
                    va="center",
                    labelpad=72,
                )


def build_si_a2_bypass_figure(
    *,
    metrics_summary_path: Path,
    tile_paths: dict[str, list[Path]],
) -> plt.Figure:
    """Build the A2 SI figure from metric JSON and generated tile PNG paths."""
    summary = _load_metrics_summary(metrics_summary_path)
    fig = plt.figure(figsize=(15.8, 9.0))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 3.1], hspace=0.10)
    _draw_metric_table(fig.add_subplot(gs[0]), summary)
    _draw_qual_grid(fig, gs[1], tile_paths)
    fig.suptitle("SI A2: Bypass probe under zero_mask_latent", fontsize=FONT_SIZE_TITLE, y=0.995)
    return fig
