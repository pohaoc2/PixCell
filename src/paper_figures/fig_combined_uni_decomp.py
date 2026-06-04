"""Combined Fig 3: UNI vs TME information content.

Two stacked halves, full-width composite consistent with the other `concat/`
figures (220 DPI raster stack, ~15.8 in target width):

    Top:    A/B/C = UNI/TME causal decomposition (08_uni_tme_decomposition.png)
    Bottom: D/E/F = UNI linear-probe overview (probe ΔR², specificity, sweep grid)

The top half is reused as-is. The bottom is re-rendered at current PANEL_SQUARE_IN
/ COMBINED_GAP_IN dimensions so D/E sizes and the A-F panel-letter font sizes are
consistent. Both halves are resized to the shared composite width and stacked.

Panel-letter font-size alignment:
  W_top ≈ 7.53 in (top, 12 pt labels) × 1.5 ≈ W_bot ≈ 11.44 in (bottom, 18 pt
  labels) ⟹  18/11.44 ≈ 12/7.53, so both appear the same size after scaling to
  composite width.

Run:  python -m src.paper_figures.fig_combined_uni_decomp
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from src.paper_figures.style import apply_style
from src.paper_figures.fig_combined_method_perf import (
    _RENDER_DPI,
    _load_rgb,
    _resize_to_width,
)
from src.a4_uni_probe.figures import (
    render_pngs_updated_combined_abc,
    render_pngs_updated_probe_delta,
    render_pngs_updated_specificity_heatmap,
)
from tools.ablation_report.shared import ROOT
from tools.ablation_report.figures import save_figure_png

# ── Inputs / outputs ────────────────────────────────────────────────────────
TOP_PNG = ROOT / "figures" / "pngs_updated" / "concat" / "08_uni_tme_decomposition.png"
UNI_PROBE_DIR = ROOT / "figures" / "pngs_updated" / "individual" / "uni_probe"
# Source data directory with probe_results.csv, specificity_full.csv, sweep/
PROBE_OUT_DIR = ROOT / "inference_output" / "a1_concat" / "a4_uni_probe"
OUT_PNG = ROOT / "figures" / "pngs_updated" / "concat" / "fig3_uni_decomposition.png"

# ── Composition geometry (mirrors the other concat figures) ─────────────────
COMPOSITE_WIDTH_IN = 15.8
_ROW_GAP_PX = 60   # white gutter between the top and bottom halves


def _render_bottom_DEF() -> np.ndarray:
    """Re-render the uni-probe overview with letters D/E/F at current sizes.

    All three sub-panels (probe ΔR², specificity, sweep grid) are regenerated
    from source data so their sizes reflect the current PANEL_SQUARE_IN /
    COMBINED_GAP_IN constants. No cached PNGs are loaded.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        render_pngs_updated_probe_delta(PROBE_OUT_DIR, tmp_path)
        render_pngs_updated_specificity_heatmap(PROBE_OUT_DIR, tmp_path)
        out = render_pngs_updated_combined_abc(
            PROBE_OUT_DIR,   # out_dir: source data (sweep/, probe_results.csv …)
            tmp_path,        # dest_dir: receives the freshly rendered D/E PNGs
            concat_dir=tmp,
            panel_letters=("D", "E", "F"),
            out_name="uni_probe_overview_def.png",
        )
        return _load_rgb(Path(out))


def build_uni_decomp_figure() -> plt.Figure:
    full_width = round(COMPOSITE_WIDTH_IN * _RENDER_DPI)

    top = _resize_to_width(_load_rgb(TOP_PNG), full_width)
    bottom = _resize_to_width(_render_bottom_DEF(), full_width)
    gutter = np.full((_ROW_GAP_PX, full_width, 3), 255, dtype=np.uint8)
    composite = np.concatenate([top, gutter, bottom], axis=0)

    fig = plt.figure(figsize=(full_width / _RENDER_DPI, composite.shape[0] / _RENDER_DPI))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(composite, interpolation="none")
    ax.axis("off")
    return fig


def main() -> None:
    apply_style()
    fig = build_uni_decomp_figure()
    save_figure_png(fig, OUT_PNG)
    print("wrote", OUT_PNG)


if __name__ == "__main__":
    main()
