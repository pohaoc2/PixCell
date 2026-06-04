"""Render the standalone SI sub-panel assets used by the SI preview composite.

The paper reorganization promoted most panels of the old composite figures into
the main figures (fig1-fig4). Two SI items survive as *single panels* of their
former composites:

* ``si_performance_ranking.png`` -- Panel B of ``performance_paired_unpaired``
  (the paired/unpaired top-3 / bottom-3 ranking tables), re-rendered standalone
  from the same ``build_comparison_table_figure`` builder.
* ``si_a1a2_qualitative_tiles.png`` -- Panel C of ``SI_A1_A2_unified`` (the
  per-variant qualitative tile grid); copied from the already-standalone
  ``individual/si_a1_a2/SI_A1_A2_section3_tiles.png`` so the preview can pick it
  up from ``concat/`` like any other figure.

Run:  python -m src.paper_figures.build_si_assets
"""
from __future__ import annotations

import shutil
from pathlib import Path

from tools.ablation_report.data import load_dataset_summary
from tools.ablation_report.figures import build_comparison_table_figure, save_figure_png
from tools.stage3.style_mapping import load_style_mapping
from src.paper_figures.style import apply_style
from src.paper_figures.main import (
    PAIRED_METRICS_ROOT,
    PAIRED_DATASET_ROOT,
    PAIRED_REFERENCE_ROOT,
    UNPAIRED_METRICS_ROOT,
    UNPAIRED_DATASET_ROOT,
    UNPAIRED_REFERENCE_ROOT,
)

ROOT = Path(__file__).resolve().parents[2]
CONCAT_DIR = ROOT / "figures" / "pngs_updated" / "concat"
SECTION3_TILES = ROOT / "figures" / "pngs_updated" / "individual" / "si_a1_a2" / "SI_A1_A2_section3_tiles.png"

OUT_RANKING = CONCAT_DIR / "si_performance_ranking.png"
OUT_TILES = CONCAT_DIR / "si_a1a2_qualitative_tiles.png"


def build() -> None:
    apply_style()

    paired = load_dataset_summary(
        slug="paired",
        title="Paired",
        metrics_root=PAIRED_METRICS_ROOT,
        dataset_root=PAIRED_DATASET_ROOT,
        reference_root=PAIRED_REFERENCE_ROOT,
        style_mapping=load_style_mapping(None),
    )
    unpaired = load_dataset_summary(
        slug="unpaired",
        title="Unpaired",
        metrics_root=UNPAIRED_METRICS_ROOT,
        dataset_root=UNPAIRED_DATASET_ROOT,
        reference_root=UNPAIRED_REFERENCE_ROOT,
        style_mapping=load_style_mapping(None),
    )
    fig = build_comparison_table_figure([paired, unpaired])
    save_figure_png(fig, OUT_RANKING)
    print(f"wrote {OUT_RANKING}")

    # Panel C qualitative tiles already exist standalone -- stage them in concat/.
    shutil.copyfile(SECTION3_TILES, OUT_TILES)
    print(f"copied {SECTION3_TILES} -> {OUT_TILES}")


if __name__ == "__main__":
    build()
