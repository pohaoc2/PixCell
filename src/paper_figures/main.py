"""Generate paper figures 01, 02, 03, and 04."""
from pathlib import Path

from src.paper_figures.fig_ablation_grid import build_representative_ablation_grid
from tools.ablation_report.figures import (
    build_channel_effect_heatmaps_figure,
    build_leave_one_out_figure,
    build_metric_trends_figure,
    build_comparison_table_figure,
    save_figure_png,
)
from tools.ablation_report.data import load_dataset_summary
from tools.stage3.style_mapping import load_style_mapping
from tools.ablation_report.shared import ROOT
from src.paper_figures.style import apply_style


PAIRED_METRICS_ROOT = ROOT / "inference_output" / "paired_ablation" / "ablation_results"
PAIRED_DATASET_ROOT = ROOT / "inference_output" / "paired_ablation"
PAIRED_REFERENCE_ROOT = ROOT / "data" / "orion-crc33"

UNPAIRED_METRICS_ROOT = ROOT / "inference_output" / "unpaired_ablation" / "ablation_results"
UNPAIRED_DATASET_ROOT = ROOT / "inference_output" / "unpaired_ablation"
UNPAIRED_REFERENCE_ROOT = ROOT / "data" / "orion-crc33"

PNG_DIR = ROOT / "figures" / "pngs"


def main() -> None:
    apply_style()
    PNG_DIR.mkdir(parents=True, exist_ok=True)

    paired_summary = load_dataset_summary(
        slug="paired",
        title="Paired",
        metrics_root=PAIRED_METRICS_ROOT,
        dataset_root=PAIRED_DATASET_ROOT,
        reference_root=PAIRED_REFERENCE_ROOT,
        style_mapping=load_style_mapping(None),
    )
    unpaired_summary = load_dataset_summary(
        slug="unpaired",
        title="Unpaired",
        metrics_root=UNPAIRED_METRICS_ROOT,
        dataset_root=UNPAIRED_DATASET_ROOT,
        reference_root=UNPAIRED_REFERENCE_ROOT,
        style_mapping=load_style_mapping(None),
    )
    summaries = [paired_summary, unpaired_summary]

    fig_trends = build_metric_trends_figure(summaries)
    save_figure_png(fig_trends, PNG_DIR / "01_metric_tradeoffs.png")

    fig_comparison = build_comparison_table_figure(summaries)
    save_figure_png(fig_comparison, PNG_DIR / "02_paired_vs_unpaired.png")

    fig_heatmap = build_channel_effect_heatmaps_figure(summaries)
    save_figure_png(fig_heatmap, PNG_DIR / "03_channel_effect_sizes.png")

    fig_loo = build_leave_one_out_figure(summaries)
    save_figure_png(fig_loo, PNG_DIR / "04_leave_one_out_impact.png")

    build_representative_ablation_grid(
        metrics_root=PAIRED_METRICS_ROOT,
        dataset_root=PAIRED_DATASET_ROOT,
        orion_root=PAIRED_REFERENCE_ROOT,
        out_png=PNG_DIR / "05_paired_ablation_grid.png",
    )

    build_representative_ablation_grid(
        metrics_root=UNPAIRED_METRICS_ROOT,
        dataset_root=UNPAIRED_DATASET_ROOT,
        orion_root=UNPAIRED_REFERENCE_ROOT,
        out_png=PNG_DIR / "06_unpaired_ablation_grid.png",
    )

    print(
        "Saved 01_metric_tradeoffs.png, 02_paired_vs_unpaired.png, "
        "03_channel_effect_sizes.png, 04_leave_one_out_impact.png, "
        "05_paired_ablation_grid.png, and 06_unpaired_ablation_grid.png to",
        PNG_DIR,
    )


if __name__ == "__main__":
    main()
