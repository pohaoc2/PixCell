"""Generate paper figures 01, 02, 03, and 04."""
from pathlib import Path

from src.paper_figures.fig_ablation_grid import build_representative_ablation_grid
from src.paper_figures.fig_inverse_decoding import build_inverse_decoding_figure
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
PROBE_ENCODERS_OUT = ROOT / "src" / "a1_probe_encoders" / "out"
T1_UNI_CSV = ROOT / "src" / "a1_probe_linear" / "out" / "linear_probe_results.csv"
T1_VIRCHOW_CSV = PROBE_ENCODERS_OUT / "virchow2_linear_probe_results.csv"
T1_CTRANSPATH_CSV = PROBE_ENCODERS_OUT / "ctranspath_linear_probe_results.csv"
T1_RESNET50_CSV = PROBE_ENCODERS_OUT / "resnet50_linear_probe_results.csv"
T1_REMEDIS_CSV = PROBE_ENCODERS_OUT / "remedis_linear_probe_results.csv"
T2_MLP_CSV = ROOT / "src" / "a1_codex_targets" / "probe_out" / "t2_mlp" / "mlp_probe_results.csv"

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

    if T2_MLP_CSV.is_file():
        fig_inverse_decoding = build_inverse_decoding_figure(
            uni_t1_csv=T1_UNI_CSV,
            virchow_t1_csv=T1_VIRCHOW_CSV if T1_VIRCHOW_CSV.is_file() else None,
            ctranspath_t1_csv=T1_CTRANSPATH_CSV if T1_CTRANSPATH_CSV.is_file() else None,
            resnet50_t1_csv=T1_RESNET50_CSV if T1_RESNET50_CSV.is_file() else None,
            remedis_t1_csv=T1_REMEDIS_CSV if T1_REMEDIS_CSV.is_file() else None,
            t2_mlp_csv=T2_MLP_CSV,
        )
        save_figure_png(fig_inverse_decoding, PNG_DIR / "07_inverse_decoding.png")
    else:
        print("Skipping 07_inverse_decoding.png; missing", T2_MLP_CSV)

    print(
        "Saved 01_metric_tradeoffs.png, 02_paired_vs_unpaired.png, "
        "03_channel_effect_sizes.png, 04_leave_one_out_impact.png, "
        "05_paired_ablation_grid.png, 06_unpaired_ablation_grid.png, "
        "and 07_inverse_decoding.png when inputs are available to",
        PNG_DIR,
    )


if __name__ == "__main__":
    main()
