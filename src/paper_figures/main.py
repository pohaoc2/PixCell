"""Generate paper figures 01 through 09 when inputs are available."""
import gc
from pathlib import Path

from src.paper_figures.fig_ablation_grid import build_representative_ablation_grid
from src.paper_figures.fig_combined_ablation_grids import build_combined_ablation_grids_figure
from src.paper_figures.fig_combined_performance import build_combined_performance_figure
from src.paper_figures.fig_inverse_decoding import build_inverse_decoding_figure
from src.paper_figures.fig_t1_spatial_multi_encoder import build_figure as build_t1_spatial_multi_encoder_figure
from src.paper_figures.fig_si_a1_a2_unified import save_split_figures as save_si_a1_a2_split_figures
from src.paper_figures.fig_si_a2_bypass import build_si_a2_bypass_figure
from src.paper_figures.fig_si_a3_zero_init import build_si_a3_zero_init_figure
from src.paper_figures.fig_uni_tme_decomposition import save_uni_tme_decomposition_figure
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


PAIRED_METRICS_ROOT = ROOT / "inference_output" / "concat_ablation_1000" / "paired_ablation" / "ablation_results"
PAIRED_DATASET_ROOT = ROOT / "inference_output" / "concat_ablation_1000" / "paired_ablation"
PAIRED_REFERENCE_ROOT = ROOT / "data" / "orion-crc33"
PROBE_ENCODERS_OUT = ROOT / "src" / "a1_probe_encoders" / "out"
T1_UNI_CSV = ROOT / "src" / "a1_probe_linear" / "out" / "linear_probe_results.csv"
T1_VIRCHOW_CSV = PROBE_ENCODERS_OUT / "virchow2_linear_probe_results.csv"
T1_CTRANSPATH_CSV = PROBE_ENCODERS_OUT / "ctranspath_linear_probe_results.csv"
T1_RESNET50_CSV = PROBE_ENCODERS_OUT / "resnet50_linear_probe_results.csv"
T1_REMEDIS_CSV = PROBE_ENCODERS_OUT / "remedis_linear_probe_results.csv"
T2_MLP_CSV = ROOT / "src" / "a1_codex_targets" / "probe_out" / "t2_mlp" / "mlp_probe_results.csv"
T2_SPATIAL_CSV = ROOT / "src" / "a1_probe_mlp_spatial" / "out" / "t2_spatial" / "mlp_spatial_probe_results.csv"
T1_SPATIAL_MULTI_ENCODER_CSVS = {
    "UNI-2h": ROOT / "src" / "a1_probe_mlp_spatial" / "out" / "uni_16" / "mlp_spatial_probe_results.csv",
    "Virchow2": ROOT / "src" / "a1_probe_mlp_spatial" / "out" / "virchow2_16" / "mlp_spatial_probe_results.csv",
    "CTransPath": ROOT / "src" / "a1_probe_mlp_spatial" / "out" / "ctranspath_07" / "mlp_spatial_probe_results.csv",
    "ResNet-50": ROOT / "src" / "a1_probe_mlp_spatial" / "out" / "resnet50_07" / "mlp_spatial_probe_results.csv",
    "REMEDIS": ROOT / "src" / "a1_probe_mlp_spatial" / "out" / "remedis_07" / "mlp_spatial_probe_results.csv",
}
DECOMPOSITION_SUMMARY_CSV = ROOT / "src" / "a2_decomposition" / "out" / "decomposition_summary.csv"

A2_METRICS_SUMMARY = ROOT / "inference_output" / "a2_bypass" / "metrics_summary.json"
A2_TILE_DIRS = {
    "production": ROOT / "inference_output" / "paired_ablation" / "production",
    "bypass_probe": ROOT / "inference_output" / "a2_bypass" / "bypass",
    "off_the_shelf": ROOT / "inference_output" / "a2_bypass" / "offshelf",
}
A2_TILE_IDS = ["10752_13824", "13056_27392", "5632_18432", "21504_8192"]

A3_METRICS_SUMMARY = ROOT / "inference_output" / "a3_zero_init" / "metrics_summary.json"
A3_STABILITY_TRUE = ROOT / "inference_output" / "a3_zero_init" / "stability_true.json"
A3_STABILITY_FALSE = ROOT / "inference_output" / "a3_zero_init" / "stability_false.json"
A3_SEEDS_TRUE_LOGS = sorted((ROOT / "checkpoints" / "pixcell_controlnet_exp").glob("seed_*/train_log.jsonl")) + sorted(
    (ROOT / "checkpoints" / "pixcell_controlnet_exp").glob("seed_*/train_log.log")
)
A3_SEEDS_FALSE_LOGS = sorted(
    (ROOT / "checkpoints" / "pixcell_controlnet_exp_a3_no_zero_init").glob("seed_*/train_log.jsonl")
) + sorted((ROOT / "checkpoints" / "pixcell_controlnet_exp_a3_no_zero_init").glob("seed_*/train_log.log"))

UNPAIRED_METRICS_ROOT = ROOT / "inference_output" / "concat_ablation_1000" / "unpaired_ablation" / "ablation_results"
UNPAIRED_DATASET_ROOT = ROOT / "inference_output" / "concat_ablation_1000" / "unpaired_ablation"
UNPAIRED_REFERENCE_ROOT = ROOT / "data" / "orion-crc33"

# Output directory structure
FIGURES_ROOT = ROOT / "figures" / "pngs_updated"
CONCAT_DIR = FIGURES_ROOT / "concat"           # multi-panel / composite figures
IND = FIGURES_ROOT / "individual"
MODEL_PERF = IND / "model_performance"         # 01-03: quantitative performance
ABLATION = IND / "ablation_analysis"           # 04-06: channel ablation & visual quality
ENCODER = IND / "encoder_probes"               # 07: encoder probe analysis
CHANNEL = IND / "channel_utility"              # 09: single-panel TME channel utility
SI_A1_A2 = IND / "si_a1_a2"                   # SI A1/A2 section panels


def main() -> None:
    apply_style()
    for d in [CONCAT_DIR, MODEL_PERF, ABLATION, ENCODER, CHANNEL, SI_A1_A2]:
        d.mkdir(parents=True, exist_ok=True)

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

    # --- model_performance: 01-03 ---
    fig_trends = build_metric_trends_figure(summaries)
    save_figure_png(fig_trends, MODEL_PERF / "01_metric_tradeoffs.png")

    fig_comparison = build_comparison_table_figure(summaries)
    save_figure_png(fig_comparison, MODEL_PERF / "02_model_comparison.png")

    fig_heatmap = build_channel_effect_heatmaps_figure(summaries)
    save_figure_png(fig_heatmap, MODEL_PERF / "03_channel_effect_sizes.png")

    # --- ablation_analysis: 04 (channel LOO) ---
    fig_loo = build_leave_one_out_figure(summaries)
    save_figure_png(fig_loo, ABLATION / "04_leave_one_out_impact.png")

    # Composite: paired + unpaired performance summary → concat/
    fig_combined = build_combined_performance_figure(summaries)
    save_figure_png(fig_combined, CONCAT_DIR / "performance_paired_unpaired.png")

    # Release summary data before the image-heavy ablation grids to reduce peak RAM.
    del paired_summary, unpaired_summary, summaries
    gc.collect()

    # --- ablation_analysis: 05-06 (visual ablation grids) ---
    build_representative_ablation_grid(
        metrics_root=PAIRED_METRICS_ROOT,
        dataset_root=PAIRED_DATASET_ROOT,
        orion_root=PAIRED_REFERENCE_ROOT,
        out_png=ABLATION / "05_paired_ablation_grid.png",
        tile_id="10752_13824",
    )

    build_representative_ablation_grid(
        metrics_root=UNPAIRED_METRICS_ROOT,
        dataset_root=UNPAIRED_DATASET_ROOT,
        orion_root=UNPAIRED_REFERENCE_ROOT,
        out_png=ABLATION / "06_unpaired_ablation_grid.png",
        tile_id="13056_27392",
        style_mapping_json=ROOT / "inference_output" / "concat_ablation_1000" / "unpaired_ablation" / "metadata" / "unpaired_mapping.json",
    )

    # Composite: paired + unpaired ablation grids side-by-side → concat/
    fig_ablation_grids = build_combined_ablation_grids_figure(
        ABLATION / "05_paired_ablation_grid.png",
        ABLATION / "06_unpaired_ablation_grid.png",
    )
    save_figure_png(fig_ablation_grids, CONCAT_DIR / "ablation_grids_combined.png")

    # --- concat/: SI A2, A3 supplemental composites ---
    if A2_METRICS_SUMMARY.is_file():
        a2_tile_paths = {
            key: [directory / f"{tile_id}.png" for tile_id in A2_TILE_IDS if (directory / f"{tile_id}.png").is_file()]
            for key, directory in A2_TILE_DIRS.items()
        }
        fig_a2 = build_si_a2_bypass_figure(
            metrics_summary_path=A2_METRICS_SUMMARY,
            tile_paths=a2_tile_paths,
        )
        save_figure_png(fig_a2, CONCAT_DIR / "SI_A2_bypass_probe.png")
    else:
        print("Skipping SI_A2_bypass_probe.png; missing", A2_METRICS_SUMMARY)

    if A3_METRICS_SUMMARY.is_file() and A3_STABILITY_TRUE.is_file() and A3_STABILITY_FALSE.is_file():
        fig_a3 = build_si_a3_zero_init_figure(
            seeds_true_logs=A3_SEEDS_TRUE_LOGS,
            seeds_false_logs=A3_SEEDS_FALSE_LOGS,
            stability_summary_true_path=A3_STABILITY_TRUE,
            stability_summary_false_path=A3_STABILITY_FALSE,
            metrics_summary_path=A3_METRICS_SUMMARY,
        )
        save_figure_png(fig_a3, CONCAT_DIR / "SI_A3_zero_init.png")
    else:
        print(
            "Skipping SI_A3_zero_init.png; missing one of",
            A3_METRICS_SUMMARY,
            A3_STABILITY_TRUE,
            A3_STABILITY_FALSE,
        )

    # --- encoder_probes/: 07 (individual); concat/: 07d (multi-encoder composite) ---
    if T2_MLP_CSV.is_file():
        fig_inverse_decoding = build_inverse_decoding_figure(
            uni_t1_csv=T1_UNI_CSV,
            virchow_t1_csv=T1_VIRCHOW_CSV if T1_VIRCHOW_CSV.is_file() else None,
            ctranspath_t1_csv=T1_CTRANSPATH_CSV if T1_CTRANSPATH_CSV.is_file() else None,
            resnet50_t1_csv=T1_RESNET50_CSV if T1_RESNET50_CSV.is_file() else None,
            remedis_t1_csv=T1_REMEDIS_CSV if T1_REMEDIS_CSV.is_file() else None,
            t2_mlp_csv=T2_MLP_CSV,
        )
        save_figure_png(fig_inverse_decoding, ENCODER / "07_inverse_decoding.png")
    else:
        print("Skipping 07_inverse_decoding.png; missing", T2_MLP_CSV)

    if any(path.is_file() for path in T1_SPATIAL_MULTI_ENCODER_CSVS.values()) and T2_SPATIAL_CSV.is_file():
        fig_t1_spatial_multi_encoder = build_t1_spatial_multi_encoder_figure(
            encoder_csvs=T1_SPATIAL_MULTI_ENCODER_CSVS,
            t2_spatial_csv=T2_SPATIAL_CSV,
        )
        save_figure_png(fig_t1_spatial_multi_encoder, CONCAT_DIR / "07d_t1_spatial_multi_encoder.png")
    else:
        print("Skipping 07d_t1_spatial_multi_encoder.png; missing multi-encoder or T2 spatial probe CSVs")

    # --- concat/: 08 TME decomposition composite, channel_utility/: 09 single panel ---
    if DECOMPOSITION_SUMMARY_CSV.is_file():
        save_uni_tme_decomposition_figure(out_png=CONCAT_DIR / "08_uni_tme_decomposition.png")
    else:
        print("Skipping 08_uni_tme_decomposition.png; missing", DECOMPOSITION_SUMMARY_CSV)

    # --- si_a1_a2: section panels (individual/) + unified (concat/) ---
    SI_A1_A2_CACHE = ROOT / "inference_output" / "si_a1_a2" / "cache.json"
    if SI_A1_A2_CACHE.is_file():
        save_si_a1_a2_split_figures(
            cache_path=SI_A1_A2_CACHE,
            tile_dir=SI_A1_A2_CACHE.parent / "tiles",
            out=CONCAT_DIR / "SI_A1_A2_unified.png",
            dpi=300,
            section_dir=SI_A1_A2,
        )
    else:
        print("Skipping SI_A1_A2_unified.png; missing", SI_A1_A2_CACHE)

    print(
        "Figures written to:\n"
        f"  concat/     → {CONCAT_DIR}\n"
        f"  individual/ → {IND}\n"
        "  Subfolders: model_performance, ablation_analysis, encoder_probes,\n"
        "              channel_utility, si_a1_a2"
    )


if __name__ == "__main__":
    main()
