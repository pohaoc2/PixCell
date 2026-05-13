"""Figure 6: Combinatorial Grammar - Emergent Signatures."""
from __future__ import annotations

from pathlib import Path

from tools.ablation_report.shared import plt

from src.paper_figures.fig_combinatorial_grammar_panels import _shared
from src.paper_figures.fig_combinatorial_grammar_panels._case_studies import render_panel_c
from src.paper_figures.fig_combinatorial_grammar_panels._diff_grid import render_panel_a
from src.paper_figures.fig_combinatorial_grammar_panels._l2_heatmap import render_panel_b


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_A3_OUT = ROOT / "src" / "a3_combinatorial_sweep" / "out"
DEFAULT_GENERATED_ROOT = DEFAULT_A3_OUT / "generated"
DEFAULT_SIGNATURES_CSV = DEFAULT_A3_OUT / "morphological_signatures.csv"
DEFAULT_RESIDUALS_CSV = DEFAULT_A3_OUT / "additive_model_residuals.csv"
DEFAULT_ABLATION_ROOT = ROOT / "inference_output" / "concat_ablation_1000" / "paired_ablation" / "ablation_results"
DEFAULT_OUT_PNG = ROOT / "figures" / "pngs" / "09_combinatorial_grammar.png"


def _reference_path(ablation_root: Path, anchor_id: str) -> Path:
    return Path(ablation_root) / anchor_id / "all" / "generated_he.png"


def build_combinatorial_grammar_figure(
    *,
    generated_root: Path = DEFAULT_GENERATED_ROOT,
    signatures_csv: Path = DEFAULT_SIGNATURES_CSV,
    residuals_csv: Path = DEFAULT_RESIDUALS_CSV,
    ablation_root: Path = DEFAULT_ABLATION_ROOT,
) -> plt.Figure:
    generated_root = Path(generated_root)
    signatures_csv = Path(signatures_csv)
    residuals_csv = Path(residuals_csv)
    ablation_root = Path(ablation_root)

    if not signatures_csv.is_file():
        raise FileNotFoundError(f"missing signatures csv: {signatures_csv}")
    if not residuals_csv.is_file():
        raise FileNotFoundError(f"missing residuals csv: {residuals_csv}")

    signature_rows = _shared.read_csv(signatures_csv)
    residual_rows = _shared.read_csv(residuals_csv)
    anchor_id = _shared.pick_representative_anchor(signature_rows)
    reference_path = _reference_path(ablation_root, anchor_id)
    if not reference_path.is_file():
        raise FileNotFoundError(
            f"missing reference render for representative anchor {anchor_id}: {reference_path}\n"
            "Run: python -m src.a3_combinatorial_sweep.generate_references "
            "--anchors src/a3_combinatorial_sweep/anchors_k20_t1_medoid.txt "
            f"--output-root {ablation_root} ..."
        )

    fig = plt.figure(figsize=(12.0, 9.0), facecolor="white")
    outer = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.6, 1.0],
        height_ratios=[1.0, 1.0],
        wspace=0.22,
        hspace=0.30,
    )

    render_panel_a(
        fig,
        outer[0, 0],
        anchor_id=anchor_id,
        generated_root=generated_root,
        reference_path=reference_path,
    )
    render_panel_b(fig, outer[1, 0], residual_rows=residual_rows)
    render_panel_c(fig, outer[:, 1], residual_rows=residual_rows)

    fig.subplots_adjust(left=0.04, right=0.97, bottom=0.06, top=0.95)
    return fig


def save_combinatorial_grammar_figure(
    *,
    out_png: Path = DEFAULT_OUT_PNG,
    generated_root: Path = DEFAULT_GENERATED_ROOT,
    signatures_csv: Path = DEFAULT_SIGNATURES_CSV,
    residuals_csv: Path = DEFAULT_RESIDUALS_CSV,
    ablation_root: Path = DEFAULT_ABLATION_ROOT,
    dpi: int = 300,
) -> Path:
    fig = build_combinatorial_grammar_figure(
        generated_root=generated_root,
        signatures_csv=signatures_csv,
        residuals_csv=residuals_csv,
        ablation_root=ablation_root,
    )
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_png

