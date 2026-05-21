"""Figure 6: Combinatorial Grammar - Emergent Signatures."""
from __future__ import annotations

from pathlib import Path

from tools.ablation_report.shared import plt


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_A3_OUT = ROOT / "src" / "a3_combinatorial_sweep" / "out"
DEFAULT_GENERATED_ROOT = DEFAULT_A3_OUT / "generated"
DEFAULT_SIGNATURES_CSV = DEFAULT_A3_OUT / "morphological_signatures.csv"
DEFAULT_RESIDUALS_CSV = DEFAULT_A3_OUT / "additive_model_residuals.csv"
DEFAULT_ABLATION_ROOT = ROOT / "inference_output" / "concat_ablation_1000" / "paired_ablation" / "ablation_results"
DEFAULT_OUT_PNG = ROOT / "figures" / "pngs_updated" / "09_combinatorial_grammar.png"


def _reference_path(ablation_root: Path, anchor_id: str) -> Path:
    return Path(ablation_root) / anchor_id / "all" / "generated_he.png"


def _draw_anchor_sweep_grid(fig, subgrid, *, anchor_id: str, generated_root: Path) -> None:
    from src.paper_figures.fig_combinatorial_grammar_panels._shared import STATES, LEVELS, condition_id, load_rgb

    outer = subgrid.subgridspec(3, 9, hspace=0.04, wspace=0.04)
    for state_idx, state in enumerate(STATES):
        for ox_idx, ox in enumerate(LEVELS):
            for gluc_idx, gluc in enumerate(LEVELS):
                col = ox_idx * len(LEVELS) + gluc_idx
                ax = fig.add_subplot(outer[state_idx, col])
                tile_path = generated_root / anchor_id / f"{condition_id(state, ox, gluc)}.png"
                ax.imshow(load_rgb(tile_path))
                ax.set_xticks([])
                ax.set_yticks([])
                if col == 0:
                    ax.set_ylabel(state, fontsize=8)
                if state_idx == 0:
                    ax.set_title(f"{ox}/{gluc}", fontsize=7, pad=1.0)
                for spine in ax.spines.values():
                    spine.set_linewidth(0.25)
                    spine.set_edgecolor("#8A8A8A")


def build_combinatorial_grammar_figure(
    *,
    generated_root: Path = DEFAULT_GENERATED_ROOT,
    signatures_csv: Path = DEFAULT_SIGNATURES_CSV,
    variance_csv: Path | None = None,
    residuals_csv: Path | None = None,
    ablation_root: Path | None = None,
) -> plt.Figure:
    from src.paper_figures.fig_combinatorial_grammar_panels._variance_bars import draw_variance_bars
    from src.paper_figures.fig_combinatorial_grammar_panels._shared import (
        compute_anchor_sweep_magnitude,
        read_csv,
    )

    generated_root = Path(generated_root)
    signatures_csv = Path(signatures_csv)
    variance_csv = variance_csv if variance_csv is not None else signatures_csv.parent / "variance_partition.csv"
    signature_rows = read_csv(signatures_csv)
    magnitudes = compute_anchor_sweep_magnitude(signature_rows)
    panel_b_anchor = sorted(magnitudes.items(), key=lambda pair: (-pair[1], pair[0]))[0][0]

    fig = plt.figure(figsize=(7.5, 12.0), facecolor="white")
    gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.0, 1.2], hspace=0.45)

    ax = fig.add_subplot(gs[0])
    draw_variance_bars(ax, Path(variance_csv), title="Variance partition (full data)")

    ax2 = fig.add_subplot(gs[1])
    within_csv = signatures_csv.parent / "variance_partition_within.csv"
    draw_variance_bars(ax2, within_csv, title="Variance partition (within-anchor)")

    _draw_anchor_sweep_grid(fig, gs[2].subgridspec(1, 1)[0, 0], anchor_id=panel_b_anchor, generated_root=generated_root)

    return fig


def save_combinatorial_grammar_figure(
    *,
    out_png: Path = DEFAULT_OUT_PNG,
    generated_root: Path = DEFAULT_GENERATED_ROOT,
    signatures_csv: Path = DEFAULT_SIGNATURES_CSV,
    variance_csv: Path | None = None,
    residuals_csv: Path | None = None,
    ablation_root: Path | None = None,
    dpi: int = 300,
) -> Path:
    fig = build_combinatorial_grammar_figure(
        generated_root=generated_root,
        signatures_csv=signatures_csv,
        variance_csv=variance_csv,
        residuals_csv=residuals_csv,
        ablation_root=ablation_root,
    )
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_png


if __name__ == "__main__":
    save_combinatorial_grammar_figure()
