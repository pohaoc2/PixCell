"""SI figure for figure 6: raw H&E sweep grids for 4 anchors."""
from __future__ import annotations

from pathlib import Path

from src.paper_figures.style import FONT_SIZE_DENSE_LABEL, FONT_SIZE_DENSE_TITLE, FONT_SIZE_INLINE, FONT_SIZE_LABEL
from tools.ablation_report.shared import INK, plt

from src.paper_figures.fig_combinatorial_grammar_panels import _shared


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_A3_OUT = ROOT / "src" / "a3_combinatorial_sweep" / "out"
DEFAULT_GENERATED_ROOT = DEFAULT_A3_OUT / "generated"
DEFAULT_SIGNATURES_CSV = DEFAULT_A3_OUT / "morphological_signatures.csv"
DEFAULT_ABLATION_ROOT = ROOT / "inference_output" / "paired_ablation" / "ablation_results"
DEFAULT_OUT_PNG = ROOT / "figures" / "pngs" / "SI_09_combinatorial_grammar_anchors.png"

STATES = _shared.STATES
LEVELS = _shared.LEVELS


def _draw_anchor_subgrid(
    fig: plt.Figure,
    subgrid,
    *,
    anchor_id: str,
    generated_root: Path,
    title: str,
) -> None:
    outer_ax = fig.add_subplot(subgrid)
    outer_ax.axis("off")
    outer_ax.text(
        0.0,
        1.02,
        title,
        transform=outer_ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=FONT_SIZE_INLINE,
        color=INK,
    )

    inner = subgrid.subgridspec(3, 9, hspace=0.04, wspace=0.04)
    for state_idx, state in enumerate(STATES):
        for oxygen_idx, oxygen_label in enumerate(LEVELS):
            for glucose_idx, glucose_label in enumerate(LEVELS):
                col = oxygen_idx * len(LEVELS) + glucose_idx
                tile_path = generated_root / anchor_id / f"{_shared.condition_id(state, oxygen_label, glucose_label)}.png"
                ax = fig.add_subplot(inner[state_idx, col])
                ax.imshow(_shared.load_rgb(tile_path))
                ax.set_xticks([])
                ax.set_yticks([])
                if col == 0:
                    ax.set_ylabel(state, fontsize=FONT_SIZE_DENSE_LABEL, color=INK)
                if state_idx == 0:
                    ax.set_title(f"{oxygen_label}/{glucose_label}", fontsize=FONT_SIZE_DENSE_TITLE, pad=1.2, color=INK)
                for spine in ax.spines.values():
                    spine.set_linewidth(0.25)
                    spine.set_edgecolor("#8A8A8A")


def build_combinatorial_grammar_si_figure(
    *,
    generated_root: Path = DEFAULT_GENERATED_ROOT,
    signatures_csv: Path = DEFAULT_SIGNATURES_CSV,
    ablation_root: Path = DEFAULT_ABLATION_ROOT,
) -> plt.Figure:
    generated_root = Path(generated_root)
    signatures_csv = Path(signatures_csv)
    ablation_root = Path(ablation_root)
    if not signatures_csv.is_file():
        raise FileNotFoundError(f"missing signatures csv: {signatures_csv}")

    signature_rows = _shared.read_csv(signatures_csv)
    representative = _shared.pick_representative_anchor(signature_rows)

    def _has_reference(anchor_id: str) -> bool:
        return (ablation_root / anchor_id / "all" / "generated_he.png").is_file()

    picks = _shared.select_si_anchors(
        signature_rows,
        representative_id=representative,
        reference_exists_fn=_has_reference,
    )
    if not picks:
        raise FileNotFoundError(f"no anchors with reference renders found under {ablation_root}")

    magnitudes = _shared.compute_anchor_sweep_magnitude(signature_rows)
    fig = plt.figure(figsize=(14.0, 10.0), facecolor="white")
    outer = fig.add_gridspec(2, 2, wspace=0.10, hspace=0.20)
    role_labels = ("representative", "low magnitude", "mid magnitude", "high magnitude")
    for idx, anchor_id in enumerate(picks[:4]):
        row, col = divmod(idx, 2)
        title = f"{role_labels[idx]}: anchor {anchor_id} (sweep magnitude={magnitudes.get(anchor_id, 0.0):.3g})"
        _draw_anchor_subgrid(fig, outer[row, col], anchor_id=anchor_id, generated_root=generated_root, title=title)

    fig.text(0.02, 0.97, "Figure S6", fontsize=FONT_SIZE_LABEL, color=INK, ha="left", va="top", fontweight="bold")
    fig.subplots_adjust(left=0.04, right=0.98, bottom=0.04, top=0.94)
    return fig


def save_combinatorial_grammar_si_figure(
    *,
    out_png: Path = DEFAULT_OUT_PNG,
    generated_root: Path = DEFAULT_GENERATED_ROOT,
    signatures_csv: Path = DEFAULT_SIGNATURES_CSV,
    ablation_root: Path = DEFAULT_ABLATION_ROOT,
    dpi: int = 300,
) -> Path:
    fig = build_combinatorial_grammar_si_figure(
        generated_root=generated_root,
        signatures_csv=signatures_csv,
        ablation_root=ablation_root,
    )
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_png
