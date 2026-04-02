"""
4×4 ablation grid figure: 16 conditions (14 ablation + All-4-ch + Real H&E) sorted by cosine similarity.

Reads ``manifest.json`` (14 ablation conditions) plus a separately supplied All-4-ch image.
Cosine scores are loaded from ``uni_cosine_scores.json`` or auto-computed via UNI-2h.

Outputs ``<cache_dir>/ablation_grid.png`` (and optionally ``.pdf``).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.stage3_ablation_cache import is_per_tile_cache_manifest_dir, list_cached_tile_ids
from tools.stage3_ablation_vis_utils import (
    FOUR_GROUP_ORDER,
    condition_metric_key,
    ordered_subset_condition_tuples,
)

# Okabe-Ito palette (colorblind-safe)
_COLOR_BY_CARD: dict[int, str] = {
    1: "#009E73",  # bluish green
    2: "#0072B2",  # blue
    3: "#D55E00",  # vermillion
    4: "#9B59B6",  # purple
}
COLOR_REF = "#999999"
COLOR_INACTIVE = "#CCCCCC"
BEST_BG = "#FFFBE6"

# CT · CS · Va · Nu (matches FOUR_GROUP_ORDER)
_GROUP_SHORT: dict[str, str] = {
    "cell_types": "CT",
    "cell_state": "CS",
    "vasculature": "Va",
    "microenv": "Nu",
}

ALL4CH_KEY: str = condition_metric_key(FOUR_GROUP_ORDER)


def _cardinality_color(n: int) -> str:
    """Return Okabe-Ito hex for a given channel count (1–4)."""
    return _COLOR_BY_CARD[n]


def _condition_label(cond: tuple[str, ...]) -> str:
    """Short label following FOUR_GROUP_ORDER: e.g. 'CT+CS+Nu'."""
    return "+".join(_GROUP_SHORT[g] for g in FOUR_GROUP_ORDER if g in cond)


def _find_real_he(orion_root: Path, tile_id: str) -> Path | None:
    """Locate real H&E tile under ``orion_root/he/``."""
    for ext in (".png", ".jpg", ".jpeg", ".tif"):
        p = orion_root / "he" / f"{tile_id}{ext}"
        if p.is_file():
            return p
    return None


def _sort_conditions_by_cosine(
    conditions: list[tuple[str, ...]],
    scores: dict[str, float],
) -> list[tuple[str, ...]]:
    """Sort 15 conditions descending by cosine score.

    Ties broken lexicographically by condition key string.
    Conditions absent from *scores* sort last (treated as -inf).
    """
    def _key(cond: tuple[str, ...]) -> tuple[float, str]:
        k = condition_metric_key(cond)
        return (-scores.get(k, float("-inf")), k)

    return sorted(conditions, key=_key)


def _parse_cosine_json(cache_dir: Path) -> dict[str, float]:
    """Load per-condition cosine scores from ``uni_cosine_scores.json``.

    Returns empty dict if file is missing or contains no valid entries.
    """
    path = cache_dir / "uni_cosine_scores.json"
    if not path.is_file():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    per = raw.get("per_condition")
    if not isinstance(per, dict):
        return {}
    out: dict[str, float] = {}
    for k, v in per.items():
        if v is None:
            continue
        fv = float(v)
        if not np.isnan(fv):
            out[str(k)] = fv
    return out


def _compute_image_cosine(
    img_path: Path,
    ref_npy: Path,
    uni_model: Path,
    device: str,
) -> float:
    """Extract UNI-2h embedding from one image and cosine-sim vs reference npy."""
    from pipeline.extract_features import UNI2hExtractor
    from tools.uni_cosine_similarity import cosine_similarity_uni, flatten_uni_npy

    extractor = UNI2hExtractor(model_path=str(uni_model), device=device)
    ref_emb = flatten_uni_npy(np.load(ref_npy))
    img = Image.open(img_path).convert("RGB")
    gen_emb = np.asarray(extractor.extract(img), dtype=np.float64).ravel()
    return float(cosine_similarity_uni(ref_emb, gen_emb))


def _load_grid_cosine_scores(
    cache_dir: Path,
    all4ch_image: Path,
    orion_root: Path,
    *,
    tile_id: str,
    auto_cosine: bool,
    uni_model: Path,
    device: str,
) -> dict[str, float]:
    """Load cosine scores for all 15 conditions (14 ablation + All-4-ch).

    1. Read existing ``uni_cosine_scores.json``.
    2. If fewer than 14 ablation scores and *auto_cosine*, run UNI extractor via
       ``compute_and_write_uni_cosine_scores`` (writes updated JSON in-place).
    3. If All-4-ch key still missing and *auto_cosine*, compute it from *all4ch_image*.
    """
    scores = _parse_cosine_json(cache_dir)

    ablation_keys = {condition_metric_key(c) for c in ordered_subset_condition_tuples() if len(c) < 4}
    have_ablation = sum(1 for k in ablation_keys if k in scores)

    if auto_cosine and have_ablation < 14:
        try:
            from tools.compute_ablation_uni_cosine import compute_and_write_uni_cosine_scores
            compute_and_write_uni_cosine_scores(
                cache_dir,
                orion_root=orion_root,
                uni_model=uni_model,
                device=device,
            )
            scores = _parse_cosine_json(cache_dir)
        except Exception as exc:
            print(f"Note: UNI cosine computation failed ({exc})", file=sys.stderr)

    if ALL4CH_KEY not in scores and auto_cosine and all4ch_image.is_file():
        ref_npy = orion_root / "features" / f"{tile_id}_uni.npy"
        if ref_npy.is_file():
            try:
                scores[ALL4CH_KEY] = _compute_image_cosine(all4ch_image, ref_npy, uni_model, device)
            except Exception as exc:
                print(f"Note: All-4-ch cosine failed ({exc})", file=sys.stderr)

    return scores


def _draw_dot_row(ax, cond: tuple[str, ...], color: str) -> None:
    """Draw 4 channel indicator dots (filled = active, hollow = inactive)."""
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-0.5, 0.5)
    ax.axis("off")
    for x, g in enumerate(FOUR_GROUP_ORDER):
        if g in cond:
            ax.scatter(x, 0, s=25, c=color, zorder=3, linewidths=0)
        else:
            ax.scatter(x, 0, s=16, facecolors="none",
                       edgecolors=COLOR_INACTIVE, linewidths=0.9, zorder=2)


def _draw_cell_border(ax, color: str, *, dashed: bool = False) -> None:
    """Color and optionally dash all four spines of an image axes."""
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2.5)
        spine.set_edgecolor(color)
        if dashed:
            spine.set_linestyle("--")


def _draw_cosine_bar_cell(ax, score: float | None, color: str) -> None:
    """Horizontal bar proportional to cosine on [-1, 1]; color at 50% alpha."""
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    if score is not None:
        ax.barh(0.5, score - (-1.0), left=-1.0, height=0.6,
                color=color, alpha=0.5, linewidth=0)


def _draw_label_ax(ax, label: str, score_text: str) -> None:
    """Two-line label: condition name on top, score text below."""
    ax.axis("off")
    ax.text(0.5, 0.72, label, ha="center", va="center", fontsize=6,
            transform=ax.transAxes, color="#222222")
    ax.text(0.5, 0.22, score_text, ha="center", va="center", fontsize=6,
            transform=ax.transAxes, color="#555555")


def _build_manifest_lookup(cache_dir: Path, manifest: dict) -> dict[str, dict]:
    """Map ``condition_metric_key`` → manifest entry dict (image_path etc.)."""
    lookup: dict[str, dict] = {}
    for section in manifest["sections"]:
        for entry in section["entries"]:
            key = condition_metric_key(tuple(entry["active_groups"]))
            lookup[key] = entry
    return lookup


def render_ablation_grid_figure(
    cache_dir: Path,
    *,
    all4ch_image: Path,
    orion_root: Path,
    tile_id: str,
    out_png: Path,
    out_pdf: Path | None = None,
    dpi: int = 300,
    auto_cosine: bool = True,
    uni_model: Path | None = None,
    device: str = "cuda",
) -> Path:
    """Render the 4×4 grid figure for one tile; return path to PNG."""
    cache_dir = Path(cache_dir).resolve()
    orion_root = Path(orion_root).resolve()
    all4ch_image = Path(all4ch_image)
    uni_model = Path(uni_model) if uni_model is not None else ROOT / "pretrained_models/uni-2h"

    manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    lookup = _build_manifest_lookup(cache_dir, manifest)

    all15 = ordered_subset_condition_tuples()  # 4+6+4+1 = 15 conditions

    scores = _load_grid_cosine_scores(
        cache_dir, all4ch_image, orion_root,
        tile_id=tile_id, auto_cosine=auto_cosine,
        uni_model=uni_model, device=device,
    )

    sorted_conds = _sort_conditions_by_cosine(all15, scores)
    best_key = condition_metric_key(sorted_conds[0]) if scores else None

    real_he_path = _find_real_he(orion_root, tile_id)
    if real_he_path is None:
        print(f"Warning: Real H&E not found for tile {tile_id!r} — cell [3,3] will be blank.", file=sys.stderr)

    # GridSpec: 16 rows (4 sub-rows per grid row × 4 grid rows) × 4 columns
    NROWS_PER_CELL = 4
    height_ratios = [0.12, 1.0, 0.08, 0.12] * 4  # dot, image, bar, label × 4 rows

    fig = plt.figure(figsize=(9.0, 10.0), facecolor="white")
    gs = gridspec.GridSpec(
        NROWS_PER_CELL * 4, 4,
        figure=fig,
        height_ratios=height_ratios,
        hspace=0.10,
        wspace=0.06,
        left=0.03, right=0.97, top=0.97, bottom=0.02,
    )

    # --- Draw 15 sorted conditions in cells [0,0] → [3,2] ---
    for cell_idx, cond in enumerate(sorted_conds):
        gr, gc = divmod(cell_idx, 4)  # grid row (0-3), grid col (0-3)
        color = _cardinality_color(len(cond))
        k = condition_metric_key(cond)
        score = scores.get(k)
        is_best = (k == best_key)

        base = gr * NROWS_PER_CELL

        # Dot row
        _draw_dot_row(fig.add_subplot(gs[base, gc]), cond, color)

        # H&E image
        if cond == tuple(FOUR_GROUP_ORDER):
            img_path = all4ch_image
        else:
            entry = lookup.get(k)
            if entry is None:
                raise KeyError(f"No manifest entry for {k!r}")
            img_path = cache_dir / entry["image_path"]

        image_ax = fig.add_subplot(gs[base + 1, gc])
        if is_best:
            image_ax.set_facecolor(BEST_BG)
        img_arr = np.asarray(Image.open(img_path).convert("RGB"))
        image_ax.imshow(img_arr)
        image_ax.axis("off")
        _draw_cell_border(image_ax, color)

        # Cosine bar
        _draw_cosine_bar_cell(fig.add_subplot(gs[base + 2, gc]), score, color)

        # Label
        score_text = (
            f"{score:.3f} \u2605" if is_best
            else (f"{score:.3f}" if score is not None else "\u2014")
        )
        _draw_label_ax(fig.add_subplot(gs[base + 3, gc]), _condition_label(cond), score_text)

    # --- Real H&E at cell [3, 3] ---
    gr, gc = 3, 3
    base = gr * NROWS_PER_CELL

    # Dot row: empty spacer
    fig.add_subplot(gs[base, gc]).axis("off")

    # Image
    image_ax = fig.add_subplot(gs[base + 1, gc])
    if real_he_path is not None:
        he_arr = np.asarray(Image.open(real_he_path).convert("RGB"))
        image_ax.imshow(he_arr)
    image_ax.axis("off")
    _draw_cell_border(image_ax, COLOR_REF, dashed=True)

    # Cosine bar: empty
    fig.add_subplot(gs[base + 2, gc]).axis("off")

    # Label
    _draw_label_ax(fig.add_subplot(gs[base + 3, gc]), "Real H\u0026E", "reference")

    # --- Save ---
    plt.savefig(out_png, dpi=dpi, bbox_inches="tight", facecolor="white", pad_inches=0.1)
    if out_pdf is not None:
        plt.savefig(out_pdf, bbox_inches="tight", facecolor="white", pad_inches=0.1)
    plt.close()
    return out_png


def _render_grid_for_cache_dir(cache_dir: Path, args: argparse.Namespace) -> None:
    """Render the grid figure for one tile cache directory."""
    cache_dir = cache_dir.resolve()
    manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    tile_id = str(manifest["tile_id"])

    orion_root = args.orion_root.resolve()

    all4ch_image = args.all4ch_image
    if all4ch_image is None:
        all4ch_image = cache_dir / "all" / f"{tile_id}.png"
    else:
        all4ch_image = Path(all4ch_image).resolve()
    if not all4ch_image.is_file():
        print(
            f"Warning: All-4-ch image not found: {all4ch_image} "
            "— cell will be blank and cosine score unavailable.",
            file=sys.stderr,
        )

    out_png = cache_dir / f"{args.output_name}.png"
    out_pdf = cache_dir / f"{args.output_name}.pdf"

    render_ablation_grid_figure(
        cache_dir,
        all4ch_image=all4ch_image,
        orion_root=orion_root,
        tile_id=tile_id,
        out_png=out_png,
        out_pdf=out_pdf,
        dpi=args.dpi,
        auto_cosine=not args.no_auto_cosine,
        uni_model=args.uni_model,
        device=args.device,
    )
    print(f"Wrote {out_png}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render 4×4 ablation grid figure sorted by cosine similarity.",
    )
    parser.add_argument(
        "--cache-dir", type=Path, required=True,
        help="Single-tile cache dir with manifest.json, or parent of per-tile dirs.",
    )
    parser.add_argument(
        "--orion-root", type=Path, default=ROOT / "data/orion-crc33",
        help="Dataset root (default: data/orion-crc33)",
    )
    parser.add_argument(
        "--all4ch-image", type=Path, default=None,
        help="Path to All-4-ch generated PNG (default: <cache-dir>/all/<tile_id>.png)",
    )
    parser.add_argument(
        "--output-name", type=str, default="ablation_grid",
        help="Basename for output files (default: ablation_grid)",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--no-auto-cosine", action="store_true",
        help="Skip UNI cosine computation; use existing JSON only.",
    )
    parser.add_argument(
        "--uni-model", type=Path, default=ROOT / "pretrained_models/uni-2h",
        help="UNI-2h weights path (default: pretrained_models/uni-2h)",
    )
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    args = parser.parse_args()

    cache_path = args.cache_dir.resolve()
    if is_per_tile_cache_manifest_dir(cache_path):
        _render_grid_for_cache_dir(cache_path, args)
        return

    try:
        cached_ids = list_cached_tile_ids(cache_path)
    except FileNotFoundError as exc:
        parser.error(str(exc))
    if not cached_ids:
        parser.error(
            f"no per-tile caches under {cache_path} "
            "(expected subdirs like <tile_id>/manifest.json)"
        )

    for tile_name in cached_ids:
        _render_grid_for_cache_dir(cache_path / tile_name, args)


if __name__ == "__main__":
    main()
