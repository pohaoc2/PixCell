"""
4×4 ablation grid figure: 16 conditions (14 ablation + All-4-ch + Real H&E) sorted by cosine similarity.

Reads ``manifest.json`` (14 ablation conditions) plus a separately supplied All-4-ch image.
Cosine scores are loaded from ``uni_cosine_scores.json`` or auto-computed via UNI-2h.

Outputs ``<cache_dir>/ablation_grid.png``.
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

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.stage3.ablation_cache import is_per_tile_cache_manifest_dir, list_cached_tile_ids
from tools.stage3.ablation_vis_utils import (
    FOUR_GROUP_ORDER,
    cache_manifest_uni_features,
    condition_metric_key,
    default_orion_he_png_path,
    ordered_subset_condition_tuples,
    parse_uni_cosine_scores_json,
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
    "vasculature": "Vas",
    "microenv": "Env",
}

ALL4CH_KEY: str = condition_metric_key(FOUR_GROUP_ORDER)


def _cardinality_color(n: int) -> str:
    """Return Okabe-Ito hex for a given channel count (1–4)."""
    return _COLOR_BY_CARD[n]


def _condition_label(cond: tuple[str, ...]) -> str:
    """Short label following FOUR_GROUP_ORDER: e.g. 'CT+CS+Nu'."""
    return "+".join(_GROUP_SHORT[g] for g in FOUR_GROUP_ORDER if g in cond)


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


def _compute_image_cosine(
    img_path: Path,
    ref_npy: Path,
    uni_model: Path,
    device: str,
    *,
    feat_cache_dir: Path | None = None,
) -> float:
    """Extract UNI-2h embedding from one image and cosine-sim vs reference npy.

    If *feat_cache_dir* is given, the embedding is loaded from
    ``<feat_cache_dir>/<img_path.stem>_uni.npy`` when present; otherwise it is
    computed and saved there for future runs.
    """
    from tools.stage3.uni_cosine_similarity import cosine_similarity_uni, flatten_uni_npy

    ref_emb = flatten_uni_npy(np.load(ref_npy))

    feat_path: Path | None = None
    if feat_cache_dir is not None:
        feat_path = Path(feat_cache_dir) / f"{img_path.stem}_uni.npy"

    if feat_path is not None and feat_path.is_file():
        gen_emb = np.load(feat_path).astype(np.float64).ravel()
    else:
        from pipeline.extract_features import UNI2hExtractor
        extractor = UNI2hExtractor(model_path=str(uni_model), device=device)
        img = Image.open(img_path).convert("RGB")
        gen_emb = np.asarray(extractor.extract(img), dtype=np.float64).ravel()
        if feat_path is not None:
            feat_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(feat_path, gen_emb)

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
    scores, _ = parse_uni_cosine_scores_json(cache_dir)

    ablation_keys = {condition_metric_key(c) for c in ordered_subset_condition_tuples() if len(c) < 4}
    have_ablation = sum(1 for k in ablation_keys if k in scores)

    if auto_cosine and have_ablation < 14:
        try:
            from tools.stage3.compute_ablation_uni_cosine import compute_and_write_uni_cosine_scores
            compute_and_write_uni_cosine_scores(
                cache_dir,
                orion_root=orion_root,
                uni_model=uni_model,
                device=device,
            )
            scores, _ = parse_uni_cosine_scores_json(cache_dir)
        except Exception as exc:
            print(f"Note: UNI cosine computation failed ({exc})", file=sys.stderr)

    if ALL4CH_KEY not in scores and auto_cosine and all4ch_image.is_file():
        ref_npy = orion_root / "features" / f"{tile_id}_uni.npy"
        if ref_npy.is_file():
            try:
                all4_rel_parent = all4ch_image.resolve().relative_to(cache_dir.resolve()).parent
            except ValueError:
                all4_rel_parent = Path("all")
            try:
                scores[ALL4CH_KEY] = _compute_image_cosine(
                    all4ch_image, ref_npy, uni_model, device,
                    feat_cache_dir=cache_dir / "features" / all4_rel_parent,
                )
            except Exception as exc:
                print(f"Note: All-4-ch cosine failed ({exc})", file=sys.stderr)

    if auto_cosine:
        try:
            cache_manifest_uni_features(
                cache_dir,
                uni_model=uni_model,
                device=device,
                force=False,
            )
        except Exception as exc:
            if str(device).lower() == "cuda":
                try:
                    cache_manifest_uni_features(
                        cache_dir,
                        uni_model=uni_model,
                        device="cpu",
                        force=False,
                    )
                except Exception as cpu_exc:
                    print(
                        "Note: manifest feature caching failed "
                        f"(cuda: {exc}; cpu: {cpu_exc})",
                        file=sys.stderr,
                    )
            else:
                print(f"Note: manifest feature caching failed ({exc})", file=sys.stderr)

    return scores


def _resolve_all4ch_image(cache_dir: Path, manifest: dict) -> Path | None:
    """Resolve All-channels image path from manifest first, then legacy fallbacks."""
    cache_dir = Path(cache_dir)
    n_groups = len(manifest.get("group_names") or FOUR_GROUP_ORDER)

    for section in manifest.get("sections", []):
        try:
            subset_size = int(section.get("subset_size", 0))
        except (TypeError, ValueError):
            continue
        if subset_size != n_groups:
            continue
        entries = section.get("entries") or []
        if not entries:
            continue
        rel = Path(entries[0].get("image_path", ""))
        if rel and (cache_dir / rel).is_file():
            return cache_dir / rel

    canonical = cache_dir / "all" / "generated_he.png"
    if canonical.is_file():
        return canonical

    all_dir = cache_dir / "all"
    if all_dir.is_dir():
        pngs = sorted(all_dir.glob("*.png"))
        if len(pngs) == 1:
            return pngs[0]
    return None


def _draw_dot_row(
    ax,
    cond: tuple[str, ...],
    color: str,
    *,
    show_labels: bool,
) -> None:
    """Draw 4 channel-indicator dots in a centered horizontal row."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    xs = np.linspace(0.18, 0.82, 4)
    dot_y = 0.32 if show_labels else 0.50
    label_y = 0.78

    for x, g in zip(xs, FOUR_GROUP_ORDER):
        short = _GROUP_SHORT[g]
        active = g in cond
        face = color if active else "white"
        edge = "black"
        ax.scatter(
            [x], [dot_y], s=100,
            c=[face], edgecolors=[edge], linewidths=0.8,
            zorder=3,
        )
        if show_labels:
            ax.text(
                x, label_y, short,
                ha="center", va="center", fontsize=7.5,
                color="black",
            )


def _draw_cell_border(ax, color: str, *, dashed: bool = False) -> None:
    """Color and optionally dash all four spines of an image axes."""
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2.5)
        spine.set_color(color)
        if dashed:
            spine.set_linestyle("--")


def _draw_cosine_bar_cell(ax, score: float | None) -> None:
    """Centered cosine bar: 0 in the middle, ±1 at the edges, monochrome.

    A light-grey track spans the full [-1, 1] range.  A thin vertical tick
    marks 0.  The filled portion runs from 0 to *score* (left for negative,
    right for positive) in dark grey/black.
    """
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    # Background track
    ax.barh(0.5, 2.0, left=-1.0, height=0.50, color="#E6E6E6", linewidth=0)
    # Centre tick
    ax.axvline(0.0, ymin=0.15, ymax=0.85, color="#AAAAAA", linewidth=0.7, zorder=2)
    if score is not None:
        ax.barh(
            0.5, abs(score), left=min(0.0, score), height=0.50,
            color="#111111", linewidth=0, zorder=3,
        )


def _draw_score_label_ax(
    ax,
    score: float | None,
    *,
    is_best: bool = False,
    show_scale_extrema: bool = False,
) -> None:
    """Annotate the cosine row with endpoint score and optional -1/+1 labels."""
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    if show_scale_extrema:
        ax.text(-1.0, 0.18, "-1", ha="left", va="center", fontsize=8.0, color="black")
        ax.text(1.0, 0.18, "+1", ha="right", va="center", fontsize=8.0, color="black")
    if score is None:
        ax.text(0.0, 0.72, "\u2014", ha="center", va="center", fontsize=9.0, color="black")
        return

    score_text = f"{score:.3f}" + (" \u2605" if is_best else "")
    score_x = float(np.clip(score, -0.95, 0.95))
    if score >= 0.0:
        score_x -= 0.02
        ha = "right"
    else:
        score_x += 0.02
        ha = "left"
    ax.text(score_x, 0.72, score_text, ha=ha, va="center", fontsize=8.5, color="black")


def _draw_reference_label_ax(ax, label_text: str, *, subtitle: str = "") -> None:
    """Centered label row for the reference H&E cell."""
    ax.axis("off")
    if subtitle:
        ax.text(0.5, 0.72, subtitle, ha="center", va="center", fontsize=7.5,
                transform=ax.transAxes, color="black")
        ax.text(0.5, 0.22, label_text, ha="center", va="center", fontsize=8.5,
                transform=ax.transAxes, color="black")
    else:
        ax.text(0.5, 0.5, label_text, ha="center", va="center", fontsize=8.5,
                transform=ax.transAxes, color="black")


def _match_ax_width_to_image(ax, image_ax) -> None:
    """Match an auxiliary axis width/x-position to the rendered square image."""
    image_pos = image_ax.get_position()
    ax_pos = ax.get_position()
    ax.set_position([image_pos.x0, ax_pos.y0, image_pos.width, ax_pos.height])


def _build_manifest_lookup(cache_dir: Path, manifest: dict) -> dict[str, dict]:
    """Map ``condition_metric_key`` → manifest entry dict (image_path etc.)."""
    lookup: dict[str, dict] = {}
    for section in manifest["sections"]:
        for entry in section["entries"]:
            key = condition_metric_key(tuple(entry["active_groups"]))
            lookup[key] = entry
    return lookup


def _load_cell_mask_array(cache_dir: Path, manifest: dict) -> np.ndarray | None:
    """Load cached reference cell mask for contour overlay when available."""
    rel = manifest.get("cell_mask_path")
    if not rel:
        return None
    path = cache_dir / rel
    if not path.is_file():
        return None
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def _maybe_contour_cell_mask(
    ax,
    cell_mask: np.ndarray | None,
    image_hw: tuple[int, int],
) -> None:
    """Overlay the reference cell-mask contour on an H&E axes."""
    if cell_mask is None:
        return
    img_h, img_w = image_hw
    mask_h, mask_w = cell_mask.shape[:2]
    if (mask_h, mask_w) != (img_h, img_w):
        resized = Image.fromarray(
            (np.clip(cell_mask, 0, 1) * 255).astype(np.uint8),
            mode="L",
        ).resize((img_w, img_h), Image.BILINEAR)
        cell_mask = np.asarray(resized, dtype=np.float32) / 255.0
    ax.contour(cell_mask, levels=[0.5], colors=["lime"], linewidths=0.7, alpha=0.85)


def render_ablation_grid_figure(
    cache_dir: Path,
    *,
    all4ch_image: Path,
    orion_root: Path,
    tile_id: str,
    out_png: Path,
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
    cell_mask = _load_cell_mask_array(cache_dir, manifest)

    all15 = ordered_subset_condition_tuples()  # 4+6+4+1 = 15 conditions

    scores = _load_grid_cosine_scores(
        cache_dir, all4ch_image, orion_root,
        tile_id=tile_id, auto_cosine=auto_cosine,
        uni_model=uni_model, device=device,
    )

    sorted_conds = _sort_conditions_by_cosine(all15, scores)
    best_key = condition_metric_key(sorted_conds[0]) if scores else None

    real_he_path = default_orion_he_png_path(orion_root, tile_id)
    if real_he_path is None:
        print(f"Warning: Real H&E not found for tile {tile_id!r} — cell [3,3] will be blank.", file=sys.stderr)

    # GridSpec: 16 rows (4 sub-rows per grid row × 4 grid rows) × 4 columns
    # Sub-rows: dot-row | image | cosine-bar | label
    # layout="constrained" keeps each imshow square and aligns bar width to it.
    NROWS_PER_CELL = 4
    height_ratios = [0.20, 1.0, 0.09, 0.13] * 4

    grid_hspace = 0.04
    grid_wspace = 0.03

    fig = plt.figure(figsize=(9.0, 10.0), facecolor="white", layout="constrained")
    layout_engine = fig.get_layout_engine()
    if layout_engine is not None:
        layout_engine.set(hspace=grid_hspace, wspace=grid_wspace, h_pad=0.01, w_pad=0.01)
    gs = gridspec.GridSpec(
        NROWS_PER_CELL * 4, 4,
        figure=fig,
        height_ratios=height_ratios,
        hspace=grid_hspace,
        wspace=grid_wspace,
    )

    aux_axes_by_image: list[tuple[plt.Axes, list[plt.Axes]]] = []

    # --- Draw 15 sorted conditions in cells [0,0] → [3,2] ---
    for cell_idx, cond in enumerate(sorted_conds):
        gr, gc = divmod(cell_idx, 4)  # grid row (0-3), grid col (0-3)
        color = _cardinality_color(len(cond))
        k = condition_metric_key(cond)
        score = scores.get(k)
        is_best = (k == best_key)

        base = gr * NROWS_PER_CELL

        # Dot row
        dot_ax = fig.add_subplot(gs[base, gc])
        _draw_dot_row(dot_ax, cond, color, show_labels=(cell_idx == 0))

        # H&E image
        if cond == tuple(FOUR_GROUP_ORDER):
            img_path = all4ch_image
        else:
            entry = lookup.get(k)
            if entry is None:
                raise KeyError(f"No manifest entry for {k!r}")
            img_path = cache_dir / entry["image_path"]

        image_ax = fig.add_subplot(gs[base + 1, gc])
        image_ax.set_box_aspect(1)
        if is_best:
            image_ax.set_facecolor(BEST_BG)
        img_arr = np.asarray(Image.open(img_path).convert("RGB"))
        image_ax.imshow(img_arr)
        _maybe_contour_cell_mask(image_ax, cell_mask, (img_arr.shape[0], img_arr.shape[1]))
        image_ax.set_xticks([])
        image_ax.set_yticks([])
        _draw_cell_border(image_ax, color)

        # Cosine bar
        bar_ax = fig.add_subplot(gs[base + 2, gc])
        _draw_cosine_bar_cell(bar_ax, score)

        # Label (score only; channel identity is conveyed by the dot row)
        label_ax = fig.add_subplot(gs[base + 3, gc])
        _draw_score_label_ax(
            label_ax,
            score,
            is_best=is_best,
            show_scale_extrema=(cell_idx == 0),
        )
        aux_axes_by_image.append((image_ax, [dot_ax, bar_ax, label_ax]))

    # --- Real H&E at cell [3, 3] ---
    gr, gc = 3, 3
    base = gr * NROWS_PER_CELL

    # Dot row: empty spacer
    fig.add_subplot(gs[base, gc]).axis("off")

    # Image
    image_ax = fig.add_subplot(gs[base + 1, gc])
    image_ax.set_box_aspect(1)
    if real_he_path is not None:
        he_arr = np.asarray(Image.open(real_he_path).convert("RGB"))
        image_ax.imshow(he_arr)
        _maybe_contour_cell_mask(image_ax, cell_mask, (he_arr.shape[0], he_arr.shape[1]))
    image_ax.set_xticks([])
    image_ax.set_yticks([])
    _draw_cell_border(image_ax, COLOR_REF, dashed=True)

    # Cosine bar: empty spacer
    fig.add_subplot(gs[base + 2, gc]).axis("off")

    # Label
    ref_label_ax = fig.add_subplot(gs[base + 3, gc])
    _draw_reference_label_ax(ref_label_ax, "reference", subtitle="Real H\u0026E")
    aux_axes_by_image.append((image_ax, [ref_label_ax]))

    # --- Save ---
    fig.canvas.draw()
    fig.set_layout_engine("none")
    for image_ax, aux_axes in aux_axes_by_image:
        for aux_ax in aux_axes:
            _match_ax_width_to_image(aux_ax, image_ax)
    plt.savefig(out_png, dpi=dpi, facecolor="white")
    plt.close()
    return out_png


def _render_grid_for_cache_dir(cache_dir: Path, args: argparse.Namespace) -> None:
    """Render the grid figure for one tile cache directory."""
    cache_dir = cache_dir.resolve()
    manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    tile_id = str(manifest["tile_id"])

    orion_root = args.orion_root.resolve()

    all4ch_image = _resolve_all4ch_image(cache_dir, manifest)
    if all4ch_image is None:
        raise FileNotFoundError(
            f"All-4-ch image not found under: {cache_dir}\n"
            "Expected an all/ image (prefer all/generated_he.png) or a subset_size==4 section in manifest.json."
        )

    out_png = cache_dir / f"{args.output_name}.png"
    render_ablation_grid_figure(
        cache_dir,
        all4ch_image=all4ch_image,
        orion_root=orion_root,
        tile_id=tile_id,
        out_png=out_png,
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
