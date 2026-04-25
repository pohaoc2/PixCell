"""
4×4 ablation grid figure with four per-condition metric bars.

Reads ``manifest.json`` (14 ablation conditions) plus a separately supplied All-4-ch
image. Metrics are loaded from ``metrics.json`` with ``uni_cosine_scores.json`` fallback,
and cosine can still be auto-computed via UNI-2h when requested.

Outputs ``<cache_dir>/ablation_grid.png``.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]

from tools.compute_ablation_metrics import (
    _merge_cosine_into_metrics,
    load_or_build_metrics,
)
from tools.stage3.ablation_cache import (
    is_per_tile_cache_manifest_dir,
    list_cached_tile_ids,
    load_manifest,
    resolve_all_image_path,
)
from tools.stage3.ablation_vis_utils import (
    FOUR_GROUP_ORDER,
    GROUP_SHORT_LABELS,
    cache_manifest_uni_features,
    condition_metric_key,
    default_orion_he_png_path,
    default_orion_uni_npy_path,
    draw_image_border,
    ordered_subset_condition_tuples,
    parse_uni_cosine_scores_json,
)
from tools.stage3.common import print_progress
from tools.stage3.style_mapping import load_style_mapping

# Okabe-Ito palette (colorblind-safe)
_COLOR_BY_CARD: dict[int, str] = {
    1: "#009E73",  # bluish green
    2: "#0072B2",  # blue
    3: "#D55E00",  # vermillion
    4: "#9B59B6",  # purple
}
COLOR_REF = "#999999"
COLOR_INACTIVE = "#CCCCCC"
COLOR_ACTIVE = "#000000"
BEST_BG = "#FFFBE6"
METRIC_BAR_FILL = "#111111"
METRIC_BAR_LABELS: dict[str, str] = {
    "cosine": "Cosine",
    "lpips": "LPIPS",
    "aji": "AJI",
    "pq": "PQ",
    "dice": "DICE",
    "fud": "FUD",
    "style_hed": "HED",
}
METRIC_ORDER: tuple[str, ...] = ("lpips", "pq", "dice", "style_hed")
SORTABLE_METRICS: tuple[str, ...] = ("cosine", "lpips", "aji", "pq", "dice", "fud", "style_hed")
METRIC_BAR_PRESETS: dict[str, tuple[str, ...]] = {
    "paired": ("lpips", "pq", "dice", "style_hed"),
    "unpaired": ("lpips", "pq", "dice", "style_hed"),
    "legacy-paired": ("cosine", "lpips", "aji", "pq"),
    "legacy-unpaired": ("aji", "pq", "style_hed"),
}
LOWER_IS_BETTER_METRICS: frozenset[str] = frozenset({"lpips", "style_hed", "fud"})
METRIC_BAR_MAX_BY_NAME: dict[str, float] = {
    "lpips": 0.50,
    "style_hed": 0.10,
    "fud": 100.0,
}

ALL4CH_KEY: str = condition_metric_key(FOUR_GROUP_ORDER)
_PAPER_GROUP_LABELS: tuple[str, ...] = ("CT", "CS", "Vas", "Nut")


def _cardinality_color(n: int) -> str:
    """Return Okabe-Ito hex for a given channel count (1–4)."""
    return _COLOR_BY_CARD[n]


def _condition_label(cond: tuple[str, ...]) -> str:
    """Short label following FOUR_GROUP_ORDER: e.g. 'CT+CS+Nu'."""
    return "+".join(GROUP_SHORT_LABELS[g] for g in FOUR_GROUP_ORDER if g in cond)


def _sort_conditions_by_metric(
    conditions: list[tuple[str, ...]],
    scores: dict[str, float],
    *,
    metric_name: str = "cosine",
) -> list[tuple[str, ...]]:
    """Sort 15 conditions by one metric, with missing values last.

    Ties broken lexicographically by condition key string.
    """
    ascending = metric_name in {"lpips", "style_hed"}

    def _key(cond: tuple[str, ...]) -> tuple[float, float, str]:
        k = condition_metric_key(cond)
        score = scores.get(k)
        if score is None:
            return (1.0, 0.0, k)
        rank_value = float(score) if ascending else -float(score)
        return (0.0, rank_value, k)

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
    style_mapping: dict[str, str] | None,
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
                style_mapping=style_mapping,
                uni_model=uni_model,
                device=device,
            )
            scores, _ = parse_uni_cosine_scores_json(cache_dir)
        except Exception as exc:
            print(f"Note: UNI cosine computation failed ({exc})", file=sys.stderr)

    if ALL4CH_KEY not in scores and auto_cosine and all4ch_image.is_file():
        ref_npy = default_orion_uni_npy_path(orion_root, tile_id, style_mapping=style_mapping)
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


def _draw_dot_row(
    ax,
    cond: tuple[str, ...],
    *,
    show_group_labels: bool = False,
) -> None:
    """Draw 4 channel-indicator circles in a transparent overlay row."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    ax.set_facecolor("none")
    ax.patch.set_alpha(0.0)

    xs = np.linspace(0.25, 0.75, 4)
    dot_y = 0.48

    for x, g in zip(xs, FOUR_GROUP_ORDER):
        active = g in cond
        face = COLOR_ACTIVE if active else "white"
        edge = "black"
        ax.scatter(
            [x], [dot_y], s=65, marker="o",
            c=[face], edgecolors=[edge], linewidths=0.8,
            zorder=3, clip_on=False,
        )
    if show_group_labels:
        for x, label in zip(xs, _PAPER_GROUP_LABELS):
            ax.text(
                x, dot_y + 0.48, label,
                ha="center", va="bottom",
                fontsize=5.5, color="black",
            )


def _metric_fill_fraction(value: float | None, metric_name: str) -> float | None:
    if value is None:
        return None
    raw_value = float(value)
    max_value = METRIC_BAR_MAX_BY_NAME.get(metric_name)
    if max_value is not None and max_value > 0.0:
        normalized = np.clip(raw_value / max_value, 0.0, 1.0)
        if metric_name in LOWER_IS_BETTER_METRICS:
            normalized = 1.0 - normalized
        return float(np.clip(normalized, 0.0, 1.0))
    if metric_name in LOWER_IS_BETTER_METRICS:
        return float(np.clip(1.0 - raw_value, 0.0, 1.0))
    return float(np.clip(raw_value, 0.0, 1.0))


def _draw_metric_bars_cell(
    ax,
    metrics: dict[str, float | None],
    metric_names: tuple[str, ...] = METRIC_ORDER,
) -> None:
    """Draw one stacked metric bar per requested metric on a simple, metric-aware scale."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, float(max(1, len(metric_names))))
    ax.axis("off")

    label_x = 0.01
    track_x = 0.28
    track_w = 0.54
    bar_h = 0.62

    top_y = float(len(metric_names)) - 0.8
    for idx, metric_name in enumerate(metric_names):
        y = top_y - idx
        value = metrics.get(metric_name)
        frac = _metric_fill_fraction(value, metric_name)

        ax.text(
            label_x,
            y + (bar_h / 2.0),
            METRIC_BAR_LABELS[metric_name],
            ha="left",
            va="center",
            fontsize=6.4,
            color="black",
        )

        track = patches.Rectangle(
            (track_x, y),
            track_w,
            bar_h,
            facecolor="#F2F2F2" if frac is not None else "white",
            edgecolor="#B3B3B3",
            linewidth=0.8,
            linestyle="-" if frac is not None else "--",
        )
        ax.add_patch(track)

        if frac is not None and frac > 0.0:
            ax.add_patch(
                patches.Rectangle(
                    (track_x, y),
                    track_w * frac,
                    bar_h,
                    facecolor=METRIC_BAR_FILL,
                    edgecolor="none",
                )
            )

        value_text = "\u2014" if value is None else f"{float(value):.3f}"
        text_x = track_x + track_w + 0.02
        ax.text(
            text_x,
            y + (bar_h / 2.0),
            value_text,
            ha="left",
            va="center",
            fontsize=6.0,
            color="black",
        )


def _draw_reference_label_ax(ax, label_text: str, *, subtitle: str = "") -> None:
    """Centered label row for the reference H&E cell."""
    ax.axis("off")
    if subtitle:
        ax.text(0.5, 0.68, subtitle, ha="center", va="center", fontsize=8.0,
                transform=ax.transAxes, color="black")
        ax.text(0.5, 0.28, label_text, ha="center", va="center", fontsize=8.0,
                transform=ax.transAxes, color="black")
    else:
        ax.text(0.5, 0.5, label_text, ha="center", va="center", fontsize=8.0,
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


def _coerce_grid_metric_value(value: object) -> float | None:
    if value is None:
        return None
    try:
        fv = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(fv):
        return None
    return fv


def _load_grid_metrics(cache_dir: Path) -> dict[str, dict[str, float | None]]:
    """Load per-condition metrics while preserving legacy ``fid`` as displayable ``fud``."""
    metrics = {
        str(key): dict(value) if isinstance(value, dict) else {}
        for key, value in load_or_build_metrics(cache_dir).items()
    }

    metrics_path = Path(cache_dir) / "metrics.json"
    if not metrics_path.is_file():
        return metrics

    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    per_condition = payload.get("per_condition", {})
    if not isinstance(per_condition, dict):
        return metrics

    for cond_key, raw_record in per_condition.items():
        if not isinstance(raw_record, dict):
            continue
        record = metrics.setdefault(str(cond_key), {})
        for metric_name in ("cosine", "lpips", "aji", "pq", "dice", "iou", "accuracy", "style_hed"):
            if metric_name in raw_record:
                record[metric_name] = _coerce_grid_metric_value(raw_record.get(metric_name))
        fud_value = raw_record.get("fud")
        if fud_value is None:
            fud_value = raw_record.get("fid")
        if fud_value is not None:
            record["fud"] = _coerce_grid_metric_value(fud_value)
    return metrics


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
    ax.contour(cell_mask, levels=[0.5], colors=["black"], linewidths=1.4, alpha=0.9)
    ax.contour(cell_mask, levels=[0.5], colors=["white"], linewidths=0.8, alpha=0.9)


def _load_cellvit_contours(image_path: Path) -> list[np.ndarray]:
    """Load imported CellViT contour polygons from the JSON sidecar when available."""
    sidecar_path = image_path.with_name(f"{image_path.stem}_cellvit_instances.json")
    if not sidecar_path.is_file():
        return []

    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    contours: list[np.ndarray] = []
    for cell in payload.get("cells", []):
        contour = cell.get("contour")
        if not isinstance(contour, list) or len(contour) < 3:
            continue
        arr = np.asarray(contour, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] < 2:
            continue
        contours.append(arr[:, :2])
    return contours


def _maybe_overlay_cellvit_contours(ax, image_path: Path, *, enabled: bool) -> None:
    """Overlay imported CellViT contours in yellow for debugging PQ/AJI mismatches."""
    if not enabled:
        return
    for contour in _load_cellvit_contours(image_path):
        ax.plot(
            contour[:, 0],
            contour[:, 1],
            color="yellow",
            linewidth=0.7,
            alpha=0.95,
            zorder=4,
        )


def _overlay_cellvit_contours_red(ax, image_path: Path) -> None:
    """Overlay CellViT contours in red on every ablation panel."""
    for contour in _load_cellvit_contours(image_path):
        ax.plot(
            contour[:, 0],
            contour[:, 1],
            color="red",
            linewidth=0.6,
            alpha=0.85,
            zorder=4,
        )


def render_ablation_grid_figure(
    cache_dir: Path,
    *,
    all4ch_image: Path,
    orion_root: Path,
    tile_id: str,
    out_png: Path,
    dpi: int = 300,
    auto_cosine: bool = True,
    sort_by: str = "cosine",
    metric_bars: tuple[str, ...] = METRIC_BAR_PRESETS["paired"],
    debug_cellvit_overlay: bool = False,
    uni_model: Path | None = None,
    device: str = "cuda",
    style_mapping: dict[str, str] | None = None,
) -> Path:
    """Render the 4×4 grid figure for one tile; return path to PNG."""
    cache_dir = Path(cache_dir).resolve()
    orion_root = Path(orion_root).resolve()
    all4ch_image = Path(all4ch_image)
    uni_model = Path(uni_model) if uni_model is not None else ROOT / "pretrained_models/uni-2h"

    manifest = load_manifest(cache_dir)
    lookup = _build_manifest_lookup(cache_dir, manifest)
    cell_mask = _load_cell_mask_array(cache_dir, manifest)

    all15 = ordered_subset_condition_tuples()  # 4+6+4+1 = 15 conditions
    metrics = _load_grid_metrics(cache_dir)
    if auto_cosine:
        cosine_scores = _load_grid_cosine_scores(
            cache_dir, all4ch_image, orion_root,
            tile_id=tile_id, style_mapping=style_mapping, auto_cosine=auto_cosine,
            uni_model=uni_model, device=device,
        )
        metrics = _merge_cosine_into_metrics(metrics, cosine_scores)

    sort_scores = {
        key: float(record[sort_by])
        for key, record in metrics.items()
        if isinstance(record, dict) and record.get(sort_by) is not None
    }
    sorted_conds = _sort_conditions_by_metric(all15, sort_scores, metric_name=sort_by)
    best_key = condition_metric_key(sorted_conds[0]) if sort_scores else None

    real_he_path = default_orion_he_png_path(orion_root, tile_id, style_mapping=style_mapping)
    if real_he_path is None:
        print(f"Warning: Real H&E not found for tile {tile_id!r} — cell [3,3] will be blank.", file=sys.stderr)

    # GridSpec: 16 rows (4 sub-rows per grid row × 4 grid rows) × 4 columns
    # Sub-rows: dot-row | image | metric-bars | label
    # layout="constrained" keeps each imshow square and aligns bar width to it.
    NROWS_PER_CELL = 4
    height_ratios = [0.10, 1.0, 0.30, 0.08] * 4

    grid_hspace = 0.012
    grid_wspace = 0.001

    fig = plt.figure(figsize=(7.2, 9.1), facecolor="white", layout="constrained")
    layout_engine = fig.get_layout_engine()
    if layout_engine is not None:
        layout_engine.set(hspace=grid_hspace, wspace=grid_wspace, h_pad=0.006, w_pad=0.001)
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
        k = condition_metric_key(cond)
        metric_record = metrics.get(k, {})
        is_best = (k == best_key)

        base = gr * NROWS_PER_CELL

        # Dot row
        dot_ax = fig.add_subplot(gs[base, gc])
        dot_ax.set_zorder(3)
        dot_ax.patch.set_alpha(0.0)
        _draw_dot_row(dot_ax, cond, show_group_labels=(cell_idx == 0))

        # H&E image
        if cond == tuple(FOUR_GROUP_ORDER):
            img_path = all4ch_image
        else:
            entry = lookup.get(k)
            if entry is None:
                raise KeyError(f"No manifest entry for {k!r}")
            img_path = cache_dir / entry["image_path"]

        image_ax = fig.add_subplot(gs[base + 1, gc])
        image_ax.set_zorder(1)
        image_ax.set_box_aspect(1)
        if is_best:
            image_ax.set_facecolor(BEST_BG)
        img_arr = np.asarray(Image.open(img_path).convert("RGB"))
        image_ax.imshow(img_arr)
        _maybe_contour_cell_mask(image_ax, cell_mask, (img_arr.shape[0], img_arr.shape[1]))
        _maybe_overlay_cellvit_contours(image_ax, img_path, enabled=debug_cellvit_overlay)
        _overlay_cellvit_contours_red(image_ax, img_path)
        image_ax.set_xticks([])
        image_ax.set_yticks([])
        draw_image_border(image_ax, COLOR_ACTIVE)

        # Metric bars
        bar_ax = fig.add_subplot(gs[base + 2, gc])
        _draw_metric_bars_cell(
            bar_ax,
            metric_record if isinstance(metric_record, dict) else {},
            metric_names=metric_bars,
        )

        # Label (primary sort metric only; channel identity is conveyed by the dot row)
        label_ax = fig.add_subplot(gs[base + 3, gc])
        label_ax.axis("off")
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
    for spine in image_ax.spines.values():
        spine.set_visible(False)
    draw_image_border(image_ax, COLOR_ACTIVE, dashed=True)

    # Metric bars: place the reference label in the middle row
    ref_bar_ax = fig.add_subplot(gs[base + 2, gc])
    _draw_reference_label_ax(ref_bar_ax, "reference", subtitle="Real H\u0026E")

    # Label row: empty spacer
    ref_label_ax = fig.add_subplot(gs[base + 3, gc])
    ref_label_ax.axis("off")
    aux_axes_by_image.append((image_ax, [ref_bar_ax, ref_label_ax]))

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
    manifest = load_manifest(cache_dir)
    tile_id = str(manifest["tile_id"])

    orion_root = args.orion_root.resolve()
    style_mapping = load_style_mapping(getattr(args, "style_mapping_json", None))

    all4ch_image = resolve_all_image_path(cache_dir, manifest, n_groups=len(manifest.get("group_names") or FOUR_GROUP_ORDER))
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
        sort_by=args.sort_by,
        metric_bars=METRIC_BAR_PRESETS[args.metric_set],
        debug_cellvit_overlay=args.debug_cellvit_overlay,
        uni_model=args.uni_model,
        device=args.device,
        style_mapping=style_mapping,
    )
    if not getattr(args, "quiet", False):
        print(f"Wrote {out_png}")


def _render_grid_for_cache_dir_job(job: tuple[str, dict]) -> str:
    """Process-pool wrapper for one tile cache directory."""
    cache_dir_str, args_dict = job
    namespace = argparse.Namespace(**args_dict)
    namespace.cache_dir = Path(namespace.cache_dir)
    namespace.orion_root = Path(namespace.orion_root)
    namespace.uni_model = Path(namespace.uni_model)
    namespace.quiet = True
    _render_grid_for_cache_dir(Path(cache_dir_str), namespace)
    return cache_dir_str


def _print_progress(completed: int, total: int, *, prefix: str = "Rendering") -> None:
    print_progress(completed, total, prefix=prefix)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render 4×4 ablation grid figure sorted by one per-condition metric.",
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
        "--style-mapping-json", type=Path, default=None,
        help="Optional layout->style mapping JSON for unpaired reference lookup.",
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
        "--metric-set",
        type=str,
        choices=list(METRIC_BAR_PRESETS),
        default="paired",
        help=(
            "Metric bars shown in each cell: paired/unpaired=LPIPS/PQ/DICE/HED. "
            "Use legacy-paired or legacy-unpaired for the older bar sets."
        ),
    )
    parser.add_argument(
        "--sort-by",
        type=str,
        choices=list(SORTABLE_METRICS),
        default="cosine",
        help="Primary metric used to sort the 15 generated conditions.",
    )
    parser.add_argument(
        "--debug-cellvit-overlay",
        action="store_true",
        help="Overlay imported CellViT contours in yellow on generated H&E panels.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of worker processes when rendering a parent cache directory (default: 1).",
    )
    parser.add_argument(
        "--uni-model", type=Path, default=ROOT / "pretrained_models/uni-2h",
        help="UNI-2h weights path (default: pretrained_models/uni-2h)",
    )
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    args = parser.parse_args()
    args.quiet = False

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

    jobs = max(1, int(args.jobs))
    if jobs == 1:
        _print_progress(0, len(cached_ids))
        for idx, tile_name in enumerate(cached_ids, start=1):
            _render_grid_for_cache_dir(cache_path / tile_name, args)
            _print_progress(idx, len(cached_ids))
        return

    if not args.no_auto_cosine and str(args.device).lower() == "cuda":
        print(
            "Note: falling back to serial rendering because parallel jobs with "
            "CUDA cosine auto-computation would contend for the GPU. Use "
            "--no-auto-cosine to parallelize cached figure rendering.",
            file=sys.stderr,
        )
        _print_progress(0, len(cached_ids))
        for idx, tile_name in enumerate(cached_ids, start=1):
            _render_grid_for_cache_dir(cache_path / tile_name, args)
            _print_progress(idx, len(cached_ids))
        return

    args_dict = vars(args).copy()
    args_dict["cache_dir"] = str(args.cache_dir)
    args_dict["orion_root"] = str(args.orion_root)
    args_dict["uni_model"] = str(args.uni_model)
    worker_count = min(jobs, len(cached_ids), os.cpu_count() or jobs)
    jobs_iter = [
        (str(cache_path / tile_name), args_dict)
        for tile_name in cached_ids
    ]
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(_render_grid_for_cache_dir_job, job) for job in jobs_iter]
        _print_progress(0, len(futures))
        completed = 0
        for future in as_completed(futures):
            future.result()
            completed += 1
            _print_progress(completed, len(futures))


if __name__ == "__main__":
    main()
