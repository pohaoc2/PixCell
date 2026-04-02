"""
Publication-style ablation figure: 14 subset conditions, channel header, UNI cosine bars.

Reads ``manifest.json`` and optional ``uni_cosine_scores.json`` next to ``--cache-dir``.

Channel header thumbnails are built from ``exp_channels/`` (see ``build_exp_channel_header_rgb``):
cell types = union of ``cell_type_*``, cell states = union of ``cell_state_*``, vasculature,
microenv = oxygen + glucose with colormaps from ``tools/color_constants``.

Layout matches ``docs/ablation_vis_spec.md``: group label | four channel dots |
generated image | UNI cosine. Optional dashed horizontal rules separate cardinality groups
in the channel + gen region. Lime mask contour on generated panels when ``cell_mask.png``
is present (same as ``stage3_ablation_full_vis``).

Outputs PNG beside the cache directory (same folder as the manifest).
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
from matplotlib.transforms import blended_transform_factory
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.stage3_ablation_cache import is_per_tile_cache_manifest_dir, list_cached_tile_ids
from tools.stage3_ablation_vis_utils import (
    FOUR_GROUP_ORDER,
    build_exp_channel_header_rgb,
    compute_rgb_pixel_cosine_scores,
    condition_metric_key,
    default_orion_uni_npy_path,
    ordered_subset_condition_tuples,
)

# Okabe–Ito (spec)
COLOR_1CH = "#009E73"
COLOR_2CH = "#0072B2"
COLOR_3CH = "#D55E00"
COLOR_INACTIVE = "#CCCCCC"
BAR_ALPHA = 0.35
BEST_ROW_BG = "#FFFBE6"

# Spec header: image + two-line labels (types / states / vasc / nutr)
_HEADER_LABELS = {
    "cell_types": "Cell\ntypes",
    "cell_state": "Cell\nstate",
    "vasculature": "Vasc",
    "microenv": "Nutr",
}


def _group_color(n_active: int) -> str:
    if n_active == 1:
        return COLOR_1CH
    if n_active == 2:
        return COLOR_2CH
    return COLOR_3CH


def _build_manifest_lookup(cache_dir: Path) -> dict[str, dict]:
    manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    lookup: dict[str, dict] = {}
    for section in manifest["sections"]:
        for entry in section["entries"]:
            key = condition_metric_key(tuple(entry["active_groups"]))
            lookup[key] = entry
    return lookup


def _parse_cosine_json(cache_dir: Path) -> tuple[dict[str, float], str] | tuple[None, str]:
    """Return ``(per_condition, column_title)`` or ``(None, default title)`` if missing/empty."""
    path = cache_dir / "uni_cosine_scores.json"
    if not path.is_file():
        return None, "UNI cosine"
    raw = json.loads(path.read_text(encoding="utf-8"))
    per = raw.get("per_condition")
    if not isinstance(per, dict) or not per:
        return None, "UNI cosine"
    out: dict[str, float] = {}
    for k, v in per.items():
        if v is None:
            continue
        fv = float(v)
        if np.isnan(fv):
            continue
        out[str(k)] = fv
    metric = raw.get("metric", "uni_cosine")
    if metric == "rgb_pixel_cosine":
        title = "RGB cosine"
    else:
        title = "UNI cosine"
    return (out if out else None), title


def _write_rgb_pixel_cosine_json(
    cache_dir: Path,
    orion_root: Path,
    *,
    tile_id: str,
) -> None:
    per, ref_meta = compute_rgb_pixel_cosine_scores(cache_dir, orion_root)
    payload = {
        "version": 1,
        "metric": "rgb_pixel_cosine",
        "tile_id": tile_id,
        "orion_root": str(Path(orion_root).resolve()),
        "reference": ref_meta,
        "per_condition": per,
    }
    (cache_dir / "uni_cosine_scores.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


def _load_or_compute_cosine_scores(
    cache_dir: Path,
    orion_root: Path,
    *,
    tile_id: str,
    auto_cosine: bool,
    uni_model: Path,
    device: str,
    reference_uni: Path | None,
    reference_he: Path | None,
) -> tuple[dict[str, float], str]:
    """
    Load ``uni_cosine_scores.json`` if present; otherwise optionally compute UNI cosine,
    then fall back to numpy RGB-pixel cosine vs ``he/{tile_id}.png`` (no torch).
    """
    scores, title = _parse_cosine_json(cache_dir)
    if scores and len(scores) >= 14:
        return scores, title

    if not auto_cosine:
        return scores or {}, title

    if not scores or len(scores) < 14:
        try:
            from tools.compute_ablation_uni_cosine import compute_and_write_uni_cosine_scores

            compute_and_write_uni_cosine_scores(
                cache_dir,
                orion_root=orion_root,
                reference_uni=reference_uni,
                reference_he=reference_he,
                uni_model=uni_model,
                device=device,
            )
            scores, title = _parse_cosine_json(cache_dir)
            if scores and len(scores) >= 14:
                print(f"UNI cosine scores loaded ({len(scores)} conditions).", file=sys.stderr)
                return scores, title
        except Exception as exc:
            print(f"Note: UNI-2h cosine unavailable ({exc}); trying RGB pixel cosine fallback.", file=sys.stderr)

    if not scores or len(scores) < 14:
        try:
            _write_rgb_pixel_cosine_json(cache_dir, orion_root, tile_id=tile_id)
            scores, title = _parse_cosine_json(cache_dir)
            if scores:
                print(
                    f"Wrote RGB pixel cosine fallback → {cache_dir / 'uni_cosine_scores.json'} "
                    f"({len(scores)} conditions). Use torch + compute_ablation_uni_cosine for UNI-2h.",
                    file=sys.stderr,
                )
                return scores or {}, title
        except Exception as exc:
            print(f"Warning: RGB pixel cosine fallback failed ({exc}).", file=sys.stderr)

    return scores or {}, title


def _load_cell_mask_array(cache_dir: Path, manifest: dict) -> np.ndarray | None:
    rel = manifest.get("cell_mask_path")
    if not rel:
        return None
    path = cache_dir / rel
    if not path.is_file():
        return None
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def _draw_axes_bottom_rule(ax, *, color: str = "#bbbbbb", lw: float = 0.65, ls: str = "--") -> None:
    """Dashed horizontal line along the bottom edge of an axes (figure coordinates)."""
    ax.plot(
        [0, 1],
        [0, 0],
        transform=ax.transAxes,
        color=color,
        linestyle=ls,
        linewidth=lw,
        clip_on=False,
        zorder=100,
    )


def _maybe_contour_cell_mask(
    ax,
    cell_mask: np.ndarray | None,
    gen_hw: tuple[int, int],
) -> None:
    """Green contour on generated H&E, matching ``stage3_ablation_full_vis``."""
    if cell_mask is None:
        return
    gh, gw = gen_hw
    mh, mw = cell_mask.shape[:2]
    if (mh, mw) != (gh, gw):
        mm = Image.fromarray((np.clip(cell_mask, 0, 1) * 255).astype(np.uint8), mode="L").resize(
            (gw, gh), Image.BILINEAR
        )
        cell_mask = np.asarray(mm, dtype=np.float32) / 255.0
    ax.contour(cell_mask, levels=[0.5], colors=["lime"], linewidths=0.7, alpha=0.85)


def render_ablation_pub_figure(
    cache_dir: Path,
    *,
    exp_channels_dir: Path,
    tile_id: str,
    orion_root: Path,
    out_png: Path,
    dpi: int = 300,
    header_thumbnail_res: int = 384,
    auto_cosine: bool = True,
    uni_model: Path | None = None,
    device: str = "cuda",
    reference_uni: Path | None = None,
    reference_he: Path | None = None,
    cardinality: int | None = None,
) -> Path:
    """Render the figure for one cardinality group; returns path to PNG.

    ``cardinality`` filters conditions to 1-, 2-, or 3-channel subsets.
    ``None`` renders all 14 conditions in a single figure (legacy mode).
    """
    cache_dir = Path(cache_dir)
    orion_root = Path(orion_root).resolve()
    uni_model = Path(uni_model) if uni_model is not None else ROOT / "pretrained_models/uni-2h"

    manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    cell_mask_full = _load_cell_mask_array(cache_dir, manifest)

    all_ordered = ordered_subset_condition_tuples()
    if cardinality is not None:
        ordered = [c for c in all_ordered if len(c) == cardinality]
    else:
        ordered = all_ordered

    lookup = _build_manifest_lookup(cache_dir)
    score_map, cosine_column_title = _load_or_compute_cosine_scores(
        cache_dir,
        orion_root,
        tile_id=tile_id,
        auto_cosine=auto_cosine,
        uni_model=uni_model,
        device=device,
        reference_uni=reference_uni,
        reference_he=reference_he,
    )

    header_rgb = build_exp_channel_header_rgb(
        exp_channels_dir,
        tile_id,
        resolution=header_thumbnail_res,
    )

    n_data = len(ordered)
    # Columns: 4× channel dots (narrow) | generated image (dominant) | cosine bar
    ncols = 6
    fig_w = 9.5
    fig_h = 3.0 + n_data * 1.5  # ~1.4 in per data row after margins
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
    gs = gridspec.GridSpec(
        1 + n_data + 1,
        ncols,
        figure=fig,
        height_ratios=[2.0] + [1.0] * n_data + [0.03],
        width_ratios=[0.38, 0.38, 0.38, 0.38, 3.5, 0.90],
        wspace=0.08,
        hspace=0.06,
        left=0.02,
        right=0.98,
        top=0.95,
        bottom=0.02,
    )

    _card_labels = {1: "1-channel", 2: "2-channel", 3: "3-channel"}
    title = "Channel conditioning ablation"
    if cardinality is not None:
        title += f" \u2014 {_card_labels.get(cardinality, f'{cardinality}-ch')}"
    fig.suptitle(title, fontsize=9, fontweight="bold", y=0.99)

    # --- Header row: 4 channel panels (image + label) | blank | cosine header ---
    for col_idx, g in enumerate(FOUR_GROUP_ORDER):
        ax_h = fig.add_subplot(gs[0, col_idx])
        ax_h.imshow(header_rgb[g])
        ax_h.axis("off")
        ax_h.set_title(
            _HEADER_LABELS.get(g, g),
            fontsize=7,
            pad=4,
            color="#222222",
        )

    fig.add_subplot(gs[0, 4]).axis("off")

    ax_uni_h = fig.add_subplot(gs[0, 5])
    ax_uni_h.axis("off")
    ax_uni_h.text(
        0.5,
        0.62,
        cosine_column_title,
        ha="center",
        va="bottom",
        fontsize=7,
        fontweight="bold",
        transform=ax_uni_h.transAxes,
    )
    ax_uni_h.text(
        0.5,
        0.18,
        "\u2192 better",
        ha="center",
        va="top",
        fontsize=6,
        color="#444444",
        transform=ax_uni_h.transAxes,
    )

    # Cosine bar axes spanning all data rows
    ax_bar = fig.add_subplot(gs[1:-1, 5])
    ax_bar.set_zorder(5)
    ax_bar.set_xlim(-1.0, 1.38)
    ax_bar.set_ylim(-0.5, n_data - 0.5)
    ax_bar.set_yticks(range(n_data))
    ax_bar.set_yticklabels([])
    ax_bar.set_xlabel("cosine", fontsize=6)
    ax_bar.tick_params(axis="x", labelsize=6)
    for spine in ("top", "right"):
        ax_bar.spines[spine].set_visible(False)

    best_idx: int | None = None
    best_val = -2.0
    if score_map:
        for i, cond in enumerate(ordered):
            k = condition_metric_key(cond)
            v = score_map.get(k)
            if v is not None and v > best_val:
                best_val = v
                best_idx = i

    x_text = blended_transform_factory(ax_bar.transData, ax_bar.transData)

    for i, cond in enumerate(ordered):
        row_gs = 1 + i
        n = len(cond)
        c = _group_color(n)

        # Dot indicators aligned with the four channel header columns
        ax_dots = fig.add_subplot(gs[row_gs, 0:4])
        ax_dots.set_xlim(-0.5, 3.5)
        ax_dots.set_ylim(-0.5, 0.5)
        ax_dots.axis("off")
        for x, g in enumerate(FOUR_GROUP_ORDER):
            if g in cond:
                ax_dots.scatter(x, 0, s=55, c=c, zorder=3)
            else:
                ax_dots.scatter(
                    x,
                    0,
                    s=28,
                    facecolors="none",
                    edgecolors=COLOR_INACTIVE,
                    linewidths=1.0,
                    zorder=2,
                )

        key = condition_metric_key(cond)
        entry = lookup.get(key)
        if entry is None:
            raise KeyError(f"No manifest entry for condition key {key!r}")
        gen_path = cache_dir / entry["image_path"]

        ax_im = fig.add_subplot(gs[row_gs, 4])
        if best_idx == i:
            ax_im.set_facecolor(BEST_ROW_BG)
        gen_im = np.asarray(Image.open(gen_path).convert("RGB"))
        ax_im.imshow(gen_im)
        _maybe_contour_cell_mask(ax_im, cell_mask_full, (gen_im.shape[0], gen_im.shape[1]))
        ax_im.axis("off")

        cos_val = score_map.get(key)
        y_bar = n_data - 1 - i
        if cos_val is not None:
            ax_bar.barh(
                y_bar,
                cos_val - (-1.0),
                left=-1.0,
                height=0.65,
                color=c,
                alpha=BAR_ALPHA,
                linewidth=0,
                zorder=4,
            )
            mark = " \u2605" if i == best_idx else ""
            ax_bar.text(
                1.12,
                y_bar,
                f"{cos_val:.3f}{mark}",
                va="center",
                ha="left",
                fontsize=7,
                clip_on=False,
                transform=x_text,
                zorder=6,
            )
        else:
            ax_bar.text(
                1.12,
                y_bar,
                "\u2014",
                va="center",
                ha="left",
                fontsize=7,
                clip_on=False,
                transform=x_text,
                zorder=6,
            )

    plt.savefig(out_png, dpi=dpi, bbox_inches="tight", facecolor="white", pad_inches=0.08)
    plt.close()
    return out_png


def _render_pub_for_cache_dir(cache_dir: Path, args: argparse.Namespace) -> None:
    """Render three PNG figures (1-ch, 2-ch, 3-ch) for one tile cache directory."""
    cache_dir = cache_dir.resolve()
    manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    tile_id = str(manifest["tile_id"])

    orion_root = args.orion_root.resolve()
    exp_channels_dir = args.exp_channels_dir
    if exp_channels_dir is None:
        exp_channels_dir = orion_root / "exp_channels"
    else:
        exp_channels_dir = Path(exp_channels_dir).resolve()

    if not exp_channels_dir.is_dir():
        print(
            f"Warning: exp_channels not found: {exp_channels_dir} — header panels may be placeholders.",
            file=sys.stderr,
        )

    for card in (1, 2, 3):
        out_png = cache_dir / f"{args.output_name}_{card}ch.png"
        render_ablation_pub_figure(
            cache_dir,
            exp_channels_dir=exp_channels_dir,
            tile_id=tile_id,
            orion_root=orion_root,
            out_png=out_png,
            dpi=args.dpi,
            header_thumbnail_res=args.header_res,
            auto_cosine=not args.no_auto_cosine,
            uni_model=args.uni_model,
            device=args.device,
            reference_uni=args.reference_uni,
            reference_he=args.reference_he,
            cardinality=card,
        )
        print(f"Wrote {out_png}")

    print(f"(Reference UNI cache: {default_orion_uni_npy_path(orion_root, tile_id)})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render ablation publication figure from cache. "
        "Pass a single-tile cache dir or a parent dir of per-tile caches.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        required=True,
        help="Directory with manifest.json, or parent of per-tile subdirs (each with manifest.json)",
    )
    parser.add_argument(
        "--orion-root",
        type=Path,
        default=ROOT / "data/orion-crc33",
        help="Dataset root (default: data/orion-crc33)",
    )
    parser.add_argument(
        "--exp-channels-dir",
        type=Path,
        default=None,
        help="exp_channels directory (default: {orion-root}/exp_channels)",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--output-name",
        type=str,
        default="ablation_pub_figure",
        help="Basename for PNG output (default: ablation_pub_figure)",
    )
    parser.add_argument(
        "--header-res",
        type=int,
        default=384,
        help="Resolution for header channel thumbnails (default: 384)",
    )
    parser.add_argument(
        "--no-auto-cosine",
        action="store_true",
        help="Do not run UNI cosine if uni_cosine_scores.json is missing",
    )
    parser.add_argument(
        "--uni-model",
        type=Path,
        default=ROOT / "pretrained_models/uni-2h",
        help="UNI-2h weights (for --auto-cosine)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="cuda or cpu (for --auto-cosine)",
    )
    parser.add_argument("--reference-uni", type=Path, default=None, help="Override reference UNI .npy")
    parser.add_argument("--reference-he", type=Path, default=None, help="Reference H&E if no .npy")
    args = parser.parse_args()

    cache_path = args.cache_dir.resolve()
    if is_per_tile_cache_manifest_dir(cache_path):
        _render_pub_for_cache_dir(cache_path, args)
        return

    try:
        cached_ids = list_cached_tile_ids(cache_path)
    except FileNotFoundError as exc:
        parser.error(str(exc))
    if not cached_ids:
        parser.error(
            f"no per-tile caches with manifest.json under {cache_path} "
            f"(expected subdirs like {cache_path}/<tile_id>/manifest.json)"
        )

    for tile_name in cached_ids:
        _render_pub_for_cache_dir(cache_path / tile_name, args)


if __name__ == "__main__":
    main()
