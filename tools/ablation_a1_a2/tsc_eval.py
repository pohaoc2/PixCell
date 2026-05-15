"""TME Spatial Concordance: nuclei-map Dice against CODEX cell masks."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.color import rgb2hed
from skimage.filters import threshold_otsu

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.ablation_a1_a2.cache_io import load_cache, merge_tsc, save_cache

load_channel = None
resolve_channel_dir = None
resolve_data_layout = None


def _ensure_tile_pipeline_imports() -> None:
    global load_channel, resolve_channel_dir, resolve_data_layout
    if load_channel is not None and resolve_channel_dir is not None and resolve_data_layout is not None:
        return

    from tools.stage3.tile_pipeline import (
        load_channel as load_channel_impl,
        resolve_channel_dir as resolve_channel_dir_impl,
        resolve_data_layout as resolve_data_layout_impl,
    )

    load_channel = load_channel_impl
    resolve_channel_dir = resolve_channel_dir_impl
    resolve_data_layout = resolve_data_layout_impl


def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """Dice coefficient for two binary masks, with both-empty = 1.0."""
    pred_mask = np.asarray(pred, dtype=bool)
    gt_mask = np.asarray(gt, dtype=bool)
    if not pred_mask.any() and not gt_mask.any():
        return 1.0
    intersection = float((pred_mask & gt_mask).sum())
    denom = float(pred_mask.sum()) + float(gt_mask.sum())
    return 0.0 if denom == 0.0 else (2.0 * intersection) / denom


def extract_nuclei_map(he_rgb: np.ndarray) -> np.ndarray:
    """Extract a boolean nuclei map from a uint8 H&E RGB tile."""
    rgb = np.asarray(he_rgb, dtype=np.uint8)
    hed = rgb2hed(rgb.astype(np.float32) / 255.0)
    hematoxylin = hed[:, :, 0]
    if float(np.ptp(hematoxylin)) < 1e-6:
        return np.mean(rgb, axis=2) < 220.0
    threshold = threshold_otsu(hematoxylin)
    return hematoxylin > threshold


def compute_tsc_tile(he_rgb: np.ndarray, codex_cell_mask: np.ndarray) -> float:
    """Dice between extracted nuclei and the CODEX cell mask."""
    return dice_score(extract_nuclei_map(he_rgb), np.asarray(codex_cell_mask, dtype=bool))


def _load_codex_cell_mask(tile_id: str, exp_channels_dir: Path, image_size: int) -> np.ndarray:
    """Load CODEX cell_masks as a boolean array."""
    _ensure_tile_pipeline_imports()
    channel_dir = resolve_channel_dir(exp_channels_dir, "cell_masks")
    channel = load_channel(channel_dir, tile_id, image_size, binary=True, channel_name="cell_masks")
    return np.asarray(channel, dtype=np.float32) > 0.5


def run_tsc(
    *,
    cache_dir: Path,
    tile_ids: list[str],
    variants: list[str],
    exp_channels_dir: Path,
    image_size: int = 256,
) -> dict[str, float]:
    """Compute mean TSC for each variant and merge it into cache.json."""
    cache_path = cache_dir / "cache.json"
    cache = load_cache(cache_path)
    scores_by_variant: dict[str, float] = {}

    for variant in variants:
        tile_dir = cache_dir / "tiles" / variant
        scores: list[float] = []
        for tile_id in tile_ids:
            tile_path = tile_dir / f"{tile_id}.png"
            if not tile_path.is_file():
                print(f"  [tsc] tile missing: {tile_path} - skip")
                continue
            he_rgb = np.asarray(Image.open(tile_path).convert("RGB"))
            codex_mask = _load_codex_cell_mask(tile_id, exp_channels_dir, image_size)
            scores.append(compute_tsc_tile(he_rgb, codex_mask))
        if scores:
            score_arr = np.asarray(scores, dtype=np.float32)
            mean_score = float(np.mean(score_arr))
            std_score = float(np.std(score_arr))
            merge_tsc(cache, variant, mean_score, std_score)
            scores_by_variant[variant] = mean_score

    save_cache(cache, cache_path)
    return scores_by_variant


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default="inference_output/si_a1_a2")
    parser.add_argument("--tile-ids-file", required=True)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["production", "a1_concat", "a1_per_channel", "a2_bypass"],
    )
    parser.add_argument("--image-size", type=int, default=256)
    args = parser.parse_args(argv)

    cache_dir = Path(args.cache_dir)
    tile_ids = [
        line.strip()
        for line in Path(args.tile_ids_file).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    _ensure_tile_pipeline_imports()
    exp_channels_dir, _, _ = resolve_data_layout(Path("data/orion-crc33"))
    run_tsc(
        cache_dir=cache_dir,
        tile_ids=tile_ids,
        variants=list(args.variants),
        exp_channels_dir=exp_channels_dir,
        image_size=args.image_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())