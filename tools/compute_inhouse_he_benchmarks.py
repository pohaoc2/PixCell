#!/usr/bin/env python3
"""Compute in-house H&E appearance benchmark bands for Fig. 2.

The benchmark samples H&E tiles from the experiment dataset and compares each
tile with another sampled tile. LPIPS uses the same AlexNet LPIPS scorer as the
ablation metrics; HED uses the same H/E moment-distance definition as
``style_hed``.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.compute_ablation_metrics import _image_to_lpips_tensor, _load_lpips_model
from tools.stage3.hed_utils import masked_mean_std, rgb_to_hed, tissue_mask_from_rgb


def _load_rgb(path: Path, *, size: tuple[int, int] | None = None) -> Image.Image:
    image = Image.open(path).convert("RGB")
    if size is not None and image.size != size:
        image = image.resize(size, Image.BILINEAR)
    return image


def _style_hed_between(reference: Image.Image, comparison: Image.Image) -> float:
    comparison = comparison if comparison.size == reference.size else comparison.resize(reference.size, Image.BILINEAR)
    ref_hed = rgb_to_hed(reference)
    cmp_hed = rgb_to_hed(comparison)
    tissue_mask = tissue_mask_from_rgb(reference) | tissue_mask_from_rgb(comparison)

    score = 0.0
    for stain_channel in (0, 1):
        ref_mean, ref_std = masked_mean_std(ref_hed[..., stain_channel], tissue_mask)
        cmp_mean, cmp_std = masked_mean_std(cmp_hed[..., stain_channel], tissue_mask)
        score += abs(cmp_mean - ref_mean) + abs(cmp_std - ref_std)
    return float(score)


def _sample_pairs(paths: list[Path], *, n: int, seed: int) -> list[tuple[Path, Path]]:
    if len(paths) < n:
        raise ValueError(f"Need at least {n} H&E images, found {len(paths)}")
    rng = random.Random(seed)
    selected = rng.sample(sorted(paths), n)
    comparisons = selected[:]
    rng.shuffle(comparisons)
    for idx, (left, right) in enumerate(zip(selected, comparisons, strict=True)):
        if left == right:
            swap_idx = (idx + 1) % len(comparisons)
            comparisons[idx], comparisons[swap_idx] = comparisons[swap_idx], comparisons[idx]
    return list(zip(selected, comparisons, strict=True))


def compute_benchmarks(
    *,
    he_dir: Path,
    n: int,
    seed: int,
    device: str,
    batch_size: int,
) -> dict[str, object]:
    paths = list(Path(he_dir).glob("*.png"))
    pairs = _sample_pairs(paths, n=n, seed=seed)

    loss_fn, resolved_device = _load_lpips_model(device)
    batch_size = max(1, int(batch_size))
    lpips_scores: list[float] = []
    hed_scores: list[float] = []

    import torch

    with torch.no_grad():
        for start in range(0, len(pairs), batch_size):
            chunk = pairs[start:start + batch_size]
            left_tensors = []
            right_tensors = []
            for left_path, right_path in chunk:
                left_img = _load_rgb(left_path)
                right_img = _load_rgb(right_path, size=left_img.size)
                hed_scores.append(_style_hed_between(left_img, right_img))
                left_tensors.append(_image_to_lpips_tensor(left_img, device=resolved_device))
                right_tensors.append(_image_to_lpips_tensor(right_img, device=resolved_device))

            left_batch = torch.cat(left_tensors, dim=0)
            right_batch = torch.cat(right_tensors, dim=0)
            scores = loss_fn(left_batch, right_batch).reshape(-1)
            lpips_scores.extend(float(score) for score in scores.tolist())

    lpips_arr = np.asarray(lpips_scores, dtype=np.float64)
    hed_arr = np.asarray(hed_scores, dtype=np.float64)
    return {
        "he_dir": str(Path(he_dir)),
        "n": int(n),
        "seed": int(seed),
        "pairing": "1000 sampled H&E tiles, each compared with one shuffled sampled H&E tile",
        "lpips": {
            "mean": float(lpips_arr.mean()),
            "std": float(lpips_arr.std()),
        },
        "style_hed": {
            "mean": float(hed_arr.mean()),
            "std": float(hed_arr.std()),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--he-dir", type=Path, default=ROOT / "data" / "orion-crc33" / "he")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260608)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "figures" / "pngs_updated" / "concat" / "inhouse_he_benchmarks.json",
    )
    args = parser.parse_args()

    payload = compute_benchmarks(
        he_dir=args.he_dir,
        n=args.n,
        seed=args.seed,
        device=args.device,
        batch_size=args.batch_size,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
