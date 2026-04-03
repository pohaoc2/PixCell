#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import warnings
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image
from scipy.linalg import sqrtm

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.stage3.ablation_cache import list_cached_tile_ids
from tools.stage3.ablation_vis_utils import FOUR_GROUP_ORDER, condition_metric_key

_RESAMPLE_BILINEAR = getattr(Image, "Resampling", Image).BILINEAR
_FID_EPS = 1e-6
_METRIC_NAME = "fid"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute dataset-level FID for every ablation condition and backfill the "
            "scores into per-tile metrics.json files."
        )
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("inference_output/cache"),
        help="Parent dir with per-tile cache subdirectories.",
    )
    parser.add_argument(
        "--orion-root",
        type=Path,
        default=Path("data/orion-crc33"),
        help="Paired dataset root with real H&E PNGs under he/<tile_id>.png.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device to use for Inception v3 inference.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Inception v3 batch size.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSON output path. Defaults to <cache-dir>/fid_scores.json.",
    )
    return parser.parse_args()


def ordered_condition_keys() -> list[str]:
    return [
        condition_metric_key(tuple(cond))
        for size in range(1, len(FOUR_GROUP_ORDER) + 1)
        for cond in combinations(FOUR_GROUP_ORDER, size)
    ]


def resolve_device(requested: str) -> str:
    import torch

    if requested.startswith("cuda") and not torch.cuda.is_available():
        warnings.warn("CUDA requested but unavailable; falling back to CPU.", stacklevel=2)
        return "cpu"
    return requested


def load_manifest(manifest_path: Path) -> dict:
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def collect_condition_paths(
    cache_dir: Path,
    orion_root: Path,
) -> tuple[list[str], list[Path], dict[str, list[Path]]]:
    condition_keys = ordered_condition_keys()
    condition_to_paths: dict[str, list[Path]] = {key: [] for key in condition_keys}
    tile_ids = list_cached_tile_ids(cache_dir)
    if not tile_ids:
        raise FileNotFoundError(f"no cached tile manifests found under {cache_dir}")

    real_paths: list[Path] = []
    all_key = condition_metric_key(FOUR_GROUP_ORDER)

    for tile_id in tile_ids:
        tile_dir = cache_dir / tile_id
        manifest = load_manifest(tile_dir / "manifest.json")

        manifest_tile_id = str(manifest.get("tile_id", "")).strip()
        if manifest_tile_id and manifest_tile_id != tile_id:
            raise ValueError(
                f"tile_id mismatch for {tile_dir}: manifest has {manifest_tile_id!r}, "
                f"directory name is {tile_id!r}"
            )

        per_tile_paths: dict[str, Path] = {}
        for section in manifest.get("sections", []):
            for entry in section.get("entries", []):
                cond_key = condition_metric_key(tuple(entry["active_groups"]))
                image_path = tile_dir / entry["image_path"]
                if not image_path.is_file():
                    raise FileNotFoundError(f"missing generated image: {image_path}")
                per_tile_paths[cond_key] = image_path

        all_path = tile_dir / "all" / "generated_he.png"
        if all_path.is_file():
            per_tile_paths[all_key] = all_path

        missing = [key for key in condition_keys if key not in per_tile_paths]
        if missing:
            raise ValueError(
                f"tile {tile_id} is missing condition images for: {', '.join(missing)}"
            )

        real_path = orion_root / "he" / f"{tile_id}.png"
        if not real_path.is_file():
            raise FileNotFoundError(f"missing real H&E image: {real_path}")

        real_paths.append(real_path)
        for cond_key in condition_keys:
            condition_to_paths[cond_key].append(per_tile_paths[cond_key])

    return tile_ids, real_paths, condition_to_paths


def build_preprocess() -> Any:
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def load_inception_model(device: str) -> Any:
    import torch
    from torchvision.models import Inception_V3_Weights, inception_v3

    weights = Inception_V3_Weights.IMAGENET1K_V1
    model = inception_v3(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval()
    model.to(device)
    return model


def iter_batches(items: list[Path], batch_size: int) -> Iterable[list[Path]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def load_image_tensor(image_path: Path, preprocess: Any) -> Any:
    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
        if rgb.size != (299, 299):
            rgb = rgb.resize((299, 299), _RESAMPLE_BILINEAR)
        return preprocess(rgb)


def extract_features(
    image_paths: list[Path],
    *,
    model: Any,
    device: str,
    batch_size: int,
    preprocess: Any,
) -> np.ndarray:
    if not image_paths:
        raise ValueError("cannot extract features from an empty image set")

    import torch

    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for batch_paths in iter_batches(image_paths, batch_size):
            batch = torch.stack(
                [load_image_tensor(path, preprocess) for path in batch_paths],
                dim=0,
            ).to(device)
            features = model(batch)
            if isinstance(features, tuple):
                features = features[0]
            outputs.append(features.detach().cpu().numpy().astype(np.float64, copy=False))

    return np.concatenate(outputs, axis=0)


def compute_statistics(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if features.ndim != 2:
        raise ValueError(f"expected 2D feature matrix, got shape {features.shape}")
    mu = np.mean(features, axis=0, dtype=np.float64)
    if features.shape[0] < 2:
        sigma = np.zeros((features.shape[1], features.shape[1]), dtype=np.float64)
    else:
        sigma = np.cov(features, rowvar=False).astype(np.float64, copy=False)
    return mu, sigma


def _stable_sqrtm(product: np.ndarray) -> np.ndarray:
    covmean = sqrtm(product)
    if not np.isfinite(covmean).all():
        offset = np.eye(product.shape[0], dtype=np.float64) * _FID_EPS
        covmean = sqrtm(product + offset)

    if np.iscomplexobj(covmean):
        imag_max = float(np.max(np.abs(covmean.imag)))
        if imag_max >= 1e-3:
            warnings.warn(
                f"large imaginary component from sqrtm ({imag_max:.3e}); taking real part",
                stacklevel=2,
            )
        covmean = covmean.real

    return covmean.astype(np.float64, copy=False)


def compute_fid_from_stats(
    real_mu: np.ndarray,
    real_sigma: np.ndarray,
    gen_mu: np.ndarray,
    gen_sigma: np.ndarray,
) -> float:
    real_sigma = np.atleast_2d(real_sigma)
    gen_sigma = np.atleast_2d(gen_sigma)
    diff = real_mu - gen_mu

    offset = np.eye(real_sigma.shape[0], dtype=np.float64) * _FID_EPS
    covmean = _stable_sqrtm((real_sigma + offset) @ (gen_sigma + offset))
    trace_term = np.trace(real_sigma) + np.trace(gen_sigma) - 2.0 * np.trace(covmean)
    fid = float(np.dot(diff, diff) + trace_term)
    return max(fid, 0.0)


def write_fid_scores(output_path: Path, fid_scores: dict[str, float]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = {key: float(fid_scores[key]) for key in ordered_condition_keys()}
    output_path.write_text(json.dumps(ordered, indent=2) + "\n", encoding="utf-8")


def backfill_metrics(cache_dir: Path, tile_ids: list[str], fid_scores: dict[str, float]) -> None:
    for tile_id in tile_ids:
        metrics_path = cache_dir / tile_id / "metrics.json"
        if not metrics_path.is_file():
            raise FileNotFoundError(f"missing metrics.json: {metrics_path}")

        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        per_condition = payload.get("per_condition")
        if not isinstance(per_condition, dict):
            raise ValueError(f"invalid per_condition payload in {metrics_path}")

        for cond_key, record in per_condition.items():
            if cond_key not in fid_scores:
                continue
            if not isinstance(record, dict):
                record = {}
                per_condition[cond_key] = record
            record[_METRIC_NAME] = float(fid_scores[cond_key])

        metrics_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    cache_dir = args.cache_dir.resolve()
    orion_root = args.orion_root.resolve()
    output_path = args.output.resolve() if args.output is not None else cache_dir / "fid_scores.json"
    device = resolve_device(args.device)

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    tile_ids, real_paths, condition_to_paths = collect_condition_paths(cache_dir, orion_root)
    preprocess = build_preprocess()
    model = load_inception_model(device)

    print(f"Extracting real H&E features once for {len(real_paths)} images on {device}.")
    real_features = extract_features(
        real_paths,
        model=model,
        device=device,
        batch_size=args.batch_size,
        preprocess=preprocess,
    )
    real_mu, real_sigma = compute_statistics(real_features)

    fid_scores: dict[str, float] = {}
    condition_keys = ordered_condition_keys()
    for index, cond_key in enumerate(condition_keys, start=1):
        gen_paths = condition_to_paths[cond_key]
        print(f"[{index}/{len(condition_keys)}] Processing {cond_key}: {len(gen_paths)} images")
        gen_features = extract_features(
            gen_paths,
            model=model,
            device=device,
            batch_size=args.batch_size,
            preprocess=preprocess,
        )
        gen_mu, gen_sigma = compute_statistics(gen_features)
        fid_scores[cond_key] = compute_fid_from_stats(real_mu, real_sigma, gen_mu, gen_sigma)

    write_fid_scores(output_path, fid_scores)
    backfill_metrics(cache_dir, tile_ids, fid_scores)
    print(f"Wrote FID scores to {output_path}")
    print(f"Backfilled fid into {len(tile_ids)} metrics.json files")


if __name__ == "__main__":
    main()
