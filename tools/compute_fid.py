#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image
from scipy.linalg import sqrtm

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.stage3.ablation_cache import list_cached_tile_ids, load_manifest
from tools.stage3.ablation_vis_utils import (
    FOUR_GROUP_ORDER,
    condition_metric_key,
    default_orion_uni_npy_path,
)

_RESAMPLE_BILINEAR = getattr(Image, "Resampling", Image).BILINEAR
_FID_EPS = 1e-6
_FEATURE_BACKENDS = ("uni", "inception")


@dataclass(frozen=True)
class ImageFeatureRecord:
    image_path: Path
    feature_path: Path | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute dataset-level Fréchet distance for every ablation condition and "
            "backfill the scores into per-tile metrics.json files."
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
        help="Torch device to use for feature extraction.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Feature extraction batch size.",
    )
    parser.add_argument(
        "--feature-backend",
        choices=_FEATURE_BACKENDS,
        default="uni",
        help=(
            "Feature space used for Fréchet distance. "
            "'uni' (default) uses histopathology UNI-2h embeddings; "
            "'inception' uses ImageNet Inception v3 features."
        ),
    )
    parser.add_argument(
        "--uni-model",
        type=Path,
        default=ROOT / "pretrained_models/uni-2h",
        help="UNI-2h model directory used when --feature-backend=uni.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSON output path. Defaults to <cache-dir>/fud_scores.json for UNI or <cache-dir>/fid_scores.json for Inception.",
    )
    return parser.parse_args()


def ordered_condition_keys() -> list[str]:
    return [
        condition_metric_key(tuple(cond))
        for size in range(1, len(FOUR_GROUP_ORDER) + 1)
        for cond in combinations(FOUR_GROUP_ORDER, size)
    ]


def _generated_uni_feature_cache_path(tile_dir: Path, image_rel: Path) -> Path:
    return tile_dir / "features" / image_rel.parent / f"{image_rel.stem}_uni.npy"


def metric_key_for_backend(feature_backend: str) -> str:
    return "fud" if feature_backend == "uni" else "fid"


def metric_label_for_backend(feature_backend: str) -> str:
    return metric_key_for_backend(feature_backend).upper()


def default_output_path(cache_dir: Path, feature_backend: str) -> Path:
    return cache_dir / f"{metric_key_for_backend(feature_backend)}_scores.json"


def resolve_device(requested: str) -> str:
    import torch

    if requested.startswith("cuda") and not torch.cuda.is_available():
        warnings.warn("CUDA requested but unavailable; falling back to CPU.", stacklevel=2)
        return "cpu"
    return requested


def collect_condition_paths(
    cache_dir: Path,
    orion_root: Path,
    *,
    feature_backend: str,
) -> tuple[list[str], list[ImageFeatureRecord], dict[str, list[ImageFeatureRecord]]]:
    condition_keys = ordered_condition_keys()
    condition_to_paths: dict[str, list[ImageFeatureRecord]] = {key: [] for key in condition_keys}
    tile_ids = list_cached_tile_ids(cache_dir)
    if not tile_ids:
        raise FileNotFoundError(f"no cached tile manifests found under {cache_dir}")

    real_paths: list[ImageFeatureRecord] = []
    all_key = condition_metric_key(FOUR_GROUP_ORDER)

    for tile_id in tile_ids:
        tile_dir = cache_dir / tile_id
        manifest = load_manifest(tile_dir)

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
                feature_path = None
                if feature_backend == "uni":
                    feature_path = _generated_uni_feature_cache_path(
                        tile_dir,
                        Path(entry["image_path"]),
                    )
                per_tile_paths[cond_key] = ImageFeatureRecord(
                    image_path=image_path,
                    feature_path=feature_path,
                )

        all_path = tile_dir / "all" / "generated_he.png"
        if all_path.is_file():
            per_tile_paths[all_key] = ImageFeatureRecord(
                image_path=all_path,
                feature_path=(
                    _generated_uni_feature_cache_path(tile_dir, Path("all/generated_he.png"))
                    if feature_backend == "uni"
                    else None
                ),
            )

        missing = [key for key in condition_keys if key not in per_tile_paths]
        if missing:
            raise ValueError(
                f"tile {tile_id} is missing condition images for: {', '.join(missing)}"
            )

        real_path = orion_root / "he" / f"{tile_id}.png"
        if not real_path.is_file():
            raise FileNotFoundError(f"missing real H&E image: {real_path}")

        real_feature_path = None
        if feature_backend == "uni":
            candidate = default_orion_uni_npy_path(orion_root, tile_id)
            if candidate.is_file():
                real_feature_path = candidate
        real_paths.append(ImageFeatureRecord(image_path=real_path, feature_path=real_feature_path))
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


def iter_batches(items: list[Any], batch_size: int) -> Iterable[list[Any]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def load_image_tensor(image_path: Path, preprocess: Any) -> Any:
    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
        if rgb.size != (299, 299):
            rgb = rgb.resize((299, 299), _RESAMPLE_BILINEAR)
        return preprocess(rgb)


def extract_inception_features(
    image_records: list[ImageFeatureRecord],
    *,
    model: Any,
    device: str,
    batch_size: int,
    preprocess: Any,
) -> np.ndarray:
    if not image_records:
        raise ValueError("cannot extract features from an empty image set")

    import torch

    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for batch_records in iter_batches(image_records, batch_size):
            batch = torch.stack(
                [load_image_tensor(record.image_path, preprocess) for record in batch_records],
                dim=0,
            ).to(device)
            features = model(batch)
            if isinstance(features, tuple):
                features = features[0]
            outputs.append(features.detach().cpu().numpy().astype(np.float64, copy=False))

    return np.concatenate(outputs, axis=0)


def _load_rgb_pil(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def load_uni_extractor(*, uni_model: Path, device: str):
    from tools.stage3.compute_ablation_uni_cosine import load_uni_extractor as _load_uni_extractor

    return _load_uni_extractor(uni_model=uni_model, device=device)


def all_features_cached(image_records: list[ImageFeatureRecord]) -> bool:
    return bool(image_records) and all(
        record.feature_path is not None and record.feature_path.is_file()
        for record in image_records
    )


def extract_uni_features(
    image_records: list[ImageFeatureRecord],
    *,
    extractor: Any | None,
    batch_size: int,
) -> np.ndarray:
    if not image_records:
        raise ValueError("cannot extract features from an empty image set")

    outputs: list[np.ndarray] = []
    pending_records: list[ImageFeatureRecord] = []

    def flush_pending() -> None:
        if not pending_records:
            return
        if extractor is None:
            raise ValueError("UNI extractor is required when cached UNI features are missing")
        images = [_load_rgb_pil(record.image_path) for record in pending_records]
        features = np.asarray(extractor.extract_batch(images), dtype=np.float64)
        if features.ndim == 1:
            features = features[np.newaxis, :]
        if len(features) != len(pending_records):
            raise ValueError(
                "UNI extractor returned a different number of features than input images"
            )
        for record, feature in zip(pending_records, features):
            if record.feature_path is not None:
                record.feature_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(record.feature_path, feature.astype(np.float32, copy=False))
            outputs.append(feature.astype(np.float64, copy=False))
        pending_records.clear()

    for record in image_records:
        if record.feature_path is not None and record.feature_path.is_file():
            outputs.append(np.load(record.feature_path).astype(np.float64).ravel())
            continue
        pending_records.append(record)
        if len(pending_records) >= batch_size:
            flush_pending()

    flush_pending()
    return np.stack(outputs, axis=0)


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


def write_metric_scores(output_path: Path, metric_scores: dict[str, float]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = {key: float(metric_scores[key]) for key in ordered_condition_keys()}
    output_path.write_text(json.dumps(ordered, indent=2) + "\n", encoding="utf-8")


def backfill_metrics(
    cache_dir: Path,
    tile_ids: list[str],
    metric_scores: dict[str, float],
    *,
    metric_key: str,
) -> None:
    for tile_id in tile_ids:
        metrics_path = cache_dir / tile_id / "metrics.json"
        if not metrics_path.is_file():
            raise FileNotFoundError(f"missing metrics.json: {metrics_path}")

        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        per_condition = payload.get("per_condition")
        if not isinstance(per_condition, dict):
            raise ValueError(f"invalid per_condition payload in {metrics_path}")

        for cond_key, record in per_condition.items():
            if cond_key not in metric_scores:
                continue
            if not isinstance(record, dict):
                record = {}
                per_condition[cond_key] = record
            record[metric_key] = float(metric_scores[cond_key])
            if metric_key == "fud":
                record.pop("fid", None)
            elif metric_key == "fid":
                record.pop("fud", None)

        metrics_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    cache_dir = args.cache_dir.resolve()
    orion_root = args.orion_root.resolve()
    device = resolve_device(args.device)
    feature_backend = str(args.feature_backend)
    uni_model = args.uni_model.resolve()
    metric_key = metric_key_for_backend(feature_backend)
    metric_label = metric_label_for_backend(feature_backend)
    output_path = args.output.resolve() if args.output is not None else default_output_path(cache_dir, feature_backend)

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    tile_ids, real_paths, condition_to_paths = collect_condition_paths(
        cache_dir,
        orion_root,
        feature_backend=feature_backend,
    )

    print(
        f"Extracting real H&E features once for {len(real_paths)} images on {device} "
        f"using {feature_backend}."
    )
    if feature_backend == "uni":
        extractor: Any | None = None
        all_uni_records = list(real_paths)
        for cond_key in ordered_condition_keys():
            all_uni_records.extend(condition_to_paths[cond_key])
        if not all_features_cached(all_uni_records):
            extractor = load_uni_extractor(uni_model=uni_model, device=device)
        real_features = extract_uni_features(
            real_paths,
            extractor=extractor,
            batch_size=args.batch_size,
        )
    else:
        preprocess = build_preprocess()
        model = load_inception_model(device)
        real_features = extract_inception_features(
            real_paths,
            model=model,
            device=device,
            batch_size=args.batch_size,
            preprocess=preprocess,
        )
    real_mu, real_sigma = compute_statistics(real_features)

    metric_scores: dict[str, float] = {}
    condition_keys = ordered_condition_keys()
    for index, cond_key in enumerate(condition_keys, start=1):
        gen_records = condition_to_paths[cond_key]
        print(f"[{index}/{len(condition_keys)}] Processing {cond_key}: {len(gen_records)} images")
        if feature_backend == "uni":
            gen_features = extract_uni_features(
                gen_records,
                extractor=extractor,
                batch_size=args.batch_size,
            )
        else:
            gen_features = extract_inception_features(
                gen_records,
                model=model,
                device=device,
                batch_size=args.batch_size,
                preprocess=preprocess,
            )
        gen_mu, gen_sigma = compute_statistics(gen_features)
        metric_scores[cond_key] = compute_fid_from_stats(real_mu, real_sigma, gen_mu, gen_sigma)

    write_metric_scores(output_path, metric_scores)
    backfill_metrics(cache_dir, tile_ids, metric_scores, metric_key=metric_key)
    print(f"Wrote {metric_label} scores to {output_path}")
    print(f"Backfilled {metric_key} into {len(tile_ids)} metrics.json files")


if __name__ == "__main__":
    main()
