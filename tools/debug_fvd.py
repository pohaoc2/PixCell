#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.compute_fid import (
    ImageFeatureRecord,
    all_features_cached,
    collect_condition_paths,
    compute_fid_from_stats,
    compute_statistics,
    condition_metric_key,
    extract_virchow2_features,
    load_virchow2_extractor,
)
from tools.stage3.ablation_vis_utils import FOUR_GROUP_ORDER


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a few sanity checks around Virchow-2 FVD computation.",
    )
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--orion-root", type=Path, required=True)
    parser.add_argument(
        "--condition",
        type=str,
        default="all",
        help=(
            "Condition key to inspect. Use 'all' for the 4-group output or pass a raw "
            "condition key such as 'cell_types__cell_state'."
        ),
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--virchow2-model",
        type=str,
        default="hf-hub:paige-ai/Virchow2",
        help="Virchow-2 timm model identifier.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed used for deterministic split-vs-split checks.",
    )
    parser.add_argument(
        "--inspect-shapes",
        action="store_true",
        help="Run one direct forward pass and print raw token output shape plus final embedding shape.",
    )
    return parser.parse_args()


def resolve_condition_key(raw_condition: str, available_keys: list[str]) -> str:
    if raw_condition == "all":
        resolved = condition_metric_key(FOUR_GROUP_ORDER)
    else:
        resolved = str(raw_condition)
    if resolved not in available_keys:
        raise KeyError(
            f"condition {raw_condition!r} resolved to {resolved!r}, "
            f"which is not present in the cache"
        )
    return resolved


def split_records(
    records: list[ImageFeatureRecord],
    *,
    seed: int,
) -> tuple[list[ImageFeatureRecord], list[ImageFeatureRecord]]:
    if len(records) < 2:
        raise ValueError("need at least two records to create a split")
    indices = np.arange(len(records))
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    midpoint = len(indices) // 2
    if midpoint == 0 or midpoint == len(indices):
        raise ValueError("split would be empty; provide at least two records")
    left = [records[int(idx)] for idx in indices[:midpoint]]
    right = [records[int(idx)] for idx in indices[midpoint:]]
    return left, right


def summarize_records(label: str, records: list[ImageFeatureRecord]) -> None:
    cached = sum(
        1
        for record in records
        if record.feature_path is not None and record.feature_path.is_file()
    )
    print(
        f"{label}: {len(records)} images "
        f"({cached}/{len(records)} cached Virchow-2 feature files present)"
    )


def extract_features(
    records: list[ImageFeatureRecord],
    *,
    extractor,
    batch_size: int,
) -> np.ndarray:
    features = extract_virchow2_features(
        records,
        extractor=extractor,
        batch_size=batch_size,
    )
    if features.ndim != 2:
        raise ValueError(f"expected 2D feature matrix, got shape {features.shape}")
    return features


def compute_fvd_between(
    features_a: np.ndarray,
    features_b: np.ndarray,
) -> float:
    mu_a, sigma_a = compute_statistics(features_a)
    mu_b, sigma_b = compute_statistics(features_b)
    return compute_fid_from_stats(mu_a, sigma_a, mu_b, sigma_b)


def inspect_shapes(sample_record: ImageFeatureRecord, *, extractor) -> None:
    if extractor is None:
        raise ValueError("--inspect-shapes requires loading the Virchow-2 model")

    import torch
    from PIL import Image

    with Image.open(sample_record.image_path) as image:
        rgb = image.convert("RGB")
    batch = torch.stack([extractor.transform(rgb)]).to(extractor.device)
    with torch.inference_mode(), extractor._autocast_context():
        output = extractor.model(batch)

    class_token = output[:, 0]
    patch_tokens = output[:, 5:]
    embedding = torch.cat([class_token, patch_tokens.mean(dim=1)], dim=-1)

    print(f"Virchow raw output shape: {tuple(output.shape)}")
    print(f"Virchow class token shape: {tuple(class_token.shape)}")
    print(f"Virchow patch token shape: {tuple(patch_tokens.shape)}")
    print(f"Final embedding shape: {tuple(embedding.shape)}")


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    cache_dir = args.cache_dir.resolve()
    orion_root = args.orion_root.resolve()
    _, real_records, condition_to_records = collect_condition_paths(
        cache_dir,
        orion_root,
        feature_backend="virchow2",
    )

    available_keys = list(condition_to_records.keys())
    condition_key = resolve_condition_key(args.condition, available_keys)
    gen_records = condition_to_records[condition_key]

    print(f"Condition: {condition_key}")
    summarize_records("Real set", real_records)
    summarize_records("Generated set", gen_records)

    need_extractor = args.inspect_shapes or not all_features_cached(real_records + gen_records)
    extractor = None
    if need_extractor:
        print(f"Loading Virchow-2 on {args.device} ...")
        extractor = load_virchow2_extractor(
            virchow2_model=str(args.virchow2_model),
            device=str(args.device),
        )

    if args.inspect_shapes:
        inspect_shapes(real_records[0], extractor=extractor)

    real_features = extract_features(
        real_records,
        extractor=extractor,
        batch_size=args.batch_size,
    )
    gen_features = extract_features(
        gen_records,
        extractor=extractor,
        batch_size=args.batch_size,
    )

    print(f"Real feature matrix shape: {tuple(real_features.shape)}")
    print(f"Generated feature matrix shape: {tuple(gen_features.shape)}")
    print(f"Single feature vector shape: {tuple(real_features[0].shape)}")

    fvd_real_vs_same = compute_fvd_between(real_features, real_features)
    fvd_real_vs_gen = compute_fvd_between(real_features, gen_features)

    print(f"FVD(real, same real set): {fvd_real_vs_same:.6f}")
    print(f"FVD(real, generated):     {fvd_real_vs_gen:.6f}")

    if len(real_records) >= 4:
        real_left, real_right = split_records(real_records, seed=args.seed)
        real_left_features = extract_features(
            real_left,
            extractor=extractor,
            batch_size=args.batch_size,
        )
        real_right_features = extract_features(
            real_right,
            extractor=extractor,
            batch_size=args.batch_size,
        )
        print(
            "FVD(real split A, real split B): "
            f"{compute_fvd_between(real_left_features, real_right_features):.6f}"
        )
    else:
        print("FVD(real split A, real split B): skipped (need at least 4 real images)")

    if len(gen_records) >= 4:
        gen_left, gen_right = split_records(gen_records, seed=args.seed)
        gen_left_features = extract_features(
            gen_left,
            extractor=extractor,
            batch_size=args.batch_size,
        )
        gen_right_features = extract_features(
            gen_right,
            extractor=extractor,
            batch_size=args.batch_size,
        )
        print(
            "FVD(gen split A, gen split B):   "
            f"{compute_fvd_between(gen_left_features, gen_right_features):.6f}"
        )
    else:
        print("FVD(gen split A, gen split B): skipped (need at least 4 generated images)")


if __name__ == "__main__":
    main()
