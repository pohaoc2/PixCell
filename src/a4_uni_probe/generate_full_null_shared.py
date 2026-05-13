"""Generate zero-UNI baseline images for a shared tile manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.a4_uni_probe.inference import GenSpec, generate_with_uni_override, load_inference_bundle


def _load_tile_ids(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [str(tile_id) for tile_id in payload["tile_ids"]]


def _shard_tile_ids(tile_ids: list[str], shard_index: int, shard_count: int) -> list[str]:
    if shard_count < 1:
        raise ValueError("shard_count must be at least 1")
    if not 0 <= shard_index < shard_count:
        raise ValueError("shard_index must be in [0, shard_count)")
    start = len(tile_ids) * shard_index // shard_count
    end = len(tile_ids) * (shard_index + 1) // shard_count
    return tile_ids[start:end]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-tiles", type=Path, required=True)
    parser.add_argument("--features-npz", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--config-path", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--exp-channels-dir", type=Path, required=True)
    parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=2.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tile-shard-index", type=int, default=0)
    parser.add_argument("--tile-shard-count", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    tile_ids = _shard_tile_ids(
        _load_tile_ids(args.shared_tiles),
        args.tile_shard_index,
        args.tile_shard_count,
    )
    features = np.load(args.features_npz, allow_pickle=True)
    zero_uni = np.zeros(int(features["uni"].shape[1]), dtype=np.float32)
    bundle = load_inference_bundle(
        checkpoint_dir=args.checkpoint_dir,
        config_path=args.config_path,
        data_root=args.data_root,
        exp_channels_dir=args.exp_channels_dir,
        num_steps=args.num_steps,
    )

    args.out_root.mkdir(parents=True, exist_ok=True)
    total = len(tile_ids)
    for index, tile_id in enumerate(tile_ids, start=1):
        out_path = args.out_root / tile_id / "tme_only.png"
        if out_path.is_file() and not args.overwrite:
            print(f"skip {index}/{total} {tile_id}", flush=True)
            continue
        generate_with_uni_override(
            GenSpec(tile_id=tile_id, uni=zero_uni, out_path=out_path),
            checkpoint_dir=args.checkpoint_dir,
            config_path=args.config_path,
            data_root=args.data_root,
            exp_channels_dir=args.exp_channels_dir,
            num_steps=args.num_steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
            bundle=bundle,
        )
        print(f"done {index}/{total} {tile_id}", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())