#!/usr/bin/env python3
"""Build a deterministic unpaired ORION-style dataset root for Stage 3 ablations.

The output dataset keeps each layout tile's ``exp_channels/`` but remaps ``he/`` and
``features/`` to a different style tile, saved under the original layout tile ID.
This lets the existing Stage 3 / ablation CLIs run unchanged against an unpaired root.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
_FILE_EXTS: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".npy")


def _link_or_copy(src: Path, dst: Path, *, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "hardlink":
        dst.hardlink_to(src)
    elif mode == "symlink":
        dst.symlink_to(src.resolve())
    elif mode == "copy":
        shutil.copy2(src, dst)
    else:
        raise ValueError(f"unsupported mode: {mode}")


def _find_tile_file(folder: Path, tile_id: str) -> Path:
    for ext in _FILE_EXTS:
        path = folder / f"{tile_id}{ext}"
        if path.is_file():
            return path
    raise FileNotFoundError(f"missing tile file for {tile_id!r} under {folder}")


def _has_any_tile_file(folder: Path, tile_ids: list[str]) -> bool:
    for tile_id in tile_ids[: min(8, len(tile_ids))]:
        try:
            _find_tile_file(folder, tile_id)
            return True
        except FileNotFoundError:
            continue
    return False


def _paired_tile_ids(cache_root: Path) -> list[str]:
    tile_ids = sorted(
        p.name
        for p in cache_root.iterdir()
        if p.is_dir() and (p / "manifest.json").is_file()
    )
    if not tile_ids:
        raise FileNotFoundError(
            f"no per-tile cache dirs found under {cache_root} "
            "(expected <tile_id>/manifest.json)"
        )
    return tile_ids


def _derangement(items: list[str], seed: int) -> list[str]:
    if len(items) < 2:
        raise ValueError("need at least 2 items to build an unpaired mapping")
    rng = random.Random(seed)
    shuffled = list(items)
    for _ in range(10_000):
        rng.shuffle(shuffled)
        if all(a != b for a, b in zip(items, shuffled, strict=True)):
            return list(shuffled)
    # Fallback: rotate a deterministic shuffle to avoid any fixed points.
    shuffled = sorted(items)
    offset = (seed % (len(items) - 1)) + 1
    return shuffled[offset:] + shuffled[:offset]


def build_unpaired_dataset(
    *,
    paired_cache_root: Path,
    data_root: Path,
    output_root: Path,
    seed: int,
    mode: str,
    overwrite: bool,
) -> Path:
    tile_ids = _paired_tile_ids(paired_cache_root)
    style_tile_ids = _derangement(tile_ids, seed)
    mapping = dict(zip(tile_ids, style_tile_ids, strict=True))

    exp_src = data_root / "exp_channels"
    feat_src = data_root / "features"
    he_src = data_root / "he"

    if not exp_src.is_dir() or not feat_src.is_dir() or not he_src.is_dir():
        raise FileNotFoundError(
            f"expected ORION-style data root with exp_channels/, features/, he/: {data_root}"
        )

    if output_root.exists():
        existing = list(output_root.iterdir())
        if existing and not overwrite:
            raise FileExistsError(
                f"output root already exists and is non-empty: {output_root} "
                "(pass --overwrite to rebuild)"
            )
    output_root.mkdir(parents=True, exist_ok=True)

    exp_out = output_root / "exp_channels"
    feat_out = output_root / "features"
    he_out = output_root / "he"
    meta_out = output_root / "metadata"
    exp_out.mkdir(parents=True, exist_ok=True)
    feat_out.mkdir(parents=True, exist_ok=True)
    he_out.mkdir(parents=True, exist_ok=True)
    meta_out.mkdir(parents=True, exist_ok=True)

    channel_dirs = sorted(
        p for p in exp_src.iterdir()
        if p.is_dir() and _has_any_tile_file(p, tile_ids)
    )
    if not channel_dirs:
        raise FileNotFoundError(f"no channel directories found under {exp_src}")

    for channel_dir in channel_dirs:
        out_dir = exp_out / channel_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        for tile_id in tile_ids:
            src = _find_tile_file(channel_dir, tile_id)
            dst = out_dir / src.name
            _link_or_copy(src, dst, mode=mode)

    for layout_tile_id, style_tile_id in mapping.items():
        he_src_path = _find_tile_file(he_src, style_tile_id)
        he_dst = he_out / f"{layout_tile_id}{he_src_path.suffix.lower()}"
        _link_or_copy(he_src_path, he_dst, mode=mode)

        feat_src_path = feat_src / f"{style_tile_id}_uni.npy"
        if not feat_src_path.is_file():
            raise FileNotFoundError(f"missing UNI feature file: {feat_src_path}")
        feat_dst = feat_out / f"{layout_tile_id}_uni.npy"
        _link_or_copy(feat_src_path, feat_dst, mode=mode)

    payload = {
        "version": 1,
        "seed": seed,
        "paired_cache_root": str(paired_cache_root.resolve()),
        "source_data_root": str(data_root.resolve()),
        "output_root": str(output_root.resolve()),
        "link_mode": mode,
        "tile_count": len(tile_ids),
        "tile_ids": tile_ids,
        "style_mapping": mapping,
    }
    mapping_path = meta_out / "unpaired_mapping.json"
    mapping_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return mapping_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create an ORION-style dataset root for unpaired ablation runs using the exact "
            "tile IDs already present in a paired ablation cache."
        ),
    )
    parser.add_argument(
        "--paired-cache-root",
        type=Path,
        default=ROOT / "inference_output" / "paired_ablation" / "ablation_results",
        help="Parent directory of the existing paired per-tile ablation caches.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=ROOT / "data" / "orion-crc33",
        help="Source ORION dataset root containing exp_channels/, features/, he/.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "inference_output" / "unpaired_ablation" / "data" / "orion-crc33-unpaired",
        help="Destination ORION-style dataset root.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the style permutation.")
    parser.add_argument(
        "--mode",
        choices=["hardlink", "symlink", "copy"],
        default="hardlink",
        help="How to populate the new dataset root (default: hardlink).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild even if the output root already exists and is non-empty.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mapping_path = build_unpaired_dataset(
        paired_cache_root=args.paired_cache_root.resolve(),
        data_root=args.data_root.resolve(),
        output_root=args.output_root.resolve(),
        seed=int(args.seed),
        mode=str(args.mode),
        overwrite=bool(args.overwrite),
    )
    print(f"Wrote unpaired dataset root -> {args.output_root.resolve()}")
    print(f"Wrote mapping -> {mapping_path}")


if __name__ == "__main__":
    main()
