#!/usr/bin/env python3
"""Flatten ablation-cache PNGs into one CellViT-friendly batch folder."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.stage3.ablation_cache import (
    is_per_tile_cache_manifest_dir,
    list_cached_tile_ids,
    load_manifest,
)


def _flat_name(tile_id: str, rel_image_path: str) -> str:
    rel_path = Path(rel_image_path)
    return f"{tile_id}__{'__'.join(rel_path.parts)}"


def _iter_manifest_png_entries(cache_dir: Path) -> list[dict[str, str]]:
    manifest = load_manifest(cache_dir)
    tile_id = str(manifest["tile_id"])
    entries: list[dict[str, str]] = []
    for section in manifest.get("sections", []):
        for entry in section.get("entries", []):
            rel_image_path = str(entry["image_path"])
            source_path = cache_dir / rel_image_path
            if not source_path.is_file():
                raise FileNotFoundError(f"missing source PNG: {source_path}")
            entries.append(
                {
                    "tile_id": tile_id,
                    "rel_image_path": rel_image_path,
                    "source_path": str(source_path.resolve()),
                    "flat_name": _flat_name(tile_id, rel_image_path),
                }
            )
    return entries


def _copy_or_link(src: Path, dst: Path, mode: str) -> None:
    if mode == "copy":
        shutil.copy2(src, dst)
        return
    if mode == "hardlink":
        os.link(src, dst)
        return
    raise ValueError(f"unsupported mode: {mode}")


def export_cellvit_batch(
    cache_root: Path,
    output_dir: Path,
    *,
    mode: str = "copy",
    overwrite: bool = False,
    make_zip: bool = False,
) -> tuple[Path, Path, int]:
    """Export one flat images folder plus manifest for CellViT processing."""
    cache_root = Path(cache_root).resolve()
    output_dir = Path(output_dir).resolve()
    images_dir = output_dir / "images"

    if output_dir.exists():
        existing = list(output_dir.iterdir())
        if existing and not overwrite:
            raise FileExistsError(
                f"output dir is not empty: {output_dir} (pass --overwrite to reuse it)"
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    cache_dirs: list[Path]
    if is_per_tile_cache_manifest_dir(cache_root):
        cache_dirs = [cache_root]
    else:
        cache_dirs = [cache_root / tile_id for tile_id in list_cached_tile_ids(cache_root)]

    manifest_entries: list[dict[str, str]] = []
    seen_names: set[str] = set()
    for cache_dir in cache_dirs:
        for entry in _iter_manifest_png_entries(cache_dir):
            flat_name = entry["flat_name"]
            if flat_name in seen_names:
                raise ValueError(f"duplicate flat export name: {flat_name}")
            seen_names.add(flat_name)
            src = Path(entry["source_path"])
            dst = images_dir / flat_name
            if dst.exists():
                if overwrite:
                    dst.unlink()
                else:
                    raise FileExistsError(f"destination already exists: {dst}")
            _copy_or_link(src, dst, mode)
            entry["flat_path"] = str(dst.resolve())
            manifest_entries.append(entry)

    manifest_payload = {
        "version": 1,
        "cache_root": str(cache_root),
        "images_dir": str(images_dir),
        "mode": mode,
        "entries": manifest_entries,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest_payload, indent=2) + "\n", encoding="utf-8")

    zip_path = output_dir.with_suffix(".zip")
    if make_zip:
        archive_base = str(output_dir)
        shutil.make_archive(archive_base, "zip", root_dir=output_dir)
    return manifest_path, zip_path, len(manifest_entries)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Flatten ablation cache PNGs into one CellViT batch folder.",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        required=True,
        help="Single tile cache dir or parent directory containing per-tile caches.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Folder to create. Output PNGs go under <output-dir>/images/.",
    )
    parser.add_argument(
        "--mode",
        choices=["copy", "hardlink"],
        default="copy",
        help="How to populate the flat images folder (default: copy).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow reusing a non-empty output directory.",
    )
    parser.add_argument(
        "--zip",
        action="store_true",
        help="Also create <output-dir>.zip for transfer.",
    )
    args = parser.parse_args()

    manifest_path, zip_path, count = export_cellvit_batch(
        args.cache_root,
        args.output_dir,
        mode=args.mode,
        overwrite=args.overwrite,
        make_zip=args.zip,
    )
    print(f"Exported {count} PNGs → {args.output_dir / 'images'}")
    print(f"Manifest → {manifest_path}")
    if args.zip:
        print(f"Zip → {zip_path}")


if __name__ == "__main__":
    main()
