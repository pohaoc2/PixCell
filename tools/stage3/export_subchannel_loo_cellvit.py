#!/usr/bin/env python3
"""Flatten subchannel-LOO PNGs into one CellViT batch folder.

Exports:
  - <tile_id>/all_baseline.png
  - <tile_id>/<sub_channel>/generated_he.png

The manifest schema matches `tools/cellvit/import_results.py`: each entry stores
the original `source_path` plus a unique `flat_name` so CellViT JSON outputs can
be copied back beside the source PNGs.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_ROOT = ROOT / "inference_output" / "subchannel_loo_n300"
SUBCHANNELS: tuple[str, ...] = (
    "cell_type_healthy",
    "cell_type_cancer",
    "cell_type_immune",
    "cell_state_prolif",
    "cell_state_nonprolif",
    "cell_state_dead",
    "vasculature",
    "oxygen",
    "glucose",
)


def _iter_png_entries(cache_root: Path, *, include_baseline: bool) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    seen_names: set[str] = set()
    for tile_dir in sorted(cache_root.iterdir()):
        if not tile_dir.is_dir() or tile_dir.name.startswith("_"):
            continue
        tile_id = tile_dir.name
        if include_baseline:
            baseline = tile_dir / "all_baseline.png"
            if baseline.is_file():
                flat_name = f"{tile_id}__all_baseline.png"
                if flat_name in seen_names:
                    raise ValueError(f"duplicate export name: {flat_name}")
                seen_names.add(flat_name)
                entries.append(
                    {
                        "tile_id": tile_id,
                        "kind": "baseline",
                        "sub_channel": "all_baseline",
                        "source_path": str(baseline.resolve()),
                        "flat_name": flat_name,
                    }
                )
        for sub_channel in SUBCHANNELS:
            image_path = tile_dir / sub_channel / "generated_he.png"
            if not image_path.is_file():
                continue
            flat_name = f"{tile_id}__{sub_channel}__generated_he.png"
            if flat_name in seen_names:
                raise ValueError(f"duplicate export name: {flat_name}")
            seen_names.add(flat_name)
            entries.append(
                {
                    "tile_id": tile_id,
                    "kind": "loo",
                    "sub_channel": sub_channel,
                    "source_path": str(image_path.resolve()),
                    "flat_name": flat_name,
                }
            )
    return entries


def export_subchannel_loo_cellvit_batch(
    cache_root: Path,
    output_dir: Path,
    *,
    overwrite: bool = False,
    make_zip: bool = False,
    include_baseline: bool = True,
) -> tuple[Path, Path, int]:
    cache_root = Path(cache_root).resolve()
    output_dir = Path(output_dir).resolve()
    images_dir = output_dir / "images"

    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"output dir is not empty: {output_dir} (pass --overwrite to reuse it)")
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    entries = _iter_png_entries(cache_root, include_baseline=include_baseline)
    for entry in entries:
        src = Path(entry["source_path"])
        dst = images_dir / entry["flat_name"]
        if dst.exists():
            if overwrite:
                dst.unlink()
            else:
                raise FileExistsError(f"destination already exists: {dst}")
        shutil.copy2(src, dst)
        entry["flat_path"] = str(dst.resolve())

    manifest = {
        "version": 1,
        "kind": "subchannel_loo_cellvit_export",
        "cache_root": str(cache_root),
        "images_dir": str(images_dir),
        "entries": entries,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    zip_path = output_dir.with_suffix(".zip")
    if make_zip:
        shutil.make_archive(str(output_dir), "zip", root_dir=output_dir)
    return manifest_path, zip_path, len(entries)


def main() -> None:
    parser = argparse.ArgumentParser(description="Flatten subchannel LOO PNGs for CellViT")
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--zip", action="store_true")
    parser.add_argument(
        "--no-baseline",
        action="store_true",
        help="Skip all_baseline.png exports (default exports baseline + LOO images).",
    )
    args = parser.parse_args()

    manifest_path, zip_path, count = export_subchannel_loo_cellvit_batch(
        args.cache_root,
        args.output_dir,
        overwrite=args.overwrite,
        make_zip=args.zip,
        include_baseline=not args.no_baseline,
    )
    print(f"Exported {count} PNGs -> {args.output_dir / 'images'}")
    print(f"Manifest -> {manifest_path}")
    if args.zip:
        print(f"Zip -> {zip_path}")


if __name__ == "__main__":
    main()