#!/usr/bin/env python3
"""Link CellViT JSON results into ablation cache directories as _cellvit_instances.json sidecars.

Each CellViT JSON in cellvit_dir is named:
    {tile_id}__{section}__{rest}.json
which corresponds to the generated image:
    ablation_results/{tile_id}/{section}/{rest}.png    (or .../generated_he.png for 'all')

The sidecar is written (hard-linked or copied) as:
    ablation_results/{tile_id}/{section}/{rest}_cellvit_instances.json
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _parse_cellvit_filename(stem: str) -> tuple[str, str, str] | None:
    """Return (tile_id, section, rest) parsed from a cellvit filename stem."""
    parts = stem.split("__", 2)
    if len(parts) != 3:
        return None
    tile_id, section, rest = parts
    if not tile_id or not section or not rest:
        return None
    return tile_id, section, rest


def link_sidecars(
    cellvit_dir: Path,
    ablation_dir: Path,
    *,
    overwrite: bool = False,
) -> tuple[int, int]:
    """Create _cellvit_instances.json sidecars. Returns (linked, skipped)."""
    linked = 0
    skipped = 0

    for json_file in sorted(cellvit_dir.glob("*.json")):
        parsed = _parse_cellvit_filename(json_file.stem)
        if parsed is None:
            print(f"  skip (bad name): {json_file.name}", file=sys.stderr)
            skipped += 1
            continue

        tile_id, section, rest = parsed
        tile_cache_dir = ablation_dir / tile_id
        if not tile_cache_dir.is_dir():
            skipped += 1
            continue

        target = tile_cache_dir / section / f"{rest}_cellvit_instances.json"
        if target.exists() and not overwrite:
            skipped += 1
            continue

        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            # Prefer hard link to avoid doubling disk usage; fall back to copy.
            if target.exists():
                target.unlink()
            os.link(json_file, target)
        except OSError:
            shutil.copy2(json_file, target)
        linked += 1

    return linked, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cellvit-dir",
        type=Path,
        default=ROOT / "inference_output/concat_ablation_1000/unpaired_ablation/cellvit",
        help="Directory containing flat CellViT JSON files",
    )
    parser.add_argument(
        "--ablation-dir",
        type=Path,
        default=ROOT / "inference_output/concat_ablation_1000/unpaired_ablation/ablation_results",
        help="Parent directory of per-tile ablation cache directories",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing sidecar files",
    )
    args = parser.parse_args()

    cellvit_dir = args.cellvit_dir.resolve()
    ablation_dir = args.ablation_dir.resolve()

    if not cellvit_dir.is_dir():
        sys.exit(f"cellvit-dir not found: {cellvit_dir}")
    if not ablation_dir.is_dir():
        sys.exit(f"ablation-dir not found: {ablation_dir}")

    print(f"Linking CellViT sidecars from {cellvit_dir}")
    print(f"  → {ablation_dir}")
    linked, skipped = link_sidecars(cellvit_dir, ablation_dir, overwrite=args.overwrite)
    print(f"Done: {linked} linked, {skipped} skipped.")


if __name__ == "__main__":
    main()
