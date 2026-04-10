"""
Cache helpers for Stage 3 single/pair/triple ablation images.

The goal is to make layout iteration cheap: generate subset images once, then reload
their PNGs to rebuild alternative combined figures without rerunning diffusion.
"""
from __future__ import annotations

from collections.abc import Sequence
from math import comb
from pathlib import Path
from typing import Protocol
import json

import numpy as np
from PIL import Image

from tools.stage3.ablation import AblationCondition
from tools.stage3.common import to_uint8_rgb

_SUBSET_DIR_NAMES = {
    1: "singles",
    2: "pairs",
    3: "triples",
    4: "all",
}


class _SectionLike(Protocol):
    title: str
    conditions: Sequence[AblationCondition]
    images: Sequence[tuple[str, np.ndarray]]


def subset_dir_name(subset_size: int) -> str:
    """Directory name used for one subset cardinality."""
    try:
        return _SUBSET_DIR_NAMES[subset_size]
    except KeyError as exc:
        raise ValueError(f"unsupported subset size {subset_size}") from exc


def condition_slug(active_groups: Sequence[str]) -> str:
    """Stable filesystem slug for one group combination."""
    if not active_groups:
        return "none"
    return "__".join(active_groups)


def _as_uint8_mask(mask: np.ndarray) -> np.ndarray:
    if mask.dtype == np.uint8:
        return mask
    return (np.clip(mask, 0.0, 1.0) * 255).astype(np.uint8)


def load_manifest(cache_dir_or_manifest_path: str | Path) -> dict:
    """Load a per-tile ablation cache manifest."""
    path = Path(cache_dir_or_manifest_path)
    manifest_path = path if path.name == "manifest.json" else path / "manifest.json"
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def resolve_all_image_path(
    cache_dir: str | Path,
    manifest: dict | None = None,
    *,
    n_groups: int | None = None,
) -> Path | None:
    """Resolve the canonical all-groups image with manifest-first fallback behavior."""
    cache_dir = Path(cache_dir)
    manifest = load_manifest(cache_dir) if manifest is None else manifest
    if n_groups is None:
        n_groups = len(manifest.get("group_names") or ())

    for section in manifest.get("sections", []):
        try:
            subset_size = int(section.get("subset_size", 0))
        except (TypeError, ValueError):
            continue
        if subset_size != n_groups:
            continue
        entries = section.get("entries") or []
        if not entries:
            continue
        rel = Path(entries[0].get("image_path", ""))
        if rel and (cache_dir / rel).is_file():
            return cache_dir / rel

    canonical = cache_dir / "all" / "generated_he.png"
    if canonical.is_file():
        return canonical

    all_dir = cache_dir / "all"
    if all_dir.is_dir():
        pngs = sorted(all_dir.glob("*.png"))
        if len(pngs) == 1:
            return pngs[0]
    return None


def save_subset_condition_cache(
    cache_dir: str | Path,
    *,
    tile_id: str,
    group_names: Sequence[str],
    sections: Sequence[_SectionLike],
    cell_mask: np.ndarray | None = None,
) -> Path:
    """Save individual subset-condition PNGs plus a manifest describing the layout."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "version": 1,
        "tile_id": tile_id,
        "group_names": list(group_names),
        "sections": [],
    }

    for section in sections:
        if not section.conditions:
            raise ValueError(f"section {section.title!r} has no conditions")
        if len(section.conditions) != len(section.images):
            raise ValueError(
                f"section {section.title!r} has {len(section.conditions)} conditions "
                f"but {len(section.images)} images"
            )

        subset_size = len(section.conditions[0].active_groups)
        section_dir_name = subset_dir_name(subset_size)
        section_dir = cache_dir / section_dir_name
        section_dir.mkdir(parents=True, exist_ok=True)

        entries = []
        for idx, (condition, (image_label, image)) in enumerate(
            zip(section.conditions, section.images, strict=True),
            start=1,
        ):
            if (
                subset_size == len(group_names)
                and len(section.conditions) == 1
                and len(section.images) == 1
            ):
                # Canonical All-channels path used by downstream figure scripts.
                rel_path = Path(section_dir_name) / "generated_he.png"
            else:
                rel_path = Path(section_dir_name) / f"{idx:02d}_{condition_slug(condition.active_groups)}.png"
            Image.fromarray(to_uint8_rgb(image, value_range="unit")).save(cache_dir / rel_path)
            entries.append(
                {
                    "active_groups": list(condition.active_groups),
                    "condition_label": condition.label,
                    "image_label": image_label,
                    "image_path": rel_path.as_posix(),
                }
            )

        manifest["sections"].append(
            {
                "title": section.title,
                "subset_size": subset_size,
                "entries": entries,
            }
        )

    if cell_mask is not None:
        cell_mask_path = cache_dir / "cell_mask.png"
        Image.fromarray(_as_uint8_mask(cell_mask)).save(cell_mask_path)
        manifest["cell_mask_path"] = cell_mask_path.name

    manifest_path = cache_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def load_subset_condition_cache(cache_dir: str | Path) -> dict:
    """Load cached subset-condition PNGs and reconstruct their manifest structure."""
    cache_dir = Path(cache_dir)
    manifest = load_manifest(cache_dir)

    sections = []
    for raw_section in manifest["sections"]:
        conditions = []
        images = []
        for entry in raw_section["entries"]:
            conditions.append(
                AblationCondition(
                    label=entry["condition_label"],
                    active_groups=tuple(entry["active_groups"]),
                )
            )
            image = np.array(Image.open(cache_dir / entry["image_path"]).convert("RGB"))
            images.append((entry["image_label"], image))
        sections.append(
            {
                "title": raw_section["title"],
                "subset_size": raw_section["subset_size"],
                "conditions": conditions,
                "images": images,
            }
        )

    cell_mask = None
    cell_mask_path = manifest.get("cell_mask_path")
    if cell_mask_path:
        cell_mask = np.array(Image.open(cache_dir / cell_mask_path).convert("L"), dtype=np.float32) / 255.0

    return {
        "tile_id": manifest["tile_id"],
        "group_names": tuple(manifest["group_names"]),
        "sections": sections,
        "cell_mask": cell_mask,
    }


def list_cached_tile_ids(cache_parent: Path) -> list[str]:
    """Tile IDs: immediate subdirs of ``cache_parent`` that contain ``manifest.json``."""
    cache_parent = Path(cache_parent)
    if not cache_parent.is_dir():
        raise FileNotFoundError(f"cache directory not found: {cache_parent}")
    return sorted(
        p.name
        for p in cache_parent.iterdir()
        if p.is_dir() and (p / "manifest.json").is_file()
    )


def is_complete_tile_cache_dir(cache_dir: Path) -> bool:
    """True when a per-tile cache manifest exists and every expected image file is present."""
    cache_dir = Path(cache_dir)
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = load_manifest(cache_dir)
    except Exception:
        return False

    group_names = tuple(manifest.get("group_names") or ())
    if not group_names:
        return False

    sections = manifest.get("sections") or []
    by_subset_size: dict[int, dict] = {}
    for section in sections:
        try:
            subset_size = int(section.get("subset_size", 0))
        except (TypeError, ValueError):
            return False
        if subset_size < 1 or subset_size > len(group_names):
            return False
        by_subset_size[subset_size] = section

    for subset_size in range(1, len(group_names) + 1):
        section = by_subset_size.get(subset_size)
        if section is None:
            return False
        entries = section.get("entries") or []
        if len(entries) != comb(len(group_names), subset_size):
            return False
        for entry in entries:
            image_path = entry.get("image_path")
            if not image_path or not (cache_dir / image_path).is_file():
                return False

    cell_mask_path = manifest.get("cell_mask_path")
    if cell_mask_path and not (cache_dir / cell_mask_path).is_file():
        return False
    return True


def list_complete_cached_tile_ids(cache_parent: Path) -> list[str]:
    """Tile IDs whose cache dirs contain a complete manifest + image set."""
    cache_parent = Path(cache_parent)
    if not cache_parent.is_dir():
        raise FileNotFoundError(f"cache directory not found: {cache_parent}")
    return sorted(
        p.name
        for p in cache_parent.iterdir()
        if p.is_dir() and is_complete_tile_cache_dir(p)
    )


def is_per_tile_cache_manifest_dir(cache_dir: Path) -> bool:
    """True if ``cache_dir`` is a single-tile cache (``manifest.json`` at this level)."""
    return (Path(cache_dir) / "manifest.json").is_file()
