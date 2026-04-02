"""
Cache helpers for Stage 3 single/pair/triple ablation images.

The goal is to make layout iteration cheap: generate subset images once, then reload
their PNGs to rebuild alternative combined figures without rerunning diffusion.
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol
import json

import numpy as np
from PIL import Image

from tools.stage3.ablation import AblationCondition

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


def _as_uint8_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    if image.dtype == np.uint8:
        return image
    clipped = np.clip(image, 0.0, 1.0)
    return (clipped * 255).astype(np.uint8)


def _as_uint8_mask(mask: np.ndarray) -> np.ndarray:
    if mask.dtype == np.uint8:
        return mask
    return (np.clip(mask, 0.0, 1.0) * 255).astype(np.uint8)


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
            Image.fromarray(_as_uint8_rgb(image)).save(cache_dir / rel_path)
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
    manifest_path = cache_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

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


def is_per_tile_cache_manifest_dir(cache_dir: Path) -> bool:
    """True if ``cache_dir`` is a single-tile cache (``manifest.json`` at this level)."""
    return (Path(cache_dir) / "manifest.json").is_file()
