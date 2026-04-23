"""Tile-ID parsing, ordering, and hashing helpers."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Iterable


_TILE_ID_RE = re.compile(r"^(?P<row>\d+)_(?P<col>\d+)$")


def parse_tile_id(tile_id: str) -> tuple[int, int]:
    """Parse a tile id like ``10240_11008`` into integer pixel coordinates."""
    match = _TILE_ID_RE.match(str(tile_id).strip())
    if match is None:
        raise ValueError(f"invalid tile_id={tile_id!r}")
    return int(match.group("row")), int(match.group("col"))


def sort_tile_ids_numeric(tile_ids: Iterable[str]) -> list[str]:
    """Sort tile IDs numerically instead of lexicographically."""
    return sorted({str(tile_id) for tile_id in tile_ids}, key=parse_tile_id)


def list_feature_tile_ids(features_dir: str | Path, suffix: str = "_uni.npy") -> list[str]:
    """List tile IDs from cached UNI feature files in stable numeric order."""
    feature_dir = Path(features_dir)
    tile_ids: list[str] = []
    for path in feature_dir.glob(f"*{suffix}"):
        tile_ids.append(path.name[: -len(suffix)])
    return sort_tile_ids_numeric(tile_ids)


def tile_ids_sha1(tile_ids: Iterable[str]) -> str:
    """Hash a tile-id sequence for downstream alignment checks."""
    digest = hashlib.sha1()
    for tile_id in tile_ids:
        digest.update(str(tile_id).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def write_tile_ids(tile_ids: Iterable[str], output_path: str | Path) -> Path:
    """Write a canonical newline-delimited tile-id list."""
    out_path = Path(output_path)
    ordered = list(tile_ids)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(ordered) + "\n", encoding="utf-8")
    return out_path
