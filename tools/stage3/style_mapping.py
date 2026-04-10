from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path


def load_style_mapping(mapping_json: str | Path | None) -> dict[str, str]:
    """Load a layout->style mapping from unpaired metadata JSON."""
    if mapping_json is None:
        return {}

    path = Path(mapping_json)
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_mapping = payload.get("style_mapping", payload) if isinstance(payload, dict) else payload
    if not isinstance(raw_mapping, dict):
        raise ValueError(f"style mapping JSON must contain an object: {path}")
    return {str(layout_tile): str(style_tile) for layout_tile, style_tile in raw_mapping.items()}


def resolve_style_tile_id(
    tile_id: str,
    *,
    style_mapping: Mapping[str, str] | None = None,
) -> str:
    """Return the mapped style tile ID for a layout tile."""
    if style_mapping is None:
        return str(tile_id)
    return str(style_mapping.get(str(tile_id), tile_id))
