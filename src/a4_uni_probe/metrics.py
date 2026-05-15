"""Per-image morphology metrics for generated H&E tiles."""

from __future__ import annotations

from pathlib import Path

from src.a4_uni_probe.labels import MORPHOLOGY_ATTR_NAMES, compute_morphology_attributes_from_cellvit
from tools.cellvit.contours import cellvit_sidecar_path


def morphology_row_for_image(image_path: str | Path) -> dict[str, float]:
    """Read a cached CellViT sidecar beside a generated PNG when present."""
    path = Path(image_path)
    sidecar_json = cellvit_sidecar_path(path)
    if sidecar_json.is_file():
        return compute_morphology_attributes_from_cellvit(sidecar_json)
    png_json = path.with_name(f"{path.name}.json")
    if png_json.is_file():
        return compute_morphology_attributes_from_cellvit(png_json)
    raw_json = path.with_suffix(".json")
    if raw_json.is_file():
        return compute_morphology_attributes_from_cellvit(raw_json)
    return {name: float("nan") for name in MORPHOLOGY_ATTR_NAMES}
