"""
cell_masks.py
Converts cell detection JSON/GeoJSON files into cell instance and type masks.

Supported input files (auto-detected in folder):
  - cells.json          – CellViT-style JSON with 'contour' polygons (preferred)
  - cell_detection.json – JSON with only bbox/centroid (fallback; draws filled rectangles)
  - cells.geojson       – GeoJSON MultiPolygon (equivalent to cells.json)
  - cell_detection.geojson – GeoJSON MultiPoint (centroid-only fallback)
"""

import json
import os
from pathlib import Path

import numpy as np
import cv2


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _bbox_to_canvas_size(cells: list, padding: int = 10) -> tuple[int, int]:
    """Infer a canvas size that fits all cells from their bbox/centroid/contour."""
    max_y, max_x = 0, 0
    for c in cells:
        if "contour" in c and c["contour"]:
            pts = np.array(c["contour"])
            max_y = max(max_y, int(pts[:, 1].max()))
            max_x = max(max_x, int(pts[:, 0].max()))
        elif "bbox" in c:
            (r0, c0), (r1, c1) = c["bbox"]
            max_y = max(max_y, r1)
            max_x = max(max_x, c1)
        elif "centroid" in c:
            cy, cx = c["centroid"]
            max_y = max(max_y, int(cy))
            max_x = max(max_x, int(cx))
    return max_y + padding, max_x + padding


def _draw_polygon(mask: np.ndarray, contour: list, value: int) -> None:
    """Fill a polygon on mask with the given integer value."""
    pts = np.array(contour, dtype=np.int32)
    # contour points are (x, y) → cv2 expects (x, y) inside pts shaped (N,1,2)
    cv2.fillPoly(mask, [pts.reshape(-1, 1, 2)], value)


def _draw_bbox(mask: np.ndarray, bbox: list, value: int, radius: int = 5) -> None:
    """Fill a rectangle (from bbox) or circle (fallback) when no contour is available."""
    (r0, c0), (r1, c1) = bbox
    # bbox is row-col → cv2 rect needs (x,y) = (col, row)
    cv2.rectangle(mask, (c0, r0), (c1, r1), value, thickness=-1)


def _draw_centroid(mask: np.ndarray, centroid: list, value: int, radius: int = 6) -> None:
    cy, cx = int(centroid[0]), int(centroid[1])
    cv2.circle(mask, (cx, cy), radius, value, thickness=-1)


# ---------------------------------------------------------------------------
# readers
# ---------------------------------------------------------------------------

def _read_cells_json(path: Path) -> tuple[list, dict]:
    with open(path) as f:
        data = json.load(f)
    return data["cells"], data.get("type_map", {})


def _read_cells_geojson(path: Path) -> tuple[list, dict]:
    """Parse GeoJSON MultiPolygon → list of cell dicts compatible with cells.json format."""
    with open(path) as f:
        features = json.load(f)

    cells = []
    type_map = {}
    for feat in features:
        geom = feat["geometry"]
        props = feat.get("properties", {})
        cell = {}

        if geom["type"] == "MultiPolygon":
            # Each ring is one cell contour; treat first ring of each polygon as the contour
            contours = [ring[0] for poly in geom["coordinates"] for ring in poly]
            cell["contour"] = contours[0] if contours else []
            # Additional cells embedded in the same feature (CellViT GeoJSON style)
            extra_contours = contours[1:]
        elif geom["type"] == "Polygon":
            cell["contour"] = geom["coordinates"][0]
            extra_contours = []
        elif geom["type"] == "MultiPoint":
            # centroid-only file
            for pt in geom["coordinates"]:
                cells.append({"centroid": [pt[1], pt[0]], "type": 0})
            continue
        elif geom["type"] == "Point":
            cells.append({"centroid": [geom["coordinates"][1], geom["coordinates"][0]], "type": 0})
            continue
        else:
            continue

        cls = props.get("classification", {})
        cell["type"] = 0  # unknown unless mapped
        cell["centroid"] = props.get("centroid", [])
        cells.append(cell)

        for ec in extra_contours:
            cells.append({"contour": ec, "type": 0, "centroid": []})

    return cells, type_map


def _read_detection_json(path: Path) -> tuple[list, dict]:
    """cell_detection.json – bbox + centroid, no contour."""
    with open(path) as f:
        data = json.load(f)
    return data["cells"], data.get("type_map", {})


# ---------------------------------------------------------------------------
# main function
# ---------------------------------------------------------------------------

def cells_to_masks(
    folder: str,
    canvas_hw: tuple[int, int] = None,
    instance_dtype: np.dtype = np.int32,
    type_dtype: np.dtype = np.uint8,
) -> dict:
    """
    Convert cell detection files in *folder* into numpy mask arrays.

    Parameters
    ----------
    folder : path-like
        Directory containing one or more of:
          cells.json, cell_detection.json, cells.geojson, cell_detection.geojson
    canvas_hw : (height, width) or None
        Canvas size. Inferred from the data when None.
    instance_dtype : numpy dtype
        Dtype for the instance mask (each cell gets a unique integer ID).
    type_dtype : numpy dtype
        Dtype for the type mask (pixel value = cell type index).

    Returns
    -------
    dict with keys:
        "instance_mask"  – (H, W) array; 0 = background, 1…N = cell IDs
        "type_mask"      – (H, W) array; 0 = background, cell type index elsewhere
        "cells"          – list of cell dicts (as parsed)
        "type_map"       – {str_index: type_name} mapping
        "n_cells"        – total number of cells rendered
    """
    folder = Path(folder)

    # --- pick the best available file ---
    cells, type_map = [], {}
    for fname, reader in [
        ("cells.json",              _read_cells_json),
        ("cells.geojson",           _read_cells_geojson),
        ("cell_detection.json",     _read_detection_json),
        ("cell_detection.geojson",  _read_cells_geojson),
    ]:
        fpath = folder / fname
        if fpath.exists():
            cells, type_map = reader(fpath)
            print(f"[cells_to_masks] loaded {len(cells)} cells from '{fname}'")
            break
    else:
        raise FileNotFoundError(f"No recognised cell file found in '{folder}'")

    # --- determine canvas size ---
    if canvas_hw is None:
        h, w = _bbox_to_canvas_size(cells)
    else:
        h, w = canvas_hw

    instance_mask = np.zeros((h, w), dtype=instance_dtype)
    type_mask     = np.zeros((h, w), dtype=type_dtype)

    for cell_id, cell in enumerate(cells, start=1):
        cell_type = int(cell.get("type", 0))

        if "contour" in cell and len(cell["contour"]) >= 3:
            _draw_polygon(instance_mask, cell["contour"], cell_id)
            _draw_polygon(type_mask,     cell["contour"], cell_type)

        elif "bbox" in cell:
            _draw_bbox(instance_mask, cell["bbox"], cell_id)
            _draw_bbox(type_mask,     cell["bbox"], cell_type)

        elif "centroid" in cell and cell["centroid"]:
            _draw_centroid(instance_mask, cell["centroid"], cell_id)
            _draw_centroid(type_mask,     cell["centroid"], cell_type)

    return {
        "instance_mask": instance_mask,
        "type_mask":     type_mask,
        "cells":         cells,
        "type_map":      type_map,
        "n_cells":       len(cells),
    }


# ---------------------------------------------------------------------------
# optional: save masks as PNG / NPZ
# ---------------------------------------------------------------------------

def save_masks(result: dict, out_dir: str) -> None:
    """
    Save instance and type masks to *out_dir*.
    Saves:
      instance_mask.npy / instance_mask_vis.png  (colorised for viewing)
      type_mask.npy     / type_mask_vis.png
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    inst = result["instance_mask"]
    typ  = result["type_mask"]

    np.save(out_dir / "instance_mask.npy", inst)
    np.save(out_dir / "type_mask.npy",     typ)

    # --- visual: instance mask (random colours per cell) ---
    n = inst.max()
    rng = np.random.default_rng(42)
    palette = np.zeros((n + 1, 3), dtype=np.uint8)
    palette[1:] = rng.integers(80, 255, size=(n, 3))
    inst_vis = palette[inst]
    cv2.imwrite(str(out_dir / "instance_mask_vis.png"), cv2.cvtColor(inst_vis, cv2.COLOR_RGB2BGR))

    # --- visual: type mask (scaled to 0-255, background always black) ---
    # Use instance mask to distinguish true background (inst==0) from type-0 cells
    n_types = max(typ.max(), inst.max() > 0)  # at least 1 if any cell exists
    max_type = int(typ[inst > 0].max()) if (inst > 0).any() else 1
    # Shift cell type values to 1-based so type-0 cells get a distinct colour
    typ_shifted = np.where(inst > 0, typ.astype(np.int32) + 1, 0).astype(np.uint8)
    # Scale to full 0-255 range across (max_type+1) levels
    scale = 255 / (max_type + 1) if max_type >= 0 else 255
    type_vis = (typ_shifted.astype(np.float32) * scale).astype(np.uint8)
    type_vis_color = cv2.applyColorMap(type_vis, cv2.COLORMAP_JET)
    type_vis_color[inst == 0] = 0  # force true background to black
    cv2.imwrite(str(out_dir / "type_mask_vis.png"), type_vis_color)

    print(f"[save_masks] saved masks to '{out_dir}'")


# ---------------------------------------------------------------------------
# quick demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    parent = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    subfolders = sorted(
        (p for p in parent.iterdir() if p.is_dir() and p.name.split("_")[-1].isdigit()),
        key=lambda p: int(p.name.split("_")[-1])
    )
    subfolders = [p for p in subfolders if p.is_dir()]

    for subfolder in subfolders:
        try:
            result = cells_to_masks(subfolder)
            save_masks(result, subfolder / "masks_output")
            print(f"[done] {subfolder.name} — {result['n_cells']} cells, canvas {result['instance_mask'].shape}")
        except FileNotFoundError as e:
            print(f"[skip] {subfolder.name} — {e}")
        except Exception as e:
            print(f"[error] {subfolder.name} — {e}")