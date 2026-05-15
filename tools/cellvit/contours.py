"""Shared helpers for CellViT instance-segmentation sidecars.

Each generated PNG may have a sibling JSON written by `tools/cellvit/import_results.py`
named `<stem>_cellvit_instances.json`. This module centralizes (a) the sidecar path
convention, (b) loading the polygon contours, and (c) plotting them as a contour overlay
on an existing matplotlib axes.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes

CELLVIT_SIDECAR_SUFFIX = "_cellvit_instances.json"


def cellvit_sidecar_path(image_path: Path) -> Path:
    """Return the conventional sidecar JSON path for a generated PNG."""
    return Path(image_path).with_name(f"{Path(image_path).stem}{CELLVIT_SIDECAR_SUFFIX}")


def load_cellvit_contours(image_path: Path) -> list[np.ndarray]:
    """Load polygon contours from the CellViT sidecar (or return [] if missing/invalid)."""
    sidecar = cellvit_sidecar_path(image_path)
    if not sidecar.is_file():
        return []
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except Exception:
        return []
    contours: list[np.ndarray] = []
    for cell in payload.get("cells", []):
        raw = cell.get("contour")
        if not isinstance(raw, list) or len(raw) < 3:
            continue
        arr = np.asarray(raw, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] < 2:
            continue
        contours.append(arr[:, :2])
    return contours


def overlay_cellvit_contours(
    ax: "Axes",
    image_path: Path,
    *,
    color,
    linewidth: float = 0.6,
    alpha: float = 0.85,
    zorder: int = 4,
) -> None:
    """Plot CellViT polygon contours on `ax`."""
    for contour in load_cellvit_contours(image_path):
        ax.plot(
            contour[:, 0],
            contour[:, 1],
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            zorder=zorder,
        )
