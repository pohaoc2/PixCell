"""Region-aware appearance metrics restricted to CellViT nuclei vs stroma.

Replaces the whole-tile pixel statistics in `appearance_metrics.py` with a
two-compartment split:
- `appearance.nuc.*`: metrics computed on nucleus pixels (CellViT mask).
- `appearance.stroma.*`: metrics computed on non-nucleus pixels.

GLCM on irregular regions is handled by replacing masked-out pixels with the
compartment median before quantization, so the same `graycomatrix` machinery
applies. Compartments with too few pixels produce NaN metrics.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from skimage.color import rgb2hed

try:
    from skimage.feature import graycomatrix, graycoprops
except ImportError:  # pragma: no cover
    from skimage.feature import greycomatrix as graycomatrix  # type: ignore
    from skimage.feature import greycoprops as graycoprops  # type: ignore

from tools.cellvit.contours import nucleus_mask_from_cellvit


_MIN_COMPARTMENT_PIXELS = 500
_GLCM_LEVELS = 32
_GLCM_DISTANCES = (1,)
_GLCM_ANGLES = (0.0, np.pi / 4.0, np.pi / 2.0, 3.0 * np.pi / 4.0)

_BASE_METRIC_KEYS = (
    "h_mean",
    "h_std",
    "e_mean",
    "e_std",
    "texture_h_contrast",
    "texture_h_homogeneity",
    "texture_h_energy",
    "texture_e_contrast",
    "texture_e_homogeneity",
    "texture_e_energy",
)


def regional_metric_keys() -> tuple[str, ...]:
    return tuple(f"appearance.{compartment}.{key}" for compartment in ("nuc", "stroma") for key in _BASE_METRIC_KEYS)


def _nan_metrics(compartment: str) -> dict[str, float]:
    return {f"appearance.{compartment}.{key}": float("nan") for key in _BASE_METRIC_KEYS}


def _quantize_with_fill(channel: np.ndarray, mask: np.ndarray) -> np.ndarray:
    pixels = channel[mask]
    pixels = pixels[np.isfinite(pixels)]
    if pixels.size == 0:
        return np.zeros(channel.shape, dtype=np.uint8)
    low = float(np.quantile(pixels, 0.01))
    high = float(np.quantile(pixels, 0.99))
    if high - low < 1e-6:
        return np.zeros(channel.shape, dtype=np.uint8)
    median = float(np.median(pixels))
    filled = np.where(mask, channel, median)
    clipped = np.clip(filled, low, high)
    scaled = np.rint((clipped - low) * (_GLCM_LEVELS - 1) / (high - low)).astype(np.uint8)
    return scaled


def _bbox_crop(mask: np.ndarray) -> tuple[slice, slice] | None:
    if not mask.any():
        return None
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return (slice(int(rows[0]), int(rows[-1]) + 1), slice(int(cols[0]), int(cols[-1]) + 1))


def _safe_mean_std(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(finite)), float(np.std(finite))


def _texture_for_channel(channel: np.ndarray, mask: np.ndarray, *, prefix: str) -> dict[str, float]:
    crop = _bbox_crop(mask)
    if crop is None:
        return {
            f"texture_{prefix}_contrast": float("nan"),
            f"texture_{prefix}_homogeneity": float("nan"),
            f"texture_{prefix}_energy": float("nan"),
        }
    sub_channel = channel[crop]
    sub_mask = mask[crop]
    quantized = _quantize_with_fill(sub_channel, sub_mask)
    glcm = graycomatrix(
        quantized,
        distances=_GLCM_DISTANCES,
        angles=_GLCM_ANGLES,
        levels=_GLCM_LEVELS,
        symmetric=True,
        normed=True,
    )
    return {
        f"texture_{prefix}_contrast": float(np.mean(graycoprops(glcm, "contrast"))),
        f"texture_{prefix}_homogeneity": float(np.mean(graycoprops(glcm, "homogeneity"))),
        f"texture_{prefix}_energy": float(np.mean(graycoprops(glcm, "energy"))),
    }


def _compartment_metrics(h: np.ndarray, e: np.ndarray, mask: np.ndarray, *, compartment: str) -> dict[str, float]:
    n = int(mask.sum())
    if n < _MIN_COMPARTMENT_PIXELS:
        return _nan_metrics(compartment)
    h_mean, h_std = _safe_mean_std(h[mask])
    e_mean, e_std = _safe_mean_std(e[mask])
    out: dict[str, float] = {
        f"appearance.{compartment}.h_mean": h_mean,
        f"appearance.{compartment}.h_std": h_std,
        f"appearance.{compartment}.e_mean": e_mean,
        f"appearance.{compartment}.e_std": e_std,
    }
    h_tex = _texture_for_channel(h, mask, prefix="h")
    e_tex = _texture_for_channel(e, mask, prefix="e")
    out.update({f"appearance.{compartment}.{k}": v for k, v in h_tex.items()})
    out.update({f"appearance.{compartment}.{k}": v for k, v in e_tex.items()})
    return out


def appearance_row_regional(image_path: str | Path, *, nucleus_mask: np.ndarray | None = None) -> dict[str, float]:
    """Return the 20-key regional appearance row for a single PNG."""
    image_path = Path(image_path)
    rgb_u8 = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    height, width = rgb_u8.shape[:2]
    hed = rgb2hed(np.asarray(rgb_u8, dtype=np.float32) / 255.0)
    hematoxylin = np.asarray(hed[:, :, 0], dtype=np.float32)
    eosin = np.asarray(hed[:, :, 1], dtype=np.float32)

    if nucleus_mask is None:
        nucleus_mask = nucleus_mask_from_cellvit(image_path, (height, width))
    nucleus_mask = np.asarray(nucleus_mask, dtype=bool)
    if nucleus_mask.shape != (height, width):
        raise ValueError(f"nucleus_mask shape {nucleus_mask.shape} != image shape {(height, width)}")
    stroma_mask = ~nucleus_mask

    row = _compartment_metrics(hematoxylin, eosin, nucleus_mask, compartment="nuc")
    row.update(_compartment_metrics(hematoxylin, eosin, stroma_mask, compartment="stroma"))
    return row
