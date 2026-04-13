from __future__ import annotations

import numpy as np
from PIL import Image

RGB_FROM_HED = np.array(
    [
        [0.65, 0.70, 0.29],
        [0.07, 0.99, 0.11],
        [0.27, 0.57, 0.78],
    ],
    dtype=np.float64,
)
HED_FROM_RGB = np.linalg.inv(RGB_FROM_HED)


def rgb_to_hed(image: Image.Image) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float64) / 255.0
    arr = np.clip(arr, 1e-6, 1.0)
    optical_density = -np.log(arr)
    return optical_density @ HED_FROM_RGB.T


def tissue_mask_from_rgb(image: Image.Image) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return np.mean(arr, axis=2) < 0.95


def masked_mean_std(values: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    data = values[np.asarray(mask, dtype=bool)]
    if data.size == 0:
        data = values.reshape(-1)
    return float(np.mean(data)), float(np.std(data))
