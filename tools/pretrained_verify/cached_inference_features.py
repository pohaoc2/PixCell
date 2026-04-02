"""Helpers for caching inference features on disk as `.npy` files."""

from pathlib import Path

import numpy as np


def load_or_compute_npy(cache_path, compute_fn):
    """
    Load a cached numpy feature if it exists, otherwise compute and save it.
    """
    cache_path = Path(cache_path)
    if cache_path.exists():
        return np.load(cache_path)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    value = np.asarray(compute_fn())
    np.save(cache_path, value)
    return value
