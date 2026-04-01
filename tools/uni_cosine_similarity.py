"""
UNI feature cosine similarity (reference vs generated H&E).

Used for ablation figures and evaluation; same normalization as ``run_evaluation._cosine_sim``.
"""
from __future__ import annotations

import numpy as np


def cosine_similarity_uni(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity between two UNI embedding vectors (same length).

    Values lie in approximately [-1, 1] for typical embeddings.
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    denom = (np.linalg.norm(a) + 1e-8) * (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b) / denom)


def flatten_uni_npy(arr: np.ndarray) -> np.ndarray:
    """Load a cached UNI .npy into a 1D float vector."""
    return np.asarray(arr, dtype=np.float64).ravel()
