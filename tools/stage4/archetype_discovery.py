"""Stage 4 archetype discovery utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np


def _validate_embeddings(embeddings: np.ndarray) -> np.ndarray:
    arr = np.asarray(embeddings)
    if arr.ndim != 2:
        raise ValueError(f"expected a 2D embedding matrix, got shape {arr.shape}")
    if arr.shape[0] == 0:
        raise ValueError("embeddings array is empty")
    return arr.astype(np.float32, copy=False)


def _squared_distances(data: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Return pairwise squared Euclidean distances with shape [N, K]."""
    data_norm = np.sum(data * data, axis=1, keepdims=True)
    centroid_norm = np.sum(centroids * centroids, axis=1, keepdims=True).T
    dists = data_norm + centroid_norm - 2.0 * data @ centroids.T
    return np.maximum(dists, 0.0)


def _kmeans_pp_init(data: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Initialize centroids using k-means++."""
    rng = np.random.default_rng(seed)
    n_samples = data.shape[0]

    first_idx = int(rng.integers(0, n_samples))
    chosen = [first_idx]
    centroids = [data[first_idx]]

    for _ in range(1, k):
        dists = _squared_distances(data, np.stack(centroids, axis=0)).min(axis=1)
        if np.allclose(dists.sum(), 0.0):
            available = [idx for idx in range(n_samples) if idx not in chosen]
            next_idx = int(rng.choice(available)) if available else int(rng.integers(0, n_samples))
        else:
            probs = dists / dists.sum()
            next_idx = int(rng.choice(n_samples, p=probs))
        chosen.append(next_idx)
        centroids.append(data[next_idx])

    return np.stack(centroids, axis=0).astype(np.float32, copy=False)


def _run_kmeans(
    data: np.ndarray,
    k: int,
    seed: int,
    max_iter: int = 100,
    tol: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """Small NumPy k-means implementation for the stage-4 analysis workloads."""
    centroids = _kmeans_pp_init(data, k, seed)
    labels = np.zeros(data.shape[0], dtype=np.int64)

    for _ in range(max_iter):
        dists = _squared_distances(data, centroids)
        new_labels = dists.argmin(axis=1).astype(np.int64, copy=False)
        new_centroids = centroids.copy()

        for cluster_id in range(k):
            members = np.flatnonzero(new_labels == cluster_id)
            if members.size == 0:
                farthest_idx = int(dists.min(axis=1).argmax())
                new_centroids[cluster_id] = data[farthest_idx]
                new_labels[farthest_idx] = cluster_id
                continue
            new_centroids[cluster_id] = data[members].mean(axis=0)

        shift = np.linalg.norm(new_centroids - centroids, axis=1).max()
        centroids = new_centroids
        labels = new_labels
        if shift <= tol:
            break

    return centroids.astype(np.float32, copy=False), labels


def _silhouette_score(data: np.ndarray, labels: np.ndarray) -> float:
    """Compute a mean silhouette score for the clustering."""
    unique_labels = np.unique(labels)
    if unique_labels.size < 2 or unique_labels.size >= data.shape[0]:
        return float("nan")

    # Use a deterministic subsample for larger inputs so the score stays tractable
    # without needing scikit-learn's vectorized implementation.
    max_points = 512
    if data.shape[0] > max_points:
        rng = np.random.default_rng(42)
        sample_idx = np.sort(rng.choice(data.shape[0], size=max_points, replace=False))
        data = data[sample_idx]
        labels = labels[sample_idx]
        unique_labels = np.unique(labels)
        if unique_labels.size < 2:
            return float("nan")

    data = np.asarray(data, dtype=np.float32, copy=False)
    norms = np.sum(data * data, axis=1, keepdims=True)
    dist2 = np.maximum(norms + norms.T - 2.0 * data @ data.T, 0.0)
    dist = np.sqrt(dist2, dtype=np.float32)
    scores = np.zeros(data.shape[0], dtype=np.float64)

    for i in range(data.shape[0]):
        same = labels == labels[i]
        same[i] = False
        a = float(dist[i, same].mean()) if same.any() else 0.0

        b = float("inf")
        for label in unique_labels:
            if label == labels[i]:
                continue
            other = labels == label
            if other.any():
                b = min(b, float(dist[i, other].mean()))

        denom = max(a, b)
        scores[i] = 0.0 if not np.isfinite(denom) or denom == 0.0 else (b - a) / denom

    return float(scores.mean())


def _sort_centroids(centroids: np.ndarray) -> np.ndarray:
    """Return a permutation that sorts centroids lexicographically."""
    if centroids.ndim != 2:
        raise ValueError(f"expected 2D centroids, got shape {centroids.shape}")
    keys = tuple(centroids[:, i] for i in reversed(range(centroids.shape[1])))
    return np.lexsort(keys)


def _remap_labels(labels: np.ndarray, order: np.ndarray) -> np.ndarray:
    """Remap cluster labels into the requested cluster order."""
    remap = {int(old): int(new) for new, old in enumerate(order)}
    return np.asarray([remap[int(label)] for label in labels], dtype=np.int64)


def load_patient_embeddings(features_dir: Path | str) -> tuple[np.ndarray, list[str]]:
    """Load all ``*_uni.npy`` files in ``features_dir``."""
    features_dir = Path(features_dir)
    paths = sorted(features_dir.glob("*_uni.npy"), key=lambda p: p.name)
    if not paths:
        raise FileNotFoundError(f"No *_uni.npy files found in {features_dir}")

    tile_ids: list[str] = []
    embeddings: list[np.ndarray] = []
    expected_shape: tuple[int, ...] | None = None

    for path in paths:
        emb = np.asarray(np.load(path))
        if emb.ndim != 1:
            raise ValueError(f"expected 1D UNI embedding in {path}, got shape {emb.shape}")
        if expected_shape is None:
            expected_shape = emb.shape
        elif emb.shape != expected_shape:
            raise ValueError(
                f"embedding dimensionality mismatch in {path}: "
                f"expected {expected_shape}, got {emb.shape}"
            )
        tile_ids.append(path.name[: -len("_uni.npy")])
        embeddings.append(emb.astype(np.float32, copy=False))

    return np.stack(embeddings, axis=0), tile_ids


def sweep_k(
    embeddings: np.ndarray,
    k_range: Iterable[int] = range(3, 8),
    seed: int = 42,
) -> dict[int, float]:
    """Return silhouette scores for each ``k`` in ``k_range``."""
    arr = _validate_embeddings(embeddings)
    scores: dict[int, float] = {}
    for k in k_range:
        k = int(k)
        if k < 2 or k >= arr.shape[0]:
            scores[k] = float("nan")
            continue
        _, labels = _run_kmeans(arr, k, seed)
        scores[k] = _silhouette_score(arr, labels)
    return scores


def select_medoids(
    embeddings: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
) -> list[int]:
    """Pick the sample nearest to each centroid within its assigned cluster."""
    emb = _validate_embeddings(embeddings)
    lab = np.asarray(labels)
    ctr = _validate_embeddings(centroids)
    if lab.ndim != 1:
        raise ValueError(f"expected 1D labels, got shape {lab.shape}")
    if emb.shape[0] != lab.shape[0]:
        raise ValueError(
            f"labels length {lab.shape[0]} does not match embeddings length {emb.shape[0]}"
        )
    if emb.shape[1] != ctr.shape[1]:
        raise ValueError(
            f"centroid dimensionality {ctr.shape[1]} does not match embeddings {emb.shape[1]}"
        )

    medoids: list[int] = []
    for k in range(ctr.shape[0]):
        indices = np.flatnonzero(lab == k)
        if indices.size == 0:
            raise ValueError(f"cluster {k} is empty; cannot choose medoid")
        dists = np.linalg.norm(emb[indices] - ctr[k], axis=1)
        medoids.append(int(indices[int(dists.argmin())]))
    return medoids


def fit_archetypes(
    embeddings: np.ndarray,
    k: int,
    seed: int = 42,
) -> dict:
    """Fit k-means and return sorted centroids, labels, and medoid indices."""
    arr = _validate_embeddings(embeddings)
    if k <= 0:
        raise ValueError("k must be positive")
    if k > arr.shape[0]:
        raise ValueError(f"k={k} cannot exceed number of samples {arr.shape[0]}")

    centroids, raw_labels = _run_kmeans(arr, k, seed)
    order = _sort_centroids(centroids)
    centroids = centroids[order]
    labels = _remap_labels(raw_labels, order)
    medoids = select_medoids(arr, labels, centroids)
    return {
        "centroids": centroids.astype(np.float32, copy=False),
        "labels": labels.astype(np.int64, copy=False),
        "medoid_indices": medoids,
    }
