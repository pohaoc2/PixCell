"""Stage 4 archetype matching utilities."""
from __future__ import annotations

import numpy as np


def _pairwise_l2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return pairwise Euclidean distances between rows of ``a`` and ``b``."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)


def assign_to_archetype(
    gn_embeddings: np.ndarray,
    centroids: np.ndarray,
) -> np.ndarray:
    """Assign each generated embedding to its nearest archetype centroid."""
    dists = _pairwise_l2(gn_embeddings, centroids)
    return dists.argmin(axis=1).astype(np.int64, copy=False)


def find_best_params(
    gn_embeddings: np.ndarray,
    centroids: np.ndarray,
    param_ids: list[str],
) -> dict[int, str]:
    """Return the closest parameter ID for each covered archetype."""
    gn_embeddings = np.asarray(gn_embeddings, dtype=np.float32)
    centroids = np.asarray(centroids, dtype=np.float32)
    if len(param_ids) != len(gn_embeddings):
        raise ValueError(
            "param_ids must match gn_embeddings length: "
            f"{len(param_ids)} != {len(gn_embeddings)}"
        )

    assignments = assign_to_archetype(gn_embeddings, centroids)
    best: dict[int, str] = {}
    for k in range(centroids.shape[0]):
        indices = np.where(assignments == k)[0]
        if indices.size == 0:
            continue
        dists = np.linalg.norm(gn_embeddings[indices] - centroids[k], axis=1)
        best[k] = param_ids[int(indices[int(dists.argmin())])]
    return best


def coverage_report(
    assignments: np.ndarray,
    k: int,
    param_ids: list[str],
) -> dict:
    """Summarize archetype coverage by assignment counts."""
    assignments = np.asarray(assignments)
    if len(param_ids) != len(assignments):
        raise ValueError(
            "param_ids must match assignments length: "
            f"{len(param_ids)} != {len(assignments)}"
        )
    counts = {int(i): int((assignments == i).sum()) for i in range(k)}
    uncovered = [i for i, count in counts.items() if count == 0]
    covered = [i for i in range(k) if i not in uncovered]
    return {
        "counts": counts,
        "covered": covered,
        "uncovered": uncovered,
    }
