"""Stage 4 style tile selection utilities."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from tools.stage4.archetype_discovery import load_patient_embeddings


def _patient_feature_dir(candidate: Path) -> Path | None:
    """Return a usable feature directory for a patient candidate, if any."""
    if candidate.is_dir() and candidate.name != "features":
        feat_dir = candidate / "features"
        if feat_dir.is_dir():
            return feat_dir
        if any(candidate.glob("*_uni.npy")):
            return candidate
    return None


def collect_other_patient_embeddings(
    data_base_dir: Path | str,
    exclude_patient_dir: Path | str,
) -> dict[str, tuple[np.ndarray, list[str]]]:
    """Load UNI embeddings from all patient folders except ``exclude_patient_dir``."""
    data_base_dir = Path(data_base_dir)
    exclude = Path(exclude_patient_dir).resolve()
    pool: dict[str, tuple[np.ndarray, list[str]]] = {}

    if not data_base_dir.is_dir():
        raise FileNotFoundError(f"Data base directory not found: {data_base_dir}")

    for candidate in sorted(data_base_dir.iterdir()):
        feat_dir = _patient_feature_dir(candidate)
        if feat_dir is None:
            continue
        if candidate.resolve() == exclude or feat_dir.resolve() == exclude:
            continue
        try:
            embeddings, tile_ids = load_patient_embeddings(feat_dir)
        except FileNotFoundError:
            continue
        pool[str(candidate.resolve())] = (embeddings, tile_ids)
    return pool


def select_style_tiles(
    centroids: np.ndarray,
    pool: dict[str, tuple[np.ndarray, list[str]]],
) -> dict[int, dict[str, str]]:
    """Pick the nearest tile from a different patient for each archetype."""
    centroids = np.asarray(centroids, dtype=np.float32)
    if not pool:
        raise ValueError("Style pool is empty")

    result: dict[int, dict[str, str]] = {}
    for k, centroid in enumerate(centroids):
        best_dist = float("inf")
        best_entry: dict[str, str] | None = None
        for patient_dir, (embs, tile_ids) in pool.items():
            dists = np.linalg.norm(embs - centroid, axis=1)
            idx = int(dists.argmin())
            dist = float(dists[idx])
            if dist < best_dist:
                best_dist = dist
                best_entry = {
                    "patient_dir": patient_dir,
                    "tile_id": tile_ids[idx],
                }
        if best_entry is None:
            raise RuntimeError(f"Could not select a style tile for archetype {k}")
        result[int(k)] = best_entry
    return result
