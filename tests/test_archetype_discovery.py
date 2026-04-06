from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest


def _write_fake_embeddings(tmp_dir: Path, n: int = 20, dim: int = 1536) -> list[str]:
    rng = np.random.default_rng(0)
    tile_ids = [f"{i * 256}_{i * 256}" for i in range(n)]
    shuffled = list(reversed(tile_ids))
    for tid in shuffled:
        np.save(tmp_dir / f"{tid}_uni.npy", rng.standard_normal(dim).astype(np.float32))
    return tile_ids


def test_load_patient_embeddings_shape_and_sorting():
    from tools.stage4.archetype_discovery import load_patient_embeddings

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        expected_ids = _write_fake_embeddings(tmp_path, n=10)

        embs, ids = load_patient_embeddings(tmp_path)

        assert embs.shape == (10, 1536)
        assert ids == sorted(expected_ids)
        assert embs.dtype == np.float32


def test_load_patient_embeddings_missing_raises():
    from tools.stage4.archetype_discovery import load_patient_embeddings

    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(FileNotFoundError):
            load_patient_embeddings(Path(tmp))


def test_sweep_k_returns_scores_for_each_k():
    from tools.stage4.archetype_discovery import sweep_k

    rng = np.random.default_rng(42)
    embeddings = rng.standard_normal((50, 8)).astype(np.float32)
    scores = sweep_k(embeddings, k_range=range(2, 5))

    assert set(scores.keys()) == {2, 3, 4}
    assert all(isinstance(v, float) for v in scores.values())


def test_fit_archetypes_shapes():
    from tools.stage4.archetype_discovery import fit_archetypes

    rng = np.random.default_rng(0)
    embeddings = rng.standard_normal((30, 8)).astype(np.float32)
    result = fit_archetypes(embeddings, k=3)

    assert result["centroids"].shape == (3, 8)
    assert result["labels"].shape == (30,)
    assert len(result["medoid_indices"]) == 3
    assert all(0 <= idx < 30 for idx in result["medoid_indices"])


def test_fit_archetypes_medoids_are_nearest_within_cluster():
    from tools.stage4.archetype_discovery import fit_archetypes

    rng = np.random.default_rng(1)
    c1 = rng.standard_normal((10, 4)) + np.array([10, 0, 0, 0])
    c2 = rng.standard_normal((10, 4)) + np.array([0, 10, 0, 0])
    c3 = rng.standard_normal((10, 4)) + np.array([0, 0, 10, 0])
    embeddings = np.vstack([c1, c2, c3]).astype(np.float32)

    result = fit_archetypes(embeddings, k=3)

    for cluster_id, medoid_idx in enumerate(result["medoid_indices"]):
        cluster_members = np.flatnonzero(result["labels"] == cluster_id)
        assert medoid_idx in cluster_members
        dists = np.linalg.norm(embeddings[cluster_members] - result["centroids"][cluster_id], axis=1)
        assert medoid_idx == int(cluster_members[int(dists.argmin())])
