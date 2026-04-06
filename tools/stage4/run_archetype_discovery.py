"""CLI for stage 4 archetype discovery on a target CRC patient."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.stage4.archetype_discovery import fit_archetypes, load_patient_embeddings, sweep_k


def _auto_select_k(scores: dict[int, float]) -> int:
    """Pick the highest silhouette score, ignoring NaNs."""
    valid = {k: v for k, v in scores.items() if np.isfinite(v)}
    if not valid:
        raise RuntimeError("No finite silhouette scores were produced")
    return max(valid, key=valid.get)


def _project_to_2d(embeddings: np.ndarray, seed: int) -> np.ndarray:
    """Compute a 2D projection for visualization, preferring UMAP when available."""
    try:
        import umap

        reducer = umap.UMAP(n_components=2, random_state=seed)
        return reducer.fit_transform(embeddings)
    except Exception:
        data = np.asarray(embeddings, dtype=np.float32)
        if data.ndim != 2:
            raise ValueError(f"expected 2D embeddings, got shape {data.shape}")
        if data.shape[1] == 1:
            return np.c_[data[:, 0], np.zeros(len(data), dtype=np.float32)]
        data = data - data.mean(axis=0, keepdims=True)
        if data.shape[0] < 2:
            return np.zeros((data.shape[0], 2), dtype=np.float32)
        _, _, vt = np.linalg.svd(data, full_matrices=False)
        coords = data @ vt[:2].T
        if coords.shape[1] == 1:
            coords = np.c_[coords[:, 0], np.zeros(len(coords), dtype=np.float32)]
        return coords.astype(np.float32, copy=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 4 archetype discovery")
    parser.add_argument("--features-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--k-min", type=int, default=3)
    parser.add_argument("--k-max", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    embeddings, tile_ids = load_patient_embeddings(args.features_dir)
    print(f"Loaded {len(tile_ids)} tiles from {args.features_dir} with shape {embeddings.shape}")

    if args.k is None:
        k_min = max(2, int(args.k_min))
        k_max = min(int(args.k_max), embeddings.shape[0] - 1)
        if k_max < k_min:
            raise ValueError(
                f"Invalid K sweep range: k_min={k_min}, k_max={k_max}, n_tiles={len(tile_ids)}"
            )
        scores = sweep_k(embeddings, range(k_min, k_max + 1), seed=args.seed)
        for k, score in sorted(scores.items()):
            print(f"  K={k}: silhouette={score:.4f}")
        selected_k = _auto_select_k(scores)
        print(f"Auto-selected K={selected_k}")
    else:
        if args.k < 2:
            raise ValueError("--k must be at least 2")
        if args.k >= embeddings.shape[0]:
            raise ValueError("--k must be smaller than the number of tiles")
        scores = {}
        selected_k = int(args.k)
        print(f"Using user-specified K={selected_k}")

    result = fit_archetypes(embeddings, k=selected_k, seed=args.seed)
    medoid_tile_ids = [tile_ids[idx] for idx in result["medoid_indices"]]
    umap_coords = _project_to_2d(embeddings, seed=args.seed)

    np.save(args.output_dir / "centroids.npy", result["centroids"])
    np.save(args.output_dir / "umap_raw.npy", umap_coords.astype(np.float32, copy=False))

    payload = {
        "k": selected_k,
        "silhouette_scores": {str(k): float(v) for k, v in scores.items()},
        "tile_ids": tile_ids,
        "labels": result["labels"].tolist(),
        "medoid_indices": result["medoid_indices"],
        "medoid_tile_ids": medoid_tile_ids,
        "centroids_path": str(args.output_dir / "centroids.npy"),
        "umap_path": str(args.output_dir / "umap_raw.npy"),
        "features_dir": str(args.features_dir),
    }
    with open(args.output_dir / "archetypes.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved archetype outputs to {args.output_dir}")
    for k, medoid in enumerate(medoid_tile_ids):
        cluster_size = int((result["labels"] == k).sum())
        print(f"  Archetype {k}: medoid={medoid}, tiles={cluster_size}")


if __name__ == "__main__":
    main()
