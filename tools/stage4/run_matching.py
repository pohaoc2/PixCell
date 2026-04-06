"""CLI for matching simulation embeddings to archetypes."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.stage4.archetype_discovery import load_patient_embeddings
from tools.stage4.matching import assign_to_archetype, coverage_report, find_best_params


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 4 archetype matching")
    parser.add_argument("--archetypes-dir", required=True, type=Path)
    parser.add_argument("--gn-features-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    archetypes_path = args.archetypes_dir / "archetypes.json"
    centroids_path = args.archetypes_dir / "centroids.npy"
    if not centroids_path.exists():
        raise FileNotFoundError(f"Missing centroids.npy at {centroids_path}")

    archetypes = {}
    if archetypes_path.exists():
        with open(archetypes_path, encoding="utf-8") as f:
            archetypes = json.load(f)
    centroids = np.load(centroids_path)
    k = int(archetypes.get("k", centroids.shape[0]))

    gn_embeddings, param_ids = load_patient_embeddings(args.gn_features_dir)
    print(f"Loaded {len(param_ids)} generated embeddings from {args.gn_features_dir}")

    assignments = assign_to_archetype(gn_embeddings, centroids)
    best_params = find_best_params(gn_embeddings, centroids, param_ids)
    report = coverage_report(assignments, k=k, param_ids=param_ids)

    medoid_tile_ids = archetypes.get("medoid_tile_ids", [])
    for idx in range(k):
        assigned = report["counts"][idx]
        best = best_params.get(idx, None)
        medoid = medoid_tile_ids[idx] if idx < len(medoid_tile_ids) else "unknown"
        print(f"  Archetype {idx}: assigned={assigned}, best_param={best or 'NONE'}, medoid={medoid}")
    if report["uncovered"]:
        print(f"Uncovered archetypes: {report['uncovered']}")
    else:
        print("All archetypes covered.")

    payload = {
        "k": k,
        "source_archetypes_dir": str(args.archetypes_dir),
        "gn_features_dir": str(args.gn_features_dir),
        "param_ids": param_ids,
        "assignments": assignments.tolist(),
        "best_params": {str(k_): v for k_, v in best_params.items()},
        "coverage": {
            "counts": {str(k_): v for k_, v in report["counts"].items()},
            "covered": report["covered"],
            "uncovered": report["uncovered"],
        },
        "medoid_tile_ids": medoid_tile_ids,
    }
    with open(args.output_dir / "matching.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved matching outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
