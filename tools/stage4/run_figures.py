"""CLI for generating the stage 4 summary figures."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.stage4.archetype_discovery import load_patient_embeddings
from tools.stage4.figures import (
    save_fig1_archetype_umap,
    save_fig2_umap_overlay,
    save_fig3_side_by_side,
    save_fig4_param_space,
)


def _project_joint_embeddings(embeddings: np.ndarray, seed: int = 42) -> np.ndarray:
    """Project embeddings to 2D for comparison plots."""
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


def _resolve_he_dir(patient_dir: Path) -> Path:
    """Prefer a dedicated he/ directory when available."""
    if (patient_dir / "he").is_dir():
        return patient_dir / "he"
    return patient_dir


def _resolve_features_dir(patient_dir: Path) -> Path:
    """Prefer a dedicated features/ directory when available."""
    if (patient_dir / "features").is_dir():
        return patient_dir / "features"
    return patient_dir


def _load_param_names(args: argparse.Namespace, param_vectors: np.ndarray) -> list[str]:
    """Resolve parameter names from CLI values or a newline-delimited text file."""
    if args.param_names:
        names = list(args.param_names)
    elif args.param_names_file is not None:
        names = [
            line.strip()
            for line in args.param_names_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        names = [f"param_{idx}" for idx in range(param_vectors.shape[1])]

    if len(names) != param_vectors.shape[1]:
        raise ValueError(
            "Parameter name count must match param_vectors columns: "
            f"{len(names)} != {param_vectors.shape[1]}"
        )
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate stage 4 figures")
    parser.add_argument("--archetypes-dir", required=True, type=Path)
    parser.add_argument("--matching-json", required=True, type=Path)
    parser.add_argument("--gn-features-dir", required=True, type=Path)
    parser.add_argument("--style-results-dir", required=True, type=Path)
    parser.add_argument("--target-patient-dir", required=True, type=Path)
    parser.add_argument("--param-vectors", required=True, type=Path)
    parser.add_argument("--param-names", nargs="+", default=None)
    parser.add_argument("--param-names-file", type=Path, default=None)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.matching_json, encoding="utf-8") as f:
        matching = json.load(f)
    with open(args.archetypes_dir / "archetypes.json", encoding="utf-8") as f:
        archetypes = json.load(f)

    k = int(matching["k"])
    labels = np.asarray(archetypes["labels"])
    medoid_indices = list(archetypes["medoid_indices"])
    tile_ids = list(archetypes["tile_ids"])
    assignments = np.asarray(matching["assignments"])

    crc_umap_path = args.archetypes_dir / "umap_raw.npy"
    if crc_umap_path.exists():
        crc_umap = np.load(crc_umap_path)
    else:
        crc_embs, _ = load_patient_embeddings(_resolve_features_dir(args.target_patient_dir))
        crc_umap = _project_joint_embeddings(crc_embs, seed=args.seed)

    he_dir = _resolve_he_dir(args.target_patient_dir)
    save_fig1_archetype_umap(
        umap_coords=crc_umap,
        labels=labels,
        medoid_indices=medoid_indices,
        tile_ids=tile_ids,
        he_dir=he_dir,
        output_path=args.output_dir / "fig1_archetype_umap.png",
        k=k,
    )

    crc_embs, _ = load_patient_embeddings(_resolve_features_dir(args.target_patient_dir))
    gn_embs, _ = load_patient_embeddings(args.gn_features_dir)
    joint_umap = _project_joint_embeddings(np.vstack([crc_embs, gn_embs]), seed=args.seed)
    crc_joint = joint_umap[: len(crc_embs)]
    gn_joint = joint_umap[len(crc_embs) :]
    save_fig2_umap_overlay(
        crc_umap=crc_joint,
        gn_umap=gn_joint,
        gn_assignments=assignments,
        output_path=args.output_dir / "fig2_umap_overlay.png",
        k=k,
    )

    save_fig3_side_by_side(
        style_results_dir=args.style_results_dir,
        output_path=args.output_dir / "fig3_side_by_side.png",
        k=k,
    )

    param_vectors = np.load(args.param_vectors)
    if param_vectors.shape[0] != assignments.shape[0]:
        raise ValueError(
            "Parameter vector rows must match the number of generated assignments: "
            f"{param_vectors.shape[0]} != {assignments.shape[0]}"
        )
    param_names = _load_param_names(args, param_vectors)
    save_fig4_param_space(
        param_vectors=param_vectors,
        assignments=assignments,
        param_names=param_names,
        output_path=args.output_dir / "fig4_param_space.png",
        k=k,
    )

    print(f"Saved figures to {args.output_dir}")


if __name__ == "__main__":
    main()
