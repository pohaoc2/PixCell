"""CLI for style-conditioned stage 3 inference per matched archetype."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
STAGE3_INFERENCE = ROOT / "stage3_inference.py"
sys.path.insert(0, str(ROOT))

from tools.stage4.style_selection import collect_other_patient_embeddings, select_style_tiles


def _resolve_he_path(patient_dir: Path, tile_id: str) -> Path | None:
    """Find the H&E tile corresponding to ``tile_id`` in a patient directory."""
    candidates = [
        patient_dir / "he" / f"{tile_id}.png",
        patient_dir / "he_images" / f"{tile_id}.png",
        patient_dir / f"{tile_id}.png",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _default_device() -> str:
    """Choose a sensible default device without requiring torch at import time."""
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 4 style-conditioned inference")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--matching-json", required=True, type=Path)
    parser.add_argument("--archetypes-dir", required=True, type=Path)
    parser.add_argument("--sim-channels-dir", required=True, type=Path)
    parser.add_argument("--data-base-dir", required=True, type=Path)
    parser.add_argument("--target-patient-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default=None)
    parser.add_argument("--guidance-scale", type=float, default=2.5)
    parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    device = args.device or _default_device()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.matching_json, encoding="utf-8") as f:
        matching = json.load(f)
    centroids = np.load(args.archetypes_dir / "centroids.npy")
    k = int(matching["k"])
    best_params = {int(key): value for key, value in matching.get("best_params", {}).items()}
    medoid_tile_ids = list(matching.get("medoid_tile_ids", []))

    pool = collect_other_patient_embeddings(args.data_base_dir, args.target_patient_dir)
    if not pool:
        raise RuntimeError(
            f"No style patients found under {args.data_base_dir} excluding {args.target_patient_dir}"
        )
    style_tiles = select_style_tiles(centroids, pool)

    for arch_k in range(k):
        sim_id = best_params.get(arch_k)
        if sim_id is None:
            print(f"Archetype {arch_k}: uncovered, skipping")
            continue

        style_info = style_tiles[arch_k]
        style_patient_dir = Path(style_info["patient_dir"])
        style_tile_id = style_info["tile_id"]
        style_he_path = _resolve_he_path(style_patient_dir, style_tile_id)
        if style_he_path is None:
            print(f"Archetype {arch_k}: missing style H&E for {style_patient_dir} / {style_tile_id}")
            continue

        medoid_he_path = None
        if arch_k < len(medoid_tile_ids):
            medoid_he_path = _resolve_he_path(args.target_patient_dir, medoid_tile_ids[arch_k])

        out_dir = args.output_dir / f"archetype_{arch_k}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "generated_he.png"
        if out_path.exists() and not args.overwrite:
            print(f"Archetype {arch_k}: skip (exists)")
            continue

        if medoid_he_path is not None:
            shutil.copy2(medoid_he_path, out_dir / "medoid_he.png")
        shutil.copy2(style_he_path, out_dir / "style_tile.png")

        cmd = [
            sys.executable,
            str(STAGE3_INFERENCE),
            "--config",
            str(args.config),
            "--checkpoint-dir",
            str(args.checkpoint_dir),
            "--sim-channels-dir",
            str(args.sim_channels_dir),
            "--sim-id",
            sim_id,
            "--reference-he",
            str(style_he_path),
            "--output",
            str(out_path),
            "--device",
            str(device),
            "--guidance-scale",
            str(args.guidance_scale),
            "--num-steps",
            str(args.num_steps),
            "--seed",
            str(args.seed),
        ]
        print(
            f"Archetype {arch_k}: sim={sim_id}, "
            f"style={style_patient_dir.name}/{style_tile_id}, output={out_path}"
        )
        result = subprocess.run(cmd, cwd=ROOT)
        if result.returncode != 0:
            raise subprocess.CalledProcessError(result.returncode, cmd)

        metadata = {
            "archetype": arch_k,
            "best_param": sim_id,
            "style_patient_dir": str(style_patient_dir),
            "style_tile_id": style_tile_id,
            "style_he_path": str(style_he_path),
            "medoid_tile_id": medoid_tile_ids[arch_k] if arch_k < len(medoid_tile_ids) else None,
            "medoid_he_path": str(medoid_he_path) if medoid_he_path is not None else None,
        }
        with open(out_dir / "selection.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    print(f"Saved style-conditioned outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
