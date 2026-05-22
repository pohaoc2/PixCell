"""Run the spatial MLP probe for each encoder at its native spatial grid."""

from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

ENCODERS = [
    {
        "name": "uni",
        "features_dir": ROOT / "data/orion-crc33/features",
        "suffix": "_uni_tokens.npy",
        "targets": ROOT / "src/a1_mask_targets_spatial/out/mask_targets_T1_spatial.npy",
        "tile_ids": ROOT / "src/a1_mask_targets_spatial/out/tile_ids.txt",
        "target_names": ROOT / "src/a1_mask_targets_spatial/out/target_names_T1_spatial.json",
        "out_dir": ROOT / "src/a1_probe_mlp_spatial/out/uni_16",
    },
    {
        "name": "virchow2",
        "features_dir": ROOT / "data/orion-crc33/features_patches/virchow2",
        "suffix": "_patches.npy",
        "targets": ROOT / "src/a1_mask_targets_spatial/out/mask_targets_T1_spatial.npy",
        "tile_ids": ROOT / "src/a1_mask_targets_spatial/out/tile_ids.txt",
        "target_names": ROOT / "src/a1_mask_targets_spatial/out/target_names_T1_spatial.json",
        "out_dir": ROOT / "src/a1_probe_mlp_spatial/out/virchow2_16",
    },
    {
        "name": "ctranspath",
        "features_dir": ROOT / "data/orion-crc33/features_patches/ctranspath",
        "suffix": "_patches.npy",
        "targets": ROOT / "src/a1_mask_targets_spatial/out_grid_07/mask_targets_T1_spatial.npy",
        "tile_ids": ROOT / "src/a1_mask_targets_spatial/out_grid_07/tile_ids.txt",
        "target_names": ROOT / "src/a1_mask_targets_spatial/out_grid_07/target_names_T1_spatial.json",
        "out_dir": ROOT / "src/a1_probe_mlp_spatial/out/ctranspath_07",
    },
    {
        "name": "resnet50",
        "features_dir": ROOT / "data/orion-crc33/features_patches/resnet50",
        "suffix": "_patches.npy",
        "targets": ROOT / "src/a1_mask_targets_spatial/out_grid_07/mask_targets_T1_spatial.npy",
        "tile_ids": ROOT / "src/a1_mask_targets_spatial/out_grid_07/tile_ids.txt",
        "target_names": ROOT / "src/a1_mask_targets_spatial/out_grid_07/target_names_T1_spatial.json",
        "out_dir": ROOT / "src/a1_probe_mlp_spatial/out/resnet50_07",
    },
]


def _has_feature_cache(features_dir: Path, suffix: str) -> bool:
    return features_dir.exists() and any(features_dir.glob(f"*{suffix}"))


def main() -> int:
    for entry in ENCODERS:
        features_dir = Path(entry["features_dir"])
        suffix = str(entry["suffix"])
        if not _has_feature_cache(features_dir, suffix):
            print(f"SKIP {entry['name']}: no cached features at {features_dir}")
            continue

        cmd = [
            "python",
            "-m",
            "src.a1_probe_mlp_spatial.main",
            "--features-dir",
            str(features_dir),
            "--targets-path",
            str(entry["targets"]),
            "--tile-ids-path",
            str(entry["tile_ids"]),
            "--target-names-path",
            str(entry["target_names"]),
            "--out-dir",
            str(entry["out_dir"]),
            "--feature-suffix",
            suffix,
            "--n-tiles",
            "800",
            "--batch-size",
            "2048",
            "--max-train-rows",
            "50000",
            "--n-jobs",
            "2",
        ]
        print(f"=== {entry['name']} ===")
        subprocess.run(cmd, check=True, cwd=ROOT)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())