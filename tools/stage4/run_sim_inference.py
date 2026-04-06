"""CLI for batch TME-only inference over simulation outputs."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STAGE3_INFERENCE = ROOT / "stage3_inference.py"


def _list_sim_ids(sim_channels_dir: Path) -> list[str]:
    """Discover sim IDs from the cell-mask channel directory."""
    for name in ("cell_mask", "cell_masks"):
        mask_dir = sim_channels_dir / name
        if mask_dir.is_dir():
            sim_ids = sorted(p.stem for p in mask_dir.glob("*.png"))
            if sim_ids:
                return sim_ids
    raise FileNotFoundError(
        f"No cell_mask/ or cell_masks/ directory with PNGs found under {sim_channels_dir}"
    )


def _default_device() -> str:
    """Choose a sensible default device without requiring torch at import time."""
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch Stage 3 TME-only inference for simulations")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--sim-channels-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default=None)
    parser.add_argument("--guidance-scale", type=float, default=2.5)
    parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    device = args.device or _default_device()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sim_ids = _list_sim_ids(args.sim_channels_dir)
    if args.limit is not None:
        sim_ids = sim_ids[: max(0, args.limit)]

    print(f"Found {len(sim_ids)} simulations in {args.sim_channels_dir}")
    for i, sim_id in enumerate(sim_ids, start=1):
        out_path = args.output_dir / sim_id / "generated_he.png"
        if out_path.exists() and not args.overwrite:
            print(f"[{i}/{len(sim_ids)}] {sim_id}: skip (exists)")
            continue

        out_path.parent.mkdir(parents=True, exist_ok=True)
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
        print(f"[{i}/{len(sim_ids)}] {sim_id} -> {out_path}")
        result = subprocess.run(cmd, cwd=ROOT)
        if result.returncode != 0:
            raise subprocess.CalledProcessError(result.returncode, cmd)

    print(f"Generated H&E saved under {args.output_dir}")


if __name__ == "__main__":
    main()
