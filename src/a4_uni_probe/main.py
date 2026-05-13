"""CLI entrypoint for the a4 UNI probe pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = ROOT / "data" / "orion-crc33"
DEFAULT_FEATURES_DIR = DEFAULT_DATA_ROOT / "features"
DEFAULT_EXP_CHANNELS_DIR = DEFAULT_DATA_ROOT / "exp_channels"
DEFAULT_HE_DIR = DEFAULT_DATA_ROOT / "he"
DEFAULT_OUT_DIR = ROOT / "src" / "a4_uni_probe" / "out"
DEFAULT_CHECKPOINT_DIR = ROOT / "checkpoints" / "concat_95470_0" / "checkpoints" / "step_0002600"
DEFAULT_CONFIG_PATH = ROOT / "configs" / "config_controlnet_exp_a1_concat.py"
DEFAULT_CELLVIT_REAL_DIR = DEFAULT_OUT_DIR / "cellvit_real"
DEFAULT_FULL_NULL_ROOT = ROOT / "src" / "a2_decomposition" / "out" / "generated"

DEFAULT_SEED = 42
DEFAULT_NUM_STEPS = 20
DEFAULT_GUIDANCE_SCALE = 2.5
DEFAULT_K_SWEEP_TILES = 50
DEFAULT_ALPHAS = (-2.0, -1.0, 0.0, 1.0, 2.0)
DEFAULT_TILE_SHARD_COUNT = 1
DEFAULT_TILE_SHARD_INDEX = 0
DEFAULT_CV_FOLDS = 5
DEFAULT_SPATIAL_BUCKET_PX = 4096


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="a4 UNI probe / sweep / null pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    p_probe = sub.add_parser("probe", help="Stage 1 linear probes")
    p_probe.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p_probe.add_argument("--features-dir", type=Path, default=DEFAULT_FEATURES_DIR)
    p_probe.add_argument("--exp-channels-dir", type=Path, default=DEFAULT_EXP_CHANNELS_DIR)
    p_probe.add_argument("--cellvit-real-dir", type=Path, default=DEFAULT_CELLVIT_REAL_DIR)
    p_probe.add_argument("--he-dir", type=Path, default=DEFAULT_HE_DIR)
    p_probe.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p_probe.add_argument("--cv-folds", type=int, default=DEFAULT_CV_FOLDS)
    p_probe.add_argument("--bucket-px", type=int, default=DEFAULT_SPATIAL_BUCKET_PX)

    p_sweep = sub.add_parser("sweep", help="Stage 2 probe-direction sweep")
    p_sweep.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p_sweep.add_argument("--features-dir", type=Path, default=DEFAULT_FEATURES_DIR)
    p_sweep.add_argument("--exp-channels-dir", type=Path, default=DEFAULT_EXP_CHANNELS_DIR)
    p_sweep.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p_sweep.add_argument("--k-tiles", type=int, default=DEFAULT_K_SWEEP_TILES)
    p_sweep.add_argument("--alphas", type=float, nargs="+", default=list(DEFAULT_ALPHAS))
    p_sweep.add_argument("--top-k-attrs", type=int, default=4)
    p_sweep.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p_sweep.add_argument("--num-steps", type=int, default=DEFAULT_NUM_STEPS)
    p_sweep.add_argument("--guidance-scale", type=float, default=DEFAULT_GUIDANCE_SCALE)
    p_sweep.add_argument("--fixed-tile-ids", type=Path, default=None)
    p_sweep.add_argument("--attr-pool", choices=["morphology", "appearance"], default="morphology")
    p_sweep.add_argument("--tile-shard-count", type=int, default=DEFAULT_TILE_SHARD_COUNT)
    p_sweep.add_argument("--tile-shard-index", type=int, default=DEFAULT_TILE_SHARD_INDEX)
    p_sweep.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    p_sweep.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)

    p_null = sub.add_parser("null", help="Stage 3 subspace nulling")
    p_null.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p_null.add_argument("--features-dir", type=Path, default=DEFAULT_FEATURES_DIR)
    p_null.add_argument("--exp-channels-dir", type=Path, default=DEFAULT_EXP_CHANNELS_DIR)
    p_null.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p_null.add_argument("--k-tiles", type=int, default=DEFAULT_K_SWEEP_TILES)
    p_null.add_argument("--top-k-attrs", type=int, default=4)
    p_null.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p_null.add_argument("--num-steps", type=int, default=DEFAULT_NUM_STEPS)
    p_null.add_argument("--guidance-scale", type=float, default=DEFAULT_GUIDANCE_SCALE)
    p_null.add_argument("--fixed-tile-ids", type=Path, default=None)
    p_null.add_argument("--attr-pool", choices=["morphology", "appearance"], default="morphology")
    p_null.add_argument("--tile-shard-count", type=int, default=DEFAULT_TILE_SHARD_COUNT)
    p_null.add_argument("--tile-shard-index", type=int, default=DEFAULT_TILE_SHARD_INDEX)
    p_null.add_argument("--full-null-root", type=Path, default=DEFAULT_FULL_NULL_ROOT)
    p_null.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    p_null.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)

    p_fig = sub.add_parser("figures", help="Render Panel A-E")
    p_fig.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)

    p_app = sub.add_parser("appearance", help="Add stain and texture metrics to existing sweep/null outputs")
    p_app.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p_app.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "probe":
        from src.a4_uni_probe.probe import run_probe

        run_probe(args)
    elif args.command == "sweep":
        from src.a4_uni_probe.edit import run_sweep

        run_sweep(args)
    elif args.command == "null":
        from src.a4_uni_probe.edit import run_null

        run_null(args)
    elif args.command == "figures":
        from src.a4_uni_probe.figures import render_all

        render_all(args.out_dir)
    elif args.command == "appearance":
        from src.a4_uni_probe.appearance_metrics import run_appearance

        run_appearance(args)
    else:  # pragma: no cover
        parser.error(f"unknown command: {args.command}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
