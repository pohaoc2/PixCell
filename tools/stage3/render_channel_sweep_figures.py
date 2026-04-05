"""Render channel sweep figures from a cached generation manifest.

Usage:
    python tools/stage3/render_channel_sweep_figures.py \
        --cache-dir inference_output/channel_sweep/cache \
        --out inference_output/channel_sweep
"""
from __future__ import annotations

import argparse
from pathlib import Path

from tools.stage3.channel_sweep_figures import render_figures_from_cache


def main() -> None:
    parser = argparse.ArgumentParser(description="Render channel sweep figures from cached PNGs")
    parser.add_argument("--cache-dir", required=True, help="Directory containing channel_sweep cache manifest.json")
    parser.add_argument("--out", required=True, help="Output directory for rendered figures")
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=["1", "2", "3"],
        default=["1", "2", "3"],
        help="Subset of experiments to render",
    )
    args = parser.parse_args()

    render_figures_from_cache(
        cache_dir=Path(args.cache_dir),
        out_dir=Path(args.out),
        experiments=args.experiments,
    )


if __name__ == "__main__":
    main()
