#!/usr/bin/env python3
"""Stable CLI wrapper for single-tile Stage 3 visualization outputs.

The implementation lives in ``tools.stage3.generate_tile_vis``; this path is
kept as the public entry point used by the docs.
"""

from tools.stage3.generate_tile_vis import main, run_vis_suite

__all__ = ["main", "run_vis_suite"]


if __name__ == "__main__":
    main()
