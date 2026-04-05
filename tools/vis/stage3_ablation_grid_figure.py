#!/usr/bin/env python3
"""Stable CLI wrapper for the static Stage 3 ablation grid.

The implementation lives in ``tools.stage3.ablation_grid_figure``; this path is
kept as the public entry point used by the docs.
"""

from tools.stage3.ablation_grid_figure import main

__all__ = ["main"]


if __name__ == "__main__":
    main()
