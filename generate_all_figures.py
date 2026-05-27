"""Generate all paper figures in one pass.

Covers three figure groups, run sequentially with explicit GC between each to
keep peak RSS well within the machine's RAM budget:

  Group 1 — paper_figures (figs 01–09 + SI)
  Group 2 — channel utility spatial (fig 09b)
  Group 3 — a4 UNI probe (probe_delta, specificity, sweep grids)

Launch with a memory cap to prevent OOM kills on the 32 GB (no-swap) machine:

    prlimit --as=24000000000 -- python generate_all_figures.py

Or via systemd scope:

    systemd-run --user --scope -p MemoryMax=24G -p MemorySwapMax=0 -- \
        python generate_all_figures.py
"""
from __future__ import annotations

import gc
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent


def _cleanup() -> None:
    plt.close("all")
    gc.collect()


def _run_paper_figures() -> None:
    print("=== Group 1: paper figures (01-09 + SI) ===")
    from src.paper_figures.main import main as _main
    _main()
    _cleanup()


def _run_channel_utility_spatial() -> None:
    print("=== Group 2: channel utility spatial (09b) ===")
    from src.paper_figures.fig_channel_utility_spatial import (
        DEFAULT_LAYOUT_CSV,
        DEFAULT_LOO_CSV,
        DEFAULT_OUT_PNG,
        DEFAULT_SPATIAL_CSV,
        save_channel_utility_spatial_figure,
    )
    if not DEFAULT_SPATIAL_CSV.is_file():
        print(f"Skipping 09b: missing {DEFAULT_SPATIAL_CSV}")
        return
    if not DEFAULT_LOO_CSV.is_file():
        print(f"Skipping 09b: missing {DEFAULT_LOO_CSV}")
        return
    try:
        save_channel_utility_spatial_figure(
            out_png=DEFAULT_OUT_PNG,
            spatial_csv=DEFAULT_SPATIAL_CSV,
            loo_csv=DEFAULT_LOO_CSV,
            layout_csv=DEFAULT_LAYOUT_CSV,
        )
        print(f"Saved {DEFAULT_OUT_PNG}")
    except Exception as exc:
        print(f"09b failed: {exc}", file=sys.stderr)
    finally:
        _cleanup()


def _run_a4_probe_figures() -> None:
    print("=== Group 3: uni_probe figures ===")
    out_dir = ROOT / "inference_output" / "a1_concat" / "a4_uni_probe"
    dest_dir = ROOT / "figures" / "pngs_updated" / "individual" / "uni_probe"
    concat_dir = ROOT / "figures" / "pngs_updated" / "concat"

    probe_csv = out_dir / "probe_results.csv"
    if not probe_csv.is_file():
        print(f"Skipping uni_probe: missing {probe_csv}")
        return

    from src.a4_uni_probe.figures import render_pngs_updated
    try:
        outputs = render_pngs_updated(out_dir, dest_dir, concat_dir=concat_dir)
        print(f"Saved {len(outputs)} uni_probe figures (individual → {dest_dir}, combined → {concat_dir})")
    except Exception as exc:
        print(f"uni_probe failed: {exc}", file=sys.stderr)
    finally:
        _cleanup()


def main() -> None:
    from src.paper_figures.style import apply_style
    apply_style()

    _run_paper_figures()
    _run_channel_utility_spatial()
    _run_a4_probe_figures()

    print("=== All figure groups complete ===")


if __name__ == "__main__":
    main()
