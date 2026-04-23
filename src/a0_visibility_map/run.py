"""CLI entry point for the visibility map task."""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path

from src._tasklib.io import ensure_directory

from .collect import (
    build_visibility_rows,
    collect_split_records,
    aggregate_visibility,
    select_inset_tiles,
    write_visibility_summary_csv,
)
from .render import render_visibility_chart


@dataclass(frozen=True)
class VisibilityMapConfig:
    """Runtime configuration for the visibility map task."""

    paired_stats_root: Path
    unpaired_stats_root: Path
    out_dir: Path
    dpi: int = 300
    n_inset_tiles: int = 6


def run_task(config: VisibilityMapConfig) -> dict[str, Path]:
    """Execute the full visibility-map workflow."""
    out_dir = ensure_directory(config.out_dir)
    paired_records = collect_split_records("paired", config.paired_stats_root)
    unpaired_records = collect_split_records("unpaired", config.unpaired_stats_root)

    paired_agg = aggregate_visibility(paired_records)
    unpaired_agg = aggregate_visibility(unpaired_records)
    rows = build_visibility_rows(paired_agg, unpaired_agg)

    csv_path = write_visibility_summary_csv(rows, out_dir / "visibility_summary_table.csv")
    chart_path = render_visibility_chart(rows, out_dir / "visibility_bar_chart.png", dpi=config.dpi)

    inset_dir = ensure_directory(out_dir / "inset_tiles")
    copied: list[Path] = []
    for inset in select_inset_tiles([*paired_records, *unpaired_records], max_tiles=config.n_inset_tiles):
        target = inset_dir / f"{inset.group}_{inset.tile_id}.png"
        shutil.copy2(inset.source_path, target)
        copied.append(target)

    return {
        "csv": csv_path,
        "chart": chart_path,
        "inset_dir": inset_dir,
        "inset_count": Path(str(len(copied))),
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Build the Phase 0 visibility map figure")
    parser.add_argument("--paired-stats-root", required=True)
    parser.add_argument("--unpaired-stats-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--n-inset-tiles", type=int, default=6)
    args = parser.parse_args(argv)

    run_task(
        VisibilityMapConfig(
            paired_stats_root=Path(args.paired_stats_root),
            unpaired_stats_root=Path(args.unpaired_stats_root),
            out_dir=Path(args.out_dir),
            dpi=args.dpi,
            n_inset_tiles=args.n_inset_tiles,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
