"""CLI entry point for the tradeoff scatter task."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from src._tasklib.io import ensure_directory

from .collect import aggregate_tradeoff, collect_tradeoff_records, mark_pareto_front, write_tradeoff_csv
from .render import render_tradeoff_panel


@dataclass(frozen=True)
class TradeoffScatterConfig:
    """Runtime configuration for the tradeoff task."""

    paired_metric_dir: Path
    unpaired_metric_dir: Path
    out_dir: Path
    dpi: int = 300


def run_task(config: TradeoffScatterConfig) -> dict[str, Path]:
    """Execute the paired and unpaired tradeoff aggregation workflow."""
    out_dir = ensure_directory(config.out_dir)
    paired_records = collect_tradeoff_records("paired", config.paired_metric_dir)
    unpaired_records = collect_tradeoff_records("unpaired", config.unpaired_metric_dir)
    rows = mark_pareto_front(aggregate_tradeoff([*paired_records, *unpaired_records]))

    csv_path = write_tradeoff_csv(rows, out_dir / "tradeoff_data.csv")
    paired_png = render_tradeoff_panel(rows, out_dir / "tradeoff_scatter_paired.png", split="paired", dpi=config.dpi)
    unpaired_png = render_tradeoff_panel(
        rows,
        out_dir / "tradeoff_scatter_unpaired.png",
        split="unpaired",
        dpi=config.dpi,
    )
    return {"csv": csv_path, "paired_png": paired_png, "unpaired_png": unpaired_png}


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Build the Phase 0 tradeoff scatter plots")
    parser.add_argument("--paired-metric-dir", required=True)
    parser.add_argument("--unpaired-metric-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args(argv)

    run_task(
        TradeoffScatterConfig(
            paired_metric_dir=Path(args.paired_metric_dir),
            unpaired_metric_dir=Path(args.unpaired_metric_dir),
            out_dir=Path(args.out_dir),
            dpi=args.dpi,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
