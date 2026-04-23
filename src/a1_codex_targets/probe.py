"""Run linear and MLP probes over CODEX T2/T3 targets."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

from src._tasklib.io import ensure_directory


def run_probe_tasks(
    *,
    features_dir: str | Path,
    tile_ids_path: str | Path,
    cv_splits_path: str | Path,
    t2_targets_path: str | Path,
    marker_names_path: str | Path,
    out_dir: str | Path,
    t3_targets_path: str | Path | None = None,
    quantile_names_path: str | Path | None = None,
    linear_runner: Callable[..., dict[str, Path]] | None = None,
    mlp_runner: Callable[..., dict[str, Path]] | None = None,
) -> dict[str, dict[str, Path]]:
    """Run linear and MLP probes for T2 and optionally T3."""
    if linear_runner is None:
        from src.a1_probe_linear.main import run_task as linear_runner
    if mlp_runner is None:
        from src.a1_probe_mlp.main import run_task as mlp_runner

    output_dir = ensure_directory(out_dir)
    results: dict[str, dict[str, Path]] = {}

    t2_linear_dir = output_dir / "t2_linear"
    results["t2_linear"] = linear_runner(
        features_dir,
        t2_targets_path,
        tile_ids_path,
        t2_linear_dir,
        target_names_path=marker_names_path,
        cv_splits_path=cv_splits_path,
    )
    results["t2_mlp"] = mlp_runner(
        features_dir,
        t2_targets_path,
        tile_ids_path,
        output_dir / "t2_mlp",
        target_names_path=marker_names_path,
        cv_splits_path=cv_splits_path,
        linear_results_json=results["t2_linear"]["json"],
    )

    if t3_targets_path is not None and quantile_names_path is not None:
        t3_linear_dir = output_dir / "t3_linear"
        results["t3_linear"] = linear_runner(
            features_dir,
            t3_targets_path,
            tile_ids_path,
            t3_linear_dir,
            target_names_path=quantile_names_path,
            cv_splits_path=cv_splits_path,
        )
        results["t3_mlp"] = mlp_runner(
            features_dir,
            t3_targets_path,
            tile_ids_path,
            output_dir / "t3_mlp",
            target_names_path=quantile_names_path,
            cv_splits_path=cv_splits_path,
            linear_results_json=results["t3_linear"]["json"],
        )
    return results


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Run linear and MLP probes for CODEX T2/T3 targets")
    parser.add_argument("--features-dir", required=True)
    parser.add_argument("--tile-ids-path", required=True)
    parser.add_argument("--cv-splits-path", required=True)
    parser.add_argument("--t2-targets-path", required=True)
    parser.add_argument("--marker-names-path", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--t3-targets-path", default=None)
    parser.add_argument("--quantile-names-path", default=None)
    args = parser.parse_args(argv)

    run_probe_tasks(
        features_dir=args.features_dir,
        tile_ids_path=args.tile_ids_path,
        cv_splits_path=args.cv_splits_path,
        t2_targets_path=args.t2_targets_path,
        marker_names_path=args.marker_names_path,
        out_dir=args.out_dir,
        t3_targets_path=args.t3_targets_path,
        quantile_names_path=args.quantile_names_path,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
