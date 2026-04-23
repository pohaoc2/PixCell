"""Train MLP probes on frozen tile embeddings."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src._tasklib.io import ensure_directory, write_json
from src._tasklib.tile_ids import tile_ids_sha1
from src.a1_probe_linear.main import (
    build_spatial_group_splits,
    load_cv_splits,
    load_feature_matrix,
    load_tile_ids,
    run_cv_regression,
    save_cv_splits,
    summarize_probe_results,
    write_probe_results,
)


def make_mlp_probe(random_state: int = 42) -> Pipeline:
    """Create the default sklearn MLP probe."""
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "mlp",
                MLPRegressor(
                    hidden_layer_sizes=(256, 64),
                    activation="relu",
                    solver="adam",
                    learning_rate_init=1e-3,
                    max_iter=200,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=10,
                    random_state=random_state,
                ),
            ),
        ]
    )


def _write_comparison_csv(
    mlp_rows: list[dict[str, float | str | list[float] | int]],
    linear_results_json: str | Path,
    output_path: str | Path,
) -> Path:
    linear_payload = json.loads(Path(linear_results_json).read_text(encoding="utf-8"))
    linear_rows = {row["target"]: row for row in linear_payload["results"]}
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["target", "linear_r2_mean", "mlp_r2_mean", "delta"])
        writer.writeheader()
        for row in mlp_rows:
            target = str(row["target"])
            linear_r2 = float(linear_rows[target]["r2_mean"])
            mlp_r2 = float(row["r2_mean"])
            writer.writerow(
                {
                    "target": target,
                    "linear_r2_mean": linear_r2,
                    "mlp_r2_mean": mlp_r2,
                    "delta": mlp_r2 - linear_r2,
                }
            )
    return out_path


def run_task(
    features_dir: str | Path,
    targets_path: str | Path,
    tile_ids_path: str | Path,
    out_dir: str | Path,
    *,
    target_names_path: str | Path | None = None,
    cv_splits_path: str | Path | None = None,
    linear_results_json: str | Path | None = None,
    n_splits: int = 5,
    block_size_px: int = 2048,
    random_state: int = 42,
) -> dict[str, Path]:
    """Run the full MLP-probe workflow and persist all outputs."""
    output_dir = ensure_directory(out_dir)
    tile_ids = load_tile_ids(tile_ids_path)
    target_names = (
        json.loads(Path(target_names_path).read_text(encoding="utf-8"))
        if target_names_path is not None
        else [f"target_{index}" for index in range(np.load(targets_path).shape[1])]
    )
    X = load_feature_matrix(features_dir, tile_ids)
    Y = np.load(targets_path).astype(np.float32)
    if Y.shape[0] != len(tile_ids):
        raise ValueError("target matrix row count does not match tile_ids.txt")

    if cv_splits_path is None:
        splits = build_spatial_group_splits(tile_ids, n_splits=n_splits, block_size_px=block_size_px)
        saved_splits_path = save_cv_splits(tile_ids, splits, output_dir / "cv_splits.json", block_size_px=block_size_px)
    else:
        splits = load_cv_splits(tile_ids, cv_splits_path)
        saved_splits_path = Path(cv_splits_path)

    fold_scores, oof_predictions, _ = run_cv_regression(
        X,
        Y,
        splits,
        estimator_factory=lambda: make_mlp_probe(random_state=random_state),
    )
    np.save(output_dir / "mlp_probe_fold_scores.npy", fold_scores)
    np.save(output_dir / "mlp_probe_oof_predictions.npy", oof_predictions)

    rows = summarize_probe_results(fold_scores, target_names)
    result_paths = write_probe_results(rows, output_dir, prefix="mlp_probe")
    manifest_path = write_json(
        {
            "version": 1,
            "tile_count": len(tile_ids),
            "tile_ids_sha1": tile_ids_sha1(tile_ids),
            "feature_dim": int(X.shape[1]),
            "n_targets": int(Y.shape[1]),
            "target_names": target_names,
            "cv_splits_path": str(saved_splits_path),
        },
        output_dir / "manifest.json",
    )

    outputs: dict[str, Path] = {
        **result_paths,
        "splits": saved_splits_path,
        "fold_scores": output_dir / "mlp_probe_fold_scores.npy",
        "oof_predictions": output_dir / "mlp_probe_oof_predictions.npy",
        "manifest": manifest_path,
    }
    if linear_results_json is not None:
        outputs["comparison"] = _write_comparison_csv(
            rows,
            linear_results_json,
            output_dir / "comparison_vs_linear.csv",
        )
    return outputs


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Run MLP probes on frozen tile embeddings")
    parser.add_argument("--features-dir", required=True)
    parser.add_argument("--targets-path", required=True)
    parser.add_argument("--tile-ids-path", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--target-names-path", default=None)
    parser.add_argument("--cv-splits-path", default=None)
    parser.add_argument("--linear-results-json", default=None)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--block-size-px", type=int, default=2048)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args(argv)

    run_task(
        args.features_dir,
        args.targets_path,
        args.tile_ids_path,
        args.out_dir,
        target_names_path=args.target_names_path,
        cv_splits_path=args.cv_splits_path,
        linear_results_json=args.linear_results_json,
        n_splits=args.n_splits,
        block_size_px=args.block_size_px,
        random_state=args.random_state,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
