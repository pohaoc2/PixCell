"""Train linear probes on frozen tile embeddings."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src._tasklib.io import ensure_directory, write_json
from src._tasklib.tile_ids import parse_tile_id, tile_ids_sha1

try:
    from joblib import Parallel, delayed
except ModuleNotFoundError:  # pragma: no cover
    Parallel = None
    delayed = None

def load_tile_ids(tile_ids_path: str | Path) -> list[str]:
    """Load newline-delimited tile IDs."""
    return [line.strip() for line in Path(tile_ids_path).read_text(encoding="utf-8").splitlines() if line.strip()]


def load_feature_matrix(features_dir: str | Path, tile_ids: list[str]) -> np.ndarray:
    """Load cached UNI embeddings in the supplied tile order."""
    feature_dir = Path(features_dir)
    rows = [np.load(feature_dir / f"{tile_id}_uni.npy").astype(np.float32) for tile_id in tile_ids]
    return np.stack(rows, axis=0)


def build_spatial_group_splits(
    tile_ids: list[str],
    *,
    n_splits: int = 5,
    block_size_px: int = 2048,
) -> list[dict[str, list[int]]]:
    """Build GroupKFold splits using coarse spatial blocks."""
    if len(tile_ids) < 2:
        raise ValueError("need at least two tile IDs to build CV splits")
    groups = []
    for tile_id in tile_ids:
        row_px, col_px = parse_tile_id(tile_id)
        groups.append(f"{row_px // block_size_px}_{col_px // block_size_px}")
    unique_groups = {group for group in groups}
    if len(unique_groups) < n_splits:
        raise ValueError(f"need at least {n_splits} unique spatial groups; got {len(unique_groups)}")

    split_builder = GroupKFold(n_splits=n_splits)
    indices = np.arange(len(tile_ids))
    splits: list[dict[str, list[int]]] = []
    for train_idx, test_idx in split_builder.split(indices, groups=groups):
        splits.append({"train_idx": train_idx.tolist(), "test_idx": test_idx.tolist()})
    return splits


def save_cv_splits(
    tile_ids: list[str],
    splits: list[dict[str, list[int]]],
    output_path: str | Path,
    *,
    block_size_px: int,
) -> Path:
    """Persist CV split indices plus an alignment hash."""
    return write_json(
        {
            "version": 1,
            "tile_count": len(tile_ids),
            "tile_ids_sha1": tile_ids_sha1(tile_ids),
            "block_size_px": block_size_px,
            "n_splits": len(splits),
            "splits": splits,
        },
        output_path,
    )


def load_cv_splits(tile_ids: list[str], cv_splits_path: str | Path) -> list[dict[str, list[int]]]:
    """Load and validate saved CV splits."""
    payload = json.loads(Path(cv_splits_path).read_text(encoding="utf-8"))
    expected_hash = tile_ids_sha1(tile_ids)
    if payload.get("tile_ids_sha1") != expected_hash:
        raise ValueError("tile_ids.txt does not match the saved CV split hash")
    return list(payload["splits"])


def make_linear_probe(alpha: float = 1.0) -> Pipeline:
    """Create the default sklearn linear probe."""
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ]
    )


def _fit_regression_target(
    X: np.ndarray,
    Y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    target_idx: int,
    *,
    estimator_factory,
) -> tuple[int, np.ndarray | None, np.ndarray | None, float, np.ndarray | None]:
    y_train = Y[train_idx, target_idx]
    y_test = Y[test_idx, target_idx]
    train_mask = np.isfinite(y_train)
    test_mask = np.isfinite(y_test)
    if train_mask.sum() == 0 or test_mask.sum() == 0:
        return target_idx, None, None, float("nan"), None

    model = estimator_factory()
    model.fit(X[train_idx][train_mask], y_train[train_mask])
    preds = np.asarray(model.predict(X[test_idx][test_mask]), dtype=np.float32)
    score = float(r2_score(y_test[test_mask], preds))

    ridge = getattr(model, "named_steps", {}).get("ridge")
    coef = None
    if ridge is not None and getattr(ridge, "coef_", None) is not None:
        coef = np.asarray(ridge.coef_, dtype=np.float64)
    return target_idx, test_idx[test_mask], preds, score, coef


def run_cv_regression(
    X: np.ndarray,
    Y: np.ndarray,
    splits: list[dict[str, list[int]]],
    *,
    estimator_factory=make_linear_probe,
    n_jobs: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run per-target CV regression and return scores, OOF predictions, and mean coefficients."""
    n_targets = Y.shape[1]
    n_features = X.shape[1]
    fold_scores = np.full((len(splits), n_targets), np.nan, dtype=np.float32)
    oof_predictions = np.full_like(Y, np.nan, dtype=np.float32)
    coef_sum = np.zeros((n_targets, n_features), dtype=np.float64)
    coef_count = np.zeros(n_targets, dtype=np.int32)

    for fold_idx, split in enumerate(splits):
        train_idx = np.asarray(split["train_idx"], dtype=np.int64)
        test_idx = np.asarray(split["test_idx"], dtype=np.int64)
        if n_jobs == 1 or Parallel is None or delayed is None or n_targets == 1:
            results = [
                _fit_regression_target(
                    X,
                    Y,
                    train_idx,
                    test_idx,
                    target_idx,
                    estimator_factory=estimator_factory,
                )
                for target_idx in range(n_targets)
            ]
        else:
            results = Parallel(n_jobs=n_jobs)(
                delayed(_fit_regression_target)(
                    X,
                    Y,
                    train_idx,
                    test_idx,
                    target_idx,
                    estimator_factory=estimator_factory,
                )
                for target_idx in range(n_targets)
            )

        for target_idx, target_test_idx, preds, score, coef in results:
            if target_test_idx is None or preds is None:
                continue
            oof_predictions[target_test_idx, target_idx] = preds
            fold_scores[fold_idx, target_idx] = score
            if coef is not None:
                coef_sum[target_idx] += coef
                coef_count[target_idx] += 1

    coef_mean = np.divide(
        coef_sum,
        coef_count[:, None],
        out=np.zeros((n_targets, n_features), dtype=np.float64),
        where=coef_count[:, None] > 0,
    ).astype(np.float32)
    return fold_scores, oof_predictions, coef_mean


def summarize_probe_results(
    fold_scores: np.ndarray,
    target_names: list[str],
) -> list[dict[str, float | str | list[float] | int]]:
    """Summarize per-target R2 across CV folds."""
    rows: list[dict[str, float | str | list[float] | int]] = []
    for target_idx, target_name in enumerate(target_names):
        column = fold_scores[:, target_idx]
        finite = np.isfinite(column)
        values = column[finite]
        rows.append(
            {
                "target": target_name,
                "r2_mean": float(np.mean(values)) if values.size else float("nan"),
                "r2_sd": float(np.std(values)) if values.size else float("nan"),
                "r2_folds": [float(value) for value in values.tolist()],
                "n_valid_folds": int(values.size),
            }
        )
    return rows


def write_probe_results(
    rows: list[dict[str, float | str | list[float] | int]],
    out_dir: str | Path,
    *,
    prefix: str,
) -> dict[str, Path]:
    """Write JSON and CSV probe summaries."""
    output_dir = ensure_directory(out_dir)
    json_path = write_json({"version": 1, "results": rows}, output_dir / f"{prefix}_results.json")
    csv_path = output_dir / f"{prefix}_results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["target", "r2_mean", "r2_sd", "n_valid_folds"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "target": row["target"],
                    "r2_mean": row["r2_mean"],
                    "r2_sd": row["r2_sd"],
                    "n_valid_folds": row["n_valid_folds"],
                }
            )
    return {"json": json_path, "csv": csv_path}


def run_task(
    features_dir: str | Path,
    targets_path: str | Path,
    tile_ids_path: str | Path,
    out_dir: str | Path,
    *,
    target_names_path: str | Path | None = None,
    cv_splits_path: str | Path | None = None,
    n_splits: int = 5,
    block_size_px: int = 2048,
    alpha: float = 1.0,
    n_jobs: int = 1,
    preloaded_X: np.ndarray | None = None,
) -> dict[str, Path]:
    """Run the full linear-probe workflow and persist all outputs."""
    output_dir = ensure_directory(out_dir)
    tile_ids = load_tile_ids(tile_ids_path)
    target_names = (
        json.loads(Path(target_names_path).read_text(encoding="utf-8"))
        if target_names_path is not None
        else [f"target_{index}" for index in range(np.load(targets_path).shape[1])]
    )
    X = preloaded_X if preloaded_X is not None else load_feature_matrix(features_dir, tile_ids)
    Y = np.load(targets_path).astype(np.float32)
    if Y.shape[0] != len(tile_ids):
        raise ValueError("target matrix row count does not match tile_ids.txt")

    if cv_splits_path is None:
        splits = build_spatial_group_splits(tile_ids, n_splits=n_splits, block_size_px=block_size_px)
        saved_splits_path = save_cv_splits(tile_ids, splits, output_dir / "cv_splits.json", block_size_px=block_size_px)
    else:
        splits = load_cv_splits(tile_ids, cv_splits_path)
        saved_splits_path = Path(cv_splits_path)

    fold_scores, oof_predictions, coef_mean = run_cv_regression(
        X,
        Y,
        splits,
        estimator_factory=lambda: make_linear_probe(alpha=alpha),
        n_jobs=n_jobs,
    )
    np.save(output_dir / "linear_probe_fold_scores.npy", fold_scores)
    np.save(output_dir / "linear_probe_oof_predictions.npy", oof_predictions)
    np.save(output_dir / "linear_probe_coef_mean.npy", coef_mean)

    rows = summarize_probe_results(fold_scores, target_names)
    result_paths = write_probe_results(rows, output_dir, prefix="linear_probe")
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
    return {
        **result_paths,
        "splits": saved_splits_path,
        "fold_scores": output_dir / "linear_probe_fold_scores.npy",
        "oof_predictions": output_dir / "linear_probe_oof_predictions.npy",
        "coef_mean": output_dir / "linear_probe_coef_mean.npy",
        "manifest": manifest_path,
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Run linear probes on frozen tile embeddings")
    parser.add_argument("--features-dir", required=True)
    parser.add_argument("--targets-path", required=True)
    parser.add_argument("--tile-ids-path", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--target-names-path", default=None)
    parser.add_argument("--cv-splits-path", default=None)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--block-size-px", type=int, default=2048)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args(argv)

    run_task(
        args.features_dir,
        args.targets_path,
        args.tile_ids_path,
        args.out_dir,
        target_names_path=args.target_names_path,
        cv_splits_path=args.cv_splits_path,
        n_splits=args.n_splits,
        block_size_px=args.block_size_px,
        alpha=args.alpha,
        n_jobs=args.n_jobs,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
