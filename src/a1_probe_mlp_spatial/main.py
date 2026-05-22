"""Train per-patch MLP probes on frozen UNI patch tokens.

For each target (one column of the per-patch target tensor) we train a single
MLP head shared across all patch positions: every (tile, patch) pair becomes
one training row of (1536,) -> scalar. Cross-validation uses the same spatial
group split scheme as the scalar probes (groups defined by a coarse pixel
block of the tile coordinate). For each test fold we compute:

* R2_global  — 1 - SS_res / Var(y_test, global). Same convention as scalar
  probes, but over per-patch values.
* R2_within  — 1 - SS_res / Var(y_test, within-tile). Asks whether the probe
  explains *within-tile* spatial variation, not just tile-mean differences.
* Pearson r — invariant to offset/scale; useful when targets are nearly
  constant per tile (oxygen, glucose).
* Delta_shuffle — drop in R2_global vs. a baseline probe trained on tokens
  permuted within tile (destroys spatial alignment, preserves marginals).

NaN patches (e.g. empty patches in CODEX targets) are masked out throughout.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src._tasklib.io import ensure_directory, write_json
from src._tasklib.tile_ids import tile_ids_sha1
from src.a1_probe_linear.main import (
    build_spatial_group_splits,
    load_cv_splits,
    load_tile_ids,
    save_cv_splits,
)

try:
    from joblib import Parallel, delayed
except ModuleNotFoundError:  # pragma: no cover
    Parallel = None
    delayed = None


def make_mlp_probe(
    random_state: int = 42,
    batch_size: int = 2048,
    hidden_layer_sizes: tuple[int, ...] = (128, 32),
    max_iter: int = 75,
) -> Pipeline:
    """Same MLP architecture as the scalar probe.

    Override batch_size; sklearn's default of min(200, n_samples) is laughable
    for our row counts (>=10k) and dominates wall time.
    """
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "mlp",
                MLPRegressor(
                    hidden_layer_sizes=hidden_layer_sizes,
                    activation="relu",
                    solver="adam",
                    learning_rate_init=1e-3,
                    max_iter=max_iter,
                    batch_size=batch_size,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=10,
                    random_state=random_state,
                ),
            ),
        ]
    )


def load_patch_token_matrix(
    features_dir: str | Path,
    tile_ids: list[str],
    *,
    memmap_path: str | Path | None = None,
    feature_suffix: str = "_uni_tokens.npy",
) -> np.ndarray:
    """Load cached UNI patch tokens row-by-row into an fp16 (memmap) tensor.

    Disk files are written by stage1 as fp16. Avoid the Python-list+np.stack
    duplication that doubles peak RAM during load (10379 tiles is ~16 GB at
    fp32). If ``memmap_path`` is provided the result is backed by an on-disk
    memmap so workers share read-only pages.
    """
    feature_dir = Path(features_dir)
    first = np.load(feature_dir / f"{tile_ids[0]}{feature_suffix}")
    if first.ndim != 2:
        raise ValueError(f"expected (P,D) tokens; got shape {first.shape}")
    n_patches, n_features = first.shape
    n_tiles = len(tile_ids)
    if memmap_path is None:
        matrix = np.empty((n_tiles, n_patches, n_features), dtype=np.float16)
    else:
        matrix = np.lib.format.open_memmap(
            memmap_path,
            mode="w+",
            dtype=np.float16,
            shape=(n_tiles, n_patches, n_features),
        )
    matrix[0] = first.astype(np.float16, copy=False)
    for idx, tile_id in enumerate(tile_ids[1:], start=1):
        arr = np.load(feature_dir / f"{tile_id}{feature_suffix}")
        matrix[idx] = arr.astype(np.float16, copy=False)
    if memmap_path is not None:
        matrix.flush()
        del matrix
        return np.load(memmap_path, mmap_mode="r")
    return matrix


def _flatten_split(
    X: np.ndarray,
    Y: np.ndarray,
    tile_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flatten tile-level (X, Y) along the patch axis for a given fold subset.

    Returns (X_flat, Y_flat, tile_id_repeat) where tile_id_repeat carries the
    parent tile index of every flattened row so per-tile baselines can be
    computed downstream.
    """
    sub_X = X[tile_idx]
    sub_Y = Y[tile_idx]
    n_tiles, n_patches, n_features = sub_X.shape
    if sub_Y.shape[:2] != (n_tiles, n_patches):
        raise ValueError(f"Y shape {sub_Y.shape} mismatched with X {sub_X.shape}")
    flat_X = sub_X.reshape(n_tiles * n_patches, n_features)
    flat_Y = sub_Y.reshape(n_tiles * n_patches)
    tile_id_repeat = np.repeat(tile_idx, n_patches)
    return flat_X, flat_Y, tile_id_repeat


def _score_fold(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    tile_id_repeat: np.ndarray,
) -> dict[str, float]:
    """Compute R2_global, R2_within (per-tile baseline), and Pearson r."""
    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    if finite.sum() < 2:
        return {
            "r2_global": float("nan"),
            "r2_within": float("nan"),
            "pearson_r": float("nan"),
            "n_patches": 0,
        }
    y_t = y_true[finite]
    y_p = y_pred[finite]
    tiles = tile_id_repeat[finite]

    r2_global = float(r2_score(y_t, y_p))

    # Per-tile mean baseline for within-tile R2.
    unique, inverse = np.unique(tiles, return_inverse=True)
    tile_sum = np.zeros(unique.size, dtype=np.float64)
    tile_count = np.zeros(unique.size, dtype=np.int64)
    np.add.at(tile_sum, inverse, y_t)
    np.add.at(tile_count, inverse, 1)
    tile_mean = tile_sum / np.maximum(tile_count, 1)
    baseline = tile_mean[inverse]
    ss_res = float(np.sum((y_t - y_p) ** 2))
    ss_tot_within = float(np.sum((y_t - baseline) ** 2))
    r2_within = float("nan") if ss_tot_within <= 0 else 1.0 - ss_res / ss_tot_within

    # Pearson r — guard against zero variance.
    if y_t.std() <= 0 or y_p.std() <= 0:
        pearson_r = float("nan")
    else:
        pearson_r = float(np.corrcoef(y_t, y_p)[0, 1])
    return {
        "r2_global": r2_global,
        "r2_within": r2_within,
        "pearson_r": pearson_r,
        "n_patches": int(finite.sum()),
    }


def _shuffle_features_within_tile(
    X_train: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Permute patch tokens within each tile; destroys spatial alignment."""
    shuffled = np.empty_like(X_train)
    for tile_idx in range(X_train.shape[0]):
        order = rng.permutation(X_train.shape[1])
        shuffled[tile_idx] = X_train[tile_idx, order]
    return shuffled


def _gather_rows(
    X: np.ndarray,
    tile_indices: np.ndarray,
    patch_indices: np.ndarray,
) -> np.ndarray:
    """Read selected (tile, patch) rows from a possibly-memmap X as fp32.

    Uses numpy fancy indexing on the leading two axes so only the requested
    rows are materialized (avoids the full fp16 train-slice copy that
    previously triggered OOM on a 32 GB box).
    """
    return X[tile_indices, patch_indices, :].astype(np.float32, copy=False)


def _run_one_target(
    X: np.ndarray,
    Y_target: np.ndarray,
    splits: list[dict[str, list[int]]],
    *,
    estimator_factory,
    compute_shuffle_baseline: bool,
    shuffle_seed: int,
    max_train_rows: int | None = 500_000,
    subsample_seed: int = 0,
) -> dict[str, list[float] | float | int]:
    """Run CV for a single target column; return per-fold and aggregate metrics.

    Training rows above ``max_train_rows`` are randomly subsampled *before*
    reading from the X memmap, so the working set never exceeds the chosen
    cap (default 500k rows * 1536 floats * 4 bytes ~= 3 GB per fit).
    """
    n_patches = Y_target.shape[1]
    fold_metrics: list[dict[str, float]] = []
    fold_shuffle_r2: list[float] = []
    for fold_idx, split in enumerate(splits):
        train_idx = np.asarray(split["train_idx"], dtype=np.int64)
        test_idx = np.asarray(split["test_idx"], dtype=np.int64)

        # ---- Training set: pick finite rows first, subsample, THEN gather. ----
        y_train_view = Y_target[train_idx]  # (n_train_tiles, n_patches)
        finite_flat = np.flatnonzero(np.isfinite(y_train_view).reshape(-1))
        if finite_flat.size < 2:
            fold_metrics.append({
                "r2_global": float("nan"),
                "r2_within": float("nan"),
                "pearson_r": float("nan"),
                "n_patches": 0,
            })
            continue
        if max_train_rows is not None and finite_flat.size > max_train_rows:
            rng_sub = np.random.default_rng(subsample_seed + fold_idx)
            chosen = rng_sub.choice(finite_flat, size=max_train_rows, replace=False)
        else:
            chosen = finite_flat
        local_tile = chosen // n_patches
        local_patch = chosen % n_patches
        global_tile = train_idx[local_tile]
        X_train_fit = _gather_rows(X, global_tile, local_patch)
        y_train_fit = y_train_view.reshape(-1)[chosen]

        # ---- Test set: keep all patches (needed for spatial R2 metrics). ----
        n_test_tiles = test_idx.size
        test_tile_repeat = np.repeat(test_idx, n_patches)
        test_patch_repeat = np.tile(np.arange(n_patches, dtype=np.int64), n_test_tiles)
        X_test_fit = _gather_rows(X, test_tile_repeat, test_patch_repeat)
        y_test_flat = Y_target[test_idx].reshape(-1)
        tile_id_repeat = test_tile_repeat

        model = estimator_factory()
        model.fit(X_train_fit, y_train_fit)
        preds = np.asarray(model.predict(X_test_fit), dtype=np.float32)
        metrics = _score_fold(y_test_flat, preds, tile_id_repeat)
        fold_metrics.append(metrics)

        if compute_shuffle_baseline:
            rng = np.random.default_rng(shuffle_seed + fold_idx)
            # Permute patch position within each chosen training tile so the
            # baseline destroys spatial alignment but keeps marginals.
            permuted_patch = local_patch.copy()
            for unique_tile in np.unique(local_tile):
                mask = local_tile == unique_tile
                permuted_patch[mask] = rng.permutation(permuted_patch[mask])
            X_train_shuffled_fit = _gather_rows(X, global_tile, permuted_patch)
            shuffled_model = estimator_factory()
            shuffled_model.fit(X_train_shuffled_fit, y_train_fit)
            shuffle_preds = np.asarray(shuffled_model.predict(X_test_fit), dtype=np.float32)
            shuffle_metrics = _score_fold(y_test_flat, shuffle_preds, tile_id_repeat)
            fold_shuffle_r2.append(shuffle_metrics["r2_global"])

    def _aggregate(key: str) -> tuple[float, float]:
        values = [m[key] for m in fold_metrics if np.isfinite(m[key])]
        if not values:
            return float("nan"), float("nan")
        return float(np.mean(values)), float(np.std(values))

    r2_global_mean, r2_global_sd = _aggregate("r2_global")
    r2_within_mean, r2_within_sd = _aggregate("r2_within")
    pearson_mean, pearson_sd = _aggregate("pearson_r")
    n_valid = sum(1 for m in fold_metrics if np.isfinite(m["r2_global"]))
    shuffle_mean = float(np.mean([v for v in fold_shuffle_r2 if np.isfinite(v)])) if fold_shuffle_r2 else float("nan")
    delta_shuffle = float(r2_global_mean - shuffle_mean) if np.isfinite(shuffle_mean) and np.isfinite(r2_global_mean) else float("nan")
    return {
        "r2_global_mean": r2_global_mean,
        "r2_global_sd": r2_global_sd,
        "r2_global_folds": [m["r2_global"] for m in fold_metrics],
        "r2_within_mean": r2_within_mean,
        "r2_within_sd": r2_within_sd,
        "r2_within_folds": [m["r2_within"] for m in fold_metrics],
        "pearson_r_mean": pearson_mean,
        "pearson_r_sd": pearson_sd,
        "pearson_r_folds": [m["pearson_r"] for m in fold_metrics],
        "r2_global_shuffle_mean": shuffle_mean,
        "delta_shuffle": delta_shuffle,
        "n_valid_folds": n_valid,
    }


def run_task(
    features_dir: str | Path,
    targets_path: str | Path,
    tile_ids_path: str | Path,
    out_dir: str | Path,
    *,
    target_names_path: str | Path | None = None,
    cv_splits_path: str | Path | None = None,
    n_splits: int = 3,
    block_size_px: int = 2048,
    random_state: int = 42,
    compute_shuffle_baseline: bool = False,
    shuffle_seed: int = 0,
    n_jobs: int = 1,
    max_train_rows: int | None = 500_000,
    batch_size: int = 2048,
    n_tiles_subsample: int | None = None,
    subsample_tile_seed: int = 42,
    preloaded_X: np.ndarray | None = None,
    feature_suffix: str = "_uni_tokens.npy",
    hidden_layer_sizes: tuple[int, ...] = (128, 32),
    max_iter: int = 75,
) -> dict[str, Path]:
    """Run the spatial MLP probe across all targets and persist results."""
    output_dir = ensure_directory(out_dir)
    tile_ids = load_tile_ids(tile_ids_path)
    Y_full = np.load(targets_path).astype(np.float32)
    if Y_full.ndim != 3:
        raise ValueError(
            f"targets at {targets_path} must be (N_tiles, n_patches, n_targets); got {Y_full.shape}"
        )
    if Y_full.shape[0] != len(tile_ids):
        raise ValueError("targets row count does not match tile_ids.txt")

    if n_tiles_subsample is not None and n_tiles_subsample < len(tile_ids):
        rng_tile = np.random.default_rng(subsample_tile_seed)
        keep = np.sort(rng_tile.choice(len(tile_ids), size=n_tiles_subsample, replace=False))
        tile_ids = [tile_ids[i] for i in keep]
        Y_full = Y_full[keep]
    n_tiles, n_patches, n_targets = Y_full.shape
    target_names = (
        json.loads(Path(target_names_path).read_text(encoding="utf-8"))
        if target_names_path is not None
        else [f"target_{idx}" for idx in range(n_targets)]
    )
    if len(target_names) != n_targets:
        raise ValueError("target_names length mismatch with targets tensor")

    # Stream tokens straight into an on-disk fp16 memmap to avoid the
    # list+stack peak (~32 GB fp32). Workers re-use the same backing store.
    memmap_path = output_dir / "_X_memmap.npy"
    if preloaded_X is not None:
        if preloaded_X.shape[:2] != (n_tiles, n_patches):
            raise ValueError(
                f"preloaded_X shape {preloaded_X.shape} mismatched with targets {Y_full.shape}"
            )
        X_mm = preloaded_X
    else:
        X_mm = load_patch_token_matrix(
            features_dir,
            tile_ids,
            memmap_path=memmap_path,
            feature_suffix=feature_suffix,
        )
    if X_mm.shape[:2] != (n_tiles, n_patches):
        raise ValueError(f"feature shape {X_mm.shape} mismatched with targets {Y_full.shape}")

    if cv_splits_path is None:
        splits = build_spatial_group_splits(tile_ids, n_splits=n_splits, block_size_px=block_size_px)
        saved_splits_path = save_cv_splits(
            tile_ids,
            splits,
            output_dir / "cv_splits.json",
            block_size_px=block_size_px,
        )
    else:
        splits = load_cv_splits(tile_ids, cv_splits_path)
        saved_splits_path = Path(cv_splits_path)

    import sys
    import time as _time

    print(f"[probe] starting {n_targets} targets x {len(splits)} folds (n_jobs={n_jobs}, max_train_rows={max_train_rows}, batch={batch_size})", flush=True)
    if n_jobs == 1 or Parallel is None or delayed is None or n_targets == 1:
        target_results = []
        for target_idx, name in enumerate(target_names):
            t0 = _time.time()
            target_results.append(
                _run_one_target(
                    X_mm,
                    Y_full[..., target_idx],
                    splits,
                    estimator_factory=lambda: make_mlp_probe(
                        random_state=random_state,
                        batch_size=batch_size,
                        hidden_layer_sizes=hidden_layer_sizes,
                        max_iter=max_iter,
                    ),
                    compute_shuffle_baseline=compute_shuffle_baseline,
                    shuffle_seed=shuffle_seed,
                    max_train_rows=max_train_rows,
                    subsample_seed=random_state + target_idx,
                )
            )
            dt = _time.time() - t0
            r2 = target_results[-1]["r2_global_mean"]
            print(f"[probe] {target_idx + 1}/{n_targets} {name} done in {dt:.1f}s r2_global={r2:.3f}", flush=True)
    else:
        # Pass X via the delayed args so joblib auto-memmaps the same backing
        # store across workers instead of forking 16 GB per worker.
        target_results = Parallel(n_jobs=n_jobs, prefer="processes", max_nbytes="1M")(
            delayed(_run_one_target)(
                X_mm,
                Y_full[..., target_idx],
                splits,
                estimator_factory=lambda: make_mlp_probe(
                    random_state=random_state,
                    batch_size=batch_size,
                    hidden_layer_sizes=hidden_layer_sizes,
                    max_iter=max_iter,
                ),
                compute_shuffle_baseline=compute_shuffle_baseline,
                shuffle_seed=shuffle_seed,
                max_train_rows=max_train_rows,
                subsample_seed=random_state + target_idx,
            )
            for target_idx in range(n_targets)
        )
    rows: list[dict[str, object]] = [
        {"target": name, **result}
        for name, result in zip(target_names, target_results)
    ]

    # Drop the temporary memmap; results are already collected.
    try:
        memmap_path.unlink()
    except OSError:
        pass

    json_path = write_json({"version": 1, "results": rows}, output_dir / "mlp_spatial_probe_results.json")
    csv_path = output_dir / "mlp_spatial_probe_results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "target",
                "r2_mean",
                "r2_sd",
                "r2_within_mean",
                "r2_within_sd",
                "pearson_r_mean",
                "pearson_r_sd",
                "delta_shuffle",
                "n_valid_folds",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "target": row["target"],
                    "r2_mean": row["r2_global_mean"],
                    "r2_sd": row["r2_global_sd"],
                    "r2_within_mean": row["r2_within_mean"],
                    "r2_within_sd": row["r2_within_sd"],
                    "pearson_r_mean": row["pearson_r_mean"],
                    "pearson_r_sd": row["pearson_r_sd"],
                    "delta_shuffle": row["delta_shuffle"],
                    "n_valid_folds": row["n_valid_folds"],
                }
            )

    manifest_path = write_json(
        {
            "version": 1,
            "tile_count": len(tile_ids),
            "tile_ids_sha1": tile_ids_sha1(tile_ids),
            "feature_dim": int(X_mm.shape[-1]),
            "n_patches": int(n_patches),
            "n_targets": int(n_targets),
            "target_names": target_names,
            "compute_shuffle_baseline": bool(compute_shuffle_baseline),
            "cv_splits_path": str(saved_splits_path),
        },
        output_dir / "manifest.json",
    )
    return {
        "json": json_path,
        "csv": csv_path,
        "splits": saved_splits_path,
        "manifest": manifest_path,
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    def _parse_hidden_sizes(raw: str) -> tuple[int, ...]:
        values = tuple(int(chunk.strip()) for chunk in raw.split(",") if chunk.strip())
        if not values:
            raise argparse.ArgumentTypeError("hidden-layer-sizes must contain at least one integer")
        return values

    parser = argparse.ArgumentParser(description="Run spatial MLP probes on UNI patch tokens")
    parser.add_argument("--features-dir", required=True)
    parser.add_argument("--targets-path", required=True)
    parser.add_argument("--tile-ids-path", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--target-names-path", default=None)
    parser.add_argument("--cv-splits-path", default=None)
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--block-size-px", type=int, default=2048)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--compute-shuffle-baseline", action="store_true")
    parser.add_argument("--shuffle-seed", type=int, default=0)
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel processes across targets")
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=500_000,
        help="Cap on training rows per fit (random subsample). Use 0 for no cap.",
    )
    parser.add_argument("--batch-size", type=int, default=2048, help="MLP Adam mini-batch size.")
    parser.add_argument(
        "--max-iter",
        type=int,
        default=75,
        help="Maximum MLP training iterations per fit.",
    )
    parser.add_argument(
        "--hidden-layer-sizes",
        type=_parse_hidden_sizes,
        default=(128, 32),
        help="Comma-separated hidden layer sizes for the MLP probe (default: 128,32).",
    )
    parser.add_argument(
        "--n-tiles",
        type=int,
        default=0,
        help="Random subsample of tiles before fitting. 0 = use all.",
    )
    parser.add_argument("--subsample-tile-seed", type=int, default=42)
    parser.add_argument(
        "--feature-suffix",
        default="_uni_tokens.npy",
        help="Feature filename suffix to load from --features-dir (default: _uni_tokens.npy).",
    )
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
        random_state=args.random_state,
        compute_shuffle_baseline=args.compute_shuffle_baseline,
        shuffle_seed=args.shuffle_seed,
        n_jobs=args.n_jobs,
        max_train_rows=None if args.max_train_rows <= 0 else args.max_train_rows,
        batch_size=args.batch_size,
        n_tiles_subsample=None if args.n_tiles <= 0 else args.n_tiles,
        subsample_tile_seed=args.subsample_tile_seed,
        feature_suffix=args.feature_suffix,
        hidden_layer_sizes=args.hidden_layer_sizes,
        max_iter=args.max_iter,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
