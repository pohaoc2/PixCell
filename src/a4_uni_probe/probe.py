"""Stage 1 linear probes for the a4 UNI semantic ablation."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src._tasklib.io import ensure_directory, write_json
from src._tasklib.tile_ids import list_feature_tile_ids, parse_tile_id
from src.a4_uni_probe.features import build_tme_baseline_features, build_uni_features, save_feature_bundle
from src.a4_uni_probe.labels import build_label_matrix, save_label_bundle


@dataclass(frozen=True)
class ProbeFitResult:
    r2_mean: float
    r2_std: float
    r2_per_fold: tuple[float, ...]
    coef: np.ndarray
    n_valid_folds: int


def spatial_bucket_groups(tile_ids: list[str], bucket_px: int) -> list[str]:
    """Assign tiles to coarse spatial buckets for GroupKFold splits."""
    groups: list[str] = []
    for tile_id in tile_ids:
        row_px, col_px = parse_tile_id(tile_id)
        groups.append(f"{row_px // bucket_px}_{col_px // bucket_px}")
    return groups


def _make_probe(alpha: float = 1.0) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ]
    )


def fit_probes_for_attribute(
    X: np.ndarray,
    y: np.ndarray,
    groups: list[str],
    *,
    cv_folds: int,
    alpha: float = 1.0,
) -> ProbeFitResult:
    valid = np.isfinite(y)
    if int(valid.sum()) < max(cv_folds, 2):
        return ProbeFitResult(float("nan"), float("nan"), tuple(), np.zeros(X.shape[1], dtype=np.float32), 0)

    X_valid = np.asarray(X[valid], dtype=np.float32)
    y_valid = np.asarray(y[valid], dtype=np.float32)
    groups_valid = [groups[index] for index, keep in enumerate(valid.tolist()) if keep]
    if len(set(groups_valid)) < cv_folds:
        return ProbeFitResult(float("nan"), float("nan"), tuple(), np.zeros(X.shape[1], dtype=np.float32), 0)

    splitter = GroupKFold(n_splits=cv_folds)
    fold_scores: list[float] = []
    coef_rows: list[np.ndarray] = []
    indices = np.arange(X_valid.shape[0])

    for train_idx, test_idx in splitter.split(indices, groups=groups_valid):
        model = _make_probe(alpha=alpha)
        model.fit(X_valid[train_idx], y_valid[train_idx])
        preds = np.asarray(model.predict(X_valid[test_idx]), dtype=np.float32)
        fold_scores.append(float(r2_score(y_valid[test_idx], preds)))

        scaler = model.named_steps["scale"]
        ridge = model.named_steps["ridge"]
        scale = np.asarray(scaler.scale_, dtype=np.float32)
        scale = np.where(scale == 0.0, 1.0, scale)
        raw_coef = np.asarray(ridge.coef_, dtype=np.float32) / scale
        coef_rows.append(raw_coef)

    coef_mean = np.mean(np.stack(coef_rows, axis=0), axis=0) if coef_rows else np.zeros(X.shape[1], dtype=np.float32)
    scores = np.asarray(fold_scores, dtype=np.float32)
    return ProbeFitResult(
        r2_mean=float(scores.mean()) if scores.size else float("nan"),
        r2_std=float(scores.std()) if scores.size else float("nan"),
        r2_per_fold=tuple(float(score) for score in scores.tolist()),
        coef=np.asarray(coef_mean, dtype=np.float32),
        n_valid_folds=int(scores.size),
    )


def _result_to_jsonable(result: ProbeFitResult) -> dict[str, object]:
    payload = asdict(result)
    payload["coef"] = result.coef.tolist()
    return payload


def run_probe(args: argparse.Namespace) -> dict[str, Path]:
    out_dir = ensure_directory(args.out_dir)
    tile_ids = list_feature_tile_ids(args.features_dir)
    labels, attr_names = build_label_matrix(tile_ids, args.exp_channels_dir, args.cellvit_real_dir)
    uni_features = build_uni_features(args.features_dir, tile_ids)
    tme_features = build_tme_baseline_features(args.exp_channels_dir, tile_ids)
    groups = spatial_bucket_groups(tile_ids, args.bucket_px)

    labels_path = save_label_bundle(out_dir, tile_ids=tile_ids, labels=labels, attr_names=attr_names)
    features_path = save_feature_bundle(out_dir, tile_ids=tile_ids, uni_features=uni_features, tme_features=tme_features)

    direction_dir = ensure_directory(out_dir / "probe_directions")
    rows: list[dict[str, object]] = []
    details: dict[str, object] = {
        "tile_ids": tile_ids,
        "attr_names": attr_names,
        "results": {},
    }

    for attr_index, attr_name in enumerate(attr_names):
        y = labels[:, attr_index]
        uni_result = fit_probes_for_attribute(uni_features, y, groups, cv_folds=args.cv_folds)
        tme_result = fit_probes_for_attribute(tme_features, y, groups, cv_folds=args.cv_folds)
        np.save(direction_dir / f"{attr_name}_uni_direction.npy", uni_result.coef.astype(np.float32))

        delta = float(uni_result.r2_mean - tme_result.r2_mean) if np.isfinite(uni_result.r2_mean) and np.isfinite(tme_result.r2_mean) else float("nan")
        rows.append(
            {
                "attr": attr_name,
                "uni_r2_mean": uni_result.r2_mean,
                "uni_r2_std": uni_result.r2_std,
                "uni_n_valid_folds": uni_result.n_valid_folds,
                "tme_r2_mean": tme_result.r2_mean,
                "tme_r2_std": tme_result.r2_std,
                "tme_n_valid_folds": tme_result.n_valid_folds,
                "delta_r2_uni_minus_tme": delta,
            }
        )
        details["results"][attr_name] = {
            "uni": _result_to_jsonable(uni_result),
            "tme": _result_to_jsonable(tme_result),
            "delta_r2_uni_minus_tme": delta,
        }

    rows.sort(key=lambda row: (float("-inf") if not np.isfinite(float(row["delta_r2_uni_minus_tme"])) else float(row["delta_r2_uni_minus_tme"])), reverse=True)

    csv_path = out_dir / "probe_results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["attr"])
        writer.writeheader()
        writer.writerows(rows)

    json_path = write_json(details, out_dir / "probe_results.json")
    return {
        "labels": labels_path,
        "features": features_path,
        "csv": csv_path,
        "json": json_path,
        "directions": direction_dir,
    }
