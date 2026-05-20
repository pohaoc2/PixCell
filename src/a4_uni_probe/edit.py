"""UNI vector editing helpers plus sweep/null runners."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from src._tasklib.io import ensure_directory, write_json
from src.a4_uni_probe.appearance_metrics import appearance_row_for_image
from src.a4_uni_probe.inference import GenSpec, generate_with_uni_override, load_inference_bundle
from src.a4_uni_probe.labels import APPEARANCE_ATTR_NAMES, MORPHOLOGY_ATTR_NAMES
from src.a4_uni_probe.metrics import morphology_row_for_image
from src.a4_uni_probe.slope_stats import bootstrap_slope_summary


def sweep_uni(uni: np.ndarray, w: np.ndarray, alphas: list[float] | tuple[float, ...] | np.ndarray) -> np.ndarray:
    base = np.asarray(uni, dtype=np.float32).reshape(-1)
    direction = np.asarray(w, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(direction))
    if norm == 0.0:
        raise ValueError("probe direction must be non-zero")
    direction = direction / norm
    scale = float(np.linalg.norm(base))
    return np.stack([base + float(alpha) * scale * direction for alpha in alphas], axis=0).astype(np.float32)


def null_uni(uni: np.ndarray, w: np.ndarray) -> np.ndarray:
    base = np.asarray(uni, dtype=np.float32).reshape(-1)
    direction = np.asarray(w, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(direction))
    if norm == 0.0:
        raise ValueError("probe direction must be non-zero")
    direction = direction / norm
    return (base - np.dot(base, direction) * direction).astype(np.float32)


def random_unit_direction(d: int, *, rng: np.random.Generator | None = None, seed: int | None = None) -> np.ndarray:
    generator = rng or np.random.default_rng(seed)
    direction = generator.normal(size=d).astype(np.float32)
    norm = float(np.linalg.norm(direction))
    if norm == 0.0:
        raise ValueError("random direction norm was zero")
    return direction / norm


def _load_bundle(npz_path: Path) -> dict[str, object]:
    data = np.load(npz_path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def _select_sweep_attrs(probe_csv: Path, top_k: int, attr_pool: str = "morphology") -> list[str]:
    pool = APPEARANCE_ATTR_NAMES if attr_pool == "appearance" else MORPHOLOGY_ATTR_NAMES
    rows: list[tuple[str, float]] = []
    with probe_csv.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            attr = row["attr"]
            if attr not in pool:
                continue
            rows.append((attr, float(row["delta_r2_uni_minus_tme"])))
    rows.sort(key=lambda item: (float("-inf") if not np.isfinite(item[1]) else item[1]), reverse=True)
    return [attr for attr, _ in rows[:top_k]]


def _load_fixed_tile_ids(path: str | Path | None) -> list[str] | None:
    if path is None:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [str(tile_id) for tile_id in payload["tile_ids"]]


def _select_sweep_tiles(labels_npz: Path, attr: str, k: int, seed: int) -> list[str]:
    bundle = _load_bundle(labels_npz)
    tile_ids = [str(tile_id) for tile_id in bundle["tile_ids"].tolist()]
    attr_names = [str(name) for name in bundle["attr_names"].tolist()]
    labels = np.asarray(bundle["labels"], dtype=np.float32)
    attr_index = attr_names.index(attr)
    y = labels[:, attr_index]
    valid = np.isfinite(y)
    if not np.any(valid):
        return []
    indices = np.flatnonzero(valid)
    sorted_indices = indices[np.argsort(y[indices])]
    if len(sorted_indices) <= k:
        return [tile_ids[index] for index in sorted_indices.tolist()]
    chosen_positions = np.linspace(0, len(sorted_indices) - 1, num=k, dtype=int)
    return [tile_ids[int(sorted_indices[pos])] for pos in chosen_positions.tolist()]


def _write_rows_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _validate_tile_shard(tile_shard_index: int, tile_shard_count: int) -> tuple[int, int]:
    if tile_shard_count < 1:
        raise ValueError("tile_shard_count must be at least 1")
    if not 0 <= tile_shard_index < tile_shard_count:
        raise ValueError("tile_shard_index must be in [0, tile_shard_count)")
    return tile_shard_index, tile_shard_count


def _shard_tile_ids(tile_ids: list[str], tile_shard_index: int, tile_shard_count: int) -> list[str]:
    tile_shard_index, tile_shard_count = _validate_tile_shard(tile_shard_index, tile_shard_count)
    if tile_shard_count == 1:
        return list(tile_ids)
    start = len(tile_ids) * tile_shard_index // tile_shard_count
    end = len(tile_ids) * (tile_shard_index + 1) // tile_shard_count
    return list(tile_ids[start:end])


def _metrics_output_path(attr_dir: Path, tile_shard_index: int, tile_shard_count: int) -> Path:
    if tile_shard_count == 1:
        return attr_dir / "metrics.csv"
    return attr_dir / f"metrics.shard_{tile_shard_index + 1:02d}of{tile_shard_count:02d}.csv"


def _aggregate_metrics(attr_dir: Path) -> Path | None:
    shard_paths = sorted(attr_dir.glob("metrics.shard_*of*.csv"))
    if not shard_paths:
        metrics_path = attr_dir / "metrics.csv"
        return metrics_path if metrics_path.is_file() else None

    rows: list[dict[str, object]] = []
    for shard_path in shard_paths:
        with shard_path.open(encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    if not rows:
        return None

    metrics_path = attr_dir / "metrics.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return metrics_path


def _summarize_slopes(metrics_csv: Path, out_json: Path, attr: str) -> None:
    rows = list(csv.DictReader(metrics_csv.open(encoding="utf-8")))
    summary: dict[str, object] = {"attr": attr}
    for direction_name in ("targeted", "random"):
        direction_rows = [row for row in rows if row.get("direction") == direction_name]
        if not direction_rows:
            continue
        alphas = np.asarray([float(row["alpha"]) for row in direction_rows], dtype=np.float32)
        values = np.asarray([float(row["target_value"]) for row in direction_rows], dtype=np.float32)
        stats = bootstrap_slope_summary(alphas, values, n_boot=1000, seed=0)
        ci = stats["slope_ci95"]
        summary[direction_name] = {
            "slope_mean": stats["slope_mean"],
            "slope_ci95": [ci[0], ci[1]],
            "n": stats["n"],
        }

    targeted = summary.get("targeted")
    random = summary.get("random")
    pass_met = False
    if isinstance(targeted, dict) and isinstance(random, dict):
        targeted_ci = targeted["slope_ci95"]
        random_slope = float(random["slope_mean"])
        pass_met = bool(targeted_ci[0] * targeted_ci[1] > 0 and abs(float(targeted["slope_mean"])) > 3.0 * abs(random_slope))
    summary["pass_criterion_met"] = pass_met
    write_json(summary, out_json)


def _summarize_nulls(metrics_csv: Path, out_json: Path, attr: str) -> None:
    rows = list(csv.DictReader(metrics_csv.open(encoding="utf-8")))
    summary: dict[str, object] = {"attr": attr}
    means: dict[str, float] = {}
    for condition_name in ("targeted", "random", "full_uni_null"):
        values = np.asarray(
            [float(row["target_value"]) for row in rows if row.get("condition") == condition_name],
            dtype=np.float32,
        )
        finite = values[np.isfinite(values)]
        means[condition_name] = float(np.mean(finite)) if finite.size else float("nan")
        summary[condition_name] = {
            "metric_mean": means[condition_name],
            "n": int(finite.size),
        }
    summary["targeted_minus_random"] = (
        float(means["targeted"] - means["random"])
        if np.isfinite(means["targeted"]) and np.isfinite(means["random"])
        else float("nan")
    )
    write_json(summary, out_json)


def _load_probe_direction(out_dir: Path, attr: str) -> np.ndarray:
    return np.asarray(np.load(out_dir / "probe_directions" / f"{attr}_uni_direction.npy"), dtype=np.float32)


def _image_metric_rows(image_path: Path) -> tuple[dict[str, float], dict[str, float]]:
    morph = morphology_row_for_image(image_path)
    appearance = appearance_row_for_image(image_path)
    return morph, appearance


def _target_value_for_attr(
    attr: str,
    *,
    morph: dict[str, float],
    appearance: dict[str, float],
) -> float:
    if attr in APPEARANCE_ATTR_NAMES:
        return float(appearance[f"appearance.{attr}"])
    return float(morph[attr])


def run_sweep(args: argparse.Namespace) -> None:
    out_dir = ensure_directory(args.out_dir)
    probe_csv = out_dir / "probe_results.csv"
    labels_npz = out_dir / "labels.npz"
    features_npz = out_dir / "features.npz"
    if not probe_csv.is_file() or not labels_npz.is_file() or not features_npz.is_file():
        raise FileNotFoundError("run `probe` first so probe_results.csv, labels.npz, and features.npz exist")

    attrs = _select_sweep_attrs(probe_csv, args.top_k_attrs, attr_pool=getattr(args, "attr_pool", "morphology"))
    features = _load_bundle(features_npz)
    tile_ids = [str(tile_id) for tile_id in features["tile_ids"].tolist()]
    uni_features = np.asarray(features["uni"], dtype=np.float32)
    sweep_root = ensure_directory(out_dir / "sweep")
    tile_shard_index, tile_shard_count = _validate_tile_shard(args.tile_shard_index, args.tile_shard_count)
    fixed_tile_ids = _load_fixed_tile_ids(getattr(args, "fixed_tile_ids", None))
    bundle = load_inference_bundle(
        checkpoint_dir=args.checkpoint_dir,
        config_path=args.config_path,
        data_root=args.data_root,
        exp_channels_dir=args.exp_channels_dir,
        num_steps=args.num_steps,
    )

    for attr in attrs:
        attr_dir = ensure_directory(sweep_root / attr)
        targeted_direction = _load_probe_direction(out_dir, attr)
        random_direction = random_unit_direction(targeted_direction.size, seed=args.seed + sum(ord(ch) for ch in attr))
        np.save(attr_dir / "w_targeted.npy", targeted_direction)
        np.save(attr_dir / "w_random.npy", random_direction)

        metrics_rows: list[dict[str, object]] = []
        all_tile_ids = fixed_tile_ids if fixed_tile_ids is not None else _select_sweep_tiles(labels_npz, attr, args.k_tiles, args.seed)
        selected_tile_ids = _shard_tile_ids(all_tile_ids, tile_shard_index, tile_shard_count)
        for tile_id in selected_tile_ids:
            tile_index = tile_ids.index(tile_id)
            base_uni = uni_features[tile_index]
            for direction_name, direction in (("targeted", targeted_direction), ("random", random_direction)):
                tile_dir = ensure_directory(attr_dir / tile_id / direction_name)
                for alpha, uni_edit in zip(args.alphas, sweep_uni(base_uni, direction, args.alphas), strict=True):
                    out_path = tile_dir / f"alpha_{float(alpha):+.2f}.png"
                    generate_with_uni_override(
                        GenSpec(tile_id=tile_id, uni=uni_edit, out_path=out_path),
                        checkpoint_dir=args.checkpoint_dir,
                        config_path=args.config_path,
                        data_root=args.data_root,
                        exp_channels_dir=args.exp_channels_dir,
                        num_steps=args.num_steps,
                        guidance_scale=args.guidance_scale,
                        seed=args.seed,
                        bundle=bundle,
                    )
                    morph, appearance = _image_metric_rows(out_path)
                    metrics_rows.append(
                        {
                            "tile_id": tile_id,
                            "direction": direction_name,
                            "alpha": float(alpha),
                            "target_attr": attr,
                            "image_path": str(out_path),
                            "target_value": _target_value_for_attr(attr, morph=morph, appearance=appearance),
                            **{f"morpho.{name}": float(value) for name, value in morph.items()},
                            **{name: float(value) for name, value in appearance.items()},
                        }
                    )
        _write_rows_csv(_metrics_output_path(attr_dir, tile_shard_index, tile_shard_count), metrics_rows)
        aggregate_metrics_path = _aggregate_metrics(attr_dir)
        if aggregate_metrics_path is not None:
            _summarize_slopes(aggregate_metrics_path, attr_dir / "slope_summary.json", attr)


def run_null(args: argparse.Namespace) -> None:
    out_dir = ensure_directory(args.out_dir)
    probe_csv = out_dir / "probe_results.csv"
    labels_npz = out_dir / "labels.npz"
    features_npz = out_dir / "features.npz"
    if not probe_csv.is_file() or not labels_npz.is_file() or not features_npz.is_file():
        raise FileNotFoundError("run `probe` first so probe_results.csv, labels.npz, and features.npz exist")

    attrs = _select_sweep_attrs(probe_csv, args.top_k_attrs, attr_pool=getattr(args, "attr_pool", "morphology"))
    features = _load_bundle(features_npz)
    tile_ids = [str(tile_id) for tile_id in features["tile_ids"].tolist()]
    uni_features = np.asarray(features["uni"], dtype=np.float32)
    null_root = ensure_directory(out_dir / "null")
    full_null_root = Path(args.full_null_root)
    tile_shard_index, tile_shard_count = _validate_tile_shard(args.tile_shard_index, args.tile_shard_count)
    fixed_tile_ids = _load_fixed_tile_ids(getattr(args, "fixed_tile_ids", None))
    bundle = load_inference_bundle(
        checkpoint_dir=args.checkpoint_dir,
        config_path=args.config_path,
        data_root=args.data_root,
        exp_channels_dir=args.exp_channels_dir,
        num_steps=args.num_steps,
    )

    for attr in attrs:
        attr_dir = ensure_directory(null_root / attr)
        targeted_direction = _load_probe_direction(out_dir, attr)
        random_direction = random_unit_direction(targeted_direction.size, seed=args.seed + sum(ord(ch) for ch in attr) + 7)
        np.save(attr_dir / "w_targeted.npy", targeted_direction)
        np.save(attr_dir / "w_random.npy", random_direction)

        metrics_rows: list[dict[str, object]] = []
        all_tile_ids = fixed_tile_ids if fixed_tile_ids is not None else _select_sweep_tiles(labels_npz, attr, args.k_tiles, args.seed)
        selected_tile_ids = _shard_tile_ids(all_tile_ids, tile_shard_index, tile_shard_count)
        for tile_id in selected_tile_ids:
            tile_index = tile_ids.index(tile_id)
            base_uni = uni_features[tile_index]
            for condition_name, uni_edit in (
                ("targeted", null_uni(base_uni, targeted_direction)),
                ("random", null_uni(base_uni, random_direction)),
            ):
                out_path = ensure_directory(attr_dir / tile_id) / f"{condition_name}.png"
                generate_with_uni_override(
                    GenSpec(tile_id=tile_id, uni=uni_edit, out_path=out_path),
                    checkpoint_dir=args.checkpoint_dir,
                    config_path=args.config_path,
                    data_root=args.data_root,
                    exp_channels_dir=args.exp_channels_dir,
                    num_steps=args.num_steps,
                    guidance_scale=args.guidance_scale,
                    seed=args.seed,
                    bundle=bundle,
                )
                morph, appearance = _image_metric_rows(out_path)
                metrics_rows.append(
                    {
                        "tile_id": tile_id,
                        "condition": condition_name,
                        "target_attr": attr,
                        "image_path": str(out_path),
                        "target_value": _target_value_for_attr(attr, morph=morph, appearance=appearance),
                        **{f"morpho.{name}": float(value) for name, value in morph.items()},
                        **{name: float(value) for name, value in appearance.items()},
                    }
                )
            full_null_path = full_null_root / tile_id / "tme_only.png"
            if full_null_path.is_file():
                morph, appearance = _image_metric_rows(full_null_path)
                metrics_rows.append(
                    {
                        "tile_id": tile_id,
                        "condition": "full_uni_null",
                        "target_attr": attr,
                        "image_path": str(full_null_path),
                        "target_value": _target_value_for_attr(attr, morph=morph, appearance=appearance),
                        **{f"morpho.{name}": float(value) for name, value in morph.items()},
                        **{name: float(value) for name, value in appearance.items()},
                    }
                )
        _write_rows_csv(_metrics_output_path(attr_dir, tile_shard_index, tile_shard_count), metrics_rows)
        aggregate_metrics_path = _aggregate_metrics(attr_dir)
        if aggregate_metrics_path is not None:
            _summarize_nulls(aggregate_metrics_path, attr_dir / "null_comparison.json", attr)
