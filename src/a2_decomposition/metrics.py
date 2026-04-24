"""Metric adapters and summaries for the UNI/TME decomposition task."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.a2_decomposition.main import DEFAULT_MODES, DEFAULT_OUT_DIR
from tools.ablation_report.shared import METRIC_SPEC_BY_KEY, TRADEOFF_METRIC_ORDER
from tools.stage3.ablation_cache import load_manifest


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GENERATED_ROOT = DEFAULT_OUT_DIR / "generated"
DEFAULT_METRICS_ROOT = DEFAULT_OUT_DIR / "decomposition_metrics"
DEFAULT_SUMMARY_CSV = DEFAULT_OUT_DIR / "decomposition_summary.csv"
DEFAULT_FUD_JSON = DEFAULT_OUT_DIR / "fud_scores.json"
DEFAULT_REPRESENTATIVE_JSON = DEFAULT_OUT_DIR / "representative_tile.json"
DECOMPOSITION_METRICS = ("fud", "lpips", "pq", "dice", "style_hed")
TILE_SELECTION_METRICS = ("lpips", "pq", "dice", "style_hed")
MODE_LABELS = {
    "uni_plus_tme": "UNI+TME",
    "uni_only": "UNI only",
    "tme_only": "TME only",
    "neither": "Neither",
}
MODE_KEYS = tuple(mode.name for mode in DEFAULT_MODES)
MODE_ACTIVE_GROUPS = {mode_key: (mode_key,) for mode_key in MODE_KEYS}


@dataclass(frozen=True)
class ModeMetricSummary:
    """Aggregate value for one metric and decomposition mode."""

    mode: str
    metric: str
    mean: float | None
    sd: float | None
    n: int
    ci95_low: float | None
    ci95_high: float | None
    direction: str


def complete_generated_tile_ids(generated_root: Path) -> list[str]:
    """Return tile directories containing all four decomposition PNGs."""
    generated_root = Path(generated_root)
    if not generated_root.is_dir():
        return []
    tile_ids: list[str] = []
    for tile_dir in sorted(path for path in generated_root.iterdir() if path.is_dir()):
        if all((tile_dir / f"{mode_key}.png").is_file() for mode_key in MODE_KEYS):
            tile_ids.append(tile_dir.name)
    return tile_ids


def _relative_path(path: Path, root: Path) -> str:
    return os.path.relpath(path.resolve(), start=root.resolve()).replace(os.sep, "/")


def write_decomposition_manifest(
    *,
    tile_id: str,
    generated_root: Path,
    metrics_root: Path,
) -> Path:
    """Write one manifest that maps four decomposition modes to generated PNGs."""
    generated_root = Path(generated_root)
    metrics_root = Path(metrics_root)
    tile_dir = generated_root / tile_id
    cache_dir = metrics_root / tile_id
    cache_dir.mkdir(parents=True, exist_ok=True)

    entries: list[dict[str, Any]] = []
    for mode_key in MODE_KEYS:
        image_path = tile_dir / f"{mode_key}.png"
        if not image_path.is_file():
            raise FileNotFoundError(f"missing decomposition image: {image_path}")
        entries.append(
            {
                "active_groups": list(MODE_ACTIVE_GROUPS[mode_key]),
                "condition_label": MODE_LABELS[mode_key],
                "image_label": MODE_LABELS[mode_key],
                "image_path": _relative_path(image_path, cache_dir),
            }
        )

    manifest = {
        "version": 1,
        "tile_id": tile_id,
        "group_names": list(MODE_KEYS),
        "decomposition_modes": [
            {
                "mode": mode.name,
                "label": MODE_LABELS[mode.name],
                "use_uni": mode.use_uni,
                "use_tme": mode.use_tme,
            }
            for mode in DEFAULT_MODES
        ],
        "sections": [
            {
                "title": "UNI/TME decomposition",
                "subset_size": 1,
                "entries": entries,
            }
        ],
    }
    out_path = cache_dir / "manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return out_path


def build_decomposition_metric_manifests(
    *,
    generated_root: Path = DEFAULT_GENERATED_ROOT,
    metrics_root: Path = DEFAULT_METRICS_ROOT,
) -> list[Path]:
    """Build manifest-style metric caches for all complete decomposition tiles."""
    tile_ids = complete_generated_tile_ids(generated_root)
    if not tile_ids:
        raise FileNotFoundError(f"no complete four-mode tiles found under {generated_root}")
    return [
        write_decomposition_manifest(
            tile_id=tile_id,
            generated_root=generated_root,
            metrics_root=metrics_root,
        )
        for tile_id in tile_ids
    ]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_tile_metric_records(metrics_root: Path) -> dict[str, dict[str, dict[str, float]]]:
    """Load per-tile decomposition metrics keyed by tile then mode."""
    metrics_root = Path(metrics_root)
    records: dict[str, dict[str, dict[str, float]]] = {}
    if not metrics_root.is_dir():
        return records

    for metrics_path in sorted(metrics_root.glob("*/metrics.json")):
        payload = _load_json(metrics_path)
        tile_id = str(payload.get("tile_id") or metrics_path.parent.name)
        per_condition = payload.get("per_condition", {})
        if not isinstance(per_condition, dict):
            continue
        tile_record: dict[str, dict[str, float]] = {}
        for mode_key in MODE_KEYS:
            raw = per_condition.get(mode_key)
            if not isinstance(raw, dict):
                continue
            values: dict[str, float] = {}
            for metric_key in DECOMPOSITION_METRICS:
                value = raw.get(metric_key)
                if value is None:
                    continue
                values[metric_key] = float(value)
            if values:
                tile_record[mode_key] = values
        if tile_record:
            records[tile_id] = tile_record
    return records


def load_fud_scores(fud_json: Path | None) -> dict[str, float]:
    """Load mode-level FUD scores from common JSON payload shapes."""
    if fud_json is None or not Path(fud_json).is_file():
        return {}
    payload = _load_json(Path(fud_json))
    if isinstance(payload.get("per_condition"), dict):
        payload = payload["per_condition"]
    if isinstance(payload.get("scores"), dict):
        payload = payload["scores"]
    scores: dict[str, float] = {}
    for mode_key in MODE_KEYS:
        value = payload.get(mode_key)
        if value is None:
            continue
        scores[mode_key] = float(value)
    return scores


def _real_he_path(orion_root: Path, tile_id: str) -> Path:
    for ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
        path = orion_root / "he" / f"{tile_id}{ext}"
        if path.is_file():
            return path
    raise FileNotFoundError(f"missing real H&E image for tile {tile_id!r} under {orion_root / 'he'}")


def _mode_image_paths_from_manifest(tile_dir: Path) -> dict[str, Path]:
    manifest = load_manifest(tile_dir)
    paths: dict[str, Path] = {}
    for section in manifest.get("sections", []):
        for entry in section.get("entries", []):
            active_groups = entry.get("active_groups") or []
            if len(active_groups) != 1:
                continue
            mode_key = str(active_groups[0])
            if mode_key in MODE_KEYS:
                paths[mode_key] = tile_dir / entry["image_path"]
    return paths


def compute_decomposition_fud(
    *,
    metrics_root: Path = DEFAULT_METRICS_ROOT,
    orion_root: Path = ROOT / "data" / "orion-crc33",
    uni_model: Path = ROOT / "pretrained_models" / "uni-2h",
    device: str = "cuda",
    batch_size: int = 64,
    out_json: Path = DEFAULT_FUD_JSON,
) -> dict[str, float]:
    """Compute UNI Fréchet distance for each decomposition mode."""
    from tools.compute_fid import (
        ImageFeatureRecord,
        all_features_cached,
        compute_fid_from_stats,
        compute_statistics,
        extract_uni_features,
        load_uni_extractor,
        resolve_device,
    )
    from tools.stage3.ablation_cache import list_cached_tile_ids

    metrics_root = Path(metrics_root)
    orion_root = Path(orion_root)
    tile_ids = list_cached_tile_ids(metrics_root)
    if not tile_ids:
        raise FileNotFoundError(f"no decomposition metric manifests found under {metrics_root}")

    real_records: list[ImageFeatureRecord] = []
    mode_records: dict[str, list[ImageFeatureRecord]] = {mode_key: [] for mode_key in MODE_KEYS}
    for tile_id in tile_ids:
        tile_dir = metrics_root / tile_id
        real_feature = orion_root / "features" / f"{tile_id}_uni.npy"
        real_records.append(
            ImageFeatureRecord(
                image_path=_real_he_path(orion_root, tile_id),
                feature_path=real_feature if real_feature.is_file() else None,
            )
        )
        mode_paths = _mode_image_paths_from_manifest(tile_dir)
        missing = [mode_key for mode_key in MODE_KEYS if mode_key not in mode_paths]
        if missing:
            raise ValueError(f"tile {tile_id} missing mode images in manifest: {', '.join(missing)}")
        for mode_key in MODE_KEYS:
            mode_records[mode_key].append(
                ImageFeatureRecord(
                    image_path=mode_paths[mode_key],
                    feature_path=tile_dir / "features" / mode_key / f"{mode_key}_uni.npy",
                )
            )

    device = resolve_device(device)
    all_records = list(real_records)
    for records in mode_records.values():
        all_records.extend(records)
    extractor = None if all_features_cached(all_records) else load_uni_extractor(uni_model=Path(uni_model), device=device)

    real_features = extract_uni_features(real_records, extractor=extractor, batch_size=batch_size)
    real_mu, real_sigma = compute_statistics(real_features)

    scores: dict[str, float] = {}
    for mode_key in MODE_KEYS:
        gen_features = extract_uni_features(mode_records[mode_key], extractor=extractor, batch_size=batch_size)
        gen_mu, gen_sigma = compute_statistics(gen_features)
        scores[mode_key] = compute_fid_from_stats(real_mu, real_sigma, gen_mu, gen_sigma)

    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(scores, indent=2) + "\n", encoding="utf-8")
    return scores


def _direction(metric_key: str) -> str:
    spec = METRIC_SPEC_BY_KEY.get(metric_key)
    higher_is_better = spec.higher_is_better if spec is not None else metric_key not in {"fud", "style_hed"}
    return "up" if higher_is_better else "down"


def _mean_sd_ci(values: list[float]) -> tuple[float | None, float | None, float | None, float | None]:
    if not values:
        return None, None, None, None
    arr = np.asarray(values, dtype=np.float64)
    mean = float(arr.mean())
    sd = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    if arr.size > 1:
        delta = 1.96 * sd / math.sqrt(float(arr.size))
    else:
        delta = 0.0
    return mean, sd, mean - delta, mean + delta


def summarize_decomposition_metrics(
    *,
    metrics_root: Path = DEFAULT_METRICS_ROOT,
    fud_json: Path | None = DEFAULT_FUD_JSON,
) -> list[ModeMetricSummary]:
    """Aggregate FUD plus per-tile metrics across decomposition modes."""
    tile_records = load_tile_metric_records(metrics_root)
    fud_scores = load_fud_scores(fud_json)
    rows: list[ModeMetricSummary] = []

    for mode_key in MODE_KEYS:
        for metric_key in DECOMPOSITION_METRICS:
            if metric_key == "fud":
                value = fud_scores.get(mode_key)
                rows.append(
                    ModeMetricSummary(
                        mode=mode_key,
                        metric=metric_key,
                        mean=value,
                        sd=None,
                        n=len(tile_records),
                        ci95_low=None,
                        ci95_high=None,
                        direction=_direction(metric_key),
                    )
                )
                continue

            values = [
                float(mode_metrics[metric_key])
                for tile in tile_records.values()
                for current_mode, mode_metrics in tile.items()
                if current_mode == mode_key and metric_key in mode_metrics
            ]
            mean, sd, ci_low, ci_high = _mean_sd_ci(values)
            rows.append(
                ModeMetricSummary(
                    mode=mode_key,
                    metric=metric_key,
                    mean=mean,
                    sd=sd,
                    n=len(values),
                    ci95_low=ci_low,
                    ci95_high=ci_high,
                    direction=_direction(metric_key),
                )
            )
    return rows


def write_summary_csv(rows: list[ModeMetricSummary], out_csv: Path = DEFAULT_SUMMARY_CSV) -> Path:
    """Write aggregate decomposition metrics to CSV."""
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ("mode", "metric", "mean", "sd", "n", "ci95_low", "ci95_high", "direction")
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "mode": row.mode,
                    "metric": row.metric,
                    "mean": "" if row.mean is None else f"{row.mean:.6f}",
                    "sd": "" if row.sd is None else f"{row.sd:.6f}",
                    "n": row.n,
                    "ci95_low": "" if row.ci95_low is None else f"{row.ci95_low:.6f}",
                    "ci95_high": "" if row.ci95_high is None else f"{row.ci95_high:.6f}",
                    "direction": row.direction,
                }
            )
    return out_csv


def _oriented_value(metric_key: str, value: float) -> float:
    return float(value) if _direction(metric_key) == "up" else -float(value)


def select_representative_tile(
    *,
    metrics_root: Path = DEFAULT_METRICS_ROOT,
    fallback_tile_id: str = "1792_10496",
) -> tuple[str | None, float | None]:
    """Select a medoid-like tile from oriented per-tile metric vectors."""
    tile_records = load_tile_metric_records(metrics_root)
    vectors: dict[str, np.ndarray] = {}
    for tile_id, per_mode in tile_records.items():
        values: list[float] = []
        complete = True
        for mode_key in MODE_KEYS:
            mode_values = per_mode.get(mode_key)
            if mode_values is None:
                complete = False
                break
            for metric_key in TILE_SELECTION_METRICS:
                value = mode_values.get(metric_key)
                if value is None:
                    complete = False
                    break
                values.append(_oriented_value(metric_key, float(value)))
            if not complete:
                break
        if complete:
            vectors[tile_id] = np.asarray(values, dtype=np.float64)

    if not vectors:
        if (Path(metrics_root) / fallback_tile_id).exists():
            return fallback_tile_id, None
        return None, None

    matrix = np.stack(list(vectors.values()), axis=0)
    center = np.median(matrix, axis=0)
    scale = np.std(matrix, axis=0)
    scale[scale == 0.0] = 1.0

    best_tile: str | None = None
    best_distance: float | None = None
    for tile_id, vector in vectors.items():
        z = (vector - center) / scale
        distance = float(np.dot(z, z))
        if best_distance is None or distance < best_distance:
            best_tile = tile_id
            best_distance = distance
    return best_tile, best_distance


def write_representative_tile_json(
    *,
    tile_id: str,
    score: float | None,
    out_json: Path = DEFAULT_REPRESENTATIVE_JSON,
) -> Path:
    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "tile_id": tile_id,
        "selection_score": score,
        "method": "z-scored median medoid over LPIPS/PQ/DICE/HED across four modes",
    }
    out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return out_json


def load_summary_csv(summary_csv: Path) -> dict[str, dict[str, ModeMetricSummary]]:
    """Load summary CSV keyed by mode then metric."""
    out: dict[str, dict[str, ModeMetricSummary]] = {}
    with Path(summary_csv).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            mode = row["mode"]
            metric = row["metric"]
            out.setdefault(mode, {})[metric] = ModeMetricSummary(
                mode=mode,
                metric=metric,
                mean=float(row["mean"]) if row.get("mean") else None,
                sd=float(row["sd"]) if row.get("sd") else None,
                n=int(row["n"]) if row.get("n") else 0,
                ci95_low=float(row["ci95_low"]) if row.get("ci95_low") else None,
                ci95_high=float(row["ci95_high"]) if row.get("ci95_high") else None,
                direction=row.get("direction") or _direction(metric),
            )
    return out


def effect_decomposition(summary: dict[str, dict[str, ModeMetricSummary]]) -> dict[str, dict[str, float | None]]:
    """Compute oriented UNI/TME/interaction effects from aggregate summaries."""
    effects = {"UNI effect": {}, "TME effect": {}, "Interaction": {}}
    for metric_key in DECOMPOSITION_METRICS:
        values: dict[str, float] = {}
        for mode_key in MODE_KEYS:
            record = summary.get(mode_key, {}).get(metric_key)
            if record is None or record.mean is None:
                continue
            values[mode_key] = _oriented_value(metric_key, record.mean)
        if set(MODE_KEYS).issubset(values):
            effects["UNI effect"][metric_key] = values["uni_plus_tme"] - values["tme_only"]
            effects["TME effect"][metric_key] = values["uni_plus_tme"] - values["uni_only"]
            effects["Interaction"][metric_key] = (
                values["uni_plus_tme"] - values["uni_only"] - values["tme_only"] + values["neither"]
            )
        else:
            for effect_name in effects:
                effects[effect_name][metric_key] = None
    return effects


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare and summarize UNI/TME decomposition metrics")
    parser.add_argument("--generated-root", type=Path, default=DEFAULT_GENERATED_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_METRICS_ROOT)
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--compute-fud", action="store_true")
    parser.add_argument("--metrics-root", type=Path, default=DEFAULT_METRICS_ROOT)
    parser.add_argument("--fud-json", type=Path, default=DEFAULT_FUD_JSON)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_SUMMARY_CSV)
    parser.add_argument("--representative-json", type=Path, default=DEFAULT_REPRESENTATIVE_JSON)
    parser.add_argument("--orion-root", type=Path, default=ROOT / "data" / "orion-crc33")
    parser.add_argument("--uni-model", type=Path, default=ROOT / "pretrained_models" / "uni-2h")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args(argv)

    if args.compute_fud:
        scores = compute_decomposition_fud(
            metrics_root=args.metrics_root,
            orion_root=args.orion_root,
            uni_model=args.uni_model,
            device=args.device,
            batch_size=args.batch_size,
            out_json=args.fud_json,
        )
        print(f"Wrote FUD scores for {len(scores)} modes -> {args.fud_json}")
        return 0

    if args.summarize:
        rows = summarize_decomposition_metrics(metrics_root=args.metrics_root, fud_json=args.fud_json)
        out_csv = write_summary_csv(rows, args.out_csv)
        tile_id, score = select_representative_tile(metrics_root=args.metrics_root)
        if tile_id is not None:
            write_representative_tile_json(tile_id=tile_id, score=score, out_json=args.representative_json)
        print(f"Wrote summary -> {out_csv}")
        if tile_id is not None:
            print(f"Representative tile -> {tile_id}")
        return 0

    paths = build_decomposition_metric_manifests(generated_root=args.generated_root, metrics_root=args.out_dir)
    print(f"Wrote {len(paths)} decomposition metric manifest(s) under {args.out_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
