"""Training-log extraction helpers for the SI A1/A2 unified figure cache."""
from __future__ import annotations

import math
from pathlib import Path

from tools.ablation_a3.aggregate_stability import _read_log


ProductionSource = Path | tuple[Path, ...]


LOG_SOURCES: dict[str, dict[str, ProductionSource]] = {
    "production": {
        "full_seed_42": (
            Path("checkpoints/production_retrain_post_fix/full_seed_42/train_log.jsonl"),
            Path("checkpoints/production_retrain/full_seed_42/train_log.jsonl"),
            Path("checkpoints/pixcell_controlnet_exp_retrain/full_seed_42/train_log.jsonl"),
            Path("checkpoints/pixcell_controlnet_exp/npy_inputs/train_log.log"),
        ),
    },
    "a1_concat": {
        "full_seed_42": (
            Path("checkpoints/concat_95470_0/train_log.jsonl"),
            Path("checkpoints/a1_concat/full_seed_42/train_log.jsonl"),
        ),
        "seed_1": Path("checkpoints/a1_concat/seed_1/train_log.jsonl"),
        "seed_2": Path("checkpoints/a1_concat/seed_2/train_log.jsonl"),
        "seed_3": Path("checkpoints/a1_concat/seed_3/train_log.jsonl"),
    },
    "a1_per_channel": {
        "full_seed_42": Path("checkpoints/a1_per_channel/full_seed_42/train_log.jsonl"),
        "seed_1": Path("checkpoints/a1_per_channel/seed_1/train_log.jsonl"),
        "seed_2": Path("checkpoints/a1_per_channel/seed_2/train_log.jsonl"),
        "seed_3": Path("checkpoints/a1_per_channel/seed_3/train_log.jsonl"),
    },
    "a2_bypass": {
        "full_seed_42": Path("checkpoints/a2_a3/a2_bypass/full_seed_42/train_log.jsonl"),
        "seed_1": Path("checkpoints/a2_a3/a2_bypass/seed_1/train_log.jsonl"),
        "seed_2": Path("checkpoints/a2_a3/a2_bypass/seed_2/train_log.jsonl"),
        "seed_3": Path("checkpoints/a2_a3/a2_bypass/seed_3/train_log.jsonl"),
        "seed_4": Path("checkpoints/a2_a3/a2_bypass/seed_4/train_log.jsonl"),
        "seed_5": Path("checkpoints/a2_a3/a2_bypass/seed_5/train_log.jsonl"),
    },
}


def serialise_float(value: float) -> float | str | None:
    """Return a JSON-stable representation for finite, NaN, and infinite floats."""
    if math.isnan(value):
        return None
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return value


def deserialise_float(value: object) -> float:
    """Invert serialise_float() for figure-side curve aggregation."""
    if value is None:
        return float("nan")
    if value == "inf":
        return float("inf")
    if value == "-inf":
        return float("-inf")
    return float(value)


def extract_run(path: Path) -> list[dict]:
    """Parse one JSONL or plain-text training log. Missing files return an empty list."""
    path = Path(path)
    if not path.exists():
        return []

    result: list[dict] = []
    for entry in _read_log(path):
        try:
            step = int(entry["step"])
        except (KeyError, TypeError, ValueError):
            continue
        loss = serialise_float(float(entry.get("loss", float("nan"))))
        grad_norm = serialise_float(float(entry.get("grad_norm", float("nan"))))
        result.append({"step": step, "loss": loss, "grad_norm": grad_norm})
    return result


def _resolve_log_source(path_or_paths: ProductionSource) -> Path:
    if isinstance(path_or_paths, Path):
        return path_or_paths
    for candidate in path_or_paths:
        if Path(candidate).exists():
            return Path(candidate)
    return Path(path_or_paths[0])


def extract_all_curves(extra_sources: dict[str, dict[str, ProductionSource]] | None = None) -> dict:
    """Return {variant: {run_id: [{step, loss, grad_norm}]}} for available logs."""
    sources: dict[str, dict[str, ProductionSource]] = {variant: dict(runs) for variant, runs in LOG_SOURCES.items()}
    if extra_sources:
        for variant, runs in extra_sources.items():
            sources.setdefault(variant, {}).update(runs)

    curves: dict[str, dict[str, list[dict]]] = {}
    for variant, runs in sources.items():
        variant_curves: dict[str, list[dict]] = {}
        for run_id, path in runs.items():
            resolved_path = _resolve_log_source(path)
            entries = extract_run(resolved_path)
            if entries:
                variant_curves[run_id] = entries
        if variant_curves:
            curves[variant] = variant_curves
    return curves
