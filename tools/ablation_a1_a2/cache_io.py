"""JSON cache helpers for the SI A1/A2 unified figure."""
from __future__ import annotations

import datetime
import json
from pathlib import Path


CACHE_VERSION = 1


def _empty_cache() -> dict:
    return {
        "version": CACHE_VERSION,
        "generated": "",
        "tile_ids": [],
        "training_curves": {},
        "metrics": {},
        "params": {},
    }


def load_cache(path: Path) -> dict:
    """Load cache.json, returning a schema-complete empty cache if absent."""
    path = Path(path)
    if not path.exists():
        return _empty_cache()
    cache = json.loads(path.read_text(encoding="utf-8"))
    empty = _empty_cache()
    for key, value in empty.items():
        cache.setdefault(key, value)
    return cache


def save_cache(cache: dict, path: Path) -> None:
    """Persist cache.json with a fresh generation date."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cache["version"] = CACHE_VERSION
    cache["generated"] = datetime.date.today().isoformat()
    path.write_text(json.dumps(cache, indent=2, ensure_ascii=True), encoding="utf-8")


def merge_curves(cache: dict, new_curves: dict) -> dict:
    """Deep-merge training curves into cache['training_curves']."""
    existing = cache.setdefault("training_curves", {})
    for variant, runs in new_curves.items():
        existing.setdefault(variant, {}).update(runs)
    return cache


def merge_metrics(cache: dict, variant: str, metrics: dict) -> dict:
    """Merge metrics for one variant without dropping previously-computed fields."""
    cache.setdefault("metrics", {}).setdefault(variant, {}).update(metrics)
    return cache


def merge_params(cache: dict, params: dict) -> dict:
    """Write or overwrite parameter counts."""
    cache.setdefault("params", {}).update(params)
    return cache


def merge_sensitivity(cache: dict, sensitivity_scores: dict) -> dict:
    """Merge per-variant sensitivity summaries without dropping other variants."""
    cache.setdefault("sensitivity", {}).update(sensitivity_scores)
    return cache


def merge_tsc(cache: dict, variant: str, tsc_score: float, tsc_std: float | None = None) -> dict:
    """Write TSC summary stats into cache['metrics'][variant]."""
    record = cache.setdefault("metrics", {}).setdefault(variant, {})
    record["tsc"] = tsc_score
    if tsc_std is not None:
        record["tsc_std"] = tsc_std
    return cache
