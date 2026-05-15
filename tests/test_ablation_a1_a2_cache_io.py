from __future__ import annotations

import pytest

from tools.ablation_a1_a2.cache_io import (
    load_cache,
    merge_curves,
    merge_metrics,
    merge_params,
    merge_sensitivity,
    merge_tsc,
    save_cache,
)


def test_load_empty(tmp_path):
    cache = load_cache(tmp_path / "nonexistent.json")
    assert cache["training_curves"] == {}
    assert cache["metrics"] == {}
    assert cache["tile_ids"] == []


def test_save_and_reload(tmp_path):
    cache_path = tmp_path / "cache.json"
    cache = load_cache(cache_path)
    cache["tile_ids"] = ["t1", "t2"]
    save_cache(cache, cache_path)
    reloaded = load_cache(cache_path)
    assert reloaded["tile_ids"] == ["t1", "t2"]
    assert "generated" in reloaded


def test_merge_curves_additive(tmp_path):
    cache = load_cache(tmp_path / "cache.json")
    merge_curves(cache, {"a1_concat": {"seed_1": [{"step": 50, "loss": 0.1, "grad_norm": 0.02}]}})
    merge_curves(cache, {"a1_concat": {"seed_2": [{"step": 50, "loss": 0.11, "grad_norm": 0.02}]}})
    assert "seed_1" in cache["training_curves"]["a1_concat"]
    assert "seed_2" in cache["training_curves"]["a1_concat"]


def test_merge_metrics(tmp_path):
    cache = load_cache(tmp_path / "cache.json")
    merge_metrics(cache, "production", {"fid": 12.3, "uni_cos": 0.85})
    assert cache["metrics"]["production"]["fid"] == pytest.approx(12.3)


def test_merge_metrics_preserves_existing_fields(tmp_path):
    cache = load_cache(tmp_path / "cache.json")
    merge_tsc(cache, "production", 0.72, 0.05)

    merge_metrics(cache, "production", {"dice": 0.8})

    assert cache["metrics"]["production"]["dice"] == pytest.approx(0.8)
    assert cache["metrics"]["production"]["tsc"] == pytest.approx(0.72)
    assert cache["metrics"]["production"]["tsc_std"] == pytest.approx(0.05)


def test_merge_params(tmp_path):
    cache = load_cache(tmp_path / "cache.json")
    merge_params(cache, {"a1_concat": 12_000_000})
    assert cache["params"]["a1_concat"] == 12_000_000


def test_merge_sensitivity_creates_key(tmp_path):
    cache = load_cache(tmp_path / "cache.json")
    scores = {
        "production": {
            "mean": 0.12,
            "std": 0.03,
            "per_group": {"cell_types": {"mean": 0.12, "std": 0.03, "per_tile": [0.10, 0.14]}},
        },
    }

    merge_sensitivity(cache, scores)

    assert cache["sensitivity"] == scores


def test_merge_sensitivity_overwrites():
    cache = {"sensitivity": {"production": {"mean": 0.99}}}

    merge_sensitivity(cache, {"production": {"mean": 0.01}})

    assert cache["sensitivity"]["production"]["mean"] == pytest.approx(0.01)


def test_merge_sensitivity_preserves_other_variants():
    cache = {"sensitivity": {"production": {"mean": 0.99}}}

    merge_sensitivity(cache, {"a2_off_shelf": {"mean": 0.0}})

    assert cache["sensitivity"]["production"]["mean"] == pytest.approx(0.99)
    assert cache["sensitivity"]["a2_off_shelf"]["mean"] == pytest.approx(0.0)


def test_merge_tsc_writes_metric(tmp_path):
    cache = load_cache(tmp_path / "cache.json")

    merge_tsc(cache, "production", 0.72, 0.05)

    assert cache["metrics"]["production"]["tsc"] == pytest.approx(0.72)
    assert cache["metrics"]["production"]["tsc_std"] == pytest.approx(0.05)
