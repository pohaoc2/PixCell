from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.a2_decomposition.metrics import (
    DECOMPOSITION_METRICS,
    MODE_KEYS,
    build_decomposition_metric_manifests,
    complete_generated_tile_ids,
    effect_decomposition,
    load_summary_csv,
    select_representative_tile,
    summarize_decomposition_metrics,
    write_summary_csv,
)


def _write_rgb(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((12, 12, 3), value, dtype=np.uint8)).save(path)


def _write_complete_generated_tile(root: Path, tile_id: str, base_value: int = 80) -> None:
    for idx, mode_key in enumerate(MODE_KEYS):
        _write_rgb(root / tile_id / f"{mode_key}.png", base_value + idx * 20)


def _write_metrics_json(metrics_root: Path, tile_id: str, offset: float) -> None:
    per_condition = {}
    for idx, mode_key in enumerate(MODE_KEYS):
        per_condition[mode_key] = {
            "lpips": 0.40 + offset + idx * 0.01,
            "pq": 0.20 + offset + idx * 0.02,
            "dice": 0.50 + offset + idx * 0.03,
            "style_hed": 0.08 + offset + idx * 0.004,
        }
    path = metrics_root / tile_id / "metrics.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"version": 2, "tile_id": tile_id, "per_condition": per_condition}, indent=2),
        encoding="utf-8",
    )


def test_build_decomposition_metric_manifests_maps_four_modes(tmp_path: Path):
    generated_root = tmp_path / "generated"
    metrics_root = tmp_path / "metrics"
    _write_complete_generated_tile(generated_root, "tile_a")
    _write_rgb(generated_root / "tile_incomplete" / "uni_plus_tme.png", 100)

    assert complete_generated_tile_ids(generated_root) == ["tile_a"]
    manifests = build_decomposition_metric_manifests(
        generated_root=generated_root,
        metrics_root=metrics_root,
    )

    assert len(manifests) == 1
    payload = json.loads(manifests[0].read_text(encoding="utf-8"))
    entries = payload["sections"][0]["entries"]
    assert [entry["condition_label"] for entry in entries] == [
        "UNI+TME",
        "UNI only",
        "TME only",
        "Neither",
    ]
    assert [entry["active_groups"] for entry in entries] == [[mode] for mode in MODE_KEYS]
    for entry in entries:
        assert (metrics_root / "tile_a" / entry["image_path"]).is_file()


def test_summarize_decomposition_metrics_with_fud(tmp_path: Path):
    metrics_root = tmp_path / "metrics"
    _write_metrics_json(metrics_root, "tile_a", 0.0)
    _write_metrics_json(metrics_root, "tile_b", 0.1)
    fud_json = tmp_path / "fud_scores.json"
    fud_json.write_text(
        json.dumps({mode_key: 10.0 + idx for idx, mode_key in enumerate(MODE_KEYS)}),
        encoding="utf-8",
    )

    rows = summarize_decomposition_metrics(metrics_root=metrics_root, fud_json=fud_json)
    out_csv = write_summary_csv(rows, tmp_path / "summary.csv")
    summary = load_summary_csv(out_csv)

    assert set(summary) == set(MODE_KEYS)
    for mode_key in MODE_KEYS:
        assert set(summary[mode_key]) == set(DECOMPOSITION_METRICS)
        assert summary[mode_key]["fud"].n == 2
        assert summary[mode_key]["fud"].direction == "down"
        assert summary[mode_key]["pq"].direction == "up"
        assert summary[mode_key]["lpips"].n == 2
    assert summary["uni_plus_tme"]["fud"].mean == pytest.approx(10.0)
    assert summary["uni_plus_tme"]["lpips"].mean == pytest.approx(0.45)


def test_effect_decomposition_orients_lower_is_better_metrics(tmp_path: Path):
    metrics_root = tmp_path / "metrics"
    _write_metrics_json(metrics_root, "tile_a", 0.0)
    fud_json = tmp_path / "fud_scores.json"
    fud_json.write_text(
        json.dumps(
            {
                "uni_plus_tme": 5.0,
                "uni_only": 7.0,
                "tme_only": 8.0,
                "neither": 9.0,
            }
        ),
        encoding="utf-8",
    )
    summary_csv = write_summary_csv(
        summarize_decomposition_metrics(metrics_root=metrics_root, fud_json=fud_json),
        tmp_path / "summary.csv",
    )

    effects = effect_decomposition(load_summary_csv(summary_csv))

    assert effects["UNI effect"]["fud"] == pytest.approx(3.0)
    assert effects["TME effect"]["fud"] == pytest.approx(2.0)
    assert effects["TME effect"]["pq"] == pytest.approx(-0.02)


def test_select_representative_tile_is_deterministic(tmp_path: Path):
    metrics_root = tmp_path / "metrics"
    _write_metrics_json(metrics_root, "tile_a", 0.0)
    _write_metrics_json(metrics_root, "tile_b", 0.5)
    _write_metrics_json(metrics_root, "tile_c", 1.0)

    tile_id, score = select_representative_tile(metrics_root=metrics_root)

    assert tile_id == "tile_b"
    assert score == pytest.approx(0.0)
