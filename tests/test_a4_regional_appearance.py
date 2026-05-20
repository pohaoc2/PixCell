"""Tests for the regional appearance pipeline (mask + metrics + slope summary)."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from src.a4_uni_probe.appearance_metrics_regional import (
    appearance_row_regional,
    regional_metric_keys,
)


def _write_synthetic_png_and_sidecar(image_dir: Path, name: str, *, rgb: np.ndarray, contour: list[list[int]]) -> Path:
    from PIL import Image

    image_path = image_dir / name
    Image.fromarray(rgb.astype(np.uint8)).save(image_path)
    sidecar = {"patch": "synthetic", "cells": [{"centroid": [16, 16], "contour": contour, "bbox": [0, 0, 32, 32], "type_cellvit": 0, "type_name": "nuc", "type_prob": 1.0}]}
    (image_path.with_name(f"{image_path.name}.json")).write_text(json.dumps(sidecar), encoding="utf-8")
    return image_path


def test_nucleus_mask_within_bounds(tmp_path: Path):
    from tools.cellvit.contours import nucleus_mask_from_cellvit

    rgb = np.full((64, 64, 3), 200, dtype=np.uint8)
    contour = [[20, 20], [44, 20], [44, 44], [20, 44]]
    image_path = _write_synthetic_png_and_sidecar(tmp_path, "tile.png", rgb=rgb, contour=contour)
    mask = nucleus_mask_from_cellvit(image_path, (64, 64))
    assert mask.dtype == np.bool_
    assert mask.shape == (64, 64)
    area = int(mask.sum())
    assert 100 <= area <= 64 * 64 // 2


def test_compartment_metrics_have_expected_keys(tmp_path: Path):
    rgb = np.random.default_rng(0).integers(80, 220, size=(96, 96, 3), dtype=np.uint8)
    contour = [[10, 10], [70, 10], [70, 70], [10, 70]]
    image_path = _write_synthetic_png_and_sidecar(tmp_path, "tile.png", rgb=rgb, contour=contour)
    row = appearance_row_regional(image_path)
    assert set(row.keys()) == set(regional_metric_keys())
    for key, value in row.items():
        assert isinstance(value, float)
        if not np.isfinite(value):
            continue
        # mean/std stay in plausible HED range, texture coprops are non-negative
        if any(token in key for token in ("contrast", "homogeneity", "energy", "std")):
            assert value >= 0.0


def test_small_compartment_returns_nan(tmp_path: Path):
    rgb = np.random.default_rng(0).integers(80, 220, size=(64, 64, 3), dtype=np.uint8)
    contour = [[0, 0], [3, 0], [3, 3], [0, 3]]  # < 500 px nucleus
    image_path = _write_synthetic_png_and_sidecar(tmp_path, "tile.png", rgb=rgb, contour=contour)
    row = appearance_row_regional(image_path)
    for key, value in row.items():
        if ".nuc." in key:
            assert not np.isfinite(value)


def test_regional_full_mask_matches_global_close(tmp_path: Path):
    """When the nucleus mask covers the whole tile, nuc metrics should equal whole-tile metrics
    computed without compartment splitting (up to median-fill which is a no-op for full mask)."""
    from src.a4_uni_probe.appearance_metrics import appearance_row_for_image

    rgb = np.random.default_rng(123).integers(60, 240, size=(96, 96, 3), dtype=np.uint8)
    contour = [[0, 0], [95, 0], [95, 95], [0, 95]]
    image_path = _write_synthetic_png_and_sidecar(tmp_path, "tile.png", rgb=rgb, contour=contour)

    full_mask = np.ones((96, 96), dtype=bool)
    regional = appearance_row_regional(image_path, nucleus_mask=full_mask)
    global_row = appearance_row_for_image(image_path)

    # H/E mean and std use the raw masked pixels, no quantize -> should be exact.
    assert np.isclose(regional["appearance.nuc.h_mean"], global_row["appearance.h_mean"], atol=1e-5)
    assert np.isclose(regional["appearance.nuc.e_mean"], global_row["appearance.e_mean"], atol=1e-5)
    # Texture: same channel, same quantization range under full-mask (median fill is a no-op).
    assert np.isclose(regional["appearance.nuc.texture_h_contrast"], global_row["appearance.texture_h_contrast"], rtol=1e-3)


def _write_sweep_with_regional_inputs(out_dir: Path, attr: str, image_paths: list[Path]) -> None:
    attr_dir = out_dir / "sweep" / attr
    attr_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    rows: list[dict[str, object]] = []
    for i, image_path in enumerate(image_paths):
        for direction in ("targeted", "random"):
            for alpha in (-1.0, 0.0, 1.0):
                rows.append({
                    "tile_id": f"tile_{i}",
                    "direction": direction,
                    "alpha": alpha,
                    "target_attr": attr,
                    "image_path": str(image_path),
                    "target_value": float(rng.normal()),
                })
    with (attr_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_run_regional_emits_expected_schema(tmp_path: Path):
    from src.a4_uni_probe.recompute_regional import run_regional

    rgb = np.random.default_rng(7).integers(80, 220, size=(64, 64, 3), dtype=np.uint8)
    contour = [[8, 8], [54, 8], [54, 54], [8, 54]]
    image_path = _write_synthetic_png_and_sidecar(tmp_path, "tile_0.png", rgb=rgb, contour=contour)
    _write_sweep_with_regional_inputs(tmp_path, "texture_h_contrast", [image_path])

    args = argparse.Namespace(out_dir=tmp_path)
    paths = run_regional(args)

    rows = list(csv.DictReader(paths["regional_summary"].open(encoding="utf-8")))
    assert len(rows) == len(regional_metric_keys())  # 1 attr * 20 metric keys

    side_by_side = list(csv.DictReader(paths["global_vs_regional"].open(encoding="utf-8")))
    expected_metric_count = len(regional_metric_keys()) // 2  # 10 base metrics
    assert len(side_by_side) == expected_metric_count
    for row in side_by_side:
        assert row["attr"] == "texture_h_contrast"
        assert row["metric"].startswith("appearance.")
        assert "global_targeted_slope" in row
        assert "nuc_targeted_slope" in row
        assert "stroma_targeted_slope" in row
