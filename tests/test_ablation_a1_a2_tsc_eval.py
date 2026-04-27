from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
import pytest

from tools.ablation_a1_a2.tsc_eval import compute_tsc_tile, dice_score, extract_nuclei_map, run_tsc


def test_dice_score_perfect():
    arr = np.array([[1, 0], [0, 1]], dtype=bool)
    assert dice_score(arr, arr) == pytest.approx(1.0)


def test_dice_score_no_overlap():
    pred = np.array([[1, 0], [0, 0]], dtype=bool)
    gt = np.array([[0, 0], [0, 1]], dtype=bool)
    assert dice_score(pred, gt) == pytest.approx(0.0)


def test_dice_score_both_empty():
    arr = np.zeros((4, 4), dtype=bool)
    assert dice_score(arr, arr) == pytest.approx(1.0)


def test_extract_nuclei_map_shape():
    rng = np.random.default_rng(0)
    he_rgb = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
    nuclei = extract_nuclei_map(he_rgb)
    assert nuclei.shape == (256, 256)
    assert nuclei.dtype == bool


def test_extract_nuclei_map_all_dark_image():
    dark_purple = np.full((64, 64, 3), [80, 40, 100], dtype=np.uint8)
    nuclei = extract_nuclei_map(dark_purple)
    assert nuclei.mean() > 0.3


def test_compute_tsc_tile_range():
    rng = np.random.default_rng(1)
    he_rgb = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
    cell_mask = rng.integers(0, 2, (256, 256), dtype=np.uint8).astype(bool)
    score = compute_tsc_tile(he_rgb, cell_mask)
    assert 0.0 <= score <= 1.0


def test_run_tsc_writes_cache(tmp_path):
    cache_dir = tmp_path / "si_a1_a2"
    cache_dir.mkdir()
    tile_dir = cache_dir / "tiles" / "production"
    tile_dir.mkdir(parents=True)

    fake_rgb = np.full((256, 256, 3), 80, dtype=np.uint8)
    Image.fromarray(fake_rgb).save(tile_dir / "tile_0.png")

    with patch("tools.ablation_a1_a2.tsc_eval._load_codex_cell_mask", return_value=np.ones((256, 256), dtype=bool)):
        run_tsc(
            cache_dir=cache_dir,
            tile_ids=["tile_0"],
            variants=["production"],
            exp_channels_dir=Path("/tmp"),
            image_size=256,
        )

    cache = json.loads((cache_dir / "cache.json").read_text(encoding="utf-8"))
    assert "tsc" in cache["metrics"]["production"]
    assert 0.0 <= cache["metrics"]["production"]["tsc"] <= 1.0
    assert "tsc_std" in cache["metrics"]["production"]