from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _write_png(path: Path, value: float, size: int = 16) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    total = size * size
    active = int(round(value * total))
    arr = np.zeros((total,), dtype=np.uint8)
    arr[:active] = 255
    arr = arr.reshape(size, size)
    Image.fromarray(arr, mode="L").save(path)


def _write_npy(path: Path, value: float, size: int = 16) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.full((size, size), value, dtype=np.float32))


def test_build_t1_targets_sorts_tiles_and_computes_expected_values(tmp_path: Path):
    from src.a1_mask_targets.main import TARGET_NAMES, build_t1_targets, run_task

    features_dir = tmp_path / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    np.save(features_dir / "2048_0_uni.npy", np.array([2.0, 0.0], dtype=np.float32))
    np.save(features_dir / "0_0_uni.npy", np.array([1.0, 0.0], dtype=np.float32))

    exp_channels = tmp_path / "exp_channels"
    for tile_id, density, cancer, healthy, immune, prolif, nonprolif, dead, vasc, oxy, glu in [
        ("0_0", 0.20, 0.10, 0.05, 0.05, 0.08, 0.10, 0.02, 0.30, 0.80, 0.70),
        ("2048_0", 0.40, 0.20, 0.10, 0.10, 0.12, 0.20, 0.08, 0.50, 0.90, 0.85),
    ]:
        _write_png(exp_channels / "cell_masks" / f"{tile_id}.png", density)
        _write_png(exp_channels / "cell_type_cancer" / f"{tile_id}.png", cancer)
        _write_png(exp_channels / "cell_type_healthy" / f"{tile_id}.png", healthy)
        _write_png(exp_channels / "cell_type_immune" / f"{tile_id}.png", immune)
        _write_png(exp_channels / "cell_state_prolif" / f"{tile_id}.png", prolif)
        _write_png(exp_channels / "cell_state_nonprolif" / f"{tile_id}.png", nonprolif)
        _write_png(exp_channels / "cell_state_dead" / f"{tile_id}.png", dead)
        _write_npy(exp_channels / "vasculature" / f"{tile_id}.npy", vasc)
        _write_npy(exp_channels / "oxygen" / f"{tile_id}.npy", oxy)
        _write_npy(exp_channels / "glucose" / f"{tile_id}.npy", glu)

    tile_ids, matrix = build_t1_targets(features_dir, exp_channels)
    assert tile_ids == ["0_0", "2048_0"]
    assert matrix.shape == (2, len(TARGET_NAMES))
    assert matrix[0, 0] == pytest.approx(0.20, abs=0.02)
    assert matrix[0, 1] == pytest.approx(0.50, abs=0.12)
    assert matrix[0, 8] == pytest.approx(0.80, abs=1e-6)
    assert matrix[0, 9] == pytest.approx(0.70, abs=1e-6)

    outputs = run_task(features_dir, exp_channels, tmp_path / "out")
    assert outputs["matrix"].is_file()
    assert outputs["tile_ids"].is_file()
    assert outputs["stats"].is_file()
    manifest = json.loads(outputs["manifest"].read_text(encoding="utf-8"))
    assert manifest["tile_count"] == 2
