from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _write_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array.astype(np.float32))


def _write_png(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.clip(array, 0, 1) * 255).astype(np.uint8), mode="L").save(path)


def test_block_mean_pool_uniform_array():
    from src.a1_mask_targets_spatial.main import block_mean_pool

    arr = np.ones((16, 16), dtype=np.float32) * 0.5
    pooled = block_mean_pool(arr, grid=4)
    assert pooled.shape == (4, 4)
    np.testing.assert_allclose(pooled, 0.5)


def test_block_mean_pool_left_right_split():
    from src.a1_mask_targets_spatial.main import block_mean_pool

    arr = np.zeros((16, 16), dtype=np.float32)
    arr[:, 8:] = 1.0  # right half hot
    pooled = block_mean_pool(arr, grid=4)
    # 4x4 grid -> columns 0-1 are 0, columns 2-3 are 1
    expected = np.zeros((4, 4), dtype=np.float32)
    expected[:, 2:] = 1.0
    np.testing.assert_allclose(pooled, expected)


def test_block_mean_pool_handles_non_divisor_grid():
    from src.a1_mask_targets_spatial.main import block_mean_pool

    arr = np.zeros((256, 256), dtype=np.float32)
    arr[:, 128:] = 1.0
    pooled = block_mean_pool(arr, grid=7)

    assert pooled.shape == (7, 7)
    assert pooled[:, 4:].mean() > 0.5
    assert pooled[:, :3].mean() < 0.5


def test_compute_tile_patch_targets_shape_and_fractions(tmp_path: Path):
    from src.a1_mask_targets_spatial.main import TARGET_NAMES, compute_tile_patch_targets

    tile_id = "0_0"
    exp_dir = tmp_path / "exp_channels"
    # Cell mask: half-filled tile so denser on the right.
    cell_mask = np.zeros((16, 16), dtype=np.float32)
    cell_mask[:, 8:] = 1.0
    _write_png(exp_dir / "cell_masks" / f"{tile_id}.png", cell_mask)

    # Cancer == cell mask -> 100% of cells are cancer where present.
    _write_png(exp_dir / "cell_type_cancer" / f"{tile_id}.png", cell_mask)
    for empty in ["cell_type_healthy", "cell_type_immune",
                  "cell_state_prolif", "cell_state_nonprolif", "cell_state_dead"]:
        _write_png(exp_dir / empty / f"{tile_id}.png", np.zeros((16, 16), dtype=np.float32))
    _write_npy(exp_dir / "vasculature" / f"{tile_id}.npy", np.zeros((16, 16), dtype=np.float32))
    _write_npy(exp_dir / "oxygen" / f"{tile_id}.npy", np.full((16, 16), 0.5, dtype=np.float32))
    _write_npy(exp_dir / "glucose" / f"{tile_id}.npy", np.full((16, 16), 0.9, dtype=np.float32))

    matrix = compute_tile_patch_targets(exp_dir, tile_id, resolution=16, grid=4)
    assert matrix.shape == (16, len(TARGET_NAMES))
    density_col = TARGET_NAMES.index("cell_density")
    cancer_col = TARGET_NAMES.index("cancer_frac")
    o2_col = TARGET_NAMES.index("oxygen_mean")

    # First two columns of the 4x4 grid: empty -> density 0, cancer fraction 0
    # Last two columns: full density, full cancer fraction.
    matrix_4x4 = matrix.reshape(4, 4, -1)
    np.testing.assert_allclose(matrix_4x4[:, :2, density_col], 0.0, atol=1e-3)
    np.testing.assert_allclose(matrix_4x4[:, 2:, density_col], 1.0, atol=1e-3)
    np.testing.assert_allclose(matrix_4x4[:, 2:, cancer_col], 1.0, atol=1e-3)
    np.testing.assert_allclose(matrix[:, o2_col], 0.5, atol=1e-3)


def test_run_task_writes_full_bundle(tmp_path: Path):
    from src.a1_mask_targets_spatial.main import run_task

    features_dir = tmp_path / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    # Use uni_tokens suffix to mirror real layout.
    for tile_id in ("0_0", "2048_0"):
        np.save(features_dir / f"{tile_id}_uni_tokens.npy", np.zeros((16, 8), dtype=np.float16))

    exp_dir = tmp_path / "exp_channels"
    for tile_id in ("0_0", "2048_0"):
        for chan in ["cell_masks", "cell_type_cancer", "cell_type_healthy", "cell_type_immune",
                     "cell_state_prolif", "cell_state_nonprolif", "cell_state_dead"]:
            _write_png(exp_dir / chan / f"{tile_id}.png", np.zeros((16, 16), dtype=np.float32))
        for chan in ["vasculature", "oxygen", "glucose"]:
            _write_npy(exp_dir / chan / f"{tile_id}.npy", np.zeros((16, 16), dtype=np.float32))

    paths = run_task(features_dir, exp_dir, tmp_path / "out", resolution=16, grid=4)
    matrix = np.load(paths["matrix"])
    assert matrix.shape == (2, 16, 10)
    # Tile IDs file should match cached features.
    tile_text = paths["tile_ids"].read_text(encoding="utf-8").splitlines()
    assert tile_text == ["0_0", "2048_0"]
