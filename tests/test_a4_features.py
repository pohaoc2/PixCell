"""Unit tests for a4_uni_probe.features."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.a4_uni_probe.features import (
    TME_FEATURE_DIM,
    TME_FEATURE_NAMES,
    build_tme_baseline_features,
    build_uni_features,
)


def _write_binary_png(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), (mask.astype(np.uint8) * 255))


def _write_float_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array.astype(np.float32))


def test_build_uni_features_stacks_rows_in_tile_order(tmp_path: Path):
    features_dir = tmp_path / "features"
    features_dir.mkdir()
    np.save(features_dir / "tile_b_uni.npy", np.array([3.0, 4.0], dtype=np.float32))
    np.save(features_dir / "tile_a_uni.npy", np.array([1.0, 2.0], dtype=np.float32))

    matrix = build_uni_features(features_dir, ["tile_a", "tile_b"])
    assert matrix.shape == (2, 2)
    np.testing.assert_allclose(matrix[0], np.array([1.0, 2.0], dtype=np.float32))
    np.testing.assert_allclose(matrix[1], np.array([3.0, 4.0], dtype=np.float32))


def test_build_tme_baseline_features_uses_mean_std_and_zero_fills_missing(tmp_path: Path):
    exp_root = tmp_path / "exp_channels"
    tile_id = "tile_0"
    binary = np.zeros((256, 256), dtype=np.uint8)
    binary[:128, :] = 1
    for channel_name in (
        "cell_type_cancer",
        "cell_type_healthy",
        "cell_type_immune",
        "cell_state_prolif",
        "cell_state_nonprolif",
        "cell_state_dead",
    ):
        _write_binary_png(exp_root / channel_name / f"{tile_id}.png", binary)

    _write_float_npy(exp_root / "vasculature" / f"{tile_id}.npy", np.ones((256, 256), dtype=np.float32))
    _write_float_npy(exp_root / "oxygen" / f"{tile_id}.npy", np.full((256, 256), 0.25, dtype=np.float32))
    # glucose intentionally missing

    matrix = build_tme_baseline_features(exp_root, [tile_id])
    assert matrix.shape == (1, TME_FEATURE_DIM)
    assert len(TME_FEATURE_NAMES) == TME_FEATURE_DIM
    glucose_mean_idx = TME_FEATURE_NAMES.index("glucose_mean")
    glucose_std_idx = TME_FEATURE_NAMES.index("glucose_std")
    assert matrix[0, glucose_mean_idx] == 0.0
    assert matrix[0, glucose_std_idx] == 0.0
