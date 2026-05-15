"""Unit tests for a4_uni_probe.appearance_metrics."""

from __future__ import annotations

import numpy as np

from src.a4_uni_probe.appearance_metrics import (
    APPEARANCE_METRIC_NAMES,
    appearance_row_for_rgb,
    stain_vector_angle_deg,
)


def _toy_rgb() -> np.ndarray:
    left = np.full((16, 16, 3), [180, 140, 170], dtype=np.uint8)
    right = np.full((16, 16, 3), [235, 190, 205], dtype=np.uint8)
    return np.concatenate([left, right], axis=1)


def test_appearance_row_has_expected_metric_keys_and_finite_stats():
    rgb = _toy_rgb()
    row = appearance_row_for_rgb(rgb, reference_rgb_u8=rgb)
    assert tuple(row.keys()) == APPEARANCE_METRIC_NAMES
    assert np.isfinite(row["appearance.h_mean"])
    assert np.isfinite(row["appearance.e_mean"])
    assert np.isfinite(row["appearance.texture_h_contrast"])
    assert np.isfinite(row["appearance.texture_e_energy"])


def test_stain_vector_angle_is_near_zero_for_identical_inputs():
    rgb = _toy_rgb()
    angle = stain_vector_angle_deg(rgb, rgb)
    assert np.isfinite(angle)
    assert angle < 1e-4


def test_reference_free_row_sets_stain_vector_distance_to_nan():
    rgb = _toy_rgb()
    row = appearance_row_for_rgb(rgb)
    assert np.isnan(row["appearance.stain_vector_angle_deg"])