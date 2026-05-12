"""Unit tests for a4_uni_probe.probe."""

from __future__ import annotations

import numpy as np

from src.a4_uni_probe.probe import fit_probes_for_attribute, spatial_bucket_groups


def test_spatial_bucket_groups_use_tile_coordinates():
    groups = spatial_bucket_groups(["0_0", "1024_1024", "4096_0"], 2048)
    assert groups == ["0_0", "0_0", "2_0"]


def test_fit_probes_for_attribute_recovers_linear_signal():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 3)).astype(np.float32)
    coef = np.array([1.5, -2.0, 0.75], dtype=np.float32)
    y = X @ coef
    groups = [str(index // 6) for index in range(30)]
    result = fit_probes_for_attribute(X, y, groups, cv_folds=5)
    assert result.n_valid_folds == 5
    assert result.r2_mean > 0.99
    assert result.coef.shape == (3,)


def test_fit_probes_for_attribute_is_deterministic_for_fixed_inputs():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(25, 4)).astype(np.float32)
    y = (0.3 * X[:, 0] - 0.7 * X[:, 1]).astype(np.float32)
    groups = [str(index // 5) for index in range(25)]
    left = fit_probes_for_attribute(X, y, groups, cv_folds=5)
    right = fit_probes_for_attribute(X, y, groups, cv_folds=5)
    assert left.r2_per_fold == right.r2_per_fold
    np.testing.assert_allclose(left.coef, right.coef)
