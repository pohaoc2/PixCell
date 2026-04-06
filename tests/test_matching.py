from __future__ import annotations

import numpy as np
import pytest


def test_assign_to_archetype_basic():
    from tools.stage4.matching import assign_to_archetype

    centroids = np.array([[0.0, 0.0], [10.0, 10.0]], dtype=np.float32)
    gn = np.array([[0.5, 0.5], [9.5, 9.5], [0.1, 0.2]], dtype=np.float32)
    assignments = assign_to_archetype(gn, centroids)

    assert list(assignments) == [0, 1, 0]


def test_assign_to_archetype_shape():
    from tools.stage4.matching import assign_to_archetype

    rng = np.random.default_rng(0)
    centroids = rng.standard_normal((4, 16)).astype(np.float32)
    gn = rng.standard_normal((100, 16)).astype(np.float32)
    out = assign_to_archetype(gn, centroids)

    assert out.shape == (100,)
    assert out.min() >= 0 and out.max() <= 3


def test_find_best_params_returns_one_per_covered_archetype():
    from tools.stage4.matching import find_best_params

    centroids = np.array([[0.0, 0.0], [10.0, 10.0]], dtype=np.float32)
    gn = np.array([[0.2, 0.2], [0.5, 0.5], [9.8, 9.8]], dtype=np.float32)
    param_ids = ["p0", "p1", "p2"]
    best = find_best_params(gn, centroids, param_ids)

    assert best[0] == "p0"
    assert best[1] == "p2"


def test_coverage_report_uncovered():
    from tools.stage4.matching import coverage_report

    assignments = np.array([0, 0, 1, 1])
    report = coverage_report(assignments, k=3, param_ids=["a", "b", "c", "d"])

    assert report["counts"][0] == 2
    assert report["counts"][1] == 2
    assert report["counts"][2] == 0
    assert 2 in report["uncovered"]


def test_coverage_report_full_coverage():
    from tools.stage4.matching import coverage_report

    assignments = np.array([0, 1, 2, 0])
    report = coverage_report(assignments, k=3, param_ids=["a", "b", "c", "d"])

    assert report["uncovered"] == []

