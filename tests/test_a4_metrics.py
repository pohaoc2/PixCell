"""Unit tests for a4_uni_probe.metrics."""

from __future__ import annotations

import json

import numpy as np

from src.a4_uni_probe.metrics import morphology_row_for_image


def test_returns_cellvit_values_when_sidecar_present(tmp_path):
    image_path = tmp_path / "sample.png"
    image_path.write_bytes(b"png")
    sidecar = tmp_path / "sample_cellvit_instances.json"
    sidecar.write_text(
        json.dumps(
            {
                "tile_area": 256.0,
                "nuclei": [
                    {"area": 100, "eccentricity": 0.5, "intensity_h": 0.4, "intensity_e": 0.2},
                    {"area": 200, "eccentricity": 0.8, "intensity_h": 0.6, "intensity_e": 0.3},
                ],
            }
        ),
        encoding="utf-8",
    )
    row = morphology_row_for_image(image_path)
    assert row["nuclear_area_mean"] == 150.0
    assert row["nuclei_density"] == 2 / 256.0


def test_returns_cellvit_values_when_png_json_sidecar_present(tmp_path):
    image_path = tmp_path / "sample.png"
    image_path.write_bytes(b"png")
    sidecar = tmp_path / "sample.png.json"
    sidecar.write_text(
        json.dumps(
            {
                "tile_area": 256.0,
                "cells": [
                    {"contour": [[0, 0], [0, 10], [10, 10], [10, 0]]},
                    {"contour": [[0, 0], [0, 20], [20, 20], [20, 0]]},
                ],
            }
        ),
        encoding="utf-8",
    )
    row = morphology_row_for_image(image_path)
    assert row["nuclear_area_mean"] == 250.0
    assert row["nuclei_density"] == 2 / 256.0


def test_returns_nan_values_when_no_sidecar_present(tmp_path):
    image_path = tmp_path / "sample.png"
    image_path.write_bytes(b"png")
    row = morphology_row_for_image(image_path)
    assert all(np.isnan(value) for value in row.values())