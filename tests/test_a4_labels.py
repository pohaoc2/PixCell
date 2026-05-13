"""Unit tests for a4_uni_probe.labels."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.a4_uni_probe.labels import (
    ALL_ATTR_NAMES,
    APPEARANCE_ATTR_NAMES,
    CHANNEL_ATTR_NAMES,
    MORPHOLOGY_ATTR_NAMES,
    build_appearance_label_matrix,
    build_label_matrix,
    compute_channel_attributes,
    compute_morphology_attributes_from_cellvit,
)


def _write_binary_png(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), (mask.astype(np.uint8) * 255))


def _write_he_png(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def _write_float_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array.astype(np.float32))


def _make_channel_major_exp_channels(tmp_path: Path) -> Path:
    exp_root = tmp_path / "exp_channels"
    tile_id = "0_0"

    cell_masks = np.zeros((256, 256), dtype=np.uint8)
    cell_masks[:128, :] = 1
    _write_binary_png(exp_root / "cell_masks" / f"{tile_id}.png", cell_masks)

    cancer = np.zeros((256, 256), dtype=np.uint8)
    cancer[:64, :] = 1
    healthy = np.zeros((256, 256), dtype=np.uint8)
    healthy[64:128, :] = 1
    immune = np.zeros((256, 256), dtype=np.uint8)
    _write_binary_png(exp_root / "cell_type_cancer" / f"{tile_id}.png", cancer)
    _write_binary_png(exp_root / "cell_type_healthy" / f"{tile_id}.png", healthy)
    _write_binary_png(exp_root / "cell_type_immune" / f"{tile_id}.png", immune)

    prolif = np.zeros((256, 256), dtype=np.uint8)
    prolif[:32, :] = 1
    nonprolif = np.zeros((256, 256), dtype=np.uint8)
    nonprolif[32:128, :] = 1
    dead = np.zeros((256, 256), dtype=np.uint8)
    _write_binary_png(exp_root / "cell_state_prolif" / f"{tile_id}.png", prolif)
    _write_binary_png(exp_root / "cell_state_nonprolif" / f"{tile_id}.png", nonprolif)
    _write_binary_png(exp_root / "cell_state_dead" / f"{tile_id}.png", dead)

    vasculature = np.zeros((256, 256), dtype=np.float32)
    vasculature[:50, :] = 1.0
    _write_float_npy(exp_root / "vasculature" / f"{tile_id}.npy", vasculature)
    _write_float_npy(exp_root / "oxygen" / f"{tile_id}.npy", np.full((256, 256), 0.5, dtype=np.float32))
    _write_float_npy(exp_root / "glucose" / f"{tile_id}.npy", np.full((256, 256), 0.25, dtype=np.float32))
    return exp_root


def test_compute_channel_attributes_returns_expected_values(tmp_path: Path):
    exp_root = _make_channel_major_exp_channels(tmp_path)
    row = compute_channel_attributes(exp_root, "0_0")
    assert set(row.keys()) == set(CHANNEL_ATTR_NAMES)
    assert row["cancer_fraction"] == pytest.approx(0.5, abs=1e-6)
    assert row["healthy_fraction"] == pytest.approx(0.5, abs=1e-6)
    assert row["immune_fraction"] == pytest.approx(0.0, abs=1e-6)
    assert row["prolif_fraction"] == pytest.approx(0.25, abs=1e-6)
    assert row["nonprolif_fraction"] == pytest.approx(0.75, abs=1e-6)
    assert row["dead_fraction"] == pytest.approx(0.0, abs=1e-6)
    assert row["vessel_area_pct"] == pytest.approx(50.0 / 256.0, abs=1e-6)
    assert row["mean_oxygen"] == pytest.approx(0.5, abs=1e-6)
    assert row["mean_glucose"] == pytest.approx(0.25, abs=1e-6)


def test_missing_optional_channel_returns_nan(tmp_path: Path):
    exp_root = _make_channel_major_exp_channels(tmp_path)
    (exp_root / "vasculature" / "0_0.npy").unlink()
    row = compute_channel_attributes(exp_root, "0_0")
    assert np.isnan(row["vessel_area_pct"])
    assert not np.isnan(row["cancer_fraction"])


def test_compute_morphology_attributes_from_cellvit_reads_schema(tmp_path: Path):
    cellvit_path = tmp_path / "0_0.json"
    cellvit_path.write_text(
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
    row = compute_morphology_attributes_from_cellvit(cellvit_path)
    assert row["nuclear_area_mean"] == pytest.approx(150.0)
    assert row["eccentricity_mean"] == pytest.approx(0.65)
    assert row["nuclei_density"] == pytest.approx(2 / 256.0)
    assert row["intensity_mean_h"] == pytest.approx(0.5)
    assert row["intensity_mean_e"] == pytest.approx(0.25)


def test_compute_morphology_attributes_from_cellvit_reads_cells_schema(tmp_path: Path):
    cellvit_path = tmp_path / "0_0.json"
    cellvit_path.write_text(
        json.dumps(
            {
                "patch": "0_0",
                "cells": [
                    {
                        "contour": [[0, 0], [4, 0], [4, 3], [0, 3]],
                        "bbox": [[0, 0], [4, 3]],
                        "type_cellvit": 0,
                        "type_name": "background",
                        "type_prob": 0.9,
                    },
                    {
                        "contour": [[0, 0], [2, 0], [2, 2], [0, 2]],
                        "bbox": [[0, 0], [2, 2]],
                        "type_cellvit": 0,
                        "type_name": "background",
                        "type_prob": 0.8,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    row = compute_morphology_attributes_from_cellvit(cellvit_path)
    assert row["nuclear_area_mean"] == pytest.approx(8.0)
    assert row["eccentricity_mean"] == pytest.approx(0.33071891388307384)
    assert row["nuclei_density"] == pytest.approx(2 / (256.0 * 256.0))
    assert np.isnan(row["intensity_mean_h"])
    assert np.isnan(row["intensity_mean_e"])


def test_appearance_attr_names_are_correct_set():
    expected = {
        "h_mean",
        "e_mean",
        "texture_h_contrast",
        "texture_h_homogeneity",
        "texture_h_energy",
        "texture_e_contrast",
        "texture_e_homogeneity",
        "texture_e_energy",
    }
    assert set(APPEARANCE_ATTR_NAMES) == expected
    assert len(APPEARANCE_ATTR_NAMES) == 8


def test_build_appearance_label_matrix_returns_finite_values(tmp_path: Path):
    he_dir = tmp_path / "he"
    rgb = np.full((32, 32, 3), [180, 140, 170], dtype=np.uint8)
    _write_he_png(he_dir / "0_0.png", rgb)
    mat = build_appearance_label_matrix(["0_0"], he_dir)
    assert mat.shape == (1, len(APPEARANCE_ATTR_NAMES))
    assert np.all(np.isfinite(mat))


def test_build_appearance_label_matrix_missing_he_gives_nan(tmp_path: Path):
    he_dir = tmp_path / "he"
    he_dir.mkdir()
    mat = build_appearance_label_matrix(["missing_tile"], he_dir)
    assert mat.shape == (1, len(APPEARANCE_ATTR_NAMES))
    assert np.all(np.isnan(mat))


def test_build_label_matrix_combines_channel_and_morphology_rows(tmp_path: Path):
    exp_root = _make_channel_major_exp_channels(tmp_path)
    cellvit_root = tmp_path / "cellvit_real"
    cellvit_root.mkdir()
    (cellvit_root / "0_0.json").write_text(
        json.dumps(
            {
                "tile_area": 65536.0,
                "nuclei": [{"area": 100.0, "eccentricity": 0.5, "intensity_h": 0.4, "intensity_e": 0.2}],
            }
        ),
        encoding="utf-8",
    )

    labels, attr_names = build_label_matrix(["0_0"], exp_root, cellvit_root)
    assert labels.shape == (1, len(ALL_ATTR_NAMES))
    assert attr_names == list(ALL_ATTR_NAMES)
    assert set(CHANNEL_ATTR_NAMES).issubset(attr_names)
    assert set(MORPHOLOGY_ATTR_NAMES).issubset(attr_names)


def test_build_label_matrix_includes_appearance_attrs_when_he_dir_given(tmp_path: Path):
    exp_root = _make_channel_major_exp_channels(tmp_path)
    cellvit_root = tmp_path / "cellvit_real"
    cellvit_root.mkdir()
    (cellvit_root / "0_0.json").write_text(
        json.dumps(
            {
                "tile_area": 65536.0,
                "nuclei": [{"area": 100.0, "eccentricity": 0.5, "intensity_h": 0.4, "intensity_e": 0.2}],
            }
        ),
        encoding="utf-8",
    )
    he_dir = tmp_path / "he"
    rgb = np.full((32, 32, 3), [180, 140, 170], dtype=np.uint8)
    _write_he_png(he_dir / "0_0.png", rgb)

    labels, attr_names = build_label_matrix(["0_0"], exp_root, cellvit_root, he_dir=he_dir)
    assert labels.shape == (1, len(ALL_ATTR_NAMES))
    assert "h_mean" in attr_names
    assert "texture_h_contrast" in attr_names
    h_mean_idx = attr_names.index("h_mean")
    assert np.isfinite(labels[0, h_mean_idx])


def test_build_label_matrix_appearance_nan_when_no_he_dir(tmp_path: Path):
    exp_root = _make_channel_major_exp_channels(tmp_path)
    cellvit_root = tmp_path / "cellvit_real"
    cellvit_root.mkdir()
    (cellvit_root / "0_0.json").write_text(
        json.dumps(
            {
                "tile_area": 65536.0,
                "nuclei": [{"area": 100.0, "eccentricity": 0.5, "intensity_h": 0.4, "intensity_e": 0.2}],
            }
        ),
        encoding="utf-8",
    )

    labels, attr_names = build_label_matrix(["0_0"], exp_root, cellvit_root)
    assert labels.shape == (1, len(ALL_ATTR_NAMES))
    h_mean_idx = attr_names.index("h_mean")
    assert np.isnan(labels[0, h_mean_idx])
