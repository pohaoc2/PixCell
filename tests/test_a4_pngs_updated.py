"""Tests for publication-ready a4 UNI probe figure outputs."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


SWEEP_ATTRS = [
    "eccentricity_mean",
    "nuclear_area_mean",
    "nuclei_density",
    "texture_e_contrast",
    "texture_h_contrast",
    "texture_h_energy",
]


def _write_probe_results(out_dir: Path) -> None:
    rows = []
    for index, attr in enumerate(SWEEP_ATTRS):
        rows.append({
            "attr": attr,
            "uni_r2_mean": f"{0.8 - index * 0.03:.4f}",
            "uni_r2_std": f"{0.01 + index * 0.002:.4f}",
            "uni_n_valid_folds": "5",
            "tme_r2_mean": f"{0.2 + index * 0.02:.4f}",
            "tme_r2_std": f"{0.02 + index * 0.001:.4f}",
            "tme_n_valid_folds": "5",
            "delta_r2_uni_minus_tme": f"{0.6 - index * 0.05:.4f}",
        })

    with (out_dir / "probe_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_specificity(out_dir: Path) -> None:
    metrics = [
        ("morpho", "morpho.eccentricity_mean"),
        ("morpho", "morpho.nuclear_area_mean"),
        ("morpho", "morpho.nuclei_density"),
        ("appearance", "appearance.texture_e_contrast"),
        ("appearance", "appearance.texture_h_contrast"),
        ("appearance", "appearance.texture_h_energy"),
    ]
    rows = []
    for attr_index, attr in enumerate(SWEEP_ATTRS):
        for metric_index, (family, metric) in enumerate(metrics):
            targeted = 0.3 + 0.05 * attr_index - 0.02 * metric_index
            random = 0.05 + 0.01 * metric_index
            rows.append({
                "edited_attr": attr,
                "measured_metric": metric,
                "family": family,
                "targeted_slope": f"{targeted:.4f}",
                "random_slope": f"{random:.4f}",
                "abs_ratio": f"{abs(targeted) / abs(random):.4f}",
                "baseline_std": f"{0.1 + 0.02 * metric_index:.4f}",
                "normalized_targeted_slope": f"{targeted / (0.1 + 0.02 * metric_index):.4f}",
            })

    with (out_dir / "specificity_full.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_rgb_png(path: Path, value: float) -> None:
    rgb = np.full((18, 18, 3), value, dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(path, rgb)


def _write_mask_png(path: Path, value: float) -> None:
    mask = np.full((18, 18), value, dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(path, mask, cmap="gray", vmin=0.0, vmax=1.0)


def _write_sweep_tree(out_dir: Path) -> None:
    for attr_index, attr in enumerate(SWEEP_ATTRS):
        attr_dir = out_dir / "sweep" / attr
        invalid_tile = attr_dir / "tile_00_invalid" / "targeted"
        invalid_tile.mkdir(parents=True, exist_ok=True)
        _write_rgb_png(invalid_tile / "alpha_-1.00.png", 0.1)
        for tile_index in range(1, 7):
            tile_name = f"tile_{tile_index:02d}"
            for direction_index, direction in enumerate(("targeted", "random")):
                for alpha_index, alpha in enumerate(("-1.00", "+0.00", "+1.00")):
                    value = 0.1 * (attr_index + 1) + 0.03 * direction_index + 0.01 * alpha_index
                    _write_rgb_png(attr_dir / tile_name / direction / f"alpha_{alpha}.png", value)


def _write_reference_inputs(out_dir: Path) -> None:
    for tile_index in range(1, 7):
        tile_id = f"tile_{tile_index:02d}"
        _write_rgb_png(out_dir / "he" / f"{tile_id}.png", 0.1 * tile_index)
        _write_mask_png(out_dir / "exp_channels" / "cell_masks" / f"{tile_id}.png", 0.1 * tile_index)


def _build_synthetic_inputs(out_dir: Path) -> None:
    _write_probe_results(out_dir)
    _write_specificity(out_dir)
    _write_sweep_tree(out_dir)
    _write_reference_inputs(out_dir)


def test_specificity_square_payload_uses_only_edited_attrs(tmp_path: Path) -> None:
    _write_specificity(tmp_path)

    from src.a4_uni_probe.figures import _read_csv_rows, _specificity_square_payload

    edited, grid, annot = _specificity_square_payload(_read_csv_rows(tmp_path / "specificity_full.csv"))

    assert edited == SWEEP_ATTRS
    assert grid.shape == (len(SWEEP_ATTRS), len(SWEEP_ATTRS))
    assert annot.shape == (len(SWEEP_ATTRS), len(SWEEP_ATTRS))


def test_render_pngs_updated_outputs(tmp_path: Path) -> None:
    _build_synthetic_inputs(tmp_path)

    from src.a4_uni_probe.figures import render_pngs_updated

    dest_dir = tmp_path / "pngs_updated" / "a4_uni_probe"
    outputs = render_pngs_updated(tmp_path, dest_dir)

    assert len(outputs) == 8
    assert set(outputs) == {
        "probe_delta_r2",
        "specificity_heatmap",
        *(f"sweep_grid_{attr}" for attr in SWEEP_ATTRS),
    }
    for path in outputs.values():
        assert path.is_file()
        assert path.stat().st_size > 1000

    probe_image = mpimg.imread(dest_dir / "probe_delta_r2.png")
    assert probe_image.shape[1] >= 1200
    assert probe_image.shape[0] >= 700

    sweep_image = mpimg.imread(dest_dir / f"sweep_grid_{SWEEP_ATTRS[0]}.png")
    assert sweep_image.shape[1] >= 3200
    assert sweep_image.shape[0] >= 500

    heatmap_image = mpimg.imread(dest_dir / "specificity_heatmap.png")
    assert heatmap_image.shape[1] >= 1100
    assert heatmap_image.shape[0] >= 1100


def test_main_pngs_updated_command(tmp_path: Path) -> None:
    _build_synthetic_inputs(tmp_path)

    from src.a4_uni_probe.main import main

    dest_dir = tmp_path / "exports"
    exit_code = main(["pngs_updated", "--out-dir", str(tmp_path), "--dest-dir", str(dest_dir)])

    assert exit_code == 0
    assert (dest_dir / "probe_delta_r2.png").is_file()
    assert (dest_dir / "specificity_heatmap.png").is_file()