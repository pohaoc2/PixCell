"""Unit tests for a4_uni_probe.edit."""

from __future__ import annotations

import csv
import json
from argparse import Namespace
from pathlib import Path

import numpy as np

from src.a4_uni_probe.edit import _select_sweep_attrs, _summarize_slopes, run_sweep, null_uni, random_unit_direction, sweep_uni


def test_sweep_uni_is_linear_and_preserves_zero_alpha():
    uni = np.array([3.0, 4.0], dtype=np.float32)
    direction = np.array([1.0, 0.0], dtype=np.float32)
    edits = sweep_uni(uni, direction, [-1.0, 0.0, 1.0])
    np.testing.assert_allclose(edits[1], uni)
    np.testing.assert_allclose(edits[2] - edits[1], edits[1] - edits[0])


def test_null_uni_removes_projection():
    uni = np.array([2.0, 1.0], dtype=np.float32)
    direction = np.array([1.0, 0.0], dtype=np.float32)
    nulled = null_uni(uni, direction)
    assert abs(float(np.dot(nulled, direction))) < 1e-6


def test_random_unit_direction_is_seeded_and_normalized():
    left = random_unit_direction(5, seed=123)
    right = random_unit_direction(5, seed=123)
    np.testing.assert_allclose(left, right)
    assert np.isclose(np.linalg.norm(left), 1.0)


def test_summarize_slopes_detects_monotonic(tmp_path):
    csv_path = tmp_path / "metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["alpha", "target_value", "direction"])
        writer.writeheader()
        for direction, slope in (("targeted", 1.0), ("random", 0.0)):
            for alpha in (-2, -1, 0, 1, 2):
                writer.writerow({"alpha": alpha, "target_value": slope * alpha + 0.01 * alpha, "direction": direction})
    out_path = tmp_path / "slope_summary.json"
    _summarize_slopes(csv_path, out_path, "test_attr")
    summary = json.loads(out_path.read_text(encoding="utf-8"))
    assert summary["targeted"]["slope_mean"] > 0.5
    assert summary["pass_criterion_met"] is True


def test_select_sweep_attrs_morphology_pool(tmp_path: Path):
    csv_path = tmp_path / "probe_results.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["attr", "delta_r2_uni_minus_tme"])
        writer.writeheader()
        writer.writerow({"attr": "eccentricity_mean", "delta_r2_uni_minus_tme": "0.23"})
        writer.writerow({"attr": "h_mean", "delta_r2_uni_minus_tme": "0.50"})
        writer.writerow({"attr": "nuclear_area_mean", "delta_r2_uni_minus_tme": "0.10"})

    attrs = _select_sweep_attrs(csv_path, top_k=2, attr_pool="morphology")
    assert "h_mean" not in attrs
    assert "eccentricity_mean" in attrs


def test_select_sweep_attrs_appearance_pool(tmp_path: Path):
    csv_path = tmp_path / "probe_results.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["attr", "delta_r2_uni_minus_tme"])
        writer.writeheader()
        writer.writerow({"attr": "eccentricity_mean", "delta_r2_uni_minus_tme": "0.23"})
        writer.writerow({"attr": "h_mean", "delta_r2_uni_minus_tme": "0.50"})
        writer.writerow({"attr": "e_mean", "delta_r2_uni_minus_tme": "0.30"})

    attrs = _select_sweep_attrs(csv_path, top_k=2, attr_pool="appearance")
    assert "eccentricity_mean" not in attrs
    assert "h_mean" in attrs
    assert "e_mean" in attrs


def test_run_sweep_uses_fixed_tile_ids_without_attr_tile_selection(tmp_path: Path, monkeypatch):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    with (out_dir / "probe_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["attr", "delta_r2_uni_minus_tme"])
        writer.writeheader()
        writer.writerow({"attr": "eccentricity_mean", "delta_r2_uni_minus_tme": "0.23"})

    np.savez_compressed(
        out_dir / "labels.npz",
        tile_ids=np.asarray(["tile_b", "tile_a"], dtype=str),
        attr_names=np.asarray(["eccentricity_mean"], dtype=str),
        labels=np.asarray([[0.1], [0.2]], dtype=np.float32),
    )
    np.savez_compressed(
        out_dir / "features.npz",
        tile_ids=np.asarray(["tile_b", "tile_a"], dtype=str),
        uni=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        tme=np.asarray([[0.0], [0.0]], dtype=np.float32),
    )
    direction_dir = out_dir / "probe_directions"
    direction_dir.mkdir()
    np.save(direction_dir / "eccentricity_mean_uni_direction.npy", np.asarray([1.0, 0.0], dtype=np.float32))

    fixed_tiles_path = tmp_path / "shared_tiles.json"
    fixed_tiles_path.write_text(json.dumps({"tile_ids": ["tile_a", "tile_b"]}), encoding="utf-8")

    def _fail_select(*_args, **_kwargs):
        raise AssertionError("_select_sweep_tiles should not be called when fixed tiles are provided")

    monkeypatch.setattr("src.a4_uni_probe.edit._select_sweep_tiles", _fail_select)
    monkeypatch.setattr("src.a4_uni_probe.edit.load_inference_bundle", lambda **_kwargs: object())
    monkeypatch.setattr("src.a4_uni_probe.edit.generate_with_uni_override", lambda spec, **_kwargs: spec.out_path.parent.mkdir(parents=True, exist_ok=True))
    monkeypatch.setattr("src.a4_uni_probe.edit.morphology_row_for_image", lambda _path: {"eccentricity_mean": 0.5})
    monkeypatch.setattr(
        "src.a4_uni_probe.edit.appearance_row_for_image",
        lambda _path: {
            "appearance.h_mean": 0.1,
            "appearance.h_std": 0.2,
            "appearance.e_mean": 0.3,
            "appearance.e_std": 0.4,
            "appearance.stain_vector_angle_deg": 5.0,
            "appearance.texture_h_contrast": 6.0,
            "appearance.texture_h_homogeneity": 0.7,
            "appearance.texture_h_energy": 0.8,
            "appearance.texture_e_contrast": 0.9,
            "appearance.texture_e_homogeneity": 1.0,
            "appearance.texture_e_energy": 1.1,
        },
    )

    args = Namespace(
        out_dir=out_dir,
        checkpoint_dir=tmp_path / "ckpt",
        config_path=tmp_path / "config.py",
        data_root=tmp_path / "data",
        exp_channels_dir=tmp_path / "exp_channels",
        num_steps=1,
        guidance_scale=1.0,
        seed=7,
        top_k_attrs=1,
        k_tiles=2,
        alphas=[0.0],
        tile_shard_index=0,
        tile_shard_count=1,
        fixed_tile_ids=fixed_tiles_path,
        attr_pool="morphology",
    )

    run_sweep(args)

    metrics_path = out_dir / "sweep" / "eccentricity_mean" / "metrics.csv"
    rows = list(csv.DictReader(metrics_path.open(encoding="utf-8")))
    assert [row["tile_id"] for row in rows] == ["tile_a", "tile_a", "tile_b", "tile_b"]


def test_run_sweep_appearance_attr_uses_appearance_metrics(tmp_path: Path, monkeypatch):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    with (out_dir / "probe_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["attr", "delta_r2_uni_minus_tme"])
        writer.writeheader()
        writer.writerow({"attr": "texture_h_contrast", "delta_r2_uni_minus_tme": "0.50"})

    np.savez_compressed(
        out_dir / "labels.npz",
        tile_ids=np.asarray(["tile_a"], dtype=str),
        attr_names=np.asarray(["texture_h_contrast"], dtype=str),
        labels=np.asarray([[0.1]], dtype=np.float32),
    )
    np.savez_compressed(
        out_dir / "features.npz",
        tile_ids=np.asarray(["tile_a"], dtype=str),
        uni=np.asarray([[1.0, 0.0]], dtype=np.float32),
        tme=np.asarray([[0.0]], dtype=np.float32),
    )
    direction_dir = out_dir / "probe_directions"
    direction_dir.mkdir()
    np.save(direction_dir / "texture_h_contrast_uni_direction.npy", np.asarray([1.0, 0.0], dtype=np.float32))

    fixed_tiles_path = tmp_path / "shared_tiles.json"
    fixed_tiles_path.write_text(json.dumps({"tile_ids": ["tile_a"]}), encoding="utf-8")

    monkeypatch.setattr("src.a4_uni_probe.edit.load_inference_bundle", lambda **_kwargs: object())
    monkeypatch.setattr("src.a4_uni_probe.edit.generate_with_uni_override", lambda spec, **_kwargs: spec.out_path.parent.mkdir(parents=True, exist_ok=True))
    monkeypatch.setattr(
        "src.a4_uni_probe.edit.morphology_row_for_image",
        lambda _path: {"eccentricity_mean": 0.5, "nuclear_area_mean": 1.0, "nuclei_density": 0.1, "intensity_mean_h": 0.2, "intensity_mean_e": 0.3},
    )
    monkeypatch.setattr(
        "src.a4_uni_probe.edit.appearance_row_for_image",
        lambda _path: {
            "appearance.h_mean": 0.1,
            "appearance.h_std": 0.2,
            "appearance.e_mean": 0.3,
            "appearance.e_std": 0.4,
            "appearance.stain_vector_angle_deg": 5.0,
            "appearance.texture_h_contrast": 6.0,
            "appearance.texture_h_homogeneity": 0.7,
            "appearance.texture_h_energy": 0.8,
            "appearance.texture_e_contrast": 0.9,
            "appearance.texture_e_homogeneity": 1.0,
            "appearance.texture_e_energy": 1.1,
        },
    )

    args = Namespace(
        out_dir=out_dir,
        checkpoint_dir=tmp_path / "ckpt",
        config_path=tmp_path / "config.py",
        data_root=tmp_path / "data",
        exp_channels_dir=tmp_path / "exp_channels",
        num_steps=1,
        guidance_scale=1.0,
        seed=7,
        top_k_attrs=1,
        k_tiles=1,
        alphas=[0.0],
        tile_shard_index=0,
        tile_shard_count=1,
        fixed_tile_ids=fixed_tiles_path,
        attr_pool="appearance",
    )

    run_sweep(args)

    metrics_path = out_dir / "sweep" / "texture_h_contrast" / "metrics.csv"
    rows = list(csv.DictReader(metrics_path.open(encoding="utf-8")))
    assert [float(row["target_value"]) for row in rows] == [6.0, 6.0]


def test_run_null_appearance_attr_uses_appearance_metrics(tmp_path: Path, monkeypatch):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    with (out_dir / "probe_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["attr", "delta_r2_uni_minus_tme"])
        writer.writeheader()
        writer.writerow({"attr": "texture_h_contrast", "delta_r2_uni_minus_tme": "0.50"})

    np.savez_compressed(
        out_dir / "labels.npz",
        tile_ids=np.asarray(["tile_a"], dtype=str),
        attr_names=np.asarray(["texture_h_contrast"], dtype=str),
        labels=np.asarray([[0.1]], dtype=np.float32),
    )
    np.savez_compressed(
        out_dir / "features.npz",
        tile_ids=np.asarray(["tile_a"], dtype=str),
        uni=np.asarray([[1.0, 0.0]], dtype=np.float32),
        tme=np.asarray([[0.0]], dtype=np.float32),
    )
    direction_dir = out_dir / "probe_directions"
    direction_dir.mkdir()
    np.save(direction_dir / "texture_h_contrast_uni_direction.npy", np.asarray([1.0, 0.0], dtype=np.float32))

    fixed_tiles_path = tmp_path / "shared_tiles.json"
    fixed_tiles_path.write_text(json.dumps({"tile_ids": ["tile_a"]}), encoding="utf-8")
    full_null_root = tmp_path / "full_null"
    (full_null_root / "tile_a").mkdir(parents=True)
    (full_null_root / "tile_a" / "tme_only.png").write_bytes(b"png")

    monkeypatch.setattr("src.a4_uni_probe.edit.load_inference_bundle", lambda **_kwargs: object())
    monkeypatch.setattr("src.a4_uni_probe.edit.generate_with_uni_override", lambda spec, **_kwargs: spec.out_path.parent.mkdir(parents=True, exist_ok=True))
    monkeypatch.setattr(
        "src.a4_uni_probe.edit.morphology_row_for_image",
        lambda _path: {"eccentricity_mean": 0.5, "nuclear_area_mean": 1.0, "nuclei_density": 0.1, "intensity_mean_h": 0.2, "intensity_mean_e": 0.3},
    )
    monkeypatch.setattr(
        "src.a4_uni_probe.edit.appearance_row_for_image",
        lambda _path: {
            "appearance.h_mean": 0.1,
            "appearance.h_std": 0.2,
            "appearance.e_mean": 0.3,
            "appearance.e_std": 0.4,
            "appearance.stain_vector_angle_deg": 5.0,
            "appearance.texture_h_contrast": 6.0,
            "appearance.texture_h_homogeneity": 0.7,
            "appearance.texture_h_energy": 0.8,
            "appearance.texture_e_contrast": 0.9,
            "appearance.texture_e_homogeneity": 1.0,
            "appearance.texture_e_energy": 1.1,
        },
    )

    from src.a4_uni_probe.edit import run_null

    args = Namespace(
        out_dir=out_dir,
        checkpoint_dir=tmp_path / "ckpt",
        config_path=tmp_path / "config.py",
        data_root=tmp_path / "data",
        exp_channels_dir=tmp_path / "exp_channels",
        num_steps=1,
        guidance_scale=1.0,
        seed=7,
        top_k_attrs=1,
        k_tiles=1,
        tile_shard_index=0,
        tile_shard_count=1,
        fixed_tile_ids=fixed_tiles_path,
        attr_pool="appearance",
        full_null_root=full_null_root,
    )

    run_null(args)

    metrics_path = out_dir / "null" / "texture_h_contrast" / "metrics.csv"
    rows = list(csv.DictReader(metrics_path.open(encoding="utf-8")))
    assert [float(row["target_value"]) for row in rows] == [6.0, 6.0, 6.0]