from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest


SYNTHETIC_CACHE = {
    "version": 1,
    "generated": "2026-04-27",
    "tile_ids": ["tile_001", "tile_002"],
    "training_curves": {
        "production": {
            "full_seed_42": [
                {"step": 50, "loss": 0.20, "grad_norm": 0.05},
                {"step": 100, "loss": 0.15, "grad_norm": 0.04},
            ]
        },
        "a1_concat": {
            "seed_1": [
                {"step": 50, "loss": 0.22, "grad_norm": 0.06},
                {"step": 100, "loss": 0.18, "grad_norm": 0.05},
            ],
            "seed_2": [
                {"step": 50, "loss": 0.24, "grad_norm": 0.07},
                {"step": 100, "loss": 0.19, "grad_norm": 0.05},
            ],
        },
        "a1_per_channel": {
            "seed_1": [
                {"step": 50, "loss": 0.25, "grad_norm": "inf"},
                {"step": 100, "loss": 0.22, "grad_norm": "inf"},
            ],
        },
        "a2_bypass_full_tme": {
            "seed_1": [
                {"step": 50, "loss": 0.21, "grad_norm": "inf"},
                {"step": 100, "loss": 0.17, "grad_norm": "inf"},
            ],
        },
        "a2_off_shelf": {},
    },
    "metrics": {
        "production": {"fud": 187.86, "dice": 0.791, "dice_std": 0.012, "pq": 0.569, "pq_std": 0.021, "lpips": 0.385, "lpips_std": 0.010, "style_hed": 0.041, "style_hed_std": 0.003},
        "a1_concat": {"fud": 187.84, "dice": 0.883, "dice_std": 0.011, "pq": 0.782, "pq_std": 0.018, "lpips": 0.344, "lpips_std": 0.009, "style_hed": 0.034, "style_hed_std": 0.002},
        "a1_per_channel": {"fud": 289.07, "dice": 0.054, "dice_std": 0.015, "pq": 0.011, "pq_std": 0.006, "lpips": 0.501, "lpips_std": 0.014, "style_hed": 0.143, "style_hed_std": 0.012},
        "a2_bypass_full_tme": {"fud": 184.42, "dice": 0.897, "dice_std": 0.010, "pq": 0.786, "pq_std": 0.017, "lpips": 0.369, "lpips_std": 0.011, "style_hed": 0.043, "style_hed_std": 0.003},
        "a2_off_shelf": {"fud": 201.11, "dice": 0.771, "dice_std": 0.014, "pq": 0.551, "pq_std": 0.022, "lpips": 0.412, "lpips_std": 0.013, "style_hed": 0.051, "style_hed_std": 0.004},
    },
    "params": {"production": 50_000_000, "a1_concat": 48_000_000, "a1_per_channel": 70_000_000},
    "sensitivity": {
        "production": {
            "mean": 0.11,
            "std": 0.04,
            "per_group": {"cell_state": {"mean": 0.18}, "microenv": {"mean": 0.14}, "cell_types": {"mean": 0.01}},
        },
        "a1_concat": {
            "mean": 0.09,
            "std": 0.03,
            "per_group": {"cell_state": {"mean": 0.14}, "microenv": {"mean": 0.16}, "cell_types": {"mean": 0.02}},
        },
        "a1_per_channel": {
            "mean": 0.16,
            "std": 0.05,
            "per_group": {"cell_state": {"mean": 0.22}, "microenv": {"mean": 0.02}, "cell_types": {"mean": 0.24}},
        },
        "a2_bypass_full_tme": {
            "mean": 0.00,
            "std": 0.00,
            "per_group": {"cell_state": {"mean": 0.00}, "microenv": {"mean": 0.00}, "cell_types": {"mean": 0.00}},
        },
        "a2_off_shelf": {
            "mean": 0.00,
            "std": 0.00,
            "per_group": {"cell_state": {"mean": 0.00}, "microenv": {"mean": 0.00}, "cell_types": {"mean": 0.00}},
        },
    },
}


def test_aggregate_curves_mean_std():
    from src.paper_figures.fig_si_a1_a2_unified import _aggregate_curves

    steps, mean, std = _aggregate_curves(SYNTHETIC_CACHE["training_curves"]["a1_concat"], "loss")
    assert list(steps) == [50, 100]
    assert mean[0] == pytest.approx(0.23)
    assert std[0] == pytest.approx(0.01, abs=0.005)


def test_aggregate_curves_single_run():
    from src.paper_figures.fig_si_a1_a2_unified import _aggregate_curves

    steps, _mean, std = _aggregate_curves(SYNTHETIC_CACHE["training_curves"]["production"], "loss")
    assert len(steps) == 2
    assert std[0] == pytest.approx(0.0)


def test_aggregate_gradnorm_inf():
    from src.paper_figures.fig_si_a1_a2_unified import _aggregate_curves

    _steps, mean, _std = _aggregate_curves(SYNTHETIC_CACHE["training_curves"]["a1_per_channel"], "grad_norm")
    assert math.isinf(mean[0])


def test_build_figure_no_error(tmp_path: Path):
    import matplotlib

    matplotlib.use("Agg")
    from PIL import Image
    from src.paper_figures.fig_si_a1_a2_unified import build_figure

    cache_path = tmp_path / "cache.json"
    cache_path.write_text(json.dumps(SYNTHETIC_CACHE), encoding="utf-8")
    tile_dir = tmp_path / "tiles"
    for variant in ("production", "a1_concat", "a1_per_channel", "a2_bypass_full_tme", "a2_off_shelf", "gt"):
        variant_dir = tile_dir / variant
        variant_dir.mkdir(parents=True)
        for tile_id in ("tile_001", "tile_002"):
            Image.fromarray(np.full((256, 256, 3), 200, dtype=np.uint8)).save(variant_dir / f"{tile_id}.png")

    fig = build_figure(cache_path=cache_path, tile_dir=tile_dir)
    assert fig is not None
    out = tmp_path / "out.png"
    fig.savefig(out, dpi=72)
    assert out.exists()


def test_build_section2_figure_with_delta_lpips(tmp_path: Path):
    import matplotlib

    matplotlib.use("Agg")
    from src.paper_figures.fig_si_a1_a2_unified import build_section2_figure

    cache_path = tmp_path / "cache.json"
    cache_path.write_text(json.dumps(SYNTHETIC_CACHE), encoding="utf-8")

    fig = build_section2_figure(cache_path=cache_path)

    assert fig is not None


def test_section1_production_and_concat_are_dashed(tmp_path: Path):
    import matplotlib

    matplotlib.use("Agg")
    from src.paper_figures.fig_si_a1_a2_unified import build_section1_figure

    cache_path = tmp_path / "cache.json"
    cache_path.write_text(json.dumps(SYNTHETIC_CACHE), encoding="utf-8")

    fig = build_section1_figure(cache_path=cache_path)

    line_styles: dict[str, set[str]] = {}
    for ax in fig.axes:
        for line in ax.lines:
            line_styles.setdefault(line.get_label(), set()).add(line.get_linestyle())

    assert "--" in line_styles["Per-ch. + Attn"]
    assert "--" in line_styles["Concat TME"]


def test_section1_excludes_off_shelf_from_legend_and_axes(tmp_path: Path):
    import matplotlib

    matplotlib.use("Agg")
    from src.paper_figures.fig_si_a1_a2_unified import build_section1_figure

    cache_path = tmp_path / "cache.json"
    cache_path.write_text(json.dumps(SYNTHETIC_CACHE), encoding="utf-8")

    fig = build_section1_figure(cache_path=cache_path)

    labels = []
    for ax in fig.axes:
        labels.extend(line.get_label() for line in ax.lines)

    assert "Off-the-shelf PixCell" not in labels


def test_read_log_carries_preceding_proj_grad(tmp_path: Path):
    from tools.ablation_a3.aggregate_stability import _read_log

    log_path = tmp_path / "train_log.log"
    log_path.write_text(
        "proj_grad[cell_identity]=1.080e-04\n"
        "proj_grad[cell_state]=1.195e-04\n"
        "proj_grad[microenv]=1.649e-04\n"
        "Epoch [1/20] Step [50/2600] Loss: 0.1074\n",
        encoding="utf-8",
    )

    entries = _read_log(log_path)

    assert len(entries) == 1
    assert entries[0]["step"] == 50
    assert entries[0]["loss"] == pytest.approx(0.1074)
    assert entries[0]["grad_norm"] == pytest.approx(1.649e-04)


def test_build_section4_figure_handles_missing_data(tmp_path: Path):
    import matplotlib

    matplotlib.use("Agg")
    from src.paper_figures.fig_si_a1_a2_unified import build_section4_figure

    cache = dict(SYNTHETIC_CACHE)
    cache.pop("sensitivity", None)
    cache_path = tmp_path / "cache.json"
    cache_path.write_text(json.dumps(cache), encoding="utf-8")

    fig = build_section4_figure(cache_path=cache_path)

    assert fig is not None


def test_section4_filters_to_cell_state_and_microenv():
    from src.paper_figures.fig_si_a1_a2_unified import _section4_sensitivity

    filtered = _section4_sensitivity(SYNTHETIC_CACHE)

    assert filtered["production"]["mean"] == pytest.approx(0.16)
    assert filtered["a1_concat"]["mean"] == pytest.approx(0.15)
    assert filtered["a1_per_channel"]["mean"] == pytest.approx(0.12)
    assert filtered["a2_bypass_full_tme"]["mean"] == pytest.approx(0.0)
    assert filtered["a2_off_shelf"]["mean"] == pytest.approx(0.0)


def test_section4_legacy_bypass_fallback():
    from src.paper_figures.fig_si_a1_a2_unified import _section4_sensitivity

    cache = json.loads(json.dumps(SYNTHETIC_CACHE))
    cache["sensitivity"]["a2_bypass"] = cache["sensitivity"].pop("a2_bypass_full_tme")

    filtered = _section4_sensitivity(cache)

    assert filtered["a2_bypass_full_tme"]["mean"] == pytest.approx(0.0)
