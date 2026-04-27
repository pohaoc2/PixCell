from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import json
import numpy as np
from PIL import Image
import pytest

from tools.ablation_a1_a2.sensitivity_eval import (
    compute_sensitivity_scores,
    generate_group_ablations,
    render_sensitivity,
    run_sensitivity,
    score_sensitivity,
    summarize_variant_sensitivity,
    variant_requires_generation,
)


def _rand_rgb(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)


def test_compute_sensitivity_scores_keys():
    pytest.importorskip("torch")
    pytest.importorskip("lpips")

    baseline = _rand_rgb(0)
    group_images = {
        "cell_types": _rand_rgb(1),
        "cell_state": _rand_rgb(2),
        "vasculature": _rand_rgb(3),
        "microenv": _rand_rgb(4),
    }

    scores = compute_sensitivity_scores(baseline, group_images)

    assert set(scores.keys()) == set(group_images.keys())
    for value in scores.values():
        assert isinstance(value, float)
        assert value >= 0.0


def test_compute_sensitivity_scores_identical_is_zero():
    pytest.importorskip("torch")
    pytest.importorskip("lpips")

    image = _rand_rgb(0)

    scores = compute_sensitivity_scores(image, {"cell_types": image.copy()})

    assert scores["cell_types"] < 0.01


def test_generate_group_ablations_returns_one_image_per_group():
    class _Cloneable:
        def clone(self):
            return self

    dummy_rgb = np.zeros((256, 256, 3), dtype=np.uint8)
    mock_context = {
        "tme_module": MagicMock(group_names=["cell_types", "cell_state"]),
        "dtype": "float32",
    }

    with (
        patch("tools.ablation_a1_a2.sensitivity_eval.prepare_tile_context", return_value=mock_context),
        patch("tools.ablation_a1_a2.sensitivity_eval._fuse_active_groups", return_value=object()),
        patch("tools.ablation_a1_a2.sensitivity_eval._render_fused_ablation_image", return_value=dummy_rgb),
        patch("tools.ablation_a1_a2.sensitivity_eval._make_fixed_noise", return_value=_Cloneable()),
    ):
        result = generate_group_ablations(
            tile_id="tile_0",
            models={},
            config=MagicMock(),
            scheduler=MagicMock(),
            uni_embeds=object(),
            device="cpu",
            exp_channels_dir=Path("/tmp"),
            guidance_scale=1.5,
            seed=42,
        )

    assert set(result.keys()) == {"cell_types", "cell_state"}
    for array in result.values():
        assert array.shape == (256, 256, 3)


def test_summarize_variant_sensitivity_averages_group_means():
    summary = summarize_variant_sensitivity(
        {
            "cell_types": [0.10, 0.30],
            "cell_state": [0.20, 0.40],
        }
    )

    assert summary["mean"] == pytest.approx(0.25)
    assert summary["std"] == pytest.approx(0.05)
    assert summary["per_group"]["cell_types"]["mean"] == pytest.approx(0.20)


def test_variant_requires_generation_flags_trivial_variants():
    assert variant_requires_generation("production") is True
    assert variant_requires_generation("a2_bypass_full_tme") is True
    assert variant_requires_generation("a2_off_shelf") is False
    assert variant_requires_generation("a2_bypass") is False


def test_render_sensitivity_writes_group_images(tmp_path):
    cache_dir = tmp_path / "si_a1_a2"
    cache_dir.mkdir()
    tile_dir = cache_dir / "tiles" / "production"
    tile_dir.mkdir(parents=True)

    fake_rgb = np.zeros((256, 256, 3), dtype=np.uint8)
    Image.fromarray(fake_rgb).save(tile_dir / "tile_0.png")

    with (
        patch("tools.ablation_a1_a2.sensitivity_eval.load_all_models", return_value={}),
        patch("tools.ablation_a1_a2.sensitivity_eval.make_inference_scheduler", return_value=MagicMock()),
        patch("tools.ablation_a1_a2.sensitivity_eval.read_config", return_value=MagicMock()),
        patch("tools.ablation_a1_a2.sensitivity_eval.resolve_uni_embedding", return_value=object()),
        patch("tools.ablation_a1_a2.sensitivity_eval._release_accelerator_memory"),
        patch(
            "tools.ablation_a1_a2.sensitivity_eval.generate_group_ablations",
            return_value={"cell_types": fake_rgb, "cell_state": fake_rgb},
        ),
    ):
        render_root = render_sensitivity(
            cache_dir=cache_dir,
            tile_ids=["tile_0"],
            device="cpu",
            exp_channels_dir=Path("/tmp"),
            features_dir=Path("/tmp"),
            variants=["production"],
        )

    assert (render_root / "production" / "cell_types" / "tile_0.png").is_file()
    assert (render_root / "production" / "cell_state" / "tile_0.png").is_file()
    assert (render_root / "manifest.json").is_file()


def test_score_sensitivity_reads_rendered_images_and_writes_cache(tmp_path):
    cache_dir = tmp_path / "si_a1_a2"
    cache_dir.mkdir()
    tile_dir = cache_dir / "tiles" / "production"
    tile_dir.mkdir(parents=True)
    render_root = cache_dir / "sensitivity_tiles"

    fake_rgb = np.zeros((256, 256, 3), dtype=np.uint8)
    Image.fromarray(fake_rgb).save(tile_dir / "tile_0.png")
    for group_name in ("cell_types", "cell_state"):
        group_dir = render_root / "production" / group_name
        group_dir.mkdir(parents=True)
        Image.fromarray(fake_rgb).save(group_dir / "tile_0.png")

    with patch(
        "tools.ablation_a1_a2.sensitivity_eval.compute_sensitivity_scores",
        return_value={"cell_types": 0.0, "cell_state": 0.2},
    ):
        score_sensitivity(
            cache_dir=cache_dir,
            tile_ids=["tile_0"],
            device="cpu",
            variants=["production"],
            render_root=render_root,
        )

    cache = json.loads((cache_dir / "cache.json").read_text(encoding="utf-8"))
    assert cache["sensitivity"]["production"]["mean"] == pytest.approx(0.1)
    assert cache["sensitivity"]["production"]["per_group"]["cell_state"]["mean"] == pytest.approx(0.2)


def test_run_sensitivity_writes_cache(tmp_path):
    cache_dir = tmp_path / "si_a1_a2"
    cache_dir.mkdir()
    tile_dir = cache_dir / "tiles" / "production"
    tile_dir.mkdir(parents=True)

    fake_rgb = np.zeros((256, 256, 3), dtype=np.uint8)
    Image.fromarray(fake_rgb).save(tile_dir / "tile_0.png")

    with (
        patch("tools.ablation_a1_a2.sensitivity_eval.load_all_models", return_value={}),
        patch("tools.ablation_a1_a2.sensitivity_eval.make_inference_scheduler", return_value=MagicMock()),
        patch("tools.ablation_a1_a2.sensitivity_eval.read_config", return_value=MagicMock()),
        patch("tools.ablation_a1_a2.sensitivity_eval.resolve_uni_embedding", return_value=object()),
        patch("tools.ablation_a1_a2.sensitivity_eval._release_accelerator_memory"),
        patch(
            "tools.ablation_a1_a2.sensitivity_eval.generate_group_ablations",
            return_value={"cell_types": fake_rgb, "cell_state": fake_rgb},
        ),
        patch(
            "tools.ablation_a1_a2.sensitivity_eval.compute_sensitivity_scores",
            return_value={"cell_types": 0.0, "cell_state": 0.2},
        ),
    ):
        run_sensitivity(
            cache_dir=cache_dir,
            tile_ids=["tile_0"],
            device="cpu",
            exp_channels_dir=Path("/tmp"),
            features_dir=Path("/tmp"),
            variants=["production"],
        )

    cache = json.loads((cache_dir / "cache.json").read_text(encoding="utf-8"))
    assert "sensitivity" in cache
    assert "production" in cache["sensitivity"]
    assert cache["sensitivity"]["production"]["mean"] == pytest.approx(0.1)
    assert cache["sensitivity"]["production"]["per_group"]["cell_state"]["mean"] == pytest.approx(0.2)