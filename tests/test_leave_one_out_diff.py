"""Tests for leave-one-out pixel diff core logic."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib.colors as mcolors
import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _make_cache(
    tmp_path: Path,
    group_names=("cell_types", "cell_state", "vasculature", "microenv"),
):
    """Write a minimal manifest plus synthetic images to tmp_path."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    all_img = np.full((4, 4, 3), 200, dtype=np.uint8)
    Image.fromarray(all_img).save(tmp_path / "generated_he.png")

    (tmp_path / "all").mkdir()
    Image.fromarray(all_img).save(tmp_path / "all" / "generated_he.png")

    (tmp_path / "triples").mkdir()
    entries_triples = []
    for i, omit in enumerate(group_names):
        active = [g for g in group_names if g != omit]
        val = 100 + i * 20
        img = np.full((4, 4, 3), val, dtype=np.uint8)
        fname = f"{i + 1:02d}_{'__'.join(active)}.png"
        Image.fromarray(img).save(tmp_path / "triples" / fname)
        entries_triples.append(
            {
                "active_groups": active,
                "condition_label": f"triples_{i}",
                "image_label": f"lbl_{i}",
                "image_path": f"triples/{fname}",
            }
        )

    manifest = {
        "version": 1,
        "tile_id": "test_tile",
        "group_names": list(group_names),
        "sections": [
            {"title": "3 active groups", "subset_size": 3, "entries": entries_triples},
            {
                "title": "4 active groups",
                "subset_size": 4,
                "entries": [
                    {
                        "active_groups": list(group_names),
                        "condition_label": "all",
                        "image_label": "all",
                        "image_path": "all/generated_he.png",
                    }
                ],
            },
        ],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return tmp_path


def _write_cell_mask(path: Path) -> None:
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[1:3, 1:3] = 255
    Image.fromarray(mask).save(path)


def test_find_loo_entry_returns_correct_triple(tmp_path):
    from tools.vis.leave_one_out_diff import find_loo_entry

    cache = _make_cache(tmp_path)
    manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
    sections = manifest["sections"]

    entry = find_loo_entry(sections, "cell_types")
    assert "cell_types" not in entry["active_groups"]
    assert len(entry["active_groups"]) == 3


def test_compute_loo_diffs_shape_and_nonzero(tmp_path):
    from tools.vis.leave_one_out_diff import compute_loo_diffs

    cache = _make_cache(tmp_path)
    diffs = compute_loo_diffs(cache)

    assert set(diffs) == {"cell_types", "cell_state", "vasculature", "microenv"}
    for group, diff in diffs.items():
        assert diff.shape == (4, 4), f"bad shape for {group}: {diff.shape}"
        assert diff.dtype == np.float32
        assert diff.min() >= 0.0


def test_compute_loo_diffs_global_normalization(tmp_path):
    from tools.vis.leave_one_out_diff import compute_loo_diffs

    cache = _make_cache(tmp_path)
    diffs = compute_loo_diffs(cache)

    all_vals = np.concatenate([d.ravel() for d in diffs.values()])
    assert all_vals.max() <= 1.0 + 1e-6
    assert all_vals.max() > 0.0


def test_relative_diff_maps_global_normalization() -> None:
    from tools.vis.leave_one_out_diff import _compute_relative_diff_maps

    baseline = np.zeros((2, 2, 3), dtype=np.uint8)
    images = [
        np.zeros((2, 2, 3), dtype=np.uint8),
        np.full((2, 2, 3), 10, dtype=np.uint8),
        np.full((2, 2, 3), 20, dtype=np.uint8),
    ]

    diff_maps = _compute_relative_diff_maps(images, baseline)

    assert len(diff_maps) == 3
    assert diff_maps[0].shape == (2, 2)
    assert diff_maps[0].dtype == np.float32
    assert float(diff_maps[0].max()) == 0.0
    assert float(diff_maps[1].max()) == 0.5
    assert float(diff_maps[2].max()) == 1.0


def test_maybe_contour_cell_mask_calls_contour_with_resized_mask() -> None:
    from tools.vis.leave_one_out_diff import _maybe_contour_cell_mask

    calls: list[tuple[np.ndarray, list[float]]] = []

    class _DummyAxes:
        def contour(self, arr, levels, **_kwargs):
            calls.append((np.asarray(arr), list(levels)))

    cell_mask = np.zeros((2, 2), dtype=np.float32)
    cell_mask[0, 0] = 1.0
    _maybe_contour_cell_mask(_DummyAxes(), cell_mask, (4, 4))

    assert len(calls) == 1
    contour_arr, levels = calls[0]
    assert contour_arr.shape == (4, 4)
    assert levels == [0.5]


def test_save_stats_and_render_figure(tmp_path):
    from tools.vis.leave_one_out_diff import compute_loo_diffs, render_loo_diff_figure, save_loo_stats

    cache = _make_cache(tmp_path)
    diffs = compute_loo_diffs(cache)

    stats_path = tmp_path / "leave_one_out_diff_stats.json"
    fig_path = tmp_path / "leave_one_out_diff.png"
    save_loo_stats(diffs, stats_path)
    render_loo_diff_figure(diffs, cache, out_path=fig_path)

    assert stats_path.is_file()
    assert fig_path.is_file()
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    assert set(stats) == {"cell_types", "cell_state", "vasculature", "microenv"}
    assert stats["cell_types"]["max_diff"] > 0.0


def test_render_loo_figure_with_cached_cell_mask(tmp_path):
    from tools.vis.leave_one_out_diff import compute_loo_diffs, render_loo_diff_figure

    cache = _make_cache(tmp_path)
    _write_cell_mask(cache / "cell_mask.png")
    manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
    manifest["cell_mask_path"] = "cell_mask.png"
    (cache / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    fig_path = tmp_path / "leave_one_out_diff_with_mask.png"
    render_loo_diff_figure(compute_loo_diffs(cache), cache, out_path=fig_path)

    assert fig_path.is_file()


def test_render_loo_cache_root_renders_multiple_tile_dirs(tmp_path):
    from tools.vis.leave_one_out_diff import render_loo_cache_root

    cache_root = tmp_path / "cache"
    cache_a = _make_cache(cache_root / "512_9728")
    cache_b = _make_cache(cache_root / "16896_40448")
    out_root = tmp_path / "loo_outputs"

    rendered = render_loo_cache_root(cache_root, out_root=out_root, workers=2, show_progress=False)

    assert len(rendered) == 2
    expected_paths = {
        out_root / cache_a.relative_to(cache_root) / "leave_one_out_diff.png",
        out_root / cache_b.relative_to(cache_root) / "leave_one_out_diff.png",
    }
    actual_paths = {fig_path for fig_path, _ in rendered}
    assert actual_paths == expected_paths
    for fig_path, stats_path in rendered:
        assert fig_path.is_file()
        assert stats_path.is_file()


def test_render_loo_figure_uses_style_mapping_for_reference(tmp_path):
    from tools.vis.leave_one_out_diff import compute_loo_diffs, render_loo_diff_figure

    cache = _make_cache(tmp_path / "cache")
    orion_root = tmp_path / "orion"
    (orion_root / "he").mkdir(parents=True, exist_ok=True)
    style_tile = "style_tile"
    Image.fromarray(np.full((4, 4, 3), 123, dtype=np.uint8)).save(orion_root / "he" / f"{style_tile}.png")
    diffs = compute_loo_diffs(cache)

    fig_path = tmp_path / "mapped_loo.png"
    render_loo_diff_figure(
        diffs,
        cache,
        orion_root=orion_root,
        style_mapping={"test_tile": style_tile},
        out_path=fig_path,
    )

    assert fig_path.is_file()


# ── _normalize_cell_masked_diff ───────────────────────────────────────────────

def test_normalize_cell_masked_diff_zeroes_background_and_normalizes_cells() -> None:
    from tools.vis.leave_one_out_diff import _normalize_cell_masked_diff

    diff = np.full((4, 4), 100.0, dtype=np.float32)
    cell_mask = np.zeros((4, 4), dtype=np.float32)
    cell_mask[1:3, 1:3] = 1.0  # 4 cell pixels in the centre

    result = _normalize_cell_masked_diff(diff, cell_mask)

    assert result.dtype == np.float32
    assert result.shape == (4, 4)
    # Background pixels are zeroed out
    assert float(result[0, 0]) == 0.0
    assert float(result[3, 3]) == 0.0
    # Cell pixels: uniform diff=100, p99=100 → normalized to 1.0
    assert float(result[1, 1]) == pytest.approx(1.0, abs=1e-5)
    assert result.min() >= 0.0
    assert result.max() <= 1.0 + 1e-6


def test_normalize_cell_masked_diff_empty_mask_returns_zeros() -> None:
    from tools.vis.leave_one_out_diff import _normalize_cell_masked_diff

    diff = np.full((4, 4), 50.0, dtype=np.float32)
    cell_mask = np.zeros((4, 4), dtype=np.float32)

    result = _normalize_cell_masked_diff(diff, cell_mask)

    assert float(result.max()) == 0.0


def test_normalize_cell_masked_diff_zero_diff_returns_zeros() -> None:
    from tools.vis.leave_one_out_diff import _normalize_cell_masked_diff

    diff = np.zeros((4, 4), dtype=np.float32)
    cell_mask = np.ones((4, 4), dtype=np.float32)

    result = _normalize_cell_masked_diff(diff, cell_mask)

    assert float(result.max()) == 0.0


# ── _render_cell_masked_overlay ───────────────────────────────────────────────

def test_render_cell_masked_overlay_returns_scalar_mappable() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as mcm
    from tools.vis.leave_one_out_diff import _render_cell_masked_overlay

    hot_cmap = mcolors.LinearSegmentedColormap.from_list(
        "hot4", ["#000000", "#ff4400", "#ffff00", "#ffffff"]
    )
    fig, ax = plt.subplots()
    raw_diff = np.full((4, 4), 80.0, dtype=np.float32)
    cell_mask = np.zeros((4, 4), dtype=np.float32)
    cell_mask[1:3, 1:3] = 1.0
    baseline_he = np.full((4, 4, 3), 128, dtype=np.uint8)

    result = _render_cell_masked_overlay(ax, raw_diff, cell_mask, baseline_he, hot_cmap)

    assert isinstance(result, mcm.ScalarMappable)
    plt.close(fig)


def test_render_cell_masked_overlay_no_crash_empty_mask() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from tools.vis.leave_one_out_diff import _render_cell_masked_overlay

    hot_cmap = mcolors.LinearSegmentedColormap.from_list(
        "hot4", ["#000000", "#ff4400", "#ffff00", "#ffffff"]
    )
    fig, ax = plt.subplots()
    raw_diff = np.full((4, 4), 30.0, dtype=np.float32)
    cell_mask = np.zeros((4, 4), dtype=np.float32)  # no cell pixels
    baseline_he = np.full((4, 4, 3), 200, dtype=np.uint8)

    _render_cell_masked_overlay(ax, raw_diff, cell_mask, baseline_he, hot_cmap)  # must not raise

    plt.close(fig)


def test_render_cell_masked_overlay_no_crash_zero_diff() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from tools.vis.leave_one_out_diff import _render_cell_masked_overlay

    hot_cmap = mcolors.LinearSegmentedColormap.from_list(
        "hot4", ["#000000", "#ff4400", "#ffff00", "#ffffff"]
    )
    fig, ax = plt.subplots()
    raw_diff = np.zeros((4, 4), dtype=np.float32)  # self-diff column (all-four baseline)
    cell_mask = np.ones((4, 4), dtype=np.float32)
    baseline_he = np.full((4, 4, 3), 128, dtype=np.uint8)

    _render_cell_masked_overlay(ax, raw_diff, cell_mask, baseline_he, hot_cmap)  # must not raise

    plt.close(fig)


# ── render_loo_diff_figure overlay integration ────────────────────────────────

def test_render_loo_figure_overlay_path_produces_png(tmp_path) -> None:
    """With a cell mask present, render_loo_diff_figure must produce a PNG."""
    from tools.vis.leave_one_out_diff import compute_loo_diffs, render_loo_diff_figure

    cache = _make_cache(tmp_path)
    _write_cell_mask(cache / "cell_mask.png")
    manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
    manifest["cell_mask_path"] = "cell_mask.png"
    (cache / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    fig_path = tmp_path / "overlay.png"
    render_loo_diff_figure(compute_loo_diffs(cache), cache, out_path=fig_path)

    assert fig_path.is_file()
    assert fig_path.stat().st_size > 0


def test_render_loo_figure_no_mask_fallback_produces_png(tmp_path) -> None:
    """Without a cell mask, render_loo_diff_figure falls back to per-map normalisation."""
    from tools.vis.leave_one_out_diff import compute_loo_diffs, render_loo_diff_figure

    cache = _make_cache(tmp_path)
    fig_path = tmp_path / "fallback.png"
    render_loo_diff_figure(compute_loo_diffs(cache), cache, out_path=fig_path)

    assert fig_path.is_file()
    assert fig_path.stat().st_size > 0


# ── ssim_loss_map ─────────────────────────────────────────────────────────────

def test_ssim_loss_map_identical_images_returns_near_zero() -> None:
    from tools.vis.leave_one_out_diff import ssim_loss_map

    img = np.full((32, 32, 3), 128, dtype=np.uint8)
    result = ssim_loss_map(img, img)

    assert result.shape == (32, 32)
    assert result.dtype == np.float32
    assert float(result.max()) < 0.05  # near-zero for identical images


def test_ssim_loss_map_different_images_has_nonzero_loss() -> None:
    from tools.vis.leave_one_out_diff import ssim_loss_map

    img_a = np.zeros((32, 32, 3), dtype=np.uint8)
    img_b = np.full((32, 32, 3), 200, dtype=np.uint8)
    result = ssim_loss_map(img_a, img_b)

    assert float(result.mean()) > 0.1


def test_ssim_loss_map_values_in_0_1() -> None:
    from tools.vis.leave_one_out_diff import ssim_loss_map

    rng = np.random.default_rng(42)
    img_a = rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)
    img_b = rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)
    result = ssim_loss_map(img_a, img_b)

    assert result.min() >= 0.0
    assert result.max() <= 1.0 + 1e-6


# ── _select_inset_region ──────────────────────────────────────────────────────

def test_select_inset_region_picks_highest_loss_region() -> None:
    from tools.vis.leave_one_out_diff import _select_inset_region

    loss = np.zeros((128, 128), dtype=np.float32)
    loss[64:80, 64:80] = 1.0  # hot spot

    y, x = _select_inset_region(loss, crop=64, stride=8)

    # The selected crop must overlap with the hot spot
    assert y <= 64 and y + 64 >= 80, f"y={y} misses hot spot rows 64–80"
    assert x <= 64 and x + 64 >= 80, f"x={x} misses hot spot cols 64–80"


def test_select_inset_region_returns_valid_crop_bounds() -> None:
    from tools.vis.leave_one_out_diff import _select_inset_region

    rng = np.random.default_rng(7)
    loss = rng.random((256, 256)).astype(np.float32)

    y, x = _select_inset_region(loss, crop=64, stride=8)

    assert 0 <= y <= 256 - 64, f"y={y} out of bounds"
    assert 0 <= x <= 256 - 64, f"x={x} out of bounds"


def test_select_inset_region_uniform_loss_returns_origin() -> None:
    """Uniform loss: first window wins (top-left corner)."""
    from tools.vis.leave_one_out_diff import _select_inset_region

    loss = np.ones((128, 128), dtype=np.float32)
    y, x = _select_inset_region(loss, crop=64, stride=8)

    assert y == 0
    assert x == 0


# ── render_loo_ssim_figure ────────────────────────────────────────────────────

def _make_cache_large(tmp_path: Path, resolution: int = 128) -> Path:
    """Like _make_cache but with resolution×resolution images for SSIM tests."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    group_names = ("cell_types", "cell_state", "vasculature", "microenv")

    all_img = np.full((resolution, resolution, 3), 200, dtype=np.uint8)
    (tmp_path / "all").mkdir()
    Image.fromarray(all_img).save(tmp_path / "all" / "generated_he.png")

    (tmp_path / "triples").mkdir()
    entries_triples = []
    for i, omit in enumerate(group_names):
        active = [g for g in group_names if g != omit]
        val = 80 + i * 30
        img = np.full((resolution, resolution, 3), val, dtype=np.uint8)
        fname = f"{i + 1:02d}_{'__'.join(active)}.png"
        Image.fromarray(img).save(tmp_path / "triples" / fname)
        entries_triples.append({
            "active_groups": active,
            "condition_label": f"triples_{i}",
            "image_label": f"lbl_{i}",
            "image_path": f"triples/{fname}",
        })

    manifest = {
        "version": 1,
        "tile_id": "large_tile",
        "group_names": list(group_names),
        "sections": [
            {"title": "3 active groups", "subset_size": 3, "entries": entries_triples},
            {
                "title": "4 active groups",
                "subset_size": 4,
                "entries": [{
                    "active_groups": list(group_names),
                    "condition_label": "all",
                    "image_label": "all",
                    "image_path": "all/generated_he.png",
                }],
            },
        ],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return tmp_path


def test_render_loo_ssim_figure_produces_png(tmp_path) -> None:
    from tools.vis.leave_one_out_diff import render_loo_ssim_figure

    cache = _make_cache_large(tmp_path / "cache")
    out_path = tmp_path / "leave_one_out_ssim.png"
    render_loo_ssim_figure(cache, out_path=out_path)

    assert out_path.is_file()
    assert out_path.stat().st_size > 0


def test_render_loo_ssim_figure_with_cell_mask(tmp_path) -> None:
    from tools.vis.leave_one_out_diff import render_loo_ssim_figure

    cache = _make_cache_large(tmp_path / "cache")
    mask = np.zeros((128, 128), dtype=np.uint8)
    mask[32:96, 32:96] = 255
    Image.fromarray(mask).save(cache / "cell_mask.png")
    manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
    manifest["cell_mask_path"] = "cell_mask.png"
    (cache / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    out_path = tmp_path / "ssim_with_mask.png"
    render_loo_ssim_figure(cache, out_path=out_path)

    assert out_path.is_file()
    assert out_path.stat().st_size > 0


def test_render_loo_cache_ssim_mode_writes_only_ssim_and_stats(tmp_path) -> None:
    from tools.vis.leave_one_out_diff import render_loo_cache

    cache = _make_cache_large(tmp_path / "cache")
    ssim_path = tmp_path / "custom_ssim.png"

    fig_path, stats_path = render_loo_cache(
        cache,
        out_path=ssim_path,
        figure_mode="ssim",
        crop_size=32,
    )

    assert fig_path == ssim_path
    assert fig_path.is_file()
    assert stats_path.is_file()
    assert not (cache / "leave_one_out_diff.png").exists()


def test_render_loo_cache_root_ssim_mode_uses_ssim_filename(tmp_path) -> None:
    from tools.vis.leave_one_out_diff import render_loo_cache_root

    cache_root = tmp_path / "cache"
    _make_cache_large(cache_root / "tile_a")
    _make_cache_large(cache_root / "tile_b")
    out_root = tmp_path / "loo_outputs"

    rendered = render_loo_cache_root(
        cache_root,
        out_root=out_root,
        workers=2,
        show_progress=False,
        figure_mode="ssim",
        crop_size=32,
    )

    assert len(rendered) == 2
    for fig_path, stats_path in rendered:
        assert fig_path.name == "leave_one_out_ssim.png"
        assert fig_path.is_file()
        assert stats_path.is_file()


def test_main_ssim_mode_writes_requested_output(tmp_path, capsys) -> None:
    from tools.vis.leave_one_out_diff import main

    cache = _make_cache_large(tmp_path / "cache")
    out_path = tmp_path / "tile_ssim.png"

    main([
        "--cache-dir", str(cache),
        "--figure", "ssim",
        "--crop-size", "32",
        "--out", str(out_path),
    ])

    captured = capsys.readouterr()
    assert "Saved stats ->" in captured.out
    assert "Saved SSIM figure ->" in captured.out
    assert out_path.is_file()
