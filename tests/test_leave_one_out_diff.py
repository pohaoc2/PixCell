"""Tests for leave-one-out pixel diff core logic."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


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
