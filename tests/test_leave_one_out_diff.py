"""Tests for leave-one-out pixel diff core logic."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
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
