from __future__ import annotations

import numpy as np


def _fake_images(count: int) -> list[tuple[str, np.ndarray]]:
    return [
        (f"cond_{idx}", np.full((16, 16, 3), idx * 10, dtype=np.uint8))
        for idx in range(count)
    ]


def test_build_subset_ablation_sections_matches_expected_counts():
    from tools.stage3_ablation_full_vis import build_subset_ablation_sections

    group_names = ("cell_types", "cell_state", "vasculature", "microenv")
    sections = build_subset_ablation_sections(
        group_names,
        single_images=_fake_images(4),
        pair_images=_fake_images(6),
        triple_images=_fake_images(4),
    )

    assert [section.title for section in sections] == [
        "1 active group",
        "2 active groups",
        "3 active groups",
    ]
    assert [len(section.conditions) for section in sections] == [4, 6, 4]
    assert [len(section.images) for section in sections] == [4, 6, 4]
    assert sections[0].conditions[0].active_groups == ("cell_types",)
    assert sections[1].conditions[-1].active_groups == ("vasculature", "microenv")


def test_build_subset_ablation_sections_includes_all_four_when_requested():
    from tools.stage3_ablation_full_vis import build_subset_ablation_sections

    group_names = ("cell_types", "cell_state", "vasculature", "microenv")
    sections = build_subset_ablation_sections(
        group_names,
        single_images=_fake_images(4),
        pair_images=_fake_images(6),
        triple_images=_fake_images(4),
        all_four_images=_fake_images(1),
    )
    assert len(sections) == 4
    assert sections[-1].title == "4 active groups"
    assert len(sections[-1].conditions) == 1
    assert len(sections[-1].images) == 1


def test_save_condition_matrix_ablation_grid_writes_png(tmp_path):
    from tools.stage3_ablation_full_vis import (
        build_subset_ablation_sections,
        save_condition_matrix_ablation_grid,
    )

    group_names = ("cell_types", "cell_state", "vasculature", "microenv")
    sections = build_subset_ablation_sections(
        group_names,
        single_images=_fake_images(4),
        pair_images=_fake_images(6),
        triple_images=_fake_images(4),
    )

    save_path = tmp_path / "ablation_group_combinations.png"
    ctrl_full = np.zeros((2, 16, 16), dtype=np.float32)
    ctrl_full[0, 4:12, 4:12] = 1.0

    save_condition_matrix_ablation_grid(
        sections,
        save_path,
        group_names=group_names,
        ctrl_full=ctrl_full,
        active_channels=["cell_masks", "vasculature"],
    )

    assert save_path.exists()
    assert save_path.stat().st_size > 0


def test_cached_subset_roundtrip_renders_combined_figure(tmp_path):
    from tools.stage3_ablation_cache import load_subset_condition_cache, save_subset_condition_cache
    from tools.stage3_ablation_full_vis import (
        build_subset_ablation_sections,
        load_cached_subset_ablation_sections,
        render_cached_subset_ablation_figure,
    )

    group_names = ("cell_types", "cell_state", "vasculature", "microenv")
    sections = build_subset_ablation_sections(
        group_names,
        single_images=_fake_images(4),
        pair_images=_fake_images(6),
        triple_images=_fake_images(4),
    )
    cell_mask = np.zeros((16, 16), dtype=np.float32)
    cell_mask[5:11, 5:11] = 1.0

    manifest_path = save_subset_condition_cache(
        tmp_path,
        tile_id="tile_001",
        group_names=group_names,
        sections=sections,
        cell_mask=cell_mask,
    )

    assert manifest_path.exists()

    cache = load_subset_condition_cache(tmp_path)
    assert cache["tile_id"] == "tile_001"
    assert cache["group_names"] == group_names
    assert [section["subset_size"] for section in cache["sections"]] == [1, 2, 3]
    assert cache["cell_mask"] is not None

    loaded_group_names, loaded_sections, ctrl_full, active_channels = load_cached_subset_ablation_sections(tmp_path)
    assert loaded_group_names == group_names
    assert [section.title for section in loaded_sections] == [
        "1 active group",
        "2 active groups",
        "3 active groups",
    ]
    assert ctrl_full is not None
    assert ctrl_full.shape == (1, 16, 16)
    assert active_channels == ["cell_masks"]

    save_path = render_cached_subset_ablation_figure(tmp_path)
    assert save_path.exists()
    assert save_path.name == "ablation_group_combinations.png"
