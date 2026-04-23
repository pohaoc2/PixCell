import pytest

from tools.stage3.ablation_vis_utils import (
    condition_metric_key,
    cosine_metric_title,
    default_orion_he_png_path,
    default_orion_uni_npy_path,
    discover_channel_pngs,
    load_uni_cosine_scores,
    normalize_active_groups,
    normalize_group_name,
    ordered_subset_condition_tuples,
    parse_uni_cosine_for_condition,
    parse_uni_cosine_scores_json,
    public_group_names,
    _fmt,
    _mean,
)


def test_public_group_names_legacy_alias():
    assert public_group_names(("vasculature", "cell_identity")) == (
        "cell_types",
        "vasculature",
    )


def test_condition_metric_key_sorts():
    assert condition_metric_key(("cell_types", "cell_state")) == "cell_state+cell_types"


def test_ordered_subset_count():
    assert len(ordered_subset_condition_tuples()) == 15


def test_load_uni_cosine_scores(tmp_path):
    p = tmp_path / "uni_cosine_scores.json"
    p.write_text(
        '{"per_condition": {"cell_types": 0.9, "cell_state+cell_types": 0.85}}\n',
        encoding="utf-8",
    )
    d = load_uni_cosine_scores(tmp_path)
    assert d["cell_types"] == 0.9
    assert d["cell_state+cell_types"] == 0.85


def test_parse_uni_cosine_scores_json_missing_is_empty(tmp_path):
    scores, title = parse_uni_cosine_scores_json(tmp_path)
    assert not scores
    assert title == "UNI cosine"


def test_parse_uni_cosine_scores_json_metric_title_and_filtering(tmp_path):
    p = tmp_path / "uni_cosine_scores.json"
    p.write_text(
        '{"metric": "rgb_pixel_cosine", "per_condition": {"cell_types": 0.9, "cell_state": null}}\n',
        encoding="utf-8",
    )
    scores, title = parse_uni_cosine_scores_json(tmp_path)
    assert title == "RGB cosine"
    assert scores == {"cell_types": 0.9}


def test_cosine_metric_title_defaults_to_uni():
    assert cosine_metric_title("uni_cosine") == "UNI cosine"
    assert cosine_metric_title(None) == "UNI cosine"


def test_normalize_group_name_unknown_returns_identity():
    assert normalize_group_name("cell_state") == "cell_state"


def test_normalize_group_name_legacy_alias():
    assert normalize_group_name("cell_identity") == "cell_types"


def test_normalize_active_groups_sorts():
    result = normalize_active_groups(("microenv", "cell_types"))
    assert result == ("cell_types", "microenv")


def test_normalize_active_groups_applies_alias():
    result = normalize_active_groups(("cell_identity", "vasculature"))
    assert "cell_types" in result
    assert "cell_identity" not in result


def test_mean_empty_returns_none():
    assert _mean([]) is None


def test_mean_single_value():
    assert _mean([0.5]) == pytest.approx(0.5)


def test_fmt_none_returns_dash():
    assert _fmt(None) == "-"


def test_fmt_value_three_decimals():
    result = _fmt(0.12345)
    assert result == "0.123"


def test_default_orion_uni_npy_path_structure(tmp_path):
    root = tmp_path
    p = default_orion_uni_npy_path(root, "tile_001")
    assert p == root / "features" / "tile_001_uni.npy"


def test_default_orion_uni_npy_path_with_style_mapping(tmp_path):
    mapping = {"tile_001": "tile_999"}
    p = default_orion_uni_npy_path(tmp_path, "tile_001", style_mapping=mapping)
    assert "tile_999" in str(p)


def test_default_orion_he_png_path_returns_none_when_missing(tmp_path):
    result = default_orion_he_png_path(tmp_path, "no_such_tile")
    assert result is None


def test_default_orion_he_png_path_finds_png(tmp_path):
    he_dir = tmp_path / "he"
    he_dir.mkdir()
    (he_dir / "tile_001.png").write_bytes(b"")
    result = default_orion_he_png_path(tmp_path, "tile_001")
    assert result is not None
    assert result.name == "tile_001.png"


def test_parse_uni_cosine_for_condition_found():
    raw = {"per_condition": {"cell_types": 0.9}}
    result = parse_uni_cosine_for_condition(raw, ("cell_types",))
    assert result == pytest.approx(0.9)


def test_parse_uni_cosine_for_condition_missing_key():
    raw = {"per_condition": {"cell_types": 0.9}}
    result = parse_uni_cosine_for_condition(raw, ("vasculature",))
    assert result is None


def test_parse_uni_cosine_for_condition_no_per_condition():
    raw = {}
    result = parse_uni_cosine_for_condition(raw, ("cell_types",))
    assert result is None


def test_discover_channel_pngs_matches_by_keyword(tmp_path):
    (tmp_path / "celltype_map.png").write_bytes(b"")
    (tmp_path / "vessel_mask.png").write_bytes(b"")
    result = discover_channel_pngs(tmp_path)
    assert "cell_types" in result
    assert "vasculature" in result


def test_discover_channel_pngs_empty_dir(tmp_path):
    result = discover_channel_pngs(tmp_path)
    assert not result


def test_load_uni_cosine_scores_empty_per_condition(tmp_path):
    p = tmp_path / "uni_cosine_scores.json"
    p.write_text('{"per_condition": {}}\n', encoding="utf-8")
    scores = load_uni_cosine_scores(tmp_path)
    assert not scores
