from tools.stage3.ablation_vis_utils import (
    cosine_metric_title,
    condition_metric_key,
    load_uni_cosine_scores,
    ordered_subset_condition_tuples,
    parse_uni_cosine_scores_json,
    public_group_names,
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
    assert scores == {}
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
