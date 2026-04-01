from tools.stage3_ablation_vis_utils import (
    condition_metric_key,
    load_uni_cosine_scores,
    ordered_subset_condition_tuples,
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
    assert len(ordered_subset_condition_tuples()) == 14


def test_load_uni_cosine_scores(tmp_path):
    p = tmp_path / "uni_cosine_scores.json"
    p.write_text(
        '{"per_condition": {"cell_types": 0.9, "cell_state+cell_types": 0.85}}\n',
        encoding="utf-8",
    )
    d = load_uni_cosine_scores(tmp_path)
    assert d["cell_types"] == 0.9
    assert d["cell_state+cell_types"] == 0.85
