from __future__ import annotations
import sys
from pathlib import Path
import json
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.stage3.ablation_grid_figure import (
    _cardinality_color,
    _condition_label,
    _draw_metric_bars_cell,
    _load_cellvit_contours,
    _sort_conditions_by_metric,
)
from tools.stage3.ablation_vis_utils import FOUR_GROUP_ORDER


def test_cardinality_color_all_four():
    assert _cardinality_color(1) == "#009E73"
    assert _cardinality_color(2) == "#0072B2"
    assert _cardinality_color(3) == "#D55E00"
    assert _cardinality_color(4) == "#9B59B6"


def test_condition_label_single():
    assert _condition_label(("cell_types",)) == "CT"
    assert _condition_label(("cell_state",)) == "CS"
    assert _condition_label(("vasculature",)) == "Vas"
    assert _condition_label(("microenv",)) == "Env"


def test_condition_label_multi_follows_four_group_order():
    # input order must not matter — output follows FOUR_GROUP_ORDER
    assert _condition_label(("cell_state", "cell_types")) == "CT+CS"


def test_condition_label_three():
    assert _condition_label(("cell_types", "cell_state", "microenv")) == "CT+CS+Env"


def test_condition_label_all_four():
    assert _condition_label(FOUR_GROUP_ORDER) == "CT+CS+Vas+Env"


from tools.stage3.ablation_vis_utils import ordered_subset_condition_tuples, condition_metric_key
from tools.stage3.ablation_grid_figure import ALL4CH_KEY


def test_sort_descending_by_score():
    conditions = ordered_subset_condition_tuples()  # 15 conditions
    # score = cardinality → 4-ch should rank first
    scores = {condition_metric_key(c): float(len(c)) for c in conditions}
    result = _sort_conditions_by_metric(conditions, scores, metric_name="cosine")
    assert result[0] == FOUR_GROUP_ORDER


def test_sort_lex_tiebreak():
    conditions = ordered_subset_condition_tuples()
    # All same score → lex order by key string
    scores = {condition_metric_key(c): 0.5 for c in conditions}
    result = _sort_conditions_by_metric(conditions, scores, metric_name="cosine")
    keys = [condition_metric_key(c) for c in result]
    assert keys == sorted(keys)


def test_sort_missing_scores_last():
    conditions = ordered_subset_condition_tuples()
    # Only 4-ch scored
    scores = {ALL4CH_KEY: 0.9}
    result = _sort_conditions_by_metric(conditions, scores, metric_name="cosine")
    assert result[0] == FOUR_GROUP_ORDER
    # unscored conditions sorted lex after the scored one
    unscored_keys = [condition_metric_key(c) for c in result[1:]]
    assert unscored_keys == sorted(unscored_keys)


def test_sort_lpips_ascending():
    conditions = ordered_subset_condition_tuples()
    scores = {condition_metric_key(c): float(len(c)) for c in conditions}
    result = _sort_conditions_by_metric(conditions, scores, metric_name="lpips")
    assert result[0] == ("cell_state",)


def test_draw_metric_bars_cell_no_crash():
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    _draw_metric_bars_cell(
        ax,
        {"cosine": 0.995, "lpips": None, "aji": 0.7, "pq": None},
        color="#0072B2",
    )
    plt.close(fig)


def test_load_cellvit_contours_from_sidecar(tmp_path: Path):
    image_path = tmp_path / "generated_he.png"
    image_path.write_bytes(b"")
    sidecar = tmp_path / "generated_he_cellvit_instances.json"
    sidecar.write_text(
        json.dumps(
            {
                "patch": image_path.name,
                "cells": [
                    {"contour": [[1, 2], [3, 4], [5, 6]]},
                    {"contour": [[10, 11], [12, 13], [14, 15]]},
                ],
            }
        ),
        encoding="utf-8",
    )
    contours = _load_cellvit_contours(image_path)
    assert len(contours) == 2
    assert contours[0].shape == (3, 2)
