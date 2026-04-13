from __future__ import annotations
import sys
from pathlib import Path
import json
import numpy as np
import pytest
import types

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if "torch" not in sys.modules:
    torch_stub = types.ModuleType("torch")

    class _DummyTensor:
        pass

    torch_stub.float16 = "float16"
    torch_stub.float32 = "float32"
    torch_stub.dtype = object
    torch_stub.Tensor = _DummyTensor
    sys.modules["torch"] = torch_stub

if "diffusers" not in sys.modules:
    diffusers_stub = types.ModuleType("diffusers")

    class _DummyScheduler:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def set_timesteps(self, *args, **kwargs) -> None:
            pass

    diffusers_stub.DDPMScheduler = _DummyScheduler
    sys.modules["diffusers"] = diffusers_stub

from tools.stage3.ablation_grid_figure import (
    _cardinality_color,
    _condition_label,
    _draw_dot_row,
    _draw_metric_bars_cell,
    _load_grid_metrics,
    _metric_fill_fraction,
    _load_cellvit_contours,
    _sort_conditions_by_metric,
    METRIC_BAR_PRESETS,
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
        {"pq": 0.41, "dice": 0.81, "fud": 45.0, "style_hed": 0.06},
        metric_names=METRIC_BAR_PRESETS["paired"],
    )
    plt.close(fig)


def test_metric_fill_fraction_supports_fud_and_inverts_lower_is_better():
    assert _metric_fill_fraction(0.40, "pq") == pytest.approx(0.40)
    assert _metric_fill_fraction(0.80, "dice") == pytest.approx(0.80)
    assert _metric_fill_fraction(50.0, "fud") == pytest.approx(0.50)
    assert _metric_fill_fraction(0.02, "style_hed") == pytest.approx(0.80)


def test_load_grid_metrics_maps_legacy_fid_to_fud(tmp_path: Path):
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "version": 2,
                "tile_id": "tile_a",
                "per_condition": {
                    "cell_types": {
                        "pq": 0.25,
                        "dice": 0.75,
                        "fid": 48.0,
                        "style_hed": 0.04,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    metrics = _load_grid_metrics(tmp_path)

    assert metrics["cell_types"]["pq"] == pytest.approx(0.25)
    assert metrics["cell_types"]["dice"] == pytest.approx(0.75)
    assert metrics["cell_types"]["fud"] == pytest.approx(48.0)
    assert metrics["cell_types"]["style_hed"] == pytest.approx(0.04)


def test_draw_dot_row_uses_circle_markers():
    import matplotlib.pyplot as plt
    from matplotlib.markers import MarkerStyle

    fig, ax = plt.subplots()
    _draw_dot_row(ax, ("cell_types", "microenv"))

    assert len(ax.collections) == 4
    expected = MarkerStyle("o").get_path().transformed(MarkerStyle("o").get_transform()).vertices
    actual = ax.collections[0].get_paths()[0].vertices
    assert np.allclose(actual, expected)
    assert np.allclose(ax.collections[0].get_offsets()[0], [0.25, 0.48])
    assert ax.collections[0].get_sizes()[0] == pytest.approx(65.0)
    assert np.allclose(ax.collections[0].get_facecolors()[0][:3], [0.0, 0.0, 0.0])
    assert np.allclose(ax.collections[1].get_facecolors()[0][:3], [1.0, 1.0, 1.0])
    assert ax.collections[0].get_clip_on() is False
    assert ax.patch.get_alpha() == pytest.approx(0.0)
    assert not ax.texts

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
