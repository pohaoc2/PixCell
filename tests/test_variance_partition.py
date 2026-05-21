"""Unit tests for src.a3_combinatorial_sweep.variance_partition."""
from __future__ import annotations

import pytest

from src.a3_combinatorial_sweep.variance_partition import variance_partition


def _row(anchor: str, state: str, ox: str, gluc: str, seed: int, value: float) -> dict:
    return {
        "anchor_id": anchor,
        "cell_state": state,
        "oxygen_label": ox,
        "glucose_label": gluc,
        "seed": seed,
        "metric": value,
    }


def _build_rows(value_fn) -> list[dict]:
    anchors = ("a1", "a2", "a3")
    states = ("prolif", "nonprolif", "dead")
    levels = ("low", "mid", "high")
    seeds = (42, 43)
    rows = []
    for anchor in anchors:
        for state in states:
            for ox in levels:
                for gluc in levels:
                    for seed in seeds:
                        rows.append(_row(anchor, state, ox, gluc, seed, value_fn(anchor, state, ox, gluc, seed)))
    return rows


def test_pure_additive_grammar_has_zero_interaction():
    state_eff = {"prolif": 1.0, "nonprolif": 0.0, "dead": -1.0}
    ox_eff = {"low": -0.5, "mid": 0.0, "high": 0.5}
    gluc_eff = {"low": -0.2, "mid": 0.0, "high": 0.2}
    rows = _build_rows(lambda a, s, o, g, _seed: state_eff[s] + ox_eff[o] + gluc_eff[g])

    shares = variance_partition(rows, metrics=("metric",))

    s = shares["metric"]
    assert s["s_x_o"] + s["s_x_g"] + s["o_x_g"] + s["s_x_o_x_g"] < 1e-9
    assert s["state"] > 0.3
    assert s["o2"] > 0.05
    assert s["gluc"] > 0.0
    assert abs(sum(s.values()) - 1.0) < 1e-9


def test_pure_anchor_variance():
    anchor_eff = {"a1": 1.0, "a2": 0.0, "a3": -1.0}
    rows = _build_rows(lambda a, *_args: anchor_eff[a])

    shares = variance_partition(rows, metrics=("metric",))

    s = shares["metric"]
    assert s["anchor"] > 0.95
    grammar = s["state"] + s["o2"] + s["gluc"] + s["s_x_o"] + s["s_x_g"] + s["o_x_g"] + s["s_x_o_x_g"]
    assert grammar < 1e-9


def test_strip_anchor_removes_anchor_share():
    anchor_eff = {'a1': 1.0, 'a2': 0.5, 'a3': 0.0}
    state_eff = {'prolif': 0.2, 'nonprolif': 0.0, 'dead': -0.2}
    rows = _build_rows(lambda a, s, o, g, _seed: anchor_eff[a] + state_eff[s])
    shares = variance_partition(rows, metrics=('metric',), strip_factor='anchor_id')
    s = shares['metric']
    assert s['anchor'] == 0.0
    assert s['state'] > 0.9
    assert abs(sum(s.values()) - 1.0) < 1e-9


def test_interaction_only_shows_up_as_interaction():
    rows = _build_rows(lambda a, s, o, g, _seed: 1.0 if (s == "prolif" and o == "high") else 0.0)

    shares = variance_partition(rows, metrics=("metric",))
    s = shares["metric"]
    assert s["s_x_o"] > 0.2


def test_seed_noise_lands_in_resid():
    rng_table = {(s, seed): 0.1 * seed for s, seed in [("prolif", 42), ("prolif", 43)]}
    rows = _build_rows(lambda a, s, o, g, seed: rng_table.get((s, seed), 0.0))

    shares = variance_partition(rows, metrics=("metric",))
    s = shares["metric"]
    assert s["resid"] > 0.0


def test_missing_metric_raises():
    rows = _build_rows(lambda *_: 0.0)
    with pytest.raises(KeyError):
        variance_partition(rows, metrics=("not_a_column",))


def test_nan_rows_are_dropped_per_metric():
    rows = _build_rows(lambda a, s, o, g, _seed: 1.0 if s == "prolif" else 0.0)
    rows[0]["metric"] = float("nan")
    rows[5]["metric"] = float("nan")
    shares = variance_partition(rows, metrics=("metric",))
    s = shares["metric"]
    assert s["state"] > 0.5
    assert abs(sum(s.values()) - 1.0) < 1e-9


def test_all_nan_metric_returns_zeros():
    rows = _build_rows(lambda *_: float("nan"))
    shares = variance_partition(rows, metrics=("metric",))
    s = shares["metric"]
    assert all(value == 0.0 for value in s.values())
