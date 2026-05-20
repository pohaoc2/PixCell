"""3-way ANOVA variance decomposition for the combinatorial sweep.

Public API: variance_partition(rows, metrics).

For each metric the model is:
    y = mu + alpha_a + beta_s + gamma_o + delta_g
        + (beta gamma)_{s,o} + (beta delta)_{s,g} + (gamma delta)_{o,g}
        + (beta gamma delta)_{s,o,g} + epsilon

Sum-of-squares for each term is computed by sequential group-mean projection
(Type I SS, ordered: anchor -> state -> o2 -> gluc -> s*o -> s*g -> o*g -> s*o*g).
Anchor*state-style 2-way terms involving the anchor are absorbed into the
anchor bucket because the anchor factor itself is structural, not grammatical.
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np


_FACTOR_COLUMNS = ("anchor_id", "cell_state", "oxygen_label", "glucose_label")
_ORDERED_TERMS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("anchor", ("anchor_id",)),
    ("state",  ("cell_state",)),
    ("o2",     ("oxygen_label",)),
    ("gluc",   ("glucose_label",)),
    ("s_x_o",  ("cell_state", "oxygen_label")),
    ("s_x_g",  ("cell_state", "glucose_label")),
    ("o_x_g",  ("oxygen_label", "glucose_label")),
    ("s_x_o_x_g", ("cell_state", "oxygen_label", "glucose_label")),
)


def _group_means(values: np.ndarray, keys: np.ndarray) -> np.ndarray:
    """Return per-row group mean using composite key array."""
    out = np.empty_like(values, dtype=np.float64)
    uniq, inverse = np.unique(keys, return_inverse=True)
    for idx in range(uniq.size):
        mask = inverse == idx
        out[mask] = float(values[mask].mean())
    return out


def _composite_key(rows_by_factor: dict[str, np.ndarray], factors: tuple[str, ...]) -> np.ndarray:
    columns = [rows_by_factor[name].astype(str) for name in factors]
    return np.array(["\x1f".join(parts) for parts in zip(*columns, strict=True)])


def variance_partition(
    rows: Iterable[dict[str, Any]],
    metrics: tuple[str, ...],
) -> dict[str, dict[str, float]]:
    """Decompose total variance per metric into named factor shares.

    Returns: {metric_name: {anchor, state, o2, gluc, s_x_o, s_x_g, o_x_g, s_x_o_x_g, resid}}
    Each inner dict sums to 1.0 (or to 0.0 if total variance is zero).
    """
    rows_list = list(rows)
    if not rows_list:
        return {metric: {term: 0.0 for term, _ in _ORDERED_TERMS} | {"resid": 0.0} for metric in metrics}

    rows_by_factor = {col: np.array([row[col] for row in rows_list]) for col in _FACTOR_COLUMNS}
    out: dict[str, dict[str, float]] = {}

    for metric in metrics:
        if metric not in rows_list[0]:
            raise KeyError(f"metric column missing from rows: {metric!r}")
        values = np.asarray([float(row[metric]) for row in rows_list], dtype=np.float64)
        finite_mask = np.isfinite(values)
        values = values[finite_mask]
        metric_rows_by_factor = {col: values_by_factor[finite_mask] for col, values_by_factor in rows_by_factor.items()}

        zero_shares = {term: 0.0 for term, _ in _ORDERED_TERMS} | {"resid": 0.0}
        if values.size < 2:
            out[metric] = zero_shares
            continue

        grand_mean = float(values.mean())
        total_ss = float(np.sum((values - grand_mean) ** 2))
        if total_ss <= 0.0 or not np.isfinite(total_ss):
            out[metric] = zero_shares
            continue

        shares: dict[str, float] = {}
        residual = values - grand_mean

        for term_name, factors in _ORDERED_TERMS:
            keys = _composite_key(metric_rows_by_factor, factors)
            term_effect = _group_means(residual, keys)
            ss_term = float(np.sum(term_effect ** 2))
            shares[term_name] = ss_term
            residual = residual - term_effect

        shares["resid"] = float(np.sum(residual ** 2))

        out[metric] = {key: max(0.0, value / total_ss) for key, value in shares.items()}

    return out
