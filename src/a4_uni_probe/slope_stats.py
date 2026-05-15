"""Shared slope + bootstrap CI helpers for the a4 sweep summaries."""

from __future__ import annotations

import numpy as np


def linear_slope(x: np.ndarray, y: np.ndarray) -> float:
    x_centered = x - float(np.mean(x))
    denom = float(np.dot(x_centered, x_centered))
    if denom == 0.0:
        return 0.0
    y_centered = y - float(np.mean(y))
    return float(np.dot(x_centered, y_centered) / denom)


def bootstrap_slope_summary(
    alphas: np.ndarray,
    values: np.ndarray,
    *,
    n_boot: int,
    seed: int = 0,
) -> dict[str, float | int | tuple[float, float]]:
    """Bootstrap slope of y~x with percentile CI95.

    Inputs are filtered to finite pairs before fitting. Returns slope_mean, slope_ci95
    (low, high), and the number of finite samples used.
    """
    alphas = np.asarray(alphas, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    valid = np.isfinite(alphas) & np.isfinite(values)
    n_valid = int(valid.sum())
    if n_valid < 2:
        return {
            "slope_mean": float("nan"),
            "slope_ci95": (float("nan"), float("nan")),
            "n": n_valid,
        }
    x = alphas[valid]
    y = values[valid]
    rng = np.random.default_rng(seed)
    slopes: list[float] = []
    for _ in range(n_boot):
        choice = rng.integers(0, x.size, size=x.size)
        slopes.append(float(linear_slope(x[choice], y[choice])))
    return {
        "slope_mean": float(np.mean(slopes)),
        "slope_ci95": (float(np.quantile(slopes, 0.025)), float(np.quantile(slopes, 0.975))),
        "n": n_valid,
    }
