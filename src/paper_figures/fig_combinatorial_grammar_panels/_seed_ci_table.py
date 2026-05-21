"""SI panel: text table of per-condition mean +/- 95% bootstrap CI across seeds."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.a3_combinatorial_sweep.main import MORPHOLOGY_METRICS
from src.paper_figures.fig_combinatorial_grammar_panels._shared import LEVELS, STATES, read_csv

_N_BOOT = 1000
_RNG_SEED = 0


def _bootstrap_ci(values: np.ndarray, *, n_boot: int = _N_BOOT) -> tuple[float, float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(_RNG_SEED)
    means = np.empty(n_boot, dtype=np.float64)
    for k in range(n_boot):
        sample = rng.choice(finite, size=finite.size, replace=True)
        means[k] = sample.mean()
    lo = float(np.quantile(means, 0.025))
    hi = float(np.quantile(means, 0.975))
    return float(finite.mean()), lo, hi


def draw_seed_ci_table(fig: plt.Figure, subgrid, *, signatures_csv: Path) -> None:
    rows = read_csv(signatures_csv)
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        key = (row["cell_state"], row["oxygen_label"], row["glucose_label"])
        grouped.setdefault(key, []).append(row)

    headline_metric = MORPHOLOGY_METRICS[0]
    table_rows: list[list[str]] = []
    for state in STATES:
        for ox in LEVELS:
            for gluc in LEVELS:
                key = (state, ox, gluc)
                group = grouped.get(key, [])
                values = np.asarray([float(r[headline_metric]) for r in group], dtype=np.float64)
                mean, lo, hi = _bootstrap_ci(values)
                table_rows.append([state, ox, gluc, f"{mean:.3g}", f"[{lo:.3g}, {hi:.3g}]", str(values.size)])

    ax = fig.add_subplot(subgrid)
    ax.set_axis_off()
    ax.set_title(f"Seed bootstrap CI — {headline_metric}", fontsize=9, pad=4)
    table = ax.table(
        cellText=table_rows,
        colLabels=["state", "O2", "glucose", "mean", "95% CI", "n_tiles"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(6)
    table.scale(1.0, 1.05)
