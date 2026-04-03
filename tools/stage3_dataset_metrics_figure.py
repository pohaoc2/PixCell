#!/usr/bin/env python3
"""
stage3_dataset_metrics_figure.py

Grouped bar figure: all five ablation metrics side-by-side for each of the
15 channel combinations (4+6+4+1).

Layout
------
* Single y-axis panel, normalized 0–1 per metric (1 = best).
  Lower-is-better metrics (FID, LPIPS) are inverted so taller always means better.
* 5 bars per combination, one per metric, colored by metric.
* Black outline on every bar; error whiskers in black.
* Dot strip below x-axis: filled black = group active, hollow = inactive.
* Vertical dashed separators partition 1→2→3→4 active groups.

Color assignment
----------------
  Metric bars (by metric):
    FID    → #9e9e9e  (gray — placeholder until dataset-level FID computed)
    Cosine → #1c4587  (deep blue)
    LPIPS  → #7b2d8b  (burgundy)
    AJI    → #1a5c38  (forest green)
    PQ     → #8b3a1a  (burnt sienna)
  Dot indicators: filled #111 (active) / hollow (inactive)

Usage
-----
  python tools/stage3_dataset_metrics_figure.py            # real cache
  python tools/stage3_dataset_metrics_figure.py --mock     # mock data
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.stage3.ablation_vis_utils import (
    FOUR_GROUP_ORDER,
    condition_metric_key,
)

# ── Palette ───────────────────────────────────────────────────────────────────
METRIC_CONFIG = [
    {"key": "fid",    "label": "FID",    "hib": False, "color": "#9e9e9e", "placeholder": True},
    {"key": "cosine", "label": "Cosine", "hib": True,  "color": "#1c4587", "placeholder": False},
    {"key": "lpips",  "label": "LPIPS",  "hib": False, "color": "#7b2d8b", "placeholder": False},
    {"key": "aji",    "label": "AJI",    "hib": True,  "color": "#1a5c38", "placeholder": False},
    {"key": "pq",     "label": "PQ",     "hib": True,  "color": "#8b3a1a", "placeholder": False},
]

DOT_ACTIVE   = "#111111"
DOT_INACTIVE = "#cccccc"
DOT_EDGE     = "#111111"

GROUP_SHORT: dict[str, str] = {
    "cell_types":  "CT",
    "cell_state":  "CS",
    "vasculature": "Vas",
    "microenv":    "Env",
}

matplotlib.rcParams.update({
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Helvetica", "Arial", "DejaVu Sans"],
    "axes.spines.top":   False,
    "axes.spines.right": False,
})


# ── Data helpers ──────────────────────────────────────────────────────────────

def _ordered_conditions() -> list[tuple[str, ...]]:
    result: list[tuple[str, ...]] = []
    for k in range(1, 5):
        result.extend(tuple(c) for c in combinations(FOUR_GROUP_ORDER, k))
    return result


def _lcg(seed: int) -> float:
    return ((seed * 1664525 + 1013904223) & 0xFFFFFFFF) / 4294967296


def load_cache_data(cache_root: Path) -> list[dict]:
    grouped: dict[str, dict[str, list[float]]] = {}
    n_tiles = 0
    for path in sorted(cache_root.glob("*/metrics.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        per_cond = payload.get("per_condition", {})
        if not isinstance(per_cond, dict):
            continue
        n_tiles += 1
        for cond_key, record in per_cond.items():
            if not isinstance(record, dict):
                continue
            bucket = grouped.setdefault(cond_key, {})
            for m in ("cosine", "lpips", "aji", "pq"):
                v = record.get(m)
                if v is not None:
                    bucket.setdefault(m, []).append(float(v))

    rows = []
    for cond_tuple in _ordered_conditions():
        cond_key = condition_metric_key(cond_tuple)
        cond_set = set(cond_tuple)
        record   = grouped.get(cond_key, {})
        metrics: dict[str, dict | None] = {}
        for m in ("cosine", "lpips", "aji", "pq"):
            vals = record.get(m, [])
            metrics[m] = (
                {"mean": statistics.mean(vals),
                 "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0}
                if vals else None
            )
        metrics["fid"] = None
        rows.append(_make_row(cond_tuple, cond_set, metrics))
    print(f"Aggregated {n_tiles} tiles, {len(rows)} conditions.")
    return rows


def make_mock_data() -> list[dict]:
    base  = {"fid": 158.0, "cosine": 0.410, "lpips": 0.478, "aji": 0.018, "pq": 0.014}
    bonus = {
        "fid":    {"cell_types": -24, "cell_state": -10, "vasculature": -17, "microenv":  -7},
        "cosine": {"cell_types": 0.052, "cell_state": 0.020, "vasculature": 0.030, "microenv": 0.014},
        "lpips":  {"cell_types": -0.011,"cell_state":-0.005,"vasculature":-0.009,"microenv":-0.003},
        "aji":    {"cell_types": 0.108, "cell_state": 0.065, "vasculature": 0.050, "microenv": 0.024},
        "pq":     {"cell_types": 0.095, "cell_state": 0.055, "vasculature": 0.048, "microenv": 0.020},
    }
    noise = {"fid": 9.5, "cosine": 0.019, "lpips": 0.007, "aji": 0.016, "pq": 0.013}
    rows  = []
    for cond_tuple in _ordered_conditions():
        cond_set = set(cond_tuple)
        metrics: dict[str, dict | None] = {}
        for m in ("fid", "cosine", "lpips", "aji", "pq"):
            mean = base[m]
            for g in cond_tuple:
                mean += bonus[m][g]
            seed   = hash(cond_tuple) ^ hash(m)
            jitter = (_lcg(seed) - 0.5) * 2 * noise[m] * 0.30
            std    = noise[m] * (0.50 + 0.50 * len(cond_tuple) / 4)
            metrics[m] = {"mean": mean + jitter, "std": std}
        rows.append(_make_row(cond_tuple, cond_set, metrics))
    return rows


def _make_row(cond_tuple, cond_set, metrics):
    return {
        "cond_tuple": cond_tuple,
        "n":          len(cond_tuple),
        "bits":       [g in cond_set for g in FOUR_GROUP_ORDER],
        "metrics":    metrics,
    }


# ── Normalization ─────────────────────────────────────────────────────────────

def normalize_metric(means: np.ndarray, stds: np.ndarray, hib: bool):
    """
    Scale means to [0, 1] where 1 = best combination.
    Lower-is-better metrics are inverted so taller bar always means better.
    Returns (norm_means, norm_stds) — stds are scaled proportionally.
    """
    valid = means[~np.isnan(means)]
    if len(valid) < 2 or valid.max() == valid.min():
        nm = np.where(np.isnan(means), np.nan, 0.5)
        return nm, stds * 0.0
    lo, hi = float(valid.min()), float(valid.max())
    span = hi - lo
    if hib:
        nm = (means - lo) / span
    else:
        nm = (hi - means) / span   # invert: lower raw → taller bar
    ns = stds / span
    return nm, ns


# ── Figure ────────────────────────────────────────────────────────────────────

def build_figure(rows: list[dict], output: Path) -> None:
    n_comb    = len(rows)       # 15
    n_metrics = len(METRIC_CONFIG)  # 5

    # Bar geometry (in data units; each comb occupies 1 unit)
    bar_w    = 0.12
    bar_gap  = 0.018
    group_w  = n_metrics * bar_w + (n_metrics - 1) * bar_gap
    dot_frac = 0.40   # dot-strip relative height

    fig_w, fig_h = 16, 7.5
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs  = gridspec.GridSpec(
        2, 1,
        height_ratios=[1.0, dot_frac],
        hspace=0.04,
        left=0.07, right=0.98, top=0.91, bottom=0.04,
    )
    bar_ax = fig.add_subplot(gs[0])
    dot_ax = fig.add_subplot(gs[1])
    bar_ax.sharex(dot_ax)

    xs = np.arange(n_comb)

    # ── Pre-compute normalized arrays ────────────────────────────────────
    norm_data = {}  # key → (norm_means, norm_stds)
    for mc in METRIC_CONFIG:
        key   = mc["key"]
        means = np.array([r["metrics"][key]["mean"] if r["metrics"][key] else np.nan for r in rows])
        stds  = np.array([r["metrics"][key]["std"]  if r["metrics"][key] else 0.0    for r in rows])
        nm, ns = normalize_metric(means, stds, mc["hib"])
        norm_data[key] = (nm, ns)

    # ── Cardinality separators ────────────────────────────────────────────
    sep_xs = [i - 0.5 for i in range(1, n_comb) if rows[i]["n"] != rows[i-1]["n"]]
    for sx in sep_xs:
        bar_ax.axvline(sx, color="#bbb", linestyle="--", linewidth=0.85,
                       alpha=0.75, zorder=1)

    # ── Bars ──────────────────────────────────────────────────────────────
    for mi, mc in enumerate(METRIC_CONFIG):
        key    = mc["key"]
        nm, ns = norm_data[key]
        color  = mc["color"]
        # x offset for this metric within group
        x_off  = -group_w / 2 + mi * (bar_w + bar_gap) + bar_w / 2

        for ci in range(n_comb):
            mean_val = nm[ci]
            std_val  = ns[ci]
            if np.isnan(mean_val):
                # Placeholder: gray hatched bar at 0 height marker
                bar_ax.bar(
                    xs[ci] + x_off, 0.04, width=bar_w,
                    color="white", edgecolor="#bbb", linewidth=0.8,
                    hatch="///", alpha=0.6, zorder=2,
                )
                continue
            bar_ax.bar(
                xs[ci] + x_off, mean_val, width=bar_w,
                color=color, edgecolor="#111", linewidth=0.7,
                alpha=0.88, zorder=2,
            )
            # Error whiskers
            err_top = min(1.0, mean_val + std_val)
            err_bot = max(0.0, mean_val - std_val)
            cap_w   = bar_w * 0.36
            bar_ax.plot(
                [xs[ci] + x_off, xs[ci] + x_off], [err_bot, err_top],
                color="#111", linewidth=1.0, zorder=3,
            )
            bar_ax.plot(
                [xs[ci] + x_off - cap_w/2, xs[ci] + x_off + cap_w/2],
                [err_top, err_top], color="#111", linewidth=1.0, zorder=3,
            )
            bar_ax.plot(
                [xs[ci] + x_off - cap_w/2, xs[ci] + x_off + cap_w/2],
                [err_bot, err_bot], color="#111", linewidth=1.0, zorder=3,
            )

    # ── Bar axis styling ──────────────────────────────────────────────────
    bar_ax.set_xlim(-0.5, n_comb - 0.5)
    bar_ax.set_ylim(0, 1.10)
    bar_ax.set_ylabel("Normalized performance  (↑ = better)", fontsize=10, labelpad=4)
    bar_ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    bar_ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
    bar_ax.tick_params(axis="x", bottom=False, labelbottom=False)
    bar_ax.tick_params(axis="y", labelsize=8)
    bar_ax.spines["bottom"].set_visible(False)
    bar_ax.axhline(0, color="#333", linewidth=0.8)

    # Cardinality labels
    boundaries = [0] + [i for i in range(1, n_comb) if rows[i]["n"] != rows[i-1]["n"]] + [n_comb]
    for bi in range(len(boundaries) - 1):
        mid  = (boundaries[bi] + boundaries[bi+1] - 1) / 2
        card = rows[boundaries[bi]]["n"]
        bar_ax.text(mid, 1.085,
                    f"{card} group{'s' if card > 1 else ''}",
                    ha="center", va="bottom", fontsize=8, color="#888")

    # ── Dot strip ─────────────────────────────────────────────────────────
    dot_ax.set_xlim(-0.5, n_comb - 0.5)
    dot_ax.set_ylim(0, 1)
    dot_ax.axis("off")

    DOT_S   = 38
    offsets = np.linspace(-0.28, 0.28, 4)
    dot_y   = 0.50

    for ci, row in enumerate(rows):
        for gi, (g, active) in enumerate(zip(FOUR_GROUP_ORDER, row["bits"])):
            dx   = xs[ci] + offsets[gi]
            face = DOT_ACTIVE if active else "white"
            edge = DOT_EDGE if active else DOT_INACTIVE
            dot_ax.scatter(
                [dx], [dot_y], s=DOT_S,
                c=[face], edgecolors=[edge], linewidths=0.9,
                zorder=3, clip_on=False,
            )

    for sx in sep_xs:
        dot_ax.axvline(sx, color="#bbb", linestyle="--", linewidth=0.85, alpha=0.75)

    # Group abbreviation label left of dot row
    short_str = " · ".join(GROUP_SHORT[g] for g in FOUR_GROUP_ORDER)
    dot_ax.text(-0.52, dot_y, short_str,
                ha="right", va="center", fontsize=8, color="#555")

    # ── Legend ────────────────────────────────────────────────────────────
    metric_handles = []
    for mc in METRIC_CONFIG:
        direction = "↑" if mc["hib"] else "↓ inv."
        label     = f"{mc['label']}  ({direction})"
        if mc.get("placeholder"):
            label += "  [placeholder]"
        h = mpatches.Patch(
            facecolor=mc["color"], edgecolor="#111", linewidth=0.7,
            alpha=0.88, label=label,
        )
        metric_handles.append(h)

    dot_active_h  = mlines.Line2D([], [], marker="o", color="w",
                                   markerfacecolor=DOT_ACTIVE,
                                   markeredgecolor=DOT_EDGE,
                                   markersize=6, label="Group active")
    dot_inactive_h = mlines.Line2D([], [], marker="o", color="w",
                                    markerfacecolor="white",
                                    markeredgecolor=DOT_INACTIVE,
                                    markersize=6, label="Group inactive")

    bar_ax.legend(
        handles=metric_handles + [dot_active_h, dot_inactive_h],
        title="Metric  (bar color)",
        title_fontsize=8, fontsize=8,
        loc="upper left", ncol=2,
        frameon=True, framealpha=0.92,
        bbox_to_anchor=(0.0, 1.0),
        handlelength=1.2, columnspacing=1.0,
    )

    fig.suptitle(
        "Dataset-Level Ablation: All Metrics vs. Channel Combination",
        fontsize=12, fontweight="bold", y=0.975,
    )

    # ── Save ─────────────────────────────────────────────────────────────
    for ext in ("png", "pdf"):
        out = output.with_suffix(f".{ext}")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved → {out}")
    plt.close(fig)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Grouped bar figure: all 5 ablation metrics per channel combination."
    )
    parser.add_argument("--cache-root", type=Path,
                        default=ROOT / "inference_output/cache")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "inference_output/dataset_metrics_figure")
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()

    if args.mock or not args.cache_root.exists():
        if not args.mock:
            print("Cache not found, using mock data.")
        rows = make_mock_data()
    else:
        rows = load_cache_data(args.cache_root)

    build_figure(rows, args.output)


if __name__ == "__main__":
    main()
