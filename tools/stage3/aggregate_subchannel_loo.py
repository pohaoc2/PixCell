#!/usr/bin/env python3
"""Aggregate per-tile sub-channel LOO JSONs into a per-channel summary CSV.

Reads <out-dir>/<tile_id>/subchannel_loo_diff_stats.json files and writes:
  - <out-dir>/per_subchannel_summary.csv   (channel, mean/SEM/n for each metric)
  - <out-dir>/group_vs_subchannel_consistency.csv  (group means from sub-channel agg)
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "inference_output" / "subchannel_loo_n300"

SUB_CHANNELS = (
    "cell_type_healthy", "cell_type_cancer", "cell_type_immune",
    "cell_state_prolif", "cell_state_nonprolif", "cell_state_dead",
    "vasculature", "oxygen", "glucose",
)

# Group → sub-channel membership for the consistency bridge.
GROUPS = {
    "cell_types": ("cell_type_healthy", "cell_type_cancer", "cell_type_immune"),
    "cell_state": ("cell_state_prolif", "cell_state_nonprolif", "cell_state_dead"),
    "vasculature": ("vasculature",),
    "microenv": ("oxygen", "glucose"),
}

METRICS = ("mean_diff", "delta_e_mean", "delta_e_p99", "pct_pixels_above_10")


def _sem(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return stdev(values) / math.sqrt(len(values))


def _load_tile_jsons(out_dir: Path) -> dict[str, dict[str, dict[str, float]]]:
    """Return tile_id -> sub_channel -> metric dict."""
    tiles: dict[str, dict[str, dict[str, float]]] = {}
    for tile_dir in sorted(out_dir.iterdir()):
        if not tile_dir.is_dir() or tile_dir.name.startswith("_"):
            continue
        stats_path = tile_dir / "subchannel_loo_diff_stats.json"
        if not stats_path.is_file():
            continue
        try:
            tiles[tile_dir.name] = json.loads(stats_path.read_text())
        except json.JSONDecodeError:
            continue
    return tiles


def aggregate(out_dir: Path) -> tuple[Path, Path]:
    tiles = _load_tile_jsons(out_dir)
    if not tiles:
        raise FileNotFoundError(f"no subchannel_loo_diff_stats.json found under {out_dir}")

    # Per sub-channel: collect each metric across tiles.
    per_channel: dict[str, dict[str, list[float]]] = {
        sub: {m: [] for m in METRICS} for sub in SUB_CHANNELS
    }
    for tile_id, sub_dict in tiles.items():
        for sub, metric_dict in sub_dict.items():
            if sub not in per_channel:
                continue
            for m in METRICS:
                v = metric_dict.get(m)
                if v is not None:
                    per_channel[sub][m].append(float(v))

    sub_csv = out_dir / "per_subchannel_summary.csv"
    fieldnames = ["sub_channel", "group", "n"]
    for m in METRICS:
        fieldnames += [f"{m}_mean", f"{m}_sem"]

    sub_to_group = {sc: g for g, members in GROUPS.items() for sc in members}

    with sub_csv.open("w", encoding="utf-8", newline="") as h:
        w = csv.DictWriter(h, fieldnames=fieldnames)
        w.writeheader()
        for sub in SUB_CHANNELS:
            row = {"sub_channel": sub, "group": sub_to_group[sub]}
            n_obs = 0
            for m in METRICS:
                vals = per_channel[sub][m]
                if vals:
                    row[f"{m}_mean"] = round(mean(vals), 4)
                    row[f"{m}_sem"] = round(_sem(vals), 4)
                    n_obs = max(n_obs, len(vals))
                else:
                    row[f"{m}_mean"] = ""
                    row[f"{m}_sem"] = ""
            row["n"] = n_obs
            w.writerow(row)

    # Group-level consistency: mean over sub-channels' tile means.
    group_csv = out_dir / "group_vs_subchannel_consistency.csv"
    with group_csv.open("w", encoding="utf-8", newline="") as h:
        w = csv.writer(h)
        w.writerow(["group", "n_subchannels", "mean_diff_mean", "delta_e_mean_mean"])
        for g, members in GROUPS.items():
            md_vals = [mean(per_channel[s]["mean_diff"]) for s in members if per_channel[s]["mean_diff"]]
            de_vals = [mean(per_channel[s]["delta_e_mean"]) for s in members if per_channel[s]["delta_e_mean"]]
            w.writerow([
                g,
                len(members),
                round(mean(md_vals), 4) if md_vals else "",
                round(mean(de_vals), 4) if de_vals else "",
            ])

    print(f"wrote {sub_csv} ({len(tiles)} tiles aggregated)", flush=True)
    print(f"wrote {group_csv}", flush=True)
    return sub_csv, group_csv


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate sub-channel LOO JSONs")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    aggregate(args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
