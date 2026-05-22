#!/usr/bin/env python3
"""Aggregate per-subchannel layout impact from existing LOO images + CellViT sidecars.

For each tile:
  baseline PQ = PQ(all_baseline.png, GT cell mask)
  loo PQ = PQ(<sub_channel>/generated_he.png, GT cell mask)
  pq_drop = baseline PQ - loo PQ

Writes <out-dir>/per_subchannel_layout_summary.csv for figure 09b panel B.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import mean, stdev

from tools.compute_ablation_metrics import _compute_pq, _load_gt_instance_mask, run_cellvit


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "inference_output" / "subchannel_loo_n300"
DEFAULT_ORION_ROOT = ROOT / "data" / "orion-crc33"
SUBCHANNELS: tuple[str, ...] = (
    "cell_type_healthy",
    "cell_type_cancer",
    "cell_type_immune",
    "cell_state_prolif",
    "cell_state_nonprolif",
    "cell_state_dead",
    "vasculature",
    "oxygen",
    "glucose",
)
SUB_TO_GROUP = {
    "cell_type_healthy": "cell_types",
    "cell_type_cancer": "cell_types",
    "cell_type_immune": "cell_types",
    "cell_state_prolif": "cell_state",
    "cell_state_nonprolif": "cell_state",
    "cell_state_dead": "cell_state",
    "vasculature": "vasculature",
    "oxygen": "microenv",
    "glucose": "microenv",
}


def _sem(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return stdev(values) / math.sqrt(len(values))


def aggregate_subchannel_layout_metrics(out_dir: Path, *, orion_root: Path) -> Path:
    out_dir = Path(out_dir)
    orion_root = Path(orion_root)

    per_subchannel: dict[str, dict[str, list[float]]] = {
        sub: {"baseline_pq": [], "loo_pq": [], "pq_drop": []} for sub in SUBCHANNELS
    }
    tiles_used = 0

    for tile_dir in sorted(out_dir.iterdir()):
        if not tile_dir.is_dir() or tile_dir.name.startswith("_"):
            continue
        baseline_path = tile_dir / "all_baseline.png"
        if not baseline_path.is_file():
            continue

        try:
            baseline_pred = run_cellvit(baseline_path)
            gt_inst = _load_gt_instance_mask(orion_root, tile_dir.name, shape=baseline_pred.shape)
            _, _, baseline_pq = _compute_pq(gt_inst, baseline_pred)
        except (FileNotFoundError, RuntimeError, ValueError):
            continue

        used_any = False
        for sub_channel in SUBCHANNELS:
            image_path = tile_dir / sub_channel / "generated_he.png"
            if not image_path.is_file():
                continue
            try:
                pred_inst = run_cellvit(image_path)
                _, _, loo_pq = _compute_pq(gt_inst, pred_inst)
            except (FileNotFoundError, RuntimeError, ValueError):
                continue
            per_subchannel[sub_channel]["baseline_pq"].append(float(baseline_pq))
            per_subchannel[sub_channel]["loo_pq"].append(float(loo_pq))
            per_subchannel[sub_channel]["pq_drop"].append(float(baseline_pq - loo_pq))
            used_any = True
        if used_any:
            tiles_used += 1

    out_csv = out_dir / "per_subchannel_layout_summary.csv"
    fieldnames = [
        "sub_channel",
        "group",
        "n",
        "baseline_pq_mean",
        "baseline_pq_sem",
        "loo_pq_mean",
        "loo_pq_sem",
        "pq_drop_mean",
        "pq_drop_sem",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for sub_channel in SUBCHANNELS:
            baseline_vals = per_subchannel[sub_channel]["baseline_pq"]
            loo_vals = per_subchannel[sub_channel]["loo_pq"]
            drop_vals = per_subchannel[sub_channel]["pq_drop"]
            writer.writerow(
                {
                    "sub_channel": sub_channel,
                    "group": SUB_TO_GROUP[sub_channel],
                    "n": len(drop_vals),
                    "baseline_pq_mean": round(mean(baseline_vals), 6) if baseline_vals else "",
                    "baseline_pq_sem": round(_sem(baseline_vals), 6) if baseline_vals else "",
                    "loo_pq_mean": round(mean(loo_vals), 6) if loo_vals else "",
                    "loo_pq_sem": round(_sem(loo_vals), 6) if loo_vals else "",
                    "pq_drop_mean": round(mean(drop_vals), 6) if drop_vals else "",
                    "pq_drop_sem": round(_sem(drop_vals), 6) if drop_vals else "",
                }
            )

    print(f"wrote {out_csv} ({tiles_used} tiles with PQ sidecars)")
    return out_csv


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate subchannel LOO PQ-drop summary")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--orion-root", type=Path, default=DEFAULT_ORION_ROOT)
    args = parser.parse_args()
    aggregate_subchannel_layout_metrics(args.out_dir, orion_root=args.orion_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())