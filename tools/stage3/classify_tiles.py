"""Two-axis tile classifier for channel impact analysis.

Usage:
    python tools/stage3/classify_tiles.py \
        --exp-root data/orion-crc33 \
        --out tile_classes.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

_PNG_CHANNELS = [
    "cell_masks",
    "cell_type_cancer",
    "cell_type_healthy",
    "cell_type_immune",
    "cell_state_prolif",
    "cell_state_nonprolif",
    "cell_state_dead",
]
_NPY_CHANNELS = ["oxygen", "glucose"]
_EPS = 1e-6


def _mean_png(path: Path) -> float:
    arr = np.array(Image.open(path).convert("L"), dtype=np.float32) / 255.0
    return float(arr.mean())


def _mean_npy(path: Path) -> float:
    return float(np.load(path).astype(np.float32).mean())


def compute_tile_stats(tile_id: str, exp_channels_dir: Path) -> dict[str, float]:
    """Compute per-tile summary stats from channel PNGs and NPYs."""
    vals: dict[str, float] = {}
    for ch in _PNG_CHANNELS:
        path = exp_channels_dir / ch / f"{tile_id}.png"
        vals[ch] = _mean_png(path) if path.is_file() else 0.0
    for ch in _NPY_CHANNELS:
        path = exp_channels_dir / ch / f"{tile_id}.npy"
        vals[ch] = _mean_npy(path) if path.is_file() else 0.0

    cell_density = vals["cell_masks"]
    denom = cell_density + _EPS
    return {
        "cell_density": cell_density,
        "cancer_frac": vals["cell_type_cancer"] / denom,
        "immune_frac": vals["cell_type_immune"] / denom,
        "healthy_frac": vals["cell_type_healthy"] / denom,
        "prolif_frac": vals["cell_state_prolif"] / denom,
        "nonprolif_frac": vals["cell_state_nonprolif"] / denom,
        "dead_frac": vals["cell_state_dead"] / denom,
        "mean_oxygen": vals["oxygen"],
        "mean_glucose": vals["glucose"],
    }


def filter_blank_tiles(
    stats_by_tile: dict[str, dict],
    min_density: float,
) -> dict[str, dict]:
    """Drop tiles below the supplied cell-density threshold."""
    # Allow a tiny tolerance so quantized PNG means and percentile rounding do not
    # accidentally drop tiles that are effectively at the cutoff.
    cutoff = max(min_density - 1e-8, 0.0)
    return {tile_id: stats for tile_id, stats in stats_by_tile.items() if stats["cell_density"] >= cutoff}


def assign_axis1(stats: dict, thresholds: dict) -> str | None:
    """Assign a cell-composition label or return None if unlabeled."""
    if stats["cancer_frac"] > thresholds["cancer_frac_p75"]:
        return "cancer"
    if stats["immune_frac"] > thresholds["immune_frac_p75"] and stats["cancer_frac"] > thresholds["cancer_frac_p25"]:
        return "immune"
    if stats["healthy_frac"] > thresholds["healthy_frac_p75"] and stats["cancer_frac"] < thresholds["cancer_frac_p25"]:
        return "healthy"
    return None


def assign_axis2(stats: dict, thresholds: dict) -> str:
    """Assign a metabolic-state label."""
    if stats["mean_oxygen"] < thresholds["oxygen_p25"]:
        return "hypoxic"
    if stats["mean_glucose"] < thresholds["glucose_p25"]:
        return "glucose_low"
    return "neutral"


def compute_percentile_thresholds(stats_by_tile: dict[str, dict]) -> dict[str, float]:
    """Compute P25/P75 thresholds used by the classifier."""
    feature_map = {
        "cancer_frac": "cancer_frac",
        "immune_frac": "immune_frac",
        "healthy_frac": "healthy_frac",
        "mean_oxygen": "oxygen",
        "mean_glucose": "glucose",
    }
    thresholds: dict[str, float] = {}
    for stats_key, threshold_key in feature_map.items():
        arr = np.array([stats[stats_key] for stats in stats_by_tile.values()], dtype=np.float32)
        thresholds[f"{threshold_key}_p25"] = float(np.percentile(arr, 25))
        thresholds[f"{threshold_key}_p75"] = float(np.percentile(arr, 75))
    return thresholds


def _rank_array(vals: list[float]) -> dict[float, float]:
    """Map each value to a rank in [0, 1]."""
    arr = np.array(vals, dtype=np.float32)
    order = arr.argsort()
    ranks = np.empty_like(order, dtype=np.float32)
    ranks[order] = np.arange(len(arr), dtype=np.float32) / max(len(arr) - 1, 1)
    return dict(zip(vals, ranks.tolist()))


def select_representatives(
    classified: dict[str, dict[str, Any]],
    thresholds: dict[str, float],
) -> dict[str, dict]:
    """Pick the highest-purity tile for each axis combination."""
    from itertools import product

    axis1_labels = ("cancer", "immune", "healthy")
    axis2_labels = ("hypoxic", "glucose_low", "neutral")

    def axis1_score(stats: dict, label: str) -> float:
        return {
            "cancer": stats["cancer_frac"],
            "immune": stats["immune_frac"],
            "healthy": stats["healthy_frac"],
        }[label]

    def axis2_score(stats: dict, label: str) -> float:
        return {
            "hypoxic": 1.0 - stats["mean_oxygen"],
            "glucose_low": 1.0 - stats["mean_glucose"],
            "neutral": (stats["mean_oxygen"] + stats["mean_glucose"]) / 2,
        }[label]

    reps: dict[str, dict] = {}
    for a1, a2 in product(axis1_labels, axis2_labels):
        combo_key = f"{a1}+{a2}"
        candidates = {
            tile_id: data
            for tile_id, data in classified.items()
            if data.get("axis1") == a1 and data.get("axis2") == a2
        }
        if not candidates:
            continue
        best_tid = max(
            candidates,
            key=lambda tid: axis1_score(classified[tid], a1) + axis2_score(classified[tid], a2),
        )
        reps[combo_key] = {"tile_id": best_tid, "scores": classified[best_tid]}
    return reps


def select_exp_tiles(
    stats_by_tile: dict[str, dict],
    threshold: float = 0.8,
) -> tuple[dict[str, dict], dict[str, dict]]:
    """Select near-pure tiles for the relabeling experiments."""
    exp2: dict[str, dict] = {}
    for label, frac_key in [
        ("cancer", "cancer_frac"),
        ("immune", "immune_frac"),
        ("healthy", "healthy_frac"),
    ]:
        candidates = {tile_id: stats for tile_id, stats in stats_by_tile.items() if stats[frac_key] >= threshold}
        if candidates:
            best = max(candidates, key=lambda tile_id: candidates[tile_id][frac_key])
            exp2[label] = {"tile_id": best, frac_key: candidates[best][frac_key]}

    exp3: dict[str, dict] = {}
    for label, frac_key in [
        ("prolif", "prolif_frac"),
        ("nonprolif", "nonprolif_frac"),
        ("dead", "dead_frac"),
    ]:
        candidates = {tile_id: stats for tile_id, stats in stats_by_tile.items() if stats[frac_key] >= threshold}
        if candidates:
            best = max(candidates, key=lambda tile_id: candidates[tile_id][frac_key])
            exp3[label] = {"tile_id": best, frac_key: candidates[best][frac_key]}

    return exp2, exp3


def _discover_tile_ids(exp_channels_dir: Path) -> list[str]:
    """Discover tile IDs from the mask channel directory."""
    for mask_name in ("cell_masks", "cell_mask"):
        mask_dir = exp_channels_dir / mask_name
        if mask_dir.is_dir():
            return sorted(p.stem for p in mask_dir.iterdir() if p.is_file() and p.suffix.lower() == ".png")
    raise FileNotFoundError(f"No cell_masks/ or cell_mask/ under {exp_channels_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Two-axis tile classifier for channel impact analysis")
    parser.add_argument("--exp-root", required=True, help="Orion dataset root (contains exp_channels/)")
    parser.add_argument("--out", default="tile_classes.json", help="Output JSON path")
    parser.add_argument(
        "--exp-threshold",
        type=float,
        default=0.8,
        help="Min dominant fraction for Exp 2/3 tile selection (default: 0.8)",
    )
    args = parser.parse_args()

    try:
        from tqdm import tqdm

        progress = tqdm
    except ImportError:
        def progress(it, **_kwargs):  # type: ignore[no-redef]
            return it

    exp_root = Path(args.exp_root)
    exp_channels_dir = exp_root / "exp_channels"
    tile_ids = _discover_tile_ids(exp_channels_dir)
    print(f"Found {len(tile_ids)} tiles")

    raw_stats: dict[str, dict] = {}
    for tile_id in progress(tile_ids, desc="Computing stats"):
        try:
            raw_stats[tile_id] = compute_tile_stats(tile_id, exp_channels_dir)
        except Exception as exc:
            print(f"  Warning: skipping {tile_id}: {exc}")

    if not raw_stats:
        raise RuntimeError(f"No tiles could be processed under {exp_channels_dir}")

    all_densities = np.array([stats["cell_density"] for stats in raw_stats.values()], dtype=np.float32)
    min_density = float(np.percentile(all_densities, 5))
    filtered = filter_blank_tiles(raw_stats, min_density=min_density)
    print(f"After filtering blanks (density < P5={min_density:.4f}): {len(filtered)} tiles")

    thresholds = compute_percentile_thresholds(filtered)
    thresholds["cell_density_p5"] = min_density

    classified: dict[str, dict] = {}
    for tile_id, stats in filtered.items():
        axis1 = assign_axis1(stats, thresholds)
        axis2 = assign_axis2(stats, thresholds)
        classified[tile_id] = {**stats, "axis1": axis1, "axis2": axis2}

    reps = select_representatives(classified, thresholds)
    exp2_tiles, exp3_tiles = select_exp_tiles(filtered, threshold=args.exp_threshold)

    output = {
        "thresholds": {key: round(value, 6) for key, value in thresholds.items()},
        "representatives": reps,
        "exp2_tiles": exp2_tiles,
        "exp3_tiles": exp3_tiles,
        "all_tiles": classified,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Saved tile_classes.json → {out_path}")
    print(f"  Representatives: {len(reps)} combos")
    print(f"  Exp2 tiles: {list(exp2_tiles.keys())}")
    print(f"  Exp3 tiles: {list(exp3_tiles.keys())}")


if __name__ == "__main__":
    main()
