# Channel Impact Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement three scripts that analyze how individual TME channels impact generated H&E appearance: a leave-one-out group diff tool, a tile classifier, and a channel sweep experiment runner.

**Architecture:** Script 1 (`leave_one_out_diff.py`) post-processes existing cached PNGs with no inference. Script 2 (`classify_tiles.py`) scans 10K tiles using only numpy/PIL and emits a `tile_classes.json`. Script 3 (`channel_sweep.py`) loads models once, reads `tile_classes.json` for tile selection, and runs three channel manipulation experiments (microenv 2D grid, cell-type relabeling, cell-state relabeling).

**Tech Stack:** Python 3.10+, numpy, PIL, matplotlib, torch (Scripts 2–3), existing repo helpers from `tools/stage3/tile_pipeline.py`, `tools/stage3/ablation_vis_utils.py`, `tools/stage3/ablation_cache.py`, `tools/channel_group_utils.py`, `train_scripts/inference_controlnet.py`.

---

## Key Constants (from `configs/config_controlnet_exp.py`)

```python
ACTIVE_CHANNELS = [
    "cell_masks",          # idx 0
    "cell_type_healthy",   # idx 1
    "cell_type_cancer",    # idx 2
    "cell_type_immune",    # idx 3
    "cell_state_prolif",   # idx 4
    "cell_state_nonprolif",# idx 5
    "cell_state_dead",     # idx 6
    "vasculature",         # idx 7
    "oxygen",              # idx 8
    "glucose",             # idx 9
]

CHANNEL_GROUPS = [
    {"name": "cell_types",  "channels": ["cell_type_healthy", "cell_type_cancer", "cell_type_immune"]},
    {"name": "cell_state",  "channels": ["cell_state_prolif", "cell_state_nonprolif", "cell_state_dead"]},
    {"name": "vasculature", "channels": ["vasculature"]},
    {"name": "microenv",    "channels": ["oxygen", "glucose"]},
]
```

## File Map

| File | Status | Responsibility |
|---|---|---|
| `tools/vis/leave_one_out_diff.py` | **Create** | LOO diff computation + figure rendering + CLI |
| `tools/stage3/classify_tiles.py` | **Create** | Per-tile stats, two-axis classification, representative selection, JSON output + CLI |
| `tools/stage3/channel_sweep.py` | **Create** | `build_*_ctrl` helpers, `generate_from_ctrl`, Exp 1/2/3 runners, figure rendering, CLI |
| `tests/test_leave_one_out_diff.py` | **Create** | Unit tests for LOO diff core logic |
| `tests/test_classify_tiles.py` | **Create** | Unit tests for stats, classification, selection |
| `tests/test_channel_sweep.py` | **Create** | Unit tests for ctrl manipulation helpers |

---

## Task 1: LOO Diff — Core Logic + Tests

**Files:**
- Create: `tools/vis/leave_one_out_diff.py`
- Create: `tests/test_leave_one_out_diff.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_leave_one_out_diff.py
"""Tests for leave-one-out pixel diff core logic."""
from __future__ import annotations
import json
import numpy as np
import pytest
from pathlib import Path


def _make_cache(tmp_path: Path, group_names=("cell_types", "cell_state", "vasculature", "microenv")):
    """Write a minimal manifest + synthetic PNGs to tmp_path."""
    from PIL import Image

    all_img = np.full((4, 4, 3), 200, dtype=np.uint8)
    Image.fromarray(all_img).save(tmp_path / "generated_he.png")

    (tmp_path / "all").mkdir()
    Image.fromarray(all_img).save(tmp_path / "all" / "generated_he.png")

    (tmp_path / "triples").mkdir()
    entries_triples = []
    for i, omit in enumerate(group_names):
        active = [g for g in group_names if g != omit]
        val = 100 + i * 20
        img = np.full((4, 4, 3), val, dtype=np.uint8)
        fname = f"{i+1:02d}_{'__'.join(active)}.png"
        Image.fromarray(img).save(tmp_path / "triples" / fname)
        entries_triples.append({
            "active_groups": active,
            "condition_label": f"triples_{i}",
            "image_label": f"lbl_{i}",
            "image_path": f"triples/{fname}",
        })

    manifest = {
        "version": 1,
        "tile_id": "test_tile",
        "group_names": list(group_names),
        "sections": [
            {"title": "3 active groups", "subset_size": 3, "entries": entries_triples},
            {"title": "4 active groups", "subset_size": 4, "entries": [{
                "active_groups": list(group_names),
                "condition_label": "all",
                "image_label": "all",
                "image_path": "all/generated_he.png",
            }]},
        ],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return tmp_path


def test_find_loo_entry_returns_correct_triple(tmp_path):
    from tools.vis.leave_one_out_diff import find_loo_entry
    cache = _make_cache(tmp_path)
    manifest = json.loads((cache / "manifest.json").read_text())
    sections = manifest["sections"]

    entry = find_loo_entry(sections, "cell_types")
    assert "cell_types" not in entry["active_groups"]
    assert len(entry["active_groups"]) == 3


def test_compute_loo_diffs_shape_and_nonzero(tmp_path):
    from tools.vis.leave_one_out_diff import compute_loo_diffs

    cache = _make_cache(tmp_path)
    diffs = compute_loo_diffs(cache)

    assert set(diffs.keys()) == {"cell_types", "cell_state", "vasculature", "microenv"}
    for group, diff in diffs.items():
        assert diff.shape == (4, 4), f"bad shape for {group}: {diff.shape}"
        assert diff.dtype == np.float32
        assert diff.min() >= 0.0


def test_compute_loo_diffs_global_normalization(tmp_path):
    from tools.vis.leave_one_out_diff import compute_loo_diffs

    cache = _make_cache(tmp_path)
    diffs = compute_loo_diffs(cache)

    all_vals = np.concatenate([d.ravel() for d in diffs.values()])
    assert all_vals.max() <= 1.0 + 1e-6, "global normalization should map max to 1.0"
    assert all_vals.max() > 0.0, "at least one non-zero diff expected"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/ec2-user/PixCell
python -m pytest tests/test_leave_one_out_diff.py -v 2>&1 | tail -20
```

Expected: `ModuleNotFoundError` or `ImportError` for `tools.vis.leave_one_out_diff`.

- [ ] **Step 3: Implement `find_loo_entry` and `compute_loo_diffs`**

```python
# tools/vis/leave_one_out_diff.py
"""Leave-one-out group pixel diff from cached ablation PNGs.

Usage:
    python tools/vis/leave_one_out_diff.py \\
        --cache-dir inference_output/cache/512_9728 \\
        --orion-root data/orion-crc33 \\
        --out inference_output/cache/512_9728/leave_one_out_diff.png
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from tools.stage3.ablation_vis_utils import FOUR_GROUP_ORDER


def find_loo_entry(sections: list[dict], omit_group: str) -> dict:
    """Return the triples manifest entry whose active_groups excludes `omit_group`."""
    for section in sections:
        if section["subset_size"] != 3:
            continue
        for entry in section["entries"]:
            if omit_group not in entry["active_groups"]:
                return entry
    raise KeyError(f"No triples entry found omitting '{omit_group}'")


def _load_rgb_float32(path: Path) -> np.ndarray:
    """Load PNG as float32 H×W×3 in [0, 255]."""
    return np.array(Image.open(path).convert("RGB"), dtype=np.float32)


def compute_loo_diffs(cache_dir: Path) -> dict[str, np.ndarray]:
    """Compute globally-normalised per-group leave-one-out absolute pixel diffs.

    Returns dict mapping group_name → float32 H×W diff map in [0, 1].
    """
    cache_dir = Path(cache_dir)
    manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    sections = manifest["sections"]
    group_names = tuple(manifest["group_names"])

    # Load all-groups baseline
    all_entry = next(
        e
        for s in sections if s["subset_size"] == len(group_names)
        for e in s["entries"]
    )
    img_all = _load_rgb_float32(cache_dir / all_entry["image_path"])

    # Compute per-group raw diff
    raw_diffs: dict[str, np.ndarray] = {}
    for group in group_names:
        entry = find_loo_entry(sections, group)
        img_loo = _load_rgb_float32(cache_dir / entry["image_path"])
        diff = np.abs(img_all - img_loo).mean(axis=2)  # H×W
        raw_diffs[group] = diff.astype(np.float32)

    # Global normalisation: divide all maps by the single global maximum
    global_max = max(d.max() for d in raw_diffs.values())
    if global_max < 1e-6:
        return {g: np.zeros_like(d) for g, d in raw_diffs.items()}
    return {g: d / global_max for g, d in raw_diffs.items()}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_leave_one_out_diff.py -v 2>&1 | tail -15
```

Expected: 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/vis/leave_one_out_diff.py tests/test_leave_one_out_diff.py
git commit -m "feat: add LOO diff core logic and tests"
```

---

## Task 2: LOO Diff — Figure Rendering + CLI

**Files:**
- Modify: `tools/vis/leave_one_out_diff.py` (add `render_loo_diff_figure`, `save_loo_stats`, `main`)

- [ ] **Step 1: Add figure rendering and CLI to `tools/vis/leave_one_out_diff.py`**

Append after the existing code:

```python
import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

ROOT = Path(__file__).resolve().parent.parent.parent


def render_loo_diff_figure(
    diffs: dict[str, np.ndarray],
    cache_dir: Path,
    *,
    orion_root: Path | None = None,
    out_path: Path,
) -> None:
    """Save a 4-column LOO diff figure: thumbnails / leave-one-out H&E / diff heatmap / stats."""
    manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    sections = manifest["sections"]
    group_names = tuple(manifest["group_names"])
    tile_id = manifest["tile_id"]

    # Load baseline and LOO images for display
    all_entry = next(
        e
        for s in sections if s["subset_size"] == len(group_names)
        for e in s["entries"]
    )
    img_all = np.array(Image.open(cache_dir / all_entry["image_path"]).convert("RGB"))

    hot_cmap = mcolors.LinearSegmentedColormap.from_list("hot4", ["#000000", "#ff4400", "#ffff00", "#ffffff"])
    n_groups = len(group_names)
    n_cols = n_groups + 1  # +1 for baseline column

    fig, axes = plt.subplots(3, n_cols, figsize=(3 * n_cols, 9))
    fig.suptitle(f"Leave-one-out group diff — tile {tile_id}", fontsize=11, y=1.01)

    # Baseline column (index 0)
    axes[0, 0].imshow(img_all)
    axes[0, 0].set_title("All groups\n(baseline)", fontsize=8)
    axes[0, 0].axis("off")
    axes[1, 0].imshow(img_all)
    axes[1, 0].axis("off")
    axes[2, 0].axis("off")

    # Per-group columns
    for col, group in enumerate(group_names, start=1):
        entry = find_loo_entry(sections, group)
        img_loo = np.array(Image.open(cache_dir / entry["image_path"]).convert("RGB"))
        diff = diffs[group]

        # Row 0: channel thumbnail (from orion_root if provided)
        if orion_root is not None:
            try:
                if str(ROOT) not in sys.path:
                    sys.path.insert(0, str(ROOT))
                from tools.stage3.ablation_vis_utils import build_exp_channel_header_rgb
                exp_ch_dir = orion_root / "exp_channels"
                thumbs = build_exp_channel_header_rgb(exp_ch_dir, tile_id, resolution=128)
                axes[0, col].imshow(thumbs.get(group, np.zeros((128, 128, 3), dtype=np.uint8)))
            except Exception:
                axes[0, col].imshow(np.zeros((128, 128, 3), dtype=np.uint8))
        else:
            axes[0, col].imshow(np.zeros((64, 64, 3), dtype=np.uint8))
        axes[0, col].set_title(group.replace("_", "\n"), fontsize=8)
        axes[0, col].axis("off")

        # Row 1: leave-one-out H&E
        axes[1, col].imshow(img_loo)
        mean_diff = float(diff.mean() * 255)
        axes[1, col].set_title(f"w/o {group}\nmean diff {mean_diff:.1f}", fontsize=7)
        axes[1, col].axis("off")

        # Row 2: diff heatmap (globally normalised)
        im = axes[2, col].imshow(diff, cmap=hot_cmap, vmin=0, vmax=1)
        axes[2, col].axis("off")

    axes[2, 0].axis("off")
    plt.colorbar(im, ax=axes[2, :], orientation="horizontal", fraction=0.02, pad=0.04,
                 label="Normalised absolute pixel diff")
    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


def save_loo_stats(diffs: dict[str, np.ndarray], out_path: Path) -> None:
    """Write per-group summary stats to JSON."""
    stats = {}
    for group, diff in diffs.items():
        diff_255 = diff * 255.0
        stats[group] = {
            "mean_diff": round(float(diff_255.mean()), 4),
            "max_diff": round(float(diff_255.max()), 4),
            "pct_pixels_above_10": round(float((diff_255 > 10).mean() * 100), 2),
        }
    out_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Leave-one-out group pixel diff from ablation cache")
    parser.add_argument("--cache-dir", required=True, help="Path to tile cache dir containing manifest.json")
    parser.add_argument("--orion-root", default=None, help="Optional: orion dataset root for channel thumbnails")
    parser.add_argument("--out", default=None, help="Output PNG path (default: <cache-dir>/leave_one_out_diff.png)")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    out_path = Path(args.out) if args.out else cache_dir / "leave_one_out_diff.png"
    orion_root = Path(args.orion_root) if args.orion_root else None

    diffs = compute_loo_diffs(cache_dir)

    stats_path = out_path.with_name("leave_one_out_diff_stats.json")
    save_loo_stats(diffs, stats_path)
    print(f"Stats → {stats_path}")

    render_loo_diff_figure(diffs, cache_dir, orion_root=orion_root, out_path=out_path)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test the script on the existing tile cache**

```bash
cd /home/ec2-user/PixCell
python tools/vis/leave_one_out_diff.py \
    --cache-dir inference_output/cache/512_9728 \
    --orion-root data/orion-crc33 \
    --out inference_output/cache/512_9728/leave_one_out_diff.png
```

Expected: PNG saved, stats JSON written, no errors.

- [ ] **Step 3: Commit**

```bash
git add tools/vis/leave_one_out_diff.py
git commit -m "feat: add LOO diff figure rendering and CLI"
```

---

## Task 3: Tile Classifier — Stats Computation + Tests

**Files:**
- Create: `tools/stage3/classify_tiles.py`
- Create: `tests/test_classify_tiles.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_classify_tiles.py
"""Tests for classify_tiles core logic."""
from __future__ import annotations
import json
import numpy as np
import pytest
from pathlib import Path
from PIL import Image


def _write_png(path: Path, value: float, size: int = 8) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.full((size, size), int(value * 255), dtype=np.uint8)
    Image.fromarray(arr, mode="L").save(path)


def _write_npy(path: Path, value: float, size: int = 8) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.full((size, size), value, dtype=np.float32)
    np.save(path, arr)


def _make_exp_channels(tmp_path: Path) -> Path:
    """Build a minimal exp_channels directory with 3 tiles."""
    ec = tmp_path / "exp_channels"
    tiles = {
        "tile_cancer": {
            "cell_masks": 0.10, "cell_type_cancer": 0.09, "cell_type_healthy": 0.005,
            "cell_type_immune": 0.005, "cell_state_prolif": 0.08, "cell_state_nonprolif": 0.01,
            "cell_state_dead": 0.01,
        },
        "tile_immune": {
            "cell_masks": 0.08, "cell_type_cancer": 0.01, "cell_type_healthy": 0.01,
            "cell_type_immune": 0.06, "cell_state_prolif": 0.02, "cell_state_nonprolif": 0.05,
            "cell_state_dead": 0.01,
        },
        "tile_blank": {
            "cell_masks": 0.0, "cell_type_cancer": 0.0, "cell_type_healthy": 0.0,
            "cell_type_immune": 0.0, "cell_state_prolif": 0.0, "cell_state_nonprolif": 0.0,
            "cell_state_dead": 0.0,
        },
    }
    png_channels = [
        "cell_masks", "cell_type_cancer", "cell_type_healthy", "cell_type_immune",
        "cell_state_prolif", "cell_state_nonprolif", "cell_state_dead",
    ]
    for tile_id, vals in tiles.items():
        for ch in png_channels:
            _write_png(ec / ch / f"{tile_id}.png", vals[ch])
        _write_npy(ec / "oxygen" / f"{tile_id}.npy", 0.9)
        _write_npy(ec / "glucose" / f"{tile_id}.npy", 0.85)
    return ec


def test_compute_tile_stats_cancer_frac(tmp_path):
    from tools.stage3.classify_tiles import compute_tile_stats
    ec = _make_exp_channels(tmp_path)
    stats = compute_tile_stats("tile_cancer", ec)
    assert stats["cell_density"] == pytest.approx(0.10, abs=0.02)
    assert stats["cancer_frac"] == pytest.approx(0.9, abs=0.1)
    assert stats["immune_frac"] < 0.2


def test_compute_tile_stats_blank(tmp_path):
    from tools.stage3.classify_tiles import compute_tile_stats
    ec = _make_exp_channels(tmp_path)
    stats = compute_tile_stats("tile_blank", ec)
    assert stats["cell_density"] == pytest.approx(0.0, abs=1e-6)


def test_filter_blank_tiles(tmp_path):
    from tools.stage3.classify_tiles import compute_tile_stats, filter_blank_tiles
    ec = _make_exp_channels(tmp_path)
    all_stats = {
        tid: compute_tile_stats(tid, ec)
        for tid in ["tile_cancer", "tile_immune", "tile_blank"]
    }
    kept = filter_blank_tiles(all_stats, min_density=0.005)
    assert "tile_blank" not in kept
    assert "tile_cancer" in kept
    assert "tile_immune" in kept


def test_axis1_assignment(tmp_path):
    from tools.stage3.classify_tiles import assign_axis1
    thresholds = {
        "cancer_frac_p75": 0.5, "immune_frac_p75": 0.4, "healthy_frac_p75": 0.6,
        "cancer_frac_p25": 0.1, "immune_frac_p25": 0.05, "healthy_frac_p25": 0.2,
    }
    assert assign_axis1({"cancer_frac": 0.8, "immune_frac": 0.05, "healthy_frac": 0.1}, thresholds) == "cancer"
    assert assign_axis1({"cancer_frac": 0.2, "immune_frac": 0.6, "healthy_frac": 0.1}, thresholds) == "immune"
    assert assign_axis1({"cancer_frac": 0.05, "immune_frac": 0.05, "healthy_frac": 0.8}, thresholds) == "healthy"
    assert assign_axis1({"cancer_frac": 0.3, "immune_frac": 0.3, "healthy_frac": 0.3}, thresholds) is None


def test_axis2_assignment(tmp_path):
    from tools.stage3.classify_tiles import assign_axis2
    thresholds = {"oxygen_p25": 0.5, "glucose_p25": 0.6}
    assert assign_axis2({"mean_oxygen": 0.3, "mean_glucose": 0.8}, thresholds) == "hypoxic"
    assert assign_axis2({"mean_oxygen": 0.8, "mean_glucose": 0.4}, thresholds) == "glucose_low"
    assert assign_axis2({"mean_oxygen": 0.8, "mean_glucose": 0.8}, thresholds) == "neutral"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_classify_tiles.py -v 2>&1 | tail -15
```

Expected: ImportError for `tools.stage3.classify_tiles`.

- [ ] **Step 3: Implement `compute_tile_stats`, `filter_blank_tiles`, `assign_axis1`, `assign_axis2`**

```python
# tools/stage3/classify_tiles.py
"""Two-axis tile classifier for the TME channel impact analysis.

Usage:
    python tools/stage3/classify_tiles.py \\
        --exp-root data/orion-crc33 \\
        --out tile_classes.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

_PNG_CHANNELS = [
    "cell_masks", "cell_type_cancer", "cell_type_healthy", "cell_type_immune",
    "cell_state_prolif", "cell_state_nonprolif", "cell_state_dead",
]
_NPY_CHANNELS = ["oxygen", "glucose"]
_EPS = 1e-6


def _mean_png(path: Path) -> float:
    arr = np.array(Image.open(path).convert("L"), dtype=np.float32) / 255.0
    return float(arr.mean())


def _mean_npy(path: Path) -> float:
    return float(np.load(path).astype(np.float32).mean())


def compute_tile_stats(tile_id: str, exp_channels_dir: Path) -> dict[str, float]:
    """Return per-tile channel statistics dict."""
    vals: dict[str, float] = {}
    for ch in _PNG_CHANNELS:
        p = exp_channels_dir / ch / f"{tile_id}.png"
        vals[ch] = _mean_png(p) if p.is_file() else 0.0
    for ch in _NPY_CHANNELS:
        p = exp_channels_dir / ch / f"{tile_id}.npy"
        vals[ch] = _mean_npy(p) if p.is_file() else 0.0

    cell_density = vals["cell_masks"]
    denom = cell_density + _EPS
    return {
        "cell_density": cell_density,
        "cancer_frac":    vals["cell_type_cancer"]    / denom,
        "immune_frac":    vals["cell_type_immune"]     / denom,
        "healthy_frac":   vals["cell_type_healthy"]    / denom,
        "prolif_frac":    vals["cell_state_prolif"]    / denom,
        "nonprolif_frac": vals["cell_state_nonprolif"] / denom,
        "dead_frac":      vals["cell_state_dead"]      / denom,
        "mean_oxygen":    vals["oxygen"],
        "mean_glucose":   vals["glucose"],
    }


def filter_blank_tiles(
    stats_by_tile: dict[str, dict],
    min_density: float,
) -> dict[str, dict]:
    """Remove tiles with cell_density below min_density."""
    return {tid: s for tid, s in stats_by_tile.items() if s["cell_density"] >= min_density}


def assign_axis1(stats: dict, thresholds: dict) -> str | None:
    """Cancer / immune / healthy / None (unlabeled)."""
    if stats["cancer_frac"] > thresholds["cancer_frac_p75"]:
        return "cancer"
    if (stats["immune_frac"] > thresholds["immune_frac_p75"]
            and stats["cancer_frac"] > thresholds["cancer_frac_p25"]):
        return "immune"
    if (stats["healthy_frac"] > thresholds["healthy_frac_p75"]
            and stats["cancer_frac"] < thresholds["cancer_frac_p25"]):
        return "healthy"
    return None


def assign_axis2(stats: dict, thresholds: dict) -> str:
    """Hypoxic / glucose_low / neutral."""
    if stats["mean_oxygen"] < thresholds["oxygen_p25"]:
        return "hypoxic"
    if stats["mean_glucose"] < thresholds["glucose_p25"]:
        return "glucose_low"
    return "neutral"
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_classify_tiles.py -v 2>&1 | tail -15
```

Expected: 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/stage3/classify_tiles.py tests/test_classify_tiles.py
git commit -m "feat: add tile classifier stats and axis assignment"
```

---

## Task 4: Tile Classifier — Percentiles, Representative Selection + CLI

**Files:**
- Modify: `tools/stage3/classify_tiles.py` (add `compute_percentile_thresholds`, `select_representatives`, `select_exp_tiles`, `main`)

- [ ] **Step 1: Append the following to `tools/stage3/classify_tiles.py`**

```python
from typing import Any


def compute_percentile_thresholds(stats_by_tile: dict[str, dict]) -> dict[str, float]:
    """Compute P25/P75 thresholds for each classification feature."""
    vals: dict[str, list] = {
        k: [] for k in [
            "cancer_frac", "immune_frac", "healthy_frac",
            "mean_oxygen", "mean_glucose",
        ]
    }
    for s in stats_by_tile.values():
        for k in vals:
            vals[k].append(s[k])

    thresholds: dict[str, float] = {}
    for k, v in vals.items():
        arr = np.array(v, dtype=np.float32)
        thresholds[f"{k}_p25"] = float(np.percentile(arr, 25))
        thresholds[f"{k}_p75"] = float(np.percentile(arr, 75))
    return thresholds


def _rank_array(vals: list[float]) -> dict[float, float]:
    """Map each value to its rank in [0, 1]."""
    arr = np.array(vals, dtype=np.float32)
    order = arr.argsort()
    ranks = np.empty_like(order, dtype=np.float32)
    ranks[order] = np.arange(len(arr)) / max(len(arr) - 1, 1)
    return dict(zip(vals, ranks.tolist()))


def select_representatives(
    classified: dict[str, dict[str, Any]],
    thresholds: dict[str, float],
) -> dict[str, dict]:
    """For each of the 9 (axis1, axis2) combinations, pick the highest-purity tile.

    Purity score = sum of axis1 feature rank + axis2 feature rank.
    """
    from itertools import product

    axis1_labels = ("cancer", "immune", "healthy")
    axis2_labels = ("hypoxic", "glucose_low", "neutral")

    # Build per-tile axis1 and axis2 feature scores
    def axis1_score(stats: dict, label: str) -> float:
        return {
            "cancer":  stats["cancer_frac"],
            "immune":  stats["immune_frac"],
            "healthy": stats["healthy_frac"],
        }[label]

    def axis2_score(stats: dict, label: str) -> float:
        return {
            "hypoxic":     1.0 - stats["mean_oxygen"],
            "glucose_low": 1.0 - stats["mean_glucose"],
            "neutral":     (stats["mean_oxygen"] + stats["mean_glucose"]) / 2,
        }[label]

    reps: dict[str, dict] = {}
    for a1, a2 in product(axis1_labels, axis2_labels):
        combo_key = f"{a1}+{a2}"
        candidates = {
            tid: data for tid, data in classified.items()
            if data.get("axis1") == a1 and data.get("axis2") == a2
        }
        if not candidates:
            continue
        best_tid = max(
            candidates,
            key=lambda tid: (
                axis1_score(classified[tid], a1) + axis2_score(classified[tid], a2)
            ),
        )
        reps[combo_key] = {"tile_id": best_tid, "scores": classified[best_tid]}
    return reps


def select_exp_tiles(
    stats_by_tile: dict[str, dict],
    threshold: float = 0.8,
) -> tuple[dict[str, dict], dict[str, dict]]:
    """Select best near-pure tiles for Experiments 2 (cell types) and 3 (cell states).

    Returns (exp2_tiles, exp3_tiles) where each maps label → {tile_id, score}.
    """
    exp2: dict[str, dict] = {}
    for label, frac_key in [("cancer", "cancer_frac"), ("immune", "immune_frac"), ("healthy", "healthy_frac")]:
        candidates = {tid: s for tid, s in stats_by_tile.items() if s[frac_key] >= threshold}
        if candidates:
            best = max(candidates, key=lambda tid: candidates[tid][frac_key])
            exp2[label] = {"tile_id": best, frac_key: candidates[best][frac_key]}

    exp3: dict[str, dict] = {}
    for label, frac_key in [("prolif", "prolif_frac"), ("nonprolif", "nonprolif_frac"), ("dead", "dead_frac")]:
        candidates = {tid: s for tid, s in stats_by_tile.items() if s[frac_key] >= threshold}
        if candidates:
            best = max(candidates, key=lambda tid: candidates[tid][frac_key])
            exp3[label] = {"tile_id": best, frac_key: candidates[best][frac_key]}

    return exp2, exp3


def main() -> None:
    parser = argparse.ArgumentParser(description="Two-axis tile classifier for channel impact analysis")
    parser.add_argument("--exp-root", required=True, help="Orion dataset root (contains exp_channels/)")
    parser.add_argument("--out", default="tile_classes.json", help="Output JSON path")
    parser.add_argument("--exp-threshold", type=float, default=0.8,
                        help="Min dominant fraction for Exp 2/3 tile selection (default: 0.8)")
    args = parser.parse_args()

    try:
        from tqdm import tqdm
        _tqdm = tqdm
    except ImportError:
        def _tqdm(it, **kw):
            return it

    exp_root = Path(args.exp_root)
    exp_channels_dir = exp_root / "exp_channels"
    mask_dir = exp_channels_dir / "cell_masks"
    if not mask_dir.is_dir():
        # Try alias
        mask_dir = exp_channels_dir / "cell_mask"
    if not mask_dir.is_dir():
        raise FileNotFoundError(f"No cell_masks/ or cell_mask/ under {exp_channels_dir}")

    tile_ids = sorted(p.stem for p in mask_dir.iterdir() if p.suffix == ".png")
    print(f"Found {len(tile_ids)} tiles")

    # Compute stats
    raw_stats: dict[str, dict] = {}
    for tid in _tqdm(tile_ids, desc="Computing stats"):
        try:
            raw_stats[tid] = compute_tile_stats(tid, exp_channels_dir)
        except Exception as exc:
            print(f"  Warning: skipping {tid}: {exc}")

    # Filter blanks
    all_densities = np.array([s["cell_density"] for s in raw_stats.values()], dtype=np.float32)
    min_density = float(np.percentile(all_densities, 5))
    filtered = filter_blank_tiles(raw_stats, min_density=min_density)
    print(f"After filtering blanks (density < P5={min_density:.4f}): {len(filtered)} tiles")

    # Percentile thresholds
    thresholds = compute_percentile_thresholds(filtered)
    thresholds["cell_density_p5"] = min_density

    # Classify
    classified: dict[str, dict] = {}
    for tid, stats in filtered.items():
        a1 = assign_axis1(stats, thresholds)
        a2 = assign_axis2(stats, thresholds)
        classified[tid] = {**stats, "axis1": a1, "axis2": a2}

    # Select representatives
    reps = select_representatives(classified, thresholds)
    exp2_tiles, exp3_tiles = select_exp_tiles(filtered, threshold=args.exp_threshold)

    output = {
        "thresholds": {k: round(v, 6) for k, v in thresholds.items()},
        "representatives": reps,
        "exp2_tiles": exp2_tiles,
        "exp3_tiles": exp3_tiles,
        "all_tiles": classified,
    }
    out_path = Path(args.out)
    out_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Saved tile_classes.json → {out_path}")
    print(f"  Representatives: {len(reps)} combos")
    print(f"  Exp2 tiles: {list(exp2_tiles.keys())}")
    print(f"  Exp3 tiles: {list(exp3_tiles.keys())}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the classifier on the full dataset (takes a few minutes)**

```bash
cd /home/ec2-user/PixCell
python tools/stage3/classify_tiles.py \
    --exp-root data/orion-crc33 \
    --out tile_classes.json
```

Expected output (example):
```
Found 10379 tiles
After filtering blanks (density < P5=...): ~9860 tiles
Saved tile_classes.json → tile_classes.json
  Representatives: N combos
  Exp2 tiles: ['cancer', 'immune', 'healthy']
  Exp3 tiles: ['prolif', 'nonprolif', 'dead']
```

If any Exp2/Exp3 class is missing, lower `--exp-threshold` (try 0.7 or 0.6).

- [ ] **Step 3: Commit**

```bash
git add tools/stage3/classify_tiles.py
git commit -m "feat: add tile classifier percentile thresholds, representative selection, and CLI"
```

---

## Task 5: Channel Sweep — Core Ctrl Manipulation Helpers + Tests

**Files:**
- Create: `tools/stage3/channel_sweep.py`
- Create: `tests/test_channel_sweep.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_channel_sweep.py
"""Tests for channel_sweep ctrl manipulation helpers."""
from __future__ import annotations
import numpy as np
import pytest
import torch


def test_build_scaled_ctrl_scales_target_channel():
    from tools.stage3.channel_sweep import build_scaled_ctrl
    ctrl = torch.ones(10, 4, 4)  # 10 channels
    result = build_scaled_ctrl(ctrl, channel_idx=2, scale=0.5)
    assert torch.allclose(result[2], ctrl[2] * 0.5)
    # Other channels unchanged
    assert torch.allclose(result[0], ctrl[0])
    assert torch.allclose(result[9], ctrl[9])
    # Original not mutated
    assert torch.allclose(ctrl[2], torch.ones(4, 4))


def test_build_2d_scaled_ctrl():
    from tools.stage3.channel_sweep import build_2d_scaled_ctrl
    ctrl = torch.ones(10, 4, 4)
    result = build_2d_scaled_ctrl(ctrl, idx_o2=8, idx_glucose=9, o2_scale=0.0, glucose_scale=0.5)
    assert torch.allclose(result[8], torch.zeros(4, 4))
    assert torch.allclose(result[9], ctrl[9] * 0.5)
    assert torch.allclose(result[0], ctrl[0])


def test_build_relabeled_ctrl_copies_and_zeros():
    from tools.stage3.channel_sweep import build_relabeled_ctrl
    ctrl = torch.zeros(10, 4, 4)
    ctrl[2] = 0.8  # cancer channel has content
    result = build_relabeled_ctrl(ctrl, idx_source=2, idx_target=3)
    # Source zeroed
    assert torch.allclose(result[2], torch.zeros(4, 4))
    # Target gets source content
    assert torch.allclose(result[3], ctrl[2])
    # Original not mutated
    assert float(ctrl[2].mean()) == pytest.approx(0.8)


def test_build_relabeled_ctrl_does_not_affect_other_channels():
    from tools.stage3.channel_sweep import build_relabeled_ctrl
    ctrl = torch.rand(10, 4, 4)
    ctrl[2] = 1.0
    result = build_relabeled_ctrl(ctrl, idx_source=2, idx_target=3)
    for ch in range(10):
        if ch in (2, 3):
            continue
        assert torch.allclose(result[ch], ctrl[ch]), f"channel {ch} should be unchanged"


def test_sweep_scales_list():
    from tools.stage3.channel_sweep import SWEEP_SCALES
    assert SWEEP_SCALES == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert len(SWEEP_SCALES) == 5
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_channel_sweep.py -v 2>&1 | tail -10
```

Expected: ImportError for `tools.stage3.channel_sweep`.

- [ ] **Step 3: Implement ctrl manipulation helpers**

```python
# tools/stage3/channel_sweep.py
"""Channel sweep experiments for TME channel impact analysis.

Three experiments driven by tile_classes.json:
  Exp 1: 5×5 O2×glucose scale grid on 3 cell-composition-representative tiles
  Exp 2: cell-type relabeling on 3 near-pure cell-type tiles
  Exp 3: cell-state relabeling on 3 near-pure cell-state tiles

Usage:
    python tools/stage3/channel_sweep.py \\
        --class-json tile_classes.json \\
        --data-root data/orion-crc33 \\
        --checkpoint-dir checkpoints/pixcell_controlnet_exp/checkpoints \\
        --out inference_output/channel_sweep/ \\
        --seed 42
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

import argparse

SWEEP_SCALES: list[float] = [0.0, 0.25, 0.5, 0.75, 1.0]

# Channel indices in ACTIVE_CHANNELS order (from configs/config_controlnet_exp.py)
_CH_IDX = {
    "cell_masks":          0,
    "cell_type_healthy":   1,
    "cell_type_cancer":    2,
    "cell_type_immune":    3,
    "cell_state_prolif":   4,
    "cell_state_nonprolif":5,
    "cell_state_dead":     6,
    "vasculature":         7,
    "oxygen":              8,
    "glucose":             9,
}


def build_scaled_ctrl(ctrl_full: torch.Tensor, channel_idx: int, scale: float) -> torch.Tensor:
    """Return a clone of ctrl_full with one channel multiplied by `scale`."""
    ctrl = ctrl_full.clone()
    ctrl[channel_idx] = ctrl_full[channel_idx] * scale
    return ctrl


def build_2d_scaled_ctrl(
    ctrl_full: torch.Tensor,
    *,
    idx_o2: int,
    idx_glucose: int,
    o2_scale: float,
    glucose_scale: float,
) -> torch.Tensor:
    """Return ctrl with oxygen scaled by `o2_scale` and glucose by `glucose_scale`."""
    ctrl = ctrl_full.clone()
    ctrl[idx_o2]      = ctrl_full[idx_o2]      * o2_scale
    ctrl[idx_glucose] = ctrl_full[idx_glucose]  * glucose_scale
    return ctrl


def build_relabeled_ctrl(
    ctrl_full: torch.Tensor,
    *,
    idx_source: int,
    idx_target: int,
) -> torch.Tensor:
    """Copy source channel to target and zero out source. All other channels unchanged."""
    ctrl = ctrl_full.clone()
    ctrl[idx_target] = ctrl_full[idx_source].clone()
    ctrl[idx_source] = torch.zeros_like(ctrl_full[idx_source])
    return ctrl
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_channel_sweep.py -v 2>&1 | tail -10
```

Expected: 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/stage3/channel_sweep.py tests/test_channel_sweep.py
git commit -m "feat: add channel sweep ctrl manipulation helpers and tests"
```

---

## Task 6: Channel Sweep — `generate_from_ctrl` + Model Loading Helpers

**Files:**
- Modify: `tools/stage3/channel_sweep.py` (add `generate_from_ctrl`, `load_sweep_models`, `_get_dtype`)

- [ ] **Step 1: Append to `tools/stage3/channel_sweep.py`**

```python
import os
import sys

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _get_dtype(device: str) -> torch.dtype:
    return torch.float16 if device == "cuda" else torch.float32


def load_sweep_models(config_path: str, ckpt_dir: Path, device: str) -> tuple[dict, Any, Any]:
    """Load all models + config + scheduler for sweep runs.

    Returns (models, config, scheduler).
    """
    from diffusers import DDPMScheduler
    from diffusion.utils.misc import read_config
    from tools.stage3.tile_pipeline import find_latest_checkpoint_dir, load_all_models

    os.chdir(ROOT)
    config = read_config(config_path)
    config._filename = config_path

    ckpt = find_latest_checkpoint_dir(ckpt_dir)
    models = load_all_models(config, config_path, ckpt, device)

    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
        prediction_type="epsilon",
        clip_sample=False,
    )
    scheduler.set_timesteps(20, device=device)
    return models, config, scheduler


def generate_from_ctrl(
    ctrl_full: torch.Tensor,
    *,
    models: dict,
    config: Any,
    scheduler: Any,
    uni_embeds: torch.Tensor,
    device: str,
    guidance_scale: float,
    fixed_noise: torch.Tensor,
    seed: int,
) -> np.ndarray:
    """Run TME → ControlNet → VAE decode for a single modified ctrl_full tensor.

    Args:
        ctrl_full: [C, H, W] float CPU tensor with modified channels.
        fixed_noise: pre-generated latent noise (same across all conditions for fair comparison).
        seed: manual seed applied before denoising (scheduler step noise).

    Returns:
        uint8 RGB numpy array [H, W, 3].
    """
    from tools.channel_group_utils import split_channels_to_groups
    from train_scripts.inference_controlnet import encode_ctrl_mask_latent, denoise

    dtype = _get_dtype(device)
    vae = models["vae"]
    vae.to(device=device, dtype=dtype).eval()

    vae_mask = encode_ctrl_mask_latent(
        ctrl_full,
        vae,
        vae_shift=config.shift_factor,
        vae_scale=config.scale_factor,
        device=device,
        dtype=dtype,
    )

    tme_dict = split_channels_to_groups(
        ctrl_full.unsqueeze(0).to(device, dtype=dtype),
        config.data.active_channels,
        config.channel_groups,
    )

    all_groups = {g["name"] for g in config.channel_groups}
    with torch.no_grad():
        fused = models["tme_module"](vae_mask, tme_dict, active_groups=all_groups)
        if getattr(config, "zero_mask_latent", False):
            fused = fused - vae_mask

    torch.manual_seed(seed)
    denoised = denoise(
        latents=fixed_noise.clone(),
        uni_embeds=uni_embeds.to(device, dtype=dtype),
        controlnet_input_latent=fused,
        scheduler=scheduler,
        controlnet_model=models["controlnet"],
        pixcell_controlnet_model=models["base_model"],
        guidance_scale=guidance_scale,
        device=device,
    )

    with torch.no_grad():
        scaled = (denoised.to(dtype) / config.scale_factor) + config.shift_factor
        gen = vae.decode(scaled, return_dict=False)[0]
    gen = (gen / 2 + 0.5).clamp(0, 1)
    return (gen.cpu().permute(0, 2, 3, 1).numpy()[0] * 255).astype(np.uint8)
```

- [ ] **Step 2: Verify syntax**

```bash
python -c "from tools.stage3.channel_sweep import generate_from_ctrl; print('ok')"
```

Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add tools/stage3/channel_sweep.py
git commit -m "feat: add generate_from_ctrl and model loading helpers"
```

---

## Task 7: Experiment 1 — Microenv 2D Grid + Figure

**Files:**
- Modify: `tools/stage3/channel_sweep.py` (add `run_exp1_microenv_grid`, `render_exp1_figure`)

- [ ] **Step 1: Append to `tools/stage3/channel_sweep.py`**

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def run_exp1_microenv_grid(
    tile_id: str,
    *,
    exp_channels_dir: Path,
    models: dict,
    config: Any,
    scheduler: Any,
    uni_embeds: torch.Tensor,
    device: str,
    guidance_scale: float,
    seed: int,
) -> dict[tuple[float, float], np.ndarray]:
    """Generate a 5×5 grid of (o2_scale, glucose_scale) → uint8 H&E image.

    Returns dict keyed by (o2_scale, glucose_scale).
    """
    from tools.stage3.tile_pipeline import load_exp_channels, _make_fixed_noise

    dtype = _get_dtype(device)
    ctrl_full = load_exp_channels(tile_id, config.data.active_channels, config.image_size, exp_channels_dir)
    fixed_noise = _make_fixed_noise(config=config, scheduler=scheduler, device=device, dtype=dtype, seed=seed)

    idx_o2      = _CH_IDX["oxygen"]
    idx_glucose = _CH_IDX["glucose"]

    results: dict[tuple[float, float], np.ndarray] = {}
    total = len(SWEEP_SCALES) ** 2
    for i, o2_s in enumerate(SWEEP_SCALES):
        for j, gl_s in enumerate(SWEEP_SCALES):
            step = i * len(SWEEP_SCALES) + j + 1
            print(f"  Exp1 [{step}/{total}] O2={o2_s:.2f} glucose={gl_s:.2f}")
            ctrl = build_2d_scaled_ctrl(ctrl_full, idx_o2=idx_o2, idx_glucose=idx_glucose,
                                        o2_scale=o2_s, glucose_scale=gl_s)
            results[(o2_s, gl_s)] = generate_from_ctrl(
                ctrl, models=models, config=config, scheduler=scheduler,
                uni_embeds=uni_embeds, device=device, guidance_scale=guidance_scale,
                fixed_noise=fixed_noise, seed=seed,
            )
    return results


def render_exp1_figure(
    images_grid: dict[tuple[float, float], np.ndarray],
    tile_id: str,
    tile_class_label: str,
    out_path: Path,
) -> None:
    """Save the 5×5 O2×glucose sweep figure with diff insets and L1 heatmap."""
    n = len(SWEEP_SCALES)
    baseline = images_grid[(1.0, 1.0)].astype(np.float32)

    # Main 5×5 grid (H&E images) + 1 row for L1 heatmap
    fig = plt.figure(figsize=(n * 2.5 + 1, n * 2.5 + 2.5))
    fig.suptitle(f"Microenv sweep — {tile_id} ({tile_class_label})", fontsize=11)

    gs = fig.add_gridspec(n + 1, n + 1, hspace=0.05, wspace=0.05,
                          height_ratios=[1] * n + [0.6],
                          width_ratios=[0.15] + [1] * n)

    # Row labels (O2 scale)
    ax_label = fig.add_subplot(gs[:n, 0])
    ax_label.axis("off")
    for i, s in enumerate(SWEEP_SCALES):
        ax_label.text(0.9, 1 - (i + 0.5) / n, f"O₂={s:.2f}",
                      ha="right", va="center", fontsize=7, transform=ax_label.transAxes)

    # H&E grid
    hot_cmap = mcolors.LinearSegmentedColormap.from_list("hot4", ["#000000", "#ff4400", "#ffff00", "#ffffff"])
    l1_grid = np.zeros((n, n), dtype=np.float32)

    for i, o2_s in enumerate(SWEEP_SCALES):
        for j, gl_s in enumerate(SWEEP_SCALES):
            ax = fig.add_subplot(gs[i, j + 1])
            img = images_grid[(o2_s, gl_s)]
            ax.imshow(img)
            ax.axis("off")
            diff = np.abs(img.astype(np.float32) - baseline).mean(axis=2)
            l1_val = float(diff.mean())
            l1_grid[i, j] = l1_val
            if o2_s == 1.0 and gl_s == 1.0:
                for spine in ax.spines.values():
                    spine.set_edgecolor("#00cc44")
                    spine.set_linewidth(2)
                    spine.set_visible(True)
            # Small diff inset (bottom-left 25% of cell)
            ax_inset = ax.inset_axes([0, 0, 0.3, 0.3])
            ax_inset.imshow(diff, cmap=hot_cmap, vmin=0, vmax=50)
            ax_inset.axis("off")
            if i == n - 1:
                ax.set_xlabel(f"Gluc={gl_s:.2f}", fontsize=6)

    # L1 heatmap strip
    ax_l1 = fig.add_subplot(gs[n, 1:])
    im = ax_l1.imshow(l1_grid, cmap="viridis", aspect="auto", vmin=0)
    ax_l1.set_xticks(range(n))
    ax_l1.set_xticklabels([f"{s:.2f}" for s in SWEEP_SCALES], fontsize=6)
    ax_l1.set_yticks(range(n))
    ax_l1.set_yticklabels([f"{s:.2f}" for s in SWEEP_SCALES], fontsize=6)
    ax_l1.set_xlabel("Glucose scale", fontsize=7)
    ax_l1.set_ylabel("O₂ scale", fontsize=7)
    plt.colorbar(im, ax=ax_l1, orientation="vertical", label="Mean L1 diff vs baseline")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")
```

- [ ] **Step 2: Syntax check**

```bash
python -c "from tools.stage3.channel_sweep import run_exp1_microenv_grid, render_exp1_figure; print('ok')"
```

Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add tools/stage3/channel_sweep.py
git commit -m "feat: add Exp1 microenv 2D grid runner and figure renderer"
```

---

## Task 8: Experiments 2 & 3 — Relabeling Runners + Figures

**Files:**
- Modify: `tools/stage3/channel_sweep.py` (add `run_exp2_cell_type_relabeling`, `run_exp3_cell_state_relabeling`, `render_relabeling_figure`)

- [ ] **Step 1: Append to `tools/stage3/channel_sweep.py`**

```python
# Cell type channel indices
_CELL_TYPE_CHANNELS = {
    "cancer":  _CH_IDX["cell_type_cancer"],
    "immune":  _CH_IDX["cell_type_immune"],
    "healthy": _CH_IDX["cell_type_healthy"],
}

# Cell state channel indices
_CELL_STATE_CHANNELS = {
    "prolif":    _CH_IDX["cell_state_prolif"],
    "nonprolif": _CH_IDX["cell_state_nonprolif"],
    "dead":      _CH_IDX["cell_state_dead"],
}


def _run_relabeling_experiment(
    tiles: dict[str, str],
    channel_map: dict[str, int],
    *,
    exp_channels_dir: Path,
    models: dict,
    config: Any,
    scheduler: Any,
    uni_embeds: torch.Tensor,
    device: str,
    guidance_scale: float,
    seed: int,
) -> dict[str, dict[str, np.ndarray]]:
    """Generic relabeling runner for Exp 2 and 3.

    Args:
        tiles: {label → tile_id}, e.g. {"cancer": "10240_11520", ...}
        channel_map: {label → ctrl_full channel index}

    Returns:
        Nested dict: {source_label → {target_label → generated_image}}
        source_label == target_label means the original (no relabeling).
    """
    from tools.stage3.tile_pipeline import load_exp_channels, _make_fixed_noise

    dtype = _get_dtype(device)
    labels = list(channel_map.keys())
    results: dict[str, dict[str, np.ndarray]] = {}

    for src_label, tile_id in tiles.items():
        print(f"  Processing tile {tile_id} (source: {src_label})")
        ctrl_full = load_exp_channels(tile_id, config.data.active_channels, config.image_size, exp_channels_dir)
        fixed_noise = _make_fixed_noise(config=config, scheduler=scheduler, device=device, dtype=dtype, seed=seed)
        results[src_label] = {}

        for tgt_label in labels:
            if src_label == tgt_label:
                # Baseline: no modification
                img = generate_from_ctrl(
                    ctrl_full, models=models, config=config, scheduler=scheduler,
                    uni_embeds=uni_embeds, device=device, guidance_scale=guidance_scale,
                    fixed_noise=fixed_noise, seed=seed,
                )
            else:
                ctrl = build_relabeled_ctrl(
                    ctrl_full,
                    idx_source=channel_map[src_label],
                    idx_target=channel_map[tgt_label],
                )
                img = generate_from_ctrl(
                    ctrl, models=models, config=config, scheduler=scheduler,
                    uni_embeds=uni_embeds, device=device, guidance_scale=guidance_scale,
                    fixed_noise=fixed_noise, seed=seed,
                )
            results[src_label][tgt_label] = img
            print(f"    {src_label} → {tgt_label}: done")

    return results


def run_exp2_cell_type_relabeling(
    tiles: dict[str, str],
    **kwargs,
) -> dict[str, dict[str, np.ndarray]]:
    """Exp 2: relabel cell type channels. `tiles` = {"cancer": tile_id, "immune": ..., "healthy": ...}"""
    return _run_relabeling_experiment(tiles, _CELL_TYPE_CHANNELS, **kwargs)


def run_exp3_cell_state_relabeling(
    tiles: dict[str, str],
    **kwargs,
) -> dict[str, dict[str, np.ndarray]]:
    """Exp 3: relabel cell state channels. `tiles` = {"prolif": tile_id, ...}"""
    return _run_relabeling_experiment(tiles, _CELL_STATE_CHANNELS, **kwargs)


def render_relabeling_figure(
    results: dict[str, dict[str, np.ndarray]],
    tiles: dict[str, str],
    exp_title: str,
    out_path: Path,
) -> None:
    """Save a (n_src × n_tgt) relabeling figure with diff panels.

    Diagonal = original (no diff panel). Off-diagonal = relabeled + diff.
    """
    labels = list(results.keys())
    n = len(labels)
    hot_cmap = mcolors.LinearSegmentedColormap.from_list("hot4", ["#000000", "#ff4400", "#ffff00", "#ffffff"])

    # Each cell = H&E + optional diff inset → 2 columns per target, 1 row per source
    fig, axes = plt.subplots(n, n, figsize=(n * 3, n * 3))
    fig.suptitle(exp_title, fontsize=11)

    for i, src in enumerate(labels):
        for j, tgt in enumerate(labels):
            ax = axes[i, j]
            img = results[src][tgt]
            ax.imshow(img)
            ax.axis("off")

            if i == 0:
                ax.set_title(f"→ {tgt}", fontsize=8)
            if j == 0:
                ax.set_ylabel(f"{src}\n({tiles[src]})", fontsize=7)

            if src != tgt:
                baseline = results[src][src].astype(np.float32)
                diff = np.abs(img.astype(np.float32) - baseline).mean(axis=2)
                ax_inset = ax.inset_axes([0.0, 0.0, 0.35, 0.35])
                ax_inset.imshow(diff, cmap=hot_cmap, vmin=0, vmax=50)
                ax_inset.axis("off")
            else:
                # Diagonal highlight
                for spine in ax.spines.values():
                    spine.set_edgecolor("#00cc44")
                    spine.set_linewidth(2)
                    spine.set_visible(True)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")
```

- [ ] **Step 2: Syntax check**

```bash
python -c "from tools.stage3.channel_sweep import run_exp2_cell_type_relabeling, run_exp3_cell_state_relabeling, render_relabeling_figure; print('ok')"
```

Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add tools/stage3/channel_sweep.py
git commit -m "feat: add Exp2/Exp3 relabeling runners and figure renderer"
```

---

## Task 9: Channel Sweep — CLI Orchestration

**Files:**
- Modify: `tools/stage3/channel_sweep.py` (add `main`)

- [ ] **Step 1: Append `main()` to `tools/stage3/channel_sweep.py`**

```python
def main() -> None:
    parser = argparse.ArgumentParser(description="Channel sweep experiments (Exp 1/2/3)")
    parser.add_argument("--class-json", required=True, help="tile_classes.json from classify_tiles.py")
    parser.add_argument("--data-root", required=True, help="Orion dataset root (contains exp_channels/)")
    parser.add_argument("--checkpoint-dir", default=None,
                        help="Parent dir of controlnet_*.pth checkpoints (default: checkpoints/pixcell_controlnet_exp/checkpoints)")
    parser.add_argument("--config", default=str(ROOT / "configs/config_controlnet_exp.py"))
    parser.add_argument("--out", required=True, help="Output directory for figures")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--guidance-scale", type=float, default=2.5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--experiments", nargs="+", choices=["1", "2", "3"], default=["1", "2", "3"],
        help="Which experiments to run (default: all three)",
    )
    parser.add_argument(
        "--null-uni", action="store_true",
        help="Use null UNI embedding (TME-only inference, no style conditioning)",
    )
    args = parser.parse_args()

    from tools.stage3.tile_pipeline import find_latest_checkpoint_dir, resolve_data_layout
    from train_scripts.inference_controlnet import null_uni_embed

    data_root = Path(args.data_root)
    exp_channels_dir, feat_dir, _ = resolve_data_layout(data_root)

    ckpt_parent = Path(args.checkpoint_dir) if args.checkpoint_dir else ROOT / "checkpoints/pixcell_controlnet_exp/checkpoints"
    models, config, scheduler = load_sweep_models(args.config, ckpt_parent, args.device)

    class_data = json.loads(Path(args.class_json).read_text(encoding="utf-8"))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    uni_embeds = null_uni_embed(device="cpu", dtype=torch.float32)
    if not args.null_uni:
        # Try to load UNI for each tile ad-hoc inside the experiments
        print("Note: using null UNI embedding (pass --null-uni to suppress this message)")

    shared_kwargs = dict(
        exp_channels_dir=exp_channels_dir,
        models=models,
        config=config,
        scheduler=scheduler,
        uni_embeds=uni_embeds,
        device=args.device,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
    )

    # --- Experiment 1: Microenv 2D grid ---
    if "1" in args.experiments:
        print("\n=== Experiment 1: Microenv 2D grid ===")
        reps = class_data.get("representatives", {})
        # Pick top-1 tile per axis1 class from neutral axis2 (fallback to any)
        axis1_tiles: dict[str, str] = {}
        for a1 in ("cancer", "immune", "healthy"):
            key = f"{a1}+neutral"
            if key in reps:
                axis1_tiles[a1] = reps[key]["tile_id"]
            else:
                # Fallback: any combo with this axis1
                for combo_key, rep in reps.items():
                    if combo_key.startswith(f"{a1}+"):
                        axis1_tiles[a1] = rep["tile_id"]
                        break
        for a1_label, tile_id in axis1_tiles.items():
            print(f"  Tile: {tile_id} (class: {a1_label})")
            grid = run_exp1_microenv_grid(tile_id, **shared_kwargs)
            render_exp1_figure(
                grid, tile_id, a1_label,
                out_path=out_dir / "exp1_microenv" / f"{a1_label}_{tile_id}.png",
            )

    # --- Experiment 2: Cell type relabeling ---
    if "2" in args.experiments:
        print("\n=== Experiment 2: Cell type relabeling ===")
        exp2 = class_data.get("exp2_tiles", {})
        if len(exp2) < 2:
            print("  WARNING: fewer than 2 exp2 tiles available — re-run classify_tiles.py with lower --exp-threshold")
        else:
            tiles_exp2 = {label: d["tile_id"] for label, d in exp2.items()}
            results2 = run_exp2_cell_type_relabeling(tiles_exp2, **shared_kwargs)
            render_relabeling_figure(
                results2, tiles_exp2,
                exp_title="Exp 2: Cell type relabeling (given cell states + microenv)",
                out_path=out_dir / "exp2_cell_type_relabeling.png",
            )

    # --- Experiment 3: Cell state relabeling ---
    if "3" in args.experiments:
        print("\n=== Experiment 3: Cell state relabeling ===")
        exp3 = class_data.get("exp3_tiles", {})
        if len(exp3) < 2:
            print("  WARNING: fewer than 2 exp3 tiles — re-run classify_tiles.py with lower --exp-threshold")
        else:
            tiles_exp3 = {label: d["tile_id"] for label, d in exp3.items()}
            results3 = run_exp3_cell_state_relabeling(tiles_exp3, **shared_kwargs)
            render_relabeling_figure(
                results3, tiles_exp3,
                exp_title="Exp 3: Cell state relabeling (given cell types + microenv)",
                out_path=out_dir / "exp3_cell_state_relabeling.png",
            )

    print(f"\nAll experiments complete. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Syntax check**

```bash
python -c "from tools.stage3.channel_sweep import main; print('ok')"
```

Expected: `ok`

- [ ] **Step 3: Smoke-test (requires models + tile_classes.json, run Exp 2 only first)**

```bash
cd /home/ec2-user/PixCell
python tools/stage3/channel_sweep.py \
    --class-json tile_classes.json \
    --data-root data/orion-crc33 \
    --out inference_output/channel_sweep/ \
    --null-uni \
    --experiments 2 \
    --seed 42
```

Expected: `exp2_cell_type_relabeling.png` written to `inference_output/channel_sweep/`.

- [ ] **Step 4: Run all experiments**

```bash
python tools/stage3/channel_sweep.py \
    --class-json tile_classes.json \
    --data-root data/orion-crc33 \
    --out inference_output/channel_sweep/ \
    --null-uni \
    --seed 42
```

Expected: `exp1_microenv/cancer_<tile>.png`, `immune_<tile>.png`, `healthy_<tile>.png`, `exp2_cell_type_relabeling.png`, `exp3_cell_state_relabeling.png` in output dir.

- [ ] **Step 5: Run all tests one final time**

```bash
python -m pytest tests/test_leave_one_out_diff.py tests/test_classify_tiles.py tests/test_channel_sweep.py -v 2>&1 | tail -20
```

Expected: All tests PASS.

- [ ] **Step 6: Final commit**

```bash
git add tools/stage3/channel_sweep.py
git commit -m "feat: add channel sweep CLI orchestration (Exp 1/2/3)"
```
