# UNI Semantic Ablation (a4_uni_probe) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `src/a4_uni_probe/` to (1) linearly probe UNI embeddings against TME-channel and CellViT-morphology labels, (2) sweep UNI along probe directions to test causal control of generated H&E, (3) null probe subspaces to verify removal degrades the target attribute.

**Architecture:** Three-stage pipeline mirroring `src/a2_decomposition/` patterns. Stage 1 is CPU sklearn; Stages 2/3 wrap existing ControlNet inference with an overridden UNI tensor. Outputs land under `src/a4_uni_probe/out/` and feed `paper_figures/`. CellViT is invoked through the existing `tools/cellvit/` batch export+import pattern.

**Tech Stack:** Python 3.12, NumPy, scikit-learn (Ridge, GroupKFold), pandas (CSV I/O), Matplotlib (figures), PyTorch + diffusers (Stages 2/3 inference, reuses `tools/stage3/tile_pipeline.py`), CellViT (external segmenter, called via existing flat-PNG export pipeline).

**Spec:** [`docs/superpowers/specs/2026-05-12-uni-probe-semantic-ablation-design.md`](../specs/2026-05-12-uni-probe-semantic-ablation-design.md)

**Role split:** Per `CLAUDE.md`, Codex executes all code edits. This plan is the input to a Codex rescue session.

---

## File Structure

| Path | Purpose |
|---|---|
| `src/a4_uni_probe/__init__.py` | Package marker. |
| `src/a4_uni_probe/main.py` | Argparse CLI entrypoint: `probe`, `sweep`, `null`, `figures`. Discovers tile IDs and dispatches per-subcommand workers. |
| `src/a4_uni_probe/labels.py` | Build the per-tile label matrix from `exp_channels/` (channel-derived) and CellViT outputs on real H&E (morphology). Cache to `out/labels.npz`. |
| `src/a4_uni_probe/features.py` | Load cached UNI embeddings into `(N, D_UNI)` matrix and build pooled TME-channel baseline features `(N, D_TME)`. Cache to `out/features.npz`. |
| `src/a4_uni_probe/probe.py` | Stage 1 linear probes with 5-fold GroupKFold by spatial bucket; persist coefficients and per-fold scores. |
| `src/a4_uni_probe/edit.py` | UNI vector editing helpers: `sweep_uni(uni, w, alphas)` and `null_uni(uni, w)`. Pure numpy. |
| `src/a4_uni_probe/inference.py` | Thin wrapper over `tools/stage3/tile_pipeline.py` accepting an explicit UNI tensor override (no embedding reload), reusing diffusion checkpoint. |
| `src/a4_uni_probe/metrics.py` | CellViT-derived per-tile morphology metrics from imported CellViT JSON sidecars + channel-derived metric reuse from existing utilities. |
| `src/a4_uni_probe/figures.py` | Panel A (probe R² bars), Panel B (sweep slope), Panel C (null drop). |
| `src/a4_uni_probe/out/` | All outputs: `labels.npz`, `features.npz`, `probe_results.{csv,json}`, `sweep/<attr>/...`, `null/<attr>/...`, `figures/*.png`. |
| `tests/test_a4_labels.py` | Unit tests for `labels.build_label_matrix` + bucket assignment. |
| `tests/test_a4_features.py` | Unit tests for UNI / TME-baseline feature builders. |
| `tests/test_a4_probe.py` | Unit tests for probe fit + ranking + GroupKFold determinism. |
| `tests/test_a4_edit.py` | Unit tests for `sweep_uni` and `null_uni`. |
| `tests/test_a4_metrics.py` | Unit tests for CellViT JSON → morphology row extractor. |

Conventions: copy `src/a2_decomposition/` for module layout, `src/a1_probe_linear/main.py` for probe code style, `tools/cellvit/import_results.py` for CellViT JSON sidecar shape.

---

## Task 1: Scaffold package and shared paths

**Files:**
- Create: `src/a4_uni_probe/__init__.py` (empty)
- Create: `src/a4_uni_probe/main.py`

- [ ] **Step 1: Create package**

```bash
mkdir -p src/a4_uni_probe/out
: > src/a4_uni_probe/__init__.py
```

- [ ] **Step 2: Write `main.py` skeleton**

```python
"""CLI entrypoint for a4_uni_probe (semantic-ablation hybrid probe)."""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = ROOT / "data" / "orion-crc33"
DEFAULT_FEATURES_DIR = DEFAULT_DATA_ROOT / "features"
DEFAULT_EXP_CHANNELS_DIR = DEFAULT_DATA_ROOT / "exp_channels"
DEFAULT_HE_DIR = DEFAULT_DATA_ROOT / "he"
DEFAULT_OUT_DIR = ROOT / "src" / "a4_uni_probe" / "out"
DEFAULT_CHECKPOINT_DIR = ROOT / "checkpoints" / "pixcell_controlnet_exp" / "npy_inputs"
DEFAULT_CONFIG_PATH = ROOT / "configs" / "config_controlnet_exp.py"
DEFAULT_CELLVIT_REAL_DIR = ROOT / "src" / "a4_uni_probe" / "out" / "cellvit_real"

DEFAULT_SEED = 42
DEFAULT_NUM_STEPS = 20
DEFAULT_GUIDANCE_SCALE = 2.5
DEFAULT_K_SWEEP_TILES = 50
DEFAULT_ALPHAS = (-2.0, -1.0, 0.0, 1.0, 2.0)
DEFAULT_CV_FOLDS = 5
DEFAULT_SPATIAL_BUCKET_PX = 4096  # bucket size for GroupKFold spatial split


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="a4 UNI probe / sweep / null pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    p_probe = sub.add_parser("probe", help="Stage 1 linear probes")
    p_probe.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p_probe.add_argument("--features-dir", type=Path, default=DEFAULT_FEATURES_DIR)
    p_probe.add_argument("--exp-channels-dir", type=Path, default=DEFAULT_EXP_CHANNELS_DIR)
    p_probe.add_argument("--cellvit-real-dir", type=Path, default=DEFAULT_CELLVIT_REAL_DIR)
    p_probe.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p_probe.add_argument("--cv-folds", type=int, default=DEFAULT_CV_FOLDS)
    p_probe.add_argument("--bucket-px", type=int, default=DEFAULT_SPATIAL_BUCKET_PX)

    p_sweep = sub.add_parser("sweep", help="Stage 2 probe-direction sweep")
    p_sweep.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p_sweep.add_argument("--features-dir", type=Path, default=DEFAULT_FEATURES_DIR)
    p_sweep.add_argument("--k-tiles", type=int, default=DEFAULT_K_SWEEP_TILES)
    p_sweep.add_argument("--alphas", type=float, nargs="+", default=list(DEFAULT_ALPHAS))
    p_sweep.add_argument("--top-k-attrs", type=int, default=4)
    p_sweep.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p_sweep.add_argument("--num-steps", type=int, default=DEFAULT_NUM_STEPS)
    p_sweep.add_argument("--guidance-scale", type=float, default=DEFAULT_GUIDANCE_SCALE)
    p_sweep.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    p_sweep.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)

    p_null = sub.add_parser("null", help="Stage 3 subspace nulling")
    p_null.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p_null.add_argument("--features-dir", type=Path, default=DEFAULT_FEATURES_DIR)
    p_null.add_argument("--k-tiles", type=int, default=DEFAULT_K_SWEEP_TILES)
    p_null.add_argument("--top-k-attrs", type=int, default=4)
    p_null.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p_null.add_argument("--num-steps", type=int, default=DEFAULT_NUM_STEPS)
    p_null.add_argument("--guidance-scale", type=float, default=DEFAULT_GUIDANCE_SCALE)
    p_null.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    p_null.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)

    p_fig = sub.add_parser("figures", help="Render Panel A/B/C")
    p_fig.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "probe":
        from src.a4_uni_probe.probe import run_probe
        run_probe(args)
    elif args.command == "sweep":
        from src.a4_uni_probe.edit import run_sweep
        run_sweep(args)
    elif args.command == "null":
        from src.a4_uni_probe.edit import run_null
        run_null(args)
    elif args.command == "figures":
        from src.a4_uni_probe.figures import render_all
        render_all(args.out_dir)
    else:
        parser.error(f"unknown command: {args.command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: Verify import path works**

Run: `python -c "from src.a4_uni_probe.main import build_parser; print(build_parser().parse_args(['probe']))"`
Expected: `Namespace(command='probe', out_dir=..., features_dir=..., ...)`

- [ ] **Step 4: Commit**

```bash
git add src/a4_uni_probe/__init__.py src/a4_uni_probe/main.py src/a4_uni_probe/out/
git commit -m "feat(a4): scaffold uni_probe package and CLI"
```

---

## Task 2: Label matrix — channel-derived attributes

**Files:**
- Create: `src/a4_uni_probe/labels.py`
- Test: `tests/test_a4_labels.py`

Spec maps each channel-derived attribute to its source. Use these eight numeric attributes:

| Attribute | Source channel | Reduction |
|---|---|---|
| `cancer_fraction` | `cell_types`, cancer mask | mean over `cell_masks` foreground |
| `healthy_fraction` | `cell_types`, healthy mask | mean over `cell_masks` foreground |
| `immune_fraction` | `cell_types`, immune mask | mean over `cell_masks` foreground |
| `prolif_fraction` | `cell_state`, prolif mask | mean over `cell_masks` foreground |
| `nonprolif_fraction` | `cell_state`, nonprolif mask | mean over `cell_masks` foreground |
| `dead_fraction` | `cell_state`, dead mask | mean over `cell_masks` foreground |
| `vessel_area_pct` | `vasculature` | mean over full tile |
| `mean_oxygen` | `microenv` oxygen | mean over full tile |
| `mean_glucose` | `microenv` glucose | mean over full tile |

If any optional channel is missing for a tile, set the attribute to NaN (downstream probe drops NaN rows per-attribute).

- [ ] **Step 1: Write failing test**

```python
# tests/test_a4_labels.py
"""Unit tests for a4_uni_probe.labels."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.a4_uni_probe.labels import CHANNEL_ATTR_NAMES, compute_channel_attributes


def _make_synthetic_channels(tmp_path: Path) -> Path:
    """Create a minimal exp_channels/<tile_id>/ tree with known content."""
    tile_id = "0_0"
    tile_dir = tmp_path / "exp_channels" / tile_id
    tile_dir.mkdir(parents=True)

    cell_masks = np.zeros((256, 256), dtype=np.uint8)
    cell_masks[:128, :] = 1  # half tile is "cell"
    np.save(tile_dir / "cell_masks.npy", cell_masks)

    cancer = np.zeros((256, 256), dtype=np.uint8); cancer[:64, :] = 1
    healthy = np.zeros((256, 256), dtype=np.uint8); healthy[64:128, :] = 1
    immune = np.zeros((256, 256), dtype=np.uint8)
    np.save(tile_dir / "cell_types_cancer.npy", cancer)
    np.save(tile_dir / "cell_types_healthy.npy", healthy)
    np.save(tile_dir / "cell_types_immune.npy", immune)

    prolif = np.zeros((256, 256), dtype=np.uint8); prolif[:32, :] = 1
    nonprolif = np.zeros((256, 256), dtype=np.uint8); nonprolif[32:128, :] = 1
    dead = np.zeros((256, 256), dtype=np.uint8)
    np.save(tile_dir / "cell_state_prolif.npy", prolif)
    np.save(tile_dir / "cell_state_nonprolif.npy", nonprolif)
    np.save(tile_dir / "cell_state_dead.npy", dead)

    vasc = np.zeros((256, 256), dtype=np.float32); vasc[:50, :] = 1.0
    np.save(tile_dir / "vasculature.npy", vasc)

    oxygen = np.full((256, 256), 0.5, dtype=np.float32)
    glucose = np.full((256, 256), 0.25, dtype=np.float32)
    np.save(tile_dir / "oxygen.npy", oxygen)
    np.save(tile_dir / "glucose.npy", glucose)

    return tmp_path / "exp_channels"


def test_compute_channel_attributes_returns_expected_values(tmp_path):
    exp_root = _make_synthetic_channels(tmp_path)
    row = compute_channel_attributes(exp_root, "0_0")
    assert set(row.keys()) == set(CHANNEL_ATTR_NAMES)
    # cancer covers 64/128 cell-masked rows = 0.5; healthy = 0.5; immune = 0.0
    assert row["cancer_fraction"] == pytest.approx(0.5, abs=1e-6)
    assert row["healthy_fraction"] == pytest.approx(0.5, abs=1e-6)
    assert row["immune_fraction"] == pytest.approx(0.0, abs=1e-6)
    assert row["prolif_fraction"] == pytest.approx(0.25, abs=1e-6)
    assert row["nonprolif_fraction"] == pytest.approx(0.75, abs=1e-6)
    assert row["dead_fraction"] == pytest.approx(0.0, abs=1e-6)
    assert row["vessel_area_pct"] == pytest.approx(50.0 / 256.0, abs=1e-6)
    assert row["mean_oxygen"] == pytest.approx(0.5, abs=1e-6)
    assert row["mean_glucose"] == pytest.approx(0.25, abs=1e-6)


def test_missing_optional_channel_returns_nan(tmp_path):
    exp_root = _make_synthetic_channels(tmp_path)
    (exp_root / "0_0" / "vasculature.npy").unlink()
    row = compute_channel_attributes(exp_root, "0_0")
    assert np.isnan(row["vessel_area_pct"])
    # required channels still present
    assert not np.isnan(row["cancer_fraction"])
```

- [ ] **Step 2: Run test, expect failure**

Run: `pytest tests/test_a4_labels.py -v`
Expected: `ImportError: cannot import name 'CHANNEL_ATTR_NAMES'`

- [ ] **Step 3: Implement `labels.py`**

```python
# src/a4_uni_probe/labels.py
"""Per-tile label extraction for channel-derived and CellViT-morphology attributes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np


CHANNEL_ATTR_NAMES = (
    "cancer_fraction",
    "healthy_fraction",
    "immune_fraction",
    "prolif_fraction",
    "nonprolif_fraction",
    "dead_fraction",
    "vessel_area_pct",
    "mean_oxygen",
    "mean_glucose",
)

MORPHOLOGY_ATTR_NAMES = (
    "nuclear_area_mean",
    "eccentricity_mean",
    "nuclei_density",
    "intensity_mean_h",
    "intensity_mean_e",
)

ALL_ATTR_NAMES = CHANNEL_ATTR_NAMES + MORPHOLOGY_ATTR_NAMES


def _load_npy_or_none(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    arr = np.load(path)
    if arr.dtype.kind in "ui":
        return arr.astype(np.float32)
    return arr.astype(np.float32, copy=False)


def _fraction_within_mask(channel: np.ndarray | None, mask: np.ndarray) -> float:
    if channel is None:
        return float("nan")
    foreground = mask > 0
    n = int(foreground.sum())
    if n == 0:
        return float("nan")
    return float(channel[foreground].mean())


def compute_channel_attributes(exp_channels_root: Path, tile_id: str) -> dict[str, float]:
    """Compute the 9 channel-derived attributes for one tile."""
    tile_dir = Path(exp_channels_root) / tile_id
    mask = _load_npy_or_none(tile_dir / "cell_masks.npy")
    if mask is None:
        raise FileNotFoundError(f"cell_masks.npy missing for tile {tile_id} in {exp_channels_root}")

    row: dict[str, float] = {}
    for prefix, members in [
        ("cell_types", ("cancer", "healthy", "immune")),
        ("cell_state", ("prolif", "nonprolif", "dead")),
    ]:
        for member in members:
            channel = _load_npy_or_none(tile_dir / f"{prefix}_{member}.npy")
            row[f"{member}_fraction"] = _fraction_within_mask(channel, mask)

    vasc = _load_npy_or_none(tile_dir / "vasculature.npy")
    row["vessel_area_pct"] = float(vasc.mean()) if vasc is not None else float("nan")

    oxygen = _load_npy_or_none(tile_dir / "oxygen.npy")
    row["mean_oxygen"] = float(oxygen.mean()) if oxygen is not None else float("nan")

    glucose = _load_npy_or_none(tile_dir / "glucose.npy")
    row["mean_glucose"] = float(glucose.mean()) if glucose is not None else float("nan")

    return row


def compute_morphology_attributes_from_cellvit(
    cellvit_json_path: Path,
) -> dict[str, float]:
    """Reduce one CellViT-per-tile JSON sidecar to mean morphology stats.

    Expects the schema used by tools/cellvit/import_results.py:
      {"nuclei": [{"area": float, "eccentricity": float,
                   "intensity_h": float, "intensity_e": float}, ...],
       "tile_area_px": int}
    Missing/empty -> NaN.
    """
    if not cellvit_json_path.is_file():
        return {name: float("nan") for name in MORPHOLOGY_ATTR_NAMES}
    data = json.loads(cellvit_json_path.read_text())
    nuclei = data.get("nuclei") or []
    if not nuclei:
        return {name: float("nan") for name in MORPHOLOGY_ATTR_NAMES}
    areas = np.array([n.get("area", np.nan) for n in nuclei], dtype=np.float64)
    eccs = np.array([n.get("eccentricity", np.nan) for n in nuclei], dtype=np.float64)
    ihs = np.array([n.get("intensity_h", np.nan) for n in nuclei], dtype=np.float64)
    ies = np.array([n.get("intensity_e", np.nan) for n in nuclei], dtype=np.float64)
    tile_area = float(data.get("tile_area_px", 256 * 256))
    return {
        "nuclear_area_mean": float(np.nanmean(areas)) if areas.size else float("nan"),
        "eccentricity_mean": float(np.nanmean(eccs)) if eccs.size else float("nan"),
        "nuclei_density": float(len(nuclei) / tile_area) if tile_area > 0 else float("nan"),
        "intensity_mean_h": float(np.nanmean(ihs)) if ihs.size else float("nan"),
        "intensity_mean_e": float(np.nanmean(ies)) if ies.size else float("nan"),
    }


def build_label_matrix(
    tile_ids: Iterable[str],
    exp_channels_root: Path,
    cellvit_real_dir: Path,
) -> tuple[np.ndarray, list[str]]:
    """Build the (N, A) label matrix in canonical ALL_ATTR_NAMES order."""
    tile_ids = list(tile_ids)
    matrix = np.full((len(tile_ids), len(ALL_ATTR_NAMES)), np.nan, dtype=np.float64)
    for i, tile_id in enumerate(tile_ids):
        ch_row = compute_channel_attributes(exp_channels_root, tile_id)
        morpho_row = compute_morphology_attributes_from_cellvit(
            Path(cellvit_real_dir) / f"{tile_id}.json"
        )
        row = {**ch_row, **morpho_row}
        for j, name in enumerate(ALL_ATTR_NAMES):
            matrix[i, j] = row[name]
    return matrix, list(ALL_ATTR_NAMES)
```

- [ ] **Step 4: Run tests, expect pass**

Run: `pytest tests/test_a4_labels.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/a4_uni_probe/labels.py tests/test_a4_labels.py
git commit -m "feat(a4): channel-derived + CellViT morphology label builder"
```

> ⚠️ **Channel filename convention:** Real `data/orion-crc33/exp_channels/<tile>/` may use different per-channel filenames (e.g., a single `cell_types.npy` with shape `(3, H, W)` rather than `cell_types_cancer.npy`). Before implementing Step 3, inspect one real tile: `ls data/orion-crc33/exp_channels/$(ls data/orion-crc33/exp_channels | head -1)/`. If the layout differs, **update both the test and the implementation** to use the real layout — keep the per-attribute dict outputs unchanged.

---

## Task 3: Feature matrix — UNI embedding + TME pooled baseline

**Files:**
- Create: `src/a4_uni_probe/features.py`
- Test: `tests/test_a4_features.py`

The UNI feature loader reuses `data/orion-crc33/features/<tile_id>_uni.npy` (one vector per tile, dim `D_UNI`). TME baseline pools per-channel mean and std over the tile, giving a low-dimensional feature representing what channel-statistics alone can predict.

TME baseline feature columns: for each of the 9 channels in `CHANNEL_ATTR_NAMES` order, append `[mean, std]` → `D_TME = 18`. Missing channel → use `[0, 0]`.

- [ ] **Step 1: Write failing test**

```python
# tests/test_a4_features.py
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.a4_uni_probe.features import (
    TME_FEATURE_DIM,
    build_tme_baseline_features,
    build_uni_features,
)


def test_build_uni_features_stacks_in_order(tmp_path):
    features_dir = tmp_path / "features"
    features_dir.mkdir()
    np.save(features_dir / "0_0_uni.npy", np.array([1.0, 2.0, 3.0]))
    np.save(features_dir / "256_0_uni.npy", np.array([4.0, 5.0, 6.0]))
    mat = build_uni_features(["0_0", "256_0"], features_dir)
    assert mat.shape == (2, 3)
    assert mat[0].tolist() == [1.0, 2.0, 3.0]
    assert mat[1].tolist() == [4.0, 5.0, 6.0]


def test_build_tme_baseline_features_handles_missing(tmp_path):
    # build only cell_masks + cell_types_cancer; everything else missing
    exp_root = tmp_path / "exp_channels"
    tile_dir = exp_root / "0_0"
    tile_dir.mkdir(parents=True)
    np.save(tile_dir / "cell_masks.npy", np.ones((4, 4), dtype=np.uint8))
    np.save(tile_dir / "cell_types_cancer.npy", np.full((4, 4), 0.5, dtype=np.float32))

    mat = build_tme_baseline_features(["0_0"], exp_root)
    assert mat.shape == (1, TME_FEATURE_DIM)
    # cancer_fraction col mean=0.5, std=0
    assert mat[0, 0] == pytest.approx(0.5)
    assert mat[0, 1] == pytest.approx(0.0)
    # missing channels -> zeros
    assert mat[0, 4:].sum() == pytest.approx(0.0)
```

- [ ] **Step 2: Run test, expect failure**

Run: `pytest tests/test_a4_features.py -v`
Expected: `ImportError: cannot import name 'TME_FEATURE_DIM'`

- [ ] **Step 3: Implement `features.py`**

```python
# src/a4_uni_probe/features.py
"""Feature matrices for the UNI / TME probes."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from src.a4_uni_probe.labels import CHANNEL_ATTR_NAMES


_TME_FILENAME = {
    "cancer_fraction": "cell_types_cancer.npy",
    "healthy_fraction": "cell_types_healthy.npy",
    "immune_fraction": "cell_types_immune.npy",
    "prolif_fraction": "cell_state_prolif.npy",
    "nonprolif_fraction": "cell_state_nonprolif.npy",
    "dead_fraction": "cell_state_dead.npy",
    "vessel_area_pct": "vasculature.npy",
    "mean_oxygen": "oxygen.npy",
    "mean_glucose": "glucose.npy",
}

TME_FEATURE_DIM = 2 * len(CHANNEL_ATTR_NAMES)  # mean + std per channel


def build_uni_features(tile_ids: Iterable[str], features_dir: Path) -> np.ndarray:
    """Load UNI features into shape (N, D_UNI)."""
    features_dir = Path(features_dir)
    vectors: list[np.ndarray] = []
    for tile_id in tile_ids:
        path = features_dir / f"{tile_id}_uni.npy"
        vec = np.load(path).astype(np.float32, copy=False).reshape(-1)
        vectors.append(vec)
    return np.stack(vectors, axis=0)


def build_tme_baseline_features(
    tile_ids: Iterable[str],
    exp_channels_root: Path,
) -> np.ndarray:
    """Per-channel [mean, std] pooled features. Missing channel -> [0, 0]."""
    tile_ids = list(tile_ids)
    exp_root = Path(exp_channels_root)
    mat = np.zeros((len(tile_ids), TME_FEATURE_DIM), dtype=np.float32)
    for i, tile_id in enumerate(tile_ids):
        tile_dir = exp_root / tile_id
        for j, attr in enumerate(CHANNEL_ATTR_NAMES):
            path = tile_dir / _TME_FILENAME[attr]
            if not path.is_file():
                continue
            arr = np.load(path).astype(np.float32, copy=False)
            mat[i, 2 * j] = float(arr.mean())
            mat[i, 2 * j + 1] = float(arr.std())
    return mat
```

- [ ] **Step 4: Run tests, expect pass**

Run: `pytest tests/test_a4_features.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/a4_uni_probe/features.py tests/test_a4_features.py
git commit -m "feat(a4): UNI and TME-baseline feature builders"
```

> ⚠️ **Same filename caveat as Task 2** — adjust `_TME_FILENAME` to real channel layout before running on data.

---

## Task 4: CellViT pass on real H&E (one-time prep)

**Files:**
- Modify: `tools/cellvit/export_batch.py` only if the existing batch flow cannot accept a flat directory of real H&E. Otherwise no code change here — this task is operational.

- [ ] **Step 1: Verify CellViT inputs/outputs exist**

```bash
ls data/orion-crc33/he/ | head -3
ls tools/cellvit/
```

Expected: `he/` contains per-tile PNG/TIFF; `tools/cellvit/` has `export_batch.py` and `import_results.py`.

- [ ] **Step 2: Export real H&E to flat batch**

```bash
mkdir -p src/a4_uni_probe/out/cellvit_real_export
python tools/cellvit/export_batch.py \
  --src-dir data/orion-crc33/he \
  --dest-dir src/a4_uni_probe/out/cellvit_real_export \
  --flat
```

Expected: directory of `<tile_id>.png` (one per tile).
If `export_batch.py` does not yet support flat-directory input (no manifest), either run the existing manifest path or symlink real H&E PNGs into the export dir directly:

```bash
mkdir -p src/a4_uni_probe/out/cellvit_real_export
for f in data/orion-crc33/he/*.png; do
  tile=$(basename "$f" .png)
  ln -sf "$(realpath "$f")" "src/a4_uni_probe/out/cellvit_real_export/${tile}.png"
done
```

- [ ] **Step 3: Run CellViT externally**

Operator step (delegated to GPU machine + CellViT environment). Place returned JSON sidecars under `src/a4_uni_probe/out/cellvit_real/`, one `<tile_id>.json` per tile, matching the schema documented in `compute_morphology_attributes_from_cellvit` (Task 2).

- [ ] **Step 4: Import JSON sidecars (sanity check)**

```bash
python - <<'PY'
import json
from pathlib import Path
root = Path("src/a4_uni_probe/out/cellvit_real")
files = list(root.glob("*.json"))
assert files, "no CellViT outputs found"
sample = json.loads(files[0].read_text())
assert "nuclei" in sample, f"unexpected schema in {files[0]}"
print(f"{len(files)} sidecars present, schema OK")
PY
```

Expected: `>0 sidecars present, schema OK`.

- [ ] **Step 5: Commit operational notes**

If you ran the symlink fallback, document it in `src/a4_uni_probe/README.md` (create a brief one-paragraph note pointing to this plan). Do not commit the symlinks or the JSON sidecars (treat them as derived data — add to `.gitignore` if not already covered).

```bash
git add src/a4_uni_probe/README.md .gitignore
git commit -m "docs(a4): document CellViT real-H&E pass"
```

> ⚠️ **If the real-H&E CellViT pass already exists under `inference_output/...`**, reuse those outputs instead of re-running. Point `--cellvit-real-dir` at the existing dir; skip Steps 2-3.

---

## Task 5: Stage 1 — linear probes with GroupKFold

**Files:**
- Create: `src/a4_uni_probe/probe.py`
- Test: `tests/test_a4_probe.py`

The probe fits one Ridge regressor per attribute on both UNI features and TME-baseline features, scored with 5-fold `GroupKFold` where groups come from `(row // bucket, col // bucket)` spatial buckets (prevents adjacency leakage).

- [ ] **Step 1: Write failing test**

```python
# tests/test_a4_probe.py
from __future__ import annotations

import numpy as np

from src.a4_uni_probe.probe import (
    ProbeFitResult,
    fit_probes_for_attribute,
    spatial_bucket_groups,
)


def test_spatial_bucket_groups_clusters_neighbors():
    tile_ids = ["0_0", "0_64", "0_128", "4096_0", "4096_4096"]
    groups = spatial_bucket_groups(tile_ids, bucket_px=4096)
    # first three share bucket (0,0); fourth is (1,0); fifth is (1,1)
    assert groups[0] == groups[1] == groups[2]
    assert groups[0] != groups[3]
    assert groups[3] != groups[4]


def test_fit_probes_recovers_perfect_linear_relationship():
    rng = np.random.default_rng(0)
    n, d = 200, 32
    X = rng.normal(size=(n, d)).astype(np.float32)
    w_true = rng.normal(size=d)
    y = X @ w_true
    tile_ids = [f"{i}_{0}" for i in range(0, n * 256, 256)]
    groups = spatial_bucket_groups(tile_ids, bucket_px=4096)

    result = fit_probes_for_attribute(X, y, groups, n_folds=5, seed=0)
    assert isinstance(result, ProbeFitResult)
    assert result.r2_mean > 0.99
    assert result.coef.shape == (d,)
    np.testing.assert_allclose(result.coef / np.linalg.norm(result.coef),
                               w_true / np.linalg.norm(w_true), atol=0.05)
```

- [ ] **Step 2: Run test, expect failure**

Run: `pytest tests/test_a4_probe.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `probe.py`**

```python
# src/a4_uni_probe/probe.py
"""Stage 1 — fit linear probes for each label attribute."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src._tasklib.io import ensure_directory, write_json
from src._tasklib.tile_ids import list_feature_tile_ids, parse_tile_id
from src.a4_uni_probe.features import build_tme_baseline_features, build_uni_features
from src.a4_uni_probe.labels import ALL_ATTR_NAMES, build_label_matrix


@dataclass(frozen=True)
class ProbeFitResult:
    r2_mean: float
    r2_std: float
    r2_per_fold: tuple[float, ...]
    coef: np.ndarray  # shape (d,), refit on full data after CV

    def as_record(self) -> dict[str, float]:
        return {
            "r2_mean": float(self.r2_mean),
            "r2_std": float(self.r2_std),
            "r2_per_fold": [float(v) for v in self.r2_per_fold],
        }


def spatial_bucket_groups(tile_ids: list[str], bucket_px: int) -> np.ndarray:
    """Map each tile_id to an integer group ID for GroupKFold."""
    seen: dict[tuple[int, int], int] = {}
    groups = np.empty(len(tile_ids), dtype=np.int64)
    for i, tid in enumerate(tile_ids):
        r, c = parse_tile_id(tid)
        key = (r // bucket_px, c // bucket_px)
        if key not in seen:
            seen[key] = len(seen)
        groups[i] = seen[key]
    return groups


def fit_probes_for_attribute(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_folds: int,
    seed: int,
) -> ProbeFitResult:
    """Fit Ridge with standardization; CV over GroupKFold; refit on full data."""
    mask = ~np.isnan(y)
    X_use, y_use, g_use = X[mask], y[mask], groups[mask]
    n_unique = int(np.unique(g_use).size)
    folds = min(n_folds, n_unique)
    if folds < 2:
        raise ValueError(f"GroupKFold needs >=2 groups, got {n_unique}")
    pipeline = Pipeline([
        ("scale", StandardScaler(with_mean=True, with_std=True)),
        ("ridge", Ridge(alpha=1.0, random_state=seed)),
    ])
    splitter = GroupKFold(n_splits=folds)
    fold_scores: list[float] = []
    for train_idx, test_idx in splitter.split(X_use, y_use, g_use):
        pipeline.fit(X_use[train_idx], y_use[train_idx])
        pred = pipeline.predict(X_use[test_idx])
        fold_scores.append(float(r2_score(y_use[test_idx], pred)))
    # Refit on all data for direction extraction
    pipeline.fit(X_use, y_use)
    # Coefficient is in scaled space; back out to original-feature direction
    scaler: StandardScaler = pipeline.named_steps["scale"]
    ridge: Ridge = pipeline.named_steps["ridge"]
    coef = ridge.coef_ / np.where(scaler.scale_ == 0, 1.0, scaler.scale_)
    return ProbeFitResult(
        r2_mean=float(np.mean(fold_scores)),
        r2_std=float(np.std(fold_scores)),
        r2_per_fold=tuple(fold_scores),
        coef=coef.astype(np.float32),
    )


def run_probe(args: argparse.Namespace) -> None:
    out_dir = ensure_directory(args.out_dir)
    tile_ids = list_feature_tile_ids(args.features_dir, suffix="_uni.npy")
    print(f"[a4] discovered {len(tile_ids)} tiles with UNI features")

    X_uni = build_uni_features(tile_ids, args.features_dir)
    X_tme = build_tme_baseline_features(tile_ids, args.exp_channels_dir)
    Y, attr_names = build_label_matrix(
        tile_ids, args.exp_channels_dir, args.cellvit_real_dir
    )
    np.savez(
        out_dir / "features.npz",
        tile_ids=np.array(tile_ids),
        X_uni=X_uni,
        X_tme=X_tme,
    )
    np.savez(out_dir / "labels.npz", tile_ids=np.array(tile_ids), Y=Y, attr_names=np.array(attr_names))

    groups = spatial_bucket_groups(tile_ids, args.bucket_px)
    csv_path = out_dir / "probe_results.csv"
    records: list[dict[str, object]] = []
    direction_dir = ensure_directory(out_dir / "probe_directions")
    with csv_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "attr", "feature_set",
            "r2_mean", "r2_std",
            "delta_r2_uni_minus_tme",
            "n_valid",
        ])
        for j, attr in enumerate(attr_names):
            y = Y[:, j]
            try:
                uni_res = fit_probes_for_attribute(X_uni, y, groups, args.cv_folds, args.seed)
                tme_res = fit_probes_for_attribute(X_tme, y, groups, args.cv_folds, args.seed)
            except ValueError as e:
                print(f"[a4] skipping {attr}: {e}")
                continue
            delta = uni_res.r2_mean - tme_res.r2_mean
            n_valid = int(np.sum(~np.isnan(y)))
            writer.writerow([attr, "uni", uni_res.r2_mean, uni_res.r2_std, delta, n_valid])
            writer.writerow([attr, "tme", tme_res.r2_mean, tme_res.r2_std, delta, n_valid])
            np.save(direction_dir / f"{attr}_uni_direction.npy", uni_res.coef)
            records.append({
                "attr": attr,
                "uni": uni_res.as_record(),
                "tme": tme_res.as_record(),
                "delta_r2_uni_minus_tme": delta,
                "n_valid": n_valid,
            })
    write_json(out_dir / "probe_results.json", records)
    print(f"[a4] wrote {csv_path} and {len(records)} per-attr directions")
```

- [ ] **Step 4: Run unit tests**

Run: `pytest tests/test_a4_probe.py -v`
Expected: 2 passed.

- [ ] **Step 5: Smoke test the end-to-end probe**

Run with cached data:

```bash
python -m src.a4_uni_probe.main probe \
  --out-dir src/a4_uni_probe/out \
  --features-dir data/orion-crc33/features \
  --exp-channels-dir data/orion-crc33/exp_channels \
  --cellvit-real-dir src/a4_uni_probe/out/cellvit_real
```

Expected: stdout lists tile count and per-attr completion; `out/probe_results.csv` and `out/probe_directions/*.npy` exist; `out/features.npz`, `out/labels.npz` present.

- [ ] **Step 6: Commit**

```bash
git add src/a4_uni_probe/probe.py tests/test_a4_probe.py
git commit -m "feat(a4): Stage 1 linear probes for UNI vs TME features"
```

---

## Task 6: UNI editing helpers (pure numpy)

**Files:**
- Create: `src/a4_uni_probe/edit.py` (helper functions only; runners added later)
- Test: `tests/test_a4_edit.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_a4_edit.py
from __future__ import annotations

import numpy as np
import pytest

from src.a4_uni_probe.edit import null_uni, random_unit_direction, sweep_uni


def test_sweep_uni_produces_expected_shifts():
    uni = np.array([1.0, 0.0, 0.0])
    w = np.array([0.0, 1.0, 0.0])
    alphas = [-1.0, 0.0, 1.0]
    out = sweep_uni(uni, w, alphas)
    assert out.shape == (3, 3)
    # alpha=0 keeps uni unchanged
    np.testing.assert_allclose(out[1], uni)
    # alpha=+1 adds w * ||uni||
    np.testing.assert_allclose(out[2], uni + np.linalg.norm(uni) * w)


def test_null_uni_removes_w_component():
    uni = np.array([1.0, 1.0, 0.0])
    w = np.array([1.0, 0.0, 0.0])  # unit
    nulled = null_uni(uni, w)
    # the w direction should be removed
    assert nulled @ w == pytest.approx(0.0, abs=1e-7)
    # the orthogonal component is preserved
    np.testing.assert_allclose(nulled, [0.0, 1.0, 0.0])


def test_random_unit_direction_is_unit_and_reproducible():
    rng = np.random.default_rng(7)
    w = random_unit_direction(d=128, rng=rng)
    assert w.shape == (128,)
    assert np.linalg.norm(w) == pytest.approx(1.0, abs=1e-7)
    rng2 = np.random.default_rng(7)
    w2 = random_unit_direction(d=128, rng=rng2)
    np.testing.assert_allclose(w, w2)
```

- [ ] **Step 2: Run test, expect failure**

Run: `pytest tests/test_a4_edit.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement edit helpers**

```python
# src/a4_uni_probe/edit.py (partial — runners added in later tasks)
"""UNI vector edits (sweep + null) plus Stage 2/3 runners."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


def _unit(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    if norm == 0.0:
        raise ValueError("cannot unit-normalize zero vector")
    return v / norm


def sweep_uni(uni: np.ndarray, w: np.ndarray, alphas: Sequence[float]) -> np.ndarray:
    """Return shape (len(alphas), D) of edited UNI vectors.

    UNI'(alpha) = UNI + alpha * unit(w) * ||UNI||
    """
    uni = np.asarray(uni, dtype=np.float32).reshape(-1)
    w = _unit(np.asarray(w, dtype=np.float32).reshape(-1))
    norm = float(np.linalg.norm(uni))
    out = np.empty((len(alphas), uni.size), dtype=np.float32)
    for i, alpha in enumerate(alphas):
        out[i] = uni + alpha * norm * w
    return out


def null_uni(uni: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Project UNI onto the subspace orthogonal to unit(w)."""
    uni = np.asarray(uni, dtype=np.float32).reshape(-1)
    w = _unit(np.asarray(w, dtype=np.float32).reshape(-1))
    return (uni - (uni @ w) * w).astype(np.float32)


def random_unit_direction(d: int, rng: np.random.Generator) -> np.ndarray:
    v = rng.normal(size=d).astype(np.float32)
    return _unit(v)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_a4_edit.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/a4_uni_probe/edit.py tests/test_a4_edit.py
git commit -m "feat(a4): UNI vector sweep/null/random-direction helpers"
```

---

## Task 7: Inference wrapper accepting UNI override

**Files:**
- Read first: `tools/stage3/tile_pipeline.py`, `train_scripts/inference_controlnet.py`, `stage3_inference.py`.
- Create: `src/a4_uni_probe/inference.py`

The goal is a single function `generate_he_with_uni_override(tile_id, uni_vector, *, tme_channels_dir, out_path, ...)` that runs the existing ControlNet inference exactly as `src/a2_decomposition/main.py` does for `uni_plus_tme` mode, but with the UNI tensor replaced by `uni_vector` instead of being loaded from `features/<tile_id>_uni.npy`.

The implementation depends on the precise inference entrypoint. Two acceptable shapes:

**(a)** If `tools/stage3/tile_pipeline.py` already exposes a function that takes a `uni_features_override: torch.Tensor | None = None` arg, call it directly.

**(b)** If not, add the override parameter to that function. Do this in a single small commit; do not refactor surrounding code. Add a test that loading the function with `uni_features_override=None` returns identical pixels to the existing path.

- [ ] **Step 1: Read existing inference pipeline**

```bash
grep -n "def " tools/stage3/tile_pipeline.py | head -30
grep -n "uni" tools/stage3/tile_pipeline.py | head -30
```

Identify the function that generates a tile from UNI + TME inputs. Confirm whether it accepts an explicit UNI tensor or always loads from disk.

- [ ] **Step 2: If override absent, add it**

Modify the relevant function signature to accept `uni_features_override: np.ndarray | None = None`. When not None, replace the disk-loaded UNI with `uni_features_override` (cast to the appropriate dtype/device). Do not change behavior when None. Add a docstring note.

```python
def run_tile(
    tile_id: str,
    *,
    uni_features_override: np.ndarray | None = None,
    # ... existing args unchanged ...
) -> ...:
    """...
    Args:
        uni_features_override: Optional (D_UNI,) array. If set, replaces the
            UNI features that would otherwise be loaded from disk. Used by
            src/a4_uni_probe for direction-edit experiments.
    ...
    """
    if uni_features_override is None:
        uni = _load_uni_from_disk(tile_id)
    else:
        uni = uni_features_override
    # ... rest unchanged ...
```

- [ ] **Step 3: Add identity test**

Add to a new `tests/test_a4_inference_override.py`:

```python
"""Verify uni_features_override=None reproduces the legacy inference output."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA-only test", allow_module_level=True)

from tools.stage3.tile_pipeline import run_tile  # adjust import to actual path


def test_override_none_matches_disk_load(tmp_path, sample_tile_id, base_kwargs):
    out_disk = run_tile(sample_tile_id, **base_kwargs)
    out_override = run_tile(
        sample_tile_id,
        uni_features_override=None,
        **base_kwargs,
    )
    np.testing.assert_allclose(out_disk, out_override, atol=0, rtol=0)
```

If existing test fixtures `sample_tile_id` and `base_kwargs` are unavailable, write a thin fixture in the same file pulling from `data/orion-crc33` (skip if not present).

- [ ] **Step 4: Write `src/a4_uni_probe/inference.py`**

```python
# src/a4_uni_probe/inference.py
"""Wrap stage3 inference with a UNI override knob for direction-edit experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# NOTE: import path may differ; align with whichever entrypoint Task 7 Step 2 updated.
from tools.stage3.tile_pipeline import run_tile


@dataclass(frozen=True)
class GenSpec:
    tile_id: str
    uni: np.ndarray  # shape (D_UNI,)
    out_path: Path  # PNG output


def generate_with_uni_override(
    spec: GenSpec,
    *,
    checkpoint_dir: Path,
    config_path: Path,
    exp_channels_dir: Path,
    num_steps: int,
    guidance_scale: float,
    seed: int,
) -> Path:
    """Run a single ControlNet generation with `uni` replacing the cached UNI vector."""
    spec.out_path.parent.mkdir(parents=True, exist_ok=True)
    run_tile(
        spec.tile_id,
        uni_features_override=spec.uni,
        out_path=spec.out_path,
        checkpoint_dir=checkpoint_dir,
        config_path=config_path,
        exp_channels_dir=exp_channels_dir,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        seed=seed,
    )
    return spec.out_path
```

> ⚠️ **Codex must inspect `tools/stage3/tile_pipeline.py` first** and adapt the call signature in `generate_with_uni_override` to whatever `run_tile` (or the actual function) accepts. The signature above is the *target* — adjust to reality.

- [ ] **Step 5: Smoke test (one tile, one alpha)**

```bash
python - <<'PY'
import numpy as np
from pathlib import Path
from src.a4_uni_probe.inference import GenSpec, generate_with_uni_override
ROOT = Path(".").resolve()
tile_id = "10240_11008"  # replace with any tile present in data/orion-crc33/features/
uni = np.load(ROOT / "data/orion-crc33/features" / f"{tile_id}_uni.npy")
out = generate_with_uni_override(
    GenSpec(tile_id=tile_id, uni=uni, out_path=ROOT / "src/a4_uni_probe/out/smoke/alpha_0.png"),
    checkpoint_dir=ROOT / "checkpoints/pixcell_controlnet_exp/npy_inputs",
    config_path=ROOT / "configs/config_controlnet_exp.py",
    exp_channels_dir=ROOT / "data/orion-crc33/exp_channels",
    num_steps=20, guidance_scale=2.5, seed=42,
)
print("smoke output:", out, out.is_file())
PY
```

Expected: PNG file present.

- [ ] **Step 6: Commit**

```bash
git add tools/stage3/tile_pipeline.py src/a4_uni_probe/inference.py tests/test_a4_inference_override.py
git commit -m "feat(a4): uni_features_override hook in stage3 inference + wrapper"
```

---

## Task 8: Stage 2 sweep runner

**Files:**
- Modify: `src/a4_uni_probe/edit.py` (add `run_sweep` and slope-fit helper)

- [ ] **Step 1: Append `run_sweep` to `edit.py`**

```python
# Append to src/a4_uni_probe/edit.py

from src._tasklib.io import ensure_directory, write_json
from src.a4_uni_probe.labels import MORPHOLOGY_ATTR_NAMES
from src.a4_uni_probe.metrics import morphology_row_for_image


def _select_sweep_attrs(probe_csv: Path, top_k: int) -> list[str]:
    """Choose top-k morphology attrs by uni_R2 - tme_R2, descending."""
    import csv
    candidates: list[tuple[str, float]] = []
    with probe_csv.open() as fh:
        reader = csv.DictReader(fh)
        seen: dict[str, float] = {}
        for row in reader:
            if row["feature_set"] != "uni":
                continue
            if row["attr"] not in MORPHOLOGY_ATTR_NAMES:
                continue
            seen[row["attr"]] = float(row["delta_r2_uni_minus_tme"])
        for attr, delta in sorted(seen.items(), key=lambda kv: -kv[1])[:top_k]:
            candidates.append((attr, delta))
    return [a for a, _ in candidates]


def _select_sweep_tiles(labels_npz: Path, attr: str, k: int, seed: int) -> list[str]:
    """Sample k tiles spanning the attribute's value range (stratified by 5 quintiles)."""
    data = np.load(labels_npz, allow_pickle=True)
    tile_ids = list(data["tile_ids"])
    attr_names = list(data["attr_names"])
    j = attr_names.index(attr)
    y = data["Y"][:, j]
    valid = ~np.isnan(y)
    rng = np.random.default_rng(seed)
    quintile_edges = np.quantile(y[valid], np.linspace(0, 1, 6))
    picks: list[str] = []
    per_bucket = max(1, k // 5)
    for b in range(5):
        lo, hi = quintile_edges[b], quintile_edges[b + 1]
        in_bucket = np.where(valid & (y >= lo) & (y <= hi))[0]
        chosen = rng.choice(in_bucket, size=min(per_bucket, len(in_bucket)), replace=False)
        picks.extend(tile_ids[i] for i in chosen)
    return picks[:k]


def run_sweep(args: argparse.Namespace) -> None:
    out_dir = ensure_directory(args.out_dir)
    sweep_root = ensure_directory(out_dir / "sweep")
    probe_csv = out_dir / "probe_results.csv"
    labels_npz = out_dir / "labels.npz"
    features_npz = out_dir / "features.npz"
    if not probe_csv.is_file():
        raise FileNotFoundError(f"run `probe` first: {probe_csv} missing")

    attrs = _select_sweep_attrs(probe_csv, args.top_k_attrs)
    print(f"[a4] sweep attrs: {attrs}")
    rng = np.random.default_rng(args.seed)
    features = np.load(features_npz, allow_pickle=True)
    tile_ids_all = list(features["tile_ids"])
    X_uni = features["X_uni"]

    for attr in attrs:
        attr_dir = ensure_directory(sweep_root / attr)
        w = np.load(out_dir / "probe_directions" / f"{attr}_uni_direction.npy")
        w_unit = w / max(np.linalg.norm(w), 1e-12)
        w_rand = random_unit_direction(d=w.size, rng=np.random.default_rng(args.seed + hash(attr) % 100))
        np.save(attr_dir / "w_targeted.npy", w_unit)
        np.save(attr_dir / "w_random.npy", w_rand)

        sweep_tiles = _select_sweep_tiles(labels_npz, attr, args.k_tiles, args.seed)
        metrics_rows: list[dict[str, object]] = []
        for tile_id in sweep_tiles:
            idx = tile_ids_all.index(tile_id)
            uni = X_uni[idx]
            for direction_name, w_use in (("targeted", w_unit), ("random", w_rand)):
                edits = sweep_uni(uni, w_use, args.alphas)
                tile_dir = ensure_directory(attr_dir / tile_id / direction_name)
                for alpha, uni_edit in zip(args.alphas, edits):
                    out_path = tile_dir / f"alpha_{alpha:+.2f}.png"
                    generate_with_uni_override(
                        GenSpec(tile_id=tile_id, uni=uni_edit, out_path=out_path),
                        checkpoint_dir=args.checkpoint_dir,
                        config_path=args.config_path,
                        exp_channels_dir=Path("data/orion-crc33/exp_channels"),
                        num_steps=args.num_steps,
                        guidance_scale=args.guidance_scale,
                        seed=args.seed,
                    )
                    morpho = morphology_row_for_image(out_path)
                    metrics_rows.append({
                        "tile_id": tile_id,
                        "direction": direction_name,
                        "alpha": float(alpha),
                        "target_attr": attr,
                        "target_value": float(morpho[attr]),
                        **{f"morpho.{k}": float(v) for k, v in morpho.items()},
                    })
        _write_sweep_csv(attr_dir / "metrics.csv", metrics_rows)
        _summarize_slopes(attr_dir / "metrics.csv", attr_dir / "slope_summary.json", attr)


def _write_sweep_csv(path: Path, rows: list[dict[str, object]]) -> None:
    import csv
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _summarize_slopes(metrics_csv: Path, out_json: Path, attr: str) -> None:
    import csv
    import math
    by_dir: dict[str, list[tuple[float, float]]] = {"targeted": [], "random": []}
    with metrics_csv.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            by_dir[row["direction"]].append((float(row["alpha"]), float(row["target_value"])))
    out: dict[str, object] = {"attr": attr}
    for direction, pairs in by_dir.items():
        if not pairs:
            continue
        a = np.array([p[0] for p in pairs])
        v = np.array([p[1] for p in pairs])
        # Bootstrap slope mean + 95% CI
        rng = np.random.default_rng(0)
        slopes = []
        for _ in range(1000):
            idx = rng.integers(0, len(a), size=len(a))
            s = np.polyfit(a[idx], v[idx], 1)[0]
            slopes.append(s)
        slopes = np.array(slopes)
        out[direction] = {
            "slope_mean": float(slopes.mean()),
            "slope_ci95": [float(np.quantile(slopes, 0.025)), float(np.quantile(slopes, 0.975))],
            "n": len(pairs),
        }
    out["pass_criterion_met"] = (
        "targeted" in out
        and "random" in out
        and out["targeted"]["slope_ci95"][0] * out["targeted"]["slope_ci95"][1] > 0  # CI excludes 0
        and abs(out["targeted"]["slope_mean"]) > 3 * abs(out["random"]["slope_mean"])
    )
    write_json(out_json, out)
```

- [ ] **Step 2: Add a regression test for `_summarize_slopes`**

```python
# add to tests/test_a4_edit.py
import csv
import json

from src.a4_uni_probe.edit import _summarize_slopes


def test_summarize_slopes_detects_monotonic(tmp_path):
    csv_path = tmp_path / "metrics.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["alpha", "target_value", "direction"])
        for direction, slope in [("targeted", 1.0), ("random", 0.0)]:
            for alpha in (-2, -1, 0, 1, 2):
                writer.writerow([alpha, slope * alpha + 0.01 * alpha, direction])
    # rewrite to dict-reader-compatible shape
    rows = []
    with csv_path.open() as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    out_path = tmp_path / "slope.json"
    _summarize_slopes(csv_path, out_path, "test_attr")
    summary = json.loads(out_path.read_text())
    assert summary["targeted"]["slope_mean"] > 0.5
    assert summary["pass_criterion_met"] is True
```

(Adjust the CSV column order in this test to match `_summarize_slopes` reader expectations; the implementation reads via `DictReader` so any column order works.)

- [ ] **Step 3: Run unit tests**

Run: `pytest tests/test_a4_edit.py -v`
Expected: 4 passed.

- [ ] **Step 4: Smoke-run the sweep on K=2 tiles, 3 alphas**

```bash
python -m src.a4_uni_probe.main sweep \
  --k-tiles 2 --alphas -1 0 1 --top-k-attrs 1 --num-steps 10
```

Expected: `src/a4_uni_probe/out/sweep/<attr>/` populated; `metrics.csv` + `slope_summary.json` created.

- [ ] **Step 5: Commit**

```bash
git add src/a4_uni_probe/edit.py tests/test_a4_edit.py
git commit -m "feat(a4): Stage 2 sweep runner + slope summary"
```

---

## Task 9: Morphology metric on generated PNGs

**Files:**
- Create: `src/a4_uni_probe/metrics.py`
- Test: `tests/test_a4_metrics.py`

CellViT runs once per tile (it's expensive). Sweep/null Stages 2/3 generate PNGs in batches; we will not call CellViT inline. Instead, write the metric extractor as a placeholder that **(a)** reads a cached `<png>.json` sidecar produced by an offline CellViT pass on the generated tiles, **(b)** falls back to a cheap pixel-statistics proxy if no sidecar present.

This separates compute-heavy CellViT runs from this Python pipeline, mirroring the existing `tools/cellvit/export_batch.py` + `import_results.py` flow used in `src/a2_decomposition`.

- [ ] **Step 1: Write failing test**

```python
# tests/test_a4_metrics.py
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from src.a4_uni_probe.metrics import (
    MORPHOLOGY_ATTR_NAMES,
    morphology_row_for_image,
)


def test_returns_cellvit_values_when_sidecar_present(tmp_path):
    png_path = tmp_path / "image.png"
    Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)).save(png_path)
    sidecar = png_path.with_suffix(".png.json")
    sidecar.write_text(json.dumps({
        "nuclei": [
            {"area": 100, "eccentricity": 0.5, "intensity_h": 0.4, "intensity_e": 0.2},
            {"area": 200, "eccentricity": 0.8, "intensity_h": 0.6, "intensity_e": 0.3},
        ],
        "tile_area_px": 256,
    }))
    row = morphology_row_for_image(png_path)
    assert row["nuclear_area_mean"] == 150.0
    assert row["nuclei_density"] == 2 / 256.0
    assert set(row.keys()) == set(MORPHOLOGY_ATTR_NAMES)


def test_falls_back_when_sidecar_missing(tmp_path):
    png_path = tmp_path / "image.png"
    Image.fromarray(np.full((16, 16, 3), 128, dtype=np.uint8)).save(png_path)
    row = morphology_row_for_image(png_path)
    # all NaN (cheap proxy yields NaN until offline CellViT pass writes sidecar)
    for v in row.values():
        assert np.isnan(v)
```

- [ ] **Step 2: Run test, expect failure**

Run: `pytest tests/test_a4_metrics.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `metrics.py`**

```python
# src/a4_uni_probe/metrics.py
"""Per-image morphology metrics for generated H&E (CellViT sidecar based)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.a4_uni_probe.labels import (
    MORPHOLOGY_ATTR_NAMES,
    compute_morphology_attributes_from_cellvit,
)


def morphology_row_for_image(png_path: Path) -> dict[str, float]:
    """Return per-image morphology metrics.

    Reads `<png_path>.json` (CellViT sidecar) when present.
    If absent, returns NaN for every attribute — the Stage 2/3 pipeline
    runs CellViT offline after generation and re-runs the figures step.
    """
    sidecar = png_path.with_suffix(png_path.suffix + ".json")
    if sidecar.is_file():
        return compute_morphology_attributes_from_cellvit(sidecar)
    return {name: float("nan") for name in MORPHOLOGY_ATTR_NAMES}
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_a4_metrics.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/a4_uni_probe/metrics.py tests/test_a4_metrics.py
git commit -m "feat(a4): morphology metric reader with CellViT sidecar fallback"
```

> ⚠️ Sweep/null `metrics.csv` will contain NaNs until the offline CellViT pass on generated PNGs completes. After CellViT, rerun `python -m src.a4_uni_probe.main figures` (Task 11) to recompute metrics from sidecars. Document this in `src/a4_uni_probe/README.md`.

---

## Task 10: Stage 3 null runner

**Files:**
- Modify: `src/a4_uni_probe/edit.py` (append `run_null`)

- [ ] **Step 1: Append `run_null`**

```python
# Append to src/a4_uni_probe/edit.py

def run_null(args: argparse.Namespace) -> None:
    out_dir = ensure_directory(args.out_dir)
    null_root = ensure_directory(out_dir / "null")
    probe_csv = out_dir / "probe_results.csv"
    labels_npz = out_dir / "labels.npz"
    features_npz = out_dir / "features.npz"
    if not probe_csv.is_file():
        raise FileNotFoundError(f"run `probe` first: {probe_csv} missing")

    attrs = _select_sweep_attrs(probe_csv, args.top_k_attrs)
    print(f"[a4] null attrs: {attrs}")
    features = np.load(features_npz, allow_pickle=True)
    tile_ids_all = list(features["tile_ids"])
    X_uni = features["X_uni"]

    for attr in attrs:
        attr_dir = ensure_directory(null_root / attr)
        w = np.load(out_dir / "probe_directions" / f"{attr}_uni_direction.npy")
        w_unit = w / max(np.linalg.norm(w), 1e-12)
        w_rand = random_unit_direction(
            d=w.size,
            rng=np.random.default_rng(args.seed + (hash(attr) % 100) + 7),
        )
        np.save(attr_dir / "w_targeted.npy", w_unit)
        np.save(attr_dir / "w_random.npy", w_rand)

        chosen_tiles = _select_sweep_tiles(labels_npz, attr, args.k_tiles, args.seed)
        rows: list[dict[str, object]] = []
        for tile_id in chosen_tiles:
            idx = tile_ids_all.index(tile_id)
            uni = X_uni[idx]
            conditions = {
                "targeted": null_uni(uni, w_unit),
                "random": null_uni(uni, w_rand),
            }
            for cond_name, uni_edit in conditions.items():
                out_path = attr_dir / tile_id / f"{cond_name}.png"
                generate_with_uni_override(
                    GenSpec(tile_id=tile_id, uni=uni_edit, out_path=out_path),
                    checkpoint_dir=args.checkpoint_dir,
                    config_path=args.config_path,
                    exp_channels_dir=Path("data/orion-crc33/exp_channels"),
                    num_steps=args.num_steps,
                    guidance_scale=args.guidance_scale,
                    seed=args.seed,
                )
                morpho = morphology_row_for_image(out_path)
                rows.append({
                    "tile_id": tile_id,
                    "condition": cond_name,
                    "target_attr": attr,
                    "target_value": float(morpho[attr]),
                    **{f"morpho.{k}": float(v) for k, v in morpho.items()},
                })
        _write_sweep_csv(attr_dir / "metrics.csv", rows)
        _summarize_null(attr_dir / "metrics.csv", attr_dir / "null_comparison.json", attr)


def _summarize_null(metrics_csv: Path, out_json: Path, attr: str) -> None:
    import csv
    rows: dict[str, list[float]] = {"targeted": [], "random": []}
    with metrics_csv.open() as fh:
        for row in csv.DictReader(fh):
            rows[row["condition"]].append(float(row["target_value"]))
    summary: dict[str, object] = {"attr": attr}
    for cond, values in rows.items():
        if not values:
            continue
        arr = np.array(values, dtype=np.float64)
        arr = arr[~np.isnan(arr)]
        summary[cond] = {
            "n": int(arr.size),
            "mean": float(arr.mean()) if arr.size else float("nan"),
            "std": float(arr.std()) if arr.size else float("nan"),
        }
    # Paired test: targeted lower (degraded) than random?
    if "targeted" in summary and "random" in summary and summary["targeted"]["n"] == summary["random"]["n"]:
        # Reconstruct paired arrays
        with metrics_csv.open() as fh:
            pairs: dict[str, dict[str, float]] = {}
            for row in csv.DictReader(fh):
                pairs.setdefault(row["tile_id"], {})[row["condition"]] = float(row["target_value"])
        diffs = [pairs[t]["targeted"] - pairs[t]["random"] for t in pairs if "targeted" in pairs[t] and "random" in pairs[t]]
        arr = np.array(diffs, dtype=np.float64)
        arr = arr[~np.isnan(arr)]
        rng = np.random.default_rng(0)
        boot = np.array([rng.choice(arr, size=arr.size, replace=True).mean() for _ in range(2000)])
        summary["paired_diff_mean"] = float(arr.mean()) if arr.size else float("nan")
        summary["paired_diff_ci95"] = [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]
        summary["pass_criterion_met"] = bool(summary["paired_diff_ci95"][0] * summary["paired_diff_ci95"][1] > 0)
    write_json(out_json, summary)
```

- [ ] **Step 2: Smoke-run on K=2 tiles, 1 attribute**

```bash
python -m src.a4_uni_probe.main null \
  --k-tiles 2 --top-k-attrs 1 --num-steps 10
```

Expected: `src/a4_uni_probe/out/null/<attr>/` populated.

- [ ] **Step 3: Commit**

```bash
git add src/a4_uni_probe/edit.py
git commit -m "feat(a4): Stage 3 subspace-null runner + paired comparison"
```

---

## Task 11: Figures

**Files:**
- Create: `src/a4_uni_probe/figures.py`

Three panels, one PNG each. Follow palette + axis conventions from `tools/color_constants.py` and existing `paper_figures/` modules.

- [ ] **Step 1: Inspect existing figure style**

```bash
head -60 src/paper_figures/fig_si_a1_a2_unified.py
head -40 tools/color_constants.py
```

Note tick fonts, color choices, panel-label conventions.

- [ ] **Step 2: Implement `figures.py`**

```python
# src/a4_uni_probe/figures.py
"""Figure panels A/B/C for the UNI semantic-ablation paper."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

from src._tasklib.io import ensure_directory


def render_panel_a(out_dir: Path) -> Path:
    """Bar chart: per-attribute R²(UNI) vs R²(TME)."""
    csv_path = Path(out_dir) / "probe_results.csv"
    rows_uni: dict[str, dict[str, float]] = {}
    rows_tme: dict[str, dict[str, float]] = {}
    with csv_path.open() as fh:
        for row in csv.DictReader(fh):
            entry = {"r2_mean": float(row["r2_mean"]), "r2_std": float(row["r2_std"])}
            (rows_uni if row["feature_set"] == "uni" else rows_tme)[row["attr"]] = entry
    attrs = sorted(rows_uni.keys(), key=lambda a: rows_uni[a]["r2_mean"] - rows_tme.get(a, {"r2_mean": 0})["r2_mean"], reverse=True)
    uni_means = [rows_uni[a]["r2_mean"] for a in attrs]
    uni_stds = [rows_uni[a]["r2_std"] for a in attrs]
    tme_means = [rows_tme.get(a, {"r2_mean": 0})["r2_mean"] for a in attrs]
    tme_stds = [rows_tme.get(a, {"r2_std": 0})["r2_std"] for a in attrs]
    x = np.arange(len(attrs))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - 0.2, uni_means, 0.4, yerr=uni_stds, label="UNI", color="#1f77b4")
    ax.bar(x + 0.2, tme_means, 0.4, yerr=tme_stds, label="TME baseline", color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels(attrs, rotation=45, ha="right")
    ax.set_ylabel("R² (5-fold spatial GroupKFold)")
    ax.set_title("Linear probe R² — UNI vs TME-pooled baseline")
    ax.legend()
    out_path = ensure_directory(Path(out_dir) / "figures") / "panel_a_probe_R2.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def render_panel_b(out_dir: Path) -> Path:
    """Slope plot: target metric vs alpha for each attr, targeted vs random."""
    sweep_root = Path(out_dir) / "sweep"
    attrs = sorted([p.name for p in sweep_root.iterdir() if p.is_dir()])
    fig, axes = plt.subplots(1, len(attrs), figsize=(4 * len(attrs), 4), sharey=False)
    if len(attrs) == 1:
        axes = [axes]
    for ax, attr in zip(axes, attrs):
        metrics_csv = sweep_root / attr / "metrics.csv"
        targeted: dict[float, list[float]] = {}
        random_: dict[float, list[float]] = {}
        with metrics_csv.open() as fh:
            for row in csv.DictReader(fh):
                a = float(row["alpha"]); v = float(row["target_value"])
                if np.isnan(v):
                    continue
                bucket = targeted if row["direction"] == "targeted" else random_
                bucket.setdefault(a, []).append(v)
        for bucket, color, label in [(targeted, "#1f77b4", "targeted"), (random_, "#bbbbbb", "random")]:
            alphas = sorted(bucket.keys())
            means = [np.mean(bucket[a]) for a in alphas]
            stds = [np.std(bucket[a]) for a in alphas]
            ax.errorbar(alphas, means, yerr=stds, marker="o", label=label, color=color)
        ax.set_xlabel("α"); ax.set_ylabel(attr)
        ax.set_title(attr); ax.legend()
    out_path = ensure_directory(Path(out_dir) / "figures") / "panel_b_sweep_slope.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def render_panel_c(out_dir: Path) -> Path:
    """Bar chart: targeted-null vs random-null mean target value, per attr."""
    null_root = Path(out_dir) / "null"
    attrs = sorted([p.name for p in null_root.iterdir() if p.is_dir()])
    targeted_means: list[float] = []; targeted_stds: list[float] = []
    random_means: list[float] = []; random_stds: list[float] = []
    for attr in attrs:
        summary = json.loads((null_root / attr / "null_comparison.json").read_text())
        targeted_means.append(summary.get("targeted", {}).get("mean", float("nan")))
        targeted_stds.append(summary.get("targeted", {}).get("std", 0.0))
        random_means.append(summary.get("random", {}).get("mean", float("nan")))
        random_stds.append(summary.get("random", {}).get("std", 0.0))
    x = np.arange(len(attrs))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - 0.2, targeted_means, 0.4, yerr=targeted_stds, label="targeted null", color="#d62728")
    ax.bar(x + 0.2, random_means, 0.4, yerr=random_stds, label="random null", color="#bbbbbb")
    ax.set_xticks(x); ax.set_xticklabels(attrs, rotation=45, ha="right")
    ax.set_ylabel("target metric after null"); ax.legend()
    ax.set_title("Subspace nulling: targeted vs random")
    out_path = ensure_directory(Path(out_dir) / "figures") / "panel_c_null_drop.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def render_all(out_dir: Path) -> list[Path]:
    paths = []
    if (Path(out_dir) / "probe_results.csv").is_file():
        paths.append(render_panel_a(out_dir))
    if (Path(out_dir) / "sweep").is_dir():
        paths.append(render_panel_b(out_dir))
    if (Path(out_dir) / "null").is_dir():
        paths.append(render_panel_c(out_dir))
    return paths
```

- [ ] **Step 3: Smoke-render with whatever stage outputs exist**

```bash
python -m src.a4_uni_probe.main figures
ls src/a4_uni_probe/out/figures/
```

Expected: `panel_a_probe_R2.png` (always, after Task 5); `panel_b_*` after Task 8; `panel_c_*` after Task 10.

- [ ] **Step 4: Commit**

```bash
git add src/a4_uni_probe/figures.py
git commit -m "feat(a4): probe / sweep / null paper panels"
```

---

## Task 12: Full run + CellViT pass + final commit

**Operational (not code).**

- [ ] **Step 1: Full Stage 1**

```bash
python -m src.a4_uni_probe.main probe
```

Inspect `probe_results.csv`. Identify the top-4 morphology attributes (manually filter to `MORPHOLOGY_ATTR_NAMES` set; channel-derived attrs are probe-only).

- [ ] **Step 2: Full Stage 2 + 3 (generates PNGs without morphology readout)**

```bash
python -m src.a4_uni_probe.main sweep --k-tiles 50
python -m src.a4_uni_probe.main null  --k-tiles 50
```

- [ ] **Step 3: CellViT batch on generated PNGs**

Use the existing batch tools to flatten and export, run CellViT externally, import sidecars back:

```bash
python tools/cellvit/export_batch.py \
  --src-dir src/a4_uni_probe/out/sweep \
  --dest-dir src/a4_uni_probe/out/cellvit_sweep_export --flat
python tools/cellvit/export_batch.py \
  --src-dir src/a4_uni_probe/out/null \
  --dest-dir src/a4_uni_probe/out/cellvit_null_export --flat
# ... run CellViT externally ...
python tools/cellvit/import_results.py \
  --cellvit-output ... --target-cache src/a4_uni_probe/out/sweep
python tools/cellvit/import_results.py \
  --cellvit-output ... --target-cache src/a4_uni_probe/out/null
```

Confirm sidecars present:

```bash
find src/a4_uni_probe/out/sweep -name "*.png.json" | head -3
find src/a4_uni_probe/out/null  -name "*.png.json" | head -3
```

- [ ] **Step 4: Re-summarize and render figures**

The summary functions read sidecars indirectly via metrics.csv. Easiest is to re-run the generation step with `--num-steps 0`-style early-exit OR add a `recompute-metrics` subcommand. Acceptable shortcut: write a one-off script that re-walks PNGs, recomputes `metrics.csv`, and re-runs `_summarize_slopes` / `_summarize_null`.

```bash
python - <<'PY'
import csv
from pathlib import Path
from src.a4_uni_probe.metrics import morphology_row_for_image
from src.a4_uni_probe.edit import _summarize_slopes, _summarize_null

out = Path("src/a4_uni_probe/out")
for stage, summarizer in (("sweep", _summarize_slopes), ("null", _summarize_null)):
    for attr_dir in (out / stage).iterdir():
        if not attr_dir.is_dir(): continue
        rows = []
        # walk PNGs and re-derive metric rows
        for png in attr_dir.rglob("*.png"):
            morpho = morphology_row_for_image(png)
            parts = png.relative_to(attr_dir).parts
            # Expected layout:
            #   sweep/<attr>/<tile>/<direction>/alpha_+1.00.png
            #   null/<attr>/<tile>/<cond>.png
            row = {"target_attr": attr_dir.name, "target_value": morpho[attr_dir.name]}
            if stage == "sweep":
                tile_id, direction, fname = parts[0], parts[1], parts[2]
                alpha = float(fname.replace("alpha_", "").replace(".png", ""))
                row.update({"tile_id": tile_id, "direction": direction, "alpha": alpha})
            else:
                tile_id, fname = parts[0], parts[1]
                cond = fname.replace(".png", "")
                row.update({"tile_id": tile_id, "condition": cond})
            row.update({f"morpho.{k}": v for k, v in morpho.items()})
            rows.append(row)
        if not rows: continue
        metrics_csv = attr_dir / "metrics.csv"
        fieldnames = sorted({k for r in rows for k in r.keys()})
        with metrics_csv.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames); w.writeheader(); w.writerows(rows)
        summarizer(metrics_csv, attr_dir / ("slope_summary.json" if stage == "sweep" else "null_comparison.json"), attr_dir.name)
        print(f"resummarized {stage}/{attr_dir.name}: {len(rows)} rows")
PY

python -m src.a4_uni_probe.main figures
```

- [ ] **Step 5: Commit results**

Commit the generated CSV/JSON summaries and figure PNGs (or, if results are large, only the summary JSONs + figures and gitignore raw PNGs).

```bash
git add src/a4_uni_probe/out/probe_results.{csv,json} \
        src/a4_uni_probe/out/sweep/**/slope_summary.json \
        src/a4_uni_probe/out/null/**/null_comparison.json \
        src/a4_uni_probe/out/figures/
git commit -m "feat(a4): full probe+sweep+null run with CellViT morphology"
```

---

## Self-Review Notes

- **Spec coverage**
  - Stage 1 → Tasks 2/3/5.
  - Stage 2 → Tasks 6/8.
  - Stage 3 → Tasks 6/10.
  - Outputs/figures → Task 11.
  - CellViT integration → Tasks 4/12 (offline pass + sidecar reader).
  - Terminology / paper claim — documented in spec, not in plan code.

- **Type consistency**
  - `ProbeFitResult.coef: np.ndarray` matches `w` consumed by `sweep_uni` / `null_uni`.
  - `morphology_row_for_image` always returns `MORPHOLOGY_ATTR_NAMES` keys; consumers in `run_sweep` / `run_null` use `morpho[attr]` (always present, possibly NaN).
  - `_select_sweep_attrs` returns morphology attrs only — guards Stage 2/3 from trying to read channel-derived attrs (unmeasurable on generated H&E).

- **Open risks documented**
  - Channel filename layout (Tasks 2 + 3 callouts).
  - Inference override signature must adapt to real `run_tile` (Task 7 callout).
  - CellViT sidecars produced offline; Stage 12 re-summarization required after CellViT pass.

---

## Execution Handoff

Plan saved to `docs/superpowers/plans/2026-05-12-uni-probe-semantic-ablation.md`. Two execution options:

1. **Subagent-Driven (recommended)** — fresh subagent per task, two-stage review between tasks. Aligns with the project's Claude/Codex role split: Claude reviews after each Codex task.
2. **Inline Execution** — Codex executes the full plan in one batch via `codex:codex-rescue`, with checkpoint review at task boundaries.

Which approach?
