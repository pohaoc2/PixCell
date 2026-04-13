# Parameter-to-Phenotype Mapping Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Map simulation parameter combinations to spatial TME phenotypes in real CRC by (1) clustering CRC patient H&E tiles into regional archetypes, (2) matching simulation outputs to those archetypes via TME-only inference in UNI feature space, and (3) visualizing matched pairs with style-conditioned inference from a different patient.

**Architecture:** Four sequential stages — archetype discovery (k-means on CRC UNI embeddings), batch TME-only inference for N simulation outputs, archetype matching + coverage analysis, style-conditioned visualization per matched archetype. Simulation outputs are assumed pre-generated as PNG files in the standard PixCell sim_channels layout.

**Tech Stack:** scikit-learn (KMeans, silhouette_score), umap-learn, numpy, matplotlib, Pillow, PyTorch (UNI encoder via existing stage1/stage3 scripts), existing PixCell inference pipeline (stage3_inference.py, stage1_extract_features.py).

**Sampling note:** With ~15 simulation parameters, use Latin Hypercube Sampling or Sobol sequences for N≥2048. Grid search is infeasible at 15D. This plan assumes simulation outputs are already available at a user-specified directory.

---

## Simulation Output Prerequisites (external — not part of PixCell)

Generate N simulation outputs (ARCADE or PhysiCell) **before** running Tasks 5–9. Tasks 1–4 (archetype discovery) can run in parallel while simulations are being generated.

**Where to put outputs:** `data/sim_outputs/` (pass as `--sim-channels-dir` to all Stage 4 scripts)

**Required directory layout:**
```
data/sim_outputs/
├── cell_mask/              {sim_id}.png   256×256 binary (pixel values 0 or 255) — REQUIRED
├── cell_type_healthy/      {sim_id}.png   256×256 binary
├── cell_type_cancer/       {sim_id}.png   256×256 binary
├── cell_type_immune/       {sim_id}.png   256×256 binary
├── cell_state_prolif/      {sim_id}.png   256×256 binary
├── cell_state_nonprolif/   {sim_id}.png   256×256 binary
├── cell_state_dead/        {sim_id}.png   256×256 binary
├── oxygen/                 {sim_id}.png   256×256 grayscale float→[0,255], or {sim_id}.npy float32
├── glucose/                {sim_id}.png   256×256 grayscale float→[0,255], or {sim_id}.npy float32
└── vasculature/            {sim_id}.png   256×256 grayscale float→[0,255]  (optional)
```

**Naming:** sim_ids must be consistent across all channel subdirs and match the row order of `param_vectors.npy`. Recommended: zero-padded integers — `sim_0000.png`, `sim_0001.png`, ...

**Parameter vectors:** Save a `[N, P]` float32 numpy array to `data/sim_outputs/param_vectors.npy` (one row per sim_id, same order). This is required by `run_figures.py --param-vectors`. Also save parameter names to `data/sim_outputs/param_names.txt` (one name per line).

**Sobol sampling (recommended):** Use `scipy.stats.qmc.Sobol` with N=2048 (power of 2, better than 1024 for 15 parameters). Example:
```python
from scipy.stats.qmc import Sobol
import numpy as np
sampler = Sobol(d=15, scramble=True, seed=42)
unit_samples = sampler.random(2048)          # [2048, 15] in [0,1]
# scale to your parameter bounds, then run simulations
```

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `tools/stage4/__init__.py` | Create | Package marker |
| `tools/stage4/archetype_discovery.py` | Create | Load UNI embeddings, sweep K, fit k-means, select medoids |
| `tools/stage4/matching.py` | Create | Assign G_N to archetypes, find best params, coverage report |
| `tools/stage4/style_selection.py` | Create | Select nearest style tile per archetype from other patients |
| `tools/stage4/figures.py` | Create | UMAP plots, side-by-side panels, parameter space plot |
| `tools/stage4/run_archetype_discovery.py` | Create | CLI: run archetype discovery on target patient, save results |
| `tools/stage4/run_sim_inference.py` | Create | CLI: batch TME-only inference for N simulation outputs |
| `tools/stage4/run_matching.py` | Create | CLI: load G_N UNI embeddings, run matching, save JSON report |
| `tools/stage4/run_style_inference.py` | Create | CLI: style-conditioned inference for each (archetype, best_param) |
| `tools/stage4/run_figures.py` | Create | CLI: generate all 4 figures from saved artifacts |
| `tests/test_archetype_discovery.py` | Create | Unit tests for archetype_discovery.py |
| `tests/test_matching.py` | Create | Unit tests for matching.py |

---

## Task 1: Archetype Discovery Utilities

**Files:**
- Create: `tools/stage4/__init__.py`
- Create: `tools/stage4/archetype_discovery.py`
- Test: `tests/test_archetype_discovery.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_archetype_discovery.py
from __future__ import annotations
import numpy as np
import pytest
from pathlib import Path
import tempfile


def _write_fake_embeddings(tmp_dir: Path, n: int = 20, dim: int = 1536) -> list[str]:
    """Write n fake _uni.npy files, return tile_ids."""
    np.random.seed(0)
    tile_ids = [f"{i * 256}_{j * 256}" for i, j in zip(range(n), range(n))]
    for tid in tile_ids:
        np.save(tmp_dir / f"{tid}_uni.npy", np.random.randn(dim).astype(np.float32))
    return tile_ids


def test_load_patient_embeddings_shape():
    from tools.stage4.archetype_discovery import load_patient_embeddings
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        tile_ids = _write_fake_embeddings(tmp, n=10)
        embs, ids = load_patient_embeddings(tmp)
        assert embs.shape == (10, 1536)
        assert len(ids) == 10


def test_load_patient_embeddings_missing_raises():
    from tools.stage4.archetype_discovery import load_patient_embeddings
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(FileNotFoundError):
            load_patient_embeddings(Path(tmp))


def test_sweep_k_returns_scores_for_each_k():
    from tools.stage4.archetype_discovery import sweep_k
    np.random.seed(42)
    embeddings = np.random.randn(50, 8).astype(np.float32)
    scores = sweep_k(embeddings, k_range=range(2, 5))
    assert set(scores.keys()) == {2, 3, 4}
    assert all(isinstance(v, float) for v in scores.values())


def test_fit_archetypes_shapes():
    from tools.stage4.archetype_discovery import fit_archetypes
    np.random.seed(0)
    embeddings = np.random.randn(30, 8).astype(np.float32)
    result = fit_archetypes(embeddings, k=3)
    assert result["centroids"].shape == (3, 8)
    assert result["labels"].shape == (30,)
    assert len(result["medoid_indices"]) == 3
    assert all(0 <= idx < 30 for idx in result["medoid_indices"])


def test_fit_archetypes_medoids_nearest_to_centroids():
    from tools.stage4.archetype_discovery import fit_archetypes
    np.random.seed(1)
    # 3 tight clusters, well separated
    c1 = np.random.randn(10, 4) + np.array([10, 0, 0, 0])
    c2 = np.random.randn(10, 4) + np.array([0, 10, 0, 0])
    c3 = np.random.randn(10, 4) + np.array([0, 0, 10, 0])
    embeddings = np.vstack([c1, c2, c3]).astype(np.float32)
    result = fit_archetypes(embeddings, k=3)
    # Each medoid should belong to its own cluster
    for k, idx in enumerate(result["medoid_indices"]):
        assert result["labels"][idx] == k
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_archetype_discovery.py -v 2>&1 | head -30
```
Expected: `ModuleNotFoundError` or `ImportError` — `tools.stage4.archetype_discovery` does not exist yet.

- [ ] **Step 3: Create package marker**

```python
# tools/stage4/__init__.py
```
(empty file)

- [ ] **Step 4: Implement archetype_discovery.py**

```python
# tools/stage4/archetype_discovery.py
"""
Archetype discovery: cluster CRC patient UNI embeddings into K regional archetypes.

Typical usage:
    from tools.stage4.archetype_discovery import load_patient_embeddings, sweep_k, fit_archetypes

    embs, tile_ids = load_patient_embeddings(Path("data/orion-crc33/features"))
    scores = sweep_k(embs)          # {k: silhouette_score}
    result = fit_archetypes(embs, k=5)
    # result["centroids"]       [K, 1536]
    # result["labels"]          [N]
    # result["medoid_indices"]  [K]  index into embs / tile_ids
"""
from __future__ import annotations

import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def load_patient_embeddings(
    features_dir: Path | str,
) -> tuple[np.ndarray, list[str]]:
    """
    Load all UNI embeddings from features_dir.

    Scans for files matching *_uni.npy.
    Returns (embeddings [N, 1536], tile_ids [N]) sorted by tile_id.

    Raises FileNotFoundError if no embeddings are found.
    """
    features_dir = Path(features_dir)
    paths = sorted(features_dir.glob("*_uni.npy"))
    if not paths:
        raise FileNotFoundError(
            f"No *_uni.npy files found in {features_dir}"
        )
    tile_ids = [p.name[: -len("_uni.npy")] for p in paths]
    embeddings = np.stack([np.load(p) for p in paths]).astype(np.float32)
    return embeddings, tile_ids


def sweep_k(
    embeddings: np.ndarray,
    k_range: range = range(3, 8),
    seed: int = 42,
) -> dict[int, float]:
    """Return {k: silhouette_score} for each k in k_range."""
    scores: dict[int, float] = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels = km.fit_predict(embeddings)
        scores[k] = float(silhouette_score(embeddings, labels))
    return scores


def select_medoids(
    embeddings: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
) -> list[int]:
    """For each cluster k, return the index of the tile nearest (L2) to centroid_k."""
    medoids: list[int] = []
    for k in range(centroids.shape[0]):
        indices = np.where(labels == k)[0]
        dists = np.linalg.norm(embeddings[indices] - centroids[k], axis=1)
        medoids.append(int(indices[dists.argmin()]))
    return medoids


def fit_archetypes(
    embeddings: np.ndarray,
    k: int,
    seed: int = 42,
) -> dict:
    """
    Fit k-means on embeddings with k clusters.

    Returns dict with keys:
        centroids      np.ndarray [k, D]   cluster centroids
        labels         np.ndarray [N]      archetype index per tile
        medoid_indices list[int]  [k]      index into embeddings for each medoid
    """
    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    labels = km.fit_predict(embeddings)
    medoids = select_medoids(embeddings, labels, km.cluster_centers_)
    return {
        "centroids": km.cluster_centers_.astype(np.float32),
        "labels": labels,
        "medoid_indices": medoids,
    }
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_archetype_discovery.py -v
```
Expected: 5 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add tools/stage4/__init__.py tools/stage4/archetype_discovery.py tests/test_archetype_discovery.py
git commit -m "feat: add stage4 archetype discovery utilities"
```

---

## Task 2: Archetype Matching Utilities

**Files:**
- Create: `tools/stage4/matching.py`
- Test: `tests/test_matching.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_matching.py
from __future__ import annotations
import numpy as np
import pytest


def test_assign_to_archetype_basic():
    from tools.stage4.matching import assign_to_archetype
    # 2 centroids at [0,0] and [10,10]
    centroids = np.array([[0.0, 0.0], [10.0, 10.0]], dtype=np.float32)
    gn = np.array([[0.5, 0.5], [9.5, 9.5], [0.1, 0.2]], dtype=np.float32)
    assignments = assign_to_archetype(gn, centroids)
    assert list(assignments) == [0, 1, 0]


def test_assign_to_archetype_shape():
    from tools.stage4.matching import assign_to_archetype
    np.random.seed(0)
    centroids = np.random.randn(4, 16).astype(np.float32)
    gn = np.random.randn(100, 16).astype(np.float32)
    out = assign_to_archetype(gn, centroids)
    assert out.shape == (100,)
    assert out.min() >= 0 and out.max() <= 3


def test_find_best_params_returns_one_per_covered_archetype():
    from tools.stage4.matching import find_best_params
    centroids = np.array([[0.0, 0.0], [10.0, 10.0]], dtype=np.float32)
    gn = np.array([[0.2, 0.2], [0.5, 0.5], [9.8, 9.8]], dtype=np.float32)
    param_ids = ["p0", "p1", "p2"]
    best = find_best_params(gn, centroids, param_ids)
    # archetype 0: p0 is closer than p1
    assert best[0] == "p0"
    # archetype 1: p2 is the only one
    assert best[1] == "p2"


def test_coverage_report_uncovered():
    from tools.stage4.matching import coverage_report
    assignments = np.array([0, 0, 1, 1])
    report = coverage_report(assignments, k=3, param_ids=["a", "b", "c", "d"])
    assert report["counts"][0] == 2
    assert report["counts"][1] == 2
    assert report["counts"][2] == 0
    assert 2 in report["uncovered"]


def test_coverage_report_full_coverage():
    from tools.stage4.matching import coverage_report
    assignments = np.array([0, 1, 2, 0])
    report = coverage_report(assignments, k=3, param_ids=["a", "b", "c", "d"])
    assert report["uncovered"] == []
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_matching.py -v 2>&1 | head -20
```
Expected: `ImportError` — `tools.stage4.matching` does not exist yet.

- [ ] **Step 3: Implement matching.py**

```python
# tools/stage4/matching.py
"""
Archetype matching: assign simulation G_N outputs to CRC regional archetypes.

Typical usage:
    from tools.stage4.matching import assign_to_archetype, find_best_params, coverage_report

    assignments = assign_to_archetype(gn_embeddings, centroids)
    best = find_best_params(gn_embeddings, centroids, param_ids)
    report = coverage_report(assignments, k=len(centroids), param_ids=param_ids)
"""
from __future__ import annotations

import numpy as np


def assign_to_archetype(
    gn_embeddings: np.ndarray,
    centroids: np.ndarray,
) -> np.ndarray:
    """
    Assign each G_N to its nearest archetype centroid (L2 distance).

    Args:
        gn_embeddings: [N, D]
        centroids:     [K, D]
    Returns:
        assignments:   [N] int, archetype index per G_N
    """
    # [N, K]
    dists = np.linalg.norm(
        gn_embeddings[:, None, :] - centroids[None, :, :], axis=2
    )
    return dists.argmin(axis=1)


def find_best_params(
    gn_embeddings: np.ndarray,
    centroids: np.ndarray,
    param_ids: list[str],
) -> dict[int, str]:
    """
    For each archetype k, return the param_id whose G_N is nearest to centroid_k.

    Only archetypes with at least one assigned G_N are included.

    Returns: {archetype_k: param_id}
    """
    assignments = assign_to_archetype(gn_embeddings, centroids)
    result: dict[int, str] = {}
    for k in range(centroids.shape[0]):
        indices = np.where(assignments == k)[0]
        if len(indices) == 0:
            continue
        dists = np.linalg.norm(gn_embeddings[indices] - centroids[k], axis=1)
        result[k] = param_ids[int(indices[dists.argmin()])]
    return result


def coverage_report(
    assignments: np.ndarray,
    k: int,
    param_ids: list[str],
) -> dict:
    """
    Return per-archetype assignment counts and uncovered archetypes.

    Returns dict with keys:
        counts      {archetype_k: int}   number of G_N assigned to each archetype
        uncovered   list[int]            archetypes with zero assigned G_N
    """
    counts = {i: int((assignments == i).sum()) for i in range(k)}
    uncovered = [i for i, c in counts.items() if c == 0]
    return {"counts": counts, "uncovered": uncovered}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_matching.py -v
```
Expected: 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/stage4/matching.py tests/test_matching.py
git commit -m "feat: add archetype matching utilities"
```

---

## Task 3: Style Tile Selection

**Files:**
- Create: `tools/stage4/style_selection.py`

No separate test file — this function is a thin wrapper over `load_patient_embeddings`; correctness is validated in the CLI run (Task 5).

- [ ] **Step 1: Implement style_selection.py**

```python
# tools/stage4/style_selection.py
"""
Style tile selection: for each archetype, find the nearest tile from a different patient.

Typical usage:
    from tools.stage4.style_selection import collect_other_patient_embeddings, select_style_tiles

    pool = collect_other_patient_embeddings(
        data_base_dir=Path("data"),
        exclude_patient_dir=Path("data/orion-crc33"),
    )
    style = select_style_tiles(centroids, pool)
    # style[k] = {"patient_dir": "data/orion-crc05", "tile_id": "1024_2048"}
"""
from __future__ import annotations

import numpy as np
from pathlib import Path

from tools.stage4.archetype_discovery import load_patient_embeddings


def collect_other_patient_embeddings(
    data_base_dir: Path | str,
    exclude_patient_dir: Path | str,
) -> dict[str, tuple[np.ndarray, list[str]]]:
    """
    Scan data_base_dir for patient subdirectories (those containing a features/ subdir).
    Load UNI embeddings from each, excluding exclude_patient_dir.

    Returns: {patient_dir_str: (embeddings [N, D], tile_ids [N])}
    """
    data_base_dir = Path(data_base_dir)
    exclude = Path(exclude_patient_dir).resolve()
    pool: dict[str, tuple[np.ndarray, list[str]]] = {}

    for candidate in sorted(data_base_dir.iterdir()):
        if not candidate.is_dir():
            continue
        if candidate.resolve() == exclude:
            continue
        feat_dir = candidate / "features"
        if not feat_dir.is_dir():
            continue
        try:
            embs, ids = load_patient_embeddings(feat_dir)
            pool[str(candidate)] = (embs, ids)
        except FileNotFoundError:
            pass
    return pool


def select_style_tiles(
    centroids: np.ndarray,
    pool: dict[str, tuple[np.ndarray, list[str]]],
) -> dict[int, dict[str, str]]:
    """
    For each archetype k, find the (patient_dir, tile_id) whose UNI embedding
    is nearest (L2) to centroid_k across all patients in pool.

    Returns: {k: {"patient_dir": str, "tile_id": str}}
    """
    result: dict[int, dict[str, str]] = {}
    for k, centroid in enumerate(centroids):
        best_dist = float("inf")
        best_entry: dict[str, str] = {}
        for patient_dir, (embs, tile_ids) in pool.items():
            dists = np.linalg.norm(embs - centroid, axis=1)
            idx = int(dists.argmin())
            if dists[idx] < best_dist:
                best_dist = float(dists[idx])
                best_entry = {"patient_dir": patient_dir, "tile_id": tile_ids[idx]}
        result[k] = best_entry
    return result
```

- [ ] **Step 2: Commit**

```bash
git add tools/stage4/style_selection.py
git commit -m "feat: add style tile selection utility"
```

---

## Pre-Task 4: Check for Existing Pathologist Annotations (Manual)

Before running archetype discovery, search for region-level annotations for the target patient in the ORION-CRC dataset. If annotations exist, they should guide or validate K selection instead of relying solely on silhouette score.

- [ ] **Search for annotations**: Check the ORION-CRC GitHub repository and associated publications for pathologist annotations (tumor core, stroma, invasive margin, necrosis) for the target patient.
- [ ] **If annotations found**: Pass `--k` matching the number of annotated region types. After running discovery, compare cluster medoids visually against the annotation map to confirm alignment.
- [ ] **If no annotations found**: Proceed with automatic K selection (omit `--k`). Manually inspect cluster medoid tiles after running to assign biological labels.

---

## Task 4: Archetype Discovery CLI Script

**Files:**
- Create: `tools/stage4/run_archetype_discovery.py`

Runs on the target patient, sweeps K, saves centroids + medoid tile IDs + labels to a JSON output.

- [ ] **Step 1: Implement run_archetype_discovery.py**

```python
# tools/stage4/run_archetype_discovery.py
"""
CLI: run archetype discovery on a target CRC patient.

Usage:
    python tools/stage4/run_archetype_discovery.py \\
        --features-dir data/orion-crc33/features \\
        --output-dir   inference_output/archetypes \\
        [--k 5]           # if omitted, auto-selects by silhouette score

Outputs to --output-dir:
    archetypes.json   centroids, medoid_tile_ids, labels, silhouette scores
    umap_raw.npy      2D UMAP coordinates for all tiles (for Fig 1)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from tools.stage4.archetype_discovery import (
    fit_archetypes,
    load_patient_embeddings,
    sweep_k,
)


def _auto_select_k(scores: dict[int, float]) -> int:
    """Return k with highest silhouette score."""
    return max(scores, key=lambda k: scores[k])


def main() -> None:
    parser = argparse.ArgumentParser(description="Archetype discovery for CRC patient tiles")
    parser.add_argument("--features-dir", required=True, type=Path,
                        help="Patient features directory containing *_uni.npy files")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="Directory to write archetypes.json and umap_raw.npy")
    parser.add_argument("--k", type=int, default=None,
                        help="Number of archetypes. If omitted, auto-selects K=3..7 by silhouette score.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading UNI embeddings from {args.features_dir} ...")
    embeddings, tile_ids = load_patient_embeddings(args.features_dir)
    print(f"  Loaded {len(tile_ids)} tiles, shape {embeddings.shape}")

    if args.k is None:
        print("Sweeping K=3..7 ...")
        scores = sweep_k(embeddings, seed=args.seed)
        for k, s in sorted(scores.items()):
            print(f"  K={k}  silhouette={s:.4f}")
        selected_k = _auto_select_k(scores)
        print(f"Auto-selected K={selected_k}")
    else:
        scores = {}
        selected_k = args.k
        print(f"Using K={selected_k} (user-specified)")

    result = fit_archetypes(embeddings, k=selected_k, seed=args.seed)
    medoid_tile_ids = [tile_ids[i] for i in result["medoid_indices"]]

    # Compute UMAP for all tiles (saved for figure generation)
    try:
        import umap
        print("Computing UMAP ...")
        reducer = umap.UMAP(n_components=2, random_state=args.seed)
        umap_coords = reducer.fit_transform(embeddings)
        np.save(args.output_dir / "umap_raw.npy", umap_coords)
        print(f"  Saved umap_raw.npy  shape={umap_coords.shape}")
    except ImportError:
        print("umap-learn not installed — skipping UMAP (install with: pip install umap-learn)")
        umap_coords = None

    output = {
        "k": selected_k,
        "silhouette_scores": {str(k): float(v) for k, v in scores.items()},
        "tile_ids": tile_ids,
        "labels": result["labels"].tolist(),
        "medoid_tile_ids": medoid_tile_ids,
        "centroids_path": str(args.output_dir / "centroids.npy"),
    }
    np.save(args.output_dir / "centroids.npy", result["centroids"])
    with open(args.output_dir / "archetypes.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved to {args.output_dir}/")
    print(f"  archetypes.json  (K={selected_k}, {len(tile_ids)} tile labels)")
    print(f"  centroids.npy    shape={result['centroids'].shape}")
    for i, tid in enumerate(medoid_tile_ids):
        n = int((result['labels'] == i).sum())
        print(f"  Archetype {i}: medoid={tid}, n_tiles={n}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run a smoke test on local data**

```bash
python tools/stage4/run_archetype_discovery.py \
    --features-dir data/orion-crc33/features \
    --output-dir   inference_output/archetypes \
    --k 5
```
Expected: prints tile count, fits K=5, writes `inference_output/archetypes/archetypes.json` and `centroids.npy`. No errors.

- [ ] **Step 3: Commit**

```bash
git add tools/stage4/run_archetype_discovery.py
git commit -m "feat: add archetype discovery CLI script"
```

---

## Task 5: Batch TME-only Inference for Simulation Outputs

**Files:**
- Create: `tools/stage4/run_sim_inference.py`

Wraps the existing `stage3_inference.py` to run all N simulation outputs in TME-only mode (no `--reference-he`), producing one generated H&E PNG per simulation.

- [ ] **Step 1: Implement run_sim_inference.py**

```python
# tools/stage4/run_sim_inference.py
"""
CLI: batch TME-only inference for N simulation outputs.

Assumes simulation outputs follow the standard sim_channels layout:
    sim_channels_dir/
    ├── cell_mask/       {sim_id}.png   (required)
    ├── cell_type_healthy/{sim_id}.png
    ├── cell_type_cancer/ {sim_id}.png
    ├── cell_type_immune/ {sim_id}.png
    ├── cell_state_prolif/{sim_id}.png
    ├── cell_state_nonprolif/{sim_id}.png
    ├── cell_state_dead/  {sim_id}.png
    ├── oxygen/           {sim_id}.png  (optional)
    ├── glucose/          {sim_id}.png  (optional)
    └── vasculature/      {sim_id}.png  (optional)

For each sim_id found in cell_mask/, runs stage3_inference.py in TME-only mode
and saves generated_he.png to --output-dir/{sim_id}/generated_he.png.

Usage:
    python tools/stage4/run_sim_inference.py \\
        --config          configs/config_controlnet_exp.py \\
        --checkpoint-dir  checkpoints/pixcell_controlnet_exp/checkpoints/step_XXXXXXX \\
        --sim-channels-dir /path/to/sim_channels \\
        --output-dir      inference_output/sim_tme_only \\
        [--device cuda]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))


def _list_sim_ids(sim_channels_dir: Path) -> list[str]:
    """Return sorted sim_ids from the cell_mask subdirectory."""
    mask_dir = sim_channels_dir / "cell_mask"
    if not mask_dir.exists():
        # try plural alias
        mask_dir = sim_channels_dir / "cell_masks"
    if not mask_dir.exists():
        raise FileNotFoundError(f"cell_mask/ not found in {sim_channels_dir}")
    return sorted(p.stem for p in mask_dir.glob("*.png"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch TME-only inference for simulation outputs")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--sim-channels-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sim_ids = _list_sim_ids(args.sim_channels_dir)
    print(f"Found {len(sim_ids)} simulation outputs in {args.sim_channels_dir}")

    for i, sim_id in enumerate(sim_ids):
        out_path = args.output_dir / sim_id / "generated_he.png"
        if out_path.exists():
            print(f"  [{i+1}/{len(sim_ids)}] {sim_id} — skip (exists)")
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable, str(ROOT / "stage3_inference.py"),
            "--config", str(args.config),
            "--checkpoint-dir", str(args.checkpoint_dir),
            "--sim-channels-dir", str(args.sim_channels_dir),
            "--sim-id", sim_id,
            "--output", str(out_path),
            "--device", args.device,
            "--guidance-scale", str(args.guidance_scale),
            "--seed", str(args.seed),
            # no --reference-he → TME-only mode
        ]
        print(f"  [{i+1}/{len(sim_ids)}] {sim_id}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"    ERROR: {result.stderr[-500:]}")
        else:
            print(f"    OK → {out_path}")

    print(f"\nDone. Generated H&E saved under {args.output_dir}/")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add tools/stage4/run_sim_inference.py
git commit -m "feat: add batch TME-only inference script for simulation outputs"
```

---

## Task 6: Compute UNI Embeddings for Generated G_N

No new code needed — use existing `stage1_extract_features.py` on the generated H&E output directory.

- [ ] **Step 1: Extract UNI embeddings for all G_N**

Point `stage1_extract_features.py` at the generated H&E directory:

```bash
python stage1_extract_features.py \
    --image-dir  inference_output/sim_tme_only \
    --output-dir inference_output/sim_tme_only/features
```

But the generated files are under `{sim_id}/generated_he.png`. Stage 1 expects flat PNGs. Create a flat symlink directory first:

```bash
mkdir -p inference_output/sim_tme_only/he_flat
for d in inference_output/sim_tme_only/*/; do
    sim_id=$(basename "$d")
    src="$d/generated_he.png"
    if [ -f "$src" ]; then
        ln -sf "$(realpath $src)" "inference_output/sim_tme_only/he_flat/${sim_id}.png"
    fi
done

python stage1_extract_features.py \
    --image-dir  inference_output/sim_tme_only/he_flat \
    --output-dir inference_output/sim_tme_only/features
```

Expected: creates `inference_output/sim_tme_only/features/{sim_id}_uni.npy` for each sim_id.

- [ ] **Step 2: Verify one embedding**

```bash
python -c "
import numpy as np, pathlib
p = sorted(pathlib.Path('inference_output/sim_tme_only/features').glob('*_uni.npy'))[0]
emb = np.load(p)
print(p.name, emb.shape, emb.dtype)
"
```
Expected output: `{sim_id}_uni.npy (1536,) float32`

---

## Task 7: Archetype Matching CLI + Coverage Report

**Files:**
- Create: `tools/stage4/run_matching.py`

- [ ] **Step 1: Implement run_matching.py**

```python
# tools/stage4/run_matching.py
"""
CLI: assign simulation G_N outputs to CRC archetypes and report coverage.

Usage:
    python tools/stage4/run_matching.py \\
        --archetypes-dir  inference_output/archetypes \\
        --gn-features-dir inference_output/sim_tme_only/features \\
        --output-dir      inference_output/matching

Outputs:
    matching.json   per-G_N archetype assignment + best param per archetype + coverage report
    gn_umap.npy     UMAP coords for G_N (projected onto same space as archetypes if umap_raw.npy exists)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from tools.stage4.archetype_discovery import load_patient_embeddings
from tools.stage4.matching import assign_to_archetype, coverage_report, find_best_params


def main() -> None:
    parser = argparse.ArgumentParser(description="Archetype matching for simulation G_N outputs")
    parser.add_argument("--archetypes-dir", required=True, type=Path,
                        help="Output dir from run_archetype_discovery.py (contains archetypes.json + centroids.npy)")
    parser.add_argument("--gn-features-dir", required=True, type=Path,
                        help="Directory of {sim_id}_uni.npy files for generated G_N")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.archetypes_dir / "archetypes.json") as f:
        arch = json.load(f)
    centroids = np.load(args.archetypes_dir / "centroids.npy")
    k = arch["k"]

    print(f"Loading G_N embeddings from {args.gn_features_dir} ...")
    gn_embeddings, param_ids = load_patient_embeddings(args.gn_features_dir)
    print(f"  {len(param_ids)} simulation outputs")

    assignments = assign_to_archetype(gn_embeddings, centroids)
    best = find_best_params(gn_embeddings, centroids, param_ids)
    report = coverage_report(assignments, k=k, param_ids=param_ids)

    print("\nCoverage report:")
    for i in range(k):
        medoid = arch["medoid_tile_ids"][i]
        count = report["counts"][i]
        best_p = best.get(i, "NONE")
        print(f"  Archetype {i} (medoid={medoid}): {count} G_N assigned, best_param={best_p}")
    if report["uncovered"]:
        print(f"\n  UNCOVERED archetypes: {report['uncovered']}")
        print("  These TME contexts are not reproduced by the current simulation parameter space.")
    else:
        print("\n  All archetypes covered.")

    output = {
        "k": k,
        "param_ids": param_ids,
        "assignments": assignments.tolist(),
        "best_params": {str(k_): v for k_, v in best.items()},
        "coverage": {
            "counts": {str(k_): v for k_, v in report["counts"].items()},
            "uncovered": report["uncovered"],
        },
        "medoid_tile_ids": arch["medoid_tile_ids"],
    }
    with open(args.output_dir / "matching.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved matching.json to {args.output_dir}/")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run matching on local data (smoke test — use any available features)**

```bash
python tools/stage4/run_matching.py \
    --archetypes-dir  inference_output/archetypes \
    --gn-features-dir inference_output/sim_tme_only/features \
    --output-dir      inference_output/matching
```
Expected: prints per-archetype counts, writes `inference_output/matching/matching.json`. No errors.

- [ ] **Step 3: Commit**

```bash
git add tools/stage4/run_matching.py
git commit -m "feat: add archetype matching CLI script"
```

---

## Task 8: Style-Conditioned Inference Per Archetype

**Files:**
- Create: `tools/stage4/run_style_inference.py`

For each archetype k, finds the best-matched parameter combination (from matching.json), selects a style tile from a different patient, and runs style-conditioned Stage 3 inference.

- [ ] **Step 1: Implement run_style_inference.py**

```python
# tools/stage4/run_style_inference.py
"""
CLI: style-conditioned inference for each (archetype, best_param) pair.

Usage:
    python tools/stage4/run_style_inference.py \\
        --config          configs/config_controlnet_exp.py \\
        --checkpoint-dir  checkpoints/pixcell_controlnet_exp/checkpoints/step_XXXXXXX \\
        --matching-json   inference_output/matching/matching.json \\
        --archetypes-dir  inference_output/archetypes \\
        --sim-channels-dir /path/to/sim_channels \\
        --data-base-dir   data \\
        --target-patient-dir data/orion-crc33 \\
        --output-dir      inference_output/style_conditioned \\
        [--device cuda]

For each archetype k:
  1. Reads best_param from matching.json
  2. Selects style tile from different patient nearest to archetype centroid
  3. Runs stage3_inference.py with --reference-he pointing to that style tile
  4. Saves generated_he.png + style_tile.png + medoid_he.png for Fig 3
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from tools.stage4.style_selection import collect_other_patient_embeddings, select_style_tiles


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--matching-json", required=True, type=Path)
    parser.add_argument("--archetypes-dir", required=True, type=Path)
    parser.add_argument("--sim-channels-dir", required=True, type=Path)
    parser.add_argument("--data-base-dir", required=True, type=Path,
                        help="Parent dir containing all patient subdirs (e.g. data/)")
    parser.add_argument("--target-patient-dir", required=True, type=Path,
                        help="Patient dir used for archetype discovery — excluded from style pool")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.matching_json) as f:
        matching = json.load(f)
    centroids = np.load(args.archetypes_dir / "centroids.npy")
    k = matching["k"]
    best_params = {int(k_): v for k_, v in matching["best_params"].items()}
    medoid_tile_ids = matching["medoid_tile_ids"]

    print(f"Collecting style embeddings from {args.data_base_dir} (excluding {args.target_patient_dir}) ...")
    pool = collect_other_patient_embeddings(args.data_base_dir, args.target_patient_dir)
    print(f"  Found {len(pool)} other patients")
    if not pool:
        print("ERROR: no other patient directories found. Check --data-base-dir.")
        sys.exit(1)

    style_tiles = select_style_tiles(centroids, pool)

    for arch_k in range(k):
        if arch_k not in best_params:
            print(f"Archetype {arch_k}: no best param (uncovered) — skipping")
            continue

        sim_id = best_params[arch_k]
        style_info = style_tiles[arch_k]
        style_patient_dir = Path(style_info["patient_dir"])
        style_tile_id = style_info["tile_id"]
        style_he_path = style_patient_dir / "he" / f"{style_tile_id}.png"

        out_dir = args.output_dir / f"archetype_{arch_k}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "generated_he.png"

        print(f"\nArchetype {arch_k}:")
        print(f"  best_param={sim_id}")
        print(f"  style={style_patient_dir.name}/{style_tile_id}")
        print(f"  medoid={medoid_tile_ids[arch_k]}")

        if not style_he_path.exists():
            print(f"  WARNING: style H&E not found at {style_he_path} — skipping")
            continue

        # Copy medoid H&E for side-by-side Fig 3
        medoid_he_src = args.target_patient_dir / "he" / f"{medoid_tile_ids[arch_k]}.png"
        if medoid_he_src.exists():
            shutil.copy(medoid_he_src, out_dir / "medoid_he.png")
        shutil.copy(style_he_path, out_dir / "style_tile.png")

        cmd = [
            sys.executable, str(ROOT / "stage3_inference.py"),
            "--config", str(args.config),
            "--checkpoint-dir", str(args.checkpoint_dir),
            "--sim-channels-dir", str(args.sim_channels_dir),
            "--sim-id", sim_id,
            "--reference-he", str(style_he_path),
            "--output", str(out_path),
            "--device", args.device,
            "--guidance-scale", str(args.guidance_scale),
            "--seed", str(args.seed),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  ERROR: {result.stderr[-500:]}")
        else:
            print(f"  OK → {out_path}")

    print(f"\nDone. Results in {args.output_dir}/")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add tools/stage4/run_style_inference.py
git commit -m "feat: add style-conditioned inference script per archetype"
```

---

## Task 9: Figures

**Files:**
- Create: `tools/stage4/figures.py`
- Create: `tools/stage4/run_figures.py`

- [ ] **Step 1: Implement figures.py**

```python
# tools/stage4/figures.py
"""
Figure generation for parameter-to-phenotype analysis.

Figures:
  Fig 1 — UMAP of CRC patient UNI embeddings, colored by archetype, with medoid thumbnails.
  Fig 2 — UMAP overlay: CRC tiles (gray) + G_N points (colored by archetype assignment).
  Fig 3 — Side-by-side per archetype: real CRC medoid | style tile | style-conditioned G_N.
  Fig 4 — Parameter space (first 2 PCA dims) colored by archetype assignment.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


_ARCHETYPE_COLORS = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628"
]


def _color(k: int) -> str:
    return _ARCHETYPE_COLORS[k % len(_ARCHETYPE_COLORS)]


def save_fig1_archetype_umap(
    umap_coords: np.ndarray,       # [N, 2]
    labels: np.ndarray,            # [N]
    medoid_indices: list[int],
    tile_ids: list[str],
    he_dir: Path,
    output_path: Path,
    k: int,
    thumbnail_size: int = 48,
) -> None:
    """Fig 1: UMAP of CRC embeddings colored by cluster, with medoid thumbnails."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for i in range(k):
        mask = labels == i
        ax.scatter(
            umap_coords[mask, 0], umap_coords[mask, 1],
            s=2, alpha=0.4, color=_color(i), label=f"Archetype {i}",
        )
    # Medoid stars
    for i, idx in enumerate(medoid_indices):
        ax.scatter(
            umap_coords[idx, 0], umap_coords[idx, 1],
            s=100, marker="*", color=_color(i), edgecolors="black", linewidths=0.5, zorder=5,
        )
    ax.legend(markerscale=3, fontsize=8)
    ax.set_title("CRC Patient — UNI Embedding UMAP by Archetype")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Fig 1 → {output_path}")


def save_fig2_umap_overlay(
    crc_umap: np.ndarray,           # [N_crc, 2]
    gn_umap: np.ndarray,            # [N_gn, 2]
    gn_assignments: np.ndarray,     # [N_gn]
    output_path: Path,
    k: int,
) -> None:
    """Fig 2: CRC tiles (gray) overlaid with G_N colored by archetype."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(crc_umap[:, 0], crc_umap[:, 1], s=2, alpha=0.15, color="lightgray", label="CRC tiles")
    for i in range(k):
        mask = gn_assignments == i
        if not mask.any():
            continue
        ax.scatter(
            gn_umap[mask, 0], gn_umap[mask, 1],
            s=15, alpha=0.8, color=_color(i), label=f"G_N archetype {i}",
        )
    ax.legend(markerscale=2, fontsize=8)
    ax.set_title("Simulation G_N vs CRC Distribution (UNI Space)")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Fig 2 → {output_path}")


def save_fig3_side_by_side(
    style_results_dir: Path,
    output_path: Path,
    k: int,
) -> None:
    """Fig 3: per archetype — real CRC medoid | style tile | style-conditioned G_N."""
    fig, axes = plt.subplots(k, 3, figsize=(9, 3 * k))
    col_labels = ["CRC Medoid", "Style Reference", "Generated G_N"]
    filenames = ["medoid_he.png", "style_tile.png", "generated_he.png"]

    for i in range(k):
        arch_dir = style_results_dir / f"archetype_{i}"
        for j, (fname, col_lbl) in enumerate(zip(filenames, col_labels)):
            ax = axes[i, j] if k > 1 else axes[j]
            p = arch_dir / fname
            if p.exists():
                img = np.array(Image.open(p).convert("RGB"))
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, "missing", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(col_lbl, fontsize=10)
            if j == 0:
                ax.set_ylabel(f"Archetype {i}", fontsize=9, color=_color(i))

    fig.suptitle("Real CRC vs Style-Conditioned Generation per Archetype", fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Fig 3 → {output_path}")


def save_fig4_param_space(
    param_vectors: np.ndarray,     # [N, P]  raw parameter vectors
    assignments: np.ndarray,       # [N]
    param_names: list[str],
    output_path: Path,
    k: int,
) -> None:
    """Fig 4: parameter space (first 2 PCA dims) colored by archetype assignment."""
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    coords = pca.fit_transform(param_vectors)
    fig, ax = plt.subplots(figsize=(7, 6))
    for i in range(k):
        mask = assignments == i
        if not mask.any():
            continue
        ax.scatter(coords[mask, 0], coords[mask, 1], s=20, alpha=0.7, color=_color(i), label=f"Archetype {i}")
    ax.legend(fontsize=8)
    var = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({var[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({var[1]*100:.1f}% var)")
    ax.set_title("Simulation Parameter Space Colored by Archetype Assignment")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Fig 4 → {output_path}")
```

- [ ] **Step 2: Implement run_figures.py**

```python
# tools/stage4/run_figures.py
"""
CLI: generate all 4 figures from saved analysis artifacts.

Usage:
    python tools/stage4/run_figures.py \\
        --archetypes-dir       inference_output/archetypes \\
        --matching-json        inference_output/matching/matching.json \\
        --gn-features-dir      inference_output/sim_tme_only/features \\
        --style-results-dir    inference_output/style_conditioned \\
        --target-patient-dir   data/orion-crc33 \\
        --param-vectors        /path/to/param_vectors.npy \\
        --param-names          oxygen_rate glucose_rate prolif_rate ... \\
        --output-dir           inference_output/figures

--param-vectors: [N, P] float32 npy file of raw parameter values (one row per sim_id,
                 in the same order as sim_ids sorted from the cell_mask directory).
--param-names:   space-separated list of P parameter names.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from tools.stage4.archetype_discovery import load_patient_embeddings
from tools.stage4.figures import (
    save_fig1_archetype_umap,
    save_fig2_umap_overlay,
    save_fig3_side_by_side,
    save_fig4_param_space,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archetypes-dir", required=True, type=Path)
    parser.add_argument("--matching-json", required=True, type=Path)
    parser.add_argument("--gn-features-dir", required=True, type=Path)
    parser.add_argument("--style-results-dir", required=True, type=Path)
    parser.add_argument("--target-patient-dir", required=True, type=Path)
    parser.add_argument("--param-vectors", required=True, type=Path,
                        help="[N, P] .npy file of parameter values, one row per sim_id (sorted)")
    parser.add_argument("--param-names", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.matching_json) as f:
        matching = json.load(f)
    centroids = np.load(args.archetypes_dir / "centroids.npy")
    k = matching["k"]
    with open(args.archetypes_dir / "archetypes.json") as f:
        arch_json = json.load(f)
    labels_crc = np.array(arch_json["labels"])
    medoid_indices = arch_json["medoid_indices"]
    tile_ids_crc = arch_json["tile_ids"]
    assignments_gn = np.array(matching["assignments"])

    umap_crc = np.load(args.archetypes_dir / "umap_raw.npy")  # [N_crc, 2]

    # Fig 1
    save_fig1_archetype_umap(
        umap_coords=umap_crc,
        labels=labels_crc,
        medoid_indices=medoid_indices,
        tile_ids=tile_ids_crc,
        he_dir=args.target_patient_dir / "he",
        output_path=args.output_dir / "fig1_archetype_umap.png",
        k=k,
    )

    # Fig 2: joint UMAP of CRC + G_N embeddings
    print("Computing joint UMAP for Fig 2 ...")
    gn_embs, _ = load_patient_embeddings(args.gn_features_dir)
    crc_embs_fresh, _ = load_patient_embeddings(args.target_patient_dir / "features")
    try:
        import umap as umap_lib
        all_embs = np.vstack([crc_embs_fresh, gn_embs])
        reducer = umap_lib.UMAP(n_components=2, random_state=42)
        all_umap = reducer.fit_transform(all_embs)
        crc_umap_joint = all_umap[:len(crc_embs_fresh)]
        gn_umap_joint = all_umap[len(crc_embs_fresh):]
    except ImportError:
        print("umap-learn not installed — using PCA for Fig 2")
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        all_embs = np.vstack([crc_embs_fresh, gn_embs])
        all_umap = pca.fit_transform(all_embs)
        crc_umap_joint = all_umap[:len(crc_embs_fresh)]
        gn_umap_joint = all_umap[len(crc_embs_fresh):]

    save_fig2_umap_overlay(
        crc_umap=crc_umap_joint,
        gn_umap=gn_umap_joint,
        gn_assignments=assignments_gn,
        output_path=args.output_dir / "fig2_umap_overlay.png",
        k=k,
    )

    # Fig 3
    save_fig3_side_by_side(
        style_results_dir=args.style_results_dir,
        output_path=args.output_dir / "fig3_side_by_side.png",
        k=k,
    )

    # Fig 4
    param_vectors = np.load(args.param_vectors)
    save_fig4_param_space(
        param_vectors=param_vectors,
        assignments=assignments_gn,
        param_names=args.param_names,
        output_path=args.output_dir / "fig4_param_space.png",
        k=k,
    )

    print(f"\nAll figures saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Commit**

```bash
git add tools/stage4/figures.py tools/stage4/run_figures.py
git commit -m "feat: add stage4 figure generation scripts"
```

---

## Task 10: End-to-End Smoke Test

- [ ] **Step 1: Run the full test suite**

```bash
pytest tests/test_archetype_discovery.py tests/test_matching.py -v
```
Expected: all 10 tests PASS.

- [ ] **Step 2: Verify all stage4 scripts are importable**

```bash
python -c "
from tools.stage4.archetype_discovery import load_patient_embeddings, sweep_k, fit_archetypes
from tools.stage4.matching import assign_to_archetype, find_best_params, coverage_report
from tools.stage4.style_selection import collect_other_patient_embeddings, select_style_tiles
from tools.stage4.figures import save_fig1_archetype_umap, save_fig2_umap_overlay, save_fig3_side_by_side, save_fig4_param_space
print('All imports OK')
"
```
Expected: `All imports OK`

- [ ] **Step 3: Commit**

```bash
git commit --allow-empty -m "chore: verify stage4 pipeline complete"
```

---

## Pipeline Run Order (Reference)

```
# 1. Discover archetypes for target patient
python tools/stage4/run_archetype_discovery.py \
    --features-dir data/<target_patient>/features \
    --output-dir   inference_output/archetypes

# 2. Batch TME-only inference for all N simulations
python tools/stage4/run_sim_inference.py \
    --config configs/config_controlnet_exp.py \
    --checkpoint-dir checkpoints/pixcell_controlnet_exp/checkpoints/step_XXXXXXX \
    --sim-channels-dir /path/to/sim_channels \
    --output-dir inference_output/sim_tme_only

# 3. Extract UNI embeddings for generated G_N
mkdir -p inference_output/sim_tme_only/he_flat
for d in inference_output/sim_tme_only/*/; do
    sim_id=$(basename "$d")
    ln -sf "$(realpath ${d}generated_he.png)" "inference_output/sim_tme_only/he_flat/${sim_id}.png"
done
python stage1_extract_features.py \
    --image-dir  inference_output/sim_tme_only/he_flat \
    --output-dir inference_output/sim_tme_only/features

# 4. Match G_N to archetypes
python tools/stage4/run_matching.py \
    --archetypes-dir  inference_output/archetypes \
    --gn-features-dir inference_output/sim_tme_only/features \
    --output-dir      inference_output/matching

# 5. Style-conditioned inference per archetype
python tools/stage4/run_style_inference.py \
    --config configs/config_controlnet_exp.py \
    --checkpoint-dir checkpoints/pixcell_controlnet_exp/checkpoints/step_XXXXXXX \
    --matching-json  inference_output/matching/matching.json \
    --archetypes-dir inference_output/archetypes \
    --sim-channels-dir /path/to/sim_channels \
    --data-base-dir  data \
    --target-patient-dir data/<target_patient> \
    --output-dir     inference_output/style_conditioned

# 6. Generate all figures
python tools/stage4/run_figures.py \
    --archetypes-dir    inference_output/archetypes \
    --matching-json     inference_output/matching/matching.json \
    --gn-features-dir   inference_output/sim_tme_only/features \
    --style-results-dir inference_output/style_conditioned \
    --target-patient-dir data/<target_patient> \
    --param-vectors     /path/to/param_vectors.npy \
    --param-names       oxygen_rate glucose_rate prolif_rate ... \
    --output-dir        inference_output/figures
```
