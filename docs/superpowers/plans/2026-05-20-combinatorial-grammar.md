# Combinatorial Grammar Fig 09 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Per `CLAUDE.md`, Claude plans/reviews only — actual file edits and shell execution are delegated to Codex via `codex:codex-rescue`.

**Goal:** Replace fig 09's bespoke a3 metric set + single-scalar interaction heatmap with the a4 metric schema, a 3-seed variance band, and a 3-way ANOVA variance-partition headline panel.

**Architecture:** New pure-numpy `variance_partition.py` module. `_compute_signature` in a3 main.py delegates to a4 helpers (`appearance_row_for_image`, `compute_morphology_attributes_from_cellvit`). Multi-seed renders go to parallel `generated_s{seed}/` dirs and roll into one tall `morphological_signatures.csv` with a `seed` column. Main fig 09 rewritten as variance-partition bars + one anchor sweep grid. Existing `additive_model_residuals.csv` and `interaction_heatmap.png` kept untouched on disk; new outputs side-by-side.

**Tech Stack:** Python 3.12, numpy, matplotlib, PIL, scikit-image (transitively via a4), pytest. CellViT sidecar pipeline already documented in `CLAUDE.md`.

**Spec:** `docs/superpowers/specs/2026-05-20-combinatorial-grammar-design.md`

---

## File map

| File | State | Responsibility |
|---|---|---|
| `src/a3_combinatorial_sweep/variance_partition.py` | new | Pure-numpy 3-way ANOVA. Public API: `variance_partition(rows, metrics)`. |
| `src/a3_combinatorial_sweep/main.py` | modify | Swap `MORPHOLOGY_METRICS` to a4 schema. `_compute_signature` delegates to a4 helpers. `_iter_signature_rows` walks all `generated*/` and emits `seed` column. `run_generate_worker` accepts repeated `--seed`. Summary worker also writes `variance_partition.csv`. |
| `src/paper_figures/fig_combinatorial_grammar.py` | rewrite | Two-panel main fig: Panel A = variance-partition stacked bars, Panel B = 1 anchor 3×9 sweep grid. |
| `src/paper_figures/fig_combinatorial_grammar_si.py` | extend | Keep existing 2×2 raw-grid section. Add: residual small-multiples, seed CI table, anchor sensitivity bar chart. |
| `src/paper_figures/fig_combinatorial_grammar_panels/_shared.py` | modify | No code changes needed if `MORPHOLOGY_METRICS` import stays — but `residual_lookup` and `compute_anchor_sweep_magnitude` exercised against new column names. Add new helper `load_variance_partition` for SI builder. |
| `tests/test_fig_combinatorial_grammar.py` | extend | Update fixtures to a4 metric names. Add new test class for variance partition (synthetic additive data → interaction shares ≈ 0). |
| `tests/test_variance_partition.py` | new | Dedicated unit tests for the new module (additive, pure-interaction, anchor-only synthetics). |

---

## Task 1: Variance partition module — failing tests

**Files:**
- Create: `tests/test_variance_partition.py`

- [ ] **Step 1: Write the failing tests**

```python
"""Unit tests for src.a3_combinatorial_sweep.variance_partition."""
from __future__ import annotations

import pytest

from src.a3_combinatorial_sweep.variance_partition import variance_partition


def _row(anchor: str, state: str, ox: str, gluc: str, seed: int, value: float) -> dict:
    return {
        "anchor_id": anchor,
        "cell_state": state,
        "oxygen_label": ox,
        "glucose_label": gluc,
        "seed": seed,
        "metric": value,
    }


def _build_rows(value_fn) -> list[dict]:
    anchors = ("a1", "a2", "a3")
    states = ("prolif", "nonprolif", "dead")
    levels = ("low", "mid", "high")
    seeds = (42, 43)
    rows = []
    for anchor in anchors:
        for state in states:
            for ox in levels:
                for gluc in levels:
                    for seed in seeds:
                        rows.append(_row(anchor, state, ox, gluc, seed, value_fn(anchor, state, ox, gluc, seed)))
    return rows


def test_pure_additive_grammar_has_zero_interaction():
    state_eff = {"prolif": 1.0, "nonprolif": 0.0, "dead": -1.0}
    ox_eff = {"low": -0.5, "mid": 0.0, "high": 0.5}
    gluc_eff = {"low": -0.2, "mid": 0.0, "high": 0.2}
    rows = _build_rows(lambda a, s, o, g, _seed: state_eff[s] + ox_eff[o] + gluc_eff[g])

    shares = variance_partition(rows, metrics=("metric",))

    s = shares["metric"]
    assert s["s_x_o"] + s["s_x_g"] + s["o_x_g"] + s["s_x_o_x_g"] < 1e-9
    assert s["state"] > 0.3
    assert s["o2"] > 0.05
    assert s["gluc"] > 0.0
    assert abs(sum(s.values()) - 1.0) < 1e-9


def test_pure_anchor_variance():
    anchor_eff = {"a1": 1.0, "a2": 0.0, "a3": -1.0}
    rows = _build_rows(lambda a, *_args: anchor_eff[a])

    shares = variance_partition(rows, metrics=("metric",))

    s = shares["metric"]
    assert s["anchor"] > 0.95
    grammar = s["state"] + s["o2"] + s["gluc"] + s["s_x_o"] + s["s_x_g"] + s["o_x_g"] + s["s_x_o_x_g"]
    assert grammar < 1e-9


def test_interaction_only_shows_up_as_interaction():
    rows = _build_rows(lambda a, s, o, g, _seed: 1.0 if (s == "prolif" and o == "high") else 0.0)

    shares = variance_partition(rows, metrics=("metric",))
    s = shares["metric"]
    assert s["s_x_o"] > 0.2


def test_seed_noise_lands_in_resid():
    rng_table = {(s, seed): 0.1 * seed for s, seed in [("prolif", 42), ("prolif", 43)]}
    rows = _build_rows(lambda a, s, o, g, seed: rng_table.get((s, seed), 0.0))

    shares = variance_partition(rows, metrics=("metric",))
    s = shares["metric"]
    assert s["resid"] > 0.0


def test_missing_metric_raises():
    rows = _build_rows(lambda *_: 0.0)
    with pytest.raises(KeyError):
        variance_partition(rows, metrics=("not_a_column",))
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `pytest tests/test_variance_partition.py -v`
Expected: all FAIL with `ImportError: cannot import name 'variance_partition'`.

---

## Task 2: Variance partition module — implementation

**Files:**
- Create: `src/a3_combinatorial_sweep/variance_partition.py`

- [ ] **Step 1: Write the module**

```python
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
        grand_mean = float(values.mean())
        total_ss = float(np.sum((values - grand_mean) ** 2))

        shares: dict[str, float] = {}
        residual = values - grand_mean
        consumed = np.zeros_like(values, dtype=np.float64)

        for term_name, factors in _ORDERED_TERMS:
            keys = _composite_key(rows_by_factor, factors)
            group_means_term = _group_means(values, keys) - grand_mean
            term_effect = group_means_term - consumed
            ss_term = float(np.sum(term_effect ** 2))
            shares[term_name] = ss_term
            consumed = consumed + term_effect

        residual_vec = values - grand_mean - consumed
        shares["resid"] = float(np.sum(residual_vec ** 2))

        if total_ss <= 0.0:
            out[metric] = {key: 0.0 for key in shares}
        else:
            out[metric] = {key: max(0.0, value / total_ss) for key, value in shares.items()}

    return out
```

- [ ] **Step 2: Run the tests and confirm they pass**

Run: `pytest tests/test_variance_partition.py -v`
Expected: all 5 tests PASS.

- [ ] **Step 3: Commit**

```bash
git add src/a3_combinatorial_sweep/variance_partition.py tests/test_variance_partition.py
git commit -m "feat: add 3-way ANOVA variance partition for combinatorial sweep"
```

---

## Task 3: a3 main.py — adopt a4 metric schema

**Files:**
- Modify: `src/a3_combinatorial_sweep/main.py` (constants `MORPHOLOGY_METRICS`, function `_compute_signature`)

- [ ] **Step 1: Replace the `MORPHOLOGY_METRICS` constant**

Replace lines 42–52 of `src/a3_combinatorial_sweep/main.py`:

```python
MORPHOLOGY_METRICS = (
    # morphology (from CellViT)
    "nuclear_area_mean",
    "eccentricity_mean",
    "nuclei_density",
    "intensity_mean_h",
    "intensity_mean_e",
    # appearance (HED + Haralick)
    "appearance.h_mean",
    "appearance.h_std",
    "appearance.e_mean",
    "appearance.e_std",
    "appearance.stain_vector_angle_deg",
    "appearance.texture_h_contrast",
    "appearance.texture_h_homogeneity",
    "appearance.texture_h_energy",
    "appearance.texture_e_contrast",
    "appearance.texture_e_homogeneity",
    "appearance.texture_e_energy",
)
```

- [ ] **Step 2: Replace `_compute_signature` with a4-backed implementation**

Replace the entire function body of `_compute_signature` (current implementation lines ~551–621). New implementation:

```python
def _compute_signature(image_path: Path) -> dict[str, float]:
    """Return a4-compatible morphology + appearance metrics for one tile."""
    from src.a4_uni_probe.appearance_metrics import (
        APPEARANCE_METRIC_NAMES,
        appearance_row_for_image,
    )
    from src.a4_uni_probe.labels import (
        MORPHOLOGY_ATTR_NAMES,
        compute_morphology_attributes_from_cellvit,
    )
    from tools.cellvit.contours import cellvit_sidecar_path

    morphology = compute_morphology_attributes_from_cellvit(cellvit_sidecar_path(image_path))
    appearance = appearance_row_for_image(image_path)

    row: dict[str, float] = {}
    for name in MORPHOLOGY_ATTR_NAMES:
        row[name] = float(morphology.get(name, float("nan")))
    for name in APPEARANCE_METRIC_NAMES:
        row[name] = float(appearance[name])
    return row
```

- [ ] **Step 3: Remove now-dead helpers**

Delete from `src/a3_combinatorial_sweep/main.py`:
- `_connected_component_sizes` (lines ~460–505)
- `_compute_glcm_features` (lines ~508–537)
- `_polygon_area` (lines ~540–548)

Also remove `from tools.cellvit.contours import load_cellvit_contours` at top — the new `_compute_signature` uses `cellvit_sidecar_path` instead.

- [ ] **Step 4: Run the existing a3 unit tests to verify nothing else broke**

Run: `pytest tests/ -k "a3 or combinatorial_sweep" -v`
Expected: tests targeting metric columns may fail (handled in Task 9). Tests for plan/job structure must still pass.

- [ ] **Step 5: Commit**

```bash
git add src/a3_combinatorial_sweep/main.py
git commit -m "refactor(a3): adopt a4 metric schema in _compute_signature"
```

---

## Task 4: Multi-seed support in generation + signature CSV

**Files:**
- Modify: `src/a3_combinatorial_sweep/main.py` (`run_generate_worker`, `run_anchor_worker`, `_iter_signature_rows`, `plan_task`, CLI)

- [ ] **Step 1: Add seed-aware output paths**

At top of `src/a3_combinatorial_sweep/main.py`, just below `DEFAULT_SEED = 42`:

```python
def _generated_subdir_name(seed: int) -> str:
    """Output subdir under out_dir for a given seed."""
    return "generated" if seed == DEFAULT_SEED else f"generated_s{seed}"


def _generated_root(out_dir: Path, seed: int) -> Path:
    return out_dir / _generated_subdir_name(seed)
```

- [ ] **Step 2: Update `run_anchor_worker` and `run_generate_worker` to use `_generated_root(seed)`**

In both functions, replace every occurrence of:

```python
config.out_dir / "generated" / anchor_id / f"{build_condition_id(condition)}.png"
```

with:

```python
_generated_root(config.out_dir, seed) / anchor_id / f"{build_condition_id(condition)}.png"
```

(There are 4 such occurrences total across the two functions; the `anchor_dir` local variable in `run_generate_worker` needs the same change.)

- [ ] **Step 3: Update `plan_task` and CLI to accept multiple seeds**

a. In `plan_task` (line ~148), change signature to `plan_task(config, runtime=None, *, seeds=(DEFAULT_SEED,))` and inside the per-anchor loop iterate seeds:

```python
for seed in seeds:
    output_dir = _generated_root(out_dir, seed)
    outputs = tuple(output_dir / anchor / f"{build_condition_id(condition)}.png" for condition in conditions)
    # ... rest of job-construction logic unchanged, but job_id becomes f"generate_{anchor}_s{seed}"
    # and command appends ("--seed", str(seed))
```

b. In CLI `main()` (line ~781), add:

```python
parser.add_argument("--seed", action="append", type=int, default=None,
                    help="Repeat for multi-seed renders. Default: [42].")
```

Resolve `seeds = tuple(args.seed) if args.seed else (DEFAULT_SEED,)` and thread into both `plan_task(seeds=...)` and `run_generate_worker(..., seeds=...)`.

c. Update `run_generate_worker` signature:

```python
def run_generate_worker(
    config,
    *,
    device=DEFAULT_DEVICE,
    guidance_scale=DEFAULT_GUIDANCE_SCALE,
    num_steps=DEFAULT_NUM_STEPS,
    seeds=(DEFAULT_SEED,),
):
    ...
    for seed in seeds:
        for index, anchor_id in enumerate(anchors, start=1):
            anchor_dir = _generated_root(config.out_dir, seed) / anchor_id
            # ... existing per-anchor loop body, passing seed= to _make_generation_noise
```

Single-anchor worker `run_anchor_worker` keeps its single `seed=` kwarg.

- [ ] **Step 4: Make `_iter_signature_rows` walk all `generated*` subtrees**

Replace `_iter_signature_rows` body with:

```python
def _iter_signature_rows(config: CombinatorialSweepConfig) -> list[dict[str, Any]]:
    anchors = _discover_summary_anchors(config)
    condition_lookup = _condition_lookup()
    rows: list[dict[str, Any]] = []
    for subdir in sorted(config.out_dir.iterdir() if config.out_dir.is_dir() else []):
        if not subdir.is_dir() or not subdir.name.startswith("generated"):
            continue
        seed = DEFAULT_SEED if subdir.name == "generated" else int(subdir.name.removeprefix("generated_s"))
        for anchor_id in anchors:
            anchor_dir = subdir / anchor_id
            if not anchor_dir.is_dir():
                continue
            for condition_id, condition in condition_lookup.items():
                image_path = anchor_dir / f"{condition_id}.png"
                if not image_path.is_file():
                    continue
                row = {
                    "anchor_id": anchor_id,
                    "cell_state": condition.cell_state,
                    "oxygen_label": condition.oxygen_label,
                    "oxygen_value": condition.oxygen_value,
                    "glucose_label": condition.glucose_label,
                    "glucose_value": condition.glucose_value,
                    "seed": seed,
                    "image_path": str(image_path),
                }
                row.update(_compute_signature(image_path))
                rows.append(row)
    return rows
```

Also update `_discover_summary_anchors` so it scans `generated*` not just `generated/`:

```python
def _discover_summary_anchors(config):
    anchors, _ = load_anchor_tile_ids(config)
    if anchors:
        return anchors
    found: set[str] = set()
    if config.out_dir.is_dir():
        for subdir in config.out_dir.iterdir():
            if subdir.is_dir() and subdir.name.startswith("generated"):
                for anchor_dir in subdir.iterdir():
                    if anchor_dir.is_dir():
                        found.add(anchor_dir.name)
    return sorted(found)
```

- [ ] **Step 5: Update signature fieldnames in `run_summary_worker`**

Replace the `signature_fieldnames` tuple to insert `seed`:

```python
signature_fieldnames = (
    "anchor_id",
    "cell_state",
    "oxygen_label",
    "oxygen_value",
    "glucose_label",
    "glucose_value",
    "seed",
    "image_path",
    *MORPHOLOGY_METRICS,
)
```

- [ ] **Step 6: Commit**

```bash
git add src/a3_combinatorial_sweep/main.py
git commit -m "feat(a3): multi-seed render + seed column in signatures"
```

---

## Task 5: Summary worker also writes `variance_partition.csv`

**Files:**
- Modify: `src/a3_combinatorial_sweep/main.py` (`_summary_output_paths`, `run_summary_worker`)

- [ ] **Step 1: Extend output-path helper**

Replace `_summary_output_paths`:

```python
def _summary_output_paths(out_dir: Path) -> tuple[Path, Path, Path, Path]:
    return (
        out_dir / "morphological_signatures.csv",
        out_dir / "additive_model_residuals.csv",
        out_dir / "interaction_heatmap.png",
        out_dir / "variance_partition.csv",
    )
```

Update both call sites:
- `plan_task`: `summary_outputs = _summary_output_paths(out_dir)` (no change needed)
- `run_summary_worker`: see step 2.

- [ ] **Step 2: Add variance partition write in `run_summary_worker`**

At the end of `run_summary_worker`, before the return, add:

```python
from src.a3_combinatorial_sweep.variance_partition import variance_partition

shares = variance_partition(signature_rows, metrics=MORPHOLOGY_METRICS)
variance_fieldnames = ("metric", "anchor", "state", "o2", "gluc", "s_x_o", "s_x_g", "o_x_g", "s_x_o_x_g", "resid")
variance_rows = [
    {"metric": metric, **shares[metric]}
    for metric in MORPHOLOGY_METRICS
]
signatures_path, residuals_path, heatmap_path, variance_path = _summary_output_paths(config.out_dir)
_write_csv(signatures_path, signature_rows, signature_fieldnames)
_write_csv(residuals_path, additive_rows, additive_fieldnames)
_write_interaction_heatmap(heatmap_path, additive_rows)
_write_csv(variance_path, variance_rows, variance_fieldnames)
return signatures_path, residuals_path, heatmap_path, variance_path
```

(Remove the old return-tuple and write block; this replaces them.)

- [ ] **Step 3: Commit**

```bash
git add src/a3_combinatorial_sweep/main.py
git commit -m "feat(a3): emit variance_partition.csv from summary worker"
```

---

## Task 6: Render seeds 43 and 44

**Files:**
- No code change. Shell only.

- [ ] **Step 1: Verify CellViT sidecars exist for seed-42 tiles**

Run: `ls src/a3_combinatorial_sweep/out/generated/10752_3072/*_cellvit_instances.json | wc -l`
Expected: `27` (one per condition).

If 0: the original seed-42 tiles never got CellViT — run the 3-step pipeline from `CLAUDE.md` for `generated/` first (script in Task 7).

- [ ] **Step 2: Render seed 43**

Run:

```bash
python -m src.a3_combinatorial_sweep.main \
  --worker generate \
  --seed 43 \
  --out-dir src/a3_combinatorial_sweep/out
```

Expected: prints `[i/20] generated <anchor>` lines and exits 0. Output count check:
`find src/a3_combinatorial_sweep/out/generated_s43 -name "*.png" | wc -l` → `540`.

- [ ] **Step 3: Render seed 44**

Run:

```bash
python -m src.a3_combinatorial_sweep.main \
  --worker generate \
  --seed 44 \
  --out-dir src/a3_combinatorial_sweep/out
```

Expected: `find src/a3_combinatorial_sweep/out/generated_s44 -name "*.png" | wc -l` → `540`.

- [ ] **Step 4: Do NOT commit PNGs** (existing repo ignores `out/generated*/`; double-check).

Run: `git status src/a3_combinatorial_sweep/out`
Expected: no PNGs listed.

---

## Task 7: CellViT sidecars for new seed tiles

**Files:**
- No code. Shell only. Follow `CLAUDE.md` "CellViT — Local Execution".

- [ ] **Step 1: Export seed-43 batch**

```bash
conda run --no-capture-output -n pixcell python tools/cellvit/export_batch.py \
  --cache-root src/a3_combinatorial_sweep/out/generated_s43 \
  --output-dir /tmp/cellvit_s43 --overwrite --zip
```

- [ ] **Step 2: Run CellViT on seed-43 batch**

```bash
set +u; source /home/ec2-user/miniconda3/etc/profile.d/conda.sh; conda activate cellvit; set -u
python /home/ec2-user/he-feature-visualizer/stages/run_cellvit_local.py \
  --zip /tmp/cellvit_s43.zip --out /tmp/cellvit_s43_results \
  --checkpoint /home/ec2-user/checkpoints/CellViT-256.pth \
  --cellvit-repo /home/ec2-user/CellViT
```

- [ ] **Step 3: Import seed-43 sidecars**

```bash
conda run --no-capture-output -n pixcell python tools/cellvit/import_results.py \
  --manifest /tmp/cellvit_s43/manifest.json --results-dir /tmp/cellvit_s43_results
```

Verify: `find src/a3_combinatorial_sweep/out/generated_s43 -name "*_cellvit_instances.json" | wc -l` → `540`.

- [ ] **Step 4: Repeat steps 1–3 for seed 44**

Substitute `s43` → `s44` in all paths.

---

## Task 8: Regenerate summary CSVs

**Files:**
- No code. Shell only.

- [ ] **Step 1: Run summary worker**

```bash
python -m src.a3_combinatorial_sweep.main \
  --worker summarize \
  --out-dir src/a3_combinatorial_sweep/out
```

- [ ] **Step 2: Sanity-check outputs**

```bash
head -1 src/a3_combinatorial_sweep/out/morphological_signatures.csv
wc -l src/a3_combinatorial_sweep/out/morphological_signatures.csv
head src/a3_combinatorial_sweep/out/variance_partition.csv
```

Expected:
- Header includes `seed` and all a4 metric names.
- Row count = `20 anchors × 27 conditions × 3 seeds + 1 header = 1621`.
- `variance_partition.csv` has 16 rows (one per metric) + header, columns: `metric,anchor,state,o2,gluc,s_x_o,s_x_g,o_x_g,s_x_o_x_g,resid`.

- [ ] **Step 3: Commit the updated CSVs (if they are tracked)**

```bash
git status src/a3_combinatorial_sweep/out/*.csv
# If they are tracked:
git add src/a3_combinatorial_sweep/out/morphological_signatures.csv \
        src/a3_combinatorial_sweep/out/additive_model_residuals.csv \
        src/a3_combinatorial_sweep/out/variance_partition.csv
git commit -m "data(a3): regenerate signatures + variance partition (3 seeds, a4 metrics)"
```

If CSVs are gitignored, skip the commit.

---

## Task 9: Update test fixtures for new metric names

**Files:**
- Modify: `tests/test_fig_combinatorial_grammar.py`

- [ ] **Step 1: Inspect the existing fixture to find the field names being used**

Run: `grep -n "hematoxylin\|glcm\|mean_cell_size\|nucleus_area" tests/test_fig_combinatorial_grammar.py`
Expected: a list of column names that were valid under the old schema.

- [ ] **Step 2: Update any fixture rows / asserts to use a4 names**

Replacement map:
- `nuclear_density` → `nuclei_density`
- `mean_cell_size`, `nucleus_area_median`, `nucleus_area_iqr` → drop; use `nuclear_area_mean`
- `hematoxylin_burden` → `appearance.h_mean`
- `hematoxylin_ratio`, `eosin_ratio` → drop
- `glcm_contrast` → `appearance.texture_h_contrast`
- `glcm_homogeneity` → `appearance.texture_h_homogeneity`

Where the test constructs synthetic rows, ensure every name in `MORPHOLOGY_METRICS` (post-Task 3) is present as a numeric column.

- [ ] **Step 3: Run the test file**

Run: `pytest tests/test_fig_combinatorial_grammar.py -v`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/test_fig_combinatorial_grammar.py
git commit -m "test: align combinatorial grammar fixtures with a4 metric schema"
```

---

## Task 10: Helper — load variance partition for fig builders

**Files:**
- Modify: `src/paper_figures/fig_combinatorial_grammar_panels/_shared.py`

- [ ] **Step 1: Append helper at end of `_shared.py`**

```python
def load_variance_partition(path: Path) -> list[dict[str, float]]:
    """Read variance_partition.csv into a list of dicts (numeric coercion)."""
    rows = read_csv(path)
    parsed: list[dict[str, float]] = []
    for row in rows:
        parsed.append(
            {
                "metric": str(row["metric"]),
                **{key: float(row[key]) for key in row if key != "metric"},
            }
        )
    return parsed
```

- [ ] **Step 2: Commit**

```bash
git add src/paper_figures/fig_combinatorial_grammar_panels/_shared.py
git commit -m "feat(fig): helper to load variance_partition.csv"
```

---

## Task 11: Variance-partition bar renderer (Panel A)

**Files:**
- Create: `src/paper_figures/fig_combinatorial_grammar_panels/_variance_bars.py`

- [ ] **Step 1: Write the renderer**

```python
"""Panel A: stacked variance-partition bars, one per metric."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.paper_figures.fig_combinatorial_grammar_panels._shared import load_variance_partition

_SEG_ORDER = ("anchor", "state", "o2", "gluc", "interactions", "resid")
_SEG_COLORS = {
    "anchor":       "#6b7280",
    "state":        "#2a5db0",
    "o2":           "#2a8a4a",
    "gluc":         "#c2a83e",
    "interactions": "#b04a2a",
    "resid":        "#cccccc",
}
_INTERACTION_KEYS = ("s_x_o", "s_x_g", "o_x_g", "s_x_o_x_g")


def _collapse(row: dict[str, float]) -> dict[str, float]:
    interactions = sum(float(row[key]) for key in _INTERACTION_KEYS)
    return {
        "anchor":       float(row["anchor"]),
        "state":        float(row["state"]),
        "o2":           float(row["o2"]),
        "gluc":         float(row["gluc"]),
        "interactions": interactions,
        "resid":        float(row["resid"]),
    }


def draw_variance_bars(ax: plt.Axes, variance_csv: Path) -> None:
    rows = load_variance_partition(variance_csv)
    if not rows:
        ax.text(0.5, 0.5, "no variance data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    collapsed = [(row["metric"], _collapse(row)) for row in rows]
    collapsed.sort(key=lambda pair: pair[1]["interactions"], reverse=True)

    metrics = [name for name, _ in collapsed]
    bar_data = np.array([[shares[k] for k in _SEG_ORDER] for _, shares in collapsed])
    cumulative = np.zeros(len(metrics), dtype=np.float64)
    y_positions = np.arange(len(metrics))[::-1]

    for col_index, seg_name in enumerate(_SEG_ORDER):
        widths = bar_data[:, col_index]
        ax.barh(
            y_positions, widths, left=cumulative,
            color=_SEG_COLORS[seg_name], edgecolor="white", linewidth=0.3,
            label=seg_name,
        )
        cumulative = cumulative + widths

    ax.set_yticks(y_positions)
    ax.set_yticklabels(metrics, fontsize=8)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("variance share")
    ax.set_title("Variance partition by metric (sorted by interaction share)", fontsize=10)
    ax.legend(loc="lower right", fontsize=7, frameon=False, ncol=3)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
```

- [ ] **Step 2: Commit**

```bash
git add src/paper_figures/fig_combinatorial_grammar_panels/_variance_bars.py
git commit -m "feat(fig): variance-partition stacked bar renderer for main fig 09"
```

---

## Task 12: Rewrite main fig 09 (Panel A + Panel B)

**Files:**
- Modify: `src/paper_figures/fig_combinatorial_grammar.py`

- [ ] **Step 1: Read the current file to confirm exports**

Run: `grep -n "def " src/paper_figures/fig_combinatorial_grammar.py`
Note the public function name (likely `build_combinatorial_grammar_figure` and `save_combinatorial_grammar_figure`).

- [ ] **Step 2: Rewrite as two-panel layout**

Replace the body of `build_combinatorial_grammar_figure` with:

```python
def build_combinatorial_grammar_figure(
    *,
    generated_root: Path = DEFAULT_GENERATED_ROOT,
    signatures_csv: Path = DEFAULT_SIGNATURES_CSV,
    variance_csv: Path | None = None,
) -> plt.Figure:
    from src.paper_figures.fig_combinatorial_grammar_panels._variance_bars import draw_variance_bars
    from src.paper_figures.fig_combinatorial_grammar_panels._shared import (
        compute_anchor_sweep_magnitude,
        pick_representative_anchor,
        read_csv,
    )

    variance_csv = variance_csv if variance_csv is not None else signatures_csv.parent / "variance_partition.csv"
    signature_rows = read_csv(signatures_csv)
    magnitudes = compute_anchor_sweep_magnitude(signature_rows)
    representative = pick_representative_anchor(signature_rows)
    # Anchor pick: max magnitude that is also the representative; if not the same, prefer max magnitude.
    panel_b_anchor = max(magnitudes.items(), key=lambda pair: (pair[1], pair[0] == representative, pair[0]))[0]

    fig = plt.figure(figsize=(7.5, 9.0), facecolor="white")
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.2], hspace=0.28)

    ax_bars = fig.add_subplot(gs[0, 0])
    draw_variance_bars(ax_bars, variance_csv)

    sub_b = gs[1, 0].subgridspec(1, 1)
    _draw_anchor_sweep_grid(fig, sub_b[0, 0], anchor_id=panel_b_anchor, generated_root=generated_root)
    return fig
```

- [ ] **Step 3: Add (or extract) `_draw_anchor_sweep_grid` helper into the same file**

```python
def _draw_anchor_sweep_grid(fig, subgrid, *, anchor_id: str, generated_root: Path) -> None:
    from src.paper_figures.fig_combinatorial_grammar_panels._shared import STATES, LEVELS, condition_id, load_rgb

    outer = subgrid.subgridspec(3, 9, hspace=0.04, wspace=0.04)
    for state_idx, state in enumerate(STATES):
        for ox_idx, ox in enumerate(LEVELS):
            for gluc_idx, gluc in enumerate(LEVELS):
                col = ox_idx * len(LEVELS) + gluc_idx
                ax = fig.add_subplot(outer[state_idx, col])
                tile_path = generated_root / anchor_id / f"{condition_id(state, ox, gluc)}.png"
                ax.imshow(load_rgb(tile_path))
                ax.set_xticks([]); ax.set_yticks([])
                if col == 0:
                    ax.set_ylabel(state, fontsize=8)
                if state_idx == 0:
                    ax.set_title(f"{ox}/{gluc}", fontsize=7, pad=1.0)
                for spine in ax.spines.values():
                    spine.set_linewidth(0.25)
                    spine.set_edgecolor("#8A8A8A")
```

- [ ] **Step 4: Regenerate the PNG**

Run:
```bash
python -m src.paper_figures.fig_combinatorial_grammar
```

If the module has no `__main__`, wire one up:

```python
if __name__ == "__main__":
    save_combinatorial_grammar_figure()
```

Verify: `ls -l figures/pngs_updated/09_combinatorial_grammar.png` updates mtime.

- [ ] **Step 5: Commit**

```bash
git add src/paper_figures/fig_combinatorial_grammar.py
git commit -m "feat(fig): main fig 09 = variance bars + anchor sweep grid"
```

---

## Task 13: SI builder — residual small-multiples

**Files:**
- Create: `src/paper_figures/fig_combinatorial_grammar_panels/_residual_small_multiples.py`

- [ ] **Step 1: Write the renderer**

```python
"""SI panel: per-metric residual heatmap small-multiples (state x (O2,gluc))."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.a3_combinatorial_sweep.main import MORPHOLOGY_METRICS
from src.paper_figures.fig_combinatorial_grammar_panels._shared import (
    LEVELS, STATES, read_csv,
)


def _residual_grid(rows: list[dict[str, str]], metric: str) -> np.ndarray:
    """Return 3 x 9 array of (actual - additive expected) for one metric."""
    by_key: dict[tuple[str, str, str], float] = {}
    for row in rows:
        key = (str(row["cell_state"]), str(row["oxygen_label"]), str(row["glucose_label"]))
        actual = float(row[f"actual_{metric}"])
        expected = float(row[f"expected_{metric}"])
        by_key[key] = actual - expected
    grid = np.zeros((len(STATES), len(LEVELS) * len(LEVELS)), dtype=np.float64)
    for s_idx, state in enumerate(STATES):
        for ox_idx, ox in enumerate(LEVELS):
            for g_idx, gluc in enumerate(LEVELS):
                col = ox_idx * len(LEVELS) + g_idx
                grid[s_idx, col] = by_key.get((state, ox, gluc), 0.0)
    return grid


def draw_residual_small_multiples(fig: plt.Figure, subgrid, *, residuals_csv: Path) -> None:
    rows = read_csv(residuals_csv)
    n_metrics = len(MORPHOLOGY_METRICS)
    ncols = 3
    nrows = (n_metrics + ncols - 1) // ncols
    inner = subgrid.subgridspec(nrows, ncols, hspace=0.55, wspace=0.25)

    for idx, metric in enumerate(MORPHOLOGY_METRICS):
        row, col = divmod(idx, ncols)
        ax = fig.add_subplot(inner[row, col])
        grid = _residual_grid(rows, metric)
        max_abs = float(np.max(np.abs(grid))) if grid.size else 1.0
        max_abs = max(max_abs, 1e-9)
        im = ax.imshow(grid, cmap="RdBu_r", vmin=-max_abs, vmax=max_abs, aspect="auto")
        ax.set_title(metric, fontsize=7)
        ax.set_xticks(range(len(LEVELS) * len(LEVELS)))
        ax.set_xticklabels([f"{ox[0]}/{g[0]}" for ox in LEVELS for g in LEVELS], fontsize=5, rotation=45)
        ax.set_yticks(range(len(STATES)))
        ax.set_yticklabels(STATES, fontsize=6)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
```

- [ ] **Step 2: Commit**

```bash
git add src/paper_figures/fig_combinatorial_grammar_panels/_residual_small_multiples.py
git commit -m "feat(fig): SI residual small-multiples per metric"
```

---

## Task 14: SI builder — seed CI table

**Files:**
- Create: `src/paper_figures/fig_combinatorial_grammar_panels/_seed_ci_table.py`

- [ ] **Step 1: Write the renderer**

```python
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
```

- [ ] **Step 2: Commit**

```bash
git add src/paper_figures/fig_combinatorial_grammar_panels/_seed_ci_table.py
git commit -m "feat(fig): SI seed-bootstrap CI table"
```

---

## Task 15: SI builder — anchor sensitivity ranking

**Files:**
- Create: `src/paper_figures/fig_combinatorial_grammar_panels/_anchor_ranking.py`

- [ ] **Step 1: Write the renderer**

```python
"""SI panel: horizontal bar chart of per-anchor sweep magnitude (||Delta metric||_2)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from src.paper_figures.fig_combinatorial_grammar_panels._shared import (
    compute_anchor_sweep_magnitude, read_csv,
)


def draw_anchor_ranking(fig: plt.Figure, subgrid, *, signatures_csv: Path) -> None:
    rows = read_csv(signatures_csv)
    magnitudes = compute_anchor_sweep_magnitude(rows)
    ordered = sorted(magnitudes.items(), key=lambda pair: pair[1])

    ax = fig.add_subplot(subgrid)
    anchor_ids = [pair[0] for pair in ordered]
    values = [pair[1] for pair in ordered]
    ax.barh(anchor_ids, values, color="#2a5db0")
    ax.set_xlabel("sweep magnitude (sum-var across metrics)")
    ax.set_title("Anchor sweep responsiveness", fontsize=9)
    ax.tick_params(axis="y", labelsize=6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
```

- [ ] **Step 2: Commit**

```bash
git add src/paper_figures/fig_combinatorial_grammar_panels/_anchor_ranking.py
git commit -m "feat(fig): SI anchor sensitivity ranking"
```

---

## Task 16: Wire SI subbuilders into `fig_combinatorial_grammar_si.py`

**Files:**
- Modify: `src/paper_figures/fig_combinatorial_grammar_si.py`

- [ ] **Step 1: Extend `build_combinatorial_grammar_si_figure` to include new panels**

After the existing 2×2 raw-grid section, append (or restructure as 2×2 outer with raw grids as one quadrant and new panels in the other three):

```python
from src.paper_figures.fig_combinatorial_grammar_panels._residual_small_multiples import draw_residual_small_multiples
from src.paper_figures.fig_combinatorial_grammar_panels._seed_ci_table import draw_seed_ci_table
from src.paper_figures.fig_combinatorial_grammar_panels._anchor_ranking import draw_anchor_ranking

DEFAULT_RESIDUALS_CSV = DEFAULT_A3_OUT / "additive_model_residuals.csv"
```

Change figure layout (in the same function):

```python
fig = plt.figure(figsize=(16.0, 18.0), facecolor="white")
outer = fig.add_gridspec(3, 2, height_ratios=[1.2, 1.0, 1.0], hspace=0.35, wspace=0.2)

# Existing 2x2 raw anchor grids -> outer[0, :]
raw_subgrid = outer[0, :].subgridspec(2, 2, wspace=0.10, hspace=0.20)
for idx, anchor_id in enumerate(picks[:4]):
    row, col = divmod(idx, 2)
    title = f"{role_labels[idx]}: anchor {anchor_id} (sweep magnitude={magnitudes.get(anchor_id, 0.0):.3g})"
    _draw_anchor_subgrid(fig, raw_subgrid[row, col], anchor_id=anchor_id, generated_root=generated_root, title=title)

# Residual small-multiples
draw_residual_small_multiples(fig, outer[1, 0], residuals_csv=DEFAULT_RESIDUALS_CSV)

# Anchor ranking
draw_anchor_ranking(fig, outer[1, 1], signatures_csv=signatures_csv)

# Seed CI table spans both bottom cells
draw_seed_ci_table(fig, outer[2, :], signatures_csv=signatures_csv)
```

- [ ] **Step 2: Regenerate the SI PNG**

```bash
python -c "from src.paper_figures.fig_combinatorial_grammar_si import save_combinatorial_grammar_si_figure; save_combinatorial_grammar_si_figure()"
```

Verify: `ls -l figures/pngs_updated/SI_09_combinatorial_grammar_anchors.png` updates mtime.

- [ ] **Step 3: Commit**

```bash
git add src/paper_figures/fig_combinatorial_grammar_si.py
git commit -m "feat(fig): SI fig 09 adds residual small-multiples + seed CI + anchor ranking"
```

---

## Task 17: Final pass — full test sweep + figure regen

- [ ] **Step 1: Run all combinatorial-grammar tests**

```bash
pytest tests/test_variance_partition.py tests/test_fig_combinatorial_grammar.py -v
```

Expected: all PASS.

- [ ] **Step 2: Regenerate both figures one final time**

```bash
python -m src.paper_figures.fig_combinatorial_grammar
python -c "from src.paper_figures.fig_combinatorial_grammar_si import save_combinatorial_grammar_si_figure; save_combinatorial_grammar_si_figure()"
```

- [ ] **Step 3: Eyeball both PNGs**

Open `figures/pngs_updated/09_combinatorial_grammar.png` and
`figures/pngs_updated/SI_09_combinatorial_grammar_anchors.png` in the IDE.

Checklist:
- Panel A bars sum to 1.0 visually (no large white gaps at right edge).
- Bars sorted by interaction share (top bar has the most red/orange segment).
- Panel B grid has all 27 tiles populated.
- SI: residual small-multiples are roughly symmetric around 0 (RdBu_r), seed CI table is legible at zoom 200%.

- [ ] **Step 4: Commit the regenerated PNGs**

```bash
git add figures/pngs_updated/09_combinatorial_grammar.png \
        figures/pngs_updated/SI_09_combinatorial_grammar_anchors.png
git commit -m "fig: regenerate combinatorial-grammar main + SI with variance partition"
```

---

## Self-review notes

- **Spec coverage**: §3 metric swap → Task 3; §4 seed band → Tasks 4, 6, 7; §5 variance partition → Tasks 1, 2, 5; §6 main fig → Tasks 11, 12; §7 SI deliverables → Tasks 13–16; §8 file map matches task `Files:` headers; §9 CellViT prereq → Task 7; §10 risks acknowledged in step expectations.
- **Placeholder scan**: clean — every code step has a complete code block; every shell step has the exact command and expected output count.
- **Type consistency**: `MORPHOLOGY_METRICS` defined in Task 3, consumed by `_shared.py` (existing import), `variance_partition.csv` schema (Task 5), bar renderer (Task 11), and small-multiples (Task 13). Variance partition keys (`anchor`, `state`, `o2`, `gluc`, `s_x_o`, `s_x_g`, `o_x_g`, `s_x_o_x_g`, `resid`) match between Task 2 module, Task 5 CSV header, and Task 11 `_INTERACTION_KEYS`. Renderer function names (`draw_variance_bars`, `draw_residual_small_multiples`, `draw_seed_ci_table`, `draw_anchor_ranking`) match between definition and wire-up sites.
