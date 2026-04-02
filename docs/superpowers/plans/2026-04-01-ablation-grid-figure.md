# Ablation Grid Figure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the ablation grid to support four evaluation metrics (cosine, LPIPS, AJI, PQ), update the matplotlib figure to show stacked metric bars, and formalize the interactive web vis as a proper CLI script.

**Architecture:** 
- `tools/stage3_ablation_grid_figure.py` — exists; needs metric-bar update and `--sort-by` flag.
- `tools/compute_ablation_metrics.py` — new; computes LPIPS + CellViT cell detection + AJI + PQ, writes `metrics.json`.
- `tools/stage3_ablation_grid_webvis.py` — new; formalizes the inline HTML generator as a proper CLI script.

**Tech Stack:** Python 3.12+, matplotlib, numpy, Pillow, `lpips`, `torch`, `scipy` (Hungarian matching).

---

## Status: Tasks 1–7 complete (2026-04-02)

### ✅ Task 1: Skeleton, constants, and pure helpers — DONE
`tools/stage3_ablation_grid_figure.py` created with `_cardinality_color`, `_condition_label`, `_find_real_he`. Tests pass.

### ✅ Task 2: Sort helper — DONE
`_sort_conditions_by_cosine` implemented. **Bug fixed:** uses `float("-inf")` for missing scores so they sort last.

### ✅ Task 3: Cosine score loading — DONE
`_parse_cosine_json`, `_compute_image_cosine`, `_load_grid_cosine_scores` implemented.

### ✅ Task 4: Cell drawing helpers — DONE
`_draw_dot_row`, `_draw_cell_border`, `_draw_cosine_bar_cell`, `_draw_label_ax` implemented.

### ✅ Task 5: Manifest lookup + render function — DONE
`_build_manifest_lookup`, `render_ablation_grid_figure` implemented. 4×4 GridSpec layout working.

### ✅ Task 6: CLI + multi-tile loop — DONE
`_render_grid_for_cache_dir`, `main()` implemented with full argument parser.

### ✅ Task 7: Smoke test — DONE
Smoke test passed against `inference_output/test_combinations/17408_32768`. Preview PNG rendered correctly. All 8 unit tests pass.

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `tools/stage3_ablation_grid_figure.py` | **Modify** | Add multi-metric bar rendering + `--sort-by` flag |
| `tools/compute_ablation_metrics.py` | **Create** | LPIPS + CellViT + AJI + PQ computation → `metrics.json` |
| `tools/stage3_ablation_grid_webvis.py` | **Create** | Standalone HTML web vis generator with CLI |
| `tests/test_stage3_ablation_grid_figure.py` | **Modify** | Add tests for updated metric bar helpers |
| `tests/test_compute_ablation_metrics.py` | **Create** | Tests for AJI/PQ pure helpers |

---

### Task 8: Unified metrics JSON + LPIPS computation

**Files:**
- Create: `tools/compute_ablation_metrics.py`
- Create: `tests/test_compute_ablation_metrics.py`

**Goal:** Compute LPIPS between ref H&E and each generated H&E; write `metrics.json`.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_compute_ablation_metrics.py
from tools.compute_ablation_metrics import (
    _merge_cosine_into_metrics,
    _empty_metrics_record,
)

def test_empty_record_has_all_keys():
    r = _empty_metrics_record()
    assert set(r.keys()) == {'cosine', 'lpips', 'aji', 'pq'}
    assert all(v is None for v in r.values())

def test_merge_cosine_preserves_existing():
    existing = {'cell_types': {'cosine': None, 'lpips': 0.3, 'aji': None, 'pq': None}}
    cosine_scores = {'cell_types': 0.9946}
    result = _merge_cosine_into_metrics(existing, cosine_scores)
    assert result['cell_types']['cosine'] == 0.9946
    assert result['cell_types']['lpips'] == 0.3  # preserved
```

- [ ] **Step 2: Run tests to confirm failure**

```bash
python3 -m pytest tests/test_compute_ablation_metrics.py -v 2>&1 | head -20
```

- [ ] **Step 3: Implement `compute_ablation_metrics.py`**

Key functions:
```python
def _empty_metrics_record() -> dict:
    return {'cosine': None, 'lpips': None, 'aji': None, 'pq': None}

def _merge_cosine_into_metrics(
    existing: dict[str, dict],
    cosine_scores: dict[str, float],
) -> dict[str, dict]:
    ...

def compute_lpips_scores(
    cache_dir: Path,
    orion_root: Path,
    *,
    device: str = 'cuda',
) -> dict[str, float]:
    """Run LPIPS(AlexNet) between ref H&E and each condition's generated H&E.
    Returns {condition_key: lpips_value}. Lower is better."""
    import lpips
    loss_fn = lpips.LPIPS(net='alex').to(device)
    ...

def load_or_build_metrics(cache_dir: Path) -> dict[str, dict]:
    """Load metrics.json if present; else migrate from uni_cosine_scores.json."""
    ...

def write_metrics(cache_dir: Path, per_condition: dict[str, dict]) -> Path:
    """Write <cache_dir>/metrics.json. Returns path."""
    ...
```

CLI:
```bash
python tools/compute_ablation_metrics.py \
  --cache-dir inference_output/test_combinations/<tile_id> \
  --orion-root data/orion-crc33 \
  --metrics lpips \       # choices: cosine lpips aji pq all
  --device cuda
```

- [ ] **Step 4: Run tests**

```bash
python3 -m pytest tests/test_compute_ablation_metrics.py -v
```

- [ ] **Step 5: Commit**

```bash
git add tools/compute_ablation_metrics.py tests/test_compute_ablation_metrics.py
git commit -m "feat: add compute_ablation_metrics with LPIPS and metrics.json schema"
```

---

### Task 9: AJI + PQ via CellViT cell detection

**Files:**
- Modify: `tools/compute_ablation_metrics.py` (add `compute_cell_metrics`)
- Modify: `tests/test_compute_ablation_metrics.py` (add AJI/PQ pure helper tests)

**Goal:** Detect cells in each generated H&E using CellViT-SAM-H, match against input `cell_masks`, compute AJI and PQ.

- [ ] **Step 1: Write failing tests for pure AJI/PQ helpers**

```python
import numpy as np
from tools.compute_ablation_metrics import _compute_aji, _compute_pq

def test_aji_perfect_match():
    # 1 GT cell = 1 pred cell, perfect overlap
    gt = np.zeros((64, 64), dtype=np.int32)
    gt[10:20, 10:20] = 1
    pred = gt.copy()
    assert _compute_aji(gt, pred) == pytest.approx(1.0)

def test_pq_no_detections():
    gt = np.zeros((64, 64), dtype=np.int32)
    gt[10:20, 10:20] = 1
    pred = np.zeros_like(gt)
    sq, rq, pq = _compute_pq(gt, pred)
    assert pq == pytest.approx(0.0)

def test_aji_no_overlap():
    gt = np.zeros((64, 64), dtype=np.int32); gt[5:15, 5:15] = 1
    pred = np.zeros((64, 64), dtype=np.int32); pred[40:50, 40:50] = 1
    assert _compute_aji(gt, pred) == pytest.approx(0.0)
```

- [ ] **Step 2: Run to confirm failure**

- [ ] **Step 3: Implement CellViT runner + AJI/PQ helpers**

```python
def run_cellvit(image_path: Path, model_path: Path, device: str) -> np.ndarray:
    """Run CellViT-SAM-H → labeled instance mask (int32, 0=bg). """
    ...

def _compute_aji(gt_inst: np.ndarray, pred_inst: np.ndarray) -> float:
    """Aggregated Jaccard Index between GT and pred instance masks."""
    ...

def _compute_pq(gt_inst: np.ndarray, pred_inst: np.ndarray) -> tuple[float, float, float]:
    """Returns (SQ, RQ, PQ=SQ*RQ). Matching: Hungarian at IoU >= 0.5."""
    from scipy.optimize import linear_sum_assignment
    ...

def compute_cell_metrics(
    cache_dir: Path,
    orion_root: Path,
    cellvit_model: Path,
    *,
    device: str = 'cuda',
) -> dict[str, dict]:
    """Run CellViT on each condition, compute AJI + PQ vs input cell_masks.
    Returns {condition_key: {'aji': float, 'pq': float}}."""
    ...
```

GT instance mask source: `skimage.measure.label` applied to binary `cell_masks` channel at `<orion_root>/exp_channels/cell_masks/<tile_id>.png`.

- [ ] **Step 4: Run tests**

- [ ] **Step 5: Commit**

```bash
git add tools/compute_ablation_metrics.py tests/test_compute_ablation_metrics.py
git commit -m "feat: add CellViT cell detection and AJI/PQ computation"
```

---

### Task 10: Update matplotlib figure — 4 stacked metric bars

**Files:**
- Modify: `tools/stage3_ablation_grid_figure.py`
- Modify: `tests/test_stage3_ablation_grid_figure.py`

**Goal:** Replace single `_draw_cosine_bar_cell` with `_draw_metric_bars_cell` (4 stacked bars). Add `--sort-by` CLI flag. Read from `metrics.json` via `compute_ablation_metrics.load_or_build_metrics`.

- [ ] **Step 1: Add tests for new bar helper**

```python
from tools.stage3_ablation_grid_figure import _draw_metric_bars_cell
import matplotlib.pyplot as plt

def test_draw_metric_bars_cell_no_crash():
    fig, ax = plt.subplots()
    metrics = {'cosine': 0.995, 'lpips': None, 'aji': 0.7, 'pq': None}
    _draw_metric_bars_cell(ax, metrics, color='#0072B2')
    plt.close()
```

- [ ] **Step 2: Run to confirm failure**

- [ ] **Step 3: Replace `_draw_cosine_bar_cell` with `_draw_metric_bars_cell`**

```python
METRIC_BAR_COLORS = {
    'cosine': '#0072B2',
    'lpips':  '#D55E00',
    'aji':    '#009E73',
    'pq':     '#9B59B6',
}
METRIC_BAR_LABELS = {'cosine': 'Co', 'lpips': 'LP', 'aji': 'AJ', 'pq': 'PQ'}
METRIC_BAR_INVERT = {'lpips'}  # lower is better; invert fill direction

def _draw_metric_bars_cell(ax, metrics: dict, color: str) -> None:
    """4 stacked thin bars: Co / LP / AJ / PQ. Placeholder = dashed gray."""
    ...
```

Update GridSpec `bars_row` height ratio from `0.08` → `0.35`.

Update `render_ablation_grid_figure`:
- Accept `metrics: dict[str, dict]` (loaded from `metrics.json`)
- Call `_draw_metric_bars_cell` instead of `_draw_cosine_bar_cell`
- Accept `sort_by: str = 'cosine'` parameter; pass to `_sort_conditions_by_cosine`

Update CLI: add `--sort-by {cosine,lpips,aji,pq}` (default: `cosine`).

Update `_sort_conditions_by_cosine` signature:
```python
def _sort_conditions_by_cosine(
    conditions: list[tuple[str, ...]],
    scores: dict[str, float],  # already keyed by condition_metric_key
) -> list[tuple[str, ...]]:
    ...  # no change to implementation
```

Rename to `_sort_conditions_by_metric` in the same function, same logic.

- [ ] **Step 4: Run all tests**

```bash
python3 -m pytest tests/test_stage3_ablation_grid_figure.py -v
```

- [ ] **Step 5: Smoke test**

```bash
python3 tools/stage3_ablation_grid_figure.py \
  --cache-dir inference_output/test_combinations/17408_32768 \
  --orion-root data/orion-crc33 \
  --all4ch-image inference_output/test_combinations/17408_32768/all/generated_he.png \
  --no-auto-cosine \
  --sort-by cosine \
  --dpi 150 \
  --output-name ablation_grid_4metrics
```

- [ ] **Step 6: Commit**

```bash
git add tools/stage3_ablation_grid_figure.py tests/test_stage3_ablation_grid_figure.py
git commit -m "feat: replace cosine bar with 4 stacked metric bars in ablation grid figure"
```

---

### Task 11: Web vis generator script

**Files:**
- Create: `tools/stage3_ablation_grid_webvis.py`

**Goal:** Formalize the inline Python HTML generator as a proper reusable CLI script. No new tests (output is HTML, tested by manual verification).

- [ ] **Step 1: Create `tools/stage3_ablation_grid_webvis.py`**

Extract the HTML generation logic from the session prototype into a proper script. Key functions:

```python
def render_ablation_grid_html(
    cache_dir: Path,
    *,
    all4ch_image: Path,
    orion_root: Path,
    tile_id: str,
    out_html: Path,
) -> Path:
    """Generate self-contained HTML ablation grid. Returns path to HTML."""
    ...

def main() -> None:
    ...
```

The HTML must:
- Embed all images as base64 (self-contained, no server needed after generation).
- Load metrics from `metrics.json` (fall back to `uni_cosine_scores.json` for cosine only).
- Show 4 metric bars per cell (Co/LP/AJ/PQ), dashed placeholder for nulls.
- Support sort by any metric via JS buttons.
- Pin Real H&E at position 16 regardless of sort.
- Show rank badge and hover tooltip with all 4 values.

- [ ] **Step 2: Smoke test**

```bash
python3 tools/stage3_ablation_grid_webvis.py \
  --cache-dir inference_output/test_combinations/17408_32768 \
  --orion-root data/orion-crc33 \
  --all4ch-image inference_output/test_combinations/17408_32768/all/generated_he.png
# Open inference_output/test_combinations/17408_32768/ablation_grid.html
```

- [ ] **Step 3: Commit**

```bash
git add tools/stage3_ablation_grid_webvis.py
git commit -m "feat: add stage3_ablation_grid_webvis CLI script"
```

---

## Self-Review Checklist

- [ ] `metrics.json` schema matches spec (version 2, keys: cosine/lpips/aji/pq)
- [ ] LPIPS lower-is-better correctly inverted in bar display
- [ ] AJI/PQ use `cell_masks` as GT (not ref H&E)
- [ ] CellViT model path defaults to `pretrained_models/cellvit-sam-h/`
- [ ] `--sort-by` flag wired through CLI → render function → sort helper
- [ ] Web vis reads `metrics.json`, falls back to `uni_cosine_scores.json`
- [ ] All existing 8 tests still pass after Task 10 changes
- [ ] Static figure GridSpec `bars_row` height ratio updated to `0.35`
