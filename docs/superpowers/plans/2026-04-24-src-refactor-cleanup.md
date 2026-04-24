# src/ Refactor & Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate ~350 LOC of duplicated utility code in `src/`, unify figure-saving conventions in `paper_figures/`, and remove dead wrapper functions.

**Architecture:** Three canonical utilities (`load_tile_ids`, `load_cv_splits`, `write_csv`) move to `_tasklib/io.py`; a new `_tasklib/plotting.py` owns matplotlib Agg setup and `save_figure`; all `paper_figures/` modules standardise on `apply_style()` + `save_figure()` with consistent PNG save args.

**Tech Stack:** Python 3.10+, matplotlib, numpy, pytest

---

## File Change Map

| File | Change |
|------|--------|
| `src/_tasklib/io.py` | Add `load_tile_ids`, `load_cv_splits`, `write_csv` + `import csv` |
| `src/_tasklib/__init__.py` | Export new symbols |
| `src/_tasklib/plotting.py` | **Create** — Agg backend + `save_figure` |
| `src/a1_probe_linear/main.py` | Remove local `load_tile_ids` / `load_cv_splits`; import from `_tasklib` |
| `src/a1_probe_encoders/main.py` | Replace try/except wrappers with direct `_tasklib` imports |
| `src/a1_generated_probe/main.py` | Remove local `load_tile_ids`; import from `_tasklib` |
| `src/a0_tradeoff_scatter/render.py` | Remove boilerplate; use `_tasklib.plotting` |
| `src/a0_visibility_map/render.py` | Remove boilerplate; use `_tasklib.plotting` |
| `src/a0_tradeoff_scatter/collect.py` | Replace DictWriter block in `write_tradeoff_csv` with `write_csv` |
| `src/a0_visibility_map/collect.py` | Replace DictWriter block in `write_visibility_summary_csv` with `write_csv` |
| `src/a1_generated_probe/main.py` | Replace DictWriter block in `_write_comparison_csv` with `write_csv` |
| `src/a1_probe_encoders/main.py` | Replace DictWriter block in `_write_encoder_comparison_csv` with `write_csv` |
| `src/a1_probe_mlp/main.py` | Replace DictWriter block in `_write_comparison_csv` with `write_csv` |
| `src/a2_decomposition/metrics.py` | Replace DictWriter block in `write_summary_csv` with `write_csv` |
| `src/paper_figures/fig_inverse_decoding.py` | Add `apply_style()` call; use `save_figure` |
| `src/paper_figures/fig_uni_tme_decomposition.py` | Replace `fig.savefig` + `plt.close` with `save_figure` |
| `src/paper_figures/fig_ablation_grid.py` | Standardise save via `save_figure` |
| `tests/test_tasklib_io.py` | **Create** — tests for the three new io utilities |
| `tests/test_tasklib_plotting.py` | **Create** — tests for `save_figure` |

---

## Task 1: Add `load_tile_ids`, `load_cv_splits`, `write_csv` to `_tasklib/io.py`

**Files:**
- Modify: `src/_tasklib/io.py`
- Modify: `src/_tasklib/__init__.py`
- Create: `tests/test_tasklib_io.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_tasklib_io.py
from __future__ import annotations
import csv
import json
from pathlib import Path

import pytest


def test_load_tile_ids_basic(tmp_path):
    from src._tasklib.io import load_tile_ids
    (tmp_path / "tile_ids.txt").write_text("0_0\n0_1\n\n1_0\n", encoding="utf-8")
    assert load_tile_ids(tmp_path / "tile_ids.txt") == ["0_0", "0_1", "1_0"]


def test_load_tile_ids_skips_blank_lines(tmp_path):
    from src._tasklib.io import load_tile_ids
    (tmp_path / "ids.txt").write_text("\n  \n2_3\n", encoding="utf-8")
    assert load_tile_ids(tmp_path / "ids.txt") == ["2_3"]


def test_load_cv_splits_roundtrip(tmp_path):
    from src._tasklib.io import load_cv_splits
    from src._tasklib.tile_ids import tile_ids_sha1
    tile_ids = ["0_0", "0_1", "1_0"]
    payload = {
        "version": 1,
        "tile_ids_sha1": tile_ids_sha1(tile_ids),
        "splits": [{"train_idx": [0, 1], "test_idx": [2]}],
    }
    splits_path = tmp_path / "splits.json"
    splits_path.write_text(json.dumps(payload), encoding="utf-8")
    result = load_cv_splits(tile_ids, splits_path)
    assert len(result) == 1
    assert result[0]["train_idx"] == [0, 1]


def test_load_cv_splits_rejects_hash_mismatch(tmp_path):
    from src._tasklib.io import load_cv_splits
    splits_path = tmp_path / "splits.json"
    splits_path.write_text(
        json.dumps({"tile_ids_sha1": "badhash", "splits": []}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="does not match"):
        load_cv_splits(["0_0"], splits_path)


def test_write_csv_creates_file(tmp_path):
    from src._tasklib.io import write_csv
    rows = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
    out = write_csv(rows, ["a", "b"], tmp_path / "out.csv")
    assert out.is_file()
    with out.open(newline="") as f:
        reader = list(csv.DictReader(f))
    assert reader[0] == {"a": "1", "b": "2"}
    assert reader[1] == {"a": "3", "b": "4"}


def test_write_csv_creates_parent_dirs(tmp_path):
    from src._tasklib.io import write_csv
    out = write_csv([{"x": 1}], ["x"], tmp_path / "nested" / "sub" / "out.csv")
    assert out.is_file()
```

- [ ] **Step 2: Run tests — expect ImportError on new symbols**

```
pytest tests/test_tasklib_io.py -v
```

Expected: 6 failures with `ImportError: cannot import name 'load_tile_ids'`

- [ ] **Step 3: Implement new functions in `src/_tasklib/io.py`**

Replace the entire file content:

```python
"""Small filesystem helpers shared by task packages."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def ensure_directory(path: str | Path) -> Path:
    """Create a directory if it does not already exist and return it."""
    out_dir = Path(path)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def write_json(payload: Any, output_path: str | Path) -> Path:
    """Write a JSON payload with stable formatting."""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return out_path


def load_tile_ids(path: str | Path) -> list[str]:
    """Load newline-delimited tile IDs, skipping blank lines."""
    return [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_cv_splits(tile_ids: list[str], cv_splits_path: str | Path) -> list[dict[str, list[int]]]:
    """Load and validate saved CV splits against the current tile list."""
    from src._tasklib.tile_ids import tile_ids_sha1

    payload = json.loads(Path(cv_splits_path).read_text(encoding="utf-8"))
    if payload.get("tile_ids_sha1") != tile_ids_sha1(tile_ids):
        raise ValueError("tile_ids.txt does not match the saved CV split hash")
    return list(payload["splits"])


def write_csv(rows: list[dict], fieldnames: list[str], output_path: str | Path) -> Path:
    """Write rows as a CSV with a header row, creating parent directories."""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return out_path
```

- [ ] **Step 4: Export new symbols from `src/_tasklib/__init__.py`**

```python
"""Shared helpers for task packages."""

from .io import ensure_directory, load_cv_splits, load_tile_ids, write_csv, write_json
from .runtime import CommandSpec, JobPlan, JobState, RuntimeProbe, TaskPlan, probe_runtime
from .tile_ids import list_feature_tile_ids, parse_tile_id, tile_ids_sha1, write_tile_ids

__all__ = [
    "CommandSpec",
    "JobPlan",
    "JobState",
    "RuntimeProbe",
    "TaskPlan",
    "ensure_directory",
    "list_feature_tile_ids",
    "load_cv_splits",
    "load_tile_ids",
    "parse_tile_id",
    "probe_runtime",
    "tile_ids_sha1",
    "write_csv",
    "write_json",
    "write_tile_ids",
]
```

- [ ] **Step 5: Run tests — expect all 6 to pass**

```
pytest tests/test_tasklib_io.py -v
```

Expected: 6 passed

- [ ] **Step 6: Commit**

```bash
git add src/_tasklib/io.py src/_tasklib/__init__.py tests/test_tasklib_io.py
git commit -m "feat: add load_tile_ids, load_cv_splits, write_csv to _tasklib/io"
```

---

## Task 2: Create `_tasklib/plotting.py`

**Files:**
- Create: `src/_tasklib/plotting.py`
- Create: `tests/test_tasklib_plotting.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_tasklib_plotting.py
from __future__ import annotations
from pathlib import Path


def test_save_figure_writes_png(tmp_path):
    from src._tasklib.plotting import save_figure, plt
    fig, ax = plt.subplots()
    ax.plot([1, 2], [3, 4])
    out = save_figure(fig, tmp_path / "test.png")
    assert out == tmp_path / "test.png"
    assert out.is_file()
    assert out.stat().st_size > 0


def test_save_figure_creates_nested_parent(tmp_path):
    from src._tasklib.plotting import save_figure, plt
    fig, _ = plt.subplots()
    out = save_figure(fig, tmp_path / "a" / "b" / "fig.png")
    assert out.is_file()


def test_save_figure_closes_figure(tmp_path):
    from src._tasklib.plotting import save_figure, plt
    fig, _ = plt.subplots()
    save_figure(fig, tmp_path / "closed.png")
    # After close, fig.number should raise an error if we try to use it
    assert not plt.fignum_exists(fig.number)
```

- [ ] **Step 2: Run tests — expect ImportError**

```
pytest tests/test_tasklib_plotting.py -v
```

Expected: 3 failures with `ModuleNotFoundError`

- [ ] **Step 3: Create `src/_tasklib/plotting.py`**

```python
"""Shared matplotlib backend setup and figure-save helper."""

from __future__ import annotations

import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(_ROOT / ".mpl-cache"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def save_figure(fig, out_path: str | Path, *, dpi: int = 300) -> Path:
    """Save figure as PNG with paper-standard settings and close it."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out
```

- [ ] **Step 4: Run tests — expect all 3 to pass**

```
pytest tests/test_tasklib_plotting.py -v
```

Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/_tasklib/plotting.py tests/test_tasklib_plotting.py
git commit -m "feat: add _tasklib/plotting with Agg backend and save_figure"
```

---

## Task 3: Update callers — replace duplicated `load_tile_ids` / `load_cv_splits`

**Files:**
- Modify: `src/a1_probe_linear/main.py`
- Modify: `src/a1_probe_encoders/main.py`
- Modify: `src/a1_generated_probe/main.py`

- [ ] **Step 1: Update `src/a1_probe_linear/main.py`**

Remove the local `load_tile_ids` definition (lines 26-28) and add to the existing `_tasklib` import:

Old imports block (top of file):
```python
from src._tasklib.io import ensure_directory, write_json
from src._tasklib.tile_ids import parse_tile_id, tile_ids_sha1
```

New imports block:
```python
from src._tasklib.io import ensure_directory, load_cv_splits, load_tile_ids, write_json
from src._tasklib.tile_ids import parse_tile_id, tile_ids_sha1
```

Remove this function entirely (it is now in `_tasklib`):
```python
def load_tile_ids(tile_ids_path: str | Path) -> list[str]:
    """Load newline-delimited tile IDs."""
    return [line.strip() for line in Path(tile_ids_path).read_text(encoding="utf-8").splitlines() if line.strip()]
```

Remove this function entirely (now in `_tasklib`):
```python
def load_cv_splits(tile_ids: list[str], cv_splits_path: str | Path) -> list[dict[str, list[int]]]:
    """Load and validate saved CV splits."""
    payload = json.loads(Path(cv_splits_path).read_text(encoding="utf-8"))
    expected_hash = tile_ids_sha1(tile_ids)
    if payload.get("tile_ids_sha1") != expected_hash:
        raise ValueError("tile_ids.txt does not match the saved CV split hash")
    return list(payload["splits"])
```

Also remove `json` from the imports at the top if it is only used by `load_cv_splits` — verify first; keep if used elsewhere.

- [ ] **Step 2: Update `src/a1_probe_encoders/main.py`**

Remove lines 35–58 (the two try/except wrapper functions):
```python
def load_tile_ids(tile_ids_path: str | Path) -> list[str]:
    try:
        from src.a1_probe_linear.main import load_tile_ids as imported_load_tile_ids
        return imported_load_tile_ids(tile_ids_path)
    except Exception:
        return [...]

def load_cv_splits(tile_ids: list[str], cv_splits_path: str | Path) -> list[dict[str, list[int]]]:
    try:
        from src.a1_probe_linear.main import load_cv_splits as imported_load_cv_splits
        return imported_load_cv_splits(tile_ids, cv_splits_path)
    except Exception:
        payload = json.loads(...)
        ...
```

Add to the existing `_tasklib` import near the top:
```python
from src._tasklib.io import ensure_directory, load_cv_splits, load_tile_ids, write_json
```

Also remove `tile_ids_sha1` from `_tasklib.tile_ids` import if it was only used by the removed `load_cv_splits` fallback — verify.

- [ ] **Step 3: Update `src/a1_generated_probe/main.py`**

Remove the local `load_tile_ids` definition (line 53-54):
```python
def load_tile_ids(tile_ids_path: str | Path) -> list[str]:
    return [line.strip() for line in Path(tile_ids_path).read_text(encoding="utf-8").splitlines() if line.strip()]
```

Add to existing `_tasklib` import:
```python
from src._tasklib.io import ensure_directory, load_tile_ids, write_json
```

- [ ] **Step 4: Run full test suite**

```
pytest tests/ -v --tb=short
```

Expected: all tests that passed before still pass; no regressions.

- [ ] **Step 5: Commit**

```bash
git add src/a1_probe_linear/main.py src/a1_probe_encoders/main.py src/a1_generated_probe/main.py
git commit -m "refactor: consolidate load_tile_ids and load_cv_splits into _tasklib/io"
```

---

## Task 4: Replace duplicated `write_csv` DictWriter blocks with `write_csv` utility

Six files contain an identical `csv.DictWriter` pattern. Replace each with the new utility.

**Files:**
- Modify: `src/a0_tradeoff_scatter/collect.py`
- Modify: `src/a0_visibility_map/collect.py`
- Modify: `src/a1_generated_probe/main.py`
- Modify: `src/a1_probe_encoders/main.py`
- Modify: `src/a1_probe_mlp/main.py`
- Modify: `src/a2_decomposition/metrics.py`

For each file, the change follows the same pattern. Below is the template; apply it to all six.

**Template — before:**
```python
import csv
...
with out_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow({...})
```

**Template — after:**
```python
from src._tasklib.io import ..., write_csv   # add write_csv to existing _tasklib import
...
write_csv(rows, fieldnames, out_path)
```

### `src/a0_tradeoff_scatter/collect.py` — `write_tradeoff_csv`

- [ ] **Step 1: Find `write_tradeoff_csv` function** and identify its DictWriter block.

Add `write_csv` to the `_tasklib` import at the top of `collect.py`:
```python
from src._tasklib.io import ensure_directory, write_csv, write_json
```

Replace the DictWriter block inside `write_tradeoff_csv` with:
```python
return write_csv(rows_dicts, fieldnames, out_path)
```

where `rows_dicts` is a list of dicts assembled from each row's fields, and `fieldnames` is the list of column names.

Also remove the `import csv` statement from the top of `collect.py` if it is now unused.

### `src/a0_visibility_map/collect.py` — `write_visibility_summary_csv`

- [ ] **Step 2:** Same pattern as above. Add `write_csv` to `_tasklib` import; replace DictWriter block; remove `import csv` if unused.

### `src/a1_generated_probe/main.py` — `_write_comparison_csv`

- [ ] **Step 3:** Same pattern. The `write_csv` import was already added in Task 3 Step 3.

### `src/a1_probe_encoders/main.py` — `_write_encoder_comparison_csv`

- [ ] **Step 4:** Same pattern. `write_csv` already in import from Task 3 Step 2.

### `src/a1_probe_mlp/main.py` — `_write_comparison_csv`

- [ ] **Step 5:** Add `write_csv` to the `_tasklib` import (or add via import from `a1_probe_linear` — but prefer `_tasklib` directly):
```python
from src._tasklib.io import write_csv
```
Replace the DictWriter block.

### `src/a2_decomposition/metrics.py` — `write_summary_csv`

- [ ] **Step 6:** Add `write_csv` to `_tasklib` import; replace DictWriter block; remove `import csv` from `metrics.py` if now unused.

- [ ] **Step 7: Run full test suite**

```
pytest tests/ -v --tb=short
```

Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add src/a0_tradeoff_scatter/collect.py src/a0_visibility_map/collect.py \
        src/a1_generated_probe/main.py src/a1_probe_encoders/main.py \
        src/a1_probe_mlp/main.py src/a2_decomposition/metrics.py
git commit -m "refactor: replace DictWriter boilerplate with _tasklib.io.write_csv"
```

---

## Task 5: Consolidate matplotlib boilerplate in render modules

**Files:**
- Modify: `src/a0_tradeoff_scatter/render.py`
- Modify: `src/a0_visibility_map/render.py`

Both files start with an identical 10-line block that sets `MPLCONFIGDIR` and switches to the Agg backend. Replace it with a single import from `_tasklib.plotting`.

- [ ] **Step 1: Update `src/a0_tradeoff_scatter/render.py`**

Remove these lines at the top of the file:
```python
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mpl-cache"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
```

Replace with:
```python
from src._tasklib.plotting import plt, save_figure
```

In `render_tradeoff_panel`, remove the `out_path.parent.mkdir(...)` line (handled by `save_figure`), and replace:
```python
fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
plt.close(fig)
return out_path
```
with:
```python
return save_figure(fig, out_path, dpi=dpi)
```

- [ ] **Step 2: Update `src/a0_visibility_map/render.py`**

Same as Step 1 for `render_visibility_chart`.

Remove same boilerplate block at top; add:
```python
from src._tasklib.plotting import plt, save_figure
```

Replace `out_path.parent.mkdir(...)`, `fig.savefig(out_path, dpi=dpi, bbox_inches="tight")`, `plt.close(fig)`, `return out_path` with:
```python
return save_figure(fig, out_path, dpi=dpi)
```

- [ ] **Step 3: Run full test suite**

```
pytest tests/ -v --tb=short
```

Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add src/a0_tradeoff_scatter/render.py src/a0_visibility_map/render.py
git commit -m "refactor: replace repeated matplotlib Agg boilerplate with _tasklib.plotting"
```

---

## Task 6: Standardise `paper_figures/` — consistent style and save

All `paper_figures/` figure builders must:
1. Call `apply_style()` from `src.paper_figures.style` before building axes.
2. Use `save_figure(fig, out_path, dpi=dpi)` for all final saves.
3. Not set `facecolor`, `format`, or `bbox_inches` individually (these are now in `save_figure`).

**Files:**
- Modify: `src/paper_figures/fig_inverse_decoding.py`
- Modify: `src/paper_figures/fig_uni_tme_decomposition.py`
- Modify: `src/paper_figures/fig_ablation_grid.py`

### `src/paper_figures/fig_inverse_decoding.py`

- [ ] **Step 1:** Add to imports:
```python
from src._tasklib.plotting import save_figure
from src.paper_figures.style import apply_style
```

- [ ] **Step 2:** At the start of `build_inverse_decoding_figure` (the public entry-point function), add:
```python
apply_style()
```

- [ ] **Step 3:** Find all `fig.savefig(...)` + `plt.close(fig)` pairs and replace with:
```python
save_figure(fig, out_path, dpi=dpi)
```

Remove any standalone `out_path.parent.mkdir(...)` that preceded the save (now inside `save_figure`).

### `src/paper_figures/fig_uni_tme_decomposition.py`

- [ ] **Step 4:** The file currently imports `plt` from `tools.ablation_report.shared`. Add:
```python
from src._tasklib.plotting import save_figure
from src.paper_figures.style import apply_style
```

- [ ] **Step 5:** At the start of `save_uni_tme_decomposition_figure`, add:
```python
apply_style()
```

- [ ] **Step 6:** Replace any `fig.savefig(out_png, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")` + `plt.close(fig)` with:
```python
save_figure(fig, out_png, dpi=dpi)
```

### `src/paper_figures/fig_ablation_grid.py`

- [ ] **Step 7:** The file delegates to `render_ablation_grid_figure` from `tools/`. Check whether that function calls `fig.savefig` internally or returns a fig object.

- If it saves internally: ensure it receives `out_png` and the function returns the path. No `save_figure` needed in the wrapper — but verify the save args match the standard (format png, bbox_inches tight, facecolor white). If not, update the call-site in `fig_ablation_grid.py` to pass those kwargs.
- If it returns a fig: replace the save call in `build_representative_ablation_grid` with `save_figure(fig, out_png, dpi=dpi)`.

Add `apply_style()` call at the start of `build_representative_ablation_grid`.

- [ ] **Step 8: Run full test suite**

```
pytest tests/ -v --tb=short
```

Expected: all pass, including `tests/test_fig_inverse_decoding.py` and `tests/test_fig_uni_tme_decomposition.py`.

- [ ] **Step 9: Commit**

```bash
git add src/paper_figures/fig_inverse_decoding.py \
        src/paper_figures/fig_uni_tme_decomposition.py \
        src/paper_figures/fig_ablation_grid.py
git commit -m "refactor: standardise paper_figures style and save via _tasklib.plotting"
```

---

## Task 7: Remove remaining dead code

- [ ] **Step 1: Remove `summarize_probe_results` fallback in `src/a1_probe_encoders/main.py`**

The function currently tries to import from `a1_probe_linear` and defines a fallback. After Task 3, `load_tile_ids` and `load_cv_splits` come from `_tasklib`. But `summarize_probe_results` is still imported from `a1_probe_linear`.

Find this pattern in `a1_probe_encoders/main.py`:
```python
def summarize_probe_results(...):
    try:
        from src.a1_probe_linear.main import summarize_probe_results as imported_fn
        return imported_fn(...)
    except Exception:
        ...  # fallback implementation
```

Replace with a direct import at the top of the file:
```python
from src.a1_probe_linear.main import summarize_probe_results
```

And remove the local wrapper function entirely.

- [ ] **Step 2: Remove `_default_target_names` duplicate**

`_default_target_names()` is defined identically in `src/a1_generated_probe/main.py` and `src/a1_probe_encoders/main.py`. Add a canonical copy to `src/a1_probe_linear/main.py` (it's the shared probe utilities module) and import it in the other two.

In `src/a1_probe_linear/main.py`, add near the top:
```python
def _default_target_names(n: int) -> list[str]:
    return [f"target_{i}" for i in range(n)]
```

In `src/a1_probe_encoders/main.py`, remove local definition and add:
```python
from src.a1_probe_linear.main import _default_target_names
```

In `src/a1_generated_probe/main.py`, remove local definition and add:
```python
from src.a1_probe_linear.main import _default_target_names
```

- [ ] **Step 3: Remove unused `csv` top-level imports**

After Tasks 3–4, several files may still have `import csv` at the top but no longer use `csv.DictWriter` directly. Scan and remove:
- `src/a0_tradeoff_scatter/collect.py`
- `src/a0_visibility_map/collect.py`
- `src/a1_generated_probe/main.py`
- `src/a1_probe_encoders/main.py`
- `src/a1_probe_mlp/main.py`
- `src/a2_decomposition/metrics.py`
- `src/a1_probe_linear/main.py` (check if `csv` is still used for `write_probe_results`)

For each: grep for `csv.` in the file body. If only `csv.DictWriter` was used (now replaced), remove the `import csv` line.

- [ ] **Step 4: Run full test suite**

```
pytest tests/ -v --tb=short
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/a1_probe_encoders/main.py src/a1_probe_linear/main.py src/a1_generated_probe/main.py \
        src/a0_tradeoff_scatter/collect.py src/a0_visibility_map/collect.py \
        src/a1_probe_mlp/main.py src/a2_decomposition/metrics.py
git commit -m "refactor: remove dead wrapper functions and unused csv imports"
```

---

## Self-Review

**Spec coverage:**
- Extract `load_tile_ids` → Task 1 + Task 3 ✓
- Extract `load_cv_splits` → Task 1 + Task 3 ✓
- Extract `write_csv` → Task 1 + Task 4 ✓
- Centralise matplotlib boilerplate → Task 2 + Task 5 ✓
- Standardise figure save → Task 6 ✓
- Remove dead wrappers (`summarize_probe_results`, `_default_target_names`) → Task 7 ✓
- Remove unused `import csv` → Task 7 ✓
- Paper figures: apply_style + consistent PNG args → Task 6 ✓

**No placeholders detected.**

**Type consistency:** `save_figure(fig, out_path, dpi=dpi)` used identically in Tasks 5, 6, 7 callers. `write_csv(rows, fieldnames, out_path)` signature stable across Task 1 definition and Task 4 call sites.
