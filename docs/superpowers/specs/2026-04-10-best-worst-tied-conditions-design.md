# Design: Tied Best/Worst Conditions in Ablation Report

**Date:** 2026-04-10
**Scope:** `tools/render_ablation_html_report.py`, `tools/summarize_ablation_report.py`, `tests/test_summarize_ablation_report.py`

## Problem

`summarize_best_worst` returns a single best and single worst condition per metric. When multiple conditions cluster near the same value (e.g., IoU ~0.65 across 4 conditions), the "best" label is arbitrary and misleading — a reader cannot tell whether there is a real winner or a near-tie.

## Design

### Tie rule

A condition is **tied-best** if its mean falls within 1 SD of the best condition's mean:

```
tied_best = [cond for cond, mean, sd in items if mean >= best_mean - best_sd]
```

Symmetric for worst:

```
tied_worst = [cond for cond, mean, sd in items if mean <= worst_mean + worst_sd]
```

where `best_sd` / `worst_sd` is the per-condition SD across tiles. Lists are sorted best→worst.

### Section 1: Data model (`summarize_ablation_report.py`)

**`summarize_best_worst`** return type changes from:

```python
dict[str, dict[str, str | float]]
# {"iou": {"best_condition": "ct+cs", "best_value": 0.654,
#           "worst_condition": "ct", "worst_value": 0.581}}
```

to:

```python
dict[str, dict[str, list[tuple[str, float, float]]]]
# {"iou": {"best": [("ct+cs+env", 0.654, 0.041), ("ct+cs+vas+env", 0.652, 0.039)],
#           "worst": [("ct", 0.581, 0.058)]}}
```

Each tuple is `(condition_key, mean, sd)`. `DatasetSummary.best_worst` type annotation updated to match.

The function gains an optional parameter `condition_stats: dict[str, dict[str, tuple[float, float]]] | None = None`. When provided (HTML report path), SDs are used for the tie rule. When `None` (markdown report path in `build_report`, which does not compute per-condition stats), each condition is treated as its own group — no ties, same behavior as before.

### Section 2: HTML rendering (`render_ablation_html_report.py`)

`render_comparison_table` and `render_best_worst_table` updated:

- Metric name cell gets `rowspan = max(len(best), len(worst))`
- Best conditions fill left side, worst fill right side; shorter list gets blank cells for extra rows
- When `len(best) > 1`: metric cell shows `(N tied)` in small italic below metric name
- Value cells: `0.654 ± 0.041` format
- Journal style: no background colors; top/bottom table borders `2px solid #000`; header row `border-bottom: 1px solid #000`; metric group separator `1px solid #000`
- Column headers: `Metric | Best conditions (CT/CS/Vas/Env) | Mean ± SD | Worst conditions | Mean ± SD`

### Section 3: Matplotlib figure (`build_comparison_table_figure`)

- Build rows by iterating `best` and `worst` lists in parallel (`zip_longest`); first row of each metric group includes dataset and metric name, subsequent rows leave those blank
- Condition text via existing `condition_indicator_text()` (● ○ symbols)
- Value formatted as `0.654 ± 0.041`
- Figure height recalculated from total row count (sum of `max(len(best), len(worst))` across all metrics and datasets)
- Column widths adjusted: condition columns slightly narrower, value columns slightly wider to accommodate `±` format

### Section 4: Tests (`tests/test_summarize_ablation_report.py`)

Two new test cases:

- **`test_summarize_best_worst_tied`**: `condition_means` where 3 conditions cluster within 1 SD of the best. Assert `best` list contains all 3, sorted correctly; `worst` list has 1 entry.
- **`test_summarize_best_worst_clear_winner`**: `condition_means` with a clear gap (>1 SD) between best and second-best. Assert `best` list has exactly 1 entry.

## Files changed

| File | Change |
|------|--------|
| `tools/summarize_ablation_report.py` | Update `summarize_best_worst` signature and return type |
| `diffusion/data/datasets/...` | *(none)* |
| `tools/render_ablation_html_report.py` | Update `render_best_worst_table`, `render_comparison_table`, `build_comparison_table_figure`, `DatasetSummary.best_worst` annotation |
| `tests/test_summarize_ablation_report.py` | Add two new test cases (create file if absent) |

## Out of scope

- No changes to the markdown report path (`summarize_ablation_report.py:build_report`) beyond the updated return type.
- No changes to other sections of the HTML report (cardinality, LOO, channel effects).
