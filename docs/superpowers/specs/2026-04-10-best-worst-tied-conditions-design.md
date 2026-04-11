# Design: Per-Metric Ranked Ablation Table

**Date:** 2026-04-10 (revised 2026-04-10)
**Scope:** `tools/render_ablation_html_report.py`, `tools/summarize_ablation_report.py`, `tests/test_summarize_ablation_report.py`

## Problem

The existing ablation table has two issues:

1. **Single best/worst per metric** — when multiple conditions cluster near the same value, the "best" label is arbitrary and misleading.
2. **Cross-metric value comparison** — a single combined table puts IoU 0.654 and LPIPS 0.312 in the same value column. Readers instinctively compare them, but the scales are incomparable.

## Design

### Layout principle

**One small table per metric per dataset.** Each table has its own isolated value column — cross-metric comparison is impossible by construction.

Within a dataset section, metric tables sit side-by-side (CSS flex). Each table shows:
- Ranks 1–3: top conditions sorted best→worst
- A dashed separator row ("···")
- Ranks (N−2)–N: worst conditions sorted from least-bad to worst (N = total conditions)

### What this table answers

- **Exact ±SD numbers** for the extremes — hard to read off scatter plots.
- **How much does condition choice matter?** The rank-1 vs rank-N gap with overlapping SDs.
- **Cross-dataset consistency** — does the same condition win on IoU in both Paired and Unpaired?
- **Which single channel is most harmful** — the worst-3 side reveals which minimal conditions fail.

This is complementary to scatter plots (which show all 15 conditions and pairwise tradeoffs), not redundant.

### Section 1: Data model (`summarize_ablation_report.py`)

**`summarize_best_worst`** gains two parameters and changes its return type:

```python
def summarize_best_worst(
    condition_means: dict[str, dict[str, float]],
    metric_keys: list[str],
    condition_stats: dict[str, dict[str, tuple[float, float]]] | None = None,
    n: int = 3,
) -> dict[str, dict[str, list[tuple[str, float, float]] | int]]:
```

Return value per metric:
```python
{
    "best":  [(cond_key, mean, sd), ...],  # top n, sorted best→worst
    "worst": [(cond_key, mean, sd), ...],  # bottom n, sorted rank (N-n+1)→N
    "total": int,                           # total condition count (for rank display)
}
```

`condition_stats` provides per-condition SDs (already computed upstream in the HTML renderer).
When `None` (markdown report path), SD defaults to `0.0`.

No tolerance-band or tied logic — pure ranking. `best_worst_notes` requires no logic changes
(it checks membership in `summary["best"]`, which still works correctly with top-3).

`DatasetSummary.best_worst` type annotation updated to match.

### Section 2: HTML rendering (`render_ablation_html_report.py`)

`render_comparison_table` is replaced. New output structure:

```
<div class="comparison-section">
  <div class="dataset-section">
    <div class="dataset-label">Paired</div>
    <div class="metric-group">
      <div class="metric-wrap">
        <div class="metric-caption">IoU ↑</div>
        <table class="ranked-table">
          <thead> Rank | CT CS Vas Env | Mean ± SD </thead>
          <tbody>
            rank 1 row
            rank 2 row
            rank 3 row
            separator row (···)
            rank 13 row
            rank 14 row
            rank 15 row
          </tbody>
        </table>
      </div>
      <!-- one .metric-wrap per metric -->
    </div>
  </div>
  <!-- one .dataset-section per DatasetSummary -->
</div>
```

Style: Times New Roman, 10px, 2px padding, 1.5px solid borders top/bottom of table,
1px solid header rule, 1px dashed separator borders. No background colors.

`render_best_worst_table` updated with same structure (single-dataset variant).

CSS block in the HTML template gains: `.comparison-section`, `.dataset-section`,
`.dataset-label`, `.metric-group`, `.metric-wrap`, `.metric-caption`,
`.ranked-table` (and `thead`, `tbody tr:last-child`, `th`, `td`, `.rank-cell`,
`.condition-cell`, `.value-cell`, `.rank-sep`).

### Section 3: Matplotlib figure (`build_comparison_table_figure`)

Rows built per-metric per-dataset, with a blank separator row between top-3 and worst-3.
Rank numbers in column 0. Condition indicator text via existing `condition_indicator_text()`.
Value formatted as `0.654 ± 0.041`. Figure height recalculated from total row count.
Column widths adjusted to accommodate rank column.

### Section 4: Tests (`tests/test_summarize_ablation_report.py`)

Changes:
- Update `test_summary_helpers_capture_expected_metric_directions`: change four assertions
  that reference old keys (`best_condition`, `best_value`) to new tuple indexing.
- Add `test_summarize_best_worst_top3`: verify top-3 are correctly ranked and SDs included.
- Add `test_summarize_best_worst_worst3`: verify worst-3 ordering and total count.

## Files changed

| File | Change |
|------|--------|
| `tools/summarize_ablation_report.py` | Replace `summarize_best_worst` with top-N/worst-N ranking; no changes to `best_worst_notes` |
| `tools/render_ablation_html_report.py` | Replace `render_best_worst_table`, `render_comparison_table`, `build_comparison_table_figure`, `DatasetSummary.best_worst` annotation; add CSS |
| `tests/test_summarize_ablation_report.py` | Update existing test; add two new ranking tests |

## Out of scope

- No changes to the markdown report path (`summarize_ablation_report.py:build_report`).
- No changes to other HTML report sections (cardinality, LOO, channel effects).
- No tied-condition SD logic.
