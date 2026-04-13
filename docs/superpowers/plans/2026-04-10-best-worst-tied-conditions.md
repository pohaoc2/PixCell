# Per-Metric Ranked Ablation Table — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the single best/worst condition per metric with a ranked top-3 / worst-3 display using one isolated table per metric, eliminating cross-metric value comparison.

**Architecture:** Update `summarize_best_worst` in the summary layer to return top-N and worst-N condition lists with SDs; replace HTML rendering functions with a flex-layout of per-metric tables; update the matplotlib figure; update tests.

**Tech Stack:** Python 3.12, pytest, matplotlib, HTML/CSS in the report renderer.

---

### Task 1: Update `summarize_best_worst`

**Files:**
- Modify: `tools/summarize_ablation_report.py`

The function is replaced with a pure top-N / worst-N ranking. No tied logic. `condition_stats` provides per-condition SDs (used in the value column). When `None`, SDs default to `0.0`. `best_worst_notes` requires **no changes** — it checks membership in `summary["best"]`, which still works.

- [ ] **Step 1: Replace `summarize_best_worst` in `tools/summarize_ablation_report.py`**

Find and replace the entire function (currently lines 124–146). Also remove the `_get_cond_sd` helper if it was added in a prior pass. New implementation:

```python
def summarize_best_worst(
    condition_means: dict[str, dict[str, float]],
    metric_keys: list[str],
    condition_stats: dict[str, dict[str, tuple[float, float]]] | None = None,
    n: int = 3,
) -> dict[str, dict[str, list[tuple[str, float, float]] | int]]:
    summary: dict[str, dict[str, list[tuple[str, float, float]] | int]] = {}
    for metric_key in metric_keys:
        items = [
            (cond_key, metrics[metric_key])
            for cond_key, metrics in condition_means.items()
            if metric_key in metrics
        ]
        if not items:
            continue
        spec = METRIC_SPEC_BY_KEY[metric_key]
        sorted_items = sorted(items, key=lambda t: t[1], reverse=spec.higher_is_better)

        def with_sd(cond_key: str, value: float) -> tuple[str, float, float]:
            sd = 0.0
            if condition_stats is not None:
                sd = condition_stats.get(cond_key, {}).get(metric_key, (0.0, 0.0))[1]
            return (cond_key, value, sd)

        top_n = [with_sd(ck, v) for ck, v in sorted_items[:n]]
        worst_start = max(n, len(sorted_items) - n)
        worst_n = [with_sd(ck, v) for ck, v in sorted_items[worst_start:]]
        summary[metric_key] = {"best": top_n, "worst": worst_n, "total": len(items)}
    return summary
```

- [ ] **Step 2: Commit**

```bash
git add tools/summarize_ablation_report.py
git commit -m "refactor: replace summarize_best_worst with top-N/worst-N ranking"
```

---

### Task 2: Update tests for `summarize_best_worst`

**Files:**
- Modify: `tests/test_summarize_ablation_report.py`

- [ ] **Step 1: Update the existing assertions in `test_summary_helpers_capture_expected_metric_directions`**

Find these four lines (around line 100):

```python
    assert best_worst["aji"]["best_condition"] == full_key
    assert best_worst["pq"]["best_condition"] == full_key
    assert "microenv" in str(best_worst["fud"]["best_condition"])
    assert "cell_state" not in str(best_worst["fud"]["best_condition"])
```

Replace with:

```python
    assert best_worst["aji"]["best"][0][0] == full_key
    assert best_worst["pq"]["best"][0][0] == full_key
    assert "microenv" in best_worst["fud"]["best"][0][0]
    assert "cell_state" not in best_worst["fud"]["best"][0][0]
```

- [ ] **Step 2: Run the updated test**

```bash
cd /home/pohaoc2/UW/bagherilab/PixCell
pytest tests/test_summarize_ablation_report.py::test_summary_helpers_capture_expected_metric_directions -v
```

Expected: PASS

- [ ] **Step 3: Add `test_summarize_best_worst_top3` at the bottom of the file**

```python
def test_summarize_best_worst_top3() -> None:
    # IoU (higher_is_better): 5 conditions; top 3 should be sorted descending with correct SDs.
    condition_means = {
        "ct+cs+env": {"iou": 0.654},
        "ct+cs+vas+env": {"iou": 0.652},
        "ct+vas+env": {"iou": 0.651},
        "ct+cs": {"iou": 0.610},
        "ct": {"iou": 0.581},
    }
    condition_stats = {
        "ct+cs+env": {"iou": (0.654, 0.041)},
        "ct+cs+vas+env": {"iou": (0.652, 0.039)},
        "ct+vas+env": {"iou": (0.651, 0.043)},
        "ct+cs": {"iou": (0.610, 0.050)},
        "ct": {"iou": (0.581, 0.058)},
    }
    result = summarize_best_worst(condition_means, ["iou"], condition_stats, n=3)

    best = result["iou"]["best"]
    assert len(best) == 3
    assert best[0][0] == "ct+cs+env"
    assert best[0][1] == pytest.approx(0.654)
    assert best[0][2] == pytest.approx(0.041)
    assert best[1][0] == "ct+cs+vas+env"
    assert best[2][0] == "ct+vas+env"
    assert result["iou"]["total"] == 5
```

- [ ] **Step 4: Add `test_summarize_best_worst_worst3` immediately after**

```python
def test_summarize_best_worst_worst3() -> None:
    # IoU (higher_is_better): 5 conditions; worst 3 should start at rank 3 (no overlap with top 3).
    # With n=3 and total=5: worst_start = max(3, 5-3) = max(3, 2) = 3 → sorted_items[3:]
    # = ["ct+cs" (rank 4), "ct" (rank 5)] — only 2 entries since 5-3=2 < n=3.
    condition_means = {
        "ct+cs+env": {"iou": 0.654},
        "ct+cs+vas+env": {"iou": 0.652},
        "ct+vas+env": {"iou": 0.651},
        "ct+cs": {"iou": 0.610},
        "ct": {"iou": 0.581},
    }
    condition_stats = {
        "ct+cs+env": {"iou": (0.654, 0.041)},
        "ct+cs+vas+env": {"iou": (0.652, 0.039)},
        "ct+vas+env": {"iou": (0.651, 0.043)},
        "ct+cs": {"iou": (0.610, 0.050)},
        "ct": {"iou": (0.581, 0.058)},
    }
    result = summarize_best_worst(condition_means, ["iou"], condition_stats, n=3)

    worst = result["iou"]["worst"]
    # No overlap: worst does not include any of the top-3 conditions
    worst_keys = [t[0] for t in worst]
    assert "ct+cs+env" not in worst_keys
    assert "ct+cs+vas+env" not in worst_keys
    assert "ct+vas+env" not in worst_keys
    # Last entry is the true worst
    assert worst[-1][0] == "ct"
    assert worst[-1][1] == pytest.approx(0.581)
    assert worst[-1][2] == pytest.approx(0.058)
```

- [ ] **Step 5: Run all tests**

```bash
pytest tests/test_summarize_ablation_report.py -v
```

Expected: all tests PASS

- [ ] **Step 6: Commit**

```bash
git add tests/test_summarize_ablation_report.py
git commit -m "test: update and add tests for top-N/worst-N ranking in summarize_best_worst"
```

---

### Task 3: Update `DatasetSummary` type annotation and call site

**Files:**
- Modify: `tools/ablation_report`

- [ ] **Step 1: Update `DatasetSummary.best_worst` type annotation**

Find (around line 133):

```python
    best_worst: dict[str, dict[str, str | float]]
```

Replace with:

```python
    best_worst: dict[str, dict[str, list[tuple[str, float, float]] | int]]
```

- [ ] **Step 2: Pass `condition_stats` to `summarize_best_worst` at the call site**

Find (around line 830):

```python
    best_worst = summarize_best_worst(condition_means, metric_keys)
```

Replace with:

```python
    best_worst = summarize_best_worst(condition_means, metric_keys, condition_stats)
```

(`condition_stats` is already computed a few lines before this call.)

- [ ] **Step 3: Commit**

```bash
git add tools/ablation_report
git commit -m "feat: pass condition_stats to summarize_best_worst for SD values"
```

---

### Task 4: Replace HTML rendering functions and add CSS

**Files:**
- Modify: `tools/ablation_report`

The existing `render_comparison_table` and `render_best_worst_table` are replaced with a flex-layout of per-metric tables. New CSS classes are added to the HTML template's `<style>` block.

- [ ] **Step 1: Add CSS for the new table layout**

Find the `<style>` block in the HTML template string inside `build_html_report` (or wherever the shared CSS is defined). Add the following rules after the existing table styles:

```css
/* Per-metric ranked tables */
.comparison-section { margin-bottom: 24px; }
.dataset-section { margin-bottom: 16px; }
.dataset-label {
  font-family: 'Times New Roman', serif;
  font-size: 11px;
  font-weight: bold;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  border-bottom: 2px solid #000;
  padding-bottom: 2px;
  margin-bottom: 6px;
}
.metric-group { display: flex; gap: 16px; flex-wrap: wrap; }
.metric-wrap { flex: 1; min-width: 160px; }
.metric-caption {
  font-family: 'Times New Roman', serif;
  font-size: 10px;
  font-weight: bold;
  margin-bottom: 2px;
}
table.ranked-table {
  border-collapse: collapse;
  width: 100%;
  font-family: 'Times New Roman', serif;
  font-size: 10px;
}
table.ranked-table thead tr {
  border-top: 1.5px solid #000;
  border-bottom: 1px solid #000;
}
table.ranked-table tbody tr:last-child td { border-bottom: 1.5px solid #000; }
table.ranked-table th, table.ranked-table td { padding: 2px 4px; }
table.ranked-table th.col-rank { text-align: center; width: 28px; }
table.ranked-table th.col-cond { text-align: left; }
table.ranked-table th.col-val  { text-align: right; white-space: nowrap; }
table.ranked-table td.rank-cell {
  text-align: center;
  color: #555;
  font-size: 9px;
}
table.ranked-table td.condition-cell { white-space: nowrap; }
table.ranked-table td.value-cell { text-align: right; white-space: nowrap; }
table.ranked-table tr.rank-sep td {
  text-align: center;
  color: #aaa;
  font-size: 9px;
  padding: 1px 4px;
  border-top: 1px dashed #ccc;
  border-bottom: 1px dashed #ccc;
}
```

- [ ] **Step 2: Replace `render_best_worst_table`**

Find and replace the entire function (starting at `def render_best_worst_table`):

```python
def render_best_worst_table(summary: DatasetSummary) -> str:
    metric_tables: list[str] = []
    for metric_key in comparison_metric_keys(summary):
        record = summary.best_worst.get(metric_key)
        if not record:
            continue
        top_list: list[tuple[str, float, float]] = record["best"]
        worst_list: list[tuple[str, float, float]] = record["worst"]
        total: int = record["total"]

        spec = METRIC_SPEC_BY_KEY[metric_key]
        direction = "&uarr;" if spec.higher_is_better else "&darr;"

        rows: list[str] = []
        for rank, (cond_key, mean, sd) in enumerate(top_list, start=1):
            rows.append(
                f"<tr>"
                f"<td class='rank-cell'>{rank}</td>"
                f"<td class='condition-cell'>{render_condition_glyph(cond_key)}</td>"
                f"<td class='value-cell'>{mean:.3f} &plusmn; {sd:.3f}</td>"
                f"</tr>"
            )
        rows.append("<tr class='rank-sep'><td colspan='3'>&#xB7;&#xB7;&#xB7;</td></tr>")
        for rank, (cond_key, mean, sd) in enumerate(
            worst_list, start=total - len(worst_list) + 1
        ):
            rows.append(
                f"<tr>"
                f"<td class='rank-cell'>{rank}</td>"
                f"<td class='condition-cell'>{render_condition_glyph(cond_key)}</td>"
                f"<td class='value-cell'>{mean:.3f} &plusmn; {sd:.3f}</td>"
                f"</tr>"
            )

        col_header = html.escape(condition_order_label())
        metric_tables.append(
            f"<div class='metric-wrap'>"
            f"<div class='metric-caption'>{html.escape(humanize_token(metric_key))} {direction}</div>"
            f"<table class='ranked-table'>"
            f"<thead><tr>"
            f"<th class='col-rank'>Rank</th>"
            f"<th class='col-cond'>{col_header}</th>"
            f"<th class='col-val'>Mean &plusmn; SD</th>"
            f"</tr></thead>"
            f"<tbody>{''.join(rows)}</tbody>"
            f"</table></div>"
        )

    return (
        f"<div class='comparison-section'>"
        f"<div class='dataset-section'>"
        f"<div class='dataset-label'>{html.escape(summary.title)}</div>"
        f"<div class='metric-group'>{''.join(metric_tables)}</div>"
        f"</div></div>"
    )
```

- [ ] **Step 3: Replace `render_comparison_table`**

Find and replace the entire function (starting at `def render_comparison_table`):

```python
def render_comparison_table(summaries: list[DatasetSummary]) -> str:
    dataset_sections: list[str] = []
    for summary in summaries:
        metric_tables: list[str] = []
        for metric_key in comparison_metric_keys(summary):
            record = summary.best_worst.get(metric_key)
            if not record:
                continue
            top_list: list[tuple[str, float, float]] = record["best"]
            worst_list: list[tuple[str, float, float]] = record["worst"]
            total: int = record["total"]

            spec = METRIC_SPEC_BY_KEY[metric_key]
            direction = "&uarr;" if spec.higher_is_better else "&darr;"

            rows: list[str] = []
            for rank, (cond_key, mean, sd) in enumerate(top_list, start=1):
                rows.append(
                    f"<tr>"
                    f"<td class='rank-cell'>{rank}</td>"
                    f"<td class='condition-cell'>{render_condition_glyph(cond_key)}</td>"
                    f"<td class='value-cell'>{mean:.3f} &plusmn; {sd:.3f}</td>"
                    f"</tr>"
                )
            rows.append("<tr class='rank-sep'><td colspan='3'>&#xB7;&#xB7;&#xB7;</td></tr>")
            for rank, (cond_key, mean, sd) in enumerate(
                worst_list, start=total - len(worst_list) + 1
            ):
                rows.append(
                    f"<tr>"
                    f"<td class='rank-cell'>{rank}</td>"
                    f"<td class='condition-cell'>{render_condition_glyph(cond_key)}</td>"
                    f"<td class='value-cell'>{mean:.3f} &plusmn; {sd:.3f}</td>"
                    f"</tr>"
                )

            col_header = html.escape(condition_order_label())
            metric_tables.append(
                f"<div class='metric-wrap'>"
                f"<div class='metric-caption'>{html.escape(humanize_token(metric_key))} {direction}</div>"
                f"<table class='ranked-table'>"
                f"<thead><tr>"
                f"<th class='col-rank'>Rank</th>"
                f"<th class='col-cond'>{col_header}</th>"
                f"<th class='col-val'>Mean &plusmn; SD</th>"
                f"</tr></thead>"
                f"<tbody>{''.join(rows)}</tbody>"
                f"</table></div>"
            )

        if not metric_tables:
            continue
        dataset_sections.append(
            f"<div class='dataset-section'>"
            f"<div class='dataset-label'>{html.escape(summary.title)}</div>"
            f"<div class='metric-group'>{''.join(metric_tables)}</div>"
            f"</div>"
        )

    return f"<div class='comparison-section'>{''.join(dataset_sections)}</div>"
```

- [ ] **Step 4: Commit**

```bash
git add tools/ablation_report
git commit -m "feat: replace best/worst table with per-metric ranked top-3/worst-3 tables"
```

---

### Task 5: Update matplotlib figure

**Files:**
- Modify: `tools/ablation_report` (`build_comparison_table_figure`)

- [ ] **Step 1: Replace `build_comparison_table_figure`**

Find and replace the entire function (starting at `def build_comparison_table_figure`):

```python
def build_comparison_table_figure(summaries: list[DatasetSummary]) -> plt.Figure:
    rows: list[list[str]] = []
    for summary in summaries:
        for metric_key in comparison_metric_keys(summary):
            record = summary.best_worst.get(metric_key)
            if not record:
                continue
            top_list: list[tuple[str, float, float]] = record["best"]
            worst_list: list[tuple[str, float, float]] = record["worst"]
            total: int = record["total"]

            spec = METRIC_SPEC_BY_KEY[metric_key]
            direction = "↑" if spec.higher_is_better else "↓"
            metric_label = f"{humanize_token(metric_key)} {direction}"

            for i, (cond_key, mean, sd) in enumerate(top_list):
                rows.append([
                    summary.title if i == 0 else "",
                    metric_label if i == 0 else "",
                    str(i + 1),
                    condition_indicator_text(cond_key),
                    f"{mean:.3f} ± {sd:.3f}",
                ])

            rows.append(["", "", "···", "", ""])

            for offset, (cond_key, mean, sd) in enumerate(worst_list):
                rank = total - len(worst_list) + 1 + offset
                rows.append([
                    "",
                    "",
                    str(rank),
                    condition_indicator_text(cond_key),
                    f"{mean:.3f} ± {sd:.3f}",
                ])

    fig_height = max(3.8, 1.7 + 0.38 * max(len(rows), 1))
    fig, ax = plt.subplots(figsize=(13.0, fig_height))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=[
            "Dataset",
            "Metric",
            "Rank",
            f"Condition ({condition_order_label()})",
            "Mean ± SD",
        ],
        cellLoc="left",
        colLoc="left",
        loc="center",
        bbox=[0.0, 0.02, 1.0, 0.92],
        colWidths=[0.12, 0.14, 0.07, 0.22, 0.14],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1.0, 1.3)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#C9C3B9")
        cell.set_linewidth(0.8)
        if row == 0:
            cell.set_facecolor("#F3F3EF")
            cell.set_text_props(weight="bold", color=INK)
        else:
            cell.set_facecolor("white")
            cell.set_text_props(color=INK, weight="normal")
            if col in {0, 1}:
                cell.set_text_props(weight="bold", color=INK)

    ax.set_title(
        "Paired vs Unpaired — Top 3 / Worst 3 Conditions per Metric",
        fontsize=14,
        fontweight="bold",
        pad=10,
    )
    return fig
```

- [ ] **Step 2: Run the full test suite**

```bash
pytest tests/ -v
```

Expected: all tests PASS

- [ ] **Step 3: Commit**

```bash
git add tools/ablation_report
git commit -m "feat: update comparison figure to show ranked top-3/worst-3 per metric"
```
