# Tied Best/Worst Conditions in Ablation Report — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the single best/worst condition per metric with a tolerance-band approach that shows all conditions within 1 SD of the best (and worst), making near-ties visible.

**Architecture:** Update `summarize_best_worst` in the summary layer to return lists of `(condition, mean, sd)` tuples; update the HTML rendering functions and matplotlib figure to display multi-row metric groups; update `best_worst_notes` and the existing test to match the new API.

**Tech Stack:** Python 3.12, pytest, matplotlib, HTML/CSS in the report renderer.

---

### Task 1: Update `summarize_best_worst` and `best_worst_notes`

**Files:**
- Modify: `tools/summarize_ablation_report.py`

The function gains an optional `condition_stats` parameter (per-condition SDs). When `None` (markdown report path), SD defaults to 0 and each condition is its own group — preserving old behavior. The return type changes from single conditions to ranked lists.

- [ ] **Step 1: Replace `summarize_best_worst` in `tools/summarize_ablation_report.py`**

Find and replace the entire function (lines 124–146). New implementation:

```python
def _get_cond_sd(
    cond_key: str,
    metric_key: str,
    condition_stats: dict[str, dict[str, tuple[float, float]]] | None,
) -> float:
    if condition_stats is None:
        return 0.0
    return condition_stats.get(cond_key, {}).get(metric_key, (0.0, 0.0))[1]


def summarize_best_worst(
    condition_means: dict[str, dict[str, float]],
    metric_keys: list[str],
    condition_stats: dict[str, dict[str, tuple[float, float]]] | None = None,
) -> dict[str, dict[str, list[tuple[str, float, float]]]]:
    summary: dict[str, dict[str, list[tuple[str, float, float]]]] = {}
    for metric_key in metric_keys:
        items = [
            (cond_key, metrics[metric_key])
            for cond_key, metrics in condition_means.items()
            if metric_key in metrics
        ]
        if not items:
            continue
        spec = METRIC_SPEC_BY_KEY[metric_key]

        if spec.higher_is_better:
            best_mean = max(v for _, v in items)
            worst_mean = min(v for _, v in items)
        else:
            best_mean = min(v for _, v in items)
            worst_mean = max(v for _, v in items)

        best_anchor = next(ck for ck, v in items if v == best_mean)
        worst_anchor = next(ck for ck, v in items if v == worst_mean)
        best_sd = _get_cond_sd(best_anchor, metric_key, condition_stats)
        worst_sd = _get_cond_sd(worst_anchor, metric_key, condition_stats)

        if spec.higher_is_better:
            tied_best = sorted(
                [(ck, v, _get_cond_sd(ck, metric_key, condition_stats)) for ck, v in items if v >= best_mean - best_sd],
                key=lambda t: t[1],
                reverse=True,
            )
            tied_worst = sorted(
                [(ck, v, _get_cond_sd(ck, metric_key, condition_stats)) for ck, v in items if v <= worst_mean + worst_sd],
                key=lambda t: t[1],
            )
        else:
            tied_best = sorted(
                [(ck, v, _get_cond_sd(ck, metric_key, condition_stats)) for ck, v in items if v <= best_mean + best_sd],
                key=lambda t: t[1],
            )
            tied_worst = sorted(
                [(ck, v, _get_cond_sd(ck, metric_key, condition_stats)) for ck, v in items if v >= worst_mean - worst_sd],
                key=lambda t: t[1],
                reverse=True,
            )

        summary[metric_key] = {"best": tied_best, "worst": tied_worst}
    return summary
```

- [ ] **Step 2: Update `best_worst_notes` in `tools/summarize_ablation_report.py`**

Find and replace the entire `best_worst_notes` function (lines 316–338):

```python
def best_worst_notes(best_worst: dict[str, dict[str, list[tuple[str, float, float]]]]) -> list[str]:
    full_condition = condition_metric_key(FOUR_GROUP_ORDER)
    full_best = [
        metric_key
        for metric_key, summary in best_worst.items()
        if any(cond == full_condition for cond, _, _ in summary["best"])
    ]
    full_worst = [
        metric_key
        for metric_key, summary in best_worst.items()
        if any(cond == full_condition for cond, _, _ in summary["worst"])
    ]

    worst_counts: dict[str, int] = {}
    for summary in best_worst.values():
        for worst_condition, _, _ in summary["worst"]:
            worst_counts[worst_condition] = worst_counts.get(worst_condition, 0) + 1

    notes: list[str] = []
    if full_best:
        metrics_text = ", ".join(f"`{metric_key}`" for metric_key in full_best)
        notes.append(f"- The full `4g` model is strongest for {metrics_text}.")
    if full_worst:
        metrics_text = ", ".join(f"`{metric_key}`" for metric_key in full_worst)
        notes.append(f"- The full `4g` model is weakest for {metrics_text}.")

    if worst_counts:
        repeated_worst = max(worst_counts.items(), key=lambda item: (item[1], item[0]))
        if repeated_worst[1] > 1:
            notes.append(f"- {_format_condition_key(repeated_worst[0])} is weak across multiple metrics.")
    return notes
```

- [ ] **Step 3: Commit**

```bash
git add tools/summarize_ablation_report.py
git commit -m "refactor: update summarize_best_worst to return tied conditions list"
```

---

### Task 2: Update tests for `summarize_best_worst`

**Files:**
- Modify: `tests/test_summarize_ablation_report.py`

Two changes: update the existing test that references the old dict keys (`best_condition`, `best_value`), and add two new tests for the tied and clear-winner cases.

- [ ] **Step 1: Update the existing test assertions in `test_summary_helpers_capture_expected_metric_directions`**

Find these four lines in the test (around line 100):

```python
    assert best_worst["aji"]["best_condition"] == full_key
    assert best_worst["pq"]["best_condition"] == full_key
    assert "microenv" in str(best_worst["fud"]["best_condition"])
    assert "cell_state" not in str(best_worst["fud"]["best_condition"])
```

Replace them with:

```python
    assert best_worst["aji"]["best"][0][0] == full_key
    assert best_worst["pq"]["best"][0][0] == full_key
    assert "microenv" in best_worst["fud"]["best"][0][0]
    assert "cell_state" not in best_worst["fud"]["best"][0][0]
```

- [ ] **Step 2: Run the updated test to verify it passes**

```bash
cd /home/pohaoc2/UW/bagherilab/PixCell
pytest tests/test_summarize_ablation_report.py::test_summary_helpers_capture_expected_metric_directions -v
```

Expected: PASS

- [ ] **Step 3: Add `test_summarize_best_worst_tied` at the bottom of `tests/test_summarize_ablation_report.py`**

```python
def test_summarize_best_worst_tied() -> None:
    # IoU (higher_is_better): three conditions cluster within 1 SD of best; one clear worst.
    # best_mean=0.654, best_sd=0.010 → threshold=0.644; all top 3 are >= 0.644.
    # worst_mean=0.500, worst_sd=0.020 → threshold=0.520; only "ct" is <= 0.520.
    condition_means = {
        "cell_types+cell_state+microenv": {"iou": 0.654},
        "cell_types+cell_state+vasculature+microenv": {"iou": 0.650},
        "cell_types+cell_state+vasculature": {"iou": 0.648},
        "cell_types": {"iou": 0.500},
    }
    condition_stats = {
        "cell_types+cell_state+microenv": {"iou": (0.654, 0.010)},
        "cell_types+cell_state+vasculature+microenv": {"iou": (0.650, 0.008)},
        "cell_types+cell_state+vasculature": {"iou": (0.648, 0.009)},
        "cell_types": {"iou": (0.500, 0.020)},
    }
    result = summarize_best_worst(condition_means, ["iou"], condition_stats)

    best = result["iou"]["best"]
    worst = result["iou"]["worst"]

    best_conds = [t[0] for t in best]
    assert "cell_types+cell_state+microenv" in best_conds
    assert "cell_types+cell_state+vasculature+microenv" in best_conds
    assert "cell_types+cell_state+vasculature" in best_conds
    assert "cell_types" not in best_conds
    # Sorted best → worst (descending for higher_is_better)
    assert best[0][0] == "cell_types+cell_state+microenv"
    assert best[0][1] == pytest.approx(0.654)
    assert len(worst) == 1
    assert worst[0][0] == "cell_types"
```

- [ ] **Step 4: Add `test_summarize_best_worst_clear_winner` immediately after**

```python
def test_summarize_best_worst_clear_winner() -> None:
    # IoU: clear gap — 2nd best (0.600) is > 1 SD below best (0.700, sd=0.010).
    # threshold = 0.700 - 0.010 = 0.690; 0.600 < 0.690 → not tied.
    condition_means = {
        "cell_types+cell_state+microenv": {"iou": 0.700},
        "cell_types+cell_state": {"iou": 0.600},
        "cell_types": {"iou": 0.500},
    }
    condition_stats = {
        "cell_types+cell_state+microenv": {"iou": (0.700, 0.010)},
        "cell_types+cell_state": {"iou": (0.600, 0.008)},
        "cell_types": {"iou": (0.500, 0.020)},
    }
    result = summarize_best_worst(condition_means, ["iou"], condition_stats)

    best = result["iou"]["best"]
    worst = result["iou"]["worst"]

    assert len(best) == 1
    assert best[0][0] == "cell_types+cell_state+microenv"
    assert len(worst) == 1
    assert worst[0][0] == "cell_types"
```

- [ ] **Step 5: Run all new tests**

```bash
pytest tests/test_summarize_ablation_report.py -v
```

Expected: all 5 tests PASS

- [ ] **Step 6: Commit**

```bash
git add tests/test_summarize_ablation_report.py
git commit -m "test: update and add tests for tied best/worst conditions"
```

---

### Task 3: Update `DatasetSummary` type annotation and call site

**Files:**
- Modify: `tools/render_ablation_html_report.py:121-143` (DatasetSummary), `tools/render_ablation_html_report.py:830` (call site)

- [ ] **Step 1: Update `DatasetSummary.best_worst` type annotation**

In `tools/render_ablation_html_report.py`, find line 133:

```python
    best_worst: dict[str, dict[str, str | float]]
```

Replace with:

```python
    best_worst: dict[str, dict[str, list[tuple[str, float, float]]]]
```

- [ ] **Step 2: Pass `condition_stats` to `summarize_best_worst` at the call site**

Find line 830:

```python
    best_worst = summarize_best_worst(condition_means, metric_keys)
```

Replace with:

```python
    best_worst = summarize_best_worst(condition_means, metric_keys, condition_stats)
```

(`condition_stats` is already computed at line 824, before this call.)

- [ ] **Step 3: Commit**

```bash
git add tools/render_ablation_html_report.py
git commit -m "feat: pass condition_stats to summarize_best_worst for SD-based tie detection"
```

---

### Task 4: Update HTML rendering functions

**Files:**
- Modify: `tools/render_ablation_html_report.py`

Two functions: `render_best_worst_table` (defined but currently unused — update for correctness) and `render_comparison_table` (used in the main HTML report).

Also add `zip_longest` to the top-level import.

- [ ] **Step 1: Add `zip_longest` to the itertools import**

Find line 13:

```python
from itertools import combinations
```

Replace with:

```python
from itertools import combinations, zip_longest
```

- [ ] **Step 2: Replace `render_best_worst_table`**

Find and replace the entire function (starting at `def render_best_worst_table`):

```python
def render_best_worst_table(summary: DatasetSummary) -> str:
    rows: list[str] = []
    for metric_key in comparison_metric_keys(summary):
        record = summary.best_worst.get(metric_key)
        if not record:
            continue
        best_list = record["best"]
        worst_list = record["worst"]
        n_best = len(best_list)
        tied_label = (
            f"<br><span style='font-style:italic;font-size:0.85em;'>({n_best} tied)</span>"
            if n_best > 1 else ""
        )
        span = max(len(best_list), len(worst_list))
        for i, (b, w) in enumerate(zip_longest(best_list, worst_list)):
            b_glyph = f"<td class='condition-cell'>{render_condition_glyph(b[0])}</td>" if b else "<td></td>"
            b_val = f"<td>{b[1]:.3f} ± {b[2]:.3f}</td>" if b else "<td></td>"
            w_glyph = f"<td class='condition-cell'>{render_condition_glyph(w[0])}</td>" if w else "<td></td>"
            w_val = f"<td>{w[1]:.3f} ± {w[2]:.3f}</td>" if w else "<td></td>"
            if i == 0:
                rows.append(
                    f"<tr>"
                    f"<td rowspan='{span}'>{metric_name_cell(metric_key)}{tied_label}</td>"
                    f"{b_glyph}{b_val}{w_glyph}{w_val}</tr>"
                )
            else:
                rows.append(f"<tr>{b_glyph}{b_val}{w_glyph}{w_val}</tr>")
    return (
        "<table class='metric-table'>"
        "<thead><tr>"
        "<th>Metric</th>"
        f"<th>Best conditions <span class='condition-order'>{html.escape(condition_order_label())}</span></th>"
        "<th>Mean ± SD</th>"
        f"<th>Worst conditions <span class='condition-order'>{html.escape(condition_order_label())}</span></th>"
        "<th>Mean ± SD</th>"
        "</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )
```

- [ ] **Step 3: Replace `render_comparison_table`**

Find and replace the entire function (starting at `def render_comparison_table`):

```python
def render_comparison_table(summaries: list[DatasetSummary]) -> str:
    all_rows: list[str] = []
    for summary in summaries:
        dataset_rows: list[str] = []
        for metric_key in comparison_metric_keys(summary):
            record = summary.best_worst.get(metric_key)
            if not record:
                continue
            best_list = record["best"]
            worst_list = record["worst"]
            n_best = len(best_list)
            tied_label = (
                f"<br><span style='font-style:italic;font-size:0.85em;'>({n_best} tied)</span>"
                if n_best > 1 else ""
            )
            span = max(len(best_list), len(worst_list))
            for i, (b, w) in enumerate(zip_longest(best_list, worst_list)):
                b_glyph = f"<td class='condition-cell'>{render_condition_glyph(b[0])}</td>" if b else "<td></td>"
                b_val = f"<td>{b[1]:.3f} ± {b[2]:.3f}</td>" if b else "<td></td>"
                w_glyph = f"<td class='condition-cell'>{render_condition_glyph(w[0])}</td>" if w else "<td></td>"
                w_val = f"<td>{w[1]:.3f} ± {w[2]:.3f}</td>" if w else "<td></td>"
                if i == 0:
                    dataset_rows.append(
                        f"<tr>"
                        f"<td class='metric-text' rowspan='{span}'>{metric_plain_text(metric_key)}{tied_label}</td>"
                        f"{b_glyph}{b_val}{w_glyph}{w_val}</tr>"
                    )
                else:
                    dataset_rows.append(f"<tr>{b_glyph}{b_val}{w_glyph}{w_val}</tr>")
        if not dataset_rows:
            continue
        dataset_rows[0] = dataset_rows[0].replace(
            "<tr>",
            f"<tr><td class='dataset-cell' rowspan='{len(dataset_rows)}'>{html.escape(summary.title)}</td>",
            1,
        )
        all_rows.extend(dataset_rows)
    return (
        "<table class='metric-table comparison-table'>"
        "<thead><tr>"
        "<th>Dataset</th>"
        "<th>Metric</th>"
        f"<th>Best conditions <span class='condition-order'>{html.escape(condition_order_label())}</span></th>"
        "<th>Mean ± SD</th>"
        f"<th>Worst conditions <span class='condition-order'>{html.escape(condition_order_label())}</span></th>"
        "<th>Mean ± SD</th>"
        "</tr></thead>"
        f"<tbody>{''.join(all_rows)}</tbody></table>"
    )
```

- [ ] **Step 4: Commit**

```bash
git add tools/render_ablation_html_report.py
git commit -m "feat: render tied best/worst conditions as multi-row groups in HTML table"
```

---

### Task 5: Update matplotlib figure

**Files:**
- Modify: `tools/render_ablation_html_report.py` (`build_comparison_table_figure`)

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
            best_list = record["best"]
            worst_list = record["worst"]
            n_best = len(best_list)
            metric_label = humanize_token(metric_key)
            if n_best > 1:
                metric_label += f"\n({n_best} tied)"
            for i, (b, w) in enumerate(zip_longest(best_list, worst_list)):
                rows.append(
                    [
                        summary.title if i == 0 else "",
                        metric_label if i == 0 else "",
                        condition_indicator_text(b[0]) if b is not None else "",
                        f"{b[1]:.3f} ± {b[2]:.3f}" if b is not None else "",
                        condition_indicator_text(w[0]) if w is not None else "",
                        f"{w[1]:.3f} ± {w[2]:.3f}" if w is not None else "",
                    ]
                )

    fig_height = max(3.8, 1.7 + 0.42 * max(len(rows), 1))
    fig, ax = plt.subplots(figsize=(15.5, fig_height))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=[
            "Dataset",
            "Metric",
            f"Best ({condition_order_label()})",
            "Mean ± SD",
            f"Worst ({condition_order_label()})",
            "Mean ± SD",
        ],
        cellLoc="left",
        colLoc="left",
        loc="center",
        bbox=[0.0, 0.02, 1.0, 0.92],
        colWidths=[0.12, 0.14, 0.18, 0.12, 0.18, 0.12],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.0)
    table.scale(1.0, 1.35)
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

    ax.set_title("Paired vs Unpaired Best/Worst Conditions", fontsize=15, fontweight="bold", pad=12)
    return fig
```

- [ ] **Step 2: Run the full test suite to catch any remaining issues**

```bash
pytest tests/ -v
```

Expected: all tests PASS

- [ ] **Step 3: Commit**

```bash
git add tools/render_ablation_html_report.py
git commit -m "feat: update paired vs unpaired figure to show tied conditions with mean ± SD"
```
