# Figure 5 Compact Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Redesign `src/paper_figures/fig_uni_tme_decomposition.py` from a 4-panel 14.5×8.3 in figure to a compact 3-panel L-shape 7.2×3.95 in figure (183 mm × 100 mm) matching Nature Communications standards.

**Architecture:** Single file edit to `fig_uni_tme_decomposition.py`. Outer layout changes from stacked (A top, BCD bottom) to L-shape (A left column, B+C right column stacked). Panel A changes from 2×4+legend to 3×2. Panel B gains a shared dot-key strip and hollow dots. Panel C gets a height-matched colorbar. Panel D is removed entirely.

**Tech Stack:** matplotlib, PIL/Pillow, numpy — all already imported.

---

## File Map

- **Modify:** `src/paper_figures/fig_uni_tme_decomposition.py`
  - Remove: `_render_mode_dots`, `_render_panel_d`, `_normalized`
  - Add: `_render_mode_indicator`, `_render_shared_dot_key`
  - Rewrite: `_render_panel_a`, `_render_panel_b`, `_render_panel_c`, `build_uni_tme_decomposition_figure`
  - Minor: `_render_image_cell` gains optional `border_color` param
- **Modify:** `tests/test_fig_uni_tme_decomposition.py`
  - Update size assertions; add aspect-ratio check

---

## Task 1: Remove Panel D, suptitle, update outer layout and figure size

**Files:**
- Modify: `src/paper_figures/fig_uni_tme_decomposition.py:390-421`
- Test: `tests/test_fig_uni_tme_decomposition.py:62-81`

- [ ] **Step 1: Update the test to expect the new compact dimensions**

Replace the size assertions in `test_save_uni_tme_decomposition_figure_renders_fixture`:

```python
    with Image.open(out_png) as image:
        assert image.width > 400
        assert image.height > 200
        assert image.width > image.height  # landscape
```

- [ ] **Step 2: Run the existing test to confirm it still passes before any code changes**

```bash
conda run -n he-multiplex python -m pytest tests/test_fig_uni_tme_decomposition.py -v
```

Expected: PASS (current figure is landscape and > 400 px wide at dpi=80).

- [ ] **Step 3: Delete `_normalized` and `_render_panel_d` (lines 333–388)**

Remove the two functions entirely:

```python
# DELETE lines 333-388: _normalized and _render_panel_d
```

- [ ] **Step 4: Rewrite `build_uni_tme_decomposition_figure` (lines 390–421)**

Replace the entire function body with:

```python
def build_uni_tme_decomposition_figure(
    *,
    generated_root: Path = DEFAULT_GENERATED_ROOT,
    metrics_root: Path = DEFAULT_METRICS_ROOT,
    summary_csv: Path = DEFAULT_SUMMARY_CSV,
    representative_json: Path = DEFAULT_REPRESENTATIVE_JSON,
    orion_root: Path = DEFAULT_ORION_ROOT,
) -> plt.Figure:
    generated_root = Path(generated_root)
    metrics_root = Path(metrics_root)
    summary_csv = Path(summary_csv)
    if not summary_csv.is_file():
        raise FileNotFoundError(f"missing decomposition summary: {summary_csv}")

    summary = load_summary_csv(summary_csv)
    tile_id = _resolve_representative_tile(
        generated_root=generated_root,
        metrics_root=metrics_root,
        representative_json=Path(representative_json),
    )

    fig = plt.figure(figsize=(7.2, 3.95))
    outer = fig.add_gridspec(1, 2, width_ratios=[0.95, 1.05], wspace=0.08)
    _render_panel_a(fig, outer[0, 0], generated_root=generated_root, orion_root=Path(orion_root), tile_id=tile_id)
    right = outer[0, 1].subgridspec(2, 1, height_ratios=[1.15, 1.0], hspace=0.38)
    _render_panel_b(fig, right[0, 0], summary)
    _render_panel_c(fig, right[1, 0], summary)
    fig.subplots_adjust(left=0.02, right=0.96, bottom=0.08, top=0.97)
    return fig
```

- [ ] **Step 5: Update default dpi in `save_uni_tme_decomposition_figure` (line 433)**

```python
    dpi: int = 300,
```

- [ ] **Step 6: Run the test to confirm it passes**

```bash
conda run -n he-multiplex python -m pytest tests/test_fig_uni_tme_decomposition.py -v
```

Expected: PASS (figure renders without Panel D; size > 400 × > 200).

- [ ] **Step 7: Commit**

```bash
git add src/paper_figures/fig_uni_tme_decomposition.py tests/test_fig_uni_tme_decomposition.py
git commit -m "refactor(fig5): remove panel D, L-shape outer layout, compact figsize"
```

---

## Task 2: Rewrite `_render_panel_a` — 3×2 grid with corner dot indicators

**Files:**
- Modify: `src/paper_figures/fig_uni_tme_decomposition.py:101-186`

- [ ] **Step 1: Add `_render_image_cell` optional `border_color` param (line 115)**

Replace the function:

```python
def _render_image_cell(ax: plt.Axes, image: Image.Image, title: str, *, border_color: str = "#333333") -> None:
    ax.imshow(image)
    ax.set_title(title, fontsize=7, pad=2)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color(border_color)
```

- [ ] **Step 2: Add `_render_mode_indicator` helper after `_render_image_cell`**

Insert this new function (after `_render_image_cell`, before `_render_panel_a`):

```python
def _render_mode_indicator(ax: plt.Axes, mode_key: str, *, show_labels: bool) -> None:
    """Draw UNI/TME ●/○ dots at bottom-right corner of an image axes."""
    positions = [(0.80, "UNI", MODE_USE_UNI[mode_key]), (0.91, "TME", MODE_USE_TME[mode_key])]
    for x_ax, label, active in positions:
        ax.scatter(
            [x_ax], [0.06],
            s=16,
            facecolors=INK if active else "white",
            edgecolors=INK,
            linewidths=0.8,
            transform=ax.transAxes,
            clip_on=False,
            zorder=5,
        )
        if show_labels:
            ax.text(x_ax, 0.20, label, transform=ax.transAxes, ha="center", fontsize=5.5, color=INK)
```

- [ ] **Step 3: Rewrite `_render_panel_a` (lines 126–186)**

Replace the entire function:

```python
_GENERATED_GRID: list[tuple[str, bool]] = [
    ("uni_plus_tme", True),
    ("uni_only", False),
    ("tme_only", False),
    ("neither", False),
]
_GENERATED_POSITIONS: list[tuple[int, int]] = [(1, 0), (1, 1), (2, 0), (2, 1)]


def _render_panel_a(
    fig: plt.Figure,
    subgrid,
    *,
    generated_root: Path,
    orion_root: Path,
    tile_id: str,
) -> None:
    outer_ax = fig.add_subplot(subgrid)
    outer_ax.axis("off")
    _panel_label(outer_ax, "A")

    grid = subgrid.subgridspec(3, 2, wspace=0.05, hspace=0.10)
    sample = _load_rgb(generated_root / tile_id / "uni_plus_tme.png")
    size = sample.size

    # Row 0: reference images
    ref_ax = fig.add_subplot(grid[0, 0])
    ref_path = orion_root / "he" / f"{tile_id}.png"
    ref_img = _load_rgb(ref_path, size=size) if ref_path.is_file() else _blank_image(size=size)
    _render_image_cell(ref_ax, ref_img, "Real H&E", border_color="#5a9a5a")

    tme_ax = fig.add_subplot(grid[0, 1])
    _render_image_cell(tme_ax, _load_tme_thumbnail(orion_root, tile_id, size=size), "TME layout", border_color="#b89a70")

    # Rows 1–2: generated modes
    for (mode_key, show_text), (row, col) in zip(_GENERATED_GRID, _GENERATED_POSITIONS, strict=True):
        ax = fig.add_subplot(grid[row, col])
        image = _load_rgb(generated_root / tile_id / f"{mode_key}.png", size=size)
        _render_image_cell(ax, image, MODE_LABELS[mode_key])
        _render_mode_indicator(ax, mode_key, show_labels=show_text)
```

- [ ] **Step 4: Remove the now-unused `MODE_GRID` constant (line 34)**

```python
# DELETE this line:
# MODE_GRID = (
#     ("uni_plus_tme", "uni_only"),
#     ("tme_only", "neither"),
# )
```

- [ ] **Step 5: Run the test**

```bash
conda run -n he-multiplex python -m pytest tests/test_fig_uni_tme_decomposition.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/paper_figures/fig_uni_tme_decomposition.py
git commit -m "refactor(fig5): panel A — 3×2 grid, corner dot indicators, no legend column"
```

---

## Task 3: Rewrite `_render_panel_b` — hollow dots, aligned dot-key row

**Files:**
- Modify: `src/paper_figures/fig_uni_tme_decomposition.py:228-294`

- [ ] **Step 1: Replace `_render_mode_dots` with `_render_dot_key_single`**

Remove `_render_mode_dots` (lines 228–250) and insert:

```python
def _render_dot_key_single(key_ax: plt.Axes, *, show_labels: bool) -> None:
    """One column of the dot-key strip. show_labels=True only for the leftmost column."""
    key_ax.set_xlim(-0.5, len(MODE_KEYS) - 0.5)
    key_ax.set_ylim(-0.5, 1.5)
    for x, mode_key in enumerate(MODE_KEYS):
        key_ax.scatter(
            x, 1, s=20,
            facecolors=INK if MODE_USE_UNI[mode_key] else "white",
            edgecolors=INK, linewidths=0.8,
        )
        key_ax.scatter(
            x, 0, s=20,
            facecolors=INK if MODE_USE_TME[mode_key] else "white",
            edgecolors=INK, linewidths=0.8,
        )
    if show_labels:
        key_ax.text(-0.8, 1, "UNI", ha="right", va="center", fontsize=6.5, color=INK)
        key_ax.text(-0.8, 0, "TME", ha="right", va="center", fontsize=6.5, color=INK)
    key_ax.axis("off")
```

- [ ] **Step 2: Rewrite `_render_panel_b` (lines 253–294)**

Use a 2-row outer grid where both rows share the same column widths and wspace. This gives exact dot alignment between metric subplots and the key strip below.

Replace the entire function:

```python
def _render_panel_b(fig: plt.Figure, subgrid, summary: dict[str, dict]) -> None:
    outer_ax = fig.add_subplot(subgrid)
    outer_ax.axis("off")
    _panel_label(outer_ax, "B")

    outer_grid = subgrid.subgridspec(2, 1, height_ratios=[5.5, 0.85], hspace=0.05)
    metric_grid = outer_grid[0, 0].subgridspec(1, len(DISPLAY_METRICS), wspace=0.38)
    key_grid = outer_grid[1, 0].subgridspec(1, len(DISPLAY_METRICS), wspace=0.38)

    x = np.arange(len(MODE_KEYS), dtype=float)
    for idx, metric_key in enumerate(DISPLAY_METRICS):
        ax = fig.add_subplot(metric_grid[0, idx])
        values, errors = _values_for_metric(summary, metric_key)
        valid_x = [xv for xv, v in zip(x, values, strict=True) if np.isfinite(v)]
        valid_y = [v for v in values if np.isfinite(v)]
        valid_err = [e if e is not None else 0.0 for v, e in zip(values, errors, strict=True) if np.isfinite(v)]
        if valid_y:
            ax.errorbar(
                valid_x,
                valid_y,
                yerr=valid_err,
                color=INK,
                linestyle="none",
                marker="o",
                markerfacecolor="white",
                markeredgecolor=INK,
                markersize=4.5,
                capsize=2.0,
                elinewidth=0.9,
                markeredgewidth=0.9,
            )
        label = METRIC_LABELS.get(metric_key, metric_key)
        row = summary.get("uni_plus_tme", {}).get(metric_key)
        direction = row.direction if row is not None else ""
        ax.set_title(f"{label} ({direction})", fontsize=7, pad=2)
        ax.set_xlim(-0.5, len(MODE_KEYS) - 0.5)
        ax.set_ylim(*_tight_ylim(values, errors))
        ax.set_xticks([])
        ax.grid(True, axis="y", color=SOFT_GRID, linewidth=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="y", labelsize=6.5, colors=INK)
        if idx > 0:
            ax.yaxis.set_visible(False)
            ax.spines["left"].set_visible(False)

        key_ax = fig.add_subplot(key_grid[0, idx])
        _render_dot_key_single(key_ax, show_labels=(idx == 0))
```

- [ ] **Step 3: Run the test**

```bash
conda run -n he-multiplex python -m pytest tests/test_fig_uni_tme_decomposition.py -v
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add src/paper_figures/fig_uni_tme_decomposition.py
git commit -m "refactor(fig5): panel B — hollow dots, shared dot-key strip, no subtitle"
```

---

## Task 4: Fix `_render_panel_c` — height-matched colorbar, remove title

**Files:**
- Modify: `src/paper_figures/fig_uni_tme_decomposition.py:297-330`

- [ ] **Step 1: Add `make_axes_locatable` import at top of file**

In the imports section, add:

```python
from mpl_toolkits.axes_grid1 import make_axes_locatable
```

- [ ] **Step 2: Rewrite `_render_panel_c` (lines 297–330)**

Replace the entire function:

```python
def _render_panel_c(fig: plt.Figure, subgrid, summary: dict[str, dict]) -> None:
    ax = fig.add_subplot(subgrid)
    _panel_label(ax, "C")
    effects = effect_decomposition(summary)
    rows = list(effects)
    cols = list(DISPLAY_METRICS)
    matrix = np.full((len(rows), len(cols)), np.nan, dtype=float)
    for row_idx, row_name in enumerate(rows):
        for col_idx, metric_key in enumerate(cols):
            value = effects[row_name].get(metric_key)
            if value is not None:
                matrix[row_idx, col_idx] = float(value)

    finite = matrix[np.isfinite(matrix)]
    if finite.size == 0:
        ax.axis("off")
        ax.text(0.5, 0.5, "Effect metrics missing", ha="center", va="center", transform=ax.transAxes)
        return

    vmax = float(np.max(np.abs(finite))) or 1.0
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([METRIC_LABELS.get(m, m) for m in cols], rotation=35, ha="right", fontsize=7)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=7)
    for row_idx in range(len(rows)):
        for col_idx in range(len(cols)):
            value = matrix[row_idx, col_idx]
            if np.isfinite(value):
                ax.text(col_idx, row_idx, f"{value:.2g}", ha="center", va="center", fontsize=6.5, color=INK)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.06)
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=6.5)
    cbar.set_label("Δ (higher-is-better)", fontsize=7)
```

- [ ] **Step 3: Run the test**

```bash
conda run -n he-multiplex python -m pytest tests/test_fig_uni_tme_decomposition.py -v
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add src/paper_figures/fig_uni_tme_decomposition.py
git commit -m "refactor(fig5): panel C — height-matched colorbar, remove title"
```

---

## Task 5: Generate final figure and verify

**Files:**
- Read: `figures/pngs/08_uni_tme_decomposition.png`

- [ ] **Step 1: Generate the updated figure**

```bash
conda run -n pixcell python -c "
from src.paper_figures.fig_uni_tme_decomposition import save_uni_tme_decomposition_figure
out = save_uni_tme_decomposition_figure()
print('Saved:', out)
from PIL import Image
with Image.open(out) as im:
    print('Size (px):', im.size)
    print('Expected ~2160×1185 at 300 dpi')
"
```

Expected output: `Size (px): (2160, 1185)` ± a few pixels (bbox_inches="tight" may trim slightly).

- [ ] **Step 2: Run full test suite**

```bash
conda run -n he-multiplex python -m pytest tests/test_fig_uni_tme_decomposition.py tests/test_a2_decomposition_metrics.py -v
```

Expected: all PASS.

- [ ] **Step 3: Commit**

```bash
git add figures/pngs/08_uni_tme_decomposition.png
git commit -m "fig: regenerate figure 5 compact redesign (183 mm × 100 mm)"
```
