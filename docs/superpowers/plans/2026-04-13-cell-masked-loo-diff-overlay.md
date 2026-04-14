# Cell-Masked LOO Diff Overlay — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the leave-one-out diff heatmap with a cell-masked overlay that zeroes diffusion background noise and shows channel-drop effects only inside cell boundaries, on a dimmed greyscale H&E background.

**Architecture:** Two new private helpers (`_normalize_cell_masked_diff`, `_render_cell_masked_overlay`) are added to `tools/vis/leave_one_out_diff.py`. `render_loo_diff_figure` is updated to compute raw diffs inline, call the overlay helper per column, and update the colorbar. When no cell mask is present, the existing per-map normalization is used as a fallback.

**Tech Stack:** numpy, matplotlib (ScalarMappable, LinearSegmentedColormap), PIL, pytest

---

## File Map

| File | Change |
|------|--------|
| `tools/vis/leave_one_out_diff.py` | Add 2 helpers; update `render_loo_diff_figure`; add `import matplotlib.cm` |
| `tests/test_leave_one_out_diff.py` | Add tests for new helpers and updated figure renderer |

---

## Task 1: `_normalize_cell_masked_diff` helper

**Files:**
- Modify: `tools/vis/leave_one_out_diff.py` (after `_compute_relative_diff_maps`, ~line 175)
- Modify: `tests/test_leave_one_out_diff.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_leave_one_out_diff.py`:

```python
# ── _normalize_cell_masked_diff ───────────────────────────────────────────────

def test_normalize_cell_masked_diff_zeroes_background_and_normalizes_cells() -> None:
    from tools.vis.leave_one_out_diff import _normalize_cell_masked_diff

    diff = np.full((4, 4), 100.0, dtype=np.float32)
    cell_mask = np.zeros((4, 4), dtype=np.float32)
    cell_mask[1:3, 1:3] = 1.0  # 4 cell pixels in the centre

    result = _normalize_cell_masked_diff(diff, cell_mask)

    assert result.dtype == np.float32
    assert result.shape == (4, 4)
    # Background pixels are zeroed out
    assert float(result[0, 0]) == 0.0
    assert float(result[3, 3]) == 0.0
    # Cell pixels: uniform diff=100, p99=100 → normalized to 1.0
    assert float(result[1, 1]) == pytest.approx(1.0, abs=1e-5)
    assert result.min() >= 0.0
    assert result.max() <= 1.0 + 1e-6


def test_normalize_cell_masked_diff_empty_mask_returns_zeros() -> None:
    from tools.vis.leave_one_out_diff import _normalize_cell_masked_diff

    diff = np.full((4, 4), 50.0, dtype=np.float32)
    cell_mask = np.zeros((4, 4), dtype=np.float32)

    result = _normalize_cell_masked_diff(diff, cell_mask)

    assert float(result.max()) == 0.0


def test_normalize_cell_masked_diff_zero_diff_returns_zeros() -> None:
    from tools.vis.leave_one_out_diff import _normalize_cell_masked_diff

    diff = np.zeros((4, 4), dtype=np.float32)
    cell_mask = np.ones((4, 4), dtype=np.float32)

    result = _normalize_cell_masked_diff(diff, cell_mask)

    assert float(result.max()) == 0.0
```

Also add `import pytest` to the test file imports if not already present.

- [ ] **Step 2: Run tests — confirm they fail**

```bash
cd /home/ec2-user/PixCell
python -m pytest tests/test_leave_one_out_diff.py::test_normalize_cell_masked_diff_zeroes_background_and_normalizes_cells tests/test_leave_one_out_diff.py::test_normalize_cell_masked_diff_empty_mask_returns_zeros tests/test_leave_one_out_diff.py::test_normalize_cell_masked_diff_zero_diff_returns_zeros -v
```

Expected: FAIL with `ImportError: cannot import name '_normalize_cell_masked_diff'`

- [ ] **Step 3: Implement `_normalize_cell_masked_diff`**

Add after `_compute_relative_diff_maps` (after its closing `return` statement, before `_load_cell_mask_array`):

```python
def _normalize_cell_masked_diff(
    diff: np.ndarray,
    cell_mask: np.ndarray,
) -> np.ndarray:
    """Normalize diff by 99th percentile of cell-region pixels; zero out background.

    Args:
        diff: H×W float32 absolute pixel diff (values in [0, 255]).
        cell_mask: H×W float32 in [0, 1]; pixels > 0.5 are treated as cells.

    Returns:
        H×W float32 in [0, 1]. Non-cell pixels are 0. Returns all-zeros when
        there are no cell pixels or when the 99th-percentile diff is zero.
    """
    cell_pixels = diff[cell_mask > 0.5]
    if len(cell_pixels) == 0 or float(cell_pixels.max()) <= 0.0:
        return np.zeros_like(diff, dtype=np.float32)
    p99 = float(np.percentile(cell_pixels, 99))
    if p99 <= 0.0:
        return np.zeros_like(diff, dtype=np.float32)
    masked = diff * (cell_mask > 0.5).astype(np.float32)
    return np.clip(masked / p99, 0.0, 1.0).astype(np.float32)
```

- [ ] **Step 4: Run tests — confirm they pass**

```bash
python -m pytest tests/test_leave_one_out_diff.py::test_normalize_cell_masked_diff_zeroes_background_and_normalizes_cells tests/test_leave_one_out_diff.py::test_normalize_cell_masked_diff_empty_mask_returns_zeros tests/test_leave_one_out_diff.py::test_normalize_cell_masked_diff_zero_diff_returns_zeros -v
```

Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
git add tools/vis/leave_one_out_diff.py tests/test_leave_one_out_diff.py
git commit -m "feat: add _normalize_cell_masked_diff helper for cell-region normalization"
```

---

## Task 2: `_render_cell_masked_overlay` helper

**Files:**
- Modify: `tools/vis/leave_one_out_diff.py` (add `import matplotlib.cm` at top; add helper after `_normalize_cell_masked_diff`)
- Modify: `tests/test_leave_one_out_diff.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_leave_one_out_diff.py`:

```python
# ── _render_cell_masked_overlay ───────────────────────────────────────────────

def test_render_cell_masked_overlay_returns_scalar_mappable() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as mcm
    from tools.vis.leave_one_out_diff import _render_cell_masked_overlay

    hot_cmap = mcolors.LinearSegmentedColormap.from_list(
        "hot4", ["#000000", "#ff4400", "#ffff00", "#ffffff"]
    )
    fig, ax = plt.subplots()
    raw_diff = np.full((4, 4), 80.0, dtype=np.float32)
    cell_mask = np.zeros((4, 4), dtype=np.float32)
    cell_mask[1:3, 1:3] = 1.0
    baseline_he = np.full((4, 4, 3), 128, dtype=np.uint8)

    result = _render_cell_masked_overlay(ax, raw_diff, cell_mask, baseline_he, hot_cmap)

    assert isinstance(result, mcm.ScalarMappable)
    plt.close(fig)


def test_render_cell_masked_overlay_no_crash_empty_mask() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from tools.vis.leave_one_out_diff import _render_cell_masked_overlay

    hot_cmap = mcolors.LinearSegmentedColormap.from_list(
        "hot4", ["#000000", "#ff4400", "#ffff00", "#ffffff"]
    )
    fig, ax = plt.subplots()
    raw_diff = np.full((4, 4), 30.0, dtype=np.float32)
    cell_mask = np.zeros((4, 4), dtype=np.float32)  # no cell pixels
    baseline_he = np.full((4, 4, 3), 200, dtype=np.uint8)

    _render_cell_masked_overlay(ax, raw_diff, cell_mask, baseline_he, hot_cmap)  # must not raise

    plt.close(fig)


def test_render_cell_masked_overlay_no_crash_zero_diff() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from tools.vis.leave_one_out_diff import _render_cell_masked_overlay

    hot_cmap = mcolors.LinearSegmentedColormap.from_list(
        "hot4", ["#000000", "#ff4400", "#ffff00", "#ffffff"]
    )
    fig, ax = plt.subplots()
    raw_diff = np.zeros((4, 4), dtype=np.float32)  # self-diff column (all-four baseline)
    cell_mask = np.ones((4, 4), dtype=np.float32)
    baseline_he = np.full((4, 4, 3), 128, dtype=np.uint8)

    _render_cell_masked_overlay(ax, raw_diff, cell_mask, baseline_he, hot_cmap)  # must not raise

    plt.close(fig)
```

Also add `import matplotlib.colors as mcolors` to the test file imports (needed for constructing the hot cmap in tests).

- [ ] **Step 2: Run tests — confirm they fail**

```bash
python -m pytest tests/test_leave_one_out_diff.py::test_render_cell_masked_overlay_returns_scalar_mappable tests/test_leave_one_out_diff.py::test_render_cell_masked_overlay_no_crash_empty_mask tests/test_leave_one_out_diff.py::test_render_cell_masked_overlay_no_crash_zero_diff -v
```

Expected: FAIL with `ImportError: cannot import name '_render_cell_masked_overlay'`

- [ ] **Step 3: Add `import matplotlib.cm` to module imports**

In `tools/vis/leave_one_out_diff.py`, after `import matplotlib` and `matplotlib.use("Agg")`, add:

```python
import matplotlib.cm
```

So the import block looks like:

```python
import matplotlib

matplotlib.use("Agg")
import matplotlib.cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
```

- [ ] **Step 4: Implement `_render_cell_masked_overlay`**

Add immediately after `_normalize_cell_masked_diff`:

```python
def _render_cell_masked_overlay(
    ax,
    raw_diff: np.ndarray,
    cell_mask: np.ndarray,
    baseline_he: np.ndarray,
    cmap,
    *,
    bg_brightness: float = 0.5,
) -> matplotlib.cm.ScalarMappable:
    """Render a cell-masked diff overlay onto `ax`.

    Cell-region pixels: hot-colormap colour keyed to the per-condition
    99th-percentile-normalised diff.  Background pixels: dimmed greyscale
    of `baseline_he`.

    Args:
        ax: matplotlib Axes to draw on.
        raw_diff: H×W float32 absolute pixel diff in [0, 255].
        cell_mask: H×W float32 in [0, 1]; pixels > 0.5 are cells.
        baseline_he: H×W×3 uint8 baseline H&E image.
        cmap: Matplotlib colormap applied to the normalised diff.
        bg_brightness: Multiplier for the greyscale background (default 0.5).

    Returns:
        ScalarMappable suitable for passing to fig.colorbar().
    """
    diff_norm = _normalize_cell_masked_diff(raw_diff, cell_mask)

    bg = baseline_he.mean(axis=2).astype(np.float32) / 255.0 * bg_brightness

    heatmap_rgba = cmap(diff_norm)  # H×W×4

    alpha = (cell_mask > 0.5).astype(np.float32)
    composite = np.stack(
        [alpha * heatmap_rgba[:, :, c] + (1.0 - alpha) * bg for c in range(3)],
        axis=2,
    )
    composite = np.clip(composite, 0.0, 1.0).astype(np.float32)

    ax.imshow(composite, vmin=0.0, vmax=1.0)

    sm = matplotlib.cm.ScalarMappable(
        cmap=cmap,
        norm=mcolors.Normalize(vmin=0.0, vmax=1.0),
    )
    sm.set_array([])
    return sm
```

- [ ] **Step 5: Run tests — confirm they pass**

```bash
python -m pytest tests/test_leave_one_out_diff.py::test_render_cell_masked_overlay_returns_scalar_mappable tests/test_leave_one_out_diff.py::test_render_cell_masked_overlay_no_crash_empty_mask tests/test_leave_one_out_diff.py::test_render_cell_masked_overlay_no_crash_zero_diff -v
```

Expected: 3 PASSED

- [ ] **Step 6: Commit**

```bash
git add tools/vis/leave_one_out_diff.py tests/test_leave_one_out_diff.py
git commit -m "feat: add _render_cell_masked_overlay helper for cell-masked diff compositing"
```

---

## Task 3: Update `render_loo_diff_figure`

**Files:**
- Modify: `tools/vis/leave_one_out_diff.py` (update `render_loo_diff_figure`)
- Modify: `tests/test_leave_one_out_diff.py` (add smoke tests for both paths)

- [ ] **Step 1: Write the failing test for fallback-path colorbar label**

Append to `tests/test_leave_one_out_diff.py`:

```python
# ── render_loo_diff_figure overlay integration ────────────────────────────────

def test_render_loo_figure_overlay_path_produces_png(tmp_path) -> None:
    """With a cell mask present, render_loo_diff_figure must produce a PNG."""
    from tools.vis.leave_one_out_diff import compute_loo_diffs, render_loo_diff_figure

    cache = _make_cache(tmp_path)
    _write_cell_mask(cache / "cell_mask.png")
    manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
    manifest["cell_mask_path"] = "cell_mask.png"
    (cache / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    fig_path = tmp_path / "overlay.png"
    render_loo_diff_figure(compute_loo_diffs(cache), cache, out_path=fig_path)

    assert fig_path.is_file()
    assert fig_path.stat().st_size > 0


def test_render_loo_figure_no_mask_fallback_produces_png(tmp_path) -> None:
    """Without a cell mask, render_loo_diff_figure falls back to per-map normalisation."""
    from tools.vis.leave_one_out_diff import compute_loo_diffs, render_loo_diff_figure

    cache = _make_cache(tmp_path)
    fig_path = tmp_path / "fallback.png"
    render_loo_diff_figure(compute_loo_diffs(cache), cache, out_path=fig_path)

    assert fig_path.is_file()
    assert fig_path.stat().st_size > 0
```

- [ ] **Step 2: Run these tests — confirm they currently pass (pre-change smoke check)**

```bash
python -m pytest tests/test_leave_one_out_diff.py::test_render_loo_figure_overlay_path_produces_png tests/test_leave_one_out_diff.py::test_render_loo_figure_no_mask_fallback_produces_png -v
```

Expected: Both PASS (they test file existence, not the overlay logic — we verify they still pass after the code change in Step 5).

- [ ] **Step 3: Replace `display_diffs` with raw diff computation in `render_loo_diff_figure`**

In `tools/vis/leave_one_out_diff.py`, inside `render_loo_diff_figure`, locate the block (around lines 221–226):

```python
    display_labels = [_display_title(None)] + [_display_title(group) for group in group_names]
    display_images = [img_all]
    for group in group_names:
        entry = find_loo_entry(sections, group)
        display_images.append(_load_rgb_float32(cache_dir / entry["image_path"]).astype(np.uint8))
    display_diffs = _compute_relative_diff_maps(display_images, img_all)
```

Replace with:

```python
    display_labels = [_display_title(None)] + [_display_title(group) for group in group_names]
    display_images = [img_all]
    for group in group_names:
        entry = find_loo_entry(sections, group)
        display_images.append(_load_rgb_float32(cache_dir / entry["image_path"]).astype(np.uint8))

    baseline_float = img_all.astype(np.float32)
    raw_diffs = [
        np.abs(img.astype(np.float32) - baseline_float).mean(axis=2).astype(np.float32)
        for img in display_images
    ]
    fallback_diffs = (
        None if cell_mask is not None
        else _compute_relative_diff_maps(display_images, img_all, per_map=True)
    )
```

- [ ] **Step 4: Update the per-column diff rendering loop**

Locate the loop (around lines 261–278):

```python
    for index, (label, image, diff_map) in enumerate(zip(display_labels, display_images, display_diffs, strict=True)):
        ...
        diff_ax = fig.add_axes([x0, bottom_row_y, panel_width, row_height])
        last_im = diff_ax.imshow(diff_map, cmap=hot_cmap, vmin=0.0, vmax=1.0)
        diff_ax.set_xticks([])
        diff_ax.set_yticks([])
        if index == 0:
            diff_ax.set_ylabel("Pixel Diff", fontsize=10, rotation=90, labelpad=2)
```

Replace the entire loop with:

```python
    for index, (label, image, raw_diff) in enumerate(zip(display_labels, display_images, raw_diffs, strict=True)):
        x0 = x_right_start + index * (panel_width + col_gap)

        image_ax = fig.add_axes([x0, top_row_y, panel_width, row_height])
        image_ax.imshow(image)
        _maybe_contour_cell_mask(image_ax, cell_mask, image.shape[:2])
        image_ax.set_title(label, fontsize=9)
        image_ax.set_xticks([])
        image_ax.set_yticks([])
        if index == 0:
            image_ax.set_ylabel("Generated H&E", fontsize=10, rotation=90, labelpad=2)

        diff_ax = fig.add_axes([x0, bottom_row_y, panel_width, row_height])
        if cell_mask is not None:
            last_im = _render_cell_masked_overlay(diff_ax, raw_diff, cell_mask, img_all, hot_cmap)
        else:
            last_im = diff_ax.imshow(
                fallback_diffs[index], cmap=hot_cmap, vmin=0.0, vmax=1.0
            )
        diff_ax.set_xticks([])
        diff_ax.set_yticks([])
        if index == 0:
            diff_ax.set_ylabel("Pixel Diff", fontsize=10, rotation=90, labelpad=2)
```

- [ ] **Step 5: Update the colorbar label**

Locate the colorbar block (around lines 282–289):

```python
        cbar = fig.colorbar(last_im, cax=cax, orientation="horizontal")
        cbar.ax.tick_params(labelsize=8, pad=1)
        cbar.set_label("Pixel diff (per-condition, 99th-pct norm.)", fontsize=8, labelpad=3)
```

Replace the `cbar.set_label` line with:

```python
        cbar = fig.colorbar(last_im, cax=cax, orientation="horizontal")
        cbar.ax.tick_params(labelsize=8, pad=1)
        cbar_label = (
            "Cell-masked pixel diff (per-condition, 99th-pct norm.)"
            if cell_mask is not None
            else "Pixel diff (per-condition, 99th-pct norm.)"
        )
        cbar.set_label(cbar_label, fontsize=8, labelpad=3)
```

- [ ] **Step 6: Run the full test suite**

```bash
python -m pytest tests/test_leave_one_out_diff.py -v
```

Expected: All tests PASS, including the two new smoke tests and the existing `test_render_loo_figure_with_cached_cell_mask`.

- [ ] **Step 7: Commit**

```bash
git add tools/vis/leave_one_out_diff.py tests/test_leave_one_out_diff.py
git commit -m "feat: cell-masked LOO diff overlay — greyscale bg + hot-colormap on cell regions"
```

---

## Final Verification

- [ ] **Run full suite one more time**

```bash
python -m pytest tests/test_leave_one_out_diff.py tests/test_leave_one_out_stats_cli.py -v
```

Expected: All existing + new tests PASS.

- [ ] **Spot-check the output PNG**

```bash
python tools/vis/leave_one_out_diff.py \
    --cache-dir inference_output/unpaired_ablation/ablation_results/10240_11008 \
    --orion-root data/orion-crc33 \
    --style-mapping-json inference_output/unpaired_ablation/data/orion-crc33-unpaired/metadata/unpaired_mapping.json \
    --out inference_output/unpaired_ablation/leave_one_out/10240_11008/leave_one_out_diff_overlay.png
```

Visually confirm: background is dimmed greyscale; cell nuclei light up in red/yellow where the dropped channel had an effect; the "All four channels" column shows black cells (zero self-diff) on a grey background.
