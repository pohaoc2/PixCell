# Cell-Masked LOO Diff Overlay — Design Spec

**Date:** 2026-04-13  
**File:** `tools/vis/leave_one_out_diff.py`  
**Status:** Approved

---

## Problem

The leave-one-out diff visualization (`render_loo_diff_figure`) shows a global pixel-diff heatmap in the bottom row. Because PixCell is a diffusion model, background pixels change stochastically every run — even with a fixed seed, the spatial layout of background texture shifts between conditions. This swamps the biologically meaningful signal: the change in **cell-nucleus pixels** when a TME channel group is dropped.

---

## Goal

Replace the current diff row with a **cell-masked overlay** that:
- Zeroes out background noise completely
- Shows hot-colormap diff intensity only inside cell boundaries
- Preserves spatial context via a dimmed greyscale version of the baseline H&E

---

## Scope

One file: `tools/vis/leave_one_out_diff.py`.  
Figure layout (top row H&E, bottom row diff, colorbar, title) is unchanged.  
`compute_loo_diffs` (stats JSON) and `_compute_relative_diff_maps` are not modified.

---

## Compositing Pipeline (per diff panel)

```
raw_diff   = |LOO_image − img_all|.mean(axis=2)          # H×W float32
masked_diff = raw_diff * (cell_mask > 0.5)                # zero out background

p99 = percentile(masked_diff[cell_mask > 0.5], 99)        # cell-region scale only
diff_norm = clip(masked_diff / p99, 0, 1)                 # [0, 1]

bg = img_all.mean(axis=2) / 255 * bg_brightness           # greyscale, 50% dim
heatmap_rgba = hot_cmap(diff_norm)                        # H×W×4

alpha = (cell_mask > 0.5).astype(float32)                 # binary
composite[c] = alpha * heatmap_rgba[c] + (1−alpha) * bg  # per channel
```

**Edge cases:**
- If no cell-mask pixels exist (`cell_mask` is all zeros, or `cell_mask` is `None`): fall back to the existing per-map normalized diff (i.e., call `_compute_relative_diff_maps(..., per_map=True)` and `imshow` as before).
- If `p99 <= 0` (all cell pixels are zero diff): return a zero diff map (black composite in cell regions, greyscale elsewhere).
- The "All four channels" column always has `raw_diff = 0`, so all cells appear black — a correct visual anchor showing no change from baseline.

---

## New Private Helpers

### `_normalize_cell_masked_diff`

```python
def _normalize_cell_masked_diff(
    diff: np.ndarray,       # H×W float32, raw absolute diff (values in [0, 255])
    cell_mask: np.ndarray,  # H×W float32 [0, 1]
) -> np.ndarray:            # H×W float32 in [0, 1]
```

Zeros non-cell pixels, then normalizes by the 99th percentile of cell-region pixels. Returns `np.zeros_like(diff)` when `p99 <= 0`.

### `_render_cell_masked_overlay`

```python
def _render_cell_masked_overlay(
    ax,
    raw_diff: np.ndarray,       # H×W float32
    cell_mask: np.ndarray,      # H×W float32 [0, 1]
    baseline_he: np.ndarray,    # H×W×3 uint8
    cmap,
    *,
    bg_brightness: float = 0.5,
) -> matplotlib.cm.ScalarMappable:
```

Executes the compositing pipeline above and calls `ax.imshow(composite)`. Returns a `ScalarMappable(cmap, Normalize(0,1))` for use as the colorbar source (because `imshow` of an RGB array has no built-in mappable).

---

## Changes to `render_loo_diff_figure`

1. Compute raw diffs inline (not via `_compute_relative_diff_maps`):
   ```python
   baseline_float = img_all.astype(np.float32)
   raw_diffs = [
       np.abs(img.astype(np.float32) - baseline_float).mean(axis=2)
       for img in display_images
   ]
   ```
2. Hoist the fallback diff maps when `cell_mask is None` (computed once before the loop):
   ```python
   fallback_diffs = (
       None if cell_mask is not None
       else _compute_relative_diff_maps(display_images, img_all, per_map=True)
   )
   ```
3. In the per-column loop, replace:
   ```python
   last_im = diff_ax.imshow(diff_map, cmap=hot_cmap, vmin=0.0, vmax=1.0)
   ```
   with:
   ```python
   if cell_mask is not None:
       last_im = _render_cell_masked_overlay(
           diff_ax, raw_diffs[index], cell_mask, img_all, hot_cmap
       )
   else:
       last_im = diff_ax.imshow(
           fallback_diffs[index], cmap=hot_cmap, vmin=0.0, vmax=1.0
       )
   ```
4. The existing `display_diffs = _compute_relative_diff_maps(...)` call is removed.

---

## Colorbar

In the overlay path, `_render_cell_masked_overlay` returns a `matplotlib.cm.ScalarMappable(cmap=hot_cmap, norm=Normalize(0, 1))` because `imshow` of an RGB composite array provides no built-in mappable. This is passed to `fig.colorbar(...)`.

In the fallback path (no cell mask), `last_im` from the final `diff_ax.imshow(...)` is used directly — the existing behavior.

Labels:
- Overlay path: `"Cell-masked pixel diff (per-condition, 99th-pct norm.)"`
- Fallback path: `"Pixel diff (per-condition, 99th-pct norm.)"`

---

## Files Changed

| File | Change |
|------|--------|
| `tools/vis/leave_one_out_diff.py` | Add `_normalize_cell_masked_diff`, `_render_cell_masked_overlay`; update `render_loo_diff_figure` loop and colorbar |

---

## Not Changed

- `compute_loo_diffs` — stats JSON, global normalization, no mask
- `_compute_relative_diff_maps` — kept as utility; no longer called by figure renderer
- Figure layout, axes geometry, contour overlays, title, reference H&E panel
