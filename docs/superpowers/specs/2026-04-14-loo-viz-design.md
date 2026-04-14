# LOO Visualization — Publication Figure Design

**Date:** 2026-04-14  
**Status:** Approved for implementation

## Problem

The existing leave-one-out (LOO) visualizations (`tools/vis/leave_one_out_diff.py`) show *where* pixels change when a channel group is dropped, but do not answer the question that matters:

> **How does dropping a channel group change what the H&E looks like — specifically cell morphology and staining texture?**

The current designs (raw pixel diff heatmap; cell-masked hot overlay) are useful for localization but fail as publication figures because the diff map is the hero instead of the actual H&E appearance.

The ablation results show that `cell_state` and `microenv` have the largest impact on generation. The figure must make that legible visually.

## Goal

A single publication-quality figure per tile showing all four LOO conditions that directly answers:

- **A — Cell morphology:** Does dropping a channel change cell shape, size, or arrangement?
- **D — Microenv texture:** Does dropping microenv change staining intensity, nuclear density, or cytoplasmic patterns?

## Layout

**3 rows × 5 columns.** Figure size: ~15 × 9 inches at 150 dpi.

### Columns (left to right)

| # | Label | Content |
|---|-------|---------|
| 1 | All channels | Baseline generation (all four TME groups active) |
| 2 | Drop Cell Types | LOO: cell_types omitted |
| 3 | Drop Cell State ★ | LOO: cell_state omitted |
| 4 | Drop Vasculature | LOO: vasculature omitted |
| 5 | Drop Microenv ★ | LOO: microenv omitted |

★ = column headers for cell_state and microenv are typographically highlighted (bold + accent color) to reinforce the ablation story.

### Row 1 — Generated H&E (full tile)

- Each panel: full 256×256 generated H&E image (square, rendered at full resolution).
- The **all-channels** panel has a teal rectangle drawn on it marking the inset crop region used in Row 2.
- Cell mask contour overlay (lime, 0.7px) when available.
- Row label (y-axis): `"Generated H&E"`

### Row 2 — Cell inset (morphology)

- Each panel: 64×64 crop from the same fixed region (identical bounding box across all columns), upsampled 4× to 256×256 with nearest-neighbour to preserve pixel sharpness.
- **Crop selection:** sliding window (stride 8) over the 256×256 image; pick the top-left corner where the mean SSIM loss averaged across all four drop conditions is highest. This auto-selects the region most impacted across conditions.
- Panel border: `#00ccaa` (teal, 2px) for the all-channels column; `#ff6644` (orange-red, 2px) for Drop Cell State; `#ddaa00` (amber, 2px) for Drop Microenv; `#555555` (neutral, 1px) for the other two.
- Row label: `"Cell inset (auto-selected)"`

### Row 3 — SSIM structural loss (cell-masked)

- Each panel: per-pixel SSIM loss map, same 256×256 size as Row 1 (square).
- **All-channels column:** shown as a uniform black panel labelled "0 (baseline)" — no diff to itself.
- **Computation:** `loss = 1 - SSIM(drop_i_gray, all_channels_gray)` using `skimage.metrics.structural_similarity` with `full=True`, `win_size=11`, `data_range=255`, grayscale input.
- **Masking:** loss values outside the cell mask (binary, `cell_mask > 0.5`) are zeroed.
- **Normalization:** global max across all four drop conditions — magnitudes are comparable across columns, so Drop Cell State and Drop Microenv will naturally appear brighter.
- **Colormap:** `hot` (black → red → yellow → white). Prints correctly in grayscale.
- Row label: `"SSIM loss (cell-masked)"`

### Colorbar

- Positioned beneath the **last column** (Drop Microenv), exactly one subplot wide.
- Horizontal orientation, tick labels at 0, 0.5, 1.
- Label: `"SSIM structural loss (globally normalized)"`

## Implementation Notes

### Inset selection algorithm

The four SSIM loss maps must be computed before inset selection. `loss_mean` is their per-pixel average.

```python
# loss_mean: H×W float32, mean SSIM loss across 4 drop conditions
crop = 64
stride = 8
best_score, best_yx = -1.0, (0, 0)
for y in range(0, H - crop + 1, stride):
    for x in range(0, W - crop + 1, stride):
        score = float(loss_mean[y:y+crop, x:x+crop].mean())
        if score > best_score:
            best_score, best_yx = score, (y, x)
```

### SSIM per-pixel map

```python
from skimage.metrics import structural_similarity as ssim
import numpy as np

def ssim_loss_map(img_all: np.ndarray, img_drop: np.ndarray) -> np.ndarray:
    """Return H×W float32 SSIM loss in [0, 1]; inputs are H×W×3 uint8."""
    gray_all  = img_all.mean(axis=2).astype(np.float64)
    gray_drop = img_drop.mean(axis=2).astype(np.float64)
    _, ssim_map = ssim(gray_all, gray_drop, full=True, win_size=11, data_range=255)
    return np.clip(1.0 - ssim_map, 0.0, 1.0).astype(np.float32)
```

### File location

New function `render_loo_ssim_figure()` added to `tools/vis/leave_one_out_diff.py`, following the same CLI interface as the existing `render_loo_diff_figure()`. Existing figure generation is preserved unchanged.

Output filename: `leave_one_out_ssim.png` (alongside existing `leave_one_out_diff.png`).

### Dependencies

- `scikit-image` (already present for `skimage.color`): add `skimage.metrics.structural_similarity`
- No new packages required

## What This Figure Answers

| Row | Question answered |
|-----|------------------|
| Full H&E | What does the overall tissue look like under each condition? |
| Cell inset | Did cell morphology (shape, arrangement) change in the most-impacted region? |
| SSIM loss | Where is structural texture (nuclear density, staining pattern) degraded by the dropout? |

The combination lets a reader directly compare: "With cell_state dropped, the cells in the inset look rounder / less differentiated" (Row 2) and "The SSIM map confirms widespread structural change concentrated in cell regions" (Row 3).
