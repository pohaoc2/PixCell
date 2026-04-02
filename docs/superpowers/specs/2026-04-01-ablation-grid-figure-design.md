# Ablation Grid Figure — Design Spec

**Date:** 2026-04-01  
**Replaces:** `docs/ablation_vis_spec.md` (original strip layout)  
**Output script:** `tools/stage3_ablation_grid_figure.py`

---

## Goal

A single publication-quality PNG/PDF showing all 16 ablation conditions (4×1-ch + 6×2-ch + 4×3-ch + 1×4-ch + 1×Real H&E reference) arranged in a 4×4 grid sorted by cosine similarity. Each cell contains the generated H&E thumbnail, active-channel dot indicators, a colored square border encoding cardinality, and a cosine score bar.

---

## Layout

### Grid structure

- **4 columns × 4 rows = 16 cells**, filled left-to-right, top-to-bottom.
- Sorted **descending by cosine similarity**: top-left = best-performing condition, bottom-right = Real H&E reference.
- Real H&E has no cosine score and is always placed last (cell [3, 3]).
- All-4-ch is a real inference run and participates in sorting alongside the 14 ablation conditions.

### Ordering of 15 scored conditions

1. Compute cosine similarity for each of the 15 conditions (14 ablation + All-4-ch).
2. Sort descending. Ties broken lexicographically by condition key.
3. Fill cells [0,0]→[3,2] with the 15 sorted conditions.
4. Cell [3,3] = Real H&E reference (always fixed here).

---

## Cell Design

Each cell has three stacked regions (top to bottom):

```
┌──────────────────┐
│  ● ● ○ ●        │  ← dot row (4 dots, one per channel)
├──────────────────┤
│                  │
│   H&E thumbnail  │  ← square image (no border-radius)
│                  │
│  [═══════░░░░]   │  ← cosine bar (full-width, thin)
├──────────────────┤
│   CT+CS+Nu  0.91 │  ← condition label + score text
└──────────────────┘
```

### Dot row

- 4 dots, left-to-right order: **CT · CS · Va · Nu** (matches channel group order in codebase).
- **Filled dot** (•): channel is active in this condition. Color = cardinality group color (see below).
- **Open ring** (○): channel inactive. Gray stroke (`#CCCCCC`), no fill.
- Dot diameter: ~5 pt. Gap between dots: ~3 pt.
- Real H&E reference: dot row omitted (empty spacer of same height).

### Image thumbnail

- Source: generated H&E PNG from `manifest.json → entry["image_path"]`.
- Real H&E source: `orion_root/he/<tile_id>.png` (or `orion_root/he_tiles/<tile_id>.png`; script tries both).
- Displayed square; aspect ratio preserved via `imshow` with equal aspect.
- **Square border** (no `border_radius`): drawn as a colored rectangle patch around the axes, 2.5 pt linewidth, color = cardinality group (see below).
- Best condition (highest cosine): faint `#FFFBE6` axes background + `★` appended to score text.
- Real H&E: dashed border (`linestyle='--'`), color `#999999`.

### Cosine bar

- Full-width horizontal bar below the image axes (separate thin axes strip).
- Bar fill proportional to cosine value on [−1, 1] scale (left edge = −1, right = 1).
- Bar color = cardinality group color at 50% alpha.
- Score text right-aligned after bar: e.g., `0.91 ★`.
- Real H&E: bar empty, label shows `"reference"`.

---

## Color Encoding (Okabe-Ito, colorblind-safe)

| Cardinality | Color | Hex |
|---|---|---|
| 1-ch | Bluish green | `#009E73` |
| 2-ch | Blue | `#0072B2` |
| 3-ch | Vermillion | `#D55E00` |
| 4-ch (all) | Purple | `#9B59B6` |
| Real H&E | Gray dashed | `#999999` |
| Inactive dot | Light gray | `#CCCCCC` |

The same color is used for: filled dots, square border, and cosine bar fill.

---

## Figure Dimensions

| Parameter | Value |
|---|---|
| Figure width | 9.0 in |
| Figure height | auto (see below) |
| DPI | 300 (print), 150 (preview) |
| Cell image size | ~1.5 in × 1.5 in |
| Column gap | 8 pt |
| Row gap | 8 pt |
| Outer margins | tight (`bbox_inches='tight'`, `pad_inches=0.1`) |

**Height formula:** `fig_h = top_margin + 4 × (dot_row + image + bar + label) + 3 × row_gap + bottom_margin`

In practice: `fig_h ≈ 10.0` for a 4×4 grid at ~1.5 in per image cell.

**GridSpec layout per cell column:**

Each column in the GridSpec holds four row-slots: dots, image, cosine bar, label. The 4 image columns share equal width. No separate group-label column.

---

## Data Requirements

### Inputs

| Source | Path | Notes |
|---|---|---|
| Manifest | `<cache_dir>/manifest.json` | 14 ablation conditions + cell mask path |
| All-4-ch image | `<cache_dir>/all/<tile_id>.png` | Fixed path inside cache dir; **required** |
| Real H&E | `<orion_root>/he/<tile_id>.png` | **Required** |
| Real H&E UNI features | `data/features/<tile_id>_uni.npy` | Cached reference embedding for cosine computation |
| Cell mask | path from manifest `cell_mask_path` | Optional; lime contour overlay if present |

### All-4-ch condition

The ablation cache stores 14 subset conditions (1-ch through 3-ch). The All-4-ch run is saved separately at `<cache_dir>/all/<tile_id>.png`. The script loads it from that fixed path and inserts it into the sorted grid alongside the 14 ablation conditions.

### Cosine similarity computation

For each generated image (14 ablation + All-4-ch = 15 total):
1. Extract UNI-2h features from the generated H&E PNG using the UNI encoder.
2. Compute cosine similarity against the cached real H&E UNI features at `data/features/<tile_id>_uni.npy`.

Results are cached to `<cache_dir>/uni_cosine_scores.json` (`per_condition` dict keyed by `condition_metric_key`). On subsequent runs the script reads the JSON directly and skips re-extraction.

Real H&E has no cosine score (it is the reference); its cell always occupies position [3, 3].

---

## CLI Interface

```bash
python tools/stage3_ablation_grid_figure.py \
  --cache-dir inference_output/test_combinations/<tile_id> \
  --orion-root data/orion-crc33 \
  --all4ch-image inference_output/all4ch/<tile_id>/generated.png \
  [--output-name ablation_grid]   # default: ablation_grid
  [--dpi 300]
  [--no-auto-cosine]
  [--device cuda]
```

Output: `<cache_dir>/ablation_grid.png` and `<cache_dir>/ablation_grid.pdf`.

---

## Implementation Notes

### New script vs existing

Create `tools/stage3_ablation_grid_figure.py` as a **new script**. Do not modify `stage3_ablation_pub_figure.py` — that script (strip layout) remains for backward compatibility.

### Shared utilities reused

- `tools/stage3_ablation_cache.py`: `is_per_tile_cache_manifest_dir`, `list_cached_tile_ids`
- `tools/stage3_ablation_vis_utils.py`: `FOUR_GROUP_ORDER`, `condition_metric_key`, `ordered_subset_condition_tuples`, `compute_rgb_pixel_cosine_scores`, `default_orion_uni_npy_path`
- `tools/compute_ablation_uni_cosine.py`: UNI cosine computation (unchanged)

### GridSpec structure

```
GridSpec rows per cell group (4 rows total):
  - dot_row:   height_ratio 0.12  (dots only)
  - image_row: height_ratio 1.0   (H&E imshow)
  - bar_row:   height_ratio 0.08  (cosine bar)
  - label_row: height_ratio 0.12  (text)

GridSpec columns: 4, equal width.
hspace = 0.10, wspace = 0.06
```

Border drawn as `ax.spines[side].set_linewidth(2.5)` + `set_edgecolor(color)` on the image axes (not as a patch), so it renders correctly in PDF vector output.

### Cosine score loading

Reuse `_parse_cosine_json` / `_load_or_compute_cosine_scores` from `stage3_ablation_pub_figure.py` (extract to shared util or copy). The All-4-ch cosine score is looked up from the same JSON if present.

### Real H&E path resolution

```python
def find_real_he(orion_root: Path, tile_id: str) -> Path | None:
    for subdir in ("he", "he_tiles"):
        p = orion_root / subdir / f"{tile_id}.png"
        if p.is_file():
            return p
    return None
```

Warn and skip the Real H&E cell if not found.

---

## Open Questions

1. **All-4-ch cosine key format** — confirm the canonical key string produced by `condition_metric_key(("cell_types","cell_state","vasculature","microenv"))`. May differ from the 14-condition keys if group name ordering differs.
2. **Real H&E directory structure** — `orion-crc33` layout needs to be confirmed before hardcoding fallback paths.
3. **Multi-tile support** — script should loop over per-tile cache subdirs (same pattern as `stage3_ablation_pub_figure.py`) when given a parent directory.
