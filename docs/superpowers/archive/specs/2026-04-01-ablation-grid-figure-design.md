# Ablation Grid Figure — Design Spec

**Date:** 2026-04-01  
**Last updated:** 2026-04-02  
**Replaces:** `docs/ablation_vis_spec.md` (original strip layout)  
**Output scripts:** `tools/stage3_ablation_grid_figure.py` (static PNG/PDF), `tools/stage3_ablation_grid_webvis.py` (interactive HTML — planned)

---

## Goal

Two complementary outputs for the same 4×4 ablation grid:

1. **Static figure** (`ablation_grid.png` / `.pdf`) — publication-quality matplotlib PNG/PDF, sorted by primary metric, with colored borders, channel dot indicators, and per-cell metric bars.
2. **Interactive web vis** (`ablation_grid.html`) — standalone self-contained HTML with sortable columns, hover tooltips, and all metrics visible. Designed for exploration and design review.

Both show 16 cells: 14 ablation conditions + All-4-ch + Real H&E reference.

---

## Metrics

### Implemented

| Metric | Key | Comparison | Range | Better |
|---|---|---|---|---|
| Cosine (UNI-2h) | `cosine` | ref H&E UNI embedding vs gen H&E UNI embedding | [−1, 1] | higher |

### Planned (next agent)

| Metric | Key | Comparison | Range | Better | Notes |
|---|---|---|---|---|---|
| LPIPS | `lpips` | ref H&E vs gen H&E (perceptual, AlexNet) | [0, 1] | **lower** | per-tile; `lpips` library |
| AJI | `aji` | input `cell_masks` vs CellViT detections on gen H&E | [0, 1] | higher | Aggregated Jaccard Index; standard in pathology |
| PQ | `pq` | input `cell_masks` vs CellViT detections on gen H&E | [0, 1] | higher | Panoptic Quality = SQ × RQ; RQ ≡ detection F1 |

**Dropped:** standalone pixel Dice, standalone F1 — both subsumed by AJI and PQ.  
**Dataset-level (future):** FID on UNI-2h embeddings across ≥ 500 tiles.

### Metrics JSON schema

All four per-condition metrics are stored in a unified `<cache_dir>/metrics.json`:

```json
{
  "version": 2,
  "tile_id": "17408_32768",
  "per_condition": {
    "cell_types": {
      "cosine": 0.9946,
      "lpips": null,
      "aji": null,
      "pq": null
    },
    ...
  }
}
```

**Migration:** existing `uni_cosine_scores.json` files should be read as a fallback when `metrics.json` is absent; `_parse_cosine_json` already handles this.

---

## Layout

### Grid structure

- **4 columns × 4 rows = 16 cells**, filled left-to-right, top-to-bottom.
- Sorted **descending by selected primary metric** (default: cosine). Ties broken lexicographically by condition key.
- Real H&E has no metric scores and is always placed last (cell [3, 3]).
- All-4-ch participates in sorting alongside the 14 ablation conditions.

---

## Cell Design

Each cell has four stacked regions (top to bottom):

```
┌──────────────────────┐
│  ● ● ○ ●            │  ← dot row (4 dots, one per channel group)
│ ┌──────────────────┐ │
│ │                  │ │
│ │   H&E thumbnail  │ │  ← square image, colored spine border
│ │                  │ │
│ └──────────────────┘ │
│  Co [████████────]   │  ← 4 stacked metric bars (Co / LP / AJ / PQ)
│  LP [──────────]     │    placeholder shown as dashed gray if not computed
│  AJ [──────────]     │
│  PQ [──────────]     │
│   CT+CS+Nu  0.9961 ★ │  ← condition label + primary metric score
└──────────────────────┘
```

### Dot row

- 4 dots, left-to-right order: **CT · CS · Va · Nu** (matches `FOUR_GROUP_ORDER`).
- **Filled dot**: channel active. Color = cardinality group color.
- **Open ring**: channel inactive. Stroke `#CCCCCC`, no fill.
- Real H&E: dot row is an empty spacer of the same height.

### Image thumbnail

- Source: generated H&E PNG from `manifest.json → entry["image_path"]` for ablation conditions; `<cache_dir>/all/generated_he.png` for All-4-ch.
- Real H&E source: `<orion_root>/he/<tile_id>.png`.
- **Important:** the All-4-ch image is stored at `<cache_dir>/all/generated_he.png`, not `<cache_dir>/all/<tile_id>.png`. Always pass `--all4ch-image` explicitly to the CLI or use the corrected default.
- Square border: `ax.spines` colored at 2.5 pt, color = cardinality group.
- Best condition (rank 1 by primary metric): faint `#FFFBE6` axes background + `★` appended to score text.
- Real H&E: dashed border (`--`), color `#999999`.

### Metric bars (static figure)

Replaces the former single cosine bar. Four thin horizontal bars stacked vertically below the image:

| Bar | Label | Color | Notes |
|---|---|---|---|
| Cosine | `Co` | `#0072B2` (blue) | normalized to observed [min, max] across all conditions for visibility |
| LPIPS | `LP` | `#D55E00` (vermillion) | inverted: lower=better → bar fill = `1 − lpips` |
| AJI | `AJ` | `#009E73` (green) | [0, 1] direct |
| PQ | `PQ` | `#9B59B6` (purple) | [0, 1] direct |

Uncomputed metrics shown as dashed gray placeholder bar (not hidden, to preserve layout stability).

**GridSpec height ratios per cell:**

```
dot_row:    0.12
image_row:  1.0
bars_row:   0.35   ← was 0.08 (single bar); expanded for 4 bars
label_row:  0.12
```

### Metric bars (web vis)

Identical color/order to static figure. Bar width proportional to within-set min-max normalized value. Uncomputed bars use CSS dashed pattern. Each bar has a 2-letter label on the left (`Co`, `LP`, `AJ`, `PQ`).

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

Metric bar colors are fixed (not cardinality-dependent) so the same metric is always the same color across cells.

---

## Figure Dimensions (static)

| Parameter | Value |
|---|---|
| Figure width | 9.0 in |
| Figure height | ≈ 10.5 in (expanded for 4 metric bars) |
| DPI | 300 (print), 150 (preview) |
| Column gap | `wspace=0.06` |
| Row gap | `hspace=0.10` |
| Outer margins | `bbox_inches='tight'`, `pad_inches=0.1` |

---

## Web Vis Features

Implemented in `tools/stage3_ablation_grid_webvis.py` (planned); currently generated inline.

| Feature | Description |
|---|---|
| Sort controls | Buttons for Cosine / LPIPS / AJI / PQ; re-ranks grid live via JS |
| Rank badge | `#N` shown top-left of each cell |
| Hover tooltip | Shows condition name, rank, all 4 metric values with mini bars |
| Real H&E pin | Always fixed at position 16 regardless of sort |
| Standalone HTML | All images base64-embedded; no server required after generation |
| Placeholder state | LPIPS/AJI/PQ bars shown as dashed gray when not yet computed |

---

## Data Requirements

### Inputs

| Source | Path | Notes |
|---|---|---|
| Manifest | `<cache_dir>/manifest.json` | 14 ablation conditions |
| All-4-ch image | `<cache_dir>/all/generated_he.png` | **Note: filename is `generated_he.png`, not `<tile_id>.png`** |
| Real H&E | `<orion_root>/he/<tile_id>.png` | Reference image |
| Metrics JSON | `<cache_dir>/metrics.json` | Unified metrics; falls back to `uni_cosine_scores.json` |
| Real H&E UNI features | `<orion_root>/features/<tile_id>_uni.npy` | For cosine computation |
| Cell masks | `<orion_root>/exp_channels/cell_masks/<tile_id>.png` | For AJI/PQ vs CellViT detections |
| CellViT model | `pretrained_models/cellvit-sam-h/` | For cell detection on generated H&E |

### AJI / PQ computation pipeline

1. Run CellViT-SAM-H on each generated H&E → instance segmentation masks.
2. Load ground-truth `cell_masks` from input channels (binary, connected components = GT instances).
3. Match GT instances to predicted instances using Hungarian matching at IoU ≥ 0.5.
4. Compute AJI (Aggregated Jaccard Index) and PQ = SQ × RQ per condition.
5. Write to `<cache_dir>/metrics.json`.

---

## CLI Interface

### Static figure

```bash
python tools/stage3_ablation_grid_figure.py \
  --cache-dir inference_output/test_combinations/<tile_id> \
  --orion-root data/orion-crc33 \
  --all4ch-image inference_output/test_combinations/<tile_id>/all/generated_he.png \
  [--output-name ablation_grid] \
  [--dpi 300] \
  [--no-auto-cosine] \
  [--sort-by cosine|lpips|aji|pq] \
  [--device cuda]
```

### Web vis (planned)

```bash
python tools/stage3_ablation_grid_webvis.py \
  --cache-dir inference_output/test_combinations/<tile_id> \
  --orion-root data/orion-crc33 \
  --all4ch-image inference_output/test_combinations/<tile_id>/all/generated_he.png \
  [--output-name ablation_grid]
```

Output: `<cache_dir>/ablation_grid.html` (self-contained, base64 images embedded).

---

## Implementation Notes

### Resolved open questions (from original spec)

1. **All-4-ch key format** — confirmed: `condition_metric_key(FOUR_GROUP_ORDER)` = `"cell_state+cell_types+microenv+vasculature"` (alphabetical within the join).
2. **Real H&E path** — confirmed: `<orion_root>/he/<tile_id>.png`. No `he_tiles/` fallback needed for current dataset.
3. **Multi-tile support** — implemented: `main()` calls `list_cached_tile_ids()` and loops when given a parent directory.

### Sort bug fix

`_sort_conditions_by_cosine` uses `float("-inf")` as the default for missing scores (not `float("inf")`). Using `float("inf")` when negated gives `-inf` which sorts missing values *first*; the fix puts them last.

### Shared utilities

- `tools/stage3_ablation_cache.py`: `is_per_tile_cache_manifest_dir`, `list_cached_tile_ids`
- `tools/stage3_ablation_vis_utils.py`: `FOUR_GROUP_ORDER`, `condition_metric_key`, `ordered_subset_condition_tuples`
- `tools/compute_ablation_uni_cosine.py`: cosine computation (unchanged)
- `tools/uni_cosine_similarity.py`: `cosine_similarity_uni`, `flatten_uni_npy`
