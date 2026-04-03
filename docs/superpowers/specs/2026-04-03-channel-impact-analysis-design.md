# Channel Impact Analysis Design
**Date:** 2026-04-03  
**Goal:** Understand how each TME channel affects generated H&E appearance through two complementary analyses: (1) leave-one-out group-level pixel diff from cached images, and (2) systematic single-tile channel manipulation experiments.

---

## Overview

Three scripts, three experiments:

| Script | Purpose |
|---|---|
| `tools/vis/leave_one_out_diff.py` | Post-process existing cache: group-level leave-one-out pixel diff |
| `tools/stage3/classify_tiles.py` | Scan all tiles, assign two-axis class labels, select representative tiles |
| `tools/stage3/channel_sweep.py` | Run three channel manipulation experiments on selected tiles |

---

## Script 1: Leave-one-out Pixel Diff (`tools/vis/leave_one_out_diff.py`)

**No new inference required** — pure post-processing of existing `inference_output/cache/<tile_id>/`.

### Inputs
```
--cache-dir   inference_output/cache/512_9728
--orion-root  data/orion-crc33          # optional, for channel thumbnails
--out         inference_output/cache/512_9728/leave_one_out_diff.png
```

### Algorithm
1. Load `manifest.json` → map `active_groups → image_path` for all 15 conditions
2. Load `all/generated_he.png` as float32 baseline `img_all`
3. For each of the 4 groups `g`:
   - Find the triples entry where `active_groups = FOUR_GROUP_ORDER − {g}`
   - Load as `img_loo[g]`
   - Compute `diff[g] = |img_all − img_loo[g]|`, mean across RGB channels
4. Normalize all 4 diff maps to a **common global max** (so magnitudes are comparable across groups)
5. Save `leave_one_out_diff_stats.json`: `{group: {mean_diff, max_diff, pct_pixels_above_10}}`

### Figure Layout
```
             cell_types   cell_state   vasculature   microenv
channel      [thumbnail]  [thumbnail]  [thumbnail]   [thumbnail]   ← from build_exp_channel_header_rgb()
leave-one-out [H&E]       [H&E]        [H&E]         [H&E]
diff heatmap  [hot cmap]  [hot cmap]   [hot cmap]    [hot cmap]    ← globally normalized
stats         mean±σ      mean±σ       mean±σ        mean±σ
```
- Optional leftmost column: `img_all` baseline + real H&E (if `--orion-root` provided)
- Uses `ablation_vis_utils.ordered_subset_condition_tuples()` to find triples index per group
- Uses `ablation_vis_utils.build_exp_channel_header_rgb()` for thumbnails

---

## Script 2: Tile Classification (`tools/stage3/classify_tiles.py`)

**Scans all ~10K tiles** in `data/orion-crc33/exp_channels/` and assigns two-axis class labels. Also selects special tiles for Experiments 2 and 3.

### Inputs
```
--exp-root  data/orion-crc33
--out       tile_classes.json
```

### Per-tile Statistics
For every tile, load PNGs + NPYs and compute:
- `cell_density` = mean(cell_masks)
- `cancer_frac` = mean(cell_type_cancer) / (mean(cell_masks) + ε)
- `immune_frac` = mean(cell_type_immune) / (mean(cell_masks) + ε)
- `healthy_frac` = mean(cell_type_healthy) / (mean(cell_masks) + ε)
- `prolif_frac` = mean(cell_state_prolif) / (mean(cell_masks) + ε)
- `nonprolif_frac` = mean(cell_state_nonprolif) / (mean(cell_masks) + ε)
- `dead_frac` = mean(cell_state_dead) / (mean(cell_masks) + ε)
- `mean_oxygen` = mean(oxygen.npy)
- `mean_glucose` = mean(glucose.npy)

**Filter:** discard tiles where `cell_density < P5` (near-blank tiles).

### Two-Axis Classification

**Axis 1 — cell composition** (mutually exclusive, priority order):
- `cancer` if `cancer_frac > P75`
- `immune` if `immune_frac > P75` AND `cancer_frac > P25`
- `healthy` if `healthy_frac > P75` AND `cancer_frac < P25`
- else: unlabeled (excluded from representative selection)

**Axis 2 — metabolic state:**
- `hypoxic` if `mean_oxygen < P25`
- `glucose_low` if `mean_glucose < P25`
- `neutral` otherwise

### Representative Selection (for cross-experiment pairing)

For each of the 9 axis combinations, select the tile maximizing a **joint purity score**:
- `cancer+hypoxic` → argmax(`cancer_frac_rank + oxygen_low_rank`)
- `immune+glucose_low` → argmax(`immune_frac_rank + glucose_low_rank`)
- etc.

### Special Tile Selection (for Experiments 2 and 3)

**Experiment 2 (cell type relabeling)** — 3 tiles, one per dominant type:
- Best cancer tile: argmax(`cancer_frac`) where `cancer_frac > 0.8`
- Best immune tile: argmax(`immune_frac`) where `immune_frac > 0.8`
- Best healthy tile: argmax(`healthy_frac`) where `healthy_frac > 0.8`

**Experiment 3 (cell state relabeling)** — 3 tiles, one per dominant state:
- Best prolif tile: argmax(`prolif_frac`) where `prolif_frac > 0.8`
- Best nonprolif tile: argmax(`nonprolif_frac`) where `nonprolif_frac > 0.8`
- Best dead tile: argmax(`dead_frac`) where `dead_frac > 0.8`

### Output Schema
```json
{
  "thresholds": {
    "cancer_frac_p75": 0.52,
    "immune_frac_p75": 0.18,
    "healthy_frac_p75": 0.71,
    "oxygen_p25": 0.85,
    "glucose_p25": 0.88,
    "cell_density_p5": 0.003
  },
  "representatives": {
    "cancer+hypoxic": {"tile_id": "10240_11520", "scores": {...}},
    "immune+neutral":  {"tile_id": "...", "scores": {...}}
  },
  "exp2_tiles": {
    "cancer":  {"tile_id": "...", "cancer_frac": 0.87},
    "immune":  {"tile_id": "...", "immune_frac": 0.91},
    "healthy": {"tile_id": "...", "healthy_frac": 0.93}
  },
  "exp3_tiles": {
    "prolif":    {"tile_id": "...", "prolif_frac": 0.84},
    "nonprolif": {"tile_id": "...", "nonprolif_frac": 0.82},
    "dead":      {"tile_id": "...", "dead_frac": 0.81}
  },
  "all_tiles": {
    "10240_11520": {
      "axis1": "cancer", "axis2": "hypoxic",
      "cancer_frac": 0.74, "mean_oxygen": 0.31, ...
    }
  }
}
```

---

## Script 3: Channel Sweep (`tools/stage3/channel_sweep.py`)

Runs all three experiments using models loaded once. Reads tile selection from `tile_classes.json`.

### Inputs
```
--class-json      tile_classes.json
--data-root       data/orion-crc33
--checkpoint-dir  checkpoints/pixcell_controlnet_exp/checkpoints
--out             inference_output/channel_sweep/
--seed            42
```

### Shared Mechanics
- Models loaded once via `load_all_models()` from `tile_pipeline`
- Fixed seed → fixed denoising noise across all conditions within a tile
- Tile A's UNI embedding used for all conditions (isolates TME effect, not style)
- Diff computed as `|generated − baseline|`, mean across RGB, globally normalized per tile

---

### Experiment 1: Microenv Value Sweep

**Question:** Given cell types and states, how does O₂/glucose concentration change H&E appearance?

**Tile selection:** Top-1 tile per cell composition class from `tile_classes.json` → 3 tiles  
Select the tile with the highest axis1 purity score from the `neutral` axis2 group (oxygen and glucose both near median), so the sweep has room to go both up and down meaningfully. Fallback to any axis2 class if `neutral` is unpopulated for that axis1.

**Sweep:** Full 2D combination grid over both channels simultaneously:
- Scales: `[0, 0.25, 0.5, 0.75, 1.0]` for each of O₂ and glucose independently
- Each combination `(o2_scale, glucose_scale)`: replace O₂ map with `original_O₂ × o2_scale`, glucose map with `original_glucose × glucose_scale`
- All other channels remain at original values
- Baseline = `(1.0, 1.0)` (unmodified tile)

**Inference runs per tile:** 5 × 5 = 25 combinations  
**Total:** 3 tiles × 25 = 75 runs

**Figure layout** (one figure per tile, 5×5 grid):
```
                 glucose=0  0.25   0.5    0.75   1.0
O₂=0:          [H&E]      [H&E]  [H&E]  [H&E]  [H&E]
O₂=0.25:       [H&E]      [H&E]  [H&E]  [H&E]  [H&E]
O₂=0.5:        [H&E]      [H&E]  [H&E]  [H&E]  [H&E]
O₂=0.75:       [H&E]      [H&E]  [H&E]  [H&E]  [H&E]
O₂=1.0(base):  [H&E]      [H&E]  [H&E]  [H&E]  [H&E ★]
```
- ★ marks baseline cell `(1.0, 1.0)` with a border highlight
- Each cell also shows a small diff heatmap thumbnail vs baseline (bottom-left inset)
- Bottom strip: LPIPS heatmap (5×5 scalar grid, colormap) — shows dose-response surface
- Header: cell composition thumbnail + tile class label

---

### Experiment 2: Cell Type Relabeling

**Question:** Given cell states and microenv, how does cancer/immune/healthy identity change H&E appearance?

**Tile selection:** 3 near-pure tiles from `exp2_tiles` in `tile_classes.json`  
**Caveat:** Cell type channels are binary masks — relabeling is in-distribution (0/1 preserved).

**Relabeling mechanism** (example: cancer tile relabeled as immune):
```python
ctrl_hybrid = ctrl_full.clone()
ctrl_hybrid[idx_immune]  = ctrl_full[idx_cancer]   # copy cancer mask → immune channel
ctrl_hybrid[idx_cancer]  = 0                        # zero out cancer channel
ctrl_hybrid[idx_healthy] = ctrl_full[idx_healthy]   # unchanged
```

**Conditions per tile:** 3 (original + 2 relabelings)  
**Total:** 3 tiles × 3 = 9 runs  

**Figure layout** (3×3 grid, one figure for the full experiment):
```
                 original       → immune      → healthy
cancer tile:    [H&E] [diff]  [H&E] [diff]  [H&E] [diff]
immune tile:    [H&E] [diff]  [H&E] [diff]  [H&E] [diff]
healthy tile:   [H&E] [diff]  [H&E] [diff]  [H&E] [diff]
```
- Diagonal cells (original) have no diff panel
- Row header: cell type thumbnail showing dominant cell type
- Column header: target label

---

### Experiment 3: Cell State Relabeling

**Question:** Given cell types and microenv, how does prolif/nonprolif/dead state change H&E appearance?

**Tile selection:** 3 near-pure tiles from `exp3_tiles` in `tile_classes.json`  
**Same relabeling mechanism as Experiment 2**, applied to `cell_state_*` channels.

**Conditions per tile:** 3 (original + 2 relabelings)  
**Total:** 3 tiles × 3 = 9 runs  

**Figure layout:** Same 3×3 grid structure as Experiment 2, with prolif/nonprolif/dead labels.

---

## Implementation Notes

### Channel index lookup
```python
# Find channel indices for a group from config
def channel_indices_for_group(active_channels, group_name, channel_groups):
    group = next(g for g in channel_groups if g["name"] == group_name)
    return [active_channels.index(ch) for ch in group["channels"] if ch in active_channels]
```

### Reused from existing codebase
- `load_all_models`, `load_exp_channels`, `resolve_data_layout`, `find_latest_checkpoint_dir` — from `tools/stage3/tile_pipeline.py`
- `split_channels_to_groups` — from `tools/channel_group_utils.py`
- `build_exp_channel_header_rgb` — from `tools/stage3/ablation_vis_utils.py`
- `load_subset_condition_cache`, `list_cached_tile_ids` — from `tools/stage3/ablation_cache.py`

### Total inference budget
| Experiment | Tiles | Runs/tile | Total |
|---|---|---|---|
| 1 (microenv 2D grid) | 3 | 25 | 75 |
| 2 (cell type relabeling) | 3 | 3 | 9 |
| 3 (cell state relabeling) | 3 | 3 | 9 |
| **Total** | | | **93 runs** |
