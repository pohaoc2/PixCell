# Figure Color Hex Reference

Inventory of hex colors used across PixCell figure code. Grouped by purpose. Canonical palette lives in `tools/color_constants.py`; per-figure overrides listed below.

---

## 1. Canonical palette (`tools/color_constants.py`)

### Cell-type RGBA (from `CELL_TYPE_COLORS`, 0–255)
| Class   | RGBA              | Hex       |
|---------|-------------------|-----------|
| cancer  | (220, 50, 50, 200)| `#dc3232` |
| immune  | (50, 100, 220, 200)| `#3264dc`|
| healthy | (50, 180, 50, 200)| `#32b432` |
| other   | (150,150,150,120) | `#969696` |

### Cell-state RGBA (from `CELL_STATE_COLORS`)
| Class         | RGBA               | Hex       |
|---------------|--------------------|-----------|
| proliferative | (230, 50, 180, 200)| `#e632b4` |
| nonprolif     | (240,140, 30, 200) | `#f08c1e` |
| dead          | (110, 40,160, 200) | `#6e28a0` |
| other         | (160,160,160,120)  | `#a0a0a0` |

### Continuous channel gradients
- Oxygen: black `#000000` → cyan `#00ffff`
- Glucose: black `#000000` → yellow `#fff31e`
- Vasculature: matplotlib `Reds`
- Cell masks: matplotlib `gray`

### Section background / text (panel role coding)
| Role      | BG        | Text      |
|-----------|-----------|-----------|
| input     | `#dce8f5` | `#1a3a5c` |
| output    | `#d5f0d5` | `#1a6b1a` |
| reference | `#fde8cc` | `#8b4500` |
| style_ref | `#ece0f0` | `#4a1a6b` |
| analysis  | `#f0f0f0` | `#333333` |

---

## 2. Okabe-Ito qualitative set (colorblind-safe)

Used in tradeoff scatter, visibility maps, ablation grids, dataset metrics.

| Name        | Hex       | Where |
|-------------|-----------|-------|
| Blue        | `#0072b2` | a0 visibility/tradeoff, dataset metrics, ablation_grid (cfg 2) |
| Vermillion  | `#d55e00` | a0 visibility/tradeoff, dataset metrics, ablation_grid (cfg 3) |
| Bluish green| `#009e73` | a0 tradeoff, dataset metrics, ablation_grid (cfg 1) |
| Purple      | `#9b59b6` | dataset metrics, ablation_grid (cfg 4), LOO baseline |
| Dark blue   | `#004c7f` | a0 visibility secondary |
| Dark orange | `#8c3a00` | a0 visibility secondary |

---

## 3. Channel-group colors (paper figures)

`fig_channel_utility.py`, `fig_marker_utility.py`, `fig_combinatorial_grammar_panels/_variance_bars.py`:

| Group       | Hex       |
|-------------|-----------|
| cell_types  | `#2a5db0` |
| cell_state  | `#b04a2a` |
| vasculature | `#2a8a4a` |
| microenv    | `#c2a83e` |
| anchor      | `#6b7280` |
| interactions| `#b04a2a` |
| resid       | `#cccccc` |

Variance-bar variants reuse: `state #2a5db0`, `o2 #2a8a4a`, `gluc #c2a83e`.

`tools/ablation_report/shared.py` muted channel-group set:

| Group       | Hex       |
|-------------|-----------|
| cell_types  | `#7c8aa5` |
| cell_state  | `#b9795f` |
| vasculature | `#6e9c92` |
| microenv    | `#9175a6` |

---

## 4. Encoder colors (`fig_inverse_decoding.py`, `fig_t1_spatial_multi_encoder.py`)

| Encoder    | Hex       |
|------------|-----------|
| UNI-2h     | `#f98866` |
| Virchow2   | `#9b59b6` |
| CTransPath | `#bed7d8` |
| REMEDIS    | `#5a5a5a` |
| ResNet-50  | `#a9a9a9` |
| fallback   | `#888888` |

---

## 5. Quadrant labels (`fig_channel_utility.py`)

| Quadrant            | Hex       | Label        |
|---------------------|-----------|--------------|
| high R², high ΔE    | `#f0c060` | Redundant    |
| low R², high ΔE     | `#d05050` | Critical     |
| low R², low ΔE      | `#a0a0a0` | Skip         |
| high R², low ΔE     | `#70b070` | MX optional  |

---

## 6. Probe / null comparison (`src/a4_uni_probe/figures.py`)

| Role               | Hex       |
|--------------------|-----------|
| UNI bars           | `#2b6cb0` |
| O₂/Glc bars        | `#dd6b20` |
| Targeted slope     | `#2f855a` |
| Random slope       | `#a0aec0` |
| Targeted null      | `#c53030` |
| Random null        | `#718096` |
| Full UNI null      | `#2b6cb0` |
| Global             | `#1a202c` |
| Nucleus            | `#c53030` |
| Stroma             | `#2b6cb0` |
| Appearance marker  | `#dd6b20` (`o`) |
| Morphology marker  | `#4a90d9` (`s`) |
| Cell-composition   | `#52b788` (`^`) |
| Quadrant bg (input)| `#eaf0f8` |
| Quadrant bg (output)| `#f8efe6`|

---

## 7. SI-A* ablations (`fig_si_a1_a2_unified.py`, `fig_si_a3_zero_init.py`)

| Variant                       | Hex       |
|-------------------------------|-----------|
| Grouped TME only (production) | `#2e7d32` |
| Concat TME encoder (a1)       | `#1565c0` |
| Per-channel encoders (a1)     | `#e65100` |
| Additive bypass + TME (a2)    | `#7b1fa2` |
| Vanilla PixCell ControlNet    | `#000000` |
| Instability marker            | `#c62828` |
| zero_init=True                | `#2b6cb0` |
| zero_init=False               | `#c53030` |

---

## 8. Stage-3 / LOO heatmaps & misc (`tools/vis/leave_one_out_diff.py`)

- Hot4 colormap stops: `#000000 → #ff4400 → #ffff00 → #ffffff`
- SSIM-loss colormap: `#000000 → #3b528b → #5ec962 → #fde725` (viridis-ish)
- LOO highlights:
  - `cell_state` `#ff6644`
  - `microenv`   `#ddaa00`
  - Reference   `#000000`
  - Baseline    `#9b59b6`
  - SSIM inset teal `#00ccaa`
  - SSIM neutral `#555555`
- Mask outline `#00d4ff`
- In-bar `#e76f51`, out-bar `#457b9d`

`tools/vis/visualize_tme_cnn_features.py` diverging custom map stops: `#1f4ed8 → #285fdf → #071329 → #000000 → #2c0708 → #b2172b → #e34a33`.

---

## 9. Stage-4 qualitative (`tools/stage4/figures.py`)

Brewer Set1 subset: `#e41a1c #377eb8 #4daf4a #984ea3 #ff7f00 #a65628`.

---

## 10. Ablation-report extras (`tools/ablation_report/`)

Okabe variants:
| Name   | Hex       |
|--------|-----------|
| Blue   | `#4c78a8` |
| Orange | `#e28e2b` |
| Green  | `#5c8f5b` |
| Purple | `#8d6a9f` |
| Red    | `#b22222` |
| Teal   | `#5b8f96` |
| Gray   | `#565656` |

Paper-style neutrals: ink `#000000`, soft-grid `#d7d6d2`, spine `#d9d5ca`, axis bg `#f6f4ef`.

HTML report CSS tokens: bg `#f7f7f5`, paper `#ffffff`, ink `#1b1b1b`, muted `#686760`, line `#dddcd6`, code-bg `#f3f3ef`, code-fg `#3f3c35`, card-border `#e6e4dd`.

---

## 11. Grid / neutral helpers (recurring)

| Use            | Hex       |
|----------------|-----------|
| Light grid     | `#e0e0e0`, `#d9d9d9`, `#d4d4d4`, `#f2f0ea` |
| Heavy grid line| `#bcbcbc`, `#bebebe`, `#d0d0d0` |
| Axhline / spine| `#4a5568`, `#555555`, `#333333` |
| Annotation arrow| `#aaaaaa` |
| Highlight bg   | `#fffbe6` (best row) |
| Bar fill (perf)| `#111111`, `#222222`, `#1b1b1b` |
| Spine grey set | `#8a8a8a`, `#9a9a9a`, `#6a6a6a` (combinatorial grammar panels) |

---

## 12. Dataset-metric panel (`tools/render_dataset_metrics.py`)

| Token        | Hex       |
|--------------|-----------|
| INK / AXIS   | `#000000` |
| SOFT         | `#aaaaaa` |
| GRID         | `#ece8e0` |
| DOT_ACTIVE   | `#4b5563` |
| DOT_INACTIVE | `#cfc7ba` |
| CARD_COLORS  | `#009e73 #0072b2 #d55e00 #9b59b6` |

---

## 13. Channel-sweep figure (`tools/stage3/channel_sweep_figures.py`)

- Sweep heatmap stops: `#000000 → #ff4400 → #ffff00 → #ffffff`
- Baseline border: `#9b59b6`

---

## 14. T1 spatial decodability (`fig_t1_spatial_decodability.py`)

| Use         | Hex       |
|-------------|-----------|
| Bar face    | `#f98866` |
| Error bar   | `#333333` |
| Zero line   | `#555555` |
| Highlight   | `#b22222` |
| Grid faint  | `#e0e0e0` |

---

## Notes

- Cell-type/state RGBA stored as 0–255 tuples in `tools/color_constants.py`; hex listed above drops alpha.
- Okabe-Ito set repeated across several modules — unify via `tools/ablation_report/shared.py` if consolidating.
- `fig_marker_utility.py` and `fig_channel_utility.py` share group colors; treat as single channel-group key.
