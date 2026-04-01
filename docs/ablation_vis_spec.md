# Ablation Visualization — Spec & Implementation Plan

## Goal

A single static PNG figure showing all 14 channel ablation conditions for the
H&E generation model. Suitable for paper supplementary figures.

---

## File Structure

```
project/
├── data/
│   └── orion/
│       └── <patch_id>/
│           ├── *.png   # channel images — auto-discovered by keyword matching
│           └── ...     # filenames not fixed; script infers by name heuristics
│
├── inference_output/
│   └── test_combination/
│       ├── fid_scores.json          # TODO: schema TBD (see below)
│       └── <condition_key>/
│           └── <patch_id>/
│               └── *.png            # generated images; script picks one representative tile
│
└── ablation_vis.py                  # visualization script
```

### Channel image auto-discovery

Script scans `data/orion/<patch_id>/` and assigns each file to a channel by
keyword matching against the filename (case-insensitive):

| Channel | Keywords to match |
|---|---|
| Cell types | `type`, `celltype`, `cell_type` |
| Cell states | `state`, `cellstate`, `cell_state` |
| Vasculature | `vasc`, `vessel`, `vasculature` |
| Nutrient | `nutrient`, `oxygen`, `glucose` |

First match wins. Unmatched files are ignored. Script warns if any channel is
not found for the selected patch.

### fid_scores.json — TODO

Schema not yet decided. Placeholder in script; insert actual key format once known.
Likely options:

```jsonc
// Option A: "+"-joined canonical channel names
{ "cell_types+cell_states": 38.1, ... }

// Option B: bitmask strings (order: types, states, vasc, nutrient)
{ "1100": 38.1, ... }

// Option C: list of objects
[{ "channels": ["cell_types", "cell_states"], "fid": 38.1 }, ...]
```

Script will have a `parse_fid(raw, condition_channels) -> float` function
that is the single point to update once the schema is confirmed.

### Representative tile selection

For each condition folder, script globs `*.png` and picks:
1. Alphabetically first file, OR
2. A file whose name contains a configurable `TILE_ID` string (set at top of script)

This makes it easy to pin a specific tile for all conditions to ensure
a fair visual comparison.

---

## Condition Registry

14 conditions in fixed display order (sorted by cardinality, then lexicographic):

```
# 1-channel (4)
[cell_types]
[cell_states]
[vasculature]
[nutrient]

# 2-channel (6)
[cell_types, cell_states]
[cell_types, vasculature]
[cell_types, nutrient]
[cell_states, vasculature]
[cell_states, nutrient]
[vasculature, nutrient]

# 3-channel (4)
[cell_types, cell_states, vasculature]
[cell_types, cell_states, nutrient]
[cell_types, vasculature, nutrient]
[cell_states, vasculature, nutrient]
```

Canonical channel order for key construction: types < states < vasculature < nutrient.

---

## Figure Layout

```
┌──────────────────────────────────────────────────────────────────┐
│  Title: "Channel conditioning ablation"                          │
├────────────┬──────────────────────────────┬──────────────────────┤
│            │  SHARED CHANNEL HEADER       │                      │
│  (blank)   │  [img] [img] [img] [img]     │  FID (UNI-2h) ↓     │
│            │  Cell  Cell  Vasc  Nutr      │       ← better       │
│            │  types states ature ient     │                      │
├────────────┼──────────────────────────────┼──────────────────────┤
│            │  dotted guide lines          │                      │
│  1-ch      │  ●  ○  ○  ○  [gen image]    │  ████░░░  42.3      │
│            │  ○  ●  ○  ○  [gen image]    │  █████░░  51.7      │
│            │  ○  ○  ●  ○  [gen image]    │  ████░░░  48.9      │
│            │  ○  ○  ○  ●  [gen image]    │  █████░░  55.2      │
│  ─ ─ ─ ─  │  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─  │  ─ ─ ─ ─ ─ ─ ─ ─   │
│  2-ch      │  ●  ●  ○  ○  [gen image]    │  ███░░░░  38.1      │
│            │  ...                         │  ...                 │
│  ─ ─ ─ ─  │  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─  │  ─ ─ ─ ─ ─ ─ ─ ─   │
│  3-ch      │  ●  ●  ●  ○  [gen image]    │  ██░░░░░  29.3 ★   │
│            │  ...                         │  ...                 │
└────────────┴──────────────────────────────┴──────────────────────┘
```

### Column definitions

| Column | Width | Content |
|---|---|---|
| Group label | ~0.4 in | "1-ch" / "2-ch" / "3-ch", rotated 90°, centered per group |
| Ch1–Ch4 indicators | equal spacing | filled circle (active) or small empty ring (inactive) |
| Generated image | ~1 in square | representative tile thumbnail |
| FID bar | ~1.2 in | horizontal bar proportional to FID value |
| FID value | ~0.4 in | numeric label, right-aligned |

### Visual encoding

**Active dot:** filled circle, radius ~5pt, color = group color (see below)  
**Inactive ring:** open circle, radius ~3pt, gray stroke, no fill  
**Guide lines:** dotted vertical lines dropped from center of each channel image
through all 14 rows — visually anchor indicators to their channel column

**Group colors (Okabe-Ito colorblind-safe palette):**
- 1-ch: `#009E73` (bluish green)
- 2-ch: `#0072B2` (blue)
- 3-ch: `#D55E00` (vermillion)

These same colors fill the FID bars at ~35% alpha.

**Best FID row:** subtle `#fffbe6` background highlight + `★` marker after value.

**FID axis annotation:** small "← better" label below the FID column header,
so directionality is unambiguous in print.

---

## Implementation Plan

### Script: `ablation_vis.py`

**Dependencies:** `matplotlib`, `numpy`, `Pillow` (PIL), `pathlib`, `json`, `glob`

#### Phase 1 — Data loading

```
load_channel_images(orion_dir, patch_id)
  → dict: {channel_name: PIL.Image}
  Auto-discovers files via keyword matching (see table above).

load_condition_image(inference_dir, condition_key, tile_id=None)
  → PIL.Image or None
  Globs condition folder, picks representative tile.

parse_fid(fid_path, condition_channels)   # TODO: stub until schema confirmed
  → float or None
```

#### Phase 2 — Figure construction (matplotlib)

Use `matplotlib.figure.Figure` with a custom `GridSpec`:

```
GridSpec rows:
  Row 0:       title (spans all columns)
  Row 1:       channel header row
               — 4 image axes (one per channel)
               — 1 spacer axis (generated image column)
               — 1 FID header text axis
  Rows 2–15:   one row per condition (14 rows)
               — group label axis (shared per group via rowspan)
               — dot indicator axis (4 dots drawn with scatter/plot)
               — generated image axis (imshow)
               — FID bar axis (barh, shared y-axis across all 14 rows)
  Row 16:      "← better" annotation below FID column
```

All image axes: `ax.imshow()`, `ax.axis('off')`, `aspect='equal'`.  
Dot indicators: `ax.scatter()` with two sizes — large filled (active) vs small open (inactive).  
FID bars: single `ax.barh()` spanning all 14 condition rows, y-ticks aligned to row centers.  
Group separators: thin dashed `hlines` drawn at cardinality boundaries.

**Figure size:** 7 in × 10 in (scales to single-column at 3.5 in by halving font sizes).  
**DPI:** 300 for print output, 150 for quick preview.  
**Font:** matplotlib default (`DejaVu Sans`) at 7 pt body / 8 pt labels / 9 pt title.

#### Phase 3 — Export

```python
fig.savefig("figures/ablation_figure.png", dpi=300, bbox_inches='tight')
fig.savefig("figures/ablation_figure.pdf", bbox_inches='tight')  # vector for submission
```

---

## Configuration Block (top of script)

All tunable parameters in one place, nothing hardcoded elsewhere:

```python
# === PATHS ===
ORION_DIR      = "data/orion"
INFERENCE_DIR  = "inference_output/test_combination"
FID_JSON       = "inference_output/test_combination/fid_scores.json"
OUTPUT_DIR     = "figures"

# === TILE SELECTION ===
PATCH_ID       = None   # folder name inside ORION_DIR; None = first found
TILE_ID        = None   # filename substring to pin a specific tile; None = first found

# === FIGURE ===
FIG_WIDTH_IN   = 7.0
DPI            = 300
FONT_SIZE_PT   = 7

# === COLORS (Okabe-Ito) ===
COLOR_1CH      = "#009E73"
COLOR_2CH      = "#0072B2"
COLOR_3CH      = "#D55E00"
COLOR_INACTIVE = "#CCCCCC"
BAR_ALPHA      = 0.35
BEST_ROW_BG    = "#FFFBE6"
```

---

## Open Questions

1. **fid_scores.json schema** — `parse_fid()` is a stub until format confirmed.
2. **Multiple patches** — spec picks one patch per figure. If you want N example
   tiles per condition side-by-side, the generated-image column becomes a 1×N strip;
   this changes the GridSpec significantly. Decide before coding.
3. **Channel image colormap** — show raw masks as-is, or apply a colormap
   (e.g. viridis for nutrient map, binary for cell type/state masks)?
4. **FID error bars** — if FID was computed over multiple samples per condition,
   are std / CI values available to add as error bars on the bar chart?
