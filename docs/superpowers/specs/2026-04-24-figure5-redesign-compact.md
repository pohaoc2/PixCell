---
title: Figure 5 Compact Redesign — UNI vs TME Decomposition
date: 2026-04-24
status: approved
---

# Figure 5 Compact Redesign — UNI vs TME Decomposition

## Goal

Redesign `figures/pngs/08_uni_tme_decomposition.png` to be maximally compact and consistent with Nature Communications figure standards. The existing 4-panel (A/B/C/D) layout is replaced by a 3-panel (A/B/C) L-shape layout.

## Overall Layout

- **Structure:** L-shape — Panel A occupies the full left column; Panels B and C stack vertically in the right column.
- **Figure size:** 183 mm × ~100 mm (full Nat Comm page width, compact height).
- **Resolution:** 300 DPI.
- **Font:** Arial, 7 pt for all axis labels and annotations; 8 pt bold for panel letters.
- **Spines:** left + bottom only (no top or right spines on any subplot).
- **Panel titles:** none — metric name + direction arrow (↑/↓) inside the subplot title only; no descriptive subtitle text on any panel.
- **Column split:** left column ~48% width (images), right column ~52% width (metrics + heatmap).

## Panel A — Left Column, Full Height

### Layout
3 rows × 2 columns image grid.

| Row | Col 1 | Col 2 |
|-----|-------|-------|
| 1   | Real H&E (reference) | TME layout (cell_masks thumbnail) |
| 2   | UNI+TME generated | UNI only generated |
| 3   | TME only generated | Neither generated |

### Styling
- No standalone legend column.
- Each generated image cell (rows 2–3) has small ●/○ UNI+TME dot indicators at the **bottom-right corner** of the cell.
- **One cell only** (UNI+TME, row 2 col 1) shows "UNI" and "TME" text beside its dots. All other generated cells show dots only.
- Real H&E cell: green tint border, label "Real H&E".
- TME thumbnail cell: warm tint border, label "TME layout"; rendered as grayscale normalised cell_masks channel.
- Cell borders: 0.8 pt, colour-coded (green for reference, warm for TME, neutral for generated).
- No row/column axis labels on the image grid.
- Panel letter **A** at top-left, 8 pt bold.

## Panel B — Right Column, Top

### Content
Five dot+errorbar subplots side by side: **FUD ↓ · LPIPS ↓ · PQ ↑ · DICE ↑ · Style HED ↓**.

### Dot+errorbar style
- Each subplot has 4 x-positions (one per mode: UNI+TME, UNI only, TME only, Neither).
- Data dot: hollow circle (○), 4.5 pt diameter, 1.5 pt edge, colour `INK` (#2c2c2c).
- Whisker: 95% CI, 1.5 pt line, cap width matches 2× dot radius.
- **No connecting lines** between dots.
- Y-axis: tight limits padded 12% beyond CI extremes; left spine only; 7 pt tick labels.
- X-axis: no tick labels, no x-axis title — condition identity is encoded entirely by the shared dot-key strip.
- Subplot title: metric name + direction arrow, 8 pt, centred above plot area.
- No top/right spines.
- Light horizontal grid lines at y-tick positions, colour `#e8e8e8`, 0.7 pt.

### Shared dot-key strip
- A single strip below all 5 subplots, spanning the full width of Panel B.
- Two rows of dots (row 1 = UNI state, row 2 = TME state), one column per mode position.
- Filled ● = component active; hollow ○ = component off.
- "UNI" and "TME" row labels at left, 7 pt.
- Thin dashed separator line above the strip.
- Strip height: ~14 pt total (two dot rows + padding).

### Panel letter
**B** at top-left of the subplot group, 8 pt bold.

## Panel C — Right Column, Bottom

### Content
Oriented-effects heatmap: 4 rows (UNI effect, TME effect, interaction, neither) × 5 columns (FUD, LPIPS, PQ, DICE, Style HED).

### Styling
- Colormap: `RdBu_r`, symmetric around 0, vmax = max absolute finite value.
- Colorbar: right side of heatmap; **height matched exactly to heatmap matrix height**; width ~8 pt; tick labels 7 pt.
- Colorbar label: "Higher-is-better Δ", 7 pt.
- Cell value annotations: formatted `:.2g`, 7 pt, colour `INK`, centred in each cell.
- X-axis tick labels: metric names, 35° rotation, right-aligned, 8 pt.
- Y-axis tick labels: effect row names, 8 pt.
- No spines around heatmap axes (imshow handles borders).
- Panel letter **C** at top-left, 8 pt bold.

## Dropped Panels

- **Panel D** (style/layout scatter) dropped from this version. Can be added later as a supplementary panel or replaced with per-tile distribution strips or additivity check scatter (see brainstorm session 2026-04-24).

## Implementation Notes

- All changes are in `src/paper_figures/fig_uni_tme_decomposition.py`.
- The `build_uni_tme_decomposition_figure` function signature is unchanged; only the internal rendering functions change.
- Remove `_render_panel_d` and the old `MODE_GRID` / legend-column logic. Keep `MODE_USE_UNI` / `MODE_USE_TME` dicts — they are still used by the shared dot-key strip in `_render_panel_b`.
- The `MODE_GRID` constant and old 2×4 grid logic in `_render_panel_a` are replaced with the 3×2 grid logic.
- The shared dot-key strip in Panel B replaces the per-subplot dot matrix rows in `_render_panel_b`.
- `DISPLAY_METRICS` order determines column order in both B and C — keep consistent.
- Figure dimensions: `figsize=(7.2, 3.95)` inches (183 mm × 100 mm at 300 DPI).
- Output path unchanged: `figures/pngs/08_uni_tme_decomposition.png`.
