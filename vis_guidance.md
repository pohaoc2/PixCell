# Visualization Guidance (Paper Figures)

High-level rules for any figure produced under `figures/` or `src/paper_figures/`.

## Layout

- Compact: prefer square panels ~3.5x3.5 in. Multi-panel use shared axes; no whitespace padding.
- Constrained_layout or tight_layout. `bbox_inches="tight"` on save.
- All 4 spines visible, black, linewidth ~0.8. No half-open axes.
- Aspect ratio: square (`set_aspect("equal")`) only when both axes share units; otherwise free.

## Text

- No text overlaps. Use `adjustText.adjust_text` with leader lines `#aaaaaa lw=0.5` for scatter labels.
- Label color: black. Group/category color is encoded by marker, not by label color.
- Font: Nimbus Sans across all figures (xlabel, ylabel, ticks, legend, in-axes text).
- Sizes: xlabel/ylabel 9, tick 8, in-axes label 6.5, legend 7, title 10. Stay within ±0.5 of these.
- Axis labels disambiguate units and metric (e.g. "Generative impact ΔE (LOO)", not "Generative impact").

## Markers and legend

- Hollow markers: white face, colored edge (lw 1.2), markersize 5–6.
- One shape per category, kept consistent across figures:
  - appearance / cell_types: circle `o`
  - morphology / cell_state: square `s`
  - cell composition / vasculature: triangle `^`
  - microenv: diamond `D`
- Color palette consistent across figures (define once per module, reuse).
- Legend below the plot: `loc="upper center", bbox_to_anchor=(0.5, -0.15..-0.18), ncol=len(handles), frameon=False`.
- No legend title unless disambiguation requires it.

## Color

- Group colors are categorical, distinguishable in CVD palettes. Avoid red/green pairings as the only signal — shape carries the distinction too.
- Quadrant / background fills: alpha 0.06 max. Never compete with foreground markers.
- Quadrant/annotation text: black, not the fill color.

## Error bars

- `elinewidth=0.7, capsize=2`, color = marker edge color.
- Always state what the bar represents in caption (std, sem, CI).

## Thresholds and reference lines

- Diagonal y=x: black dashed, lw 0.8.
- Quadrant splits: gray dotted `#888 lw 0.6`.
- Threshold values justified in caption (perceptual JND, median, etc), not arbitrary.

## File outputs

- PNG at dpi 200–300, white facecolor.
- Filename matches the figure number used in paper: `09_channel_utility.png`, `SI_09_*.png`.
- Regenerable from a single function: `save_<name>_figure()`.

---

When in doubt, mimic `src/a4_uni_probe/figures.py::render_pngs_updated_probe_delta`
(output: `figures/pngs_updated/a4_uni_probe/probe_delta_r2.png`).
