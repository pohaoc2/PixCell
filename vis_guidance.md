# Visualization Guidance (Paper Figures)

High-level requirements for any figure under `figures/` or `src/paper_figures/`.
These are principles, not recipes — pick concrete values (colors, linewidths,
exact point sizes) in code to satisfy them; keep that detail out of this file.

## Layout

- Compact: minimal whitespace, panels close together, no large empty bands between panels.
- Square plotting areas when both axes share units (e.g. scatter with equal ranges, correlation heatmaps).
- Equal-status panels share size: panels in the same row share height, and the **same kind of plotting area is the same physical size across panels** (e.g. a scatter square and a heatmap square should match).
- Panel-size relationships hold **by construction** (e.g. the width of a stacked panel equals the combined width of the panels above it), never by stretching a rendered panel.
- All four axis spines visible and black; no half-open axes.

## Panel labels

- Bold letters (A, B, C, …) on every panel.
- **Aligned**: labels in the same row share a top baseline; labels in the same column share a left edge.
- The same size as one another, rendered at the figure's resolution (not rescaled).

## Text

- **No text overlaps — anywhere.** Includes labels overlapping each other, text cut by a panel border, text crossing into a neighbouring panel, and colorbar/axis labels running off the figure edge. Check the seams and all four borders, not just the plot interior.
- **Large enough to read** comfortably at print size. Prefer a compact layout with generous text over a sparse layout with tiny text.
- One font family across the whole figure.
- Use leader lines for crowded scatter labels.
- Axis labels name the metric and units, not just the quantity.
- Tick labels on the x and y axes use the **same number of decimal places** — and a shared axis (the same quantity plotted in multiple panels) uses the same number of decimals in every panel.

## Cross-panel / cross-figure consistency (REQUIRED)

- The same kind of text is the **same physical size in every panel and every figure** (axis labels match axis labels, ticks match ticks, …).
- Achieve this with a single source of truth for sizes and one resolution per figure. **Do not rescale an already-rendered panel** to make it fit — rescaling changes that panel's text size relative to the others. Size panels to fit at render time instead.

## Markers, legend, color

- Hollow markers (white face, colored edge); one shape per category, kept consistent across all figures.
- Color palette and marker shapes defined once in code and reused; categories distinguishable in colorblind-safe palettes, with shape (not color alone) carrying the distinction.
- Legend below the plot, no frame, no title unless needed to disambiguate.
- Background/quadrant fills subtle enough never to compete with foreground markers; annotation text in black.

## Reference lines & error bars

- Reference lines (y = x diagonal, thresholds) visually distinct from the data and from each other.
- Error bars thin and unobtrusive; the caption states what they represent (std, sem, CI) and justifies any threshold.

## File outputs

- High-resolution PNG, white background.
- Filename matches the figure number used in the paper.
- Regenerable from a single function.

## Process

- Open the rendered image and look at it — never declare a figure done from code alone.
- Iterate render → inspect → fix until every rule above holds.

---

Reference implementation: the combined overview in
`src/a4_uni_probe/figures.py` (`render_pngs_updated_combined_abc` plus its
`FONT_*` / `PANEL_*` constants).
