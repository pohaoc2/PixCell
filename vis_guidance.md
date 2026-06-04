# Visualization Guidance (Paper Figures)

High-level requirements for any figure under `figures/` or `src/paper_figures/`.
These are principles, not recipes — pick concrete values (colors, linewidths,
exact point sizes) in code to satisfy them; keep that detail out of this file.

## Layout

- Compact: minimal whitespace, panels close together, no large empty bands between panels.
- Square plotting areas when both axes share units (e.g. scatter with equal ranges, correlation heatmaps).
- Equal-status panels share size: panels in the same row share height, and the **same kind of plotting area is the same physical size across panels** (e.g. a scatter square and a heatmap square should match).
- Panel-size relationships hold **by construction** (e.g. the width of a stacked panel equals the combined width of the panels above it), never by stretching a rendered panel.
- **Never distort a rendered image.** To make a panel narrower/wider/taller, **re-render it** at the target size (so its data area and text re-flow), or scale it **proportionally** (same factor on both axes). Never change only one dimension of an already-rendered raster (e.g. cropping or squashing a panel's width to fit a column) — that warps its aspect ratio and its glyphs. This applies to data panels, schematics, and tile images alike.
- **Align by data region, not figure box.** When panels sit beside or below each other (including a schematic next to data panels), line them up on their **plotting areas** — the axes box / bars / web — *together with the axis and tick labels you want to read across panels* — not on their outer figure bounding boxes. A figure box carries margins whose size depends on label lengths, legends, and titles, so aligning boxes leaves the actual data (and its `x`/`y` labels) misaligned. Measure the data-region extent at render time (its top/bottom or left/right, extended to include the shared axis/tick labels) and size and place each panel to that. (Reference: `src/paper_figures/fig_combined_per_channel.py` — the schematic's green box is scaled and positioned so it spans exactly B's bar-area top to C's bar-area bottom, and F's plot square spans D's bar top to E's bar bottom.)
- All four axis spines visible and black; no half-open axes.

### Image grids / tile showcases

- For a grid of image tiles (e.g. an ablation strip), the gap between adjacent tiles must be **equal horizontally and vertically** (`wspace == hspace` *visually*). A reader should not be able to tell the row gap from the column gap.
- Do **not** rely on GridSpec `wspace`/`hspace` numbers to achieve this: they are fractions of the *average cell width/height*, so equal numbers give unequal physical gaps, and the gaps drift whenever the figure width or height is tuned. Instead lay the grid out in **absolute inches** — fixed square cell size + a single gap constant used for both axes — and derive `figsize` from those, so equal gaps hold by construction at any size. (Reference: `src/paper_figures/fig_channel_ablation_strip.py`, the `CELL_IN` / `GAP_IN` layout.)
- Tiles carry no solid frame unless a border encodes meaning (e.g. a dashed border marking a reference/ground-truth tile).
- **Header labels / dot indicators above tiles: position them in absolute inches, not axis fractions.** A column label, channel-name, or dot row placed at a *fraction* of a fixed-height header axis barely moves when you change that fraction (the band is only a fraction of an inch tall) and its physical distance to the tiles below shifts whenever figure height changes. Cause: a fraction times a small/variable height is not a stable offset. Fix: draw header glyphs on one figure-wide overlay axis whose coordinates are inches (`set_xlim(0, fig_w)`, `set_ylim(0, fig_h)`), and place them at explicit inch offsets above the first tile row (e.g. dots `DOT_OFF_IN` above the row, names `NAME_OFF_IN` above the dots). "Move the label closer" then becomes a one-number inch edit that holds at any figure size. (Reference: `src/paper_figures/fig_channel_ablation_strip.py`, `_draw_header` + the `DOT_OFF_IN` / `NAME_OFF_IN` overlay.)

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
