# Paper Figure Plan — PixCell (target: Nature Communications)

Organizing principle: **method-as-narrative** — the paper follows the 3-stage
pipeline (generate paired data → train → ablate), each main figure pairing a
stage schematic with its real results. Stage colors (blue / orange / green) from
the overview act as a paper-wide section color code.

Source roots: `figures/pngs_updated/{methods,concat,individual}/`.

## Graphical abstract
- `methods/overview_workflow` — the 4-box pipeline flowchart.
- **TODO:** export as PNG (or vector PDF/SVG); only a `.jpg` exists today and JPG
  smears line-art/text edges at print.

## Main figures

### Fig 1 — Approach & data (Stage 1)
- `methods/stage_1.jpg` (sub-step schematic)
- `methods/stage_1_results.png` (real H&E + CODEX + segmentation + O₂/glucose)
- `individual/cell_composition/cell_assignment.png` (marker z-scores validating
  cell-type / state labels)

### Fig 2 — Training, conditioning design & performance (Stage 2)  ✅ BUILT
File: `concat/fig2_architecture_performance.png`
Builder: `src/paper_figures/fig_combined_method_perf.py`
(`python -m src.paper_figures.fig_combined_method_perf`)
- **A** `methods/stage_2_svg.svg` — training/architecture schematic (rasterized via cairosvg)
- **B** training loss (top) + ΔLPIPS encoder-ablation bars + shared variant legend
  (from `fig_si_a1_a2_unified`: `_plot_loss_curves` + `_draw_section4_sensitivity`)
- **C** performance metric trade-offs (`build_metric_trends_figure`, = performance A)
- **D** channel-group effect heatmaps (`build_channel_effect_heatmaps_figure`, = performance C)
- **E** per-channel ablation tile strip (`individual/channel_ablation_paired_unpaired.png`)
- Notes: panels rendered at final pixel size (no rescaled text); A & B share
  height; E's dashed paired|unpaired divider auto-aligned to D's heatmap gutter.

### Fig 3 — UNI vs TME decomposition (information content)
- `concat/08_uni_tme_decomposition.png`
- `concat/uni_probe_overview.png`

### Fig 4 — Per-channel impact → channel-selection guide (Stage 3)
- `methods/stage_3.jpg` (sub-step schematic + quadrant guide)
- `concat/07d_t1_spatial_multi_encoder.png` — reframed as step 3.1 (how much UNI
  already encodes each channel). **TODO:** highlight UNI-2h, gray the other
  encoders, so it doesn't read as an encoder-choice figure.
- `individual/ablation_analysis/04_leave_one_out_impact.png`
- `concat/09b_channel_color_layout_impact.png`

## Supplementary
- `concat/ablation_grids_combined.png` — full ablation atlas
- `performance_paired_unpaired.png` **Panel B** — paired/unpaired ranking tables
- `individual/si_a1_a2/SI_A1_A2_section1_curves.png` — gradient-norm curves
  (training-loss half now promoted to Fig 2B)
- `individual/si_a1_a2/SI_A1_A2_section2_metrics.png` — full metric table
- `individual/si_a1_a2/SI_A1_A2_section3_tiles.png` — encoder-variant tile grid
- Full 4-encoder version of `07d` (encoder justification) if `07d` in Fig 4 is UNI-only

## Open styling to-dos (deferred — not blocking organization)
- `methods/stage_2_detail.png` / `stage_3_detail.png`: move off black background,
  fill/remove empty boxes (not currently used in the main plan).
- Unify schematic font → Nimbus Sans to match data panels. (Also: Nimbus Sans is
  not installed in the render env → panels currently fall back to DejaVu Sans.)
- Propagate stage colors (blue/orange/green) into the data figures.
- **Deliberately NOT fixed (user decision):** benchmark-label overlap on the data
  in Fig 2 panel C / standalone performance A.
