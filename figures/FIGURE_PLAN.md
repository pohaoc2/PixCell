# Paper Figure Plan — PixCell (target: Nature Communications)

Organizing principle: **method-as-narrative** — the paper follows the 3-stage
pipeline (generate paired data → train → ablate), each main figure pairing a
stage schematic with its real results. Stage colors (blue / orange / green) from
the overview act as a paper-wide section color code.

Source roots: `figures/pngs_updated/{methods,concat,individual}/`.

Reorganized 2026-06-04: the paper now ships four `fig1`–`fig4` composites as the
main figures and three SI items (S1–S3). The former standalone panels (`08`,
`uni_probe_overview`, `07d`, `09b`, the performance composite) are embedded in
the composites or split, with one panel each surviving in the SI. Captions live
in `figures.md`; print-preview composites
(`concat/_all_figures_preview.png` main, `concat/_all_SI_figures.png` SI) are
rebuilt with `python -m src.paper_figures.build_concat_preview`.

## Graphical abstract
- `methods/overview_workflow.png` — the 4-box pipeline flowchart (RGBA, ~12 in
  wide). Sits at the top of the main preview (`concat/_all_figures_preview.png`),
  unnumbered, ahead of Fig 1. Caption in `figures.md`.
- **TODO:** for final submission consider a vector (PDF/SVG) export; the PNG is
  fine for the preview but line-art is crisper as vector at print.

## Main figures

### Fig 1 — Approach & data (Stage 1)  ✅ BUILT
File: `concat/fig1_approach_data.png`
Builder: `src/paper_figures/fig_combined_stage1.py`
- **A** stage-1 sub-step schematic (matched H&E/MX → CellViT → markers → k-means
  → PDE nutrients)
- **B(i)** per-cell-type morphology (nuclear area, circularity; mean ± SD) and
  **B(ii)** marker z-score heatmap validating cancer/immune/healthy assignments
- **C** real-data montage: WSI → registered H&E + CODEX channels → segmentation
  → PDE-estimated O₂/glucose fields

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

### Fig 3 — UNI vs TME decomposition (information content)  ✅ BUILT
File: `concat/fig3_uni_decomposition_v2.png`
Builder: `src/paper_figures/fig_combined_uni_decomp_v2.py`
- Left **A/B/C** = UNI/TME decomposition panels (tile + 4-condition metrics +
  two-way decomposition heatmap)
- Right **D/E/F** = UNI-vs-TME decodability scatter + probe specificity matrix +
  edit sweeps in the side-by-side v2 layout
- Sources (now embedded, archived in `figures.md`): `concat/08_uni_tme_decomposition.png`,
  `concat/uni_probe_overview.png`

### Fig 4 — Per-channel impact → channel-selection guide (Stage 3)  ✅ BUILT
File: `concat/fig4_per_channel_impact.png` (v1; `fig4_*_v2.png` is the radar D/E
alternative, **not** in the paper)
Builder: `src/paper_figures/fig_combined_per_channel.py`
- **A** stage-3 sub-step schematic (probe → leave-one-out → per-channel impact)
- **B** within-tile decodability boxplots across 4 encoders (= `07d` reframed as
  step 3.1). **TODO:** highlight UNI-2h, gray the other encoders.
- **C** individual MX-marker decodability bars (UNI-2h)
- **D/E** per-channel color impact (ΔE) + layout impact (ΔPQ) — **placeholder
  bars**, pending the leave-one-out metric pass (`04_leave_one_out_impact`)
- **F** decodability-vs-impact quadrant scatters (= `09b_channel_color_layout_impact.png`)

## Supplementary  (S1–S3; preview = `concat/_all_SI_figures.png`)
- **S1** `concat/si_performance_ranking.png` — paired/unpaired ranking tables
  (was performance **Panel B**; re-rendered standalone by
  `src/paper_figures/build_si_assets.py`)
- **S2** `concat/ablation_grids_combined.png` — full qualitative ablation atlas
- **S3** `concat/si_a1a2_qualitative_tiles.png` — conditioning-architecture
  qualitative tiles (was SI_A1_A2 **Panel C**; copied from
  `individual/si_a1_a2/SI_A1_A2_section3_tiles.png` by `build_si_assets.py`)
- Supporting (not in the SI preview): `individual/si_a1_a2/SI_A1_A2_section2_metrics.png`
  (full metric table referenced by S3), `SI_A1_A2_section1_curves.png` (gradient-norm
  curves; training-loss half promoted to Fig 2B), full 4-encoder `07d`
  (encoder justification, if Fig 4B is reframed UNI-first)

## Open styling to-dos (deferred — not blocking organization)
- `methods/stage_2_detail.png` / `stage_3_detail.png`: move off black background,
  fill/remove empty boxes (not currently used in the main plan).
- Unify schematic font → Nimbus Sans to match data panels. (Also: Nimbus Sans is
  not installed in the render env → panels currently fall back to DejaVu Sans.)
- Propagate stage colors (blue/orange/green) into the data figures.
- **Deliberately NOT fixed (user decision):** benchmark-label overlap on the data
  in Fig 2 panel C / standalone performance A.
