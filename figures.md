# Figures

Caption registry for the publication figures in `figures/pngs_updated/concat/`.
Each section is keyed by the rendered PNG. The "huge" print-preview composite
(`figures/pngs_updated/concat/_all_figures_preview.png`) pulls its captions from
this file.

> Removed 2026-06-01: the `methods` and `cell_summary_figure` entries — their
> `figures_ec2/pngs/` paths no longer exist anywhere in the repo. If those
> figures are still in the paper, re-add them here with current paths.

## performance_paired_unpaired (`figures/pngs_updated/concat/performance_paired_unpaired.png`)

**Outline name:** Channel-group ablation results

### Figure caption

**Figure X. Channel-group ablations characterize ControlNet performance under paired and unpaired inference.** **(A)** Per-tile mean ± SD (standard deviation across tiles) for each of the 32 channel-group on/off combinations across five metrics: FUD (lower is better), LPIPS (lower is better), PQ (higher is better), DICE (higher is better), and HED (lower is better). Triangles indicate paired inference; squares indicate unpaired inference. The shaded band shows the benchmark mean ± SD. **(B)** Top-3 and bottom-3 ranked conditions per metric for paired (upper) and unpaired (lower). The dot column encodes active channel groups in the order CT, CS, VA, NU, UNI; filled = included. CT (Cell types) groups the healthy / cancer / immune density maps; CS (Cell state) groups the proliferative / nonproliferative / dead density maps; VA (Vasculature) is the vasculature density channel; NU (Nutrient) groups the oxygen and glucose channels; UNI is the UNI-2h H&E reference embedding (paired-only). **(C)** Per-group effect-size heatmaps showing mean ± SD change in each metric attributable to including that group (paired top, unpaired bottom), color-scaled from blue (positive) to red (negative); rows correspond to the CT / CS / VA / NU groups defined above.

### Key takeaway

- Paired (triangle) beats unpaired (square) across all 5 metrics. Reference H&E pins style; absent it, FUD/LPIPS/HED drift up.
- Cell-state and Nutrient co-dominate nuclear-segmentation metrics (ΔPQ, ΔDICE) in both regimes — tied magnitudes, not cell-state alone.
- FUD direction flips for the two strongest groups between regimes: with reference H&E, dense conditioning lowers realism distance; without it, the same channels over-constrain the generator and FUD blows up.
- Cell-types are small everywhere and Vasculature ≈ 0 on every metric — the vasculature channel is currently near-dead weight.
- LPIPS and HED move little for any single group (|Δ| ≤ 0.04); ablation differentiation lives mostly in FUD, PQ, DICE.

## uni_probe_overview (`figures/pngs_updated/concat/uni_probe_overview.png`)

**Outline name:** UNI embeddings linearly encode interpretable H&E morphology

### Figure caption

**Figure X. Frozen UNI-2h embeddings linearly encode interpretable H&E morphology and appearance attributes, and edits along the learned probe directions produce specific, visible changes in generated histology.** **(A)** Tile-level decodability of each H&E attribute from frozen UNI features (y-axis) versus from the spatial TME channels (x-axis), reported as held-out R². Each point is one attribute, colored and shaped by family: appearance (orange circle), morphology (blue square), and cell composition (green triangle). The shaded diagonal band marks attributes equally recoverable from either source. **(B)** Specificity matrix for UNI probe edits: each column edits the UNI embedding along one attribute's probe direction and each row reports the resulting Pearson correlation with the measured change in a tile-level morphology metric, color-scaled from −1 (red) to +1 (blue). A strong diagonal indicates that editing one attribute moves that attribute's metric and not the others. **(C)** Targeted edit sweeps for six attributes (columns: Eccentricity, Nuclear area, Nuclei density, E contrast, H contrast, H energy). The top row (gray dashed border) is the real reference H&E; the lower rows are H&E generated after shifting the UNI embedding along that attribute's probe direction by α = −1, 0, +1. Each column shows three example tiles.

### Key takeaway

- UNI features and TME channels are near-interchangeable for many attributes (A) — UNI already carries the appearance/morphology signal the TME channels encode.
- Probe edits are specific (B): the response matrix is diagonally dominant, so each direction isolates one morphology axis.
- The edit sweeps (C) make the directions visually interpretable — α = −1/+1 produces monotonic, attribute-consistent changes in the generated tissue relative to the reference.

## 08_uni_tme_decomposition (`figures/pngs_updated/concat/08_uni_tme_decomposition.png`)

**Outline name:** UNI–TME factorial decomposition

### Figure caption

**Figure X. UNI conditioning and TME conditioning contribute near-orthogonally to ControlNet output, separating realism from nuclear geometry.** The four conditions toggle UNI conditioning and the full TME channel stack independently: full (UNI on, TME on), UNI-only (UNI on, TME off), TME-only (UNI off, TME on), and unconditional (UNI off, TME off). **(A)** A representative tile. The top row shows the inputs to the model: the real H&E used as the style reference (left) and a visualization of the TME conditioning (right), which aggregates per-cell location, type, state, vasculature, and nutrient channels. The lower rows show generated H&E (left) for two of the four UNI/TME settings; the dot pair beside each row indicates which of UNI and TME is active (filled = on). **(B)** Per-tile mean ± SD for each of the four conditions across five metrics: FUD (lower is better), LPIPS (lower is better), PQ (higher is better), DICE (higher is better), and HED (lower is better). The 2-dot label under each marker encodes UNI/TME state. **(C)** Two-way decomposition of each metric across the four conditioning settings, reported as mean ± SD across tiles. The three rows are: the UNI effect (full − tme_only, the marginal gain from adding UNI on top of TME); the TME effect (full − uni_only, the marginal gain from adding TME on top of UNI); and the UNI × TME interaction (full − uni_only − tme_only + unconditional), where positive values indicate synergy and negative values indicate redundancy or competition. Cells are color-scaled from blue (most positive) to red (most negative); numeric values are printed in each cell.

### Key takeaway

- Clean 2 × 2 factorial with formal main-effect + interaction terms — the strongest evidence in the paper for what each signal does.
- **Two signals, two jobs.** UNI = realism/style anchor (drives FUD, LPIPS, HED); TME = nuclear geometry (drives PQ, DICE). Near-orthogonal contributions.
- Large FUD interaction = UNI and TME partly redundant for realism; either alone recovers most of the gain (so TME-only inference still passes the realism check).
- PQ/DICE interaction ≈ 0 = TME's contribution to segmentation does not depend on UNI, validating TME-only inference for downstream cell-counting tasks.

## 07d_t1_spatial_multi_encoder (`figures/pngs_updated/concat/07d_t1_spatial_multi_encoder.png`)

**Outline name:** Multi-encoder spatial (within-tile) decodability

### Figure caption

**Figure SI-X. Aggregate TME channels are recoverable within-tile from multiple frozen H&E encoders, but individual MX marker intensities are not.** Each target is regressed on spatially-resolved (within-tile patch) features using a probe with k-fold cross-validation; held-out within-tile R² is reported. **(A)** Within-tile decoding R² for aggregate TME targets (per-patch density or mean intensity) across four pretrained H&E encoders — UNI-2h, Virchow2, CTransPath, and ResNet-50 — with one boxplot per encoder per target. Targets are ordered left-to-right by descending median R²; boxplots show CV-fold spread with the central line at the median. **(B)** Within-tile decoding R² for individual MX marker mean intensities using the best encoder (UNI-2h), one bar per marker, ordered by descending R²; error bars are the CV-fold standard error. The y-axis extends below zero where the probe underperforms the mean predictor.

### Key takeaway

- The information case for MX-derived conditioning: high-R² aggregate targets (proliferation, density, cancer/healthy maps) contain "free" H&E signal, while low-R² targets (vasculature, immune, dead) and essentially all individual markers (B) do not — so the MX channels carry signal H&E lacks.
- Encoder ranking is consistent (UNI-2h ≈ Virchow2 > CTransPath > ResNet-50) but margins narrow on the easy targets.
- `dead` and rare markers (e.g. FOXP3) sit near or below zero R² — consistent with their low prevalence; expect little conditioning leverage there.

## 09b_channel_color_layout_impact (`figures/pngs_updated/concat/09b_channel_color_layout_impact.png`)

**Outline name:** Decodability vs color and layout generative impact

### Figure caption

**Figure SI-X. Sub-channel decodability from H&E is largely decoupled from each channel's generative impact on color and on nuclear layout.** Both panels share the x-axis — within-tile decodability (R²) of each TME sub-channel from the UNI-2h spatial probe — and the same point identities, colored and shaped by group: cell types (blue circle), cell state (red square), vasculature (green triangle), and microenv (yellow diamond). The x-axis carries a break to accommodate the low-R² microenv and dead channels. **(A)** Generative color impact (ΔE) of each sub-channel under leave-one-out ablation (the color shift when that channel is removed). **(B)** Generative layout impact (ΔPQ, the Panoptic-Quality drop under the same leave-one-out), computed as a metric pass over the cached LOO generations with no probe refit. Dotted guides mark the per-panel quadrants.

### Key takeaway

- Decodability and generative impact are not the same axis: channels poorly recovered from H&E (Glucose, O₂, Vasculature) drive the largest color shifts (A), while the cell-state channels drive layout/segmentation (B).
- Microenv channels (Glucose, O₂) are low-R² but high color-impact — they supply appearance information H&E cannot reconstruct on its own.
- The split between panels A and B mirrors the UNI/TME decomposition: color impact ↔ realism, layout impact ↔ nuclear geometry.

## ablation_grids_combined (`figures/pngs_updated/concat/ablation_grids_combined.png`)

**Outline name:** Qualitative ablation grids

### Figure caption

**Figure SI-X. Generated H&E patches under channel-group ablations illustrate ControlNet output for paired and unpaired inference.** **(A)** Paired-inference results for one representative tile. **(B)** Unpaired-inference results for one representative tile. Each block is a 4 × 4 grid; the first cell (gray dashed border, top-left) is the real H&E reference for that tile, and the remaining 15 cells are generated patches sorted by Panoptic Quality (PQ; higher is better) from best to worst across channel-group on/off combinations. The 4-dot header above each cell encodes active channel groups in the order CT, CS, VA, NU (Cell types, Cell state, Vasculature, Nutrient); a filled dot indicates the group is included. Per-cell bar plots report LPIPS (lower is better), PQ (higher is better), DICE (higher is better), and HED (lower is better) for that single patch, with the numeric value printed beside each bar.

### Key takeaway

- Sorting by PQ makes the visual gradient direct — top-left = best segmentation match, bottom-right = worst — so the metric ranking can be eyeballed against image quality.
- Side-by-side paired (A) vs unpaired (B) shows paired holds stain and gross structure further down the rank list; unpaired drifts in color and nucleus contour faster.
- CS-off conditions are consistently low-PQ in both regimes, confirming cell-state as load-bearing for nuclear geometry independent of the reference H&E.
- SI-tier qualitative confirmation of the main-figure trends.

## SI_A1_A2_unified (`figures/pngs_updated/concat/SI_A1_A2_unified.png`)

**Outline name:** TME-conditioning architecture ablation (training, metrics, qualitative)

### Figure caption

**Figure SI-X. Conditioning-architecture ablation: how the TME signal is injected affects training stability and output quality.** Variants compared: Grouped TME only, Concat TME encoder, Per-channel TME encoders, additive Mask + TME, and an off-the-shelf Vanilla PixCell ControlNet baseline. **(A)** Training loss (left) and gradient norm (right) versus optimization step for the conditioning-architecture variants; per-channel and mask-only-bypass variants exhibit the unstable trajectories. **(B)** ΔLPIPS of each conditioning variant relative to the baseline (lower is better). **(C)** Qualitative tiles across 10 columns. Rows are the real reference H&E followed by each variant's generated H&E; overlaid contours mark segmented nuclei (red = generated, gray = reference) so segmentation fidelity can be compared per tile. The per-variant metrics table — FUD (↓), DICE (↑), PQ (↑), LPIPS (↓), HED (↓) and trainable parameter count, best value per column in bold — is provided as a separate table (`SI_A1_A2_section2_metrics.png`).

### Key takeaway

- The Concat and Grouped TME encoders train stably and lead the quantitative metrics (B); the per-channel and mask-only-bypass variants destabilize training (A).
- The qualitative overlays (D) show the stable variants reproduce nuclear contours closely, while the Vanilla baseline drifts.
- Justifies the production choice of grouped/concat TME conditioning over the per-channel and bypass alternatives.
