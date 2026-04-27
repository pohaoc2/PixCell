# Figures

## methods (`figures_ec2/pngs/methods.png`)

**Outline name:** Paired H&E–CODEX training and inference framework

### Figure caption

**Figure 1. Paired H&E–CODEX framework for training and applying a TME-conditioned H&E generator.** **(Stage 1) Context extraction.** Whole-slide ORION-CRC samples provide co-registered H&E and 19-channel CODEX. Representative multiplex channels — cell type (PanCK, CD45, CD3e), cell state (Ki67), and vasculature (CD31) — are segmented into per-cell categorical masks via k-means. Continuous microenvironment fields (oxygen, glucose) are estimated from distance-to-vasculature and fit with a reaction–diffusion PDE. **(Stage 2) Multi-group ControlNet training.** Matched H&E tiles are encoded to SD3.5-VAE spatial features and UNI-2h semantic features; both encoders are frozen during training (indicated by hatched blocks). Grouped TME channels and a binary cell mask pass through per-group CNN encoders; mask latents supply Q tokens to cross-attention blocks reading K/V from channel features, yielding per-group additive residuals Δg aggregated as ΣΔg and passed to the PixCell ControlNet. The per-group CNN encoders and ControlNet are trainable (solid, unhatched blocks); the base PixCell-256 transformer is frozen. The model is trained to predict the noise ε added by forward diffusion to the clean H&E latent. **(Stage 3) Histology generation.** At inference, the generator consumes TME channels plus an optional UNI-2h reference, producing matched H&E when reference and channels share a tile, and mismatched (style-transferred) H&E when decoupled — enabling TME perturbation at fixed style. Scale bars: 4 mm (WSI), 100 µm (tile), 20 µm (zoom).

### Key takeaway

- Hatched blocks = frozen (SD3.5-VAE, UNI-2h, base PixCell-256 transformer). Solid blocks = trainable (per-group CNN encoders, ControlNet). Distinction justifies compute budget: only ControlNet branch updates.
- Three-stage flow is linear: extract → train → generate. Each stage is independently runnable, matching the `stage0`–`stage3` script structure.
- Dual conditioning paths (VAE spatial + UNI semantic) are explicit in Stage 2 — sets up the 2×2 UNI/TME factorial decomposition in Fig X.
- Style-transfer vs. matched inference fork in Stage 3 is the key inference affordance — enables TME perturbation without retraining.

## cell_summary_figure (`figures_ec2/pngs/cell_summary_figure.png`)

**Outline name:** Dataset cell characterization

### Figure caption

**Figure X. Broad cell-type and cell-state assignments on slide ORION-CRC33 yield distinct per-cell morphology and MX marker profiles.** **(A)** Cell area (µm², left) and circularity (right) for the three broad cell-type classes (n = 110,491 cancer, 37,123 immune, 115,832 healthy). Bars show medians; whiskers span the 25th–75th percentile. Pairwise significance bars compare the three classes within each panel. **(B)** Median z-scored MX marker intensity (winsorized to the 1st–99th percentile, then globally z-scored) for each cell type × cell-state stratum, with stratum counts shown beneath each column. The colormap spans negative (red) to positive (blue) median z; values are printed in each cell. Markers analyzed: Hoechst, Pan-CK, E-cadherin, CD45, CD3e, CD4, CD8a, CD20, CD68, Ki67. Significance was examined by two-tailed Mann–Whitney *U* test (A); \*\*\* denotes *p* < 0.001.

### Key takeaway

- 3-class partition (cancer/immune/healthy) backed by independent signal — morphology *and* protein markers separate them. Not arbitrary label.
- Lineage markers light up expected boxes: E-cad/Pan-CK = cancer, CD45/CD3e/CD4 = immune, no nuclear marker = healthy/stromal. Dataset clean.
- Ki67 + proliferative state co-vary across all 3 types, so `cell_state` channel carry real biology, not redundant with `cell_type`.
- Class imbalance: healthy 44% / cancer 42% / immune 14%; dead = 0.27% (rare, descriptive only). Justifies per-group dropout + channel weighting in `configs/config_controlnet_exp.py`.
- Bottom line: TME channel groups encode biologically separable populations, so ControlNet conditioning signal is real, not label noise.

## fig_paired_unpaired_performance (`figures_ec2/pngs/fig_paired_unpaired_performance.png`)

**Outline name:** Channel-group ablation results

### Figure caption

**Figure X. Channel-group ablations characterize ControlNet performance under paired and unpaired inference.** **(A)** Per-tile mean ± SD (standard deviation across tiles) for each of the 32 channel-group on/off combinations across five metrics: FUD (lower is better), LPIPS (lower is better), PQ (higher is better), DICE (higher is better), and HED (lower is better). Triangles indicate paired inference; squares indicate unpaired inference. The shaded band shows the benchmark mean ± SD. **(B)** Top-3 and bottom-3 ranked conditions per metric for paired (upper) and unpaired (lower). The dot column encodes active channel groups in the order CT, CS, VA, NU, UNI; filled = included. CT (Cell types) groups the healthy / cancer / immune density maps; CS (Cell state) groups the proliferative / nonproliferative / dead density maps; VA (Vasculature) is the vasculature density channel; NU (Nutrient) groups the oxygen and glucose channels; UNI is the UNI-2h H&E reference embedding (paired-only). **(C)** Per-group effect-size heatmaps showing mean ± SD change in each metric attributable to including that group (paired top, unpaired bottom), color-scaled from blue (positive) to red (negative); rows correspond to the same CT / CS / VA / NU groups defined above.

### Key takeaway

- Paired (triangle) beat unpaired (square) across all 5 metrics. Reference H&E pin style; absent it, FUD/LPIPS/HED drift up.
- Cell-state and Nutrient co-dominate nuclear-segmentation metrics. ΔPQ +0.20 / +0.26 paired and +0.15 / +0.19 unpaired; ΔDICE +0.29 / +0.28 paired and +0.27 / +0.27 unpaired. Tied magnitudes — not cell-state alone.
- FUD direction flips for the two strongest groups between regimes. Cell-state: −6.48 paired vs +5.39 unpaired. Nutrient: −2.29 paired vs +10.31 unpaired. With reference H&E, dense conditioning lower realism distance; without reference, the same channels over-constrain the generator and FUD blow up.
- Cell-types small everywhere (ΔFUD −0.91 paired, −0.76 unpaired; near-zero on PQ/DICE). Broad lineage map alone is coarse signal.
- Vasculature ≈ 0 on every metric in both regimes. Channel currently dead weight; candidate for removal or stronger encoder.
- LPIPS and HED move very little for any single group (|Δ| ≤ 0.04). The metric-level differentiation in this ablation lives mostly in FUD, PQ, DICE.
- Benchmark band in (A) is tight relative to between-condition spread, so ablation differences exceed measurement noise.

## fig_ablation_grids (`figures_ec2/pngs/fig_ablation_grids.png`)

**Outline name:** Qualitative ablation grids

### Figure caption

**Figure SI-X. Generated H&E patches under channel-group ablations illustrate ControlNet output for paired and unpaired inference.** **(A)** Paired-inference results for one representative tile. **(B)** Unpaired-inference results for one representative tile. Each block is a 4 × 4 grid; the first cell (red dashed border, top-left) is the real H&E reference for that tile, and the remaining 15 cells are generated patches sorted by Panoptic Quality (PQ; higher is better) from best to worst across channel-group on/off combinations. The 4-dot header above each cell encodes active channel groups in the order CT, CS, VA, NU (Cell types, Cell state, Vasculature, Nutrient); a filled dot indicates the group is included. Per-cell bar plots report LPIPS (lower is better), PQ (higher is better), DICE (higher is better), and HED (lower is better) for that single patch, with the numeric value printed beside each bar.

### Key takeaway

- Sort by PQ make the visual gradient direct — top-left = best segmentation match to ref, bottom-right = worst. Eye can verify metric ranking against image quality.
- Side-by-side paired (A) vs unpaired (B) lets reviewer compare regimes directly: paired holds stain and gross structure further down the rank list; unpaired drift in color and nucleus contour faster.
- CS off (2nd dot empty) consistently low PQ in both regimes. Confirms cell-state as load-bearing for nuclear geometry, independent of reference H&E.
- Per-cell bars inside grid let reviewer eyeball-correlate numeric metrics with visual artifacts, defending the metric choice.
- SI-tier figure: qualitative confirmation of main-figure trends. Keep in supplement.

## 08_uni_tme_decomposition (`figures_ec2/pngs/08_uni_tme_decomposition.png`)

**Outline name:** UNI–TME factorial decomposition

### Figure caption

**Figure X. UNI conditioning and TME conditioning contribute near-orthogonally to ControlNet output, separating realism from nuclear geometry.** The four conditions toggle UNI conditioning and the full TME channel stack independently: full (UNI on, TME on), UNI-only (UNI on, TME off), TME-only (UNI off, TME on), and unconditional (UNI off, TME off). **(A)** A representative tile. The top row shows the inputs to the model: the real H&E used as the style reference (left) and a visualization of the TME conditioning (right), which aggregates per-cell location, type, state, vasculature, and nutrient channels. The lower rows show generated H&E (left) for two of the four UNI/TME settings; the dot pair beside each row indicates which of UNI and TME is active (filled = on). **(B)** Per-tile mean ± SD for each of the four conditions across five metrics: FUD (lower is better), LPIPS (lower is better), PQ (higher is better), DICE (higher is better), and HED (lower is better). The 2-dot label under each marker encodes UNI/TME state. **(C)** Two-way decomposition of each metric across the four conditioning settings, reported as mean ± SD across tiles. The three rows are: the UNI effect, defined as full − tme_only (the marginal gain from adding UNI on top of TME); the TME effect, defined as full − uni_only (the marginal gain from adding TME on top of UNI); and the UNI × TME interaction, defined as full − uni_only − tme_only + unconditional, where positive values indicate synergy (UNI and TME combined exceed the sum of their individual gains) and negative values indicate redundancy or competition. Cells are color-scaled from blue (most positive) to red (most negative); numeric values are printed in each cell.

### Key takeaway

- Clean 2 × 2 factorial. Decompose effect of UNI vs TME on output quality with formal main-effect + interaction terms. Strongest evidence in paper for what each signal does.
- **Two signals, two jobs.** UNI = realism/style anchor (drives FUD, LPIPS, HED). TME = nuclear geometry (drives PQ, DICE). Near-orthogonal contributions.
- FUD interaction = +70: UNI and TME partially redundant for realism — either alone recovers most of the gain. Practical: unpaired inference with TME-only still passes UNI-feature realism check.
- PQ/DICE interaction ≈ 0: TME contribution to segmentation metrics does *not* depend on UNI presence. Validates TME-only inference for downstream cell-counting tasks.
- A shows two of the four conditioning settings; the unconditional baseline (UNI off, TME off) is captured quantitatively in B/C where it confirms baseline collapse — generator without conditioning is texture noise, so all measured effects are real signal not chance.
- This figure is the headline justification for the dual-conditioning design. Each branch carry distinct, complementary information.

## 07_inverse_decoding (`figures_ec2/pngs/07_inverse_decoding.png`)

**Outline name:** H&E encoder probe decoding

### Figure caption

**Figure X. Aggregate TME channels are recoverable from frozen H&E encoders, but individual MX marker intensities are not.** Each target is regressed on per-tile features from four pretrained H&E encoders (UNI-2h, Virchow2, CTransPath, ResNet-50) using a ridge probe with k-fold cross-validation; held-out R² is reported. **(A)** Decoding R² for aggregate TME channels (per-tile mean intensity or density), with one boxplot per encoder per target. Targets are ordered left-to-right by descending median R². Boxplots show CV-fold spread; central line is the median. **(B)** Decoding R² for individual MX marker mean intensities using the best encoder, one bar per marker, ordered left-to-right by descending R². Error bars are the CV-fold standard error. The y-axis in (B) extends below zero where the probe underperforms the mean predictor.

### Key takeaway

- **The information case for MX-derived conditioning.** If H&E alone could already recover a channel, conditioning on it would be redundant. (B) shows individual markers are *not* recoverable, so MX channels carry signal H&E does not.
- (A) sorts channels by how much "free" H&E signal they contain. High-R² targets (density, prolif, healthy/cancer maps) — generator could plausibly synthesize the H&E from coarse hints. Low-R² targets (vasculature, immune density, dead) — generator *needs* the channel.
- `dead` near-zero R² across all encoders, consistent with the 0.27 % prevalence in the cell summary. Rare events resist both probing and conditioning; expect this channel to contribute little.
- UNI-2h and Virchow2 are roughly matched, and both outperform CTransPath, which outperforms ResNet-50. Backbone choice for downstream feature work matter for higher-order channels but margins narrow on easy targets.
- (B) is the cleanest argument that MX panels add information beyond H&E. Without this panel, a reviewer ask "why not just train H&E to H&E?" Answer: most markers invisible.
- Negative R² on FOXP3 = probe systematically worse than mean — likely reflects sparse Treg distribution: model picks up correlated tissue features that anti-correlate with rare FOXP3⁺ cells. Don't over-interpret beyond "not decodable."
