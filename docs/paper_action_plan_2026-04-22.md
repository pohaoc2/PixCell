# Paper Action Plan — Morphological Grammar of the Tumor Microenvironment

**Date**: 2026-04-22
**Scope**: paired H&E ↔ multiplex (CODEX) analysis using PixCell + he-feature-visualizer
**Working title**: *Decoding the morphological grammar of the tumor microenvironment: what H&E sees, what it misses, and what combinations encode*

---

## 1. Storyline

We combine paired H&E and multiplex imaging with a controllable generative model (ControlNet + multi-group TME module) to systematically map the morphological imprint of tumor microenvironment (TME) variables on H&E. Four claims organize the paper:

1. **Visibility map** — systematic perturbation of TME channels reveals which variables (cell state, microenvironment, vasculature, lineage) leave visible morphological signatures on H&E.
2. **Inverse decoding** — linear probes on frozen H&E embeddings quantify per-channel MX information content, yielding an interpretable lower bound on what morphology encodes.
3. **Representation decomposition** — the frozen pathology foundation model (UNI-2h) and the MX conditioning carry orthogonal information: UNI controls *style* (stain, tissue class), TME controls *layout* (cells, niche).
4. **Combinatorial grammar** — systematic combinatorial perturbation of (cell state × oxygen × glucose) reveals non-additive emergent H&E signatures, generating testable mechanistic hypotheses about TME-morphology coupling.

**Differentiation from prior work**:
- Not virtual staining: we quantify information content with frozen-feature probes, not image-level reconstruction with a tuned generator.
- Not another ControlNet: the controllable model is instrument, not product; ablation and probing are the scientific contribution.
- Not interpretability hand-waving: claims are quantitative (R², AUROC, pixel visibility %, emergent signature metrics).

### 1.1 Probe architecture (clarification)

The inverse probe is **asymmetric** — H&E goes through a frozen pathology encoder; CODEX does not. This is critical because pathology foundation models (UNI-2h, Virchow, CONCH) are pretrained on H&E only and produce meaningless features on fluorescence channels.

```
H&E tile ──► [frozen UNI-2h / Virchow / raw CNN] ──► embedding (R^D) ──► [small head] ──► prediction
                                                                                              │
                                                                                              │
CODEX tile ─► [direct per-channel summary, no encoder] ──────────────────────────────── target
```

Target options (CODEX side, no neural encoder):
- **Tile-mean intensity**: one scalar per marker per tile (K markers → target ∈ R^K). Regression with MSE.
- **Mask-derived summary**: fraction of tile labeled {prolif, cancer, vessel, ...} from our existing channel masks. Regression or classification.
- **Per-channel histogram**: quantile bins of intensity distribution per marker per tile.
- *Not* pixel-level reconstruction — that is virtual-staining territory.

Head options:
- Linear: `R^D → R^K`.
- 2-layer MLP: adds nonlinearity without becoming a generator.

Encoder ablations (H&E side only):
- UNI-2h (primary)
- Virchow (secondary)
- Small CNN on raw pixels (baseline, controls for pretraining bias)

Agreement across all three encoders on an R²≈0 channel = robust "H&E is blind to this signal" claim.

### 1.2 Differentiation vs prior H&E → molecular work

| method | input | output | head | claim form |
|--------|-------|--------|------|------------|
| Virtual staining (Rivenson 2019, Bai 2023, VirtualMultiplexer 2024) | H&E tile | pixel-level synthesized marker image | end-to-end generator (U-Net / GAN / diffusion) | "image looks like IF" |
| HE2RNA (Schmauch 2020) | WSI (MIL) | bulk gene expression per gene | attention MIL + linear | per-gene AUC / correlation |
| ST-Net / HisToGene / Hist2ST (2020–2022) | H&E tile | spatial gene expression per spot | CNN/transformer + regression | per-gene Pearson R |
| iStar (Zhang 2024) | H&E WSI | super-resolved spatial transcriptomics | hierarchical CNN | per-gene R² |
| BLEEP (Xie 2023) | H&E tile | contrastive alignment with ST embeddings | contrastive encoder | retrieval accuracy |
| **Our probe** | **H&E tile** | **tile-level CODEX summary (mean / mask fraction) per channel** | **frozen foundation encoder + linear/MLP** | **R² / AUROC per channel → information bound** |

Positioning sentence:
> *"Prior work maps H&E to RNA (HE2RNA, ST-Net, iStar) or synthesizes stained images (virtual staining). We ask the complementary, largely unaddressed question: given a controlled generative model of paired H&E + CODEX, which TME axes leave morphological traces, and which do not? We answer with systematic perturbation-based visibility mapping and frozen-feature probes, yielding an information map rather than a reconstruction."*

---

## 2. Paper Outline

### Title (candidates)
1. *Morphological grammar of the tumor microenvironment: systematic ablation of paired H&E and multiplex imaging*
2. *What H&E sees and misses: an information-theoretic dissection of tumor microenvironment morphology*
3. *Decoding H&E: mapping cellular, state, and metabolic signatures with paired multiplex ablation*

### Abstract (~200 words, draft after Phase 3)

### 1. Introduction
- 1.1 Motivation: H&E is universal; MX is costly. What does morphology already encode about the microenvironment?
- 1.2 Gap: prior work (virtual staining, pathology foundation models) treats H&E as input to prediction, not as object of scientific inquiry.
- 1.3 Contributions: (i) visibility map, (ii) frozen-feature inverse probe, (iii) foundation-model decomposition, (iv) combinatorial grammar.

### 2. Related Work
- 2.1 Virtual staining (Rivenson, Ounkomol, Bai, Burlingame)
- 2.2 Pathology foundation models (UNI, Virchow, Prov-GigaPath)
- 2.3 Multiplex imaging and spatial atlases (CODEX, MIBI, Xenium, HuBMAP, HTAN)
- 2.4 Conditional generative models for pathology (PathLDM, PixCell, MorphDiff)

### 3. Methods
- 3.1 Paired dataset construction (ORION-CRC, tile registration, channel groups)
- 3.2 Controllable generator (ControlNet + MultiGroupTME, zero-mask residual path)
- 3.3 Ablation protocol (group presence/absence, leave-one-out, relabeling, channel sweep)
- 3.4 Evaluation metrics (AJI, PQ, LPIPS, cosine, FID, style-HED, pixel visibility)
- 3.5 Inverse probe protocol (frozen UNI → linear/MLP head, per-channel decoding)

### 4. Results

#### 4.1 Visibility map — what H&E sees
- *Main claim*: cell_state and microenv leave strong morphological signatures (~20% pixels affected); coarse cell_types and vasculature are near-invisible (~2%).
- Figure: visibility bar chart + representative leave-one-out diffs.
- Quantitative table: mean diff, max diff, %pixels>10, paired and unpaired.

#### 4.2 Specificity–realism tradeoff
- *Main claim*: adding groups improves structural faithfulness (AJI, PQ, LPIPS) but hurts unconditional realism (FID).
- Figure: tradeoff scatter.
- Discussion: metric mismatch, not model failure.

#### 4.3 Inverse decoding — what H&E encodes
- *Main claim*: frozen UNI embeddings linearly decode cell_state and microenv well, lineage poorly.
- Figure: per-channel R² / AUROC bar plot; comparison on real vs generated H&E.
- Table: upper bound (real H&E) vs generator self-consistency (generated H&E).

#### 4.4 Foundation-model decomposition
- *Main claim*: UNI carries global style, MX conditioning carries local layout; the two axes are separable.
- Figure: 2×2 panel (UNI+TME / UNI-only / TME-only / neither) × representative tiles.
- Table: style similarity vs layout fidelity per mode.

#### 4.5 Combinatorial grammar — emergent signatures
- *Main claim*: (cell_state × O2 × glucose) combinations produce non-additive H&E phenotypes matching known biology (e.g., hypoxia + proliferation → necrotic rim).
- Figure: 3×3×3 grid of generated patches + per-combination signature heatmap.
- Table: interaction tests, biological referents.

### 5. Discussion
- 5.1 Implications for pathology foundation-model pretraining
- 5.2 Limitations: coarse cell_types (mitigated in Supplementary), single-cohort (ORION-CRC only), CODEX-panel bias
- 5.3 Future work: fine-grained lineage, multi-cohort validation, spatial-transcriptomics extension

### 6. Conclusion

### Supplementary
- S1. Fine-grained cell_types ablation (Phase 4)
- S2. Training details, hyperparameters, compute budget
- S3. Full ablation tables (all 15 group combinations × 6 metrics, paired + unpaired)
- S4. Inverse probe ablations (linear vs MLP, layer choice)
- S5. Failure cases

---

## 3. Deliverables (ranked by priority)

| # | Deliverable | Phase | Owner | Status |
|---|-------------|-------|-------|--------|
| D1 | Visibility map table + figure (paired + unpaired) | 0 | — | data exists, needs packaging |
| D2 | Specificity–realism tradeoff figure | 0 | — | data exists, needs plot |
| D3 | Per-channel linear-probe R²/AUROC (mask-fraction targets; UNI / Virchow / raw-CNN × linear / MLP) | 1 | — | not started, target data already available |
| D4 | Per-channel regression R² (raw CODEX mean intensity + histogram targets) | 1 | — | not started, needs raw CODEX tile alignment check |
| D5 | UNI/TME 2×2 decomposition panel + metrics | 2 | — | qualitative result known, needs quantification |
| D6 | Combinatorial sweep grid + interaction analysis | 3 | — | partial (exp1 microenv sweep exists) |
| D7 | Fine-grained cell_types re-ablation | 4 (supp) | — | not started |
| D8 | Method schematic figure | writeup | — | not started |
| D9 | Abstract + full draft | writeup | — | after Phase 3 |

---

## 4. Phases and Tasks

### Phase 0 — Reframe existing data (1 week)
No new training. Repackage the ablation already completed.

- [ ] A0.1 Convert `ablation_summary_2026-04-03.md` §5 leave-one-out numbers into a visibility-map bar chart (paired + unpaired side-by-side).
- [ ] A0.2 Build specificity–realism scatter: x=groups active, y=AJI/PQ, dual-axis FID. Annotate Pareto front.
- [ ] A0.3 Build summary table combining paired + unpaired deltas (cell_types, cell_state, vasculature, microenv).
- [ ] A0.4 Write §4.1 and §4.2 first drafts.
- **Blockers**: none.
- **Output**: D1, D2, drafts of §4.1, §4.2.

### Phase 1 — Inverse probe (2 weeks)
Frozen-feature probes to quantify per-channel decoding. **Tile-level only** — no pixel-level / spatial reconstruction (keeps probe cheap and distinguishes from virtual staining).

Target forms (per CODEX tile):
- T1 Mask-derived fractions: % prolif, % cancer, % immune, % vessel per tile (already available from existing channel masks).
- T2 Per-channel mean intensity: one scalar per marker per tile (needs raw CODEX).
- T3 Per-channel histogram quantiles: 4 quantiles per marker per tile (needs raw CODEX, optional).

Encoder ablations (H&E side):
- E1 UNI-2h (primary)
- E2 Virchow-2 (secondary)
- E3 Small 4-layer CNN on raw 256×256 pixels (baseline for pretraining bias)

Tasks:
- [ ] A1.1 Cache frozen encoder embeddings for all evaluation tiles. UNI already cached; add Virchow pass if not cached. Raw-pixel CNN trained from scratch during probe step.
- [ ] A1.2 Build target T1 (mask fractions) from existing `exp_channels/` masks. This is free — no new data needed.
- [ ] A1.3 Train linear probe on (E1 × T1). 5-fold CV by tile. Report R² per target dimension.
- [ ] A1.4 Train MLP probe (2 hidden layers, ReLU, dropout 0.1) on (E1 × T1). Compare to linear.
- [ ] A1.5 Repeat A1.3 with E2, E3. Build the E × probe × target matrix (robust null check).
- [ ] A1.6 Repeat A1.3 using **generated** H&E as input (self-consistency: does the generator preserve probeable information?).
- [ ] A1.7 Check ORION-CRC33 data contract for raw CODEX tile alignment.
- [ ] A1.8 If raw CODEX available:
  - A1.8a Build target T2 (per-channel mean intensity).
  - A1.8b Train linear + MLP probes on (E1 × T2), report per-marker R².
  - A1.8c Optional: T3 histogram target.
- [ ] A1.9 Write §4.3 draft with encoder-ablation table.
- **Blockers**: Virchow-2 access / weights; raw CODEX alignment for T2, T3.
- **Output**: D3 (mask-level probe), D4 (raw-marker probe if feasible), §4.3 draft.

### Phase 2 — Foundation-model decomposition (1 week)
Quantify UNI vs TME contribution. Builds on existing qualitative finding.

- [ ] A2.1 Inference sweep: {UNI+TME, UNI-only (TME=0), TME-only (UNI=0), neither} × N=500 tiles.
- [ ] A2.2 Metrics per mode: style-HED distance to reference, AJI/PQ vs ground truth, FID, nuclei count error.
- [ ] A2.3 Build 2×2 qualitative panel (1 tile × 4 modes) + metric bar plot.
- [ ] A2.4 Write §4.4 draft.
- **Blockers**: none (cfg_dropout already supports UNI=0; TME=0 via mask zeroing).
- **Output**: D5, §4.4 draft.

### Phase 3 — Combinatorial grammar (2–3 weeks)
Main mechanistic result. Extends existing microenv sweeps.

- [ ] A3.1 Define sweep grid: cell_state ∈ {prolif, nonprolif, dead} × O2 ∈ {low, mid, high} × glucose ∈ {low, mid, high} = 27 conditions.
- [ ] A3.2 Select K=20 anchor layouts (diverse cell arrangements from held-out tiles).
- [ ] A3.3 Generate 27 × 20 = 540 tiles.
- [ ] A3.4 Define morphological signature metrics per tile: nuclear density, eosin/hematoxylin ratio, GLCM texture descriptors, mean cell size.
- [ ] A3.5 Fit additive model: signature ~ state + O2 + glucose. Residuals = non-additive interaction.
- [ ] A3.6 Identify top-K interaction conditions. Compare to known biology (hypoxia→necrosis, glucose-starvation→shrinkage).
- [ ] A3.7 Figure: 3×3×3 grid for 1 anchor + interaction heatmap.
- [ ] A3.8 Write §4.5 draft.
- **Blockers**: need biology reference list for interaction validation.
- **Output**: D6, §4.5 draft.

### Phase 4 — Fine-grained cell_types (3–4 weeks, supplementary)
Address reviewer concern about coarse labels.

- [ ] A4.1 Audit raw CODEX panel: which markers distinguish CD4/CD8/Treg/macrophage/B/NK/fibroblast?
- [ ] A4.2 Re-process masks at fine resolution. Filter types with <1% prevalence.
- [ ] A4.3 Extend `MultiGroupTMEModule.cell_types` encoder to N channels.
- [ ] A4.4 Retrain with extended group. Reuse other groups unchanged.
- [ ] A4.5 Re-run ablation (at least presence/absence + leave-one-out).
- [ ] A4.6 Compare coarse vs fine: does signal appear?
- [ ] A4.7 Write Supplementary S1.
- **Blockers**: raw CODEX panel quality, training compute.
- **Output**: D7, Supplementary S1.

### Phase 5 — Writeup and revision (3 weeks, after Phase 3)
- [ ] A5.1 Method schematic figure (pipeline + TME module architecture).
- [ ] A5.2 Assemble main figures, standardize style.
- [ ] A5.3 Write §1 Introduction, §2 Related Work, §3 Methods.
- [ ] A5.4 Write §5 Discussion, §6 Conclusion.
- [ ] A5.5 Abstract.
- [ ] A5.6 Internal review + revision.

---

## 5. Figure Plan

### Figure budget (main text target: 6 figures)

| Fig | Title | Content | Source | Status |
|-----|-------|---------|--------|--------|
| F1 | Pipeline and architecture | Schematic: paired data → ControlNet + MultiGroupTME → H&E. Inset: MultiGroupTME detail (4 encoders, cross-attn, zero-mask residual). | new | to draw |
| F2 | Visibility map | Bar chart per channel group: mean pixel diff, % pixels changed, paired vs unpaired. 4–6 representative leave-one-out diff tiles inset. | existing data | needs plotting |
| F3 | Specificity–realism tradeoff | Scatter: AJI/PQ vs FID as groups added (1g→4g). Pareto front highlighted. | existing data | needs plotting |
| F4 | Inverse decoding | (a) Per-channel linear probe R²/AUROC bar plot, real vs generated H&E. (b) Example decoded masks overlaid on H&E. | new (Phase 1) | not started |
| F5 | UNI/TME decomposition | 2×2 image panel (modes) + metric bar plot (style sim, layout fidelity, FID). | partially done (qualitative known) | needs sweep + quantification |
| F6 | Combinatorial grammar | (a) 3×3×3 grid for 1 anchor tile, (b) interaction heatmap (residuals after additive fit), (c) exemplar interaction annotated with biology. | partial (exp1 microenv) | needs extension |

### Supplementary figures (target: 8–12)
- S1 Full ablation grid (all 15 combinations × N tiles)
- S2 Fine vs coarse cell_types comparison
- S3 Per-metric per-condition boxplots (paired)
- S4 Per-metric per-condition boxplots (unpaired)
- S5 Linear vs MLP probe comparison
- S6 Probe transfer: real→generated and generated→real
- S7 Failure cases and attribution
- S8 Style-HED distribution per mode
- S9 Full combinatorial sweep for 5 additional anchors
- S10 Training curves, compute

### Figures locally available (WSL path root: `\\wsl.localhost\Ubuntu-22.04\home\pohaoc2\UW\bagherilab\PixCell`)

> All inference outputs previously on `ec2-user` are now synced locally.

| File (Linux path relative to repo root) | Usable for | Notes |
|------------------------------------------|------------|-------|
| `figures/pngs/01_metric_tradeoffs.png` | F3 (specificity–realism tradeoff) | reformat axes, add Pareto front |
| `figures/pngs/02_paired_vs_unpaired.png` | F3 (paired vs unpaired panel) | combine with 01 |
| `figures/pngs/03_channel_effect_sizes.png` | F2 (visibility map bar chart) | verify group labels |
| `figures/pngs/04_leave_one_out_impact.png` | F2 (leave-one-out bar chart) | crop/relabel axes |
| `figures/pngs/05_paired_summary.png` | §4.1 / F2 | verify metric alignment |
| `figures/pngs/06_unpaired_summary.png` | §4.2 / F2 | verify metric alignment |
| `figures/dataset_metrics_manual.png` | F3 (backup tradeoff) | check axes |
| `inference_output/paired_ablation/ablation_results/<tile>/leave_one_out_diff.png` | F2 (inset tiles) | crop, relabel; pick 4–6 representative tiles |
| `inference_output/paired_ablation/ablation_results/<tile>/leave_one_out_diff_stats.json` | F2 (bar chart data) | aggregate across all tiles |
| `inference_output/paired_ablation/channel_sweep/cache/exp1_microenv/cancer_12800_16384.png` | F6 (microenv sweep) | extend grid from 1D to 3D |
| `inference_output/paired_ablation/channel_sweep/exp2_cell_type_relabeling.png` | S2 (coarse cell_types baseline) | keep as-is |
| `inference_output/paired_ablation/channel_sweep/exp3_cell_state_relabeling.png` | F6 (state axis exemplar) | keep + extend |
| `inference_output/unpaired_ablation/ablation_results/<tile>/ablation_grid.png` | S1 (ablation grid supp) | many tiles available; pick diverse set |
| `inference_output/unpaired_ablation/ablation_results/<tile>/metrics.json` | F3 / S3 | per-tile unpaired metrics |
| `inference_output/unpaired_ablation/ablation_results/<tile>/uni_cosine_scores.json` | §4.4 (UNI decomposition) | per-tile cosine similarity |
| `inference_output/unpaired_ablation/leave_one_out/10240_11008/leave_one_out_diff_stats.json` | F2 (unpaired bar chart) | single tile; expand by aggregating more tiles |

### Figures to create from scratch

- F1 method schematic (vector graphics, ~1 day)
- F4 inverse decoding panel (after Phase 1)
- F5 UNI/TME decomposition (partial data, needs N=500 inference sweep + metric extraction)
- F6 full 3×3×3 sweep (needs Phase 3 generation)
- S5 probe ablation (needs Phase 1)
- S7 failure case curation (~1 day after Phase 0)

---

## 6. Timeline (compressed)

| Week | Phase | Milestone |
|------|-------|-----------|
| W1 | Phase 0 + start Phase 1, 2 | Visibility map + tradeoff done (D1, D2) |
| W2 | Phase 1, 2 continue | UNI/TME decomposition quantified (D5) |
| W3 | Phase 1 finish, Phase 3 start | Probe R² table done (D3, D4) |
| W4–5 | Phase 3 | Combinatorial sweep + interaction analysis (D6) |
| W6–7 | Phase 5 writeup | Full draft, main figures assembled |
| W8 | Internal review | Revision pass 1 |
| W9 | Phase 4 (supp, parallel) | Fine-grained cell_types if data permits |
| W10 | Submission prep | Final figures, supplementary, formatting |

**Minimum viable submission**: end of W7 (~mid-June 2026), Phases 0–3 + 5.
**Strong submission**: end of W10 (~early July 2026), all phases.

---

## 7. Open Questions

1. **RESOLVED** — Raw CODEX data is available locally at `\\wsl.localhost\Ubuntu-22.04\home\pohaoc2\UW\bagherilab\he-feature-visualizer\data` (Linux: `/home/pohaoc2/UW/bagherilab/he-feature-visualizer/data`). Contents confirmed:
   - `mx_crc33.ome.tiff` — full-resolution 19-channel multiplex image (19 × 52740 × 36354, float)
   - `features_crc33.csv` — per-cell feature table: 443 514 cells × (CellID, 19 marker intensities, X/Y centroid, Area, morphology stats)
   - `markers.csv` — channel list: Hoechst, AF1, CD31, CD45, CD68, Argo550, CD4, FOXP3, CD8a, CD45RO, CD20, PD-L1, CD3e, CD163, E-cadherin, PD-1, Ki67, Pan-CK, SMA
   - `he_crc33.ome.tif`, `mask_crc33.tif` — full-resolution H&E and cell mask
   - For T2/T3 probe targets (D4): tile-level mean intensities can be derived by joining `features_crc33.csv` (cell centroids) to the 256×256 tile grid. Tile alignment feasible — no new data required.
2. Which journal target? (Nature Methods, Nat Biomed Eng, Cell Systems, Med Image Anal) — affects scope and figure budget.
3. Fine-grained cell_types panel: which 5–8 lineage markers to prioritize? (candidate set from markers.csv: CD4, CD8a, FOXP3, CD20, CD68, CD163, CD3e → T-cell subtypes + macrophage + B-cell)
4. Ethics/data-use for cross-cohort extension (if attempted later).

---

## 8. Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| Linear probe too weak (all R²≈0 on real H&E) | Escalate to MLP, ViT probe; report as negative finding with discussion |
| Combinatorial interactions don't match known biology | Frame as exploratory hypothesis generation, not confirmation; invite follow-up wet-lab |
| Raw CODEX intensities not per-tile aligned | Stick with mask-level probes; note in limitations |
| Reviewer pushback on coarse cell_types | Phase 4 supplementary addresses head-on |
| FID story confuses reviewers | Reframe as "conditioning specificity vs unconditional realism are distinct objectives" with clear section |
| Compute for Phase 4 retraining prohibitive | Only retrain TME module with extended cell_types, freeze everything else |

---

## 9. Useful References to Cite

### Virtual staining (H&E → synthesized marker image)
- Rivenson et al. (2019, *Nat Biomed Eng*) — H&E → IHC deep virtual staining.
- Ounkomol et al. (2018, *Nat Methods*) — label-free prediction of 3D fluorescence.
- Bai et al. (2023, *Nat Rev Bioeng*) — review.
- Burlingame et al. (2020) — H&E → Ki67.
- Ghahremani et al. (2023) — multi-marker virtual IHC.
- Pati et al. (2024) — **VirtualMultiplexer**, H&E → multi-marker IF (closest to "H&E → CODEX").

### H&E → gene expression (bulk and spatial)
- Schmauch et al. (2020, *Nat Commun*) — **HE2RNA**, tile-MIL → bulk RNA. Closest methodological analog to our probe.
- Pizurica et al. (2024, *Nat Commun*) — **SEQUOIA**, WSI transformer → bulk RNA.
- Alsaafin et al. (2023) — **tRNAsformer**.
- He et al. (2020, *Nat Biomed Eng*) — **ST-Net**, H&E → spatial transcriptomics.
- Pang et al. (2021) — **HisToGene**.
- Zeng et al. (2022) — **Hist2ST**.
- Zhang et al. (2024, *Nat Biomed Eng*) — **iStar**, super-resolved ST from H&E.
- Xie et al. (2023, NeurIPS) — **BLEEP**, contrastive H&E ↔ ST alignment.

### Pathology foundation models (H&E encoders used for probing)
- Chen et al. (2024, *Nat Med*) — **UNI / UNI-2h**. Our primary frozen encoder.
- Vorontsov et al. (2024) — **Virchow / Virchow-2**.
- Xu et al. (2024) — **Prov-GigaPath**.
- Lu et al. (2024) — **CONCH** (vision-language, optional secondary encoder).

### Conditional generative models for pathology
- Yellapragada et al. (2024) — **PathLDM**.
- PixCell (baseline / scaffold for our work).
- **MorphDiff** (cell-morphology conditional diffusion).

### Multiplex and paired H&E + MX datasets
- Lin et al. (2023) — **ORION-CRC**, source of our paired data.
- Human Tumor Atlas Network (HTAN).
- HuBMAP.
- Human Cell Atlas (HCA).

### Probing methodology (representation-learning literature)
- Alain & Bengio (2016) — linear probe canonical reference.
- Belinkov (2022) — probing classifier methodology survey.
- Hewitt & Liang (2019) — control tasks for probe validity.
- Tenney et al. (2019) — edge probing framework.

### Controllable generation / ControlNet
- Zhang et al. (2023) — **ControlNet** original.
- Stable Diffusion 3 / 3.5 — base architecture underlying PixCell.

---

## 10. Parallelization Map

Tasks with no mutual dependencies can be executed simultaneously. The map below groups tasks by dependency tier.

### Tier 0 — Start immediately in parallel (no blockers)

| Task | Phase | Why parallel-safe |
|------|-------|-------------------|
| A0.1 Visibility map bar chart | 0 | data already local; pure plotting |
| A0.2 Specificity–realism scatter | 0 | data already local; pure plotting |
| A0.3 Summary table (paired + unpaired) | 0 | data already local; pure aggregation |
| A1.2 Build T1 mask-fraction targets | 1 | uses existing `exp_channels/` masks; no GPU |
| A1.7 Check raw CODEX data contract | 1 | pure data audit; no compute |
| A5.1 Method schematic figure | 5 | design work; no data dependency |
| A5.3 §2 Related Work draft | 5 | pure writing; no results needed |

### Tier 1 — Start after Tier 0 completes (selected subtasks)

| Task | Depends on | Notes |
|------|------------|-------|
| A0.4 Write §4.1 and §4.2 drafts | A0.1, A0.2, A0.3 | all three Phase 0 data tasks must finish |
| A1.1 Cache Virchow-2 embeddings | (none, but GPU required) | can start in parallel with A1.2; listed here because GPU scheduling is a constraint |
| A1.8a–c Raw CODEX probe targets | A1.7 | only if contract check confirms availability |

### Tier 2 — Phase 1 probe training (fully parallel among themselves)

All three can launch simultaneously once **A1.1** (encoder cache) and **A1.2** (mask targets) are ready:

| Task | Description |
|------|-------------|
| A1.3 | Linear probe: E1 (UNI) × T1 (mask fractions), 5-fold CV |
| A1.4 | MLP probe: E1 (UNI) × T1, compare to A1.3 |
| A1.5 | Linear probes: E2 (Virchow), E3 (raw CNN) × T1 |

### Phase 2 — Fully independent of Phase 1; run in parallel

A2.1 → A2.2 → A2.3 → A2.4 can proceed concurrently with all Phase 1 probe tasks. No shared compute or data dependencies.

### Tier 3 — Phase 3 setup (parallel among themselves)

| Task | Notes |
|------|-------|
| A3.1 Define 27-condition sweep grid | pure design |
| A3.2 Select K=20 anchor layouts | pure data selection |
| A3.4 Define morphological signature metrics | pure design; no generation needed |

A3.3 (generation, 540 tiles) starts after A3.1 + A3.2. A3.5 (model fitting) starts after A3.3 + A3.4.

### Tier 4 — Writeup (partial parallelism with Phases 1–3)

| Task | Can start when |
|------|----------------|
| A5.3 §3 Methods | After Phase 0 draft; Phase 1–3 methods are fixed |
| A5.3 §1 Introduction | After §4.1–4.2 drafted (Phase 0 complete) |
| A5.4 §5 Discussion, §6 Conclusion | After Phase 1 + 2 results in hand |
| A5.5 Abstract | Only after §4.1–4.5 are drafted |

### Dependency graph (simplified)

```
[A0.1, A0.2, A0.3]  ──►  A0.4 (§4.1, §4.2)
[A1.2, A1.7]         ──►  A1.1 + embeddings ready  ──►  [A1.3 ∥ A1.4 ∥ A1.5]  ──►  A1.6  ──►  A1.9
                                                                                    A1.8 (if CODEX OK)
[A2.1]  ──►  A2.2  ──►  A2.3  ──►  A2.4                        (parallel to all Phase 1)
[A3.1 ∥ A3.2 ∥ A3.4]  ──►  A3.3  ──►  A3.5  ──►  A3.6  ──►  A3.7  ──►  A3.8
[A5.1, A5.3 §2]  (start now, parallel to everything)
```

> **Phase 4 (fine-grained cell_types)**: kept as TODO — not listed as active tasks. See §5.1 Limitations and Supplementary S1 placeholder.

---

## 11. Task Source Layout (`src/`)

Each task lives in its own isolated folder under `src/<task_name>/`.

Current implementation convention:
- Specs remain centralized in this document (no duplicated per-task `README.md` files yet).
- Each task folder has a Python entry module (`run.py`, `main.py`, `build.py`, or `probe.py`) depending on task shape.
- No task-specific dependency files are needed yet; all current implementations use the repo environment.

**Implementation status (2026-04-23)**:
- CPU-complete and tested on this machine: `a0_visibility_map`, `a0_tradeoff_scatter`, `a1_mask_targets`, `a1_probe_linear`, `a1_probe_mlp`, `a1_codex_targets`
- Planner/wrapper complete and tested on this machine; execution deferred to GPU machine: `a1_probe_encoders`, `a1_generated_probe`, `a2_decomposition`, `a3_combinatorial_sweep`

```
src/
  a0_visibility_map/          # A0.1 + A0.3 bar chart + summary table
  a0_tradeoff_scatter/        # A0.2 specificity–realism scatter
  a1_mask_targets/            # A1.2 build T1 mask-fraction targets
  a1_probe_linear/            # A1.3 linear probe (UNI × T1)
  a1_probe_mlp/               # A1.4 MLP probe (UNI × T1)
  a1_probe_encoders/          # A1.5 Virchow + raw-CNN probes
  a1_generated_probe/         # A1.6 probe on generated H&E
  a1_codex_targets/           # A1.7–1.8 raw CODEX T2/T3 targets + probes
  a2_decomposition/           # A2.1–2.4 UNI/TME 2×2 decomposition
  a3_combinatorial_sweep/     # A3.1–3.8 3×3×3 combinatorial grammar
```

---

### `src/a0_visibility_map/`

**Purpose**: Aggregate `leave_one_out_diff_stats.json` across all tiles (paired and unpaired) and produce the visibility map bar chart (F2) and summary table (D1).

**Inputs**:
- `inference_output/paired_ablation/ablation_results/*/leave_one_out_diff_stats.json` — per-tile stats; keys: `cell_types`, `cell_state`, `vasculature`, `microenv`; fields: `mean_diff`, `max_diff`, `pct_pixels_above_10`
- `inference_output/unpaired_ablation/leave_one_out/*/leave_one_out_diff_stats.json` — same schema, unpaired
- Representative diff tiles: `inference_output/paired_ablation/ablation_results/*/leave_one_out_diff.png` (select 4–6 visually distinct)

**Outputs**:
- `out/visibility_bar_chart.png` — side-by-side paired/unpaired bar chart; x-axis = channel group, y-axis = mean pixel diff (primary) with % pixels > 10 as secondary; error bars = ±1 SD across tiles
- `out/visibility_summary_table.csv` — columns: `group`, `paired_mean_diff`, `paired_pct>10`, `unpaired_mean_diff`, `unpaired_pct>10`, `n_paired_tiles`, `n_unpaired_tiles`
- `out/inset_tiles/` — copies of 4–6 selected `leave_one_out_diff.png` with labels

**Method**:
1. Glob all `leave_one_out_diff_stats.json` under paired and unpaired dirs.
2. For each split × group, collect `mean_diff` and `pct_pixels_above_10` across tiles; compute mean ± SD.
3. Plot grouped bar chart (matplotlib); sort groups by paired mean_diff descending.
4. Write CSV summary.

**Acceptance criteria**:
- All 4 groups present in both paired and unpaired bars.
- Error bars visible and non-zero.
- CSV matches chart values.

---

### `src/a0_tradeoff_scatter/`

**Purpose**: Build the specificity–realism tradeoff scatter (F3). x-axis = number of active channel groups; y-axes = AJI/PQ (structural fidelity, primary) and FID (realism, secondary, reversed); highlight Pareto front.

**Inputs**:
- `inference_output/paired_ablation/ablation_results/*/metrics.json` — per-tile, per-condition metrics; relevant conditions: `cell_state`, `cell_state+cell_types`, `cell_state+cell_types+microenv`, `cell_state+cell_types+microenv+vasculature`; relevant fields: `aji`, `pq`, `fid`, `lpips`
- `inference_output/unpaired_ablation/ablation_results/*/metrics.json` — same schema

**Outputs**:
- `out/tradeoff_scatter_paired.png`
- `out/tradeoff_scatter_unpaired.png`
- `out/tradeoff_data.csv` — columns: `split`, `n_groups`, `condition`, `aji_mean`, `aji_sd`, `pq_mean`, `pq_sd`, `fid_mean`, `fid_sd`, `n_tiles`

**Method**:
1. Parse all `metrics.json`; map condition name → group count (1–4).
2. Aggregate per (split, n_groups): mean ± SD of AJI, PQ, FID across tiles.
3. Scatter: x = n_groups (jittered), left y = AJI/PQ, right y = FID (inverted axis); color by metric; annotate Pareto front condition names.
4. Save both paired and unpaired panels; optionally a combined 1×2 figure.

**Acceptance criteria**:
- 4 points per metric (1g, 2g, 3g, 4g).
- Dual y-axis with correct inversion for FID.
- Pareto front annotated.

---

### `src/a1_mask_targets/`

**Purpose**: Build T1 target vectors (mask-fraction per tile) from existing `exp_channels/` binary maps. Output is an `(N_tiles, K_targets)` numpy array aligned to the UNI embedding cache.

**Inputs**:
- `data/orion-crc33/exp_channels/` — subdirs per channel; binary PNG (uint8 0/255) for cell-type/state channels; float32 NPY for `oxygen`, `glucose`; bool NPY for `vasculature`
  - Cell-type channels (binary PNG): `cell_type_cancer`, `cell_type_healthy`, `cell_type_immune`
  - Cell-state channels (binary PNG): `cell_state_prolif`, `cell_state_nonprolif`, `cell_state_dead`
  - `vasculature` (bool NPY), `oxygen` (float32 NPY, range ~0.78–1.0), `glucose` (float32 NPY, range ~0.66–1.0)
- `data/orion-crc33/features/*.npy` — UNI embeddings; filenames `<row>_<col>_uni.npy`; shape (1536,); defines the tile list

**Outputs**:
- `out/mask_targets_T1.npy` — shape (N, 8); float32; columns in order: `cancer_frac`, `healthy_frac`, `immune_frac`, `prolif_frac`, `nonprolif_frac`, `dead_frac`, `vasculature_frac`, `oxygen_mean`, `glucose_mean`
- `out/tile_ids.txt` — one tile ID per line (`<row>_<col>`), same row order as T1 array
- `out/target_stats.csv` — mean, SD, min, max per column (sanity check)

**Method**:
1. Enumerate tile IDs from `data/orion-crc33/features/` glob `*_uni.npy`.
2. For each tile, load each channel; compute fraction of non-zero pixels (binary) or spatial mean (continuous).
3. Stack into (N, 8) array. Handle missing optional channels with NaN (no tiles expected to be missing any of the 8 here, but log if so).
4. Save outputs.

**Acceptance criteria**:
- N ≈ 10 379 (tile count from `exp_channels/cell_masks/`).
- All 8 columns; no all-NaN columns.
- `target_stats.csv` shows `cancer_frac`, `prolif_frac` values consistent with known biology (< 0.5 mean, non-trivial SD).

---

### `src/a1_probe_linear/`

**Purpose**: Train a linear probe (ridge regression / logistic regression) on frozen UNI-2h embeddings → T1 mask-fraction targets. Report per-target R² via 5-fold tile-level CV.

**Inputs**:
- `data/orion-crc33/features/*_uni.npy` — UNI-2h embeddings (1536-d, float32); one file per tile
- `src/a1_mask_targets/out/mask_targets_T1.npy` — shape (N, 8)
- `src/a1_mask_targets/out/tile_ids.txt` — row alignment

**Outputs**:
- `out/linear_probe_results.json` — per-target `{r2_mean, r2_sd, r2_folds}` across 5 CV folds
- `out/linear_probe_results.csv` — same in tabular form: columns `target`, `r2_mean`, `r2_sd`
- `out/coef_heatmap.png` — (optional) heatmap of ridge coefficients (1536 × 8)

**Method**:
1. Load all UNI embeddings; stack into (N, 1536) matrix.
2. Load T1 array; verify row alignment via `tile_ids.txt`.
3. 5-fold CV (stratify by spatial region if possible; otherwise random tile split).
4. Per fold: fit `sklearn.linear_model.Ridge(alpha=1.0)` for regression targets (all 8 are continuous in [0,1]); compute R² on held-out fold.
5. Aggregate R² mean ± SD per target across folds.
6. Save outputs.

**Acceptance criteria**:
- 5-fold CV completed for all 8 targets.
- JSON contains `r2_mean` and `r2_sd` for every target.
- At least one target with R² > 0.1 (sanity: prolif or cancer expected to be non-trivially decodable).

---

### `src/a1_probe_mlp/`

**Purpose**: Replace the linear head with a 2-hidden-layer MLP; compare R² to `a1_probe_linear` to quantify nonlinearity benefit.

**Inputs**: Same as `a1_probe_linear`.

**Outputs**:
- `out/mlp_probe_results.json` / `.csv` — same schema as linear results
- `out/comparison_table.csv` — columns: `target`, `linear_r2`, `mlp_r2`, `delta`

**Method**:
1. Same CV splits as `a1_probe_linear` (save/load split indices from that task's `out/` to ensure exact comparison).
2. MLP architecture: `Linear(1536→256) → ReLU → Dropout(0.1) → Linear(256→64) → ReLU → Linear(64→8)`; trained with Adam, lr=1e-3, up to 200 epochs with early stopping (val loss patience 10).
3. Regression: MSE loss; R² computed on held-out fold.
4. Produce comparison table.

**Acceptance criteria**:
- Uses identical CV splits as `a1_probe_linear`.
- `comparison_table.csv` present with delta column.
- Training does not overfit (val loss tracked).

---

### `src/a1_probe_encoders/`

**Purpose**: Repeat linear probe with two additional encoders — Virchow-2 (if weights available) and a small 4-layer CNN trained from scratch — to assess pretraining bias. A target where all three encoders give R²≈0 is a robust "H&E is blind" claim.

**Inputs**:
- Virchow-2 embeddings: to be cached by this task if not already present (model path TBD — flag as blocker if weights unavailable)
- Raw H&E tiles: `data/orion-crc33/he/*.png` — 256×256 RGB
- `src/a1_mask_targets/out/mask_targets_T1.npy` and `tile_ids.txt`

**Outputs**:
- `out/cnn_embeddings.npy` — (N, D_cnn) raw-CNN embeddings after training
- `out/virchow_embeddings.npy` — (N, D_virchow) if available; else `out/virchow_SKIPPED.txt`
- `out/encoder_comparison.csv` — columns: `target`, `uni_r2`, `virchow_r2`, `cnn_r2`

**Method**:
1. **Raw CNN**: 4-layer conv net (32→64→128→256 channels, 3×3 kernels, stride 2, ReLU, global avg pool → 256-d vector). Train end-to-end with same ridge probe using leave-one-fold-out on T1 targets.
2. **Virchow-2**: If weights available, run forward pass on all H&E tiles; cache embeddings. Else skip and note in outputs.
3. Fit ridge probe per encoder; report R² per target.
4. Write comparison CSV.

**Acceptance criteria**:
- CNN baseline completes (no external model required).
- Comparison CSV has `uni_r2` column matching `a1_probe_linear` output.
- `virchow_SKIPPED.txt` acceptable if weights unavailable; flag as blocker in report.

---

### `src/a1_generated_probe/`

**Purpose**: Run the linear probe (same head as `a1_probe_linear`) but using generated H&E as encoder input instead of real H&E. Measures generator self-consistency: does generated H&E preserve TME-decodable information?

**Inputs**:
- Generated H&E tiles from inference: `inference_output/paired_ablation/ablation_results/*/all/` — the "all groups active" generated image per tile
- Frozen UNI-2h encoder weights (`pretrained_models/uni-2h/`)
- `src/a1_mask_targets/out/mask_targets_T1.npy` and `tile_ids.txt`
- CV split indices from `src/a1_probe_linear/out/cv_splits.json`

**Outputs**:
- `out/generated_uni_embeddings.npy` — (N_eval, 1536) UNI embeddings of generated H&E
- `out/generated_probe_results.json` / `.csv`
- `out/real_vs_generated_r2.csv` — columns: `target`, `real_r2`, `generated_r2`, `ratio`

**Method**:
1. For each tile in the paired ablation eval set, load the "all groups" generated PNG; embed with frozen UNI-2h.
2. Fit ridge probe using same T1 targets and same CV splits (eval only — do not re-train; use probe weights from `a1_probe_linear` if desired, or refit).
3. Compare R² real vs generated.

**Acceptance criteria**:
- Covers all tiles present in `inference_output/paired_ablation/ablation_results/`.
- `real_vs_generated_r2.csv` present.
- Generated R² ≤ real R² (sanity check: generator can't create information not in the input).

---

### `src/a1_codex_targets/`

**Purpose**: Build T2 (per-channel mean intensity) and T3 (histogram quantiles) from raw CODEX `features_crc33.csv`, aligned to the 256×256 tile grid. Then run linear + MLP probes for D4.

**Inputs**:
- `/home/pohaoc2/UW/bagherilab/he-feature-visualizer/data/features_crc33.csv` — 443 514 cells × (CellID, 19 marker intensities: Hoechst, AF1, CD31, CD45, CD68, Argo550, CD4, FOXP3, CD8a, CD45RO, CD20, PD-L1, CD3e, CD163, E-cadherin, PD-1, Ki67, Pan-CK, SMA, X_centroid, Y_centroid, ...)
- `/home/pohaoc2/UW/bagherilab/he-feature-visualizer/data/mx_crc33.ome.tiff` — full-resolution 19-channel image (19 × 52740 × 36354); use for pixel-level mean if needed
- `src/a1_mask_targets/out/tile_ids.txt` — defines tile grid (row=y, col=x in 256-px units at full resolution)
- `data/orion-crc33/features/*_uni.npy` — defines valid tile set

**Outputs**:
- `out/codex_T2_mean.npy` — (N, 19) per-tile mean marker intensity; float32
- `out/codex_T3_quantiles.npy` — (N, 19×4) per-tile 4-quantile (Q10, Q25, Q75, Q90) per marker; float32
- `out/codex_tile_ids.txt` — tile IDs with ≥1 cell; some tiles may be excluded (no cells)
- `out/T2_probe_results.csv`, `out/T3_probe_results.csv` — linear + MLP R² per marker

**Method**:
1. Load `features_crc33.csv`; parse X/Y centroids. Map each cell to tile ID: `tile_row = (Y_centroid // 256) * 256`, `tile_col = (X_centroid // 256) * 256`.
2. Group cells by tile ID; compute (a) mean of each marker and (b) 4 quantiles per marker per tile.
3. Align to `tile_ids.txt` ordering; fill NaN for tiles with no cells (< 5 cells threshold → NaN).
4. Run linear and MLP probe (same CV splits from `a1_probe_linear`) for T2, report per-marker R².
5. Optional: T3 quantile probe.

**Acceptance criteria**:
- T2 array shape (N, 19); no NaN for tiles with ≥ 5 cells.
- Tile ID alignment verified against `a1_mask_targets/out/tile_ids.txt`.
- Probe results CSV covers all 19 markers.

---

### `src/a2_decomposition/`

**Purpose**: Quantify UNI vs TME contribution with a 4-mode inference sweep: `{UNI+TME, UNI-only (TME=0), TME-only (UNI=0), neither}` × N=500 tiles. Measure per-mode: style-HED distance, AJI/PQ vs ground truth, FID, cosine similarity.

**Inputs**:
- Trained model checkpoint (path from `configs/config_controlnet_exp.py`)
- `data/orion-crc33/` — full tile set for inference
- `data/orion-crc33/features/*_uni.npy` — UNI embeddings for "UNI+TME" and "UNI-only" modes; zeroed for "TME-only" and "neither"
- `data/orion-crc33/exp_channels/` — TME channels; zeroed for "UNI-only" and "neither"
- `data/orion-crc33/he/*.png` — ground-truth H&E for metrics

**Outputs**:
- `out/generated/` — 500 × 4 generated tiles (organized as `<tile_id>/<mode>.png`)
- `out/mode_metrics.csv` — columns: `mode`, `tile_id`, `aji`, `pq`, `lpips`, `fid`, `cosine`, `style_hed`
- `out/mode_summary.csv` — per-mode mean ± SD across 500 tiles
- `out/qualitative_panel/` — 5 selected tiles × 4 modes (visual panel for F5)

**Method**:
1. Select N=500 tiles randomly from eval split (seed=42 for reproducibility).
2. For each mode, run `stage3_inference.py` with appropriate flag: `--cfg_scale 0` for UNI=0, `--zero_tme` for TME=0.
3. Compute metrics per tile per mode using existing `tools/compute_ablation_metrics.py`.
4. Aggregate; select 5 diverse tiles for qualitative panel.

**Acceptance criteria**:
- All 4 modes × 500 tiles generated (2000 images total).
- `mode_summary.csv` shows UNI+TME has highest AJI/PQ; neither has lowest.
- FID ordering: neither > UNI-only (style only matters less) — confirm with data.

---

### `src/a3_combinatorial_sweep/`

**Purpose**: Systematic 3×3×3 combinatorial sweep of `cell_state ∈ {prolif, nonprolif, dead}` × `oxygen ∈ {low, mid, high}` × `glucose ∈ {low, mid, high}` = 27 conditions × K=20 anchor tiles = 540 generated tiles. Fit additive model; extract interaction residuals.

**Inputs**:
- Trained model checkpoint
- K=20 anchor tile layouts selected from `data/orion-crc33/exp_channels/cell_masks/` (diverse spatial arrangements; selection script in task)
- Channel value definitions:
  - `cell_state`: relabeling — "prolif" = `cell_state_prolif` set to 1, others 0; similarly for nonprolif, dead
  - `oxygen`: low=0.5, mid=0.75, high=1.0 (uniform fill of oxygen channel)
  - `glucose`: low=0.5, mid=0.75, high=1.0 (uniform fill of glucose channel)
- `data/orion-crc33/exp_channels/cell_masks/`, `cell_type_cancer/`, `cell_type_healthy/`, `cell_type_immune/` — kept fixed from anchor tile (layout preserved)

**Outputs**:
- `out/generated/` — 540 PNGs organized as `<anchor_id>/<state>_<o2>_<glucose>.png`
- `out/morphological_signatures.csv` — per-tile: `anchor_id`, `cell_state`, `oxygen`, `glucose`, `nuclear_density`, `eosin_ratio`, `hematoxylin_ratio`, `mean_cell_size`, `glcm_contrast`, `glcm_homogeneity`
- `out/additive_model_residuals.csv` — per condition: expected (additive fit) vs actual signature; residual norm per metric
- `out/interaction_heatmap.png` — heatmap of residual norms across 27 conditions
- `out/top_interactions.json` — top-5 non-additive combinations with biology annotation placeholders

**Method**:
1. Select K=20 anchor tiles: cluster `mask_targets_T1` embeddings with k-means (k=20); pick medoid per cluster.
2. For each of 27 conditions, modify anchor's channel channels: relabel cell_state channels; fill oxygen and glucose arrays with scalar value.
3. Run `stage3_inference.py` in batch for all 27 × 20 conditions.
4. Compute morphological signatures per generated tile: nuclear density (CellViT detections from `inference_output/.../cellvit/`), H/E ratio (HED decomposition), GLCM texture (scikit-image).
5. Fit additive model: `signature ~ state + O2 + glucose` per metric using OLS. Residuals = interaction terms.
6. Rank conditions by L2 residual norm; annotate top 5 with known biology.

**Acceptance criteria**:
- 540 images generated (27 conditions × 20 anchors).
- `morphological_signatures.csv` has 540 rows × 6 metrics.
- Additive model fit report (R² for additive fit per metric).
- At least one interaction condition with L2-residual > 1 SD (non-trivial non-additivity).

---

*Last updated: 2026-04-23. Edit as phases complete.*
