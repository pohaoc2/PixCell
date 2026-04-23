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

### Figures that already exist (on remote `ec2-user`)

| File | Usable for | Modification needed |
|------|------------|---------------------|
| `/home/ec2-user/PixCell/figures/dataset_metrics_filtered.png` | F3 (tradeoff) | reformat axes, add Pareto front |
| `/home/ec2-user/PixCell/inference_output/cache/512_9728/leave_one_out_diff.png` | F2 (visibility map inset) | crop, relabel |
| `/home/ec2-user/PixCell/inference_output/cache/512_9728/leave_one_out_diff_stats.json` | F2 (bar chart data) | aggregate across tiles |
| `/home/ec2-user/PixCell/inference_output/channel_sweep/exp1_microenv/cancer_12800_16384.png` | F6 (microenv sweep) | extend grid from 1D to 3D |
| `/home/ec2-user/PixCell/inference_output/channel_sweep/exp2_cell_type_relabeling.png` | S2 (coarse cell_types baseline) | keep as-is |
| `/home/ec2-user/PixCell/inference_output/channel_sweep/exp3_cell_state_relabeling.png` | F6 (state axis exemplar) | keep + extend |
| `/home/ec2-user/PixCell/inference_output/unpaired_ablation/dataset_metrics_filtered.png` | F3 (unpaired panel) | pair with paired version |
| `/home/ec2-user/PixCell/inference_output/unpaired_ablation/ablation_results/29952_46080/ablation_grid.png` | S1 (ablation grid supp) | keep, verify labels |
| `/home/ec2-user/PixCell/inference_output/unpaired_ablation/leave_one_out/29952_46080/leave_one_out_diff.png` | F2 (unpaired inset) | crop |
| `/home/ec2-user/PixCell/inference_output/unpaired_ablation/channel_sweep/cache/exp1_microenv/cancer_22528_34304/*.png` | F6 (microenv continuous axis) | combine with glucose axis |

### Figures locally available (WSL)

| File | Usable for | Notes |
|------|------------|-------|
| `dataset_metrics_all.png` | S3/S4 | full per-metric boxplot |
| `dataset_metrics_option_a.png` | possibly F3 | subset variant |
| `tmp_option_a_ref_crop*.png` | qualitative inset | crops, check labels |
| `tmp_option_a_mid_crop.png` | qualitative inset | single-tile example |

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

1. Are raw CODEX marker intensities available per tile (for D4)? Need to confirm in ORION-CRC33 data contract.
2. Which journal target? (Nature Methods, Nat Biomed Eng, Cell Systems, Med Image Anal) — affects scope and figure budget.
3. Fine-grained cell_types panel: which 5–8 lineage markers to prioritize?
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

*Last updated: 2026-04-22. Edit as phases complete.*
