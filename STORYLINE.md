# Storyline — MultiChannel PixCell ControlNet

## Central Gap
Which aspects of cellular composition, cell state, and TME are reflected in H&E morphology — and how do these underlying biological factors **collectively** give rise to the emergent phenotypes pathologists recognize? H&E is cheap and ubiquitous; MX is rich but rare. No principled framework links them at the level of generative morphology.

---

## 1. Paired H&E ↔ MX data generation

**Question.** Can we build a faithful paired H&E + MX dataset at scale, so model training is grounded in real co-registered biology rather than simulation?

**Gap.** Sim-only pipelines lack ground-truth morphology; existing MX datasets lack co-registered H&E or are too small for diffusion training.

**Figures.**
- Main: `figures/pngs/methods.png` — registration + channel extraction pipeline schematic.
- SI: `figures/pngs/cell_summary_figure.png` — per-tile cell-type/state distributions, channel coverage, registration QC.

---

## 2. Channels → H&E: does the model use MX signal?

**Question.** Which channels carry morphologically decodable signal, and does adding them monotonically improve generation?

**Gap.** Prior ControlNet work conditions on coarse masks; nobody has done systematic per-group ablation on MX channels driving H&E synthesis.

**Figures.**
- Main quant: `figures/pngs/01_metric_tradeoffs.png` (FID / UNI-cos vs. channel set), `figures/pngs/02_paired_vs_unpaired.png` (paired vs. unpaired generalization), `figures/pngs/03_channel_effect_sizes.png` (per-channel effect sizes).
- Main qual: one representative tile from `figures/pngs/05_paired_ablation_grid.png`.
- SI qual: `figures/pngs/05_paired_ablation_grid.png` (paired), `figures/pngs/06_unpaired_ablation_grid.png` (unpaired).

---

## 2.5. UNI ↔ TME interaction

**Question.** Are UNI (global style/context) and TME (spatial biology) complementary, redundant, or competing?

**Gap.** Unclear whether TME conditioning duplicates information already in UNI, or fills a distinct spatial axis.

**Figure.** Main: `figures/pngs/08_uni_tme_decomposition.png` — 2×2 ±UNI×±TME ablation, decomposes appearance vs. structure.

---

## 3. Leave-One-Out: causal contribution of each channel

**Question.** For a given tile, which pixels does each channel control?

**Gap.** Aggregate metrics hide local effects. Need spatially resolved attribution.

**Figures.**
- Main qual: `figures/pngs/04_leave_one_out_impact.png` — LOO tile panels.
- SI quant: `figures/pngs/leave_one_out_diff.png` — per-channel pixel-level impact with generated H&E, ΔE-CIELAB magnitude, 1−SSIM structural loss, and inside/outside causal-ratio strips on shared row scales.
- Stats: `leave_one_out_diff_stats.json` now keeps legacy `mean_diff`/`max_diff` fields and adds `delta_e_mean`, `delta_e_p99`, `ssim_loss_mean`, `ssim_loss_p99`, `causal_inside_mean_dE`, `causal_outside_mean_dE`, `causal_ratio`, and nullable `uni_cosine_drop`.

---

## 4. Inverse direction: H&E → MX prediction

**Question.** Given the model encodes H&E↔MX coupling, how much MX signal is **decodable** from H&E alone?

**Gap.** Generative quality alone doesn't prove information sufficiency — need a discriminative readout.

**Figure.** Main: `figures/pngs/07_inverse_decoding.png` — per-channel R²/AUROC predicting MX from H&E (UNI) embeddings. Decodable channels = morphologically visible biology; non-decodable = invisible (e.g. specific molecular states).

---

## 5. Use case — combinatorial grammar of MX → morphology

**Question.** Which **combinations** of MX channels jointly produce specific H&E phenotypes (TLS, immune-excluded edge, necrotic core)?

**Gap.** Single-channel attribution misses interactions; pathologist phenotypes emerge from co-occurring factors.

**Figures.**
- Main: `figures/pngs/09_combinatorial_grammar.png` — systematic on/off sweep across channel groups, cluster morphologies → known phenotypes.
- SI: `figures/pngs/SI_09_combinatorial_grammar_anchors.png` — extended grammar (rare combinations, dose-response, failure modes).

---

## Narrative arc
data (1) → model learns coupling (2, 2.5) → causal per-channel (3) → information sufficiency (4) → emergent grammar / use case (5).

---

# Reviewer (Nat Comm) — Honest Weakness Assessment

## Story-level

**Overall.** Coherent arc but currently reads as **"we built a controlled generator + ablations,"** not **"we discovered how MX maps to H&E."** Generation quality is a *means*; the paper needs a clearer scientific claim. Right now the use case (Section 5) carries the biological-discovery burden alone — that's risky for Nat Comm. Either elevate Section 5 (with validated phenotype claims) or reframe the paper as a methods paper (better fit: Nature Methods / Comm Biol).

## W1. Single-cohort, single-platform validation
- ORION-CRC only. One tissue, one MX platform, one staining lab.
- **Reviewer ask:** repeat on a second cohort (e.g. CODEX/MIBI on a different tumor) or at minimum a held-out site/scanner. Without it, the "grammar" claim doesn't generalize.

## W2. "Decodable from H&E" claim needs a pathologist
- Section 4's R²/AUROC alone won't convince. Reviewer wants: blinded pathologist scoring of generated tiles, agreement on key phenotypes (necrosis, TLS, immune infiltration), and inter-rater κ.
- Without humans in the loop, "morphologically visible biology" is just a model claim about itself.

## W3. Causal language is overreached
- LOO ablation shows **counterfactual sensitivity**, not causation. The model could learn shortcuts (e.g. coarse spatial priors) that correlate with channels.
- **Reviewer ask:** swap-tests (replace channel A with a permuted/randomized version vs. a mismatched-tile version), and shape-randomized controls. Without these, causal claim is fragile.

## W4. Combinatorial grammar — risk of post-hoc storytelling
- Section 5 sweeps and clusters morphologies, then maps to known phenotypes. Reviewer will ask: were phenotype labels pre-registered? Or did you cherry-pick clusters that look like TLS/necrosis?
- **Reviewer ask:** pre-registered phenotype list, prospective hold-out sweep, quantitative match score against pathologist annotations, **and** at least one *novel* (i.e. not previously described) MX-combination → morphology hypothesis with experimental or external-cohort validation.

## W5. Generative-model evaluation is shallow
- FID + UNI-cosine are global. They don't measure cell-level fidelity (right number of lymphocytes? correct nuclear morphology?).
- **Reviewer ask:** cell-detection-based metrics (HoVer-Net or similar): cell count, cell-type composition, nuclear morphology distribution match between generated and real.

## W6. Comparison to baselines is missing
- No comparison to: (a) image-to-image translation baselines (pix2pix-HD, CUT), (b) other ControlNet conditioning schemes (concat-only, no TME module), (c) recent histology diffusion models.
- Without these, "PixCell ControlNet + multi-group TME" is unfalsified as the right design.

## W7. UNI is doing a lot of work — and it's not yours
- Heavy reliance on a frozen UNI-2h encoder. If UNI fails on your cohort (different scanner, stain), the model fails. CFG dropout helps but doesn't solve it.
- **Reviewer ask:** stain-augmentation ablation, scanner-shift robustness, and an ablation with a non-UNI encoder (CTransPath, retrained ResNet) to show the framework not the encoder is the contribution.

## W8. Channel set is small and curated
- 4 groups, ~7 channels. Real CODEX panels have 30–60 markers. Story under-sells what's hard: which channels matter, why these, and how to scale.
- **Reviewer ask:** show the framework extends to a denser panel; quantify diminishing returns vs. channel count.

## W9. Inverse decoding (Section 4) and forward generation (Section 2) aren't tied together
- They feel like two separate experiments. The paper would be stronger if decodability **predicts** which channels matter for generation (or vice versa).
- **Reviewer ask:** correlate per-channel decodability (R²) with per-channel LOO impact (ΔE / SSIM). High correlation = unified information-theoretic story.

## W10. Reproducibility
- Paired ORION dataset access, exact tile IDs, registration QC thresholds, train/val/test split — must all be released. Diffusion models are notoriously irreproducible from text descriptions; share code + checkpoints + manifests.

## W11. `leave_one_out_diff.png` (current) under-delivers
- As-is, the figure shows raw RGB |Δ|; reviewer can't separate color from morphology shifts and can't tell if signal is local-to-channel-mask. Plan in `LOO_DIFF_CHANGE_PLAN.md` addresses this — must land before submission.

## What would push it from Comm Biol → Nat Comm
1. Second cohort or platform (W1).
2. Pathologist-blinded evaluation (W2).
3. ≥1 novel, prospectively validated MX→morphology hypothesis (W4).
4. Cell-level fidelity metrics (W5).
5. Forward↔inverse unification (W9).

Without ≥3 of these, this is a strong Comm Biol / Nature Methods paper, not Nat Comm.

---

# Methods-Paper Reframe — Where More Analysis Is Needed

Reframing as a methods paper (Nat Methods / Nat BME / Comm Biol) shifts the bar from biological discovery to **rigor of the framework**. The story becomes: *a controllable, attributable H&E generator conditioned on multiplex biology, with a unified forward/inverse evaluation suite.* Below are the missing analytical blocks ordered by priority.

## A. Design-justification ablations (P0 — biggest current gap)

A methods paper must prove **every architectural choice was necessary**. Several are currently asserted, not tested. No figure exists for any of A1–A6 yet.

- **A1. Multi-group TME vs. naive concat ControlNet.** (i) single-encoder all-channels concat, (ii) per-channel encoders no grouping, (iii) current grouped design. Same compute budget. Metrics: FID, UNI-cos, cell-level fidelity, training stability. → new figure `SI_A1_tme_design.png`.
- **A2. `zero_mask_latent=True` post-TME subtraction.** With vs. without subtraction; bypass-path probe (zero TME, leak only mask latent — does H&E recover?). → `SI_A2_bypass_probe.png`.
- **A3. Zero-init residual gating.** Ablate; show training-stability curves (loss variance, divergence rate across seeds). → `SI_A3_zero_init.png`.
- **A4. CFG dropout sweep on UNI.** {0, 0.1, 0.15, 0.3, 0.5}; TME-only vs. paired quality tradeoff. → `SI_A4_cfg_sweep.png`.
- **A5. Per-group dropout schedule.** Currently configured but never ablated. Sweep. → `SI_A5_group_dropout.png`.
- **A6. TME injection mechanism.** Cross-attention vs. spatial broadcast vs. FiLM. → `SI_A6_injection.png`.

## B. Robustness / generalization (P2 — lifts impact)

- **B1. Scanner shift.** Held-out scanner; or Macenko/Vahadane stain-augment stress test at increasing severity.
- **B2. Tissue shift.** ≥1 OOD tissue (qualitative is acceptable).
- **B3. Channel-noise injection.** Gaussian noise/dropout on MX channels at inference; degradation curve. Tells users how clean MX must be.
- **B4. Resolution invariance.** Train at 256, infer at 512; or ½×/2× resampled MX.
- **B5. Sample-size scaling.** Train on {25, 50, 100%} of paired tiles; learning curve. Tells users how much paired data they need.

## C. Cell-level fidelity (P0 — replaces FID-only evaluation)

FID + UNI-cos are insufficient.

- **C1. HoVer-Net (or CellViT) on real vs. generated.** Per-cell-type count match, nuclear-morphology distribution KS-distance, Ripley-K spatial agreement.
- **C2. Conditional cell-type accuracy.** When cell-type channel says "lymphocyte at (x,y)," is there a lymphocyte-shaped nucleus there in the generated H&E?
- **C3. Tile-level pathology classifier transfer.** Train on real H&E (tumor vs. stroma); evaluate on generated H&E. Performance gap = methods metric.

## D. Causal attribution rigor (P1 — extends Section 3)

LOO alone isn't enough.

- **D1. Shuffle/swap controls.** Per channel: (a) zero-out (current LOO), (b) spatial shuffle within tile, (c) channel from a *different* tile. Distinguishes "channel content matters" from "channel presence matters."
- **D2. Counterfactual local edits.** Add an immune cell at (x,y); show generated H&E changes specifically and locally. Most convincing experiment for the "controllable" claim.
- **D3. LOO redesign quantitative summary.** Per the new ΔE / 1−SSIM / inside-outside ratio (already in `LOO_DIFF_CHANGE_PLAN.md`), plot each channel as a point in (color-shift, structure-shift, locality) space.

## E. Forward ↔ inverse unification (P2 — cheap, high payoff)

Make Sections 2 and 4 a single argument:

- **E1. Per-channel forward LOO impact vs. inverse decodability (R²)**, scatter, one point per channel. Strong correlation = framework recovers the information-theoretic structure of the H&E↔MX mapping. Pure methods contribution, no biology claim required. → new main-text figure.

## F. Baselines (P1 — cannot skip)

- **F1.** pix2pix-HD or CUT on the same paired data.
- **F2.** SD3.5 ControlNet with naive channel concat (no TME module, no UNI).
- **F3.** Frozen-PixCell with a learned linear adapter on channels (cheap baseline).
Compare on FID + UNI-cos + cell-level fidelity + LOO causal-ratio.

## G. Compute / practical reporting (P3 — required, mechanical)

- Train cost (GPU-hours), per-tile inference latency, memory footprint.
- Throughput vs. tile size and channel count.
- Failure-mode catalog: out-of-range MX, missing channels, very sparse cells.

## Priority Triage

| Priority | Block | Why |
|---|---|---|
| P0 | A (design ablations) | Methods-paper minimum bar; biggest current gap |
| P0 | C (cell-level fidelity) | Replaces shallow FID story |
| P1 | D (causal rigor + LOO redesign) | LOO redesign already planned; finish + add D1/D2 |
| P1 | F (baselines) | Reviewers will block on this |
| P2 | B (robustness/generalization) | Strongly elevates impact |
| P2 | E (forward↔inverse unification) | Cheap, very high payoff |
| P3 | G (compute reporting) | Required but mechanical |

**Most underdeveloped: A and C.** Existing Sections 2/2.5/3/4/5 give *what the model does*; A and C give *why this design is right and how good it really is*. Without those, a methods reviewer will say "interesting tool, insufficient methodological evidence."

## Methods-paper revised story arc

data (1) → **design justified** (A) → model learns coupling (2, 2.5) → **fidelity quantified at cell level** (C) → causal per-channel + controls (3 + D) → **information sufficiency unified with forward generation** (4 + E) → robustness & baselines (B, F) → use case (5, demoted to "demonstration" rather than headline).
