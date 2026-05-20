# Figures Storyline (post-update, May 2026)

Source figures: `/home/ec2-user/PixCell/figures/pngs_updated/` and `/home/ec2-user/PixCell/inference_output/a1_concat/a4_uni_probe/figures/`.

This document maps each updated figure to (i) the question it answers, (ii) the main quantitative/qualitative deliverable, (iii) its role in the overall storyline, and (iv) suggestions to tighten the story without introducing new concepts.

---

## Storyline (one paragraph)

H&E is cheap and ubiquitous; multiplex (MX/CODEX) is rich but rare. We ask: **how much of the MX-defined TME is morphologically encoded in H&E, and can we build a generator that respects that encoding while supplying what H&E cannot?** Fig 1 (cell assignments + decodability) defines the boundary — broad cell categories and aggregate TME signals are decodable from H&E, but per-marker intensities and rare populations are not, motivating MX-derived conditioning. Fig 2 (architecture ablations) justifies the multi-group ControlNet + UNI design by showing alternative encoder combinations are unstable or strictly worse. Fig 3 (per-channel quant + qual) demonstrates that the realized model uses each channel group differently and that the effects are visible. Fig 4 (UNI×TME 2×2) decomposes the contributions: UNI anchors realism/style, TME anchors nuclear geometry; the two are redundant on realism (large +502 FUD interaction — either alone recovers most of the gain) but independent on geometry (PQ/DICE interactions ≈ 0), validating the dual-conditioning design. Fig 4.5 (UNI probe) opens the UNI black box, showing that biologically and morphologically meaningful axes are linearly readable from UNI but distributed across many directions, so projecting any single direction out does not collapse the appearance — explaining why TME and UNI cannot trivially substitute. Fig 5 (combinatorial grammar) closes the loop: holding UNI fixed and walking TME states reveals that the same cell types render different morphologies under different microenvironment conditions, giving a use case for in-silico TME perturbation. Net arc: **what H&E can/can't say (1) → why this architecture (2) → what each channel does (3) → how UNI and TME divide labor (4) → what UNI actually encodes (4.5) → what the model enables (5).**

---

## Figure 1 — Decoding Boundary

### 1A. `cell_assignment.png`
- **Question.** Are the broad cell-type/state assignments derived from CODEX biologically coherent (separable morphology + protein expression)?
- **Deliverable.**
  - Cancer / immune / healthy show distinguishable area + circularity (Mann–Whitney p<0.001 across class pairs).
  - Lineage markers light up expected boxes (E-cad/Pan-CK→cancer; CD45/CD3e/CD4→immune; Hoechst-only→healthy/stromal).
  - Ki67 co-varies with proliferative state across all three cell types — `cell_state` channel carries non-redundant biology.
  - Class composition: healthy 44% / cancer 42% / immune 14% / dead 0.27% — justifies per-group dropout + channel weighting.
- **Role in story.** Establishes that the conditioning channels are not label noise; they index real, separable populations.

### 1B. `07_inverse_decoding.png`
- **Question.** What information in CODEX is decodable from H&E alone (i.e. what would conditioning on MX add beyond what an H&E foundation encoder already captures)?
- **Deliverable.**
  - (A) Aggregate TME channels (cell-type density, cell-state fractions, tissue regions): UNI-2h ≈ Virchow2 > CTransPath > ResNet-50; high R² on cancer/healthy density and proliferative fraction.
  - (B) Per-marker intensities: most markers are *not* recoverable from any H&E encoder; many rare/specific markers (e.g. FOXP3) trend negative. `dead` ~0 R² consistent with 0.27% prevalence.
- **Role in story.** Defines the "decoding boundary." Aggregate phenotype is in H&E; molecular specificity is not. This motivates why the generator must take MX as input rather than re-deriving it from H&E.

### Narrative hand-off
1A says "the labels are real biology"; 1B says "H&E alone partially recovers aggregates, fails on specifics." Together they bound what conditioning is *necessary* for.

---

## Figure 2 — Architecture Ablation (`SI_A1_A2_unified.png`)

- **Question.** Why this architecture? Does the multi-group TME module + UNI co-conditioning beat alternative encoder combinations and remain trainable?
- **Sections (one unified figure built from `SI_A1_A2_section{1..4}.png`).**
  - Section 1 — training-stability curves across encoder/conditioning variants.
  - Section 2 — per-variant generation metrics (FUD/LPIPS/PQ/DICE/HED).
  - Section 3 — qualitative tile comparison across variants.
  - Section 4 — sensitivity / robustness sweep.
- **Deliverable.** Identifies the chosen design (per-group CNN encoders + cross-attention to mask Q + zero-init residuals + UNI co-conditioning) as Pareto-best across stability and quality. Alternatives (single-encoder concat, naive ControlNet, no UNI, no zero-init) are dominated either on stability (training divergence / loss variance) or on quality (worse FUD or PQ).
- **Role in story.** Removes the "did you try X?" reviewer question for the headline architecture decision before any per-channel claim is made.

---

## Figure 3 — Per-Channel Effect (Quant + Qual)

### 3A. `fig_paired_unpaired_performance.png` (quant)
- **Question.** What is the marginal effect of each channel group on H&E quality, in paired (with reference H&E) and unpaired (TME-only) inference?
- **Deliverable.**
  - (A) 32 on/off combinations × 5 metrics; paired (▲) consistently beats unpaired (■).
  - (B) Top-3/bottom-3 per metric per regime — Cell-state (CS) and Nutrient (NU) groups dominate PQ/DICE in both regimes.
  - (C) Per-group effect-size heatmaps (FUD/LPIPS/HED display with sign convention: **negative = improvement**, positive = worse):
    - **CS:** ΔPQ +0.32 paired / +0.32 unpaired; ΔDICE +0.32 paired / +0.32 unpaired.
    - **NU:** ΔPQ +0.24 paired / +0.17 unpaired; ΔDICE +0.23 paired / +0.13 unpaired; **biggest single contributor to FUD improvement** (ΔFUD −31.68 paired / −22.78 unpaired) and the only group that meaningfully moves HED (−0.11 both regimes).
    - **CT:** comparable to CS on segmentation (ΔPQ +0.30 paired / +0.28 unpaired; ΔDICE +0.30 paired / +0.26 unpaired); small on FUD/HED.
    - **VA:** small but non-zero in paired (ΔPQ/ΔDICE ≈ +0.11, ΔFUD −8.99); collapses to ≈0 in unpaired.
    - **FUD direction flip (paired→unpaired):** CS −4.37 → +11.39; CT −8.25 → +7.19. Adding these groups *improves* FUD when a style anchor (UNI) is available but *worsens* it without one — consistent with the dual-conditioning interpretation (Fig 4): TME geometry alone over-constrains color/texture without UNI to absorb it. NU and VA do **not** flip (both stay improvements or near-zero).
- **Role in story.** Quantifies which channels carry usable signal and exposes the regime-dependence of "more channels ⇒ better."

### 3B. `fig_ablation_grids.png` (qual)
- **Question.** Does the visual quality track the metric ranking?
- **Deliverable.** Two 4×4 PQ-sorted grids (paired top, unpaired bottom) for one representative tile each; per-tile bars for LPIPS/PQ/DICE/HED beside each cell. CS-off cells consistently low PQ in both regimes; paired holds stain/structure further down the rank list.
- **Role in story.** Anchors the metric-driven claims to visual reality so a reviewer can eyeball the metric ↔ image correspondence.

### Hand-off
Fig 3 says *which* channels move the needle and *how much*. Fig 4 explains *why* the ones that look like they should move the needle (UNI vs TME) divide their labor the way they do.

---

## Figure 4 — UNI × TME Decomposition (`08_uni_tme_decomposition.png`)

- **Question.** Are UNI (global style) and TME (spatial biology) complementary, redundant, or competing?
- **Deliverable.**
  - 2×2 factorial (UNI on/off × TME on/off) × 5 metrics, with formal main-effect + interaction terms.
  - Raw main-effect magnitudes (Y_only − Y_neither):
    - **UNI effect:** FUD −604, LPIPS −0.33, PQ +0.05, DICE +0.12, HED −0.45 → dominates style/realism axes (FUD, LPIPS, HED), tiny on segmentation.
    - **TME effect:** FUD −551, LPIPS −0.30, PQ +0.76, DICE +0.84, HED −0.42 → dominates nuclear geometry (PQ/DICE) while also recovering most of the realism gain on its own.
  - **FUD interaction = +502** → strongly sub-additive: adding both factors recovers only Y_both ≈ 133 against an additive prediction of Y_uni + Y_tme − Y_neither ≈ −369 (i.e. each factor alone already collapses most of the realism gap, so they are highly redundant for FUD).
  - **PQ interaction = −0.07, DICE = −0.14** → small relative to TME's main effect (~+0.8); TME's segmentation contribution is essentially independent of UNI, validating TME-only inference for cell-counting tasks.
  - **LPIPS/HED interactions** (+0.18 / +0.28) are also small relative to main effects, reinforcing redundancy on realism.
  - Unconditional baseline (Y_neither: FUD 785, LPIPS 0.81, PQ 0.04, DICE 0.04, HED 0.64) collapses to texture noise → all measured effects are real signal.
- **Role in story.** Headline justification for the dual-conditioning design and the design choice that lets the generator be used in two modes (paired with style anchor; unpaired for TME perturbation).

---

## Figure 4.5 — UNI Probe: What UNI Encodes (`a4_uni_probe/figures/`)

Five panels (`panel_a..panel_e`) plus tables in `probe_results.csv`, `appearance_sweep_summary.csv`, `appearance_null_summary.csv`.

- **Question.** What biological/morphological features are linearly readable from UNI features, and are they concentrated in single directions?
- **Deliverables.**
  - **Panel A — probe R² per attribute.** Linear ridge probes from UNI to per-tile attributes (cell-type fractions, nuclear morphology, stain statistics, Haralick texture). High R² on texture and stain stats (e.g. `texture_h_homogeneity` R² 0.91, `texture_h_energy` 0.86, `texture_h_contrast` 0.81); cell-fraction attributes also recoverable. UNI vs TME-feature comparison shown side-by-side.
  - **Panel B — sweep slope.** Walking along the probe direction in UNI space and re-generating: appearance metrics shift monotonically with sweep magnitude → directions are causal handles, not just correlates.
  - **Panel C — null drop.** Projecting one probe direction out (`targeted`) ≈ projecting out a random direction (`random`) ≪ zeroing the whole UNI (`full_uni_null`). Single-axis edits do not collapse appearance; full-UNI-null shifts stain angle from ~3° → ~10–13°, collapses eosin, flattens H-texture.
  - **Panel D — appearance sweep, all metrics.** Same sweep as B across stain/texture metric panel — confirms multi-axis effect.
  - **Panel E — appearance under full UNI null, all metrics.** Documents the full-null collapse pattern.
- **Role in story.** Opens the UNI black box: UNI carries decodable biological/appearance signal but distributes it across many axes. This explains *why* UNI is required despite TME conditioning (TME can't recover the distributed appearance basis) and *why* UNI alone can't substitute for TME (no spatial geometry handle).

---

## Figure 5 — Combinatorial Grammar (`09_combinatorial_grammar.png`, anchors in SI)

- **Question.** When the same cell types are placed under different microenvironment conditions, do the generated morphologies change in biologically interpretable ways?
- **Deliverable.** Systematic on/off sweep across channel groups while holding cell-type layout fixed; cluster generated morphologies and map to known phenotypes (e.g. proliferative cancer core, hypoxic boundary, immune-excluded edge). Anchors figure (`SI_09_combinatorial_grammar_anchors.png`) extends to rare combinations and dose-response.
- **Role in story.** Concrete use case — in-silico TME perturbation. Demonstrates the model is not just a quality-controlled image generator but a tool for asking "what would this tissue look like under condition X?"

---

## How the figures connect (one-liners)

1. **Fig 1 → Fig 2:** because per-marker MX is invisible to H&E, we need a generator that takes MX as input — and that generator must be defensible architecturally.
2. **Fig 2 → Fig 3:** with the architecture justified, what does each channel actually do?
3. **Fig 3 → Fig 4:** Fig 3 mixes UNI + TME effects in the paired regime; Fig 4 disentangles them formally.
4. **Fig 4 → Fig 4.5:** Fig 4 says UNI and TME divide labor; Fig 4.5 explains *what* UNI encodes that TME can't substitute.
5. **Fig 4.5 → Fig 5:** because TME has independent control over geometry (Fig 4) and UNI anchors style across many distributed axes (Fig 4.5), holding UNI fixed and varying TME yields interpretable in-silico perturbations.

---

## Suggestions to tighten without adding concepts

These are zero-new-concept moves; each reuses what is already in the figures or tables.

### S1. Make the decoding-boundary explicit (Fig 1B → Fig 3 link)
Current Fig 1B reports R² per marker; current Fig 3 reports per-channel ablation impact. **Add a single scatter** as a small sub-panel inside Fig 1B (or as the last panel of Fig 3): per-channel decodability (R²) on x-axis vs. per-channel forward-LOO impact (ΔPQ or ΔFUD) on y-axis. One point per channel. The expected anti-correlation ("channels invisible to H&E are the ones the generator most needs") makes the forward-and-inverse story a single sentence in the paper rather than two sections. No new experiments required — both axes already exist.

### S2. Promote the UNI×TME interaction terms to bold callouts in Fig 4
The current heatmap buries the **+502** FUD interaction (additive prediction undershoots by 502 → strong redundancy for realism) and the small PQ/DICE interactions (−0.07/−0.14, near-zero vs TME main effect ≈ +0.8). Adding a single annotated arrow per row ("redundant for realism" / "independent for geometry") turns a numerical table into the figure's takeaway. No new data.

### S3. Re-use Fig 4.5 panel C inside Fig 4 as an inset
Fig 4 establishes UNI ≠ TME at the metric level. Fig 4.5 panel C (targeted ≈ random ≪ full-UNI-null) is a different test of the same claim — the appearance space cannot be collapsed by removing a single UNI axis. Putting one bar from panel C as an inset in Fig 4 keeps Fig 4.5 self-contained while making Fig 4's "UNI is doing distributed work" claim immediately visible.

### S4. Tie Fig 5 anchors to Fig 4.5 sweep directions
Fig 5 already shows TME-driven morphology changes under fixed UNI. If any of the Fig 4.5 probe directions correspond to a phenotype shown in Fig 5 (e.g. nuclei-density direction matches the proliferative-core panel of Fig 5), label that correspondence on Fig 5 and reference Fig 4.5. This stitches the use-case to the mechanism without new experiments.

### S5. Drop or demote the Vasculature group everywhere
Fig 3 shows VA is the **smallest** contributor: paired ΔPQ/ΔDICE ≈ +0.11 (about a third of CS/CT/NU) and ΔFUD −8.99; unpaired VA collapses to ≈0 across all metrics. Carrying it through Fig 4/5 dilutes the per-channel claims. Two options:
- (a) Fold VA into Fig 1B/3 as an explicit "small effect" result ("the channel we expected to matter most for tissue architecture is the weakest contributor — H&E already encodes vasculature geometry, which Fig 1B confirms via high R² on cancer/healthy density"). The unpaired-VA-≈0 result strengthens the claim.
- (b) Remove VA from the on/off matrix in Fig 3A, freeing one bit of dot-encoding for clearer rows.

### S6. Use a single cohesive panel layout for Fig 1
Right now Fig 1 has two figures (`cell_assignment.png` + `07_inverse_decoding.png`) that share a question. Combining them as Fig 1A (cell assignments + markers) and Fig 1B (decoding R² aggregate) and Fig 1C (decoding R² per marker) gives the reader one coherent "decoding boundary" figure rather than two. No new content, just composition.

### S7. Order Fig 3 and Fig 4 callouts so paired-regime effects in Fig 3 are *predicted* by Fig 4
Currently Fig 3's regime-flip on FUD for CS (−4.37 paired → +11.39 unpaired) and CT (−8.25 → +7.19) reads as a surprise. Fig 4's "UNI anchors realism, TME anchors geometry" makes this expected: without UNI, adding CT/CS geometry shifts color/texture away from the real-H&E manifold. Reordering the prose so the FUD flip in Fig 3 is framed as a prediction of Fig 4 (cite-forward) — or just stating it parenthetically — converts a confusing regime-dependence into evidence for the dual-conditioning design. Note that NU does **not** flip (improves FUD in both regimes), consistent with NU acting as a non-geometric texture-modulating signal that benefits realism with or without UNI.

### S8. Cap each figure with a one-line "What this rules out"
Reviewers love negative claims. For each main figure:
- F1: rules out "H&E already encodes everything MX does."
- F2: rules out "any reasonable encoder combination would work."
- F3: rules out "all channels matter equally" and rules in "CS+NU are the load-bearing groups."
- F4: rules out "UNI and TME duplicate each other."
- F4.5: rules out "UNI = a single style direction" and rules in "UNI = distributed appearance basis."
- F5: rules out "the model is just memorizing H&E patches."
These do not lengthen the paper materially but make every figure earn its slot.

---

## What I deliberately did NOT add
- No new experiments, encoders, datasets, or metrics.
- No pathologist-validation or second-cohort suggestions (those are scope-expanding and addressed in `STORYLINE.md` already).
- No reframing of the paper as methods-only or biology-only — kept the dual framing because both Fig 4.5 (mechanism) and Fig 5 (use case) are present.
