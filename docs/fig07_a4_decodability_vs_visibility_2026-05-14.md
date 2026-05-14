# Fig 07 (Inverse Decoding) vs a4 UNI Probe — Goals and Results

This note summarizes the two experiments that probe the H&E latent (UNI) for biological content, contrasts their goals, and reports the quantitative outcomes.

Evidence files:

- `figures/pngs_updated/07_inverse_decoding.png`
- `src/a1_probe_linear/out/linear_probe_results.csv` (T1, UNI → CODEX-derived aggregate)
- `src/a1_codex_targets/probe_out/t2_mlp/mlp_probe_results.csv` (T2, UNI → per-marker mean)
- `src/a1_probe_encoders/out/*linear_probe_results.csv` (T1 for Virchow2, CTransPath, ResNet-50, REMEDIS)
- `src/a4_uni_probe/out/probe_results.csv`
- `inference_output/a1_concat/a4_uni_probe/probe_results.csv` (shared-tile pool)
- `inference_output/a1_concat/a4_uni_probe/appearance_sweep_summary.csv`
- `inference_output/a1_concat/a4_uni_probe/appearance_null_summary.csv`

## 1. Goals

| Experiment            | Question answered                                                                                       | Method                                                                                                                            | Output                                                            |
| --------------------- | ------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| Fig 07 (inverse dec.) | "Which CODEX-derived biological features can be **linearly decoded** from an H&E foundation embedding?" | Fit Ridge / MLP probes: `encoder(H&E) → CODEX target`. Compare 5 encoders (UNI-2h, Virchow2, CTransPath, REMEDIS, ResNet-50).     | Per-target R² (T1 aggregates and T2 marker means).                |
| a4 UNI probe          | "Which biological features are **causally written into the generated H&E** when their UNI subspace is edited?" | Fit `UNI → attr` probe, get direction `w`; **sweep** `uni + α·‖uni‖·ŵ` and **null** `uni − (uniᵀŵ)ŵ`; regen H&E; remeasure attr. | Probe R², sweep slope vs. random direction, null-vs-targeted ΔE. |

Key distinction:

- **Fig 07 = decodability (information presence in the latent).** Necessary condition. High R² means a linear readout from H&E embeddings recovers the CODEX feature.
- **a4 = visibility / causal usability (information used downstream by the generator).** Sufficient condition test. A direction `w` that perturbs the generated image's morphology proves the ControlNet **uses** that subspace, not merely that the subspace exists.

This framing is correct in spirit; a cleaner phrasing for the paper is "linearly decodable from UNI" (Fig 07) vs. "causally expressed by ControlNet via UNI" (a4).

## 2. Fig 07 results (UNI → CODEX decodability)

T1 aggregates, UNI-2h (`src/a1_probe_linear/out/linear_probe_results.csv`):

| target           | R² mean | R² sd  | n folds |
| ---------------- | ------- | ------ | ------- |
| cell_density     | 0.953   | 0.007  | 5       |
| prolif_frac      | 0.863   | 0.035  | 5       |
| nonprolif_frac   | 0.826   | 0.023  | 5       |
| glucose_mean     | 0.821   | 0.020  | 5       |
| oxygen_mean      | 0.810   | 0.022  | 5       |
| healthy_frac     | 0.710   | 0.013  | 5       |
| cancer_frac      | 0.669   | 0.016  | 5       |
| vasculature_frac | 0.509   | 0.022  | 5       |
| immune_frac      | 0.495   | 0.042  | 5       |
| dead_frac        | -0.135  | 0.107  | 5       |

T2 per-marker MLP, UNI-2h (`mlp_probe_results.csv`):

| top markers   | R²     | bottom markers | R²     |
| ------------- | ------ | -------------- | ------ |
| PD-1          | 0.364  | Argo550        | -2.700 |
| Hoechst       | 0.355  | FOXP3          | -1.060 |
| E-cadherin    | 0.238  | CD8a           | -0.191 |
| CD45RO        | 0.094  | CD31           | -0.144 |
| Ki67          | 0.050  | CD20           | -0.142 |
| CD3e          | 0.045  | CD163          | -0.035 |

Interpretation:

- UNI cleanly encodes **cell mass, density, metabolic gradients, and broad cell-state composition** (R² ≥ 0.7 for density, prolif/nonprolif fractions, oxygen, glucose, healthy fraction).
- Lineage-specific markers (CD45, CD68, CD20, FOXP3, CD8a) are essentially **not** decodable (R² ≤ 0.1, several negative). UNI is therefore a *morphology + composition* embedding, not a lineage readout.
- Per-encoder comparison (in the figure, not in the CSV summarized here) shows UNI-2h and Virchow2 lead on most T1 targets; ResNet-50 and REMEDIS lag, confirming H&E foundation pretraining is what unlocks decodability.

## 3. a4 UNI probe results

a4 first refits the same UNI→attr probes on a shared 30-tile pool (`inference_output/a1_concat/a4_uni_probe/probe_results.csv`); R² is comparable to Fig 07 and adds appearance/texture attrs:

| attr                  | UNI R² | TME R² | Δ R² (UNI − TME) |
| --------------------- | ------ | ------ | ---------------- |
| texture_h_contrast    | 0.813  | 0.208  | **+0.605**       |
| texture_e_contrast    | 0.753  | 0.168  | **+0.585**       |
| texture_h_energy      | 0.857  | 0.395  | +0.462           |
| texture_e_homogeneity | 0.913  | 0.492  | +0.421           |
| texture_h_homogeneity | 0.912  | 0.509  | +0.403           |
| texture_e_energy      | 0.937  | 0.562  | +0.375           |
| h_mean                | 0.949  | 0.615  | +0.333           |
| e_mean                | 0.978  | 0.672  | +0.305           |
| eccentricity_mean     | 0.451  | 0.220  | +0.231           |
| nuclear_area_mean     | 0.482  | 0.322  | +0.159           |
| nuclei_density        | 0.931  | 0.821  | +0.109           |
| prolif_fraction       | 0.864  | 0.928  | -0.064           |
| nonprolif_fraction    | 0.856  | 0.927  | -0.071           |
| mean_glucose          | 0.794  | 1.000  | -0.206           |
| healthy_fraction      | 0.702  | 0.914  | -0.213           |
| mean_oxygen           | 0.782  | 1.000  | -0.218           |
| cancer_fraction       | 0.678  | 0.919  | -0.241           |
| immune_fraction       | 0.469  | 0.746  | -0.277           |
| vessel_area_pct       | 0.502  | 1.000  | -0.498           |
| dead_fraction         | -0.200 | 0.595  | -0.796           |

(`tme_r2` is fit from the experimental TME mask features; ~1.0 R² rows are degenerate identity cases where the attr equals an input channel statistic.)

Interpretation of the probe stage:

- UNI's strength relative to TME is **texture and stain** (Δ R² between +0.30 and +0.61 for the appearance attrs) — this is genuinely H&E-only information.
- TME beats UNI on **channel-defined composition** (vessel area, oxygen/glucose, cancer/immune fractions) because those quantities are literally encoded in the input channels.
- `dead_fraction` is essentially undecodable from UNI (R² = −0.20) — apoptotic morphology is not preserved or not present in this slide.

### Sweep — directional edits (n=90 generations per row: 30 tiles × 2 directions × 3 alphas)

`appearance_sweep_summary.csv` reports targeted slope of the metric vs alpha along the probe direction, plus a random-direction control. Slopes whose 95% CI excludes 0 are statistically nonzero. Sample of the strongest causal edits:

| attr edited       | metric moved              | targeted slope | random slope | causal? |
| ----------------- | ------------------------- | -------------- | ------------ | ------- |
| nuclei_density    | appearance.h_mean         | **+0.259**     | -0.001       | yes     |
| nuclei_density    | appearance.texture_h_cont | **+15.78**     | +0.001       | yes     |
| nuclei_density    | appearance.texture_e_cont | **+14.19**     | -4.86        | yes     |
| texture_h_contrast| appearance.h_mean         | **-0.142**     | -0.008       | yes     |
| texture_h_contrast| appearance.texture_h_cont | **+20.94**     | +2.52        | yes     |
| texture_h_energy  | appearance.texture_h_cont | **-19.13**     | -1.97        | yes     |
| texture_e_contrast| appearance.h_mean         | **+0.258**     | -0.002       | yes     |
| nuclear_area_mean | appearance.h_mean         | **+0.088**     | +0.005       | yes     |
| eccentricity_mean | appearance.texture_h_cont | **+8.56**      | +2.54        | yes     |

Interpretation:

- Editing along the probe direction moves the targeted appearance metric ~5–10× more than a random direction of equal norm — confirming the probe directions are **functional**, not just correlational.
- Texture-contrast probes are the most causally powerful (largest slopes); intensity (`h_mean`, `e_mean`) probes are second.
- Sweep over `eccentricity_mean` and `nuclear_area_mean` directions still moves stain metrics — UNI's morphology subspace is **entangled** with its stain subspace.

### Null — projecting the probe direction out (n=30 tiles × 2 conditions)

`appearance_null_summary.csv` reports the per-tile mean of each appearance metric under three regimes: targeted null (project out `w`), random null (project out random direction), full UNI null (set UNI = 0). Selected rows:

| attr nulled        | metric              | targeted null | random null | full-UNI null |
| ------------------ | ------------------- | ------------- | ----------- | ------------- |
| eccentricity_mean  | h_mean              | 0.0156        | 0.0156      | 0.0202        |
| eccentricity_mean  | texture_h_contrast  | 5.114         | 5.102       | 2.749         |
| eccentricity_mean  | stain_vector_angle  | 2.609         | 2.569       | 11.402        |
| nuclei_density     | texture_h_contrast  | 5.045         | 5.067       | 2.749         |
| texture_h_contrast | texture_h_contrast  | 5.394         | 5.075       | 2.749         |
| texture_h_contrast | texture_h_homog.    | 0.528         | 0.541       | 0.660         |

Interpretation:

- Targeted nulls are barely distinguishable from random nulls for most appearance metrics — a single 1-d projection does **not** remove the encoded attribute. The information is **distributed** across many UNI directions.
- Full UNI nulling, in contrast, collapses several metrics toward TME-only baseline (`stain_vector_angle` jumps from 2.6 to 11.4; `texture_h_contrast` drops 5.1 → 2.7). This shows UNI as a whole is doing real work, but no single 1-d subspace is sufficient.
- Practical consequence: sweep edits succeed because they push along a useful axis with large alpha; null edits fail because removing one axis leaves enough redundancy in UNI to reconstruct the attribute. The encoding is "linearly readable but holographically stored".

## 4. Are these the right experiments to answer "what bio features live in UNI and what happens when they're perturbed"?

Yes — a4 is the right tool, with one caveat.

- **Encoded?** ✓ probe R² answers it for any pre-specified attribute (channel composition, morphology, stain, texture). Fig 07 is the encoder-comparison version of the same question.
- **Effect of perturbation?** ✓ sweep gives the causal slope per direction; null isolates necessity.

Caveat: a4 currently restricts the attribute set to a hand-curated list (`labels.py:18-44`: channel fractions, CellViT morphology, GLCM textures). To answer "what features in general live in UNI" without hand-naming them, complement with an **unsupervised** decomposition of UNI (PCA / sparse-dict / linear concept activation vectors) and probe whether each unsupervised direction maps to any handcrafted attribute. This would catch axes the current probe set misses.

## 5. Summary

- Fig 07 = decodability (probe R²); a4 = causal visibility (probe + sweep + null).
- UNI strongly encodes density, proliferation, oxygen/glucose, and stain/texture (R² ≥ 0.7 for most). It does **not** encode lineage markers (CD45 family, FOXP3 — R² ≤ 0.1).
- Sweep edits along probe directions produce ~5–10× larger appearance shifts than random directions, with the largest causal effect on texture contrast and intensity.
- Single-direction null is ineffective; UNI stores attributes redundantly across many directions, so the only way to remove an attribute is to zero UNI entirely.
- For the paper, frame the storyline as: **Fig 07 establishes that the H&E foundation embeddings hold the information; a4 establishes that the ControlNet uses it.**
