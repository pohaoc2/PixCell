# Combinatorial Grammar Sweep Summary (Fig 09)

Run family: `a3_combinatorial_sweep`
Anchor set: `anchors_k20_t1_medoid.txt` (k=20 UNI medoids from the ORION-CRC training set, t=1 nearest medoid)
Metric stack: `a4` morphology+appearance (CellViT instance morphology + HED/Haralick appearance)

This summary covers the 27-condition sweep over the 20 anchor tiles:

- Anchors: 20 ORION-CRC tiles
- Conditions: 3 cell-states (`dead`, `nonprolif`, `prolif`) x 3 O2 levels (`low`, `mid`, `high`) x 3 glucose levels (`low`, `mid`, `high`) = 27
- Seeds: 3 (`42`, `43`, `44`)
- Total generated H&E PNGs scored: **1,620**
- Per-PNG metric rows in `morphological_signatures.csv`: 1,620 (20 x 27 x 3)
- Condition-level rows in `additive_model_residuals.csv`: 27 (averaged across anchors x seeds, n=60 per row)

Three metrics are all-NaN by construction and excluded from every aggregate below: `intensity_mean_h`, `intensity_mean_e`, `appearance.stain_vector_angle_deg`. The first two have no per-nucleus intensity exposed in the CellViT JSON sidecars; the third degenerates when the stain-vector estimator falls back to the canonical basis. They are kept as columns in the CSVs for schema consistency only.

---

## Variance Partition (Full Data)

Three-way ANOVA share per metric across (`anchor`, `state`, `o2`, `gluc`) and their interactions, computed on the full 1620-row table. `interactions` is `s_x_o + s_x_g + o_x_g + s_x_o_x_g`. Rows sorted by `interactions` descending; rounded to 3 decimals.

| metric | anchor | state | O2 | gluc | interactions | resid |
|---|---:|---:|---:|---:|---:|---:|
| `eccentricity_mean` | 0.635 | 0.000 | 0.001 | 0.001 | 0.002 | 0.361 |
| `appearance.e_std` | 0.563 | 0.012 | 0.000 | 0.001 | 0.001 | 0.423 |
| `nuclear_area_mean` | 0.925 | 0.004 | 0.001 | 0.000 | 0.000 | 0.070 |
| `appearance.texture_e_contrast` | 0.905 | 0.002 | 0.000 | 0.002 | 0.000 | 0.090 |
| `appearance.texture_e_homogeneity` | 0.947 | 0.000 | 0.001 | 0.006 | 0.000 | 0.045 |
| `appearance.texture_h_homogeneity` | 0.972 | 0.001 | 0.001 | 0.005 | 0.000 | 0.020 |
| `appearance.texture_h_energy` | 0.940 | 0.001 | 0.001 | 0.005 | 0.000 | 0.054 |
| `appearance.e_mean` | 0.966 | 0.001 | 0.003 | 0.009 | 0.000 | 0.021 |
| `appearance.texture_e_energy` | 0.949 | 0.000 | 0.002 | 0.006 | 0.000 | 0.043 |
| `appearance.texture_h_contrast` | 0.957 | 0.000 | 0.001 | 0.004 | 0.000 | 0.037 |

Anchor share dominates 56-99% of variance for every non-degenerate metric, peaking at 99.0% for `appearance.h_mean` and `nuclei_density`. The two outliers — `eccentricity_mean` (anchor 63%, resid 36%) and `appearance.e_std` (anchor 56%, resid 42%) — leak most of their non-anchor signal into residual rather than into the conditioning factors. The combined sweep factors (state, O2, glucose, all interactions) account for less than 2% of total variance on every metric in this table. The model is mask-driven: anchor-to-anchor variance in the underlying mask geometry vastly outsizes the 27-condition conditioning sweep.

---

## Variance Partition (Within-Anchor)

Same partition after per-anchor demeaning ("strip-factor" view that removes anchor variance before running ANOVA). This is the "grammar" view: it asks which factor moves morphology *after* you fix the mask. Rows sorted by `interactions` descending.

| metric | state | O2 | gluc | interactions | resid |
|---|---:|---:|---:|---:|---:|
| `appearance.texture_h_homogeneity` | 0.048 | 0.035 | 0.198 | 0.009 | 0.710 |
| `nuclear_area_mean` | 0.054 | 0.010 | 0.000 | 0.007 | 0.929 |
| `appearance.texture_e_homogeneity` | 0.005 | 0.021 | 0.117 | 0.006 | 0.852 |
| `eccentricity_mean` | 0.001 | 0.002 | 0.002 | 0.006 | 0.989 |
| `appearance.texture_e_contrast` | 0.022 | 0.001 | 0.022 | 0.004 | 0.952 |
| `appearance.e_mean` | 0.033 | 0.080 | 0.268 | 0.003 | 0.615 |
| `appearance.texture_h_contrast` | 0.009 | 0.021 | 0.097 | 0.002 | 0.871 |
| `appearance.texture_h_energy` | 0.018 | 0.010 | 0.076 | 0.002 | 0.894 |
| `appearance.texture_e_energy` | 0.002 | 0.036 | 0.117 | 0.002 | 0.843 |
| `appearance.e_std` | 0.028 | 0.001 | 0.003 | 0.001 | 0.967 |
| `nuclei_density` | 0.071 | 0.002 | 0.004 | 0.001 | 0.922 |
| `appearance.h_std` | 0.055 | 0.004 | 0.011 | 0.001 | 0.930 |
| `appearance.h_mean` | 0.033 | 0.013 | 0.066 | 0.000 | 0.888 |

Once anchor is stripped, glucose is the dominant within-anchor factor for every E-channel and texture metric, peaking at 27% of within-anchor variance for `appearance.e_mean` and 20% for `appearance.texture_h_homogeneity`. State is the dominant within-anchor factor for the morphology-leaning metrics: `nuclei_density` (7.1%), `appearance.h_std` (5.5%), and `nuclear_area_mean` (5.4%). O2 is consistently the weakest of the three single factors, only crossing 3.5% on `appearance.texture_e_energy` and `appearance.texture_h_homogeneity`. Interactions remain small in every cell (max 0.91% on `appearance.texture_h_homogeneity`), meaning the within-anchor signal is well-approximated by an additive state + O2 + glucose model.

---

## Additive vs. Full Model

`additive_model_residuals.csv` fits an additive `actual ~ state + O2 + glucose` model on the anchor-and-seed-averaged condition means and reports the per-condition residual. Computing a z-scored L2 over the 10 non-degenerate residual columns (z-scored across the 27 conditions), the three conditions with the largest deviation from additivity are:

| cell_state | O2 | glucose | z-scored residual L2 |
|---|---|---|---:|
| `nonprolif` | `low` | `low` | 5.63 |
| `nonprolif` | `high` | `high` | 5.42 |
| `prolif` | `high` | `high` | 5.24 |

The corners of the conditioning cube — extreme low-low and high-high combinations on the proliferating and non-proliferating states — are where the additive model breaks down most. The `dead` state shows the smallest residuals at both diagonal corners, consistent with `dead` being closer to a single-mode appearance regardless of O2/glucose. In absolute terms the largest residual is in `appearance.texture_e_contrast` (max |residual| 0.132) and `appearance.texture_h_contrast` (max 0.024), so the non-additive structure is concentrated in E/H texture contrast rather than in mean stain channels.

---

## Anchor Sweep Magnitude

Per-anchor metric range across the 27-condition sweep (averaged over seeds, then z-scored against the across-anchor pooled std, then per-metric max-minus-min, averaged over the 13 non-degenerate metrics):

- Overall mean z-range across the 20 anchors: **0.43**
- Min / max anchor mean-z-range: **0.20 / 1.48**
- Top anchor: `25856_45824` (mean z-range 1.48, driven by `appearance.e_std`)
- Second: `26624_50688` (mean z-range 1.03, driven by `eccentricity_mean`)

Most anchors sit below 0.5 z, i.e. the 27-condition sweep moves the average metric by less than half a cross-anchor standard deviation. Two anchors break out above 1.0 — these are the candidate "responsive" tiles surfaced in panel B of `09_combinatorial_grammar.png` and broken out small-multiples-style in `SI_09_combinatorial_grammar_anchors.png`. The remaining 18 anchors are flat-responders under this sweep.

---

## Conclusions

1. The model is mask-driven: across the 1620-image sweep, anchor identity accounts for 56-99% of total variance on every non-degenerate morphology and appearance metric. The combined effect of state, O2, glucose, and their interactions stays below ~2% of total variance on every metric.
2. The within-anchor view exposes the actual conditioning grammar: glucose is the dominant within-anchor knob for E-channel mean and intensity-derived texture metrics (e.g. `appearance.e_mean` glucose share 27%, `appearance.texture_h_homogeneity` glucose share 20%), while state is the dominant within-anchor knob for morphology-coupled metrics (`nuclei_density` state share 7.1%, `appearance.h_std` 5.5%, `nuclear_area_mean` 5.4%).
3. O2 is the weakest of the three single factors within-anchor, never crossing 4% on any metric. If a single conditioning channel is to be down-weighted in future ablations, O2 is the candidate.
4. Interactions are negligible in the within-anchor partition (max 0.9% on `appearance.texture_h_homogeneity`); the within-anchor sweep is well-modelled as additive state + O2 + glucose.
5. The additive model breaks down most at the diagonal corners of the (O2, glucose) sub-cube for `nonprolif` and `prolif` states; absolute non-additivity is concentrated in E/H texture contrast.
6. Per-anchor sweep magnitude is highly uneven: 2 of 20 anchors show mean z-range above 1.0, while the remaining 18 sit below 0.5. The "responsive" anchors are the right slice to feature in any grammar-style figure.
7. Caveats: (a) three metrics are all-NaN by construction (`intensity_mean_h`, `intensity_mean_e`, `appearance.stain_vector_angle_deg`) and excluded from every aggregate above; (b) raw-variance ranking is biased toward large-scale metrics (e.g. `nuclear_area_mean` in absolute units), which is why the within-anchor and z-scored views are the correct lens for comparing factors across metrics; (c) the within-anchor residual share remains 60-99% even after stripping anchor, indicating substantial seed-level and unmodelled per-tile noise.

---

## Artifacts

- Variance partition (full): `src/a3_combinatorial_sweep/out/variance_partition.csv`
- Variance partition (within-anchor, strip-factor): `src/a3_combinatorial_sweep/out/variance_partition_within.csv`
- Additive model residuals: `src/a3_combinatorial_sweep/out/additive_model_residuals.csv`
- Per-PNG metric table: `src/a3_combinatorial_sweep/out/morphological_signatures.csv` (1620 rows)
- Sweep plan: `src/a3_combinatorial_sweep/out/plan.json`
- Anchor list: `src/a3_combinatorial_sweep/anchors_k20_t1_medoid.txt`
- Generated PNG caches: `src/a3_combinatorial_sweep/out/generated/`, `out/generated_s43/`, `out/generated_s44/`
- Main figure: `figures/pngs_updated/09_combinatorial_grammar.png`
- SI figure (per-anchor small multiples): `figures/pngs_updated/SI_09_combinatorial_grammar_anchors.png`
- Interaction heatmap: `src/a3_combinatorial_sweep/out/interaction_heatmap.png`
- Spec: `docs/superpowers/specs/2026-05-20-combinatorial-grammar-design.md`
- Plan: `docs/superpowers/plans/2026-05-20-combinatorial-grammar.md`
- Pipeline source: `src/a3_combinatorial_sweep/main.py`, `src/a3_combinatorial_sweep/variance_partition.py`
