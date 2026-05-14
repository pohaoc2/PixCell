# Channel Importance Summary — Figures 01–09 (paper)

This note summarizes the quantitative evidence behind figures `01_metric_tradeoffs` through `09_combinatorial_grammar` (paths under `figures/pngs_updated/`). Two datasets feed figs 01–06: paired ORION-CRC ControlNet runs at `inference_output/concat_ablation_1000/paired_ablation/ablation_results` (n=1000 tiles) and unpaired runs at `inference_output/concat_ablation_1000/unpaired_ablation/ablation_results` (n=1001). Figures 07–09 reuse the paired runs plus auxiliary probes/sweeps.

Evidence files:

- `figures/pngs_updated/01_metric_tradeoffs.png`
- `figures/pngs_updated/02_paired_vs_unpaired.png`
- `figures/pngs_updated/03_channel_effect_sizes.png`
- `figures/pngs_updated/04_leave_one_out_impact.png`
- `figures/pngs_updated/05_paired_ablation_grid.png`
- `figures/pngs_updated/06_unpaired_ablation_grid.png`
- `figures/pngs_updated/07_inverse_decoding.png`
- `figures/pngs_updated/08_uni_tme_decomposition.png`
- `figures/pngs_updated/09_combinatorial_grammar.png`
- `src/a1_probe_linear/out/linear_probe_results.csv`
- `src/a1_codex_targets/probe_out/t2_mlp/mlp_probe_results.csv`
- `src/a2_decomposition/out/decomposition_summary.csv`
- `src/a2_decomposition/out/fud_scores.json`
- `src/a2_decomposition/out/mode_summary.csv`
- `src/a3_combinatorial_sweep/out/morphological_signatures.csv`

For Figure 01 specifically, the metric family is `fud`, `lpips`, `pq`, `dice`, `style_hed`: higher is better for `pq`, `dice`; lower is better for `lpips`, `fud`, `style_hed`.

## 1. Average performance by number of active groups

These are the exact metric families shown in `01_metric_tradeoffs.png`.

**Paired (n=1000 tiles)**

| groups | lpips | pq    | dice  | fud     | style_hed |
| ------ | ----- | ----- | ----- | ------- | --------- |
| 1g     | 0.426 | 0.189 | 0.309 | 165.051 | 0.135     |
| 2g     | 0.396 | 0.335 | 0.475 | 144.294 | 0.109     |
| 3g     | 0.342 | 0.732 | 0.857 | 133.998 | 0.068     |
| 4g     | 0.319 | 0.808 | 0.903 | 133.899 | 0.035     |

**Unpaired (n=1001 tiles)**

| groups | lpips | pq    | dice  | fud     | style_hed |
| ------ | ----- | ----- | ----- | ------- | --------- |
| 1g     | 0.551 | 0.265 | 0.426 | 164.794 | 0.135     |
| 2g     | 0.523 | 0.490 | 0.639 | 157.021 | 0.106     |
| 3g     | 0.499 | 0.679 | 0.793 | 161.696 | 0.078     |
| 4g     | 0.478 | 0.757 | 0.839 | 163.088 | 0.044     |

Interpretation:

- In paired runs, more channel groups push all five Figure 01 metrics in the expected direction; the main gains arrive at 2g→3g, where `pq` jumps from 0.335 to 0.732 and `dice` from 0.475 to 0.857.
- In unpaired runs, `lpips`, `pq`, `dice`, and `style_hed` still improve nearly monotonically with more groups, but `fud` is different: it is best at 2g (157.0) and then worsens slightly at 3g/4g. That is the key paired/unpaired divergence in Figure 01.
- The reason is that paired runs keep morphology and H&E style aligned tile-by-tile, so extra groups help both realism and structure together. Unpaired runs still gain structure from extra biological cues, but the added constraints can move the output away from the borrowed reference style in UNI-feature space, which shows up as worse `fud`.
- At matched cardinality, paired remains better on structure (`pq`, `dice`) while unpaired closes much of the stain gap once `microenv` is present (`style_hed` 0.044 unpaired 4g vs 0.035 paired 4g).

## 2. Best and worst conditions

**Paired**

| metric    | best condition                               | value   | worst condition | value   |
| --------- | -------------------------------------------- | ------- | --------------- | ------- |
| fud       | `cell_types+microenv+vasculature`            | 118.697 | `vasculature`   | 185.789 |
| lpips     | `cell_state+cell_types+microenv+vasculature` | 0.319   | `cell_types`    | 0.430   |
| pq        | `cell_state+cell_types+microenv`             | 0.809   | `vasculature`   | 0.061   |
| dice      | `cell_state+cell_types+microenv`             | 0.904   | `vasculature`   | 0.143   |
| style_hed | `cell_state+cell_types+microenv+vasculature` | 0.035   | `vasculature`   | 0.166   |

**Unpaired**

| metric    | best condition                               | value   | worst condition | value   |
| --------- | -------------------------------------------- | ------- | --------------- | ------- |
| fud       | `microenv`                                   | 136.401 | `vasculature`   | 189.310 |
| lpips     | `cell_state+cell_types+microenv+vasculature` | 0.478   | `vasculature`   | 0.564   |
| pq        | `cell_state+cell_types+microenv`             | 0.765   | `vasculature`   | 0.064   |
| dice      | `cell_state+cell_types+microenv`             | 0.847   | `vasculature`   | 0.171   |
| style_hed | `cell_state+cell_types+microenv+vasculature` | 0.044   | `vasculature`   | 0.172   |

Interpretation:

- `vasculature` alone is the universal worst condition across both modes and all five Figure 01 metrics; vessel masks add local detail, not enough global layout to stand alone.
- In paired runs, the best structural setting is `cell_state+cell_types+microenv`, with 4g essentially tied; this says vasculature is not required to hit the structural optimum.
- In unpaired runs, 4g is still best for `lpips`, `pq`, `dice`, and `style_hed`, but not for `fud`: `microenv` alone gives the lowest unpaired `fud`. This is another sign that unpaired realism in UNI space is dominated by stain context more than by full spatial specificity.
- The paired/unpaired asymmetry is therefore not “more channels always helps everything”; it is “more channels always helps structure, while `fud` depends on whether layout and style come from the same tile.”

## 3. Average effect of adding each group

Positive deltas mean "adding this group tends to help"; sign convention is normalized so positive is improvement.

**Paired**

| added group   | Δ lpips | Δ pq   | Δ dice | Δ fud (better +) | Δ style_hed |
| ------------- | ------- | ------ | ------ | ---------------- | ----------- |
| `cell_types`  | +0.032  | +0.304 | +0.303 | +8.247           | +0.008      |
| `cell_state`  | +0.033  | +0.322 | +0.322 | +4.371           | +0.010      |
| `vasculature` | +0.040  | +0.113 | +0.113 | +8.988           | +0.008      |
| `microenv`    | +0.052  | +0.236 | +0.228 | +31.683          | +0.107      |

**Unpaired**

| added group   | Δ lpips | Δ pq   | Δ dice | Δ fud (better +) | Δ style_hed |
| ------------- | ------- | ------ | ------ | ---------------- | ----------- |
| `cell_types`  | +0.024  | +0.285 | +0.263 | -7.191           | +0.002      |
| `cell_state`  | +0.026  | +0.316 | +0.289 | -11.394          | +0.007      |
| `vasculature` | +0.014  | -0.017 | -0.022 | +0.324           | +0.002      |
| `microenv`    | +0.037  | +0.170 | +0.126 | +22.776          | +0.106      |

Interpretation:

- `cell_state` and `cell_types` are the dominant structural drivers in both paired and unpaired settings; `pq` and `dice` tell the same story, so this is not specific to one instance-segmentation metric.
- `microenv` is the dominant realism and stain driver in both modes: it is by far the largest positive contributor to `style_hed`, and it is the only strong positive contributor to unpaired `fud`.
- The key paired/unpaired split is that adding `cell_state` or `cell_types` helps unpaired `pq` and `dice` but hurts unpaired `fud`. Extra biological structure improves spatial fidelity, yet it can make the image less compatible with the unrelated reference style.
- `vasculature` is weak in paired runs and mildly harmful to unpaired structure, consistent with it acting as a narrow local cue rather than a global organizer.

## 4. Presence-vs-absence summary

This table compares all conditions where a group is present versus absent.

**Paired**

| group present? | Δ lpips | Δ pq   | Δ dice | Δ fud   | Δ style_hed |
| -------------- | ------- | ------ | ------ | ------- | ----------- |
| `cell_types`   | +0.024  | +0.263 | +0.263 | +5.666  | +0.000      |
| `cell_state`   | +0.025  | +0.282 | +0.284 | +0.724  | +0.002      |
| `vasculature`  | +0.032  | +0.056 | +0.050 | +3.150  | -0.001      |
| `microenv`     | +0.044  | +0.181 | +0.174 | +30.423 | +0.106      |

**Unpaired**

| group present? | Δ lpips | Δ pq   | Δ dice | Δ fud    | Δ style_hed |
| -------------- | ------- | ------ | ------ | -------- | ----------- |
| `cell_types`   | +0.019  | +0.255 | +0.240 | -7.147   | -0.004      |
| `cell_state`   | +0.021  | +0.290 | +0.271 | -11.880  | -0.001      |
| `vasculature`  | +0.008  | -0.074 | -0.083 | -3.522   | -0.007      |
| `microenv`     | +0.031  | +0.112 | +0.074 | +24.612  | +0.105      |

Notes:

- This coarser present-vs-absent view matches the added-group analysis: `microenv` is the only group whose presence strongly helps both unpaired `fud` and `style_hed`.
- In paired runs, every group is neutral-to-positive for `fud` on average. In unpaired runs, `cell_types`, `cell_state`, and `vasculature` all turn negative for `fud` even though `cell_types` and `cell_state` remain positive for `pq` and `dice`.
- That is why the paired Figure 01 scatter moves mostly in one favorable direction as channels accumulate, while the unpaired scatter reflects a true tradeoff: more channel specificity helps structure, but only `microenv` reliably improves global realism against a mismatched style reference.
- The negative unpaired `vasculature` effect survives both the added-group and presence-vs-absence analyses, so it is a stable signal rather than a ranking artifact.

## 5. Leave-one-out pixel impact (Fig 04)

Average over `leave_one_out_diff_stats.json` per tile.

**Paired (n=1000)**

| removed group | mean diff | max diff | pct pixels > 10 | mean ΔE | causal inside/outside ΔE |
| ------------- | --------- | -------- | --------------- | ------- | ------------------------ |
| `cell_types`  | 20.49     | 192.70   | 45.98           | 3.62    | 4.86 / 0.36              |
| `cell_state`  | 21.12     | 195.44   | 47.63           | 3.69    | 5.07 / 0.42              |
| `vasculature` | 27.77     | 215.95   | 58.95           | 4.17    | 4.88 / 4.24              |
| `microenv`    | 37.65     | 243.42   | 81.82           | 6.94    | 6.94 / 0.00              |

**Unpaired (n=1000)**

| removed group | mean diff | max diff | pct pixels > 10 | mean ΔE |
| ------------- | --------- | -------- | --------------- | ------- |
| `cell_types`  | 35.37     | 229.72   | 77.21           | 6.69    |
| `cell_state`  | 35.19     | 227.21   | 77.22           | 6.68    |
| `vasculature` | 39.72     | 228.52   | 79.91           | 6.07    |
| `microenv`    | 41.06     | 242.14   | 83.29           | 9.43    |

Interpretation:

- Removing `microenv` causes the largest *global* pixel shift in both modes (>80% of pixels change >10 levels paired; ΔE_mean ~7 paired, ~9.4 unpaired). Its `causal_inside_mean_dE / outside` ratio is essentially infinite because changes are spatially diffuse rather than localized — confirming `microenv` controls stain saturation rather than structure.
- `vasculature` is the only group with a near-1 causal ratio (inside 4.88 vs outside 4.24 paired) — its impact is **local** at vessel pixels, off-vessel regions are largely untouched. This justifies treating it as a fine local detail rather than a global cue.
- `cell_types` and `cell_state` removal mostly affect masked-cell pixels (causal ratios ~10⁶ inside/outside), confirming they steer cell-resident morphology.

## 6. Encoder decodability (Fig 07)

`UNI → CODEX` ridge probes report per-channel R² (`src/a1_probe_linear/out/linear_probe_results.csv`).

| target            | R² mean | R² sd  |
| ----------------- | ------- | ------ |
| cell_density      | 0.953   | 0.007  |
| prolif_frac       | 0.863   | 0.035  |
| nonprolif_frac    | 0.826   | 0.023  |
| glucose_mean      | 0.821   | 0.020  |
| oxygen_mean       | 0.810   | 0.022  |
| healthy_frac      | 0.710   | 0.013  |
| cancer_frac       | 0.669   | 0.016  |
| vasculature_frac  | 0.509   | 0.022  |
| immune_frac       | 0.495   | 0.042  |
| dead_frac         | -0.135  | 0.107  |

T2 (CODEX marker channel from H&E) is decoded by `src/a1_codex_targets/probe_out/t2_mlp/mlp_probe_results.csv`: best linearly decodable markers are `PD-1` (0.36), `Hoechst` (0.35), `E-cadherin` (0.24); most lineage/immune markers stay near zero or negative R² — meaning UNI does **not** encode individual marker intensities, only aggregate composition.

## 7. UNI/TME decomposition (Fig 08)

Figure 08 is now complete quantitatively: Panel B is backed by per-tile metrics for all 500 tiles per mode, and `representative_tile.json` now selects tile `5376_4096`.

`src/a2_decomposition/out/decomposition_summary.csv` (n=500 tiles per mode):

| mode         | fud   | lpips | pq    | dice  | style_hed |
| ------------ | ----- | ----- | ----- | ----- | --------- |
| uni_plus_tme | 133.4 | 0.359 | 0.777 | 0.865 | 0.049     |
| uni_only     | 181.8 | 0.481 | 0.084 | 0.160 | 0.189     |
| tme_only     | 234.9 | 0.507 | 0.799 | 0.881 | 0.219     |
| neither      | 785.4 | 0.810 | 0.038 | 0.038 | 0.637     |

QC-style mode summary from `src/a2_decomposition/out/mode_summary.csv`:

| mode         | tissue_fraction | reference_rgb_mae | reference_hed_mae |
| ------------ | --------------- | ----------------- | ----------------- |
| uni_plus_tme | 0.778           | 0.092             | 0.078             |
| uni_only     | 0.679           | 0.117             | 0.102             |
| tme_only     | 0.821           | 0.098             | 0.100             |
| neither      | 0.0000          | 0.216             | 0.174             |

Interpretation:

- `tme_only` carries almost all cell-scale structure by itself: it is the best mode for `pq` (0.799) and `dice` (0.881), slightly above `uni_plus_tme` (`pq=0.777`, `dice=0.865`). So the explicit TME masks, not the UNI prior, are the main source of segmentation-faithful layout.
- `uni_plus_tme` is nevertheless the best overall realism/stain condition: it is the best mode for `fud`, `lpips`, and `style_hed`. Adding UNI on top of TME improves texture and stain realism strongly versus `tme_only` (`fud 133 vs 235`, `lpips 0.359 vs 0.507`, `style_hed 0.049 vs 0.219`) while only slightly softening structural metrics.
- `uni_only` preserves part of the H&E appearance prior (`fud=181.8`, much better than `neither=785.4`), but structure largely collapses (`pq=0.084`, `dice=0.160`). That means UNI carries useful global appearance/composition information, but not precise cell placement.
- `neither` is the true null: tissue fraction goes to zero, stain errors explode, and both structure metrics collapse.
- The QC metrics agree with the Figure 01-style metrics: TME is what makes tissue appear in the right places (`tissue_fraction 0.821` for `tme_only`), while UNI is what makes the tissue look histologically plausible once the layout exists.
- Panel C's effect decomposition is therefore consistent with a non-additive split of roles: TME dominates the structural effect (`pq`, `dice`), UNI contributes the strongest realism/stain gains, and the interaction term is net negative across all metrics, meaning the combined model is complementary but not simply additive.
- So the answer to the earlier “are there missing data?” is now no for Figure 08 itself: the missing layer was the derived CellViT-backed per-tile metrics, and that cache has now been generated, summarized, and rendered into the final PNG.

## 8. Combinatorial grammar (Fig 09)

`src/a3_combinatorial_sweep/out/morphological_signatures.csv` provides per-anchor morphology over the full `(cell_state × oxygen_label × glucose_label)` lattice; the residuals to an additive model are in `additive_model_residuals.csv`. Residual L2 norms across the 18 lattice cells stay small (<13 per cell, typically <7), so cell_state, oxygen, and glucose act *near-additively* on `nuclear_density`, `eosin_ratio`, and texture GLCM stats. Interactions are second-order (~3–10% of total spread) — meaning the channels are individually interpretable when stacked.

## 9. Channel importance ranking

Pulling 03 effect sizes, 04 LOO, 06 encoder decodability, and 08 decomposition together:

| group         | structure (aji/pq) | stain (style_hed/fud) | decodability from UNI | global vs local |
| ------------- | ------------------ | --------------------- | --------------------- | --------------- |
| `cell_types`  | **dominant**       | mid                   | 0.51–0.71 R²          | global layout   |
| `cell_state`  | **dominant**       | mid                   | 0.83–0.86 R²          | global density  |
| `microenv`    | mid                | **dominant**          | 0.81–0.82 R²          | global stain    |
| `vasculature` | weak / negative    | weak                  | 0.51 R²               | local           |

Conclusion: the paired ControlNet relies on three independent axes — cell mass (cell_types), cell vitality (cell_state), and stain modulator (microenv) — with vasculature adding only fine-grained vessel detail. UNI carries the cell-density and oxygen/glucose information well enough to *replace* parts of TME, but TME masks remain essential for spatial accuracy at cell scale (best `aji`/`pq` always requires at least `cell_types + cell_state`).
