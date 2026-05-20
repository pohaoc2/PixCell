# Combinatorial Grammar — Design Spec

- Date: 2026-05-20
- Scope: metric schema swap + 2–3 new quantitative figs for fig 09 / SI_09
- Companion HTML mockup: `docs/proposal_combinatorial_grammar_2026-05-20.html`

## 1. Goal

Replace the bespoke a3 metric set + single-scalar `interaction_heatmap.png` with the a4 metric schema and a variance-partition view that directly answers: **is TME grammar (state × O₂ × glucose) additive?**

Outputs:
- Refreshed `09_combinatorial_grammar.png` (main) with new headline layout.
- Refreshed `SI_09_combinatorial_grammar_anchors.png` plus three new SI panels.
- New CSV `variance_partition.csv` alongside existing artifacts (no consumer break).

## 2. Non-goals

- No UMAP/PCA of UNI embeddings.
- No cross-attention entropy probe.
- No channel-group inference ablation (covered elsewhere).
- No real-vs-generated marginal match.
- No new H&E generation beyond two additional seeds.

## 3. Metric schema

Adopt `src/a4_uni_probe/labels.py::MORPHOLOGY_ATTR_NAMES` + `APPEARANCE_ATTR_NAMES`. Anchor-level channel fractions (constant under sweep) are written once to a side table and excluded from per-tile rows.

| Current (a3) | Action | Replacement |
|---|---|---|
| `nuclear_density` | keep | `nuclei_density` (CellViT contours / tissue area) |
| `mean_cell_size`, `nucleus_area_median`, `nucleus_area_iqr` | collapse | `nuclear_area_mean` (median+IQR → SI table only) |
| — | add | `eccentricity_mean`, `intensity_mean_h`, `intensity_mean_e` |
| `hematoxylin_burden` | rename | `h_mean` |
| `hematoxylin_ratio`, `eosin_ratio` | drop | redundant (sum=1); keep `h_mean`, `e_mean` |
| `glcm_contrast`, `glcm_homogeneity` (gray) | replace | `texture_{h,e}_{contrast,homogeneity,energy}` on HED |

Implementation hook: import `appearance_row_for_image` from `src/a4_uni_probe/appearance_metrics.py` inside `src/a3_combinatorial_sweep/main.py::_compute_signature`. CellViT-derived morphology stats are already loaded via `tools.cellvit.contours.load_cellvit_contours`; just rename fields to match a4.

## 4. Seed variance band

Render seeds 43 and 44 in addition to existing seed 42:

```
src/a3_combinatorial_sweep/out/generated/<anchor>/<condition>.png       # seed 42 (existing)
src/a3_combinatorial_sweep/out/generated_s43/<anchor>/<condition>.png   # new
src/a3_combinatorial_sweep/out/generated_s44/<anchor>/<condition>.png   # new
```

Compute budget: 20 anchors × 27 conditions × 2 extra seeds = 1 080 renders (~30–60 min on one A10).

`morphological_signatures.csv` gains a `seed` column. Point estimates: mean across seeds. Error bars: 95% bootstrap CI across seeds (anchor-mean first, then percentile bootstrap over anchors).

## 5. Variance partition (3-way ANOVA)

For each metric *m*, model the per-tile value *y* as:

```
y_{a,s,o,g,r} = μ + α_a + β_s + γ_o + δ_g
              + (αβ)_{a,s} + (βγ)_{s,o} + (βδ)_{s,g} + (γδ)_{o,g}
              + (βγδ)_{s,o,g} + ε_r
```

- `a` = anchor, `s` = cell_state, `o` = oxygen_label, `g` = glucose_label, `r` = seed.
- Compute sum-of-squares per term by group-mean projection (Type I sequential SS, ordered as listed).
- Output: fraction of total SS per term, summing to 1.0 per metric.

Panel A in the main fig collapses all 2-way and 3-way grammar interactions `(βγ), (βδ), (γδ), (βγδ)` into a single **interactions** bar segment. Anchor variance `α_a + (αβ)_{a,s}` is shown as a separate **anchor** segment because it is structural, not grammatical.

Module: `src/a3_combinatorial_sweep/variance_partition.py` (pure numpy). Public API:

```python
def variance_partition(
    rows: list[dict[str, Any]],
    metrics: tuple[str, ...],
) -> dict[str, dict[str, float]]:
    """Return {metric: {anchor, state, o2, gluc, s_x_o, s_x_g, o_x_g, s_x_o_x_g, resid}}."""
```

Unit test: synthetic additive data (`y = β_s + γ_o + δ_g + ε`) must yield interaction shares < 1e-6.

## 6. Main figure 09 — new layout

Two stacked panels.

**Panel A — Variance-partition bars.** One horizontal stacked bar per metric, six segments: `anchor`, `state`, `O2`, `glucose`, `interactions`, `seed/residual`. Diverging-free categorical palette matching a4 conventions. Bars sorted by interaction share (largest at top) so the nonadditive finding is visually first.

**Panel B — One representative anchor sweep (3 × 9).** Rows = state (prolif / nonprolif / dead). Columns = (O₂ × glucose) ∈ {low, mid, high}². Same tile rendering as SI_09 subgrid. Anchor pick: highest sweep magnitude under new metrics AND median anchor-variance contribution (ties → first by anchor_id sort) to keep the panel visually consistent with Panel A's claim.

Figure size: 7.5 × 9.0 in, 300 dpi.

## 7. SI deliverables

- **SI_09a — Raw anchor sweep grids.** Existing 2×2 layout (representative + low/mid/high sweep magnitude). Refresh anchor picks from new `morphological_signatures.csv`. Same builder, no API change.
- **SI_09b — Per-metric residual small-multiples.** Grid of 9 mini heatmaps (one per a4 metric), each showing actual − additive prediction over (state × (O₂, glucose)). Diverging colormap (e.g., `RdBu_r`) centered at 0. Replaces the single-scalar `interaction_heatmap.png` for fig-builder consumers (file is kept on disk for back-compat).
- **SI_09c — Seed CI table.** Compact table: per metric × condition, mean ± 95% bootstrap CI across 3 seeds, averaged over 20 anchors. Establishes noise floor.
- **SI_09d — Anchor sensitivity ranking.** Horizontal bar chart, one bar per anchor, length = sweep ‖Δmetric‖₂ averaged over metrics. Justifies anchor picks used in SI_09a and main Panel B.

## 8. File-by-file changes

| File | Change |
|---|---|
| `src/a3_combinatorial_sweep/main.py` | `_compute_signature` calls `appearance_row_for_image`; remove ad-hoc GLCM. Update `MORPHOLOGY_METRICS` tuple to a4 schema. `run_generate_worker` accepts repeated `--seed` flag, writes to `generated_s{seed}/`. `_iter_signature_rows` walks all `generated*` subtrees and adds `seed` column. `_fit_additive_rows` augmented (or replaced) by call to new variance-partition module. |
| `src/a3_combinatorial_sweep/variance_partition.py` (new) | 3-way ANOVA decomposition, pure numpy, public API per §5. |
| `src/a3_combinatorial_sweep/anchors_k20_t1_medoid.txt` | unchanged |
| `src/paper_figures/fig_combinatorial_grammar.py` | Rewrite main fig as Panel A (variance bars) + Panel B (sweep grid). Remove slope-style content (moved to SI). |
| `src/paper_figures/fig_combinatorial_grammar_si.py` | Add three new SI subbuilders (small-multiples, seed CI table, anchor ranking). Keep existing 2×2 raw-grid section. |
| `src/paper_figures/fig_combinatorial_grammar_panels/_shared.py` | Audit metric-column refs; update `pick_representative_anchor` and `compute_anchor_sweep_magnitude` to use new metric names. |
| `tests/test_fig_combinatorial_grammar.py` | Update fixtures for new metric names. Add variance-partition unit test (synthetic additive data → interactions ≈ 0). |

`additive_model_residuals.csv` is **kept unchanged** so any external consumer does not break; new file `variance_partition.csv` is written alongside.

## 9. CellViT prerequisite

Before recomputing signatures, sidecar JSONs must exist for tiles in all `generated*/` subtrees. Run the standard 3-step pipeline documented in `CLAUDE.md`:

```bash
# 1. export
conda run --no-capture-output -n pixcell python tools/cellvit/export_batch.py \
  --cache-root src/a3_combinatorial_sweep/out/generated_s43 \
  --output-dir /tmp/cellvit_s43 --overwrite --zip
# 2. run CellViT (cellvit env)
# 3. import back beside source PNGs
```

Repeat for `generated_s44`. Existing `generated/` tiles already have sidecars from prior run (verify before counting on it).

## 10. Risks

| Risk | Mitigation |
|---|---|
| CellViT sidecars missing for new seed tiles | Run 3-step pipeline before recompute. Worker should fail loud, not silently fall back to skimage CC. |
| Seed noise ≥ interaction signal → variance partition flat | That **is** the finding. SI_09c table makes it visible. Don't oversell. |
| Existing `additive_model_residuals.csv` consumers | Keep old CSV untouched; new file `variance_partition.csv` alongside. |
| Renaming a3 metric columns breaks `_shared.py` | Audit in same PR; tests catch the rest. |
| 1 080 new renders exceed budget | Job is deferred-runnable; can split across two GPU sessions. |

## 11. Approval checklist

- [x] Metric schema = a4 `MORPHOLOGY_ATTR_NAMES + APPEARANCE_ATTR_NAMES`
- [x] 3 seeds (42 / 43 / 44), parallel `generated_s*/` dirs
- [x] Main fig 09 = variance-partition bars + 1 anchor sweep grid
- [x] SI = refreshed raw grids + residual small-multiples + seed CI table + anchor ranking
- [x] Variance partition implemented in standalone module with synthetic-additive unit test
- [x] No new H&E generation beyond the 2 extra seeds
- [x] `additive_model_residuals.csv` left untouched; new outputs side-by-side

## 12. Handoff

Per `CLAUDE.md` role split (Claude plans/reviews, Codex implements):
- Next step: invoke `writing-plans` skill to produce a step-by-step implementation plan with verification checkpoints.
- Implementation itself is delegated to Codex via `codex:codex-rescue` once the plan is approved.
