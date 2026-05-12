# UNI Semantic Ablation — Hybrid Probe + Behavioral Edit

**Date:** 2026-05-12
**Status:** Draft for review
**Module:** `src/a4_uni_probe/`

---

## Purpose

Existing ablations (`src/a2_decomposition`, `docs/ablation_summary_*`) show that **UNI + TME > UNI-only ≈ TME-only > neither** on global metrics (FUD, LPIPS, PQ, DICE, HED). They do not answer:

> *What biological information does UNI encode that TME channels cannot supply, and does the model causally use it during generation?*

This spec adds a **representational probe** (what UNI encodes) plus **behavioral edits** (what UNI controls in pixels), validated against TME-channel baselines.

## Terminology

- **Appearance prior (UNI)** — global pooled embedding from UNI-2h pretrained on H&E. Encodes stain, texture, morphology distribution. Replaces previous label "semantic features."
- **Structural / biological layout (TME channels)** — per-pixel maps of cell type, cell state, vasculature, microenv from CODEX. Encodes spatial identity and gradients. Replaces previous label "spatial features."

Avoid the "semantic vs spatial" dichotomy: both modalities carry biological identity; the real axis is **global appearance prior** vs **spatially-resolved biological maps**.

## Scope

- Single slide: ORION-CRC33, ~10k paired patches.
- No cross-slide batch axis → slide-ID probe omitted.
- Reuse cached UNI features and VAE latents from Stage 1 pipeline.
- Reuse cached generated H&E from `src/a2_decomposition/out/generated/` where applicable.
- CellViT is the canonical cell segmenter; do not introduce HoVer-Net.

## Stages

### Stage 1 — Representational probe

**Inputs**
- Cached UNI embeddings (`features/`) for N paired tiles.
- Cached TME channels (`exp_channels/`).
- CellViT segmentation on real H&E tiles.

**Bio attribute labels per tile**

| Group | Attribute | Source |
|---|---|---|
| Channel-derived (ii) | cancer_fraction, healthy_fraction, immune_fraction | `exp_channels/cell_types` |
| Channel-derived (ii) | prolif_fraction, nonprolif_fraction, dead_fraction | `exp_channels/cell_state` |
| Channel-derived (ii) | vessel_area_pct, mean_oxygen, mean_glucose | `exp_channels/{vasculature,microenv}` |
| Morphology (iv) | nuclear_area_mean, eccentricity_mean, nuclei_density, intensity_mean_h, intensity_mean_e | CellViT on real H&E |

**Probes**
- Linear probe (Ridge regression for continuous, Logistic for any categorical).
- 5-fold CV stratified by spatial coordinate buckets (no adjacency leakage).
- Train two parallel probes per attribute:
  - **P_UNI**: features = UNI embedding (dim D_UNI).
  - **P_TME**: features = TME channel pooled stats (per-channel mean / std / area; dim ≈ 8 × 3).

**Outputs**
- `out/probe_results.csv`: per-attribute R² (or AUC), with P_UNI vs P_TME, plus delta and 95% CI from CV folds.
- Attributes ranked by **R²(UNI) − R²(TME)** = unique appearance information beyond channels.
- Carry forward to Stage 2/3 the **top-4 attributes that are measurable on generated H&E** (i.e., CellViT-derived morphology). Channel-derived attributes are reported in Stage 1 only — they are unobservable from generated pixels without ground-truth channels, so sweep/null cannot read them out.

### Stage 2 — Probe-direction sweep (causal edit)

For each top-ranked attribute `a` with probe weight vector `w_a`:

1. Unit-norm `w_a` over UNI feature dim.
2. Sample K=50 tiles spanning the attribute range.
3. For each tile, generate H&E at α ∈ {−2, −1, 0, +1, +2}: `UNI'(α) = UNI + α · w_a · ‖UNI‖`.
4. TME held fixed at the tile's real channels.
5. Run the attribute's metric extractor on each generated H&E:
   - Channel-derived attrs → cannot be measured on H&E directly; instead use CellViT-derived proxies (e.g., immune_fraction ≈ CellViT-classified small-round-cell fraction) where available, else skip (probe-only).
   - Morphology attrs → CellViT on generated H&E.

**Control**
- Random unit direction `w_rand` in UNI space (matched norm), same α sweep, K tiles. Expect flat response.

**Pass criterion**
- Linear regression of target metric vs α has slope significantly ≠ 0 (one-sided test, p < 0.01 after Bonferroni across attributes), with |slope_target| > 3 × |slope_random|.

**Outputs**
- `out/sweep/<attr>/<tile_id>/alpha=*.png`
- `out/sweep/<attr>/metrics.csv`: per (tile, α) target metric
- `out/sweep/<attr>/slope_summary.json`: slope, CI, p-value, comparison to random

### Stage 3 — Subspace nulling (causal removal)

For each top-ranked attribute:

1. **Targeted null**: `UNI_null = UNI − (UNI · w_a) w_a`.
2. **Random null**: `UNI_rand = UNI − (UNI · w_rand) w_rand` (matched norm projection).
3. **Full UNI null**: reuse `tme_only` mode from `src/a2_decomposition/` (already cached).

Generate H&E for conditions 1, 2 over the same K=50 tiles; pull condition 3 from cache.

**Pass criterion**
- Mean degradation of target metric under targeted null > random null (paired test, p < 0.01).
- Full-UNI-null degradation is upper-bounded interpretation: "how much of the attribute lives in UNI as a whole."

**Outputs**
- `out/null/<attr>/<tile_id>/{targeted,random}.png`
- `out/null/<attr>/metrics.csv`
- `out/null/<attr>/null_comparison.json`

## Module layout

```
src/a4_uni_probe/
  __init__.py
  main.py                 # CLI: probe | sweep | null | figures
  probe.py                # Stage 1 linear probes + CV + ranking
  edit.py                 # UNI vector editing helpers (sweep + null)
  inference.py            # thin wrapper around existing stage3 ControlNet inference, accepting edited UNI tensors
  metrics.py              # bio-attr extraction (channel stats + CellViT wrappers)
  figures.py              # paper panels A/B/C
  out/
    probe_results.csv
    sweep/<attr>/<tile_id>/alpha=*.png
    sweep/<attr>/{metrics.csv, slope_summary.json}
    null/<attr>/<tile_id>/{targeted,random}.png
    null/<attr>/{metrics.csv, null_comparison.json}
    figures/
      panel_a_probe_R2.png
      panel_b_sweep_slope.png
      panel_c_null_drop.png
```

## Dependencies on existing code

- `diffusion/model/nets/multi_group_tme.py` — inference path must accept overridden UNI tensors without touching TME.
- `stage3_inference.py` / `tools/stage3/tile_pipeline.py` — extend with `--uni-override` path or call `inference.py` directly.
- `src/a2_decomposition/main.py` — reuse `tme_only` cached outputs as full-UNI-null baseline.
- CellViT integration — locate existing wrapper (likely `tools/`); reuse, do not re-implement.

## Scale and compute

- Probe (Stage 1): pure feature math + CellViT on real H&E only — runs on ~10k tiles; CellViT pass is the dominant cost.
- Sweep (Stage 2): 50 tiles × 5 α × 4 attrs = 1000 diffusion generations + CellViT.
- Null (Stage 3): 50 tiles × 2 conditions × 4 attrs = 400 diffusion generations + CellViT.
- Total new generations: ~1400, comparable to `a2_decomposition`'s 2000.

## Paper claim (target sentence)

> UNI uniquely encodes morphology axes {A, B, C} that TME channels cannot recover (preregistered threshold: probe ΔR² ≥ 0.05). Editing UNI along these axes causally moves the corresponding morphology metric in generated H&E (sweep slope ≠ 0, p < 0.01); nulling the axis degrades that metric significantly more than nulling a random direction (paired p < 0.01). This decomposes the model: TME provides spatially-resolved biological layout, UNI provides appearance-level morphology beyond what channels supply.

## Risks / open questions

- **CellViT label noise** — morphology metrics on generated H&E may be noisier than on real H&E (artifacts). Mitigation: filter by CellViT confidence.
- **Probe direction non-linearity** — if a bio attribute is non-linear in UNI, linear-probe direction is suboptimal. Mitigation: report MLP-probe R² as upper bound; use linear `w` for edits because non-linear edits are hard to invert.
- **Channel-derived attributes not measurable on generated H&E** — `mean_oxygen`, fractions etc. have no direct readout from pixels. They appear in Stage 1 (probe) only; sweep/null is restricted to CellViT-measurable morphology.
- **Sweep K=50 may be underpowered** — pilot on K=10 first; scale up if slopes are borderline.

## Out of scope

- New UNI fine-tuning.
- Multi-slide generalization (single slide only).
- Pathologist annotation (none available).
- Non-linear probes for edits.
- HoVer-Net or alternative segmenters.

## Next step

After approval, invoke `superpowers:writing-plans` to produce the implementation plan (broken into milestones for Codex execution per project Claude/Codex role split).
