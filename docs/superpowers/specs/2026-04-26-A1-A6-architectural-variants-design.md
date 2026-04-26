# A1/A6 Architectural Variants — Spec

**Date:** 2026-04-26
**Storyline anchor:** Methods-paper reframe, Section A (P0). Design-justification ablations.
**Scope:** A1 (multi-group TME vs. naive concat ControlNet) and A6 (TME injection mechanism: cross-attention vs. spatial broadcast vs. FiLM). A2, A3, A4, A5 covered by separate specs.

---

## 1. Motivation

The current design has two architectural choices that have not been measured against alternatives:
- **A1:** the multi-group TME module (per-group encoder + per-group cross-attention) versus simpler ControlNet conditioning paths.
- **A6:** the cross-attention injection of TME residuals into ControlNet, versus alternative injection mechanisms (spatial broadcast addition, FiLM modulation).

A methods paper must show these choices were not arbitrary. This spec runs a small head-to-head and produces two SI figures.

## 2. Out of scope

- A2 `zero_mask_latent`, A3 `zero_init_conv_out` (separate spec).
- A4 CFG dropout sweep, A5 per-group dropout sweep (separate spec).
- Sections B, C (we use the existing CellViT pipeline), D, E, F, G.

## 3. Definitions

- **Production design:** multi-group TME with cross-attention injection (current `MultiGroupTMEModule`). Already trained.
- **A1 variants:**
  - **A1.i Concat:** single-encoder ControlNet conditioned on all channels concatenated along the channel axis (mask + cell_types + cell_state + vasculature + microenv → 10-channel input). No TME module. Same ControlNet architecture; the conditioning input projection is widened to 10 channels.
  - **A1.ii Per-channel-no-grouping:** one encoder per individual channel (10 encoders) feeding into the same cross-attention path as production, but without group-level pooling. Tests whether grouping itself helps.
  - **A1.iii Production:** existing multi-group design (4 groups). Reused.
- **A6 variants:**
  - **A6.cross_attn:** production cross-attention injection. Reused.
  - **A6.broadcast:** TME-encoder output is spatially broadcast and added to the ControlNet conditioning latent (no attention, no per-group Q/K/V). Group encoders unchanged.
  - **A6.film:** TME-encoder output is reduced to per-group γ/β scale-and-shift parameters that modulate the ControlNet conditioning latent (FiLM). Group encoders unchanged.

## 4. Experimental matrix

| ID | Variants | Seeds | Schedule |
|----|----------|-------|----------|
| A1 | i Concat, ii Per-channel-no-grouping, iii Production (reused) | 3 (i, ii) | short proxy + 1 full-headline run per variant |
| A6 | cross_attn (reused), broadcast, film | 3 (broadcast, film) | short proxy + 1 full-headline run per variant |

3 seeds chosen because architectural variants are far more expensive than the A2/A3 config flips. A 3-seed run is the minimum for an SD-bar that reviewers will accept.

**Compute-budget normalization:** parameter count and training step count are matched across variants within ±10%. Record actual parameter counts and per-step wall-clock in a spec-results JSON for transparency. If a variant cannot be matched (e.g. A1.ii has many small encoders summing to fewer params than production), train it with extra steps to equalize total FLOPs as best as possible and record both numbers.

## 5. A1 — TME design ablation (`SI_A1_tme_design.png`)

### 5.1 Three-row SI table (top panel)

| Row | Variant | Conditioning path |
|-----|---------|-------------------|
| 1. Concat | A1.i | 10-channel concat → single-encoder ControlNet |
| 2. Per-channel | A1.ii | 10 encoders → cross-attn (no grouping) |
| 3. Production | A1.iii | 4-group encoders → group-wise cross-attn |

### 5.2 Qualitative grid (bottom panel)

3 rows × 4 tiles. Same fixed test tile IDs as the A2/A3 spec, for visual continuity.

## 6. A6 — TME injection mechanism (`SI_A6_injection.png`)

### 6.1 Three-row SI table

| Row | Variant | Injection |
|-----|---------|-----------|
| 1. Cross-attn (production) | A6.cross_attn | per-group K/V into ControlNet conditioning Q |
| 2. Spatial broadcast | A6.broadcast | per-group encoder output → mean-pooled → broadcast-added |
| 3. FiLM | A6.film | per-group encoder output → γ/β → channel-wise affine |

### 6.2 Qualitative grid

Same layout as §5.2, same tile IDs.

## 7. Metric set (both A1 and A6)

Identical to the A2/A3 spec (§7 there): FID, UNI-cos, CellViT cell-count r, CellViT cell-type composition KL, CellViT nuclear-morphology KS. Plus a parameter-count column for A1, since variants differ structurally.

## 8. Eval set

Paired ORION-CRC test split only.

## 9. Components (file-level)

New code:

- `diffusion/model/nets/concat_controlnet.py` — A1.i variant. Subclasses or wraps the existing ControlNet so its conditioning input projection accepts 10 channels. Reuses everything else.
- `diffusion/model/nets/per_channel_tme.py` — A1.ii variant. A `PerChannelTMEModule` with one encoder per individual channel feeding the existing cross-attention head (no grouping).
- `diffusion/model/nets/multi_group_tme_broadcast.py` — A6.broadcast variant. Drop-in replacement for `MultiGroupTMEModule` that returns a broadcast-added residual instead of cross-attention output.
- `diffusion/model/nets/multi_group_tme_film.py` — A6.film variant. Per-group FiLM modulator.
- `configs/config_controlnet_exp_a1_concat.py`
- `configs/config_controlnet_exp_a1_per_channel.py`
- `configs/config_controlnet_exp_a6_broadcast.py`
- `configs/config_controlnet_exp_a6_film.py`
- `tools/ablation_a1/__init__.py` (empty)
- `tools/ablation_a6/__init__.py` (empty)
- `src/paper_figures/fig_si_a1_tme_design.py` — figure builder.
- `src/paper_figures/fig_si_a6_injection.py` — figure builder.
- `tests/test_concat_controlnet.py` — shape + forward smoke test.
- `tests/test_per_channel_tme.py` — shape + active-channel gating test.
- `tests/test_multi_group_tme_broadcast.py` — shape + zero-init identity.
- `tests/test_multi_group_tme_film.py` — shape + zero-γ identity (γ=0,β=0 ⇒ identity).

Reused without change:
- `train_scripts/train_controlnet_exp.py` — config-flag driven; new variants register their module class via the `tme_model` config string.
- Paired-exp dataset, latent caches, UNI embeddings, CellViT pipeline.

## 10. Output artifacts

- `figures/pngs/SI_A1_tme_design.png` and `figures/pngs_updated/SI_A1_tme_design.png`.
- `figures/pngs/SI_A6_injection.png` and `figures/pngs_updated/SI_A6_injection.png`.
- `inference_output/a1_tme_design/<variant>/<tile_id>.png`.
- `inference_output/a6_injection/<variant>/<tile_id>.png`.
- `inference_output/a1_tme_design/metrics_summary.json`.
- `inference_output/a6_injection/metrics_summary.json`.

## 11. Failure modes and fallbacks

- **Variant fails to train at production schedule.** This is itself the result. Report best achievable and mark training-instability cell.
- **Compute-budget gap > 10%.** Train the cheaper variant for proportionally more steps; record both raw and FLOP-matched final numbers in the SI table caption.
- **A1.ii with no grouping leaks too many parameters.** Cap per-channel encoder hidden width so total parameter count ≈ production within 10%.
- **A6.broadcast spatial broadcast loses spatial structure.** Acceptable — that's what we're measuring. Don't tune it post-hoc.

## 12. Testing

Each new module ships a unit test:

- `test_concat_controlnet.py`: forward pass with 10-channel input produces a tensor of the same shape ControlNet expects from the production conditioning path.
- `test_per_channel_tme.py`: 10 encoders are instantiated, each is exercised by its respective channel, and a missing-channel input falls through cleanly.
- `test_multi_group_tme_broadcast.py`: zero-initialized residual yields identity (same as production module's identity test).
- `test_multi_group_tme_film.py`: γ=0 and β=0 yields identity.

No new tests for figure builders.

## 13. Caption requirements

- A1 caption must report parameter counts and per-step wall-clock for each variant.
- A6 caption must describe each injection mechanism in one sentence.
- Both captions must report seeds, short-proxy step count, and headline-run step count.

## 14. Open assumptions to verify before training

- ControlNet input projection can be widened to 10 channels without breaking the loaded production ControlNet weights — needs a clean re-init of that one layer.
- TME registry (`tme_model = "MultiGroupTMEModule"`) supports plug-in via config string; if not, factor a small registry into `diffusion/model/builder.py` first.
- Compute budget for 4 architectural variants × 3 seeds × short-proxy + 4 full-headline runs is available.

## 15. Acceptance criteria

1. Both SI figures exist with all panels populated.
2. All metrics in §7 populated for every row.
3. Captions meet §13.
4. Unit tests in §12 pass.
5. Compute-budget table is recorded in `inference_output/a1_tme_design/metrics_summary.json` (per-variant params and per-step wall-clock).
