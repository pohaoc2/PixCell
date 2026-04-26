# A2/A3 Design-Justification Ablations — Spec

**Date:** 2026-04-26
**Storyline anchor:** Methods-paper reframe, Section A (P0). Design-justification ablations.
**Scope:** A2 (`zero_mask_latent` post-TME subtraction) and A3 (zero-init residual gating). A1, A4–A6 covered by separate specs.

---

## 1. Motivation

A methods paper must justify each architectural choice with evidence, not assertion. Two of the current design's load-bearing flags — `zero_mask_latent=True` (post-TME subtraction) and `zero_init_conv_out=True` (residual gating) — are currently asserted as best practice but not measured against alternatives. This spec runs minimal ablations on both flags and produces two SI figures.

## 2. Out of scope

- A1 multi-group TME vs. naive concat (separate spec).
- A4 CFG dropout sweep and A5 per-group dropout (separate spec).
- A6 TME injection mechanism (cross-attention vs. spatial broadcast vs. FiLM) (separate spec).
- Section B (robustness/generalization), C (cell-level fidelity infrastructure — we *use* the existing CellViT pipeline but do not extend it here), D, E, F, G.

## 3. Definitions

- **Production design:** `zero_mask_latent=True`, `zero_init_conv_out=True`, full TME at inference. Already trained.
- **Bypass design (A2 variant):** `zero_mask_latent=False`. Newly trained for this spec.
- **No-zero-init design (A3 variant):** `zero_init_conv_out=False`. Newly trained for this spec.
- **Bypass probe:** at inference, force TME outputs to zero on the bypass-design checkpoint, leaving condition = `mask_latent`.
- **Off-the-shelf reference:** original PixCell ControlNet checkpoint at `pretrained_models/pixcell-256-controlnet`, run inference on the same paired ORION test tiles with mask-only spatial conditioning + matching UNI embeddings. **No fine-tuning.**

## 4. Experimental matrix

| ID | Flag | Variants | Seeds | Schedule |
|----|------|----------|-------|----------|
| A2 | `zero_mask_latent` | True (production), False (newly trained) | 5 | short proxy + 1 full-headline run for the False variant |
| A3 | `zero_init_conv_out` | True (production), False (newly trained) | 5 | short proxy + 1 full-headline run for the False variant |

Production checkpoints (True/True) are reused — no retraining. Only False variants are newly trained.

**Short proxy** = a fixed fraction of full schedule sufficient to rank designs; final value chosen empirically against an existing run (target: ≤ 25% of full steps; verified by checking that paired-test FID ranking at the proxy length matches ranking at full length on a single reference seed). Locked at decomposition.

**Full-headline** = the production training schedule, used once per ablation only for the variant being defended (False), so the SI table reports at least one apples-to-apples full-budget comparison vs. production.

## 5. A2 — bypass probe

### 5.1 Three-row SI table (`SI_A2_bypass_probe.png`, top panel)

| Row | Training | Inference condition | Purpose |
|-----|----------|--------------------|---------|
| 1. Production | `zero_mask_latent=True` | full TME (= `tme(mask_latent, channels)`) | reference |
| 2. Bypass probe | `zero_mask_latent=False` | TME outputs zeroed → condition = `mask_latent` | measures how much the False model leans on the mask-latent shortcut |
| 3. Off-the-shelf PixCell | original PixCell ControlNet (no fine-tune) | mask-only conditioning + UNI | anchors the bypass-probe interpretation |

### 5.2 Architectural note recorded in caption

Under `zero_mask_latent=True` the conditioning fed to ControlNet is `mask_latent + (tme(mask_latent, channels) - mask_latent) = tme(mask_latent, channels)`; the mask latent is routed entirely through TME. Zeroing TME at inference on this model collapses the conditioning to zero — a degenerate probe. The bypass probe is therefore meaningful only on the `zero_mask_latent=False` variant, where the conditioning is additive (`mask_latent + tme(...)`) and zeroing TME cleanly reduces to `mask_latent`. This must be stated in the caption to pre-empt reviewer confusion with the production design.

### 5.3 Qualitative grid (bottom panel)

3 rows × N tiles (N = 4, fixed test tile IDs from existing paired-test split). Same tiles as the existing main paired ablation grid for visual continuity. Each row is one of the three table conditions; columns are tiles; each cell shows generated H&E.

### 5.4 Off-the-shelf reference: implementation notes

- Use existing UNI-2h embeddings cached for paired test tiles.
- Feed `cell_masks` channel as the spatial control input.
- Verify the original checkpoint's expected mask format matches our cached representation; convert if needed (one-shot script under `tools/baselines/pixcell_offshelf_inference.py`).
- No fine-tuning. Inference only.

## 6. A3 — zero-init residual gating

### 6.1 Variants

- True (production, reused).
- False, 5 seeds, short proxy. One additional full-headline run on the False variant for a fair apples-to-apples row.

### 6.2 SI figure layout (`SI_A3_zero_init.png`)

Three panels, top to bottom:

1. **Loss-curve plot.** Mean ± std loss across 5 seeds for True and False, log-x step axis. Highlights early-training divergence behavior of the False variant.
2. **Divergence-rate bar chart.** For each variant, fraction of seeds that diverged (NaN loss, gradient explosion threshold, or final FID worse than fixed cutoff). Same 5-seed pool.
3. **Summary table.** Columns: variant, mean loss at fixed step (e.g., step 10k of short proxy), std loss across seeds at that step, divergence count, paired-test FID, paired-test UNI-cos, CellViT cell-count match, cell-type composition KL, nuclear-morphology KS.

A3's contribution to the methodological story is the **stability difference**, not just the final-FID difference. The figure must visibly show that.

### 6.3 Divergence definition

A seed is marked "diverged" if any of:
- NaN appears in loss within the short-proxy run, OR
- Gradient norm exceeds a fixed threshold (logged; threshold set at 100× the median gradient norm of the True variant on the first 1k steps), OR
- Final paired-test FID is more than 2× the True variant's mean final FID.

Threshold values are recorded in the spec results, not chosen post-hoc.

## 7. Metric set (both A2 and A3)

| Metric | Source | Notes |
|--------|--------|-------|
| FID | existing pipeline | paired-test split |
| UNI-cos | existing pipeline | paired-test split |
| CellViT cell-count match | existing CellViT outputs for production; new runs reuse the same pipeline | per cell-type Pearson r between real and generated tile counts |
| CellViT cell-type composition | as above | KL divergence between real and generated tile composition vectors |
| CellViT nuclear-morphology KS | as above | nucleus area distribution KS-distance, real vs. generated |
| Training-stability (A3 only) | new training logs | loss mean/std at fixed step, divergence count |

CellViT pipeline reuse: the existing CellViT runner already accepts a directory of PNGs and produces per-tile JSON with cell-type counts and nuclear morphology. New variant inference outputs PNGs to a sibling directory; same runner is invoked.

## 8. Eval set

- Paired ORION-CRC test split only (per scoping decision Q2c).
- Unpaired generalization is the existing main figure's job.

## 9. Components (file-level)

New code or scripts:

- `train_scripts/train_controlnet_exp.py` — already supports config-flag overrides; no code change. Two new config files added under `configs/` for the False-variant runs (one per ablation).
- `tools/baselines/pixcell_offshelf_inference.py` — new. Inference-only wrapper for the original PixCell ControlNet checkpoint over paired test tiles. Outputs PNGs in the standard ablation-output directory layout so existing CellViT and metric tools work unchanged.
- `tools/ablation_a2/run_bypass_probe.py` — new. Loads the trained False-variant checkpoint, performs inference with TME output zeroed, writes PNGs.
- `tools/ablation_a3/aggregate_stability.py` — new. Reads training logs from short-proxy runs, computes loss-mean/std at fixed step and divergence flags, writes a JSON summary used by the figure builder.
- `src/paper_figures/fig_si_a2_bypass.py` — new. Builds `SI_A2_bypass_probe.png` (table + qualitative grid).
- `src/paper_figures/fig_si_a3_zero_init.py` — new. Builds `SI_A3_zero_init.png` (loss curves + divergence bar + summary table).
- `src/paper_figures/main.py` — register the two new SI figure builders, save to `figures/pngs/` and `figures/pngs_updated/`.
- `src/paper_figures/style.py` — no change expected; use existing constants. New per-figure metric panel may need `FONT_SIZE_TICK` / `FONT_SIZE_LABEL` only.

Reused without change:

- `MultiGroupTMEModule` — config flag `zero_mask_latent` already toggles the post-TME subtraction.
- ControlNet `zero_init_conv_out` flag.
- Paired-exp dataset, latent caches, UNI embeddings.
- CellViT pipeline.

## 10. Output artifacts

- `figures/pngs/SI_A2_bypass_probe.png` and `figures/pngs_updated/SI_A2_bypass_probe.png`.
- `figures/pngs/SI_A3_zero_init.png` and `figures/pngs_updated/SI_A3_zero_init.png`.
- `inference_output/a2_bypass/<variant>/<tile_id>.png` for generated tiles.
- `inference_output/a3_zero_init/<variant>_seed<k>/...` for per-seed outputs.
- `inference_output/a2_bypass/metrics_summary.json`.
- `inference_output/a3_zero_init/stability_summary.json`.

## 11. Failure modes and fallbacks

- **Off-the-shelf checkpoint mask-format mismatch.** If the original PixCell ControlNet expects a different cell-mask representation (e.g., binary vs. probability map), document the conversion in `pixcell_offshelf_inference.py` and verify outputs visually before computing metrics.
- **CellViT pipeline mismatch on new tile outputs.** If output tile resolution or filename pattern differs from production runs, normalize at inference write-time, not at CellViT time.
- **Short proxy ranking flip.** If at the proxy length the FID ranking between True and False contradicts the full-headline run, report both numbers in the SI and use the full-headline ranking for the headline conclusion. Do not retroactively extend the proxy.
- **All seeds diverge on False zero-init variant.** This is itself the result. Report 5/5 divergence in the bar chart and use a single short-trajectory loss curve for visualization.

## 12. Testing

New unit tests:

- `tests/test_pixcell_offshelf_inference.py` — smoke test that the wrapper loads the checkpoint, processes a single tile, produces a non-empty PNG of expected size.
- `tests/test_bypass_probe.py` — verifies that `run_bypass_probe.py` zeroes TME outputs by inspecting the conditioning tensor at a fixed step and checking it equals the mask latent (within float tolerance).
- `tests/test_aggregate_stability.py` — fixture training logs with a known divergence pattern; assert the stability summary correctly identifies divergence cases per the threshold rules.

No new tests for figure builders; matplotlib output is verified by visual inspection.

## 13. Caption requirements (for both SI figures)

- A2 caption must include the architectural note from §5.2 explaining why the bypass probe is meaningful only on the False variant.
- A3 caption must report the divergence-threshold values used.
- Both captions must report the short-proxy step count and the headline-run step count.
- Both captions must list the seeds used.

## 14. Open assumptions to verify before training

- Production True/True checkpoints have logs sufficient to compute the A3 loss-curve mean/std at the chosen fixed step. If older runs lack the necessary log granularity, retrain one True seed under the new logging scheme as a reference.
- Original PixCell ControlNet checkpoint accepts `cell_masks` channel directly. If not, conversion script needed.
- 5 seeds × short-proxy training fits within available compute. If not, drop to 3 seeds for A3 (loss-stability claim becomes weaker but still defensible) — record the reduction in spec results, not the spec itself.

## 15. Acceptance criteria

The spec is implemented when:

1. `figures/pngs/SI_A2_bypass_probe.png` and `SI_A3_zero_init.png` exist and contain all panels described in §5–§6.
2. All metrics in §7 are populated for every row of every table, with no missing cells.
3. Captions meet §13.
4. Unit tests in §12 pass.
5. The spec's interpretation in §5.2 is reflected in the figure caption.
