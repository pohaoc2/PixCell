# A4/A5 Numerical Sweeps — Spec

**Date:** 2026-04-26
**Storyline anchor:** Methods-paper reframe, Section A (P0). Design-justification ablations.
**Scope:** A4 (CFG dropout sweep on UNI) and A5 (per-group dropout sweep). A1, A2, A3, A6 covered by separate specs.

---

## 1. Motivation

Two probability hyperparameters are currently asserted-not-measured:
- **A4 `cfg_dropout_prob = 0.15`** — controls the probability that UNI embeddings are zeroed during training, enabling TME-only inference. The choice of 0.15 is undocumented.
- **A5 `group_dropout_probs = {cell_types: 0.10, cell_state: 0.10, vasculature: 0.15, microenv: 0.20}`** — heterogeneous per-group dropout rates. Never ablated.

A methods paper must show these were tuned, not guessed. This spec runs lightweight sweeps over both and produces two SI figures.

## 2. Out of scope

- A1 multi-group vs. concat, A6 injection mechanism (separate spec).
- A2 `zero_mask_latent`, A3 `zero_init_conv_out` (separate spec).
- Sections B, C (we use existing CellViT pipeline), D, E, F, G.

## 3. Definitions

- **A4 sweep set:** `cfg_dropout_prob ∈ {0.0, 0.1, 0.15 (production), 0.3, 0.5}`. Five points.
- **A5 sweep mode:** uniform-rate sweep across all four groups simultaneously: `p ∈ {0.0, 0.1, 0.2, 0.3}`, plus the production heterogeneous setting (cell_types=0.10, cell_state=0.10, vasculature=0.15, microenv=0.20) as a fifth anchor. Five points total. Per-group independent sweeps are deferred to a follow-up spec if A5 results suggest heterogeneity matters.
- **TME-only inference:** at inference, set UNI embeddings to zero so generation depends entirely on TME conditioning. Tests A4's main claim that high CFG dropout enables a useful TME-only mode.

## 4. Experimental matrix

| ID | Sweep | Points | Seeds per point | Schedule |
|----|-------|--------|-----------------|----------|
| A4 | `cfg_dropout_prob` | 5 (0.0, 0.1, 0.15, 0.3, 0.5) | 1 per non-anchor; 3 at the production anchor (0.15) for variance | short proxy |
| A5 | uniform `p` across groups + production heterogeneous | 5 | 1 per non-anchor; 3 at production anchor for variance | short proxy |

This gives `4 + 3 = 7` runs per sweep × 2 sweeps = 14 short-proxy runs, plus reuse of production seeds where they exist. The 3-seed anchor provides an SD bar at one point so the sweep curves can be read against it.

No full-headline runs in this spec — the sweep's job is the ranking and the curve shape, not absolute headline numbers. The A2/A3 spec already provides one full-budget production reference.

## 5. A4 — CFG dropout sweep (`SI_A4_cfg_sweep.png`)

### 5.1 Two-panel SI figure

**Top panel — Paired-quality vs. CFG dropout.** Line plot, x-axis `cfg_dropout_prob ∈ {0.0, 0.1, 0.15, 0.3, 0.5}`. Two lines:
- FID (paired test, full UNI + TME inference).
- UNI-cos (paired test, full UNI + TME inference).
Production anchor (0.15) drawn with mean ± SD bar across its 3 seeds.

**Bottom panel — TME-only quality vs. CFG dropout.** Same x-axis. Two lines:
- FID (paired test, UNI = 0 at inference, TME only).
- UNI-cos.
This is the panel that tests the A4 design claim: as `cfg_dropout_prob` rises, the model should learn to operate without UNI; FID under TME-only should drop monotonically (or plateau).

### 5.2 Summary table (caption-side, optional)

CellViT triple (cell-count r, type KL, nuc KS) for each point under paired (full) inference. Compact 5-row table; no qualitative grid for A4.

## 6. A5 — Per-group dropout sweep (`SI_A5_group_dropout.png`)

### 6.1 Single-panel SI figure

Line plot, x-axis = uniform `p ∈ {0.0, 0.1, 0.2, 0.3}` plus a single point labeled "Het. (production)". Three lines:
- FID
- UNI-cos
- Cell-count r

Production anchor drawn with mean ± SD across 3 seeds. The interpretation is: if the heterogeneous production setting beats every uniform-`p` point, heterogeneity is justified; if not, replace with the best uniform-`p`.

### 6.2 Optional follow-up note

If the heterogeneous setting wins by a margin > the production anchor's SD, the spec results recommend keeping it and the paper text claims "tuned." If it loses, the paper text recommends the best uniform `p` and the production design is updated in a follow-up PR.

## 7. Metric set

Identical to A2/A3 spec §7: FID, UNI-cos, CellViT cell-count r, CellViT cell-type composition KL, CellViT nuclear-morphology KS. A4 additionally reports the same metrics under TME-only inference (UNI = 0 at inference time).

## 8. Eval set

Paired ORION-CRC test split only.

## 9. Components (file-level)

New code:

- `configs/config_controlnet_exp_a4_cfg{0_0,0_1,0_3,0_5}.py` — four variant configs differing only in `cfg_dropout_prob`. (Production 0.15 reused.)
- `configs/config_controlnet_exp_a5_group{0_0,0_1,0_2,0_3}.py` — four variant configs differing only in `group_dropout_probs` (uniform value).
- `tools/ablation_a4/__init__.py` (empty).
- `tools/ablation_a4/run_tme_only_inference.py` — wraps existing inference but forces UNI embeddings to zero.
- `tools/ablation_a4/aggregate_sweep.py` — reads metrics across A4 sweep points, emits a single sweep_summary.json keyed by `cfg_dropout_prob`.
- `tools/ablation_a5/__init__.py` (empty).
- `tools/ablation_a5/aggregate_sweep.py` — same idea for A5, keyed by sweep label (`p=0.0`, …, `het_production`).
- `src/paper_figures/fig_si_a4_cfg_sweep.py`
- `src/paper_figures/fig_si_a5_group_dropout.py`
- `tests/test_run_tme_only_inference.py` — verifies UNI zeroing at the call site.
- `tests/test_aggregate_sweep_a4.py` — verifies sweep summary schema and key ordering.
- `tests/test_aggregate_sweep_a5.py` — same for A5.

Reused without change:
- `train_scripts/train_controlnet_exp.py` — config-flag driven.
- Paired-exp dataset, latent caches, UNI embeddings, CellViT pipeline.
- `tools/baselines/pixcell_offshelf_inference.py` — not used here.
- `tools/ablation_a3/aggregate_stability.py` — not used here.

## 10. Output artifacts

- `figures/pngs/SI_A4_cfg_sweep.png` and `figures/pngs_updated/SI_A4_cfg_sweep.png`.
- `figures/pngs/SI_A5_group_dropout.png` and `figures/pngs_updated/SI_A5_group_dropout.png`.
- `inference_output/a4_cfg_sweep/<point>/{paired,tme_only}/<tile_id>.png`.
- `inference_output/a5_group_dropout/<point>/<tile_id>.png`.
- `inference_output/a4_cfg_sweep/sweep_summary.json`.
- `inference_output/a5_group_dropout/sweep_summary.json`.

## 11. Failure modes and fallbacks

- **TME-only inference produces noise across the entire sweep.** Acceptable result for low CFG-dropout points; this is the test of A4's claim. Plot a zero or "off-chart" marker rather than dropping the point.
- **A5 heterogeneous setting underperforms uniform.** Report honestly. Update the paper text to recommend the better setting.
- **Sweep ranking unstable across seeds at a single anchor.** If the 3-seed production anchor's SD overlaps adjacent sweep points, widen the anchor seed pool to 5 and re-render.

## 12. Testing

- `test_run_tme_only_inference.py`: assert that the inference helper passes `y = zeros_like(uni_embedding)` to the model when `tme_only=True`.
- `test_aggregate_sweep_a4.py`: feed a small fixture metrics tree, assert sweep_summary.json has 5 points keyed by sweep value, each with `paired` and `tme_only` blocks containing the metric set.
- `test_aggregate_sweep_a5.py`: same shape check, 5 points keyed by sweep label including `het_production`.

## 13. Caption requirements

- A4 caption must explicitly state which panel uses full UNI inference and which uses TME-only.
- A5 caption must list the production heterogeneous values for transparency.
- Both captions must report short-proxy step count, seeds, and the metric definitions used (especially KL base, KS distance variant).

## 14. Open assumptions to verify before running

- Same short-proxy step count from the A2/A3 spec applies; if not, lock independently here using a single uniform-anchor pilot.
- CellViT pipeline runs on TME-only outputs without modification (file naming and tile resolution match).
- Existing production seeds are sufficient for the 3-seed anchor; if not, train two additional production seeds at short-proxy length.

## 15. Acceptance criteria

1. Both SI figures exist with all panels populated.
2. All metrics in §7 populated for every sweep point (under both inference modes for A4).
3. Captions meet §13.
4. Unit tests in §12 pass.
5. `sweep_summary.json` files exist and conform to the documented schema.
