# SI A1+A2 Unified Ablation Figure — Design Spec

**Date:** 2026-04-27
**Anchor:** Methods paper, SI section — design justification ablations A1 (TME architecture) and A2 (bypass path).
**Output file:** `figures/pngs/SI_A1_A2_unified.png`

---

## 1. Purpose

Single SI figure consolidating A1 and A2 ablation results. Justifies two architectural choices:

- **A1:** multi-group TME (per-group encoder + cross-attention) over naive concat or ungrouped per-channel conditioning.
- **A2:** `zero_mask_latent=True` (bypass-closed production) over bypass-probe and off-the-shelf baselines.

Reviewer question answered: *"Why this architecture and not something simpler?"*

---

## 2. Variants

### A1 — TME design
| ID | Variant | Conditioning path |
|----|---------|-------------------|
| A1.i | Concat | 10 channels concatenated → single ControlNet encoder, no TME module |
| A1.ii | Per-channel | 10 individual encoders → cross-attn (no semantic grouping) |
| A1.iii ★ | Production | 4-group encoders → group-wise cross-attn (current design) |

### A2 — Bypass path
| ID | Variant | Description |
|----|---------|-------------|
| A2.i | Bypass probe | `zero_mask_latent=False`, TME=0 — bypass path open |
| A2.ii | Off-the-shelf | Mask-only PixCell, no fine-tuning |
| A2.iii ★ | Production | `zero_mask_latent=True`, full TME (same checkpoint as A1.iii) |

★ Production is the same checkpoint for both axes.

---

## 3. Figure Layout (3 sections, top → bottom)

### Section 1 — Training Curves

Three subplots side by side:

1. **A1 Loss over steps** — one curve per A1 variant (Production, Concat, Per-channel), 3 seeds shaded ±1σ band.
2. **A1 Grad norm** (log scale) — same variants. Per-channel curve clips at axis ceiling with upward arrow and "∞" annotation; red zone band at top of plot.
3. **A2 Loss over steps** — Production, Bypass probe, Off-the-shelf. Off-the-shelf has no fine-tuning so shown as eval-only horizontal marker.

**Conditional A2 grad norm subplot:** If A2 weights (still downloading as of 2026-04-27) show gradient explosion on inspection, add a fourth subplot mirroring the A1 grad norm format. Implementation must check for this and include the subplot when warranted. The figure script should accept a flag `--a2_grad_norm` to enable it.

Per-channel (A1.ii) styling: red curve (`#f44336`), solid line, jagged to convey instability. Annotate explosion point with circle callout + "∞" label.

### Section 2 — Master Ablation Table

Single table covering both axes, grouped by axis. Columns:

| Design axis | Variant | FID↓ | UNI-cos↑ | CellViT r↑ | Type KL↓ | Nuc KS↓ | Params |
|-------------|---------|------|----------|------------|----------|---------|--------|

- A1 group: 3 rows (Concat, Per-channel ⚠, Production ★). Params column populated for A1 only (variants differ structurally).
- A2 group: 3 rows (Bypass, Off-the-shelf, Production ★). Params column shows "shared" (same architecture).
- Production ★ row repeated once per axis group (same numbers, makes within-group comparison self-contained).
- Per-channel row: amber/red background, metrics reported from last finite-loss checkpoint, footnote: "⚠ training unstable (∞ grad norm); metrics from last finite checkpoint."
- Best-performing row per axis group bold-highlighted.

### Section 3 — Qualitative Tile Grid

Fixed test tile IDs (same 4 tiles used across all ablation SI figures for visual continuity).

Row order:
1. GT H&E (reference)
2. A1.i Concat
3. A1.ii Per-channel ⚠ — diagonal hatch background, red border, "∞" badge bottom-right corner
4. ★ Production (shared)
5. *(dashed separator line labeled "A2 variants")*
6. A2.i Bypass probe
7. A2.ii Off-the-shelf

Columns: 4 test tiles.

---

## 4. Styling

Matches existing `src/paper_figures/style.py` (Nature Communications compatible):
- Font: DejaVu Sans, sizes per `style.py` constants
- Per-channel instability: red (`#f44336`), diagonal hatch (`////`), amber row background in table
- Production: green (`#4caf50`), bold border in tile grid
- A1 Concat: blue (`#2196f3`), dashed line in curves
- A2 Bypass: orange (`#ff9800`), dashed line
- A2 Off-the-shelf: purple (`#9c27b0`), dotted line

---

## 5. Output Artifacts

- `figures/pngs/SI_A1_A2_unified.png`
- `figures/pngs_updated/SI_A1_A2_unified.png`
- `inference_output/a1_tme_design/<variant>/<tile_id>.png` — per-variant tiles
- `inference_output/a2_bypass/<variant>/<tile_id>.png`
- `inference_output/a1_tme_design/metrics_summary.json`
- `inference_output/a2_bypass/metrics_summary.json`

---

## 6. Implementation Files

- `src/paper_figures/fig_si_a1_a2_unified.py` — figure builder (replaces separate `fig_si_a1_tme_design.py` and existing `fig_si_a2_bypass.py`)
- Reuses: `tools/stage3/tile_pipeline.py`, `tools/stage3/run_evaluation.py`, `src/paper_figures/style.py`
- Inference must be run per variant before figure build; see Section 7.

---

## 7. Inference Prerequisite

Before building the figure, run inference for each variant against the 4 fixed test tiles:

```
conda activate pixcell
python stage3_inference.py \
  --config configs/config_controlnet_exp_a1_concat.py \
  --checkpoint checkpoints/a1_concat/full_seed_42/checkpoint/step_0002600/controlnet_epoch_20_step_2600.pth \
  --tme_checkpoint checkpoints/a1_concat/full_seed_42/checkpoint/step_0002600/tme_module.pth \
  --output_dir inference_output/a1_tme_design/concat

python stage3_inference.py \
  --config configs/config_controlnet_exp_a1_per_channel.py \
  --checkpoint checkpoints/a1_per_channel/full_seed_42/checkpoint/step_0002600/controlnet_epoch_20_step_2600.pth \
  --tme_checkpoint checkpoints/a1_per_channel/full_seed_42/checkpoint/step_0002600/tme_module.pth \
  --output_dir inference_output/a1_tme_design/per_channel

python stage3_inference.py \
  --config configs/config_controlnet_exp.py \
  --checkpoint checkpoints/pixcell_controlnet_exp/npy_inputs/controlnet_epoch_20_step_2600.pth \
  --tme_checkpoint checkpoints/pixcell_controlnet_exp/npy_inputs/tme_module.pth \
  --output_dir inference_output/a1_tme_design/production

# A2 variants — run once weights finish downloading
```

All A1 checkpoints confirmed present as of 2026-04-27. A2 weights still pending.

---

## 8. Metrics

FID, UNI-cos, CellViT cell-count r, CellViT cell-type composition KL, CellViT nuclear-morphology KS. Computed on paired ORION-CRC test split only.

Per-channel metrics: report from last finite-loss checkpoint. Note in caption.

---

## 9. Caption Requirements

- Report seeds, proxy step count, headline step count per variant.
- A1 caption: param count and per-step wall-clock for each variant.
- A2 caption: one sentence per variant describing conditioning path.
- Per-channel footnote: explain ∞ grad norm, which checkpoint metrics are from.
- Production note: same checkpoint serves as reference for both A1 and A2.

---

## 10. Open Items (as of 2026-04-27)

- A2 weights still downloading — confirm whether bypass/off-the-shelf runs show gradient explosion; add A2 grad norm subplot if so (use `--a2_grad_norm` flag).
- All A1 checkpoints now complete (confirmed 2026-04-27).
- A2 weights still downloading — unblock A2 inference once complete.
- Verify 4 fixed test tile IDs match those used in A2/A3 specs for visual continuity.
- Note: `checkpoints/checkpoint/step_0002600/` exists as a stray dir — confirm whether this is production or an artifact before using.
- `src/paper_figures/fig_si_a2_bypass.py` already exists and is superseded by the unified script; archive or delete it to avoid confusion.
