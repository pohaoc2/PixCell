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
| A1.ii | Per-channel ⚠ | 10 individual encoders → cross-attn (no semantic grouping); ∞ grad norm from step 50 |
| A1.iii ★ | Production | 4-group encoders → group-wise cross-attn (current design) |

### A2 — Bypass path
| ID | Variant | Description |
|----|---------|-------------|
| A2.i | Bypass probe ⚠ | `zero_mask_latent=False`, TME=0 — bypass path open; ∞ grad norm from step 50 (all seeds) |
| A2.ii | Off-the-shelf | Mask-only PixCell, no fine-tuning |
| A2.iii ★ | Production | `zero_mask_latent=True`, full TME (same checkpoint as A1.iii) |

★ Production is the same checkpoint for both axes.
⚠ Both A1.ii and A2.i show ∞ grad norm from step 50 across all seeds — confirmed 2026-04-27.

---

## 3. Weight-Free Reproducibility

**Core principle:** weights are needed once to build the cache; the figure rebuilds from the cache alone.

### 3.1 Cache layout

```
inference_output/si_a1_a2/
  cache.json           # all numeric data for the figure
  tiles/
    gt/                # ground-truth H&E crops
    a1_concat/
    a1_per_channel/
    production/
    a2_bypass/
    a2_off_shelf/
```

### 3.2 `cache.json` schema

```json
{
  "generated": "YYYY-MM-DD",
  "tile_ids": ["tile_001", "tile_002", "tile_003", "tile_004"],
  "training_curves": {
    "<variant>": {
      "<run_id>": [{"step": int, "loss": float, "grad_norm": float_or_inf}, ...]
    }
  },
  "metrics": {
    "<variant>": {
      "fid": float,
      "uni_cos": float,
      "cellvit_count_r": float,
      "cellvit_type_kl": float,
      "cellvit_nuc_ks": float,
      "note": "optional string, e.g. last finite checkpoint"
    }
  },
  "params": {
    "a1_concat": int,
    "a1_per_channel": int,
    "production": int
  }
}
```

`grad_norm` values of `Infinity` serialized as the JSON string `"inf"` (not `null`); the figure builder converts back to `float('inf')` for plotting.

### 3.3 Cache builder script

New file: `tools/ablation_a1_a2/build_cache.py`

Steps (run once with weights present, `conda activate pixcell`):

1. **Extract training curves** — read all `train_log.jsonl` files listed below; write into `cache.json["training_curves"]`. No weights needed for this step.
2. **Run inference** — generate tiles for each variant on the 4 fixed tile IDs; save to `tiles/<variant>/`. Needs weights.
3. **Compute metrics** — FID, UNI-cos, CellViT suite on generated vs GT tiles; write into `cache.json["metrics"]`.
4. **Record params** — count trainable parameters for each A1 variant; write into `cache.json["params"]`.

After `build_cache.py` succeeds, weights may be deleted. Figure is fully reproducible from `cache.json` + `tiles/`.

### 3.4 Training log sources

| Variant | Run | Path |
|---------|-----|------|
| a1_concat | full_seed_42 | `checkpoints/a1_concat/full_seed_42/train_log.jsonl` |
| a1_concat | seed_1 | `checkpoints/a1_concat/seed_1/train_log.jsonl` |
| a1_concat | seed_2 | `checkpoints/a1_concat/seed_2/train_log.jsonl` |
| a1_concat | seed_3 | `checkpoints/a1_concat/seed_3/train_log.jsonl` |
| a1_per_channel | full_seed_42 | `checkpoints/a1_per_channel/full_seed_42/train_log.jsonl` |
| a1_per_channel | seed_1 | `checkpoints/a1_per_channel/seed_1/train_log.jsonl` |
| a1_per_channel | seed_2 | `checkpoints/a1_per_channel/seed_2/train_log.jsonl` |
| a1_per_channel | seed_3 | `checkpoints/a1_per_channel/seed_3/train_log.jsonl` |
| a2_bypass | full_seed_42 | `checkpoints/a2_a3/a2_bypass/full_seed_42/train_log.jsonl` |
| a2_bypass | seed_1 | `checkpoints/a2_a3/a2_bypass/seed_1/train_log.jsonl` |
| a2_bypass | seed_2 | `checkpoints/a2_a3/a2_bypass/seed_2/train_log.jsonl` |
| production | (from pixcell_controlnet_exp log) | `checkpoints/pixcell_controlnet_exp/train_log.log` |

### 3.5 Inference checkpoints (one-time, then deletable)

| Variant | Controlnet | TME module |
|---------|-----------|-----------|
| a1_concat | `checkpoints/a1_concat/full_seed_42/checkpoint/step_0002600/controlnet_epoch_20_step_2600.pth` | `…/tme_module.pth` |
| a1_per_channel | `checkpoints/a1_per_channel/full_seed_42/checkpoint/step_0002600/controlnet_epoch_20_step_2600.pth` | `…/tme_module.pth` |
| production | `checkpoints/pixcell_controlnet_exp/npy_inputs/controlnet_epoch_20_step_2600.pth` | `…/tme_module.pth` |
| a2_bypass | `checkpoints/a2_a3/a2_bypass/full_seed_42/…` | pending |
| a2_off_shelf | N/A — uses pretrained PixCell-256 directly | — |

---

## 4. Figure Layout (3 sections, top → bottom)

### Section 1 — Training Curves

Four subplots side by side:

1. **A1 Loss over steps** — Production (green), Concat (blue dashed), Per-channel (red); 3 seeds ±1σ shaded.
2. **A1 Grad norm** (log scale) — same variants. Per-channel clips at axis ceiling; upward arrow + "∞" annotation; red zone band at top.
3. **A2 Loss over steps** — Production, Bypass probe (orange dashed), Off-the-shelf (purple dotted; eval-only horizontal marker, no fine-tuning).
4. **A2 Grad norm** (log scale) — Bypass probe clips at ceiling with same styling as A1 per-channel. Production stable.

Instability annotation style (both A1.ii and A2.i): curve enters red zone, clips at axis top, upward arrow, "∞ from step 50" text annotation.

### Section 2 — Master Ablation Table

Single table, grouped by axis. Columns: Design axis | Variant | FID↓ | UNI-cos↑ | CellViT r↑ | Type KL↓ | Nuc KS↓ | Params

- **A1 group** (3 rows): Concat, Per-channel ⚠, Production ★. Params column populated (variants differ structurally).
- **A2 group** (3 rows): Bypass ⚠, Off-the-shelf, Production ★. Params column "shared".
- Production ★ repeated once per axis group (same numbers; makes within-group comparison self-contained).
- ⚠ rows: amber/red background. Metrics from best finite-loss checkpoint. Footnote: "⚠ ∞ grad norm from step 50 across all seeds; metrics from last finite checkpoint."
- Best row per group: bold.

### Section 3 — Qualitative Tile Grid

4 fixed test tile IDs (same across all ablation SI figures).

Row order:
1. GT H&E (reference)
2. A1.i Concat
3. A1.ii Per-channel ⚠ — diagonal hatch, red border, "∞" badge
4. ★ Production (shared)
5. *(dashed separator — "A2 variants")*
6. A2.i Bypass probe ⚠ — same hatch + badge treatment as A1.ii
7. A2.ii Off-the-shelf

Columns: 4 tiles.

---

## 5. Styling

Matches `src/paper_figures/style.py` (Nature Communications):
- Production: `#4caf50` green, solid, bold border in tile grid
- A1 Concat: `#2196f3` blue, dashed
- A1/A2 instability (⚠): `#f44336` red, solid; diagonal hatch `////` in tiles; amber row in table
- A2 Bypass: `#ff9800` orange, dashed (curves); red treatment for ⚠ instability
- A2 Off-the-shelf: `#9c27b0` purple, dotted
- Instability red zone in grad norm plots: `#f44336` at 10% opacity, top of y-axis

---

## 6. Implementation Files

| File | Role |
|------|------|
| `tools/ablation_a1_a2/build_cache.py` | One-time: extract curves, run inference, compute metrics, write `cache.json` |
| `tools/ablation_a1_a2/__init__.py` | Empty |
| `src/paper_figures/fig_si_a1_a2_unified.py` | Figure builder — reads `cache.json` + tile PNGs only; no weights |
| `src/paper_figures/style.py` | Reused unchanged |
| `tools/stage3/tile_pipeline.py` | Reused for inference in `build_cache.py` |
| `tools/stage3/run_evaluation.py` | Reused for metrics in `build_cache.py` |

`src/paper_figures/fig_si_a2_bypass.py` is superseded — archive after the unified figure is validated.

---

## 7. Metrics

FID, UNI-cos, CellViT cell-count r, CellViT cell-type composition KL, CellViT nuclear-morphology KS.
Computed on paired ORION-CRC test split only.
⚠ variants: metrics from last finite-loss checkpoint (loss still converging despite ∞ grad norm).

---

## 8. Caption Requirements

- Seeds, proxy step count, headline step count per variant.
- A1: param count and per-step wall-clock per variant.
- A2: one sentence per variant describing conditioning path.
- ⚠ footnote: "∞ grad norm observed from step 50 across all seeds for A1.ii and A2.i; this is itself a result — ungrouped conditioning and open bypass paths are training-unstable. Metrics reported from last finite checkpoint."
- Production: same checkpoint for both axes.

---

## 9. Open Items (as of 2026-04-27)

- A2 bypass inference weights present but a2_off_shelf uses pretrained only — confirm exact checkpoint/config.
- Stray dir `checkpoints/checkpoint/step_0002600/` — confirm identity before using or delete.
- Verify 4 fixed tile IDs match those in A2/A3 specs.
- Production `train_log.log` is a plain text log (not JSONL) — parser in `build_cache.py` must handle both formats.
