# Multi-Encoder Spatial Decodability Probe — Spec

## Background

`figures/pngs_updated/07_inverse_decoding.png` panel A ranks T1 targets by
**tile-mean R²** using a scalar probe per encoder (UNI-2h, Virchow2,
CTransPath, REMEDIS optional, ResNet-50). Tile-mean R² inflates targets
that are nearly constant within a tile (oxygen σ ≈ 0.05) — the probe just
predicts the slide-wide mean and scores well.

The new `src/a1_probe_mlp_spatial` already runs UNI-2h on a 16×16 per-patch
grid and writes `r2_global`, `r2_within`, `pearson_r`. Result (800-tile pilot,
50K row cap): proliferation, density, nonprolif show real within-tile signal;
oxygen / glucose collapse (R² < 0). Story is robust enough to surface as an
SI figure.

To make the per-patch ranking **comparable across encoders**, each encoder
should be probed at its own native spatial grid (no resampling that
artificially smooths or upsamples features). Each encoder's bar then answers
"what fraction of within-tile variance does this encoder's natural feature
grid recover?"

## Goal

Produce `figures/pngs_updated/07d_t1_spatial_multi_encoder.png`: per-encoder
spatial decodability bars for T1 targets, ranked by within-tile R², with
each encoder probed at its own native grid.

## Non-goals

- No fine-tuning, no new trainable modules.
- No replacement of fig 07A. Spatial figure is supplementary.
- No T2 marker rerun (that's separately tracked).

## Encoder coverage and native grids

| Encoder | Native grid (224 input) | Feature dim | Probe?
|---------|-------------------------|-------------|-------|
| UNI-2h ViT (patch_14) | 16×16 = 256 | 1536 | already done |
| Virchow2 ViT (patch_14) | 16×16 = 256 | 1280 | new |
| CTransPath Swin (window7) | 7×7 = 49 (stage-4 output) | 768 | new |
| ResNet-50 | 7×7 = 49 (`layer4` output) | 2048 | new |
| REMEDIS ResNet | 7×7 (if weights available) | 2048 | optional |

For each non-UNI encoder the patch features come from a forward hook on the
last spatial stage. No model edits.

## Targets

Use the existing T1 channel set (`cell_density`, 3 cell-type fracs, 3
cell-state fracs, `vasculature_frac`, `oxygen_mean`, `glucose_mean`). For each
encoder grid `(H, W)` block-mean-pool the 256×256 channel maps to `(H, W)`
(use the existing `block_mean_pool` if `256 % H == 0`, else interpolate via
average pooling kernel). For UNI/Virchow2: 16×16. For CTransPath/ResNet-50:
7×7. The new `T1_spatial` builder takes `--grid` as a CLI flag; one builder
output per grid resolution.

## Probe

`src/a1_probe_mlp_spatial/main.py` already supports arbitrary grid sizes via
the `(N_tiles, n_patches, n_targets)` target tensor and arbitrary `(N, P, D)`
feature tensors. No probe code changes needed beyond a new feature loader
hook for non-UNI encoders.

## Per-encoder peak RAM budget

| Encoder | rows per fit (800 tiles, 16×16 → 50K cap) | working set fp32 |
|---------|-----|-----|
| UNI-2h | 50K × 1536 = 77 M floats | ~310 MB |
| Virchow2 | 50K × 1280 = 64 M | ~256 MB |
| CTransPath | 800 × 49 = 39 K rows (no cap needed) × 768 | ~120 MB |
| ResNet-50 | 39 K × 2048 | ~320 MB |

All well within 24 GB AS cap. Workers can stay at n_jobs=2.

## Figure layout

`07d_t1_spatial_multi_encoder.png`. Single figure, two panels:

- **Panel A**: per-target within-tile R². X-axis: 10 T1 targets, sorted by
  UNI-2h `r2_within`. Y-axis: R². Bars grouped per encoder (4–5 bars per
  target), one color per encoder. Clip floor at -2.0 with annotation. Error
  bars: per-fold SD.
- **Panel B**: per-target Pearson r. Same x-axis order. Cleaner ranking;
  invariant to scale.

Color palette: reuse `_ENCODER_COLORS` from `fig_inverse_decoding.py`. Use
the same `apply_style()` (Nimbus Sans, 4 black spines, legend below).

## Files

| Status | Path | Purpose |
|--------|------|---------|
| modify | `src/a1_mask_targets_spatial/main.py` | accept `--grid` already does; verify non-16 grid block_mean_pool path |
| modify | `src/a1_probe_encoders/main.py` | add `extract_*_patch_features` for each encoder |
| create | `pipeline/patch_extractors.py` *(new module)* | per-encoder patch extractors via hooks; pure functions, reusable |
| create | `tests/test_patch_extractors.py` | shape/grid asserts for each extractor on small synthetic input |
| create | `src/paper_figures/fig_t1_spatial_multi_encoder.py` | new figure 07d |
| create | `tests/test_fig_t1_spatial_multi_encoder.py` | smoke test for figure builder |
| run | `src/a1_mask_targets_spatial/out/grid_07/` | T1 targets at 7×7 grid |
| run | `data/orion-crc33/features_patches/<encoder>/<tile>_patches.npy` | patch features per encoder |
| run | `src/a1_probe_mlp_spatial/out/<encoder>_<grid>/` | probe results per encoder |

## Out of scope

- REMEDIS: if local weights present, include; else mark TODO and let figure
  render with whatever encoders are available.
- 10 379-tile rerun: 800-tile pilot is enough for SI figure. Note in caption.

## Success criteria

1. `07d_t1_spatial_multi_encoder.png` renders with at least 3 encoders
   (UNI-2h, Virchow2, one of CTransPath/ResNet-50). REMEDIS optional.
2. Each encoder's bar uses its native grid (no interpolation).
3. UNI-2h numbers match the existing 07c (sanity check that the multi-encoder
   pipeline didn't change the UNI ranking).
4. All probe outputs include `r2_within_mean` and `pearson_r_mean` columns.
5. Caveat in figure caption: bars not directly comparable across grids; they
   each report decodability at the encoder's native granularity.
