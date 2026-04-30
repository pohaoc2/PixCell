# Handover — Debug Compare 500 / CellViT Refresh

## TL;DR

The compare_500 debug pipeline is now back in a consistent post-CellViT state.

- All 5 compare_500 variants were rendered for the same 500 metric tiles.
- A flat CellViT export was created and zipped.
- External CellViT results were imported back into the compare_500 tile cache.
- Summary metrics were recomputed into `inference_output/debug_compare_500/cache.json`.
- Unified and split figure outputs were regenerated under `figures/pngs_updated/`.

This handoff is about the debug comparison workflow, not the earlier TME gradient-explosion investigation.

## What this compare_500 batch contains

The standard debug comparison now covers these 5 variants over 500 tiles:

- `production`
- `a1_concat`
- `a1_per_channel`
- `a2_bypass_full_tme`
- `a2_off_shelf`

Ground-truth H&E tiles are also present under `inference_output/debug_compare_500/tiles/gt`, but they are not part of the CellViT export batch.

## Important distinction: compare_500 vs full15 concat

Two different downstream pipelines exist under `inference_output/debug_compare_500/`:

1. Standard compare_500 debug batch
   - Location: `inference_output/debug_compare_500/tiles/`
   - Variants: the 5 standard debug variants above
   - Import/export helpers: `tools/ablation_a1_a2/metrics_io.py`
   - Figure builder: `src/paper_figures/fig_si_a1_a2_unified.py`

2. Full15 concat ablation batch
   - Location: `inference_output/debug_compare_500/a1_concat_full_ablation/`
   - Scope: 15 condition combinations for concat only
   - Import/export helpers: `tools/cellvit/import_results.py`, `tools/cellvit/export_batch.py`
   - Metrics helper: `tools/compute_ablation_metrics.py`

The user-provided `all_variants` CellViT run belongs to the standard compare_500 debug batch, not the full15 concat ablation batch.

## Completed work

### 1. Generated missing compare_500 variants

Previously only `production` and `a1_concat` were present. The following were added to complete the 5-way compare set:

- `a1_per_channel`
- `a2_bypass_full_tme`
- `a2_off_shelf`

For `a2_off_shelf`, prior generated H&E was reused where tile IDs overlapped because the off-shelf model weights were unchanged:

- Reused from earlier debug outputs: 300 / 500 tiles
- Newly generated remainder: 200 / 500 tiles

Final tile coverage is complete for all variants:

- `production`: 500 PNGs
- `a1_concat`: 500 PNGs
- `a1_per_channel`: 500 PNGs
- `a2_bypass_full_tme`: 500 PNGs
- `a2_off_shelf`: 500 PNGs

### 2. Exported the all-variant CellViT batch

Export location:

- Folder: `inference_output/debug_compare_500/cellvit_all_variants/`
- Images: `inference_output/debug_compare_500/cellvit_all_variants/images/`
- Manifest: `inference_output/debug_compare_500/cellvit_all_variants/manifest.json`
- Zip: `inference_output/debug_compare_500/cellvit_all_variants.zip`

Verified export counts:

- 2500 PNGs total
- Expected shape: `5 variants x 500 tiles = 2500`

### 3. Consumed external CellViT results

User-provided results were found under:

- `inference_output/debug_compare_500/all_variants/cellvit/`

Matching manifest:

- `inference_output/debug_compare_500/all_variants/cellvit_all_variants/manifest.json`

Validation before import:

- `2500` JSON files found
- `2500` manifest entries found
- `0` missing filenames after manifest/result-name comparison

Import step used the standard compare_500 importer in `tools/ablation_a1_a2/metrics_io.py`.

Import report written to:

- `inference_output/debug_compare_500/all_variants/cellvit_all_variants/import_report.json`

### 4. Verified imported CellViT sidecars

After import, every compare_500 variant has full sidecar coverage:

- `production`: 500 sidecars
- `a1_concat`: 500 sidecars
- `a1_per_channel`: 500 sidecars
- `a2_bypass_full_tme`: 500 sidecars
- `a2_off_shelf`: 500 sidecars

Sidecar naming is the standard:

- `<tile_id>_cellvit_instances.json`

These live beside the generated PNGs in each variant directory under:

- `inference_output/debug_compare_500/tiles/<variant>/`

### 5. Recomputed compare_500 summary metrics

Summary metrics were recomputed with:

```bash
python -m tools.ablation_a1_a2.metrics_io compute \
       --cache-dir inference_output/debug_compare_500 \
       --orion-root data/orion-crc33 \
       --device cuda \
       --variants production a1_concat a1_per_channel a2_bypass_full_tme a2_off_shelf
```

Important note:

- This path uses imported `*_cellvit_instances.json` sidecars.
- It does not rerun CellViT.

Updated summary now lives in:

- `inference_output/debug_compare_500/cache.json`

## Final metric summary

### production

- `n_tiles`: 500
- `fud`: 171.87160145288192
- `dice`: 0.8975547679383322
- `pq`: 0.7906668635006808
- `lpips`: 0.34848187780380246
- `style_hed`: 0.03926316789241031

### a1_concat

- `n_tiles`: 500
- `fud`: 161.64002588165343
- `dice`: 0.898415829513277
- `pq`: 0.7934131734309054
- `lpips`: 0.3166726744472981
- `style_hed`: 0.034889157695458375

### a1_per_channel

- `n_tiles`: 500
- `fud`: 195.83224800839898
- `dice`: 0.6156159898251067
- `pq`: 0.4260592438379538
- `lpips`: 0.40702725905179976
- `style_hed`: 0.09882909470437808

### a2_bypass_full_tme

- `n_tiles`: 500
- `fud`: 171.3847321450176
- `dice`: 0.8987132091882496
- `pq`: 0.7947529087128214
- `lpips`: 0.34821229112148283
- `style_hed`: 0.03911856083868174

### a2_off_shelf

- `n_tiles`: 500
- `fud`: 214.7971599479886
- `dice`: 0.8625126287956607
- `pq`: 0.7007343693446132
- `lpips`: 0.35224138030409813
- `style_hed`: 0.34373103087246376

## Figure outputs regenerated

The compare_500 cache was rendered with:

```bash
python -m src.paper_figures.fig_si_a1_a2_unified \
       --cache-dir inference_output/debug_compare_500 \
       --out figures/pngs_updated/SI_A1_A2_unified.png
```

Outputs created:

- `figures/pngs_updated/SI_A1_A2_unified.png`
- `figures/pngs_updated/SI_A1_A2_section1_curves.png`
- `figures/pngs_updated/SI_A1_A2_section2_metrics.png`
- `figures/pngs_updated/SI_A1_A2_section3_tiles.png`
- `figures/pngs_updated/SI_A1_A2_section4_sensitivity.png`

Verified nonzero file sizes:

- Unified: `13133248`
- Section 1: `388980`
- Section 2: `197993`
- Section 3: `11262282`
- Section 4: `43752`

## Current known-good artifact paths

### Compare_500 cache and tiles

- `inference_output/debug_compare_500/cache.json`
- `inference_output/debug_compare_500/metric_tile_ids.txt`
- `inference_output/debug_compare_500/tiles/production/`
- `inference_output/debug_compare_500/tiles/a1_concat/`
- `inference_output/debug_compare_500/tiles/a1_per_channel/`
- `inference_output/debug_compare_500/tiles/a2_bypass_full_tme/`
- `inference_output/debug_compare_500/tiles/a2_off_shelf/`
- `inference_output/debug_compare_500/tiles/gt/`

### Standard compare_500 CellViT export/import artifacts

- Export folder: `inference_output/debug_compare_500/cellvit_all_variants/`
- Export zip: `inference_output/debug_compare_500/cellvit_all_variants.zip`
- User-provided result folder: `inference_output/debug_compare_500/all_variants/cellvit/`
- Matching manifest: `inference_output/debug_compare_500/all_variants/cellvit_all_variants/manifest.json`
- Import report: `inference_output/debug_compare_500/all_variants/cellvit_all_variants/import_report.json`

### Full15 concat ablation artifacts

These are separate and remain valid:

- `inference_output/debug_compare_500/a1_concat_full_ablation/`
- `inference_output/debug_compare_500/cellvit_full15_concat/manifest.json`
- `inference_output/debug_compare_500/cellvit_full15_concat.zip`

## Repro / maintenance commands

### Re-import all-variant CellViT results

```bash
python -m tools.ablation_a1_a2.metrics_io import-cellvit \
       --manifest inference_output/debug_compare_500/all_variants/cellvit_all_variants/manifest.json \
       --results-dir inference_output/debug_compare_500/all_variants/cellvit
```

### Recompute compare_500 metrics

```bash
python -m tools.ablation_a1_a2.metrics_io compute \
       --cache-dir inference_output/debug_compare_500 \
       --orion-root data/orion-crc33 \
       --device cuda \
       --variants production a1_concat a1_per_channel a2_bypass_full_tme a2_off_shelf
```

### Rebuild the unified and split figures

```bash
python -m src.paper_figures.fig_si_a1_a2_unified \
       --cache-dir inference_output/debug_compare_500 \
       --out figures/pngs_updated/SI_A1_A2_unified.png
```

## Practical notes for the next person

- The user-reported `all_variants` path was one level above the actual raw CellViT JSON directory. The JSONs were under `all_variants/cellvit/`, not directly under `all_variants/`.
- The all-variant manifest used for import was nested under `all_variants/cellvit_all_variants/manifest.json`.
- `metrics_io.compute` is slow and mostly silent after model setup. It can sit for several minutes with no new terminal text while still using the GPU actively.
- For `a2_off_shelf`, reuse of prior generated H&E is valid when tile IDs overlap, because the off-shelf model weights were unchanged.
- The compare_500 standard batch and the concat full15 ablation batch are separate. Do not import one batch into the other cache tree.

## If new CellViT results arrive later

1. Identify which pipeline they belong to:
   - standard compare_500 debug batch, or
   - full15 concat ablation batch
2. Match them against the correct manifest before importing.
3. For standard compare_500, use `tools.ablation_a1_a2.metrics_io.py`.
4. For full15 concat ablation, use `tools/cellvit/import_results.py` and `tools/compute_ablation_metrics.py`.

## Files / configs of interest

- Smoke configs (untracked): `configs/config_controlnet_exp_smoke_depth18_bs2_*_fixed.py`
- Probe entry point: `MultiGroupTMEModule.enable_debug_tme_probe()` (called once in train loop on first sync step when `debug_tme_probe=True`)
- Offender attribution: `_grad_health(..., top_k=8)` returns top-8 tensors by `max_abs` — useful when investigating future blow-ups.

## Why "safe clip" alone wasn't enough — for future readers

`_safe_clip_grad_norm_` returns `total_norm` **before** scaling. A reported norm of 1e31 means the optimizer sees `clip_coef = max_norm / 1e31 ≈ 1e-31`, effectively zeroing the TME update. Training looks stable (no NaNs, controlnet trains fine) while the TME branch silently freezes. Always check pre-clip `max_abs` per-tensor, not just the post-clip norm.
