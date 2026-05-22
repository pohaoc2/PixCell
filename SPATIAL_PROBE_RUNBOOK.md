# Spatial Probe Runbook

Per-patch MLP probe on UNI patch tokens. Replaces the scalar tile-mean probe
(`a1_probe_mlp`) with a per-patch evaluation that asks: can a small MLP head,
applied independently to each UNI patch token, recover spatial structure in the
target channels? Adds harder baselines (per-tile within-R², Pearson r,
within-tile shuffle) so trivially-flat targets like oxygen don't get
artificially high R².

## Inputs

* H&E tiles already cached in `data/orion-crc33/he/` (or wherever stage1 ran).
* Experimental channel maps in `data/orion-crc33/exp_channels/{cell_masks,...}/`.
* Per-cell CODEX feature CSV (the same one consumed by `a1_codex_targets`).

## Pipeline

### 1. Re-extract UNI features with patch tokens

Adds `<tile_id>_uni_tokens.npy` of shape `(256, 1536)` float16 next to the
existing `<tile_id>_uni.npy` CLS embedding (≈8 GB total for 10 379 tiles).

```bash
python stage1_extract_features.py \
    --image-dir   data/orion-crc33/he \
    --output-dir  data/orion-crc33/features \
    --skip-vae \
    --save-uni-tokens
```

The default `--uni-tokens-prefix uni_tokens` produces `<tile_id>_uni_tokens.npy`.
Pass `--skip-uni` to avoid rewriting the CLS embedding files. To run this on a
fresh tile set, drop both skip flags.

### 2. Build per-patch T1 targets

Block-pools each 256×256 channel map (`cell_masks`, `cell_type_*`,
`cell_state_*`, `vasculature`, `oxygen`, `glucose`) down to a 16×16 grid;
cell-type and cell-state outputs are per-patch fractions of local density.

```bash
python -m src.a1_mask_targets_spatial.main \
    --features-dir       data/orion-crc33/features \
    --exp-channels-dir   data/orion-crc33/exp_channels \
    --out-dir            src/a1_mask_targets_spatial/out \
    --grid               16
```

Writes `mask_targets_T1_spatial.npy` of shape `(N_tiles, 256, 10)` plus
`target_names_T1_spatial.json` and `tile_ids.txt`.

### 3. Build per-patch T2 marker targets

Bins per-cell CODEX intensities into the same 16×16 patch grid using each
cell's centroid. Empty patches (no cells) are NaN; the probe masks them out.

```bash
python -m src.a1_codex_targets_spatial.build \
    --features-csv   path/to/codex_per_cell_features.csv \
    --markers-csv    path/to/markers.csv \
    --tile-ids-path  src/a1_mask_targets_spatial/out/tile_ids.txt \
    --out-dir        src/a1_codex_targets_spatial/out \
    --grid           16
```

Writes `codex_T2_spatial_mean.npy` of shape `(N_tiles, 256, n_markers)` plus
`codex_marker_names.json` and `codex_cell_counts_per_patch.npy`.

### 4. Run the spatial probe

One MLP per target column. Shared across patch positions (every (tile, patch)
pair is one row of `(1536,) -> scalar`). Spatial GroupKFold matches the
existing scalar-probe scheme (2048 px blocks).

```bash
# T1 spatial probe
python -m src.a1_probe_mlp_spatial.main \
    --features-dir         data/orion-crc33/features \
    --targets-path         src/a1_mask_targets_spatial/out/mask_targets_T1_spatial.npy \
    --tile-ids-path        src/a1_mask_targets_spatial/out/tile_ids.txt \
    --target-names-path    src/a1_mask_targets_spatial/out/target_names_T1_spatial.json \
    --out-dir              src/a1_probe_mlp_spatial/out/t1_spatial \
    --compute-shuffle-baseline

# T2 spatial probe
python -m src.a1_probe_mlp_spatial.main \
    --features-dir         data/orion-crc33/features \
    --targets-path         src/a1_codex_targets_spatial/out/codex_T2_spatial_mean.npy \
    --tile-ids-path        src/a1_codex_targets_spatial/out/tile_ids.txt \
    --target-names-path    src/a1_codex_targets_spatial/out/codex_marker_names.json \
    --out-dir              src/a1_probe_mlp_spatial/out/t2_spatial \
    --compute-shuffle-baseline
```

Each output directory contains:

* `mlp_spatial_probe_results.csv` — per-target summary with columns
  `r2_mean, r2_sd, r2_within_mean, r2_within_sd, pearson_r_mean, pearson_r_sd,
  delta_shuffle, n_valid_folds`.
* `mlp_spatial_probe_results.json` — full per-fold breakdown.
* `cv_splits.json` — saved fold assignments (reuse via `--cv-splits-path`).
* `manifest.json` — provenance + `tile_ids_sha1`.

`--compute-shuffle-baseline` doubles training time (one extra fit per fold per
target with within-tile-shuffled tokens) but is required to compute
`delta_shuffle`. Drop the flag for a fast first pass.

## Interpreting the new metrics

| Metric | What it asks |
|--------|--------------|
| `r2_mean`        | Global per-patch R² (same convention as scalar probes). |
| `r2_within_mean` | Variance explained **inside each tile**. Trivially-flat targets (oxygen) score near 0 here even if `r2_mean` is high. |
| `pearson_r_mean` | Correlation between predicted and actual per-patch values; scale/offset invariant. |
| `delta_shuffle`  | `r2_mean` minus shuffled-feature baseline. Real spatial signal ⇒ Δ ≫ 0; trivial mean-prediction ⇒ Δ ≈ 0. |

## Switching the figures over

Once `mlp_spatial_probe_results.csv` exists, point the figure builders at the
spatial CSV instead of the scalar one:

* `src/paper_figures/fig_marker_utility.py:22` — `PROBE_CSV` constant. Use
  `r2_within_mean` for the x-axis if you want to test "within-tile spatial
  decodability"; keep `r2_mean` for backward compatibility.
* `src/paper_figures/main.py` — set `T2_MLP_CSV` to the new spatial CSV (and
  add a `T1_SPATIAL_CSV` for fig 07A if rewriting that panel too).

These switches are intentionally not automated — pick which R² flavor the
final paper figure should show before re-running `main.py`.
