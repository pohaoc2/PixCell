# Ablation CLI Reference

This file summarizes the paired and unpaired ablation workflows.

## Conventions

- `paired_ablation/` uses the original paired ORION-style root: `data/orion-crc33`
- `unpaired_ablation/` uses a remapped ORION-style root where:
  - `exp_channels/` stay attached to the layout tile
  - `he/` and `features/` are remapped to a different style tile
- Latest checkpoint directory used here:
  - `checkpoints/pixcell_controlnet_exp/npy_inputs`

## Directory layout

### Paired

- cache / per-tile ablations:
  - `inference_output/paired_ablation/ablation_results`
- CellViT export/import:
  - `inference_output/paired_ablation/cellvit_batch`
  - `inference_output/paired_ablation/cellvit`
- dataset figures / analyses:
  - `inference_output/paired_ablation/leave_one_out`
  - `inference_output/paired_ablation/channel_sweep`

### Unpaired

- remapped dataset root:
  - `inference_output/unpaired_ablation/data/orion-crc33-unpaired`
- style mapping JSON:
  - `inference_output/unpaired_ablation/data/orion-crc33-unpaired/metadata/unpaired_mapping.json`
- cache / per-tile ablations:
  - `inference_output/unpaired_ablation/ablation_results`
- CellViT export/import:
  - `inference_output/unpaired_ablation/cellvit_batch`
  - `inference_output/unpaired_ablation/cellvit`
- dataset figures / analyses:
  - `inference_output/unpaired_ablation/leave_one_out`
  - `inference_output/unpaired_ablation/channel_sweep`

## Step 1-2. Build the unpaired dataset root

Comment:
- Use this only for the unpaired workflow.
- This reuses the same 1000 tile IDs already present in paired ablation results.
- It writes a deranged style mapping JSON and creates a remapped ORION-style root.

```bash
python tools/stage3/prepare_unpaired_ablation_dataset.py \
  --paired-cache-root inference_output/paired_ablation/ablation_results \
  --data-root data/orion-crc33 \
  --output-root inference_output/unpaired_ablation/data/orion-crc33-unpaired \
  --seed 42
```

Inspect the saved mapping:

```bash
cat inference_output/unpaired_ablation/data/orion-crc33-unpaired/metadata/unpaired_mapping.json
```

## Step 3. Generate Stage 3 ablation caches

Comment:
- Paired uses the original ORION root.
- Unpaired uses the remapped ORION root.
- For unpaired generation, the remapped `features/<tile>_uni.npy` files already exist, so `--cache-uni-features` is not needed.
- `--jobs 2` is reasonable on a single T4 after the worker reuse fix; start lower if VRAM is tight.

### Paired

```bash
python tools/stage3/generate_ablation_subset_cache.py \
  --data-root data/orion-crc33 \
  --checkpoint-dir checkpoints/pixcell_controlnet_exp/npy_inputs \
  --n-tiles 1000 \
  --output-dir inference_output/paired_ablation/ablation_results \
  --seed 42 \
  --tile-sample-seed 42 \
  --device cuda \
  --guidance-scale 2.5 \
  --num-steps 20 \
  --jobs 2
```

### Unpaired

```bash
python tools/stage3/generate_ablation_subset_cache.py \
  --data-root inference_output/unpaired_ablation/data/orion-crc33-unpaired \
  --checkpoint-dir checkpoints/pixcell_controlnet_exp/npy_inputs \
  --n-tiles 1000 \
  --output-dir inference_output/unpaired_ablation/ablation_results \
  --seed 42 \
  --tile-sample-seed 42 \
  --device cuda \
  --guidance-scale 2.5 \
  --num-steps 20 \
  --jobs 2
```

## Step 4. Export CellViT batch

Comment:
- Same CLI shape for paired and unpaired; only the cache root and output dir change.

### Paired

```bash
python tools/cellvit/export_batch.py \
  --cache-root inference_output/paired_ablation/ablation_results \
  --output-dir inference_output/paired_ablation/cellvit_batch \
  --mode copy \
  --overwrite
```

### Unpaired

```bash
python tools/cellvit/export_batch.py \
  --cache-root inference_output/unpaired_ablation/ablation_results \
  --output-dir inference_output/unpaired_ablation/cellvit_batch \
  --mode copy \
  --overwrite
```

## Step 5. Import external CellViT results

Comment:
- Replace `/path/to/cellvit/results` with your actual external CellViT output folder.

### Paired

```bash
python tools/cellvit/import_results.py \
  --manifest inference_output/paired_ablation/cellvit_batch/manifest.json \
  --results-dir /path/to/cellvit/results
```

### Unpaired

```bash
python tools/cellvit/import_results.py \
  --manifest inference_output/unpaired_ablation/cellvit_batch/manifest.json \
  --results-dir /path/to/cellvit/results
```

## Step 6. Compute ablation metrics

Comment:
- Paired main metrics usually keep: `all`
- Unpaired main metrics usually keep: `aji pq style_hed`
- `all` still means `cosine lpips aji pq`
- `style_hed` is optional and must be requested explicitly

### Paired

```bash
python tools/compute_ablation_metrics.py \
  --cache-dir inference_output/paired_ablation/ablation_results \
  --orion-root data/orion-crc33 \
  --metrics all \
  --device cuda \
  --jobs 2
```

### Unpaired

```bash
python tools/compute_ablation_metrics.py \
  --cache-dir inference_output/unpaired_ablation/ablation_results \
  --orion-root inference_output/unpaired_ablation/data/orion-crc33-unpaired \
  --metrics aji pq style_hed \
  --device cuda \
  --jobs 2
```

If you want the legacy metrics plus `style_hed` for unpaired:

```bash
python tools/compute_ablation_metrics.py \
  --cache-dir inference_output/unpaired_ablation/ablation_results \
  --orion-root inference_output/unpaired_ablation/data/orion-crc33-unpaired \
  --metrics all style_hed \
  --device cuda \
  --jobs 2
```

## Step 7. Compute FID

Comment:
- FID is dataset-level and uses all tiles in the cache by default.
- It is written once per condition and then backfilled into every tile's `metrics.json`.

### Paired

```bash
python tools/compute_fid.py \
  --cache-dir inference_output/paired_ablation/ablation_results \
  --orion-root data/orion-crc33 \
  --device cuda \
  --batch-size 64 \
  --output inference_output/paired_ablation/ablation_results/fid_scores.json
```

### Unpaired

```bash
python tools/compute_fid.py \
  --cache-dir inference_output/unpaired_ablation/ablation_results \
  --orion-root inference_output/unpaired_ablation/data/orion-crc33-unpaired \
  --device cuda \
  --batch-size 64 \
  --output inference_output/unpaired_ablation/ablation_results/fid_scores.json
```

## Step 8. Render per-tile ablation grid figures

Comment:
- Renders `ablation_grid.png` inside each per-tile cache directory.

### Paired

```bash
python tools/vis/stage3_ablation_grid_figure.py \
  --cache-dir inference_output/paired_ablation/ablation_results \
  --orion-root data/orion-crc33 \
  --output-name ablation_grid \
  --device cuda \
  --jobs 2
```

### Unpaired

```bash
python tools/vis/stage3_ablation_grid_figure.py \
  --cache-dir inference_output/unpaired_ablation/ablation_results \
  --orion-root inference_output/unpaired_ablation/data/orion-crc33-unpaired \
  --output-name ablation_grid \
  --metric-set unpaired \
  --sort-by style_hed \
  --no-auto-cosine \
  --device cpu \
  --jobs 8
```

## Step 9. Render dataset-level metrics figure

Comment:
- `--min-gt-cells` filters tiles when aggregating the plotted metrics.
- This is especially helpful for `PQ/AJI`.
- It does not truly recompute FID on the filtered subset; it only filters the plotting inputs.

### Paired

```bash
python tools/render_dataset_metrics.py \
  --metric-dir inference_output/paired_ablation/ablation_results \
  --output inference_output/paired_ablation/dataset_metrics_filtered.png \
  --orion-root data/orion-crc33 \
  --min-gt-cells 20
```

### Unpaired

```bash
python tools/render_dataset_metrics.py \
  --metric-dir inference_output/unpaired_ablation/ablation_results \
  --output inference_output/unpaired_ablation/dataset_metrics_filtered.png \
  --orion-root inference_output/unpaired_ablation/data/orion-crc33-unpaired \
  --min-gt-cells 20
```

## Step 10. Run leave-one-out analysis

Comment:
- Writes one `leave_one_out_diff.png` and one `leave_one_out_diff_stats.json` per tile.

### Paired

```bash
python tools/vis/leave_one_out_diff.py \
  --cache-root inference_output/paired_ablation/ablation_results \
  --orion-root data/orion-crc33 \
  --out-root inference_output/paired_ablation/leave_one_out
```

### Unpaired

```bash
python tools/vis/leave_one_out_diff.py \
  --cache-root inference_output/unpaired_ablation/ablation_results \
  --orion-root inference_output/unpaired_ablation/data/orion-crc33-unpaired \
  --out-root inference_output/unpaired_ablation/leave_one_out
```

## Step 11. Run experiments 1/2/3

Comment:
- `classify_tiles.py` chooses representative tiles from the dataset root.
- `channel_sweep.py` then runs Exp 1, Exp 2, and Exp 3 and writes both caches and figures.

### Paired: classify representative tiles

```bash
python tools/stage3/classify_tiles.py \
  --exp-root data/orion-crc33 \
  --out inference_output/paired_ablation/channel_sweep/tile_classes.json
```

### Paired: run channel sweep

```bash
python tools/stage3/channel_sweep.py \
  --class-json inference_output/paired_ablation/channel_sweep/tile_classes.json \
  --data-root data/orion-crc33 \
  --checkpoint-dir checkpoints/pixcell_controlnet_exp/npy_inputs \
  --out inference_output/paired_ablation/channel_sweep \
  --cache-dir inference_output/paired_ablation/channel_sweep/cache \
  --experiments 1 2 3 \
  --seed 42
```

### Unpaired: classify representative tiles

```bash
python tools/stage3/classify_tiles.py \
  --exp-root inference_output/unpaired_ablation/data/orion-crc33-unpaired \
  --out inference_output/unpaired_ablation/channel_sweep/tile_classes.json
```

### Unpaired: run channel sweep

```bash
python tools/stage3/channel_sweep.py \
  --class-json inference_output/unpaired_ablation/channel_sweep/tile_classes.json \
  --data-root inference_output/unpaired_ablation/data/orion-crc33-unpaired \
  --checkpoint-dir checkpoints/pixcell_controlnet_exp/npy_inputs \
  --out inference_output/unpaired_ablation/channel_sweep \
  --cache-dir inference_output/unpaired_ablation/channel_sweep/cache \
  --experiments 1 2 3 \
  --seed 42
```

## Practical notes

- If you want a fresh rerun, remove or rename the previous output folder first.
- For unpaired evaluation:
  - keep `PQ`, `AJI`, `FID`
  - prefer `style_hed` as the style metric
  - avoid using `LPIPS` and `cosine` as the main conclusions unless you explicitly want paired-reference similarity
- For paired evaluation:
  - `all` is still the default metric bundle
  - `style_hed` is available but usually unnecessary
