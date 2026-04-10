# Ablation CLI Reference

This file summarizes the paired and unpaired ablation workflows.

## Conventions

- `paired_ablation/` uses the original paired ORION-style root: `data/orion-crc33`
- `unpaired_ablation/` can use either:
  - a remapped ORION-style root, or
  - the original `data/orion-crc33` plus a small `unpaired_mapping.json`
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

- style mapping JSON:
  - `inference_output/unpaired_ablation/metadata/unpaired_mapping.json`
- optional remapped dataset root:
  - `inference_output/unpaired_ablation/data/orion-crc33-unpaired`
- cache / per-tile ablations:
  - `inference_output/unpaired_ablation/ablation_results`
- CellViT export/import:
  - `inference_output/unpaired_ablation/cellvit_batch`
  - `inference_output/unpaired_ablation/cellvit`
- dataset figures / analyses:
  - `inference_output/unpaired_ablation/leave_one_out`
  - `inference_output/unpaired_ablation/channel_sweep`

## Rerun from scratch

Comment:
- Use this when you want a clean paired or unpaired regeneration instead of extending an existing cache.
- If you keep old outputs around, rename them first so they do not mix with the new mapping or cache contents.
- The Stage 3 cache generator now batches ablation conditions internally, so you do not need an extra CLI flag to enable batching.
- On a single A100, start with `--jobs 1`. Multiple jobs on one GPU can still compete for VRAM.

### Paired: clean rerun to 5000 tiles

```bash
mv inference_output/paired_ablation inference_output/paired_ablation_old

python tools/stage3/generate_ablation_subset_cache.py \
  --data-root data/orion-crc33 \
  --checkpoint-dir checkpoints/pixcell_controlnet_exp/npy_inputs \
  --output-dir inference_output/paired_ablation/ablation_results \
  --target-total-tiles 5000 \
  --seed 42 \
  --tile-sample-seed 42 \
  --device cuda \
  --guidance-scale 2.5 \
  --num-steps 20 \
  --jobs 1
```

### Unpaired: clean rerun to 5000 tiles

```bash
mv inference_output/unpaired_ablation inference_output/unpaired_ablation_old

python tools/stage3/prepare_unpaired_ablation_dataset.py \
  --paired-cache-root inference_output/paired_ablation/ablation_results \
  --data-root data/orion-crc33 \
  --metadata-only \
  --mapping-output inference_output/unpaired_ablation/metadata/unpaired_mapping.json \
  --seed 42

python tools/stage3/generate_ablation_subset_cache.py \
  --data-root data/orion-crc33 \
  --style-mapping-json inference_output/unpaired_ablation/metadata/unpaired_mapping.json \
  --checkpoint-dir checkpoints/pixcell_controlnet_exp/npy_inputs \
  --output-dir inference_output/unpaired_ablation/ablation_results \
  --target-total-tiles 5000 \
  --seed 42 \
  --tile-sample-seed 42 \
  --device cuda \
  --guidance-scale 2.5 \
  --num-steps 20 \
  --jobs 1
```

## Step 1-2. Build the unpaired style mapping

Comment:
- Use this only for the unpaired workflow.
- This reuses the same tile IDs already present in paired ablation results.
- `--metadata-only` is the lean option when you do not want `inference_output/` to contain a copied or linked `data/` tree.

```bash
python tools/stage3/prepare_unpaired_ablation_dataset.py \
  --paired-cache-root inference_output/paired_ablation/ablation_results \
  --data-root data/orion-crc33 \
  --metadata-only \
  --mapping-output inference_output/unpaired_ablation/metadata/unpaired_mapping.json \
  --seed 42
```

Inspect the saved mapping:

```bash
cat inference_output/unpaired_ablation/metadata/unpaired_mapping.json
```

## Step 3. Generate Stage 3 ablation caches

Comment:
- Paired uses the original ORION root.
- Unpaired can now use the original ORION root plus `--style-mapping-json`.
- The generator now batches ablation conditions internally.
- On a single GPU, start with `--jobs 1`; only raise it if you have verified the extra worker helps on your hardware.

### Paired

```bash
python tools/stage3/generate_ablation_subset_cache.py \
  --data-root data/orion-crc33 \
  --checkpoint-dir checkpoints/pixcell_controlnet_exp/npy_inputs \
  --target-total-tiles 5000 \
  --output-dir inference_output/paired_ablation/ablation_results \
  --seed 42 \
  --tile-sample-seed 42 \
  --device cuda \
  --guidance-scale 2.5 \
  --num-steps 20 \
  --jobs 1
```

### Unpaired

```bash
python tools/stage3/generate_ablation_subset_cache.py \
  --data-root data/orion-crc33 \
  --style-mapping-json inference_output/unpaired_ablation/metadata/unpaired_mapping.json \
  --checkpoint-dir checkpoints/pixcell_controlnet_exp/npy_inputs \
  --target-total-tiles 5000 \
  --output-dir inference_output/unpaired_ablation/ablation_results \
  --seed 42 \
  --tile-sample-seed 42 \
  --device cuda \
  --guidance-scale 2.5 \
  --num-steps 20 \
  --jobs 1
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
- `all` now means `cosine lpips aji pq style_hed`
- Request a subset explicitly if you want to skip the slower HED pass

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
  --orion-root data/orion-crc33 \
  --style-mapping-json inference_output/unpaired_ablation/metadata/unpaired_mapping.json \
  --metrics aji pq style_hed \
  --device cuda \
  --jobs 2
```

If you want all metrics explicitly for unpaired:

```bash
python tools/compute_ablation_metrics.py \
  --cache-dir inference_output/unpaired_ablation/ablation_results \
  --orion-root data/orion-crc33 \
  --style-mapping-json inference_output/unpaired_ablation/metadata/unpaired_mapping.json \
  --metrics all style_hed \
  --device cuda \
  --jobs 2
```

## Step 7. Compute FUD

Comment:
- FUD is dataset-level and uses all tiles in the cache by default when using UNI features.
- It is written once per condition and then backfilled into every tile's `metrics.json`.

### Paired

```bash
python tools/compute_fid.py \
  --cache-dir inference_output/paired_ablation/ablation_results \
  --orion-root data/orion-crc33 \
  --device cuda \
  --batch-size 64 \
  --output inference_output/paired_ablation/ablation_results/fud_scores.json
```

### Unpaired

```bash
python tools/compute_fid.py \
  --cache-dir inference_output/unpaired_ablation/ablation_results \
  --orion-root data/orion-crc33 \
  --style-mapping-json inference_output/unpaired_ablation/metadata/unpaired_mapping.json \
  --device cuda \
  --batch-size 64 \
  --output inference_output/unpaired_ablation/ablation_results/fud_scores.json
```

## Step 7b. Compute FVD

Comment:
- FVD uses Virchow-2 tile embeddings and is written to `fvd_scores.json`.
- Access to `paige-ai/Virchow2` must be approved on Hugging Face before first use.

### Paired

```bash
python tools/compute_fid.py \
  --cache-dir inference_output/paired_ablation/ablation_results \
  --orion-root data/orion-crc33 \
  --feature-backend virchow2 \
  --virchow2-model hf-hub:paige-ai/Virchow2 \
  --device cuda \
  --batch-size 64 \
  --output inference_output/paired_ablation/ablation_results/fvd_scores.json
```

### Unpaired

```bash
python tools/compute_fid.py \
  --cache-dir inference_output/unpaired_ablation/ablation_results \
  --orion-root data/orion-crc33 \
  --style-mapping-json inference_output/unpaired_ablation/metadata/unpaired_mapping.json \
  --feature-backend virchow2 \
  --virchow2-model hf-hub:paige-ai/Virchow2 \
  --device cuda \
  --batch-size 64 \
  --output inference_output/unpaired_ablation/ablation_results/fvd_scores.json
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
  --orion-root data/orion-crc33 \
  --style-mapping-json inference_output/unpaired_ablation/metadata/unpaired_mapping.json \
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
- It does not truly recompute FUD on the filtered subset; it only filters the plotting inputs.
- `--metric-set unpaired` swaps `Cosine/LPIPS` out for `HED` so the figure matches the unpaired main metrics.

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
  --metric-set unpaired \
  --orion-root data/orion-crc33 \
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
  --orion-root data/orion-crc33 \
  --style-mapping-json inference_output/unpaired_ablation/metadata/unpaired_mapping.json \
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
  --exp-root data/orion-crc33 \
  --out inference_output/unpaired_ablation/channel_sweep/tile_classes.json
```

### Unpaired: run channel sweep

```bash
python tools/stage3/channel_sweep.py \
  --class-json inference_output/unpaired_ablation/channel_sweep/tile_classes.json \
  --data-root data/orion-crc33 \
  --style-mapping-json inference_output/unpaired_ablation/metadata/unpaired_mapping.json \
  --checkpoint-dir checkpoints/pixcell_controlnet_exp/npy_inputs \
  --out inference_output/unpaired_ablation/channel_sweep \
  --cache-dir inference_output/unpaired_ablation/channel_sweep/cache \
  --experiments 1 2 3 \
  --seed 42
```

## Practical notes

- If you want a fresh rerun, remove or rename the previous output folder first.
- For unpaired evaluation:
  - keep `PQ`, `AJI`, `FUD`
  - prefer `style_hed` as the style metric
  - avoid using `LPIPS` and `cosine` as the main conclusions unless you explicitly want paired-reference similarity
- For paired evaluation:
  - `all` is still the default metric bundle
  - `style_hed` is available but usually unnecessary
