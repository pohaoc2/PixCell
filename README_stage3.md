# Stage 3: Inference, Validation, and Analysis

This guide covers Stage 3 of the PixCell pipeline: running inference from simulation channels, validating checkpoints, and producing ablation and reporting artifacts.

[Open Paired Ablation Colab](https://colab.research.google.com/github/pohaoc2/PixCell/blob/main/notebook/stage3_paired_ablation_a100_colab.ipynb)

Prerequisites:

1. Dependencies installed from `[README.md](README.md)`.
2. Models and features prepared from `[stage1.md](stage1.md)`.
3. A trained checkpoint from `[stage2.md](stage2.md)`, or public weights for the pretrained verification flow.

---

## Stage 3: Inference

Generate experimental-like H&E from simulation channel images. At inference, CODEX-compatible experimental channels are replaced by simulation outputs with the same spatial format.

### Style-Conditioned (recommended)

Pass a reference H&E image to set tissue appearance such as staining and cell density:

```bash
python stage3_inference.py \
    --config           configs/config_controlnet_exp.py \
    --checkpoint-dir   checkpoints/pixcell_controlnet_exp/checkpoints/step_0000004 \
    --sim-channels-dir inference_data \
    --sim-id           sim_0001 \
    --reference-uni    data/orion-crc/features/0_256_uni.npy \
    --output           inference_data/generated_he.png
```

### TME-Only (no reference H&E)

Generate purely from TME layout. This requires `cfg_dropout_prob > 0` during training:

```bash
python stage3_inference.py \
    --config           configs/config_controlnet_exp.py \
    --checkpoint-dir   checkpoints/pixcell_controlnet_exp/checkpoints/step_XXXXXXX \
    --sim-channels-dir /path/to/sim_channels \
    --sim-id           sim_0001 \
    --output           generated_he.png
```

### Raw Batch Generation

For batch inference without extra visualizations:

```bash
python stage3_inference.py \
    --config           configs/config_controlnet_exp.py \
    --checkpoint-dir   checkpoints/pixcell_controlnet_exp/checkpoints/step_XXXXXXX \
    --sim-channels-dir /path/to/sim_channels \
    --output-dir       ./inference_output \
    --n-tiles          50
```

---

## Batch Evaluation and Visualization

### Batch Generation with Visualizations

Generate H&E plus the two Stage 3 visuals used in evaluation:

```bash
python tools/stage3/run_evaluation.py \
    --checkpoint-dir checkpoints/pixcell_controlnet_exp/checkpoints/zero_out_mask_post \
    --output-dir     inference_output/zero_out_mask_post \
    --n-tiles        3
```

Per patch, outputs under `{output_dir}/{tile_id}/{paired,unpaired}/`:


| File                | Contents                                               |
| ------------------- | ------------------------------------------------------ |
| `generated_he.png`  | Generated H&E image                                    |
| `overview.png`      | TME input channels -> generated H&E                    |
| `ablation_grid.png` | H&E + mask overlay, delta maps, and channel composites |


Paired means same tile UNI + TME. Unpaired means next tile UNI + current tile TME. The script also writes `metrics.json` with per-tile UNI cosine similarity scores.

Add `--no-metrics` to skip cosine similarity computation.

### Single-Tile Visualization + Ablation Test Suite

Generate the full Stage 3 visualization bundle for one tile:

```bash
python tools/stage3/generate_tile_vis.py \
    --config         configs/config_controlnet_exp.py \
    --checkpoint-dir checkpoints/pixcell_controlnet_exp/checkpoints/zero_out_mask_post \
    --data-root      data/orion-crc33 \
    --tile-id        YOUR_TILE_ID \
    --output-dir     inference_output/YOUR_TILE_ID
```

Add `--null-uni` for TME-only generation, or override style inputs with `--uni-npy` and `--reference-he`.

Outputs under `{output_dir}/`:


| File                         | Contents                                               |
| ---------------------------- | ------------------------------------------------------ |
| `overview.png`               | Input channels, reference style H&E, and generated H&E |
| `ablation_grid.png`          | Default progressive group-addition sweep               |
| `ablation_single_groups.png` | 4 single-group tests                                   |
| `ablation_group_pairs.png`   | All 6 two-group combinations                           |
| `ablation_group_triples.png` | All 4 three-group combinations                         |
| `ablation_orders/`           | 24 progressive addition orders                         |


The exhaustive suite is built from the four Stage 3 groups: `cell_types`, `cell_state`, `vasculature`, and `microenv`.

For unpaired analysis, keep using the original ORION root for layout and pass a mapping JSON when you need style references to come from a different tile. The mapping-only workflow avoids creating `inference_output/unpaired_ablation/data/...`, which keeps S3 uploads smaller.

---

## Channel Impact Analysis

Three CLIs support the channel-impact workflow:


| Script                            | Purpose                                                     | Typical command                                                                                                                                                                                                                                                                                            |
| --------------------------------- | ----------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `tools/vis/leave_one_out_diff.py` | Render leave-one-out figures from cached ablations          | `conda run -n pixcell python tools/vis/leave_one_out_diff.py --cache-root inference_output/paired_ablation/ablation_results --orion-root data/orion-crc33 --out-root inference_output/paired_ablation/leave_one_out --figure both`                                                                                  |
| `tools/stage3/classify_tiles.py`  | Classify tiles and write `tile_classes.json`                | `conda run -n pixcell python tools/stage3/classify_tiles.py --exp-root data/orion-crc33 --out tile_classes.json`                                                                                                                                                                                           |
| `tools/stage3/channel_sweep.py`   | Generate caches plus rendered figures for experiments 1/2/3 | `conda run -n pixcell python tools/stage3/channel_sweep.py --class-json tile_classes.json --data-root data/orion-crc33 --checkpoint-dir checkpoints/pixcell_controlnet_exp/npy_inputs --out inference_output/channel_sweep --cache-dir inference_output/channel_sweep/cache --experiments 1 2 3 --seed 42` |


Useful commands:

```bash
conda run -n pixcell python tools/vis/leave_one_out_diff.py \
    --cache-root inference_output/paired_ablation/ablation_results \
    --orion-root data/orion-crc33 \
    --out-root inference_output/paired_ablation/leave_one_out \
    --figure both

conda run -n pixcell python tools/vis/leave_one_out_diff.py \
    --cache-dir inference_output/paired_ablation/ablation_results/29952_46080 \
    --figure ssim \
    --crop-size 64 \
    --out inference_output/paired_ablation/leave_one_out/29952_46080/leave_one_out_ssim.png

conda run -n pixcell python tools/stage3/classify_tiles.py \
    --exp-root data/orion-crc33 \
    --out tile_classes.json

conda run -n pixcell python tools/stage3/channel_sweep.py \
    --class-json tile_classes.json \
    --data-root data/orion-crc33 \
    --checkpoint-dir checkpoints/pixcell_controlnet_exp/npy_inputs \
    --out inference_output/channel_sweep \
    --cache-dir inference_output/channel_sweep/cache \
    --experiments 1 2 3 \
    --seed 42
```

Unpaired variants can use the same `data/orion-crc33` layout root plus a mapping JSON:

```bash
conda run -n pixcell python tools/vis/leave_one_out_diff.py \
    --cache-root inference_output/unpaired_ablation/ablation_results \
    --orion-root data/orion-crc33 \
    --style-mapping-json inference_output/unpaired_ablation/metadata/unpaired_mapping.json \
    --out-root inference_output/unpaired_ablation/leave_one_out \
    --figure both

conda run -n pixcell python tools/stage3/channel_sweep.py \
    --class-json inference_output/unpaired_ablation/channel_sweep/tile_classes.json \
    --data-root data/orion-crc33 \
    --style-mapping-json inference_output/unpaired_ablation/metadata/unpaired_mapping.json \
    --checkpoint-dir checkpoints/pixcell_controlnet_exp/npy_inputs \
    --out inference_output/unpaired_ablation/channel_sweep \
    --cache-dir inference_output/unpaired_ablation/channel_sweep/cache \
    --experiments 1 2 3 \
    --seed 42
```

Re-render from an existing cache without rerunning inference:

```bash
conda run -n pixcell python tools/stage3/render_channel_sweep_figures.py \
    --cache-dir inference_output/channel_sweep/cache \
    --out inference_output/channel_sweep \
    --experiments 1 2 3
```

Targeted tests:

```bash
conda run -n pixcell python -m pytest \
    tests/test_leave_one_out_diff.py \
    tests/test_classify_tiles.py \
    tests/test_channel_sweep.py \
    tests/test_channel_sweep_cache.py -v
```

---

## Ablation + Metrics Workflow

Stage 3 ablation workflow, end to end:

1. Generate full-ablation caches.
2. Export cached H&E PNGs for CellViT.
3. Run CellViT externally.
4. Import CellViT outputs back into the cache tree.
5. Compute per-condition metrics.
6. Compute dataset-level FUD.
7. Render tile-level and dataset-level figures.
8. Render the paired-vs-unpaired scientific HTML report.

For the consolidated paired + unpaired workflow reference, see `[ablation_cli.md](ablation_cli.md)`.

When running unpaired ablations, prefer the mapping-only flow:

```bash
python tools/stage3/prepare_unpaired_ablation_dataset.py \
    --paired-cache-root inference_output/paired_ablation/ablation_results \
    --data-root data/orion-crc33 \
    --metadata-only \
    --mapping-output inference_output/unpaired_ablation/metadata/unpaired_mapping.json \
    --seed 42
```

That JSON lets the unpaired tools keep layout inputs from `data/orion-crc33` while pulling style references from mapped tiles, so you do not need `inference_output/unpaired_ablation/data/...`.

For clean reruns:

- regenerate paired to `5000` tiles first
- rebuild the unpaired mapping JSON from that paired cache
- then regenerate unpaired from scratch
- the cache generator now batches ablation conditions internally, so there is no extra batching flag to pass
- on a single A100, start with `--jobs 1`


| Script                                           | Purpose                                                                       | Typical command                                                                                                                                                                                                                        |
| ------------------------------------------------ | ----------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `tools/stage3/generate_ablation_subset_cache.py` | Generate cached single/pair/triple/all H&E PNGs plus `manifest.json`          | `python tools/stage3/generate_ablation_subset_cache.py --config configs/config_controlnet_exp.py --checkpoint-dir checkpoints/pixcell_controlnet_exp/checkpoints/zero_out_mask_post --data-root data/orion-crc33 --n-tiles 8 --jobs 4` |
| `tools/cellvit/export_batch.py`                  | Flatten cached PNGs for external CellViT processing                           | `python tools/cellvit/export_batch.py --cache-root inference_output/paired_ablation/ablation_results --output-dir inference_output/paired_ablation/cellvit_batch --zip`                                                                 |
| `tools/cellvit/import_results.py`                | Copy flat CellViT JSON results back beside each cached image                  | `python tools/cellvit/import_results.py --manifest inference_output/paired_ablation/cellvit_batch/manifest.json --results-dir inference_output/paired_ablation/cellvit`                                                                  |
| `tools/compute_ablation_metrics.py`              | Write `<cache-dir>/metrics.json` with cosine / LPIPS / AJI / PQ / `style_hed` | `conda run -n pixcell python tools/compute_ablation_metrics.py --cache-dir inference_output/paired_ablation/ablation_results --orion-root data/orion-crc33`                                                                             |
| `tools/compute_fid.py`                           | Compute dataset-level FUD for all 15 ablation conditions                      | `python tools/compute_fid.py --cache-dir inference_output/paired_ablation/ablation_results --device cuda`                                                                                                                               |
| `tools/stage3/ablation_grid_figure.py`           | Render the static ranked 4x4 matplotlib figure                                | `python tools/stage3/ablation_grid_figure.py --cache-dir inference_output/paired_ablation/ablation_results/YOUR_TILE_ID --orion-root data/orion-crc33 --sort-by pq --no-auto-cosine --jobs 8`                                          |
| `tools/render_dataset_metrics.py`                | Render the standalone dataset-level summary figure                            | `python tools/render_dataset_metrics.py --metric-dir inference_output/paired_ablation/ablation_results --output figures/dataset_metrics.png --dpi 400`                                                                                  |
| `tools/ablation_report`                          | Render the paired-vs-unpaired scientific HTML report                          | `python -m tools.ablation_report --output docs/ablation_scientific_report.html`                                                                                                                                                        |


Typical sequence:

```bash
python tools/stage3/generate_ablation_subset_cache.py \
    --config         configs/config_controlnet_exp.py \
    --checkpoint-dir checkpoints/pixcell_controlnet_exp/checkpoints/zero_out_mask_post \
    --data-root      data/orion-crc33 \
    --tile-id        YOUR_TILE_ID

python tools/stage3/generate_ablation_subset_cache.py \
    --config             configs/config_controlnet_exp.py \
    --checkpoint-dir     checkpoints/pixcell_controlnet_exp/checkpoints/zero_out_mask_post \
    --data-root          data/orion-crc33 \
    --output-dir         inference_output/paired_ablation/ablation_results \
    --target-total-tiles 5000 \
    --jobs               1

python tools/stage3/generate_ablation_subset_cache.py \
    --checkpoint-dir     checkpoints/pixcell_controlnet_exp/npy_inputs \
    --data-root          data/orion-crc33 \
    --output-dir         inference_output/paired_ablation/ablation_results \
    --target-total-tiles 5000 \
    --seed               42 \
    --tile-sample-seed   42 \
    --device             cuda \
    --guidance-scale     2.5 \
    --num-steps          20 \
    --jobs               1

python tools/stage3/prepare_unpaired_ablation_dataset.py \
    --paired-cache-root  inference_output/paired_ablation/ablation_results \
    --data-root          data/orion-crc33 \
    --metadata-only \
    --mapping-output     inference_output/unpaired_ablation/metadata/unpaired_mapping.json \
    --seed               42

python tools/stage3/generate_ablation_subset_cache.py \
    --data-root          data/orion-crc33 \
    --style-mapping-json inference_output/unpaired_ablation/metadata/unpaired_mapping.json \
    --checkpoint-dir     checkpoints/pixcell_controlnet_exp/npy_inputs \
    --output-dir         inference_output/unpaired_ablation/ablation_results \
    --target-total-tiles 5000 \
    --seed               42 \
    --tile-sample-seed   42 \
    --device             cuda \
    --guidance-scale     2.5 \
    --num-steps          20 \
    --jobs               1

python tools/stage3/ablation_grid_figure.py \
    --cache-dir inference_output/paired_ablation/ablation_results/YOUR_TILE_ID \
    --orion-root data/orion-crc33 \
    --sort-by pq \
    --no-auto-cosine

python tools/render_dataset_metrics.py \
    --metric-dir inference_output/paired_ablation/ablation_results \
    --output figures/dataset_metrics.png \
    --dpi 400

python -m tools.ablation_report \
    --paired-metrics-root inference_output/paired_ablation/ablation_results \
    --paired-dataset-root inference_output/paired_ablation \
    --paired-reference-root data/orion-crc33 \
    --unpaired-metrics-root inference_output/unpaired_ablation/ablation_results \
    --unpaired-dataset-root inference_output/unpaired_ablation \
    --unpaired-reference-root data/orion-crc33 \
    --unpaired-style-mapping-json inference_output/unpaired_ablation/metadata/unpaired_mapping.json \
    --output docs/ablation_scientific_report.html
```

If CellViT outputs have already been imported, compute metrics across the full cache root with:

```bash
conda run --no-capture-output -n pixcell \
    python -u tools/compute_ablation_metrics.py \
    --cache-dir inference_output/paired_ablation/ablation_results \
    --orion-root data/orion-crc33 \
    --metrics lpips aji pq \
    --lpips-batch-size 8
```

For unpaired metric computation without a remapped dataset tree:

```bash
conda run --no-capture-output -n pixcell \
    python -u tools/compute_ablation_metrics.py \
    --cache-dir inference_output/unpaired_ablation/ablation_results \
    --orion-root data/orion-crc33 \
    --style-mapping-json inference_output/unpaired_ablation/metadata/unpaired_mapping.json \
    --metrics aji pq style_hed \
    --lpips-batch-size 8

python tools/compute_fid.py \
    --cache-dir inference_output/unpaired_ablation/ablation_results \
    --orion-root data/orion-crc33 \
    --style-mapping-json inference_output/unpaired_ablation/metadata/unpaired_mapping.json \
    --device cuda \
    --output inference_output/unpaired_ablation/ablation_results/fud_scores.json
```

---

## Dataset-Level FUD

Use `tools/compute_fid.py` after `tools/compute_ablation_metrics.py` to compute Fréchet UNI Distance once per ablation condition across the full cached dataset by default. The script writes `<cache-dir>/fud_scores.json` and backfills `fud` into each per-tile `metrics.json`. If you explicitly use `--feature-backend inception`, it instead writes canonical `fid_scores.json` and backfills `fid`.

```bash
python tools/compute_fid.py \
    --cache-dir inference_output/paired_ablation/ablation_results \
    --device cuda
```

Optional flags:

- `--batch-size N` controls feature batching; default is `64`.
- `--output PATH` writes the JSON summary somewhere other than the backend-specific default (`<cache-dir>/fud_scores.json` for UNI, `<cache-dir>/fid_scores.json` for Inception).

Afterward, re-run `tools/render_dataset_metrics.py` to populate the FUD panel.

---

## Group Control at Inference

Selectively include or exclude TME channel groups:

```bash
python stage3_inference.py \
    --config configs/config_controlnet_exp.py \
    --checkpoint-dir checkpoints/pixcell_controlnet_exp/checkpoints/step_XXXXXXX \
    --sim-channels-dir /path/to/sim_channels \
    --sim-id sim_0001 \
    --active-groups cell_types vasculature \
    --output generated_he.png

python stage3_inference.py \
    --config configs/config_controlnet_exp.py \
    --checkpoint-dir checkpoints/pixcell_controlnet_exp/checkpoints/step_XXXXXXX \
    --sim-channels-dir /path/to/sim_channels \
    --sim-id sim_0001 \
    --drop-groups microenv \
    --output generated_he.png
```

---

## All Inference Flags


| Flag                 | Default  | Description                                |
| -------------------- | -------- | ------------------------------------------ |
| `--sim-channels-dir` | required | Root dir with per-channel subdirectories   |
| `--sim-id`           | -        | Single snapshot ID (file stem)             |
| `--output`           | -        | Output PNG for single-tile mode            |
| `--output-dir`       | -        | Output directory for batch mode            |
| `--n-tiles`          | all      | Max tiles in batch mode                    |
| `--reference-he`     | -        | Reference H&E image for style conditioning |
| `--reference-uni`    | -        | Precomputed UNI `.npy`                     |
| `--active-groups`    | all      | TME groups to include                      |
| `--drop-groups`      | none     | TME groups to exclude                      |
| `--guidance-scale`   | `2.5`    | CFG guidance scale                         |
| `--num-steps`        | `20`     | Denoising steps                            |
| `--device`           | `cuda`   | Device                                     |


---

## Simulation Channel Directory Layout

```text
sim_channels/
├── cell_mask/              {sim_id}.png   binary (required)
├── cell_type_healthy/      {sim_id}.png   (optional)
├── cell_type_cancer/       {sim_id}.png   (optional)
├── cell_type_immune/       {sim_id}.png   (optional)
├── cell_state_prolif/      {sim_id}.png   (optional)
├── cell_state_nonprolif/   {sim_id}.png   (optional)
├── cell_state_dead/        {sim_id}.png   (optional)
├── vasculature/            {sim_id}.png   (optional)
├── oxygen/                 {sim_id}.png or .npy
└── glucose/                {sim_id}.png or .npy
```

Only channels listed in `configs/config_controlnet_exp.py -> data.active_channels` are loaded.

---

## Validation

Evaluate sim-to-exp alignment by generating H&E from simulation TME channels and measuring cosine similarity against precomputed experimental UNI features:

```bash
python pipeline/validate_sim_to_exp.py \
    --config          configs/config_controlnet_exp.py \
    --sim-root        /path/to/sim_data_root \
    --exp-feat        /path/to/exp_data_root/features \
    --controlnet-ckpt checkpoints/pixcell_controlnet_exp/checkpoints/step_XXXXXXX \
    --tme-ckpt        checkpoints/pixcell_controlnet_exp/checkpoints/step_XXXXXXX \
    --uni-model       ./pretrained_models/uni-2h \
    --n-tiles         50 \
    --guidance-scale  2.5 \
    --output-dir      ./validation_output
```


| Flag                | Description                                      |
| ------------------- | ------------------------------------------------ |
| `--sim-root`        | Sim data root (contains `sim_channels/`)         |
| `--exp-feat`        | Directory of `{tile_id}_uni.npy` target features |
| `--controlnet-ckpt` | Checkpoint directory                             |
| `--tme-ckpt`        | Checkpoint directory, usually the same as above  |
| `--reference-uni`   | Optional `.npy` for style-conditioned mode       |
| `--n-tiles`         | Number of sim snapshots to evaluate              |
| `--guidance-scale`  | CFG guidance scale                               |
| `--output-dir`      | Optional output directory for generated H&E      |


Example output:

```text
  snap_0001: cosine_sim=0.7821
  snap_0002: cosine_sim=0.7543
  ...
=== Validation Results ===
N tiles:          50
Mean cosine sim:  0.771
Std cosine sim:   0.032
```

---

## Analysis Tools

Most visualization-facing CLIs live under `tools/vis/`, while CellViT batch import and export lives under `tools/cellvit/`. Shared Stage 3 implementation code remains in `tools/stage3/`, including:

- `tools/stage3/figures.py`
- `tools/stage3/ablation_cache.py`
- `tools/stage3/ablation_grid_figure.py`
- `tools/stage3/tile_pipeline.py`
- `tools/color_constants.py`

### Dataset Metrics Figure Renderer

Use `tools/render_dataset_metrics.py` to export the standalone dataset-level summary figure:

```bash
python tools/render_dataset_metrics.py \
    --metric-dir inference_output/paired_ablation/ablation_results
```

Optional flags:


| Flag                                 | Default                          | Description                                               |
| ------------------------------------ | -------------------------------- | --------------------------------------------------------- |
| `--metric-dir PATH`                  | `inference_output/paired_ablation/ablation_results` | Parent directory containing per-tile `metrics.json` files |
| `--output PATH`                      | `dataset_metrics.png`            | Output PNG path                                           |
| `--dpi N`                            | `300`                            | Export resolution                                         |
| `--metric-set {all,paired,unpaired}` | `paired`                         | Metric preset                                             |
| `--min-gt-cells N`                   | `0`                              | Filter out tiles with fewer than `N` GT instances         |
| `--orion-root PATH`                  | `data/orion-crc33`               | Dataset root used when GT lookup is needed                |


### Paired-vs-Unpaired Scientific HTML Report

Use `python -m tools.ablation_report` to build a single HTML report that compares paired and unpaired ablation runs side by side:

```bash
python -m tools.ablation_report
```

Optional flags:


| Flag                                 | Default                                               | Description                                                         |
| ------------------------------------ | ----------------------------------------------------- | ------------------------------------------------------------------- |
| `--paired-metrics-root PATH`         | `inference_output/paired_ablation/ablation_results`   | Parent directory with paired `metrics.json` files                   |
| `--paired-dataset-root PATH`         | `inference_output/paired_ablation`                    | Paired dataset root for figure lookup                               |
| `--paired-reference-root PATH`       | `data/orion-crc33`                                    | Reference H&E root for paired HED backfill                          |
| `--paired-style-mapping-json PATH`   | `None`                                                | Optional paired style mapping JSON for remapped-reference workflows |
| `--unpaired-metrics-root PATH`       | `inference_output/unpaired_ablation/ablation_results` | Parent directory with unpaired `metrics.json` files                 |
| `--unpaired-dataset-root PATH`       | `inference_output/unpaired_ablation`                  | Unpaired dataset root for figure lookup                             |
| `--unpaired-reference-root PATH`     | `data/orion-crc33`                                    | Layout dataset root used for unpaired reference lookup              |
| `--unpaired-style-mapping-json PATH` | `None`                                                | Mapping JSON that resolves unpaired style-reference tiles           |
| `--output PATH`                      | `docs/ablation_scientific_report.html`                | Output HTML path                                                    |
| `--title TEXT`                       | `Channel Ablation Scientific Report`                  | HTML page title                                                     |
| `--self-contained`                   | `False`                                               | Embed representative evidence images                                |
