# Task CLIs

Run all commands from the repository root:

```bash
cd /home/pohaoc2/UW/bagherilab/PixCell
```

Data roots used below:

```bash
export EXP_ROOT=data/orion-crc33
export HE_FEATURE_ROOT=/home/pohaoc2/UW/bagherilab/he-feature-visualizer/data
```

## Status

- CPU-runnable on this machine: `a0_visibility_map`, `a0_tradeoff_scatter`, `a1_mask_targets`, `a1_probe_linear`, `a1_probe_mlp`, `a1_codex_targets`
- Planner-only on this machine; execution deferred to GPU machine: `a1_probe_encoders`, `a1_generated_probe`, `a2_decomposition`, `a3_combinatorial_sweep`

## Recommended order

```bash
python -m src.a0_visibility_map.run \
  --paired-stats-root inference_output/paired_ablation \
  --unpaired-stats-root inference_output/unpaired_ablation \
  --out-dir src/a0_visibility_map/out

python -m src.a0_tradeoff_scatter.run \
  --paired-metric-dir inference_output/paired_ablation \
  --unpaired-metric-dir inference_output/unpaired_ablation \
  --out-dir src/a0_tradeoff_scatter/out

python -m src.a1_mask_targets.main \
  --features-dir "$EXP_ROOT/features" \
  --exp-channels-dir "$EXP_ROOT/exp_channels" \
  --out-dir src/a1_mask_targets/out

python -m src.a1_probe_linear.main \
  --features-dir "$EXP_ROOT/features" \
  --targets-path src/a1_mask_targets/out/mask_targets_T1.npy \
  --tile-ids-path src/a1_mask_targets/out/tile_ids.txt \
  --target-names-path src/a1_mask_targets/out/target_names_T1.json \
  --out-dir src/a1_probe_linear/out

python -m src.a1_probe_mlp.main \
  --features-dir "$EXP_ROOT/features" \
  --targets-path src/a1_mask_targets/out/mask_targets_T1.npy \
  --tile-ids-path src/a1_mask_targets/out/tile_ids.txt \
  --target-names-path src/a1_mask_targets/out/target_names_T1.json \
  --cv-splits-path src/a1_probe_linear/out/cv_splits.json \
  --linear-results-json src/a1_probe_linear/out/linear_probe_results.json \
  --out-dir src/a1_probe_mlp/out

python -m src.a1_codex_targets.build \
  --features-csv "$HE_FEATURE_ROOT/features_crc33.csv" \
  --markers-csv "$HE_FEATURE_ROOT/markers.csv" \
  --tile-ids-path src/a1_mask_targets/out/tile_ids.txt \
  --out-dir src/a1_codex_targets/out

python -m src.a1_codex_targets.probe \
  --features-dir "$EXP_ROOT/features" \
  --tile-ids-path src/a1_mask_targets/out/tile_ids.txt \
  --cv-splits-path src/a1_probe_linear/out/cv_splits.json \
  --t2-targets-path src/a1_codex_targets/out/codex_T2_mean.npy \
  --marker-names-path src/a1_codex_targets/out/codex_marker_names.json \
  --t3-targets-path src/a1_codex_targets/out/codex_T3_quantiles.npy \
  --quantile-names-path src/a1_codex_targets/out/codex_T3_feature_names.json \
  --out-dir src/a1_codex_targets/probe_out
```

## Task-by-task CLI summary

### `a0_visibility_map`

Build the paired vs unpaired leave-one-out visibility chart and summary CSV.

```bash
python -m src.a0_visibility_map.run \
  --paired-stats-root inference_output/paired_ablation \
  --unpaired-stats-root inference_output/unpaired_ablation \
  --out-dir src/a0_visibility_map/out \
  --dpi 300 \
  --n-inset-tiles 6
```

Outputs:
- `src/a0_visibility_map/out/visibility_bar_chart.png`
- `src/a0_visibility_map/out/visibility_summary_table.csv`
- `src/a0_visibility_map/out/inset_tiles/`

### `a0_tradeoff_scatter`

Build paired and unpaired specificity-realism tradeoff panels.

```bash
python -m src.a0_tradeoff_scatter.run \
  --paired-metric-dir inference_output/paired_ablation \
  --unpaired-metric-dir inference_output/unpaired_ablation \
  --out-dir src/a0_tradeoff_scatter/out \
  --dpi 300
```

Outputs:
- `src/a0_tradeoff_scatter/out/tradeoff_data.csv`
- `src/a0_tradeoff_scatter/out/tradeoff_scatter_paired.png`
- `src/a0_tradeoff_scatter/out/tradeoff_scatter_unpaired.png`

### `a1_mask_targets`

Build T1 tile-level targets from `exp_channels/`.

```bash
python -m src.a1_mask_targets.main \
  --features-dir "$EXP_ROOT/features" \
  --exp-channels-dir "$EXP_ROOT/exp_channels" \
  --out-dir src/a1_mask_targets/out \
  --resolution 256
```

Outputs:
- `src/a1_mask_targets/out/mask_targets_T1.npy`
- `src/a1_mask_targets/out/tile_ids.txt`
- `src/a1_mask_targets/out/target_names_T1.json`
- `src/a1_mask_targets/out/target_stats.csv`

### `a1_probe_linear`

Run the linear probe on UNI embeddings against T1 targets.

```bash
python -m src.a1_probe_linear.main \
  --features-dir "$EXP_ROOT/features" \
  --targets-path src/a1_mask_targets/out/mask_targets_T1.npy \
  --tile-ids-path src/a1_mask_targets/out/tile_ids.txt \
  --target-names-path src/a1_mask_targets/out/target_names_T1.json \
  --out-dir src/a1_probe_linear/out \
  --n-splits 5 \
  --block-size-px 2048 \
  --alpha 1.0
```

Outputs:
- `src/a1_probe_linear/out/linear_probe_results.json`
- `src/a1_probe_linear/out/linear_probe_results.csv`
- `src/a1_probe_linear/out/cv_splits.json`
- `src/a1_probe_linear/out/linear_probe_fold_scores.npy`
- `src/a1_probe_linear/out/linear_probe_oof_predictions.npy`
- `src/a1_probe_linear/out/linear_probe_coef_mean.npy`

### `a1_probe_mlp`

Run the MLP probe on the same feature/target set and reuse the linear CV splits.

```bash
python -m src.a1_probe_mlp.main \
  --features-dir "$EXP_ROOT/features" \
  --targets-path src/a1_mask_targets/out/mask_targets_T1.npy \
  --tile-ids-path src/a1_mask_targets/out/tile_ids.txt \
  --target-names-path src/a1_mask_targets/out/target_names_T1.json \
  --cv-splits-path src/a1_probe_linear/out/cv_splits.json \
  --linear-results-json src/a1_probe_linear/out/linear_probe_results.json \
  --out-dir src/a1_probe_mlp/out \
  --random-state 42
```

Outputs:
- `src/a1_probe_mlp/out/mlp_probe_results.json`
- `src/a1_probe_mlp/out/mlp_probe_results.csv`
- `src/a1_probe_mlp/out/comparison_vs_linear.csv`

### `a1_codex_targets.build`

Build T2 means and T3 quantiles from CRC33 per-cell CODEX features.

```bash
python -m src.a1_codex_targets.build \
  --features-csv "$HE_FEATURE_ROOT/features_crc33.csv" \
  --markers-csv "$HE_FEATURE_ROOT/markers.csv" \
  --tile-ids-path src/a1_mask_targets/out/tile_ids.txt \
  --out-dir src/a1_codex_targets/out \
  --min-cells 1
```

Outputs:
- `src/a1_codex_targets/out/codex_T2_mean.npy`
- `src/a1_codex_targets/out/codex_T3_quantiles.npy`
- `src/a1_codex_targets/out/codex_cell_counts.npy`
- `src/a1_codex_targets/out/codex_marker_names.json`
- `src/a1_codex_targets/out/codex_T3_feature_names.json`

### `a1_codex_targets.probe`

Run linear and MLP probes over T2 and optionally T3.

```bash
python -m src.a1_codex_targets.probe \
  --features-dir "$EXP_ROOT/features" \
  --tile-ids-path src/a1_mask_targets/out/tile_ids.txt \
  --cv-splits-path src/a1_probe_linear/out/cv_splits.json \
  --t2-targets-path src/a1_codex_targets/out/codex_T2_mean.npy \
  --marker-names-path src/a1_codex_targets/out/codex_marker_names.json \
  --t3-targets-path src/a1_codex_targets/out/codex_T3_quantiles.npy \
  --quantile-names-path src/a1_codex_targets/out/codex_T3_feature_names.json \
  --out-dir src/a1_codex_targets/probe_out
```

Outputs:
- `src/a1_codex_targets/probe_out/t2_linear/`
- `src/a1_codex_targets/probe_out/t2_mlp/`
- `src/a1_codex_targets/probe_out/t3_linear/`
- `src/a1_codex_targets/probe_out/t3_mlp/`

## Planner CLIs for GPU tasks

These commands are safe to run on this machine. They do not execute the heavy GPU jobs. They only validate inputs, inspect runtime availability, and write `plan.json` into the task output directory.

### `a1_probe_encoders`

```bash
python -m src.a1_probe_encoders.main \
  --he-dir "$EXP_ROOT/he" \
  --targets-path src/a1_mask_targets/out/mask_targets_T1.npy \
  --tile-ids-path src/a1_mask_targets/out/tile_ids.txt \
  --cv-splits-path src/a1_probe_linear/out/cv_splits.json \
  --out-dir src/a1_probe_encoders/out \
  --virchow-weights /path/to/virchow_weights.pt
```

Output:
- `src/a1_probe_encoders/out/plan.json`

### `a1_generated_probe`

```bash
python -m src.a1_generated_probe.main \
  --generated-root inference_output/paired_ablation \
  --uni-model-path pretrained_models/uni-2h \
  --targets-path src/a1_mask_targets/out/mask_targets_T1.npy \
  --tile-ids-path src/a1_mask_targets/out/tile_ids.txt \
  --cv-splits-path src/a1_probe_linear/out/cv_splits.json \
  --out-dir src/a1_generated_probe/out
```

Output:
- `src/a1_generated_probe/out/plan.json`

### `a2_decomposition`

```bash
python -m src.a2_decomposition.main \
  --config-path configs/config_controlnet_exp.py \
  --checkpoint-dir /path/to/checkpoint_dir \
  --data-root "$EXP_ROOT" \
  --out-dir src/a2_decomposition/out \
  --sample-n 500
```

Output:
- `src/a2_decomposition/out/plan.json`

### `a3_combinatorial_sweep`

First create a newline-delimited anchor list, for example `src/a3_combinatorial_sweep/anchors.txt`:

```text
10240_11008
10496_11776
12800_16384
```

Then run:

```bash
python -m src.a3_combinatorial_sweep.main \
  --config-path configs/config_controlnet_exp.py \
  --checkpoint-dir /path/to/checkpoint_dir \
  --data-root "$EXP_ROOT" \
  --out-dir src/a3_combinatorial_sweep/out \
  --anchor-tile-ids-path src/a3_combinatorial_sweep/anchors.txt
```

Output:
- `src/a3_combinatorial_sweep/out/plan.json`

## Notes for the GPU machine

- The planner CLIs above intentionally stop at `plan.json` on a CPU-only machine.
- Do not pass `--worker` manually on this machine; those code paths are placeholders for the GPU execution environment.
- The most useful files to move to the GPU machine are:
  - `src/a1_mask_targets/out/`
  - `src/a1_probe_linear/out/cv_splits.json`
  - `src/a1_codex_targets/out/`
  - the generated `plan.json` files for the GPU-bound tasks

## Validation

Current focused test suite for the task CLIs and packages:

```bash
pytest -q \
  tests/test_task_a0_visibility_map.py \
  tests/test_task_a0_tradeoff_scatter.py \
  tests/test_task_a1_mask_targets.py \
  tests/test_task_a1_probe_linear.py \
  tests/test_task_a1_probe_mlp.py \
  tests/test_task_a1_codex_targets.py \
  tests/test_task_a1_codex_probe_cli.py \
  tests/test_task_gpu_wrappers.py
```
