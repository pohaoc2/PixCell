# Local CellViT Runbook For a4 UNI Probe

This note summarizes how CellViT was run locally for the `a4_uni_probe` outputs under `inference_output/a1_concat/a4_uni_probe`.

## Purpose

CellViT is used here as a post-processing stage on generated H&E tiles.

- Input: generated PNG tiles from the a4 sweep/null outputs
- Model: local CellViT checkpoint
- Output: one JSON per tile with detected cells
- Follow-up: import those JSONs back beside the source PNGs and refresh a4 metrics

## Local Prerequisites

- Repo root: `/home/ec2-user/PixCell`
- CellViT repo: `/home/ec2-user/CellViT`
- CellViT checkpoint: `/home/ec2-user/checkpoints/CellViT-256.pth`
- CellViT runner: `/home/ec2-user/he-feature-visualizer/stages/run_cellvit_local.py`
- Conda init script: `/home/ec2-user/miniconda3/etc/profile.d/conda.sh`
- Conda env: `cellvit`

The local runner expects a dedicated `cellvit` environment because its dependency stack conflicts with the main PixCell environment.

## One-Command Path For The Main a4 Batch

The standard local path for the generated a4 images is:

```bash
cd /home/ec2-user/PixCell
bash inference_output/a1_concat/a4_uni_probe/run_generated_cellvit_local.sh
```

That helper script does four things in order:

1. Export all nested a4 sweep/null PNGs into one flat CellViT batch zip.
2. Activate the local `cellvit` conda environment and run CellViT on the zip.
3. Import the resulting JSON files back beside the original PNGs.
4. Refresh the a4 metrics CSVs and summary JSONs.

The exact helper script is `/home/ec2-user/PixCell/inference_output/a1_concat/a4_uni_probe/run_generated_cellvit_local.sh`.

## What The Helper Actually Runs

### 1. Export generated images to a flat CellViT batch

```bash
python -m src.a4_uni_probe.postprocess export-cellvit-batch \
  --out-dir /home/ec2-user/PixCell/inference_output/a1_concat/a4_uni_probe \
  --batch-dir /home/ec2-user/PixCell/inference_output/a1_concat/a4_uni_probe/cellvit_generated_batch \
  --overwrite \
  --zip
```

This produces:

- Batch manifest: `/home/ec2-user/PixCell/inference_output/a1_concat/a4_uni_probe/cellvit_generated_batch/manifest.json`
- Batch zip: `/home/ec2-user/PixCell/inference_output/a1_concat/a4_uni_probe/cellvit_generated_batch.zip`

In this run, the exported generated batch contained `1,440` PNGs.

### 2. Run CellViT locally

Because the shell uses `set -u`, the helper temporarily disables nounset around conda activation.

```bash
set +u
source /home/ec2-user/miniconda3/etc/profile.d/conda.sh
conda activate cellvit
set -u

python /home/ec2-user/he-feature-visualizer/stages/run_cellvit_local.py \
  --zip /home/ec2-user/PixCell/inference_output/a1_concat/a4_uni_probe/cellvit_generated_batch.zip \
  --out /home/ec2-user/PixCell/inference_output/a1_concat/a4_uni_probe/cellvit_generated_results \
  --checkpoint /home/ec2-user/checkpoints/CellViT-256.pth \
  --cellvit-repo /home/ec2-user/CellViT
```

The local runner:

- unzips the flat PNG batch
- loads the CellViT checkpoint on local GPU if available
- writes one JSON per patch into the output directory
- optionally bundles those JSONs into `<out>.zip`

### 3. Import JSON results back beside the source PNGs

```bash
python tools/cellvit/import_results.py \
  --manifest /home/ec2-user/PixCell/inference_output/a1_concat/a4_uni_probe/cellvit_generated_batch/manifest.json \
  --results-dir /home/ec2-user/PixCell/inference_output/a1_concat/a4_uni_probe/cellvit_generated_results \
  --prefer-ext .json \
  --sidecar-suffix .png
```

For the main generated batch, the import step writes sidecars in the `.png.json` form next to the original images.

In this run, the import report matched `1,440` CellViT outputs.

### 4. Refresh a4 metrics after import

```bash
python -m src.a4_uni_probe.postprocess refresh-metrics \
  --out-dir /home/ec2-user/PixCell/inference_output/a1_concat/a4_uni_probe
```

This re-reads the imported CellViT sidecars, updates the per-image metric tables, and rebuilds the slope/null summary JSONs.

## Manual Full-Null Baseline Path

The shared `full_uni_null` baseline used the same local CellViT runner, but on a smaller flat batch produced from:

`/home/ec2-user/PixCell/inference_output/a1_concat/a4_uni_probe/full_uni_null_shared`

### Export the 30 full-null baseline PNGs

```bash
python tools/cellvit/export_batch.py \
  --cache-root /home/ec2-user/PixCell/inference_output/a1_concat/a4_uni_probe/full_uni_null_shared \
  --output-dir /home/ec2-user/PixCell/inference_output/a1_concat/a4_uni_probe/cellvit_full_null_batch \
  --overwrite \
  --zip
```

### Run CellViT on the full-null zip

```bash
set +u
source /home/ec2-user/miniconda3/etc/profile.d/conda.sh
conda activate cellvit
set -u

python /home/ec2-user/he-feature-visualizer/stages/run_cellvit_local.py \
  --zip /home/ec2-user/PixCell/inference_output/a1_concat/a4_uni_probe/cellvit_full_null_batch.zip \
  --out /home/ec2-user/PixCell/inference_output/a1_concat/a4_uni_probe/cellvit_full_null_results \
  --checkpoint /home/ec2-user/checkpoints/CellViT-256.pth \
  --cellvit-repo /home/ec2-user/CellViT
```

### Import the full-null JSON files back to the baseline images

```bash
python tools/cellvit/import_results.py \
  --manifest /home/ec2-user/PixCell/inference_output/a1_concat/a4_uni_probe/cellvit_full_null_batch/manifest.json \
  --results-dir /home/ec2-user/PixCell/inference_output/a1_concat/a4_uni_probe/cellvit_full_null_results \
  --prefer-ext .json \
  --sidecar-suffix _cellvit_instances
```

In this run:

- full-null batch size: `30` PNGs
- imported full-null JSONs: `30`

## Output Locations

Main generated-image run:

- Batch folder: `/home/ec2-user/PixCell/inference_output/a1_concat/a4_uni_probe/cellvit_generated_batch`
- Batch zip: `/home/ec2-user/PixCell/inference_output/a1_concat/a4_uni_probe/cellvit_generated_batch.zip`
- Raw CellViT JSONs: `/home/ec2-user/PixCell/inference_output/a1_concat/a4_uni_probe/cellvit_generated_results`
- Import report: `/home/ec2-user/PixCell/inference_output/a1_concat/a4_uni_probe/cellvit_generated_batch/import_report.json`

Shared full-null run:

- Batch folder: `/home/ec2-user/PixCell/inference_output/a1_concat/a4_uni_probe/cellvit_full_null_batch`
- Batch zip: `/home/ec2-user/PixCell/inference_output/a1_concat/a4_uni_probe/cellvit_full_null_batch.zip`
- Raw CellViT JSONs: `/home/ec2-user/PixCell/inference_output/a1_concat/a4_uni_probe/cellvit_full_null_results`
- Import report: `/home/ec2-user/PixCell/inference_output/a1_concat/a4_uni_probe/cellvit_full_null_batch/import_report.json`

## Practical Notes

- Use `conda.sh` directly instead of `source ~/.bashrc` when running non-interactive helpers.
- If the shell has `set -u`, wrap `conda activate` and `conda deactivate` with `set +u` / `set -u`.
- For the nested a4 sweep/null layout, use `python -m src.a4_uni_probe.postprocess export-cellvit-batch` rather than the generic exporter.
- After importing results, always rerun `python -m src.a4_uni_probe.postprocess refresh-metrics` so the summary CSVs and JSONs reflect the new CellViT sidecars.