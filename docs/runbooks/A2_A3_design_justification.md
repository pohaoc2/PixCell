# A2 / A3 Runbook (Colab + Local)

## 0. Prereq: Short-Proxy Length

Run one seed of `configs/config_controlnet_exp.py` with the schedule reduced to
about 25% of the full run. Confirm paired-test FID ranking against the
production run matches at full schedule on one reference seed. Lock that step
count as the proxy length for A2/A3 short-proxy runs and record it in the spec
results.

## 1. Colab: Per-Seed Training

Open `notebook/multichannel_controlnet.ipynb`. Keep the setup cells unchanged
(clone, install, AWS credentials, data download, stage0/stage1, dependencies).
After setup, replace the single training cell with this per-seed pattern and
rerun it once per seed.

### A2 Short-Proxy Seeds 1-5

```python
%cd /content/PixCell
!git pull origin main

VARIANT  = "a2_bypass"
CONFIG   = "configs/config_controlnet_exp_a2_bypass.py"
SEED     = 1
WORK_DIR = f"/content/PixCell/checkpoints/pixcell_controlnet_exp_{VARIANT}/seed_{SEED}"
S3_BASE  = f"s3://bagherilab-working/pohao/share_space/a2_a3/{VARIANT}/seed_{SEED}"

!accelerate launch stage2_train.py {CONFIG} \
    --work-dir {WORK_DIR} \
    --seed {SEED}

!aws s3 cp {WORK_DIR}/train_log.jsonl {S3_BASE}/train_log.jsonl
!aws s3 cp {WORK_DIR}/checkpoints/ {S3_BASE}/checkpoint/ --recursive --quiet
```

Change `SEED` to `2`, `3`, `4`, and `5` across runs.

### A2 Full-Headline Seed

Use the same cell with `SEED = 42`, production-length schedule, `WORK_DIR`
ending in `full_seed_42`, and `S3_BASE` ending in `/full_seed_42`.

### A3 `zero_init=False`

Use the same pattern with:

```python
VARIANT = "a3_no_zero_init"
CONFIG = "configs/config_controlnet_exp_a3_no_zero_init.py"
```

Run seeds `1` through `5` for the short proxy plus one full-headline seed.

### A3 Production-True Reference Logs

If existing production True checkpoints do not have `train_log.jsonl`, retrain a
short-proxy True reference with:

```python
VARIANT = "a3_zero_init_true"
CONFIG = "configs/config_controlnet_exp.py"
```

## 2. Colab: Inference

After training artifacts are uploaded, run inference cells in the same notebook
or a sibling notebook.

### Production Row

Reuse `inference_output/paired_ablation/production/` if available. Otherwise
rerun production inference and sync outputs to:

```text
s3://bagherilab-working/pohao/share_space/a2_a3/a2_bypass/inference/production/
```

### Bypass-Probe Row

```python
%cd /content/PixCell
!git pull origin main

!aws s3 cp s3://bagherilab-working/pohao/share_space/a2_a3/a2_bypass/full_seed_42/checkpoint/ \
    /content/PixCell/checkpoints/pixcell_controlnet_exp_a2_bypass/full_seed_42/checkpoints/ \
    --recursive --quiet

!python -m tools.ablation_a2.run_bypass_probe \
    --checkpoint /content/PixCell/checkpoints/pixcell_controlnet_exp_a2_bypass/full_seed_42 \
    --config configs/config_controlnet_exp_a2_bypass.py \
    --tile_ids tools/ablation_report/paired_test_tile_ids.txt \
    --out_dir inference_output/a2_bypass/bypass

!aws s3 cp inference_output/a2_bypass/bypass/ \
    s3://bagherilab-working/pohao/share_space/a2_a3/a2_bypass/inference/bypass/ \
    --recursive --quiet
```

### Off-The-Shelf PixCell Row

```python
!python -m tools.baselines.pixcell_offshelf_inference \
    --controlnet pretrained_models/pixcell-256-controlnet/controlnet/diffusion_pytorch_model.safetensors \
    --base pretrained_models/pixcell-256/transformer \
    --vae pretrained_models/sd-3.5-vae/vae \
    --uni data/orion-crc33/features \
    --tile_ids tools/ablation_report/paired_test_tile_ids.txt \
    --out_dir inference_output/a2_bypass/offshelf

!aws s3 cp inference_output/a2_bypass/offshelf/ \
    s3://bagherilab-working/pohao/share_space/a2_a3/a2_bypass/inference/offshelf/ \
    --recursive --quiet
```

### A3 Inference

For each A3 seed, pull the checkpoint from S3 and run the existing paired-test
inference flow to populate:

```text
inference_output/a3_zero_init/<variant>/seed_<k>/
```

Upload each generated directory back to S3.

## 3. Colab: CellViT Pass

Run the existing CellViT export/import flow for each generated PNG directory:

```python
!python -m tools.cellvit.export_batch \
    --input_dir inference_output/a2_bypass/bypass \
    --output_dir inference_output/a2_bypass/bypass/cellvit
```

Repeat for production, off-the-shelf, and each A3 variant/seed directory, then
merge with `tools.cellvit.import_results`.

## 4. Colab: Metric Aggregation

A2 metrics should be written to:

```text
inference_output/a2_bypass/metrics_summary.json
```

with schema:

```json
{
  "rows": [
    {"key": "production", "fid": 0, "uni_cos": 0, "cellvit_count_r": 0, "cellvit_type_kl": 0, "cellvit_nuc_ks": 0},
    {"key": "bypass_probe", "fid": 0, "uni_cos": 0, "cellvit_count_r": 0, "cellvit_type_kl": 0, "cellvit_nuc_ks": 0},
    {"key": "off_the_shelf", "fid": 0, "uni_cos": 0, "cellvit_count_r": 0, "cellvit_type_kl": 0, "cellvit_nuc_ks": 0}
  ]
}
```

Upload it to:

```text
s3://bagherilab-working/pohao/share_space/a2_a3/a2_bypass/metrics_summary.json
```

A3 stability:

```python
!mkdir -p /tmp/a3_logs/true /tmp/a3_logs/false
!aws s3 sync s3://bagherilab-working/pohao/share_space/a2_a3/a3_zero_init_true/ /tmp/a3_logs/true/
!aws s3 sync s3://bagherilab-working/pohao/share_space/a2_a3/a3_no_zero_init/ /tmp/a3_logs/false/

!python -m tools.ablation_a3.aggregate_stability \
    --seed_dirs /tmp/a3_logs/true/seed_1 /tmp/a3_logs/true/seed_2 /tmp/a3_logs/true/seed_3 \
                /tmp/a3_logs/true/seed_4 /tmp/a3_logs/true/seed_5 \
    --out inference_output/a3_zero_init/stability_true.json \
    --fixed_step 10000 \
    --grad_threshold 100.0

!python -m tools.ablation_a3.aggregate_stability \
    --seed_dirs /tmp/a3_logs/false/seed_1 /tmp/a3_logs/false/seed_2 /tmp/a3_logs/false/seed_3 \
                /tmp/a3_logs/false/seed_4 /tmp/a3_logs/false/seed_5 \
    --out inference_output/a3_zero_init/stability_false.json \
    --fixed_step 10000 \
    --grad_threshold 100.0

!aws s3 cp inference_output/a3_zero_init/stability_true.json \
    s3://bagherilab-working/pohao/share_space/a2_a3/a3_zero_init/stability_true.json
!aws s3 cp inference_output/a3_zero_init/stability_false.json \
    s3://bagherilab-working/pohao/share_space/a2_a3/a3_zero_init/stability_false.json
```

A3 metric summary should be written to:

```text
inference_output/a3_zero_init/metrics_summary.json
```

with the schema expected by `src/paper_figures/fig_si_a3_zero_init.py`.

## 5. Local EC2: Figure Rendering

After all S3 uploads complete:

```bash
cd /home/ec2-user/PixCell

aws s3 sync s3://bagherilab-working/pohao/share_space/a2_a3/a2_bypass/ inference_output/a2_bypass/
aws s3 sync s3://bagherilab-working/pohao/share_space/a2_a3/a3_zero_init/ inference_output/a3_zero_init/
aws s3 sync s3://bagherilab-working/pohao/share_space/a2_a3/a3_zero_init_true/ checkpoints/pixcell_controlnet_exp/
aws s3 sync s3://bagherilab-working/pohao/share_space/a2_a3/a3_no_zero_init/ checkpoints/pixcell_controlnet_exp_a3_no_zero_init/

conda run -n pixcell python -m src.paper_figures.main
```

Expected outputs:

- `figures/pngs/SI_A2_bypass_probe.png`
- `figures/pngs_updated/SI_A2_bypass_probe.png`
- `figures/pngs/SI_A3_zero_init.png`
- `figures/pngs_updated/SI_A3_zero_init.png`
