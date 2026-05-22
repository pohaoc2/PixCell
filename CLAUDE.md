# PixCell Quick Context

- Goal: diffusion ControlNet maps spatial TME channels → realistic H&E.
- Train: paired ORION-CRC tiles (H&E + CODEX channels). Inference: unpaired sim channels.
- Primary path: paired-exp ControlNet + raw/concat conditioning. Grouped TME + sim-only kept for ablations.

## Pipeline

1. `stage0_setup.py` — fetch PixCell-256, ControlNet init, UNI-2h, SD3.5 VAE into `pretrained_models/`.
2. `stage1_extract_features.py` — cache UNI embeddings + SD3.5 VAE latents.
3. `stage2_train.py` — train via `configs/config_controlnet_exp.py`.
4. `stage3_inference.py` — gen H&E from exp channels, optional ref-H&E style.

## Model

- Frozen: PixCell-256 transformer, SD3.5 VAE, UNI-2h.
- Trainable: ControlNet + active conditioning module (`RawConditioningPassthrough` default; `MultiGroupTMEModule` for grouped ablations).
- Channel groups: `cell_types` (healthy/cancer/immune), `cell_state` (prolif/nonprolif/dead), `vasculature`, `microenv` (oxygen/glucose). Per-group CNN + cross-attn, zero-init residuals.
- `zero_mask_latent=True` (post-TME): `fused = tme(vae_mask) - vae_mask`. Closes bypass, keeps spatial structure. Apply post-TME in train/inference/helpers.
- `cfg_dropout_prob=0.15` zeros UNI embeddings → enables TME-only inference.

## Data Contract

- Paired root: `data/orion-crc33`. Subdirs: `exp_channels/`, `features/`, `vae_features/`, `metadata/exp_index.hdf5`.
- First channel `cell_masks` (legacy alias `cell_mask` accepted).
- Required binary: `cell_masks`, 3 cell-type, 3 cell-state. Optional: `vasculature`, `oxygen`, `glucose`.
- Missing `mask_sd3_vae` → zeros. Missing optional sim channels skipped at inference.

## Important Files

- Config: `configs/config_controlnet_exp.py`
- Train loop: `train_scripts/train_controlnet_exp.py`
- Dataset: `diffusion/data/datasets/paired_exp_controlnet_dataset.py`
- Group helpers: `tools/channel_group_utils.py`
- Inference: `stage3_inference.py`; batch vis: `tools/stage3/run_evaluation.py`
- Vis: `tools/stage3/figures.py`, `tools/stage3/tile_pipeline.py`
- Palette: `tools/color_constants.py`
- Architecture: `MODEL_ARCHITECTURE.md`

## Tests

- `tests/test_paired_exp_dataset.py` — paired dataset contract + alias.
- `tests/test_multi_group_tme.py` — shape, zero-init, group gating, residuals.
- `tests/test_channel_group_utils.py` — group splitting.
- `tests/test_train_controlnet_exp.py` — CFG dropout, conditioning logic, helpers.

## Working Assumptions

- Prefer paired-exp + concat unless task targets legacy grouped/sim code.
- Preserve `cell_masks`/`cell_mask` compatibility in dataset/inference touches.
- Update tests when changing inference/training group logic.
- `README.md` = full setup; this file = short agent handoff.

## CellViT — Local Execution

Full runbook: `CELLVIT_LOCAL_RUNBOOK.md`. Repo: `/home/ec2-user/CellViT`. Checkpoint: `/home/ec2-user/checkpoints/CellViT-256.pth`. Runner: `/home/ec2-user/he-feature-visualizer/stages/run_cellvit_local.py`. Env: `cellvit` (separate from `pixcell`, conflicting deps).

Three-step pattern for any ablation cache:

```bash
# 1. Export PNGs to flat batch
conda run --no-capture-output -n pixcell python tools/cellvit/export_batch.py \
  --cache-root <cache_dir> --output-dir /tmp/cellvit_batch --overwrite --zip

# 2. Run CellViT (cellvit env)
set +u; source /home/ec2-user/miniconda3/etc/profile.d/conda.sh; conda activate cellvit; set -u
python /home/ec2-user/he-feature-visualizer/stages/run_cellvit_local.py \
  --zip /tmp/cellvit_batch.zip --out /tmp/cellvit_results \
  --checkpoint /home/ec2-user/checkpoints/CellViT-256.pth \
  --cellvit-repo /home/ec2-user/CellViT

# 3. Import JSON sidecars back beside source PNGs (suffix: _cellvit_instances)
conda run --no-capture-output -n pixcell python tools/cellvit/import_results.py \
  --manifest /tmp/cellvit_batch/manifest.json --results-dir /tmp/cellvit_results
```

Then re-run `compute_ablation_metrics.py --metrics dice` (or `all`) on cache dir.

## Token Efficiency — Codex for Heavy Commands

Before running commands with large output (git diffs, recursive listings, large logs, full dataset inspection), delegate to Codex subagent. See `CODEX_COMMANDS.md`. Trigger: "Use Codex to run `<cmd>` and summarize." Codex bills OpenAI, not Claude. **Always** delegate any command listed in `CODEX_COMMANDS.md`.

## Claude Role Split — DISABLED 2026-05-21

Stay in Claude for plan + implement. Use Edit/Write directly; do **not** delegate to `codex:codex-rescue`. (Re-enable by removing this note. Original policy: Claude plans/reviews, Codex implements via `codex:codex-rescue`.)

## Memory Limits — Cap Long-Running Jobs

Box: 32 GB RAM, no swap. Wrap any job allocating > ~10 GB with a memory cap so OOM cannot kill user session.

- Launch wrap: `prlimit --as=24000000000 -- <cmd>` (24 GB AS cap) or `systemd-run --user --scope -p MemoryMax=24G -p MemorySwapMax=0 -- bash -c "<cmd>"` (tree-wide).
- Live PID: `prlimit --pid <pid> --as=24000000000`.
- Default cap: **24 GB**. Raise only after profiling shows a single fit needs more.
- Mandatory for: probe training, full-dataset feature extraction, anything loading the 10 379-tile UNI cache, anything `np.stack`ing per-tile arrays.
- Pre-launch math: `rows × features × dtype-bytes × (n_jobs + 1)`. If > 24 GB → subsample or shrink `n_jobs`.

## Figure / Visualization Guidance

Paper figures follow `vis_guidance.md` (compact layout, no text overlaps, consistent marker shapes, Nimbus Sans, 4 black spines, legend below). Exemplar: `src/a4_uni_probe/figures.py::render_pngs_updated_probe_delta`.
