# PixCell Quick Context

- Goal: train a diffusion ControlNet that maps spatial TME channels to realistic H&E patches.
- Training uses paired ORION-CRC experimental tiles (H&E + CODEX-derived channels); inference uses unpaired simulation channels.
- Current primary path is paired experimental ControlNet + multi-group TME; older sim-only code still exists but is not the default workflow.

## Pipeline

1. `stage0_setup.py`: download PixCell-256, PixCell ControlNet init, UNI-2h, and SD3.5 VAE into `pretrained_models/`.
2. `stage1_extract_features.py`: cache UNI embeddings and SD3.5 VAE latents for H&E and cell-mask images.
3. `stage2_train.py`: train ControlNet + TME module using `configs/config_controlnet_exp.py`.
4. `stage3_inference.py`: generate H&E from exp channels, optionally with a reference H&E for style conditioning.

## Model

- Frozen pieces: base PixCell-256 transformer, SD3.5 VAE, and UNI-2h encoder.
- Trainable pieces: ControlNet + `MultiGroupTMEModule` in `diffusion/model/nets/multi_group_tme.py`.
- Channel groups: `cell_identity` (healthy/cancer/immune), `cell_state` (prolif/nonprolif/dead), `vasculature`, `microenv` (oxygen/glucose).
- Each group has its own CNN encoder + cross-attention; outputs are zero-init additive residuals.
- `zero_mask_latent=True` (post-TME): TME uses real mask latent for spatial Q, then subtracts — `fused = tme(vae_mask) - vae_mask`. Closes bypass path, preserves spatial structure. Must be applied post-TME in train, inference, and all pipeline helpers.
- `cfg_dropout_prob=0.15` zeros UNI embeddings during training, enabling TME-only inference.
- Per-group dropout configured in `configs/config_controlnet_exp.py`.

## Data Contract

- Default paired dataset root: `data/orion-crc33`.
- Required subdirs: `exp_channels/`, `features/`, `vae_features/`, `metadata/exp_index.hdf5`.
- Canonical first channel is `cell_masks`; legacy alias `cell_mask` is still accepted in utilities/tests.
- Required binary channels: `cell_masks`, 3 cell-type maps, 3 cell-state maps.
- Optional/approximate channels: `vasculature`, `oxygen`, `glucose`.
- Missing `mask_sd3_vae` falls back to zeros; missing optional sim channels are skipped at inference.

## Important Files

- Config: `configs/config_controlnet_exp.py`
- Training loop: `train_scripts/train_controlnet_exp.py`
- Dataset: `diffusion/data/datasets/paired_exp_controlnet_dataset.py`
- Group helpers: `tools/channel_group_utils.py`
- Inference: `stage3_inference.py`; batch vis: `tools/run_evaluation.py`
- Visualization: `tools/stage3_figures.py` (figures), `tools/stage3_tile_pipeline.py` (inference helpers)
- Color palette: `tools/color_constants.py`
- Architecture notes: `MODEL_ARCHITECTURE.md`

## Tests

- `tests/test_paired_exp_dataset.py`: paired dataset contract + `cell_mask` alias.
- `tests/test_multi_group_tme.py`: shape, zero-init identity, active-group gating, residual/attention outputs.
- `tests/test_channel_group_utils.py`: group splitting + group dropout.
- `tests/test_train_controlnet_exp.py`: CFG dropout + channel-weighting tensor logic.

## Working Assumptions

- Prefer the paired-exp + multi-group path unless the task explicitly targets legacy sim code.
- Preserve `cell_masks`/`cell_mask` compatibility when touching datasets or inference.
- If changing inference/training group logic, update both code and tests.
- Read `README.md` only for full setup instructions; this file is the short agent handoff.
