# PixCell Codebase Cleanup — Design Spec

**Date:** 2026-03-28
**Approach:** A (Minimal/Targeted)

---

## 1. Dead Code Removal — `train_scripts/inference_controlnet.py`

Delete 9 functions with zero callers outside the file. All have been superseded by `_from_checkpoint` variants or were one-off debug helpers.

| Function | Reason |
|---|---|
| `load_base_model_checkpoint` | Replaced by `load_pixcell_controlnet_model_from_checkpoint` |
| `load_controlnet_model` | Replaced by `load_controlnet_model_from_checkpoint` |
| `load_pixcell_controlnet_model` | Same — `_from_checkpoint` version used everywhere |
| `initialize_pixcell_controlnet_model` | Unused setup helper |
| `initialize_controlnet_model` | Unused setup helper |
| `save_keys_comparison_controlnet` | Debug helper, never called outside file |
| `test_load_controlnet` | Debug helper, never called outside file |
| `prepare_controlnet_input` | Hardcoded index-based helper from old sim path |
| `decode_latents` | Unused decode helper (decoding is inline everywhere) |

**Constraint:** Do not touch any function that is imported by another module. Confirmed callers of remaining functions: `null_uni_embed`, `encode_ctrl_mask_latent`, `load_vae`, `denoise`, `load_controlnet_model_from_checkpoint`, `load_pixcell_controlnet_model_from_checkpoint`, `load_controlnet_weights_flexible`, `load_model_weights_flexible`.

---

## 2. Batch Script Cleanup

### 2a. Delete `run_zero_out_mask_batch.py`

`tools/generate_orion_paired_unpaired_batch.py` already covers paired+unpaired batch generation with `--n-tiles`, `--checkpoint-dir`, `--data-root`, and `--output-dir`. The only thing `run_zero_out_mask_batch.py` adds is hardcoded paths to a specific checkpoint — not reusable logic.

### 2b. Move + refactor `run_stage3_full.py` → `tools/run_evaluation.py`

Unique logic to preserve:
- Separate inference + validation tile sets (no overlap)
- Per-tile UNI cosine-similarity metric vs ground-truth H&E features
- Summary metrics JSON written to `--output-dir`

Hardcoded constants become CLI flags:

```
python tools/run_evaluation.py \
    --config           configs/config_controlnet_exp.py \
    --checkpoint-dir   checkpoints/pixcell_controlnet_exp/checkpoints \
    --data-root        data/orion-crc33 \
    --output-dir       inference_output/evaluation \
    [--n-inference  20] \
    [--n-validation 20] \
    [--n-vis-tiles   5] \
    [--seed         42] \
    [--guidance-scale 2.5] \
    [--num-steps    20] \
    [--device       cuda]
```

The internal structure of the script (tile loop, paired/unpaired modes, figure generation) is preserved as-is; only the path constants are lifted to argparse.

---

## 3. New Tests

### 3a. `tests/test_inference_core.py`

Covers `train_scripts/inference_controlnet.py` and `stage3_inference.py`.

| Test | What it checks |
|---|---|
| `test_load_sim_channels_binary_thresholding` | `cell_mask` channel is {0,1}; continuous channels are not clamped |
| `test_encode_ctrl_mask_latent_shape` | Output shape is `[1, 16, H/8, W/8]` |
| `test_generate_zero_mask_latent_applied` | With `zero_mask_latent=True`: `fused = tme_out - vae_mask` (not equal to `tme_out`) |
| `test_generate_zero_mask_latent_off` | With `zero_mask_latent=False`: fused equals raw TME output |

### 3b. `tests/test_stage3_tile_pipeline.py`

Covers `tools/stage3_tile_pipeline.py`.

| Test | What it checks |
|---|---|
| `test_load_channel_binary` | PNG → float [0,1]; binary channel values snap to {0,1} |
| `test_load_channel_reflect_pad` | Non-binary channels are reflect-padded before resize |
| `test_resolve_data_layout_orion` | Directory with `exp_channels/` resolves all three paths correctly |
| `test_generate_tile_zero_mask_latent` | `zero_mask_latent=True` produces `fused - vae_mask`, not raw fused |

**Constraints for all new tests:** Small tensors (16×16 or 32×32), mock/stub VAE and model forward passes, no GPU required, no disk I/O beyond `tmp_path` fixtures.

---

## Out of Scope

- Merging `load_models` / `generate` duplication between `stage3_inference.py` and `stage3_tile_pipeline.py` (Approach B)
- Structural reorganization of `tools/` (Approach C)
- Any changes to `diffusion/`, `configs/`, or existing tests
