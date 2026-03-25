"""
config_0epoch_verify.py

Zero-epoch training config for stage2 model-loading verification.

Inherits from config_controlnet_exp.py — loads the same pretrained base model
and ControlNet, builds a fresh MultiGroupTMEModule (zero-init), then immediately
saves a checkpoint and exits.  The saved checkpoint preserves the original
pretrained weights unchanged (no gradient steps taken), which means running
stage3_inference.py from this checkpoint should produce output identical to
verify_pretrained_inference.py given the same mask and UNI embedding.

Usage:
    python stage2_train.py configs/config_0epoch_verify.py

Then:
    python stage3_inference.py \\
        --config configs/config_controlnet_exp.py \\
        --checkpoint-dir checkpoints/pixcell_controlnet_0epoch_verify/checkpoints/step_0000000 \\
        --sim-channels-dir inference_data/sample \\
        --sim-id test_mask \\
        --reference-uni inference_data/sample/test_control_image_uni.npy \\
        --output inference_data/results/stage2_verify_output.png
"""

_base_ = ['./config_controlnet_exp.py']

# ── Zero-epoch: just load + save checkpoint, no training steps ────────────────
num_epochs = 0

# ── Dummy dataset path (created by tools/verify_stage2_model_loading.py) ──────
exp_data_root = "./inference_data/dummy_exp_train_data"

# ── Separate work_dir so we don't overwrite real training checkpoints ──────────
work_dir = "./checkpoints/pixcell_controlnet_0epoch_verify"
