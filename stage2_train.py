"""
Stage 2: Training
=================
Train PixCell ControlNet on PAIRED experimental H&E + multichannel TME data.

Goal: Learn the mapping from experimental multichannel images (CODEX-derived TME
channels: cell types, cell states, vasculature, metabolic fields) to the paired
H&E image in the same spatial location.

Architecture:
    - PixCell-256 transformer (FROZEN)
    - ControlNet (trainable, initialized from pretrained weights)
    - TMEConditioningModule (trainable, ~300K params)
      ResNet CNN encodes TME channels → cross-attention fuse with cell_mask VAE latent

Training features:
    - CFG dropout (--cfg-dropout-prob, default 0.15): randomly zero UNI embedding
      to enable TME-only inference at no extra cost
    - Channel reliability weighting: attenuate approximate channels (vasculature,
      oxygen, glucose) which have registration uncertainty

Inference after training:
    python stage3_inference.py \\
        --checkpoint-dir  checkpoints/pixcell_controlnet_exp/checkpoints/step_XXXXXXX \\
        --sim-channels-dir  /path/to/sim_channels \\
        --sim-id            my_simulation_001 \\
        --output            generated_he.png

Usage:
    # Single GPU
    python stage2_train.py configs/config_controlnet_exp.py

    # Multi-GPU (recommended)
    accelerate launch stage2_train.py configs/config_controlnet_exp.py

    # Resume from checkpoint
    accelerate launch stage2_train.py configs/config_controlnet_exp.py \\
        --resume-from checkpoints/pixcell_controlnet_exp/checkpoints/step_XXXXXXX

Requirements:
    Stage 0 (model setup) and Stage 1 (feature extraction) must be complete.
    Dataset path must be set in config: exp_data_root = "/path/to/exp_paired"
"""
from train_scripts.train_controlnet_exp import main

if __name__ == "__main__":
    main()
