"""
Stage 1: Feature Extraction
============================
Extract UNI-2h embeddings and SD 3.5 VAE latents from paired experimental H&E images.

These precomputed features are required by Stage 2 (training) to avoid re-encoding
images on every training step. Run this once on your paired experimental dataset
(H&E tiles + cell_mask images).

Output per image:
    {stem}_uni.npy       — UNI-2h embedding [1536]  (histopathology style features)
    {stem}_sd3_vae.npy   — SD3.5 VAE latent  [2, 16, H/8, W/8]  (mean + std)

Two passes are required — one for H&E images, one for cell_mask images:

    # Pass 1: H&E → UNI embeddings + VAE latents
    python stage1_extract_features.py \\
        --image-dir  ./data/exp_paired/he_images \\
        --output-dir ./data/exp_paired/features

    # Pass 2: cell_mask → VAE latents (used for ControlNet conditioning)
    python stage1_extract_features.py \\
        --image-dir  ./data/exp_paired/exp_channels/cell_mask \\
        --output-dir ./data/exp_paired/vae_features \\
        --vae-prefix mask_sd3_vae \\
        --skip-uni

After both passes, proceed to:
    python stage2_train.py configs/config_controlnet_exp.py

Requirements:
    Stage 0 (model setup) must be complete.
    export HF_TOKEN=<your_huggingface_token>
"""
from pipeline.extract_features import main

if __name__ == "__main__":
    main()
