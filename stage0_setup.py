"""
Stage 0: Model Setup
====================
Download all pretrained models required for the PixCell ControlNet pipeline:

    1. PixCell-256         — base diffusion transformer (frozen during training)
    2. PixCell-256-ControlNet — pretrained ControlNet initialization weights
    3. UNI-2h              — histopathology foundation model (feature extraction + style conditioning)
    4. SD 3.5 VAE          — image encoder/decoder for latent diffusion

After running this stage, proceed to:
    python stage1_extract_features.py --image-dir <he_images> --output-dir <features>

Usage:
    python stage0_setup.py [--save-dir ./pretrained_models] [--model all]

Requirements:
    export HF_TOKEN=<your_huggingface_token>
"""
from pipeline.setup_pretrained_model import download_all_models, download_pixcell_256, \
    download_pixcell_controlnet, download_uni_2h, download_sd3_vae

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Stage 0: Download pretrained models for PixCell ControlNet"
    )
    parser.add_argument("--save-dir", type=str, default="./pretrained_models",
                        help="Directory to save pretrained models (default: ./pretrained_models)")
    parser.add_argument("--model",
                        choices=["pixcell", "pixcell-controlnet", "uni2h", "sd3_vae", "all"],
                        default="all",
                        help="Which model to download (default: all)")
    parser.add_argument("--token", type=str, default=None,
                        help="HuggingFace token (defaults to $HF_TOKEN env variable)")
    args = parser.parse_args()

    if args.model == "all":
        download_all_models(args.save_dir, args.token)
    elif args.model == "pixcell":
        download_pixcell_256(args.save_dir, args.token)
    elif args.model == "pixcell-controlnet":
        download_pixcell_controlnet(args.save_dir, args.token)
    elif args.model == "uni2h":
        download_uni_2h(args.save_dir, args.token)
    elif args.model == "sd3_vae":
        download_sd3_vae(args.save_dir, args.token)
