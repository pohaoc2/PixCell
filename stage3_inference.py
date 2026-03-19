"""
Stage 3: Inference
==================
Generate experimental-like H&E images from simulation channel inputs.

The trained model (Stage 2) learns sim/exp channel → H&E from PAIRED experimental
data. At inference, we pass UNPAIRED simulation channels to generate realistic H&E
images matching the simulated TME layout.

Two inference modes:

    Style-conditioned (recommended when a reference tissue is available):
        Pass a reference H&E image (--reference-he). Its UNI-2h embedding guides
        the global tissue appearance (staining style, cell density, collagen patterns).

    TME-only (when no reference is available):
        Omit --reference-he. A null UNI embedding is used, relying entirely on the
        TME channel layout for generation. Requires CFG dropout during training
        (cfg_dropout_prob > 0 in config, default 0.15).

Usage:
    # Style-conditioned generation
    python stage3_inference.py \\
        --config           configs/config_controlnet_exp.py \\
        --checkpoint-dir   checkpoints/pixcell_controlnet_exp/checkpoints/step_XXXXXXX \\
        --sim-channels-dir /path/to/sim_channels \\
        --sim-id           my_simulation_001 \\
        --reference-he     /path/to/reference.png \\
        --output           generated_he.png

    # TME-only generation (no reference H&E)
    python stage3_inference.py \\
        --config           configs/config_controlnet_exp.py \\
        --checkpoint-dir   checkpoints/pixcell_controlnet_exp/checkpoints/step_XXXXXXX \\
        --sim-channels-dir /path/to/sim_channels \\
        --sim-id           my_simulation_001 \\
        --output           generated_he.png

    # Batch: generate for all sims in a directory
    python stage3_inference.py \\
        --config           configs/config_controlnet_exp.py \\
        --checkpoint-dir   checkpoints/pixcell_controlnet_exp/checkpoints/step_XXXXXXX \\
        --sim-channels-dir /path/to/sim_channels \\
        --output-dir       ./inference_output \\
        --n-tiles          50

Simulation channel directory layout:
    sim_channels/
    ├── cell_mask/        {sim_id}.png     binary cell segmentation (required)
    ├── cell_type_healthy/{sim_id}.png     (optional, exp-compatible channels)
    ├── cell_type_cancer/ {sim_id}.png
    ├── cell_type_immune/ {sim_id}.png
    ├── cell_state_prolif/{sim_id}.png
    ├── cell_state_nonprolif/{sim_id}.png
    ├── cell_state_dead/  {sim_id}.png
    ├── oxygen/           {sim_id}.png/.npy
    ├── glucose/          {sim_id}.png/.npy
    └── vasculature/      {sim_id}.png     (optional)

Channel availability is controlled by --active-channels (default: all 10 exp channels).
Missing optional channels are silently skipped.
"""
import argparse
import os
from pathlib import Path

import numpy as np
import torch
from diffusers import DDPMScheduler
from PIL import Image

from diffusion.model.builder import build_model
from diffusion.utils.misc import read_config
from diffusion.data.datasets.sim_controlnet_dataset import _find_file, _load_spatial_file, _BINARY_CHANNELS
from diffusion.data.datasets.paired_exp_controlnet_dataset import _BINARY_CHANNELS as EXP_BINARY
from train_scripts.inference_controlnet import (
    load_vae,
    null_uni_embed,
    denoise,
    load_controlnet_model_from_checkpoint,
    load_pixcell_controlnet_model_from_checkpoint,
)
from train_scripts.training_utils import load_tme_checkpoint

_ALL_BINARY = _BINARY_CHANNELS | EXP_BINARY


# ── Channel loading ───────────────────────────────────────────────────────────

def load_sim_channels(sim_channels_dir: Path, sim_id: str,
                      active_channels: list[str], resolution: int) -> torch.Tensor:
    """
    Load simulation channel images for a single snapshot → [C, H, W] tensor.

    Args:
        sim_channels_dir: Root directory containing per-channel subdirectories.
        sim_id:           Simulation snapshot identifier (file stem).
        active_channels:  Ordered channel list (cell_mask must be first).
        resolution:       Target spatial resolution.

    Returns:
        Float32 tensor [C, resolution, resolution].
    """
    planes = []
    for ch in active_channels:
        ch_dir = sim_channels_dir / ch
        fpath  = _find_file(ch_dir, sim_id)
        arr    = _load_spatial_file(fpath, resolution=resolution,
                                    binary=(ch in _ALL_BINARY))
        planes.append(arr)
    return torch.from_numpy(np.stack(planes, axis=0))


# ── Model loading ─────────────────────────────────────────────────────────────

def load_models(config, checkpoint_dir: str, device: str):
    """
    Load VAE, ControlNet, base model, and TME module from a training checkpoint.

    Args:
        config:         Training config (read_config output).
        checkpoint_dir: Path to a step checkpoint directory
                        (contains controlnet_*.pth and tme_module.pth).
        device:         'cuda' or 'cpu'.

    Returns:
        dict: vae, controlnet, base_model, tme_module
    """
    import glob as _glob
    config_path = config._filename if hasattr(config, '_filename') else None
    checkpoint_dir = str(checkpoint_dir)

    vae = load_vae(config.vae_pretrained, device)

    n_tme_channels = len(config.data.active_channels) - 1
    tme_module = build_model(
        "TMEConditioningModule", False, False,
        n_tme_channels=n_tme_channels,
        base_ch=getattr(config, "tme_base_ch", 32),
    )
    load_tme_checkpoint(checkpoint_dir, tme_module, device=device)
    _dtype = torch.float16 if device == 'cuda' else torch.float32
    tme_module.to(device=device, dtype=_dtype).eval()

    # Find the controlnet .pth file inside the checkpoint directory
    controlnet_pths = sorted(_glob.glob(
        os.path.join(checkpoint_dir, "controlnet_*.pth")
    ))
    if not controlnet_pths:
        raise FileNotFoundError(
            f"No controlnet_*.pth found in {checkpoint_dir}"
        )
    controlnet_pth = controlnet_pths[-1]   # take latest if multiple
    print(f"Loading controlnet from {controlnet_pth}")
    controlnet = load_controlnet_model_from_checkpoint(
        config_path, controlnet_pth, device
    )

    # Base model is always the frozen pretrained transformer (never saved to checkpoint)
    base_model_path = getattr(config, "load_from", config.base_model_path)
    # load_from may point to a directory — resolve to the safetensors file inside it
    if os.path.isdir(base_model_path):
        candidates = (
            _glob.glob(os.path.join(base_model_path, "*.safetensors")) +
            _glob.glob(os.path.join(base_model_path, "*.pth"))
        )
        if not candidates:
            raise FileNotFoundError(
                f"No .safetensors or .pth found in base_model_path={base_model_path}"
            )
        base_model_path = sorted(candidates)[0]
    print(f"Loading base model from {base_model_path}")
    base_model = load_pixcell_controlnet_model_from_checkpoint(
        config_path, base_model_path
    )
    base_model.to(device).eval()

    return dict(vae=vae, controlnet=controlnet, base_model=base_model,
                tme_module=tme_module)


# ── Single-tile generation ────────────────────────────────────────────────────

def generate(
    sim_channels_dir: Path,
    sim_id: str,
    models: dict,
    config,
    uni_embeds: torch.Tensor,
    scheduler,
    guidance_scale: float,
    device: str,
) -> np.ndarray:
    """
    Generate a single experimental-like H&E image from simulation channels.

    Args:
        sim_channels_dir: Directory with per-channel subdirectories.
        sim_id:           Simulation snapshot ID (file stem).
        models:           Dict from load_models().
        config:           Training config.
        uni_embeds:       UNI embedding [1,1,1,1536] — null or reference style.
        scheduler:        Configured DDPMScheduler.
        guidance_scale:   CFG guidance scale (higher = more TME adherence).
        device:           'cuda' or 'cpu'.

    Returns:
        RGB image as uint8 numpy array [H, W, 3].
    """
    vae        = models['vae']
    controlnet = models['controlnet']
    base_model = models['base_model']
    tme_module = models['tme_module']

    active_channels = config.data.active_channels
    vae_scale = config.scale_factor
    vae_shift = config.shift_factor
    dtype = torch.float16 if device == 'cuda' else torch.float32
    vae.to(device=device, dtype=dtype).eval()
    # 1. Load sim TME channels [C, H, W]
    ctrl_full = load_sim_channels(
        sim_channels_dir, sim_id, active_channels, resolution=config.image_size
    )

    # 2. VAE-encode cell_mask (channel 0) → conditioning latent
    cell_mask_img = ctrl_full[0:1].unsqueeze(0).repeat(1, 3, 1, 1)  # [1, 3, H, W]
    cell_mask_img = 2 * (cell_mask_img - 0.5)
    with torch.no_grad():
        vae_mask = vae.encode(
            cell_mask_img.to(device, dtype=dtype)
        ).latent_dist.mean
        vae_mask = (vae_mask - vae_shift) * vae_scale

    # 3. Fuse TME channels through TMEConditioningModule
    tme_channels = ctrl_full[1:].unsqueeze(0).to(device, dtype=dtype)
    with torch.no_grad():
        fused_cond = tme_module(vae_mask.to(dtype), tme_channels)

    # 4. Diffusion denoising
    latent_shape = (1, 16, config.image_size // 8, config.image_size // 8)
    latents = torch.randn(latent_shape, device=device, dtype=dtype)
    latents = latents * scheduler.init_noise_sigma
    denoised = denoise(
        latents=latents,
        uni_embeds=uni_embeds.to(device, dtype=dtype),
        controlnet_input_latent=fused_cond,
        scheduler=scheduler,
        controlnet_model=controlnet,
        pixcell_controlnet_model=base_model,
        guidance_scale=guidance_scale,
        device=device,
    )

    # 5. Decode latents → RGB image
    with torch.no_grad():
        scaled_latents = (denoised / vae_scale) + vae_shift
        gen_img = vae.decode(scaled_latents, return_dict=False)[0]
    gen_img = (gen_img / 2 + 0.5).clamp(0, 1)
    return (gen_img.cpu().permute(0, 2, 3, 1).numpy()[0] * 255).astype(np.uint8)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Stage 3: Generate experimental H&E from simulation channels"
    )
    parser.add_argument("--config",           required=True,
                        help="Training config path (configs/config_controlnet_exp.py)")
    parser.add_argument("--checkpoint-dir",   required=True,
                        help="Stage 2 checkpoint directory (contains controlnet_*.pth + tme_module.pth)")
    parser.add_argument("--sim-channels-dir", required=True,
                        help="Root directory with per-channel subdirectories")

    # Single-tile mode
    parser.add_argument("--sim-id",           default=None,
                        help="Simulation ID to generate (file stem, e.g. 'sim_0001')")
    parser.add_argument("--output",           default=None,
                        help="Output PNG path for single-tile mode")

    # Batch mode
    parser.add_argument("--output-dir",       default=None,
                        help="Output directory for batch generation")
    parser.add_argument("--n-tiles",          type=int,   default=None,
                        help="Max number of tiles to generate in batch mode")

    # Style conditioning
    parser.add_argument("--reference-he",     default=None,
                        help="Reference H&E image (PNG) for style conditioning. "
                             "If omitted, uses null UNI embedding (TME-only mode).")
    parser.add_argument("--reference-uni",    default=None,
                        help="Precomputed reference UNI embedding (.npy). "
                             "Alternative to --reference-he (skips UNI extraction).")

    # Generation settings
    parser.add_argument("--guidance-scale",   type=float, default=2.5)
    parser.add_argument("--num-steps",        type=int,   default=20)
    parser.add_argument("--device",           default="cuda")
    args = parser.parse_args()

    # ── Validate args ─────────────────────────────────────────────────────────
    if args.sim_id is None and args.output_dir is None:
        parser.error("Provide --sim-id (single tile) or --output-dir (batch).")
    if args.sim_id is not None and args.output is None:
        parser.error("--sim-id requires --output.")

    config = read_config(args.config)
    # Attach config filename for model loader
    config._filename = args.config
    device = args.device

    # ── Load models ───────────────────────────────────────────────────────────
    print("Loading models...")
    models = load_models(config, args.checkpoint_dir, device)

    # ── UNI embedding ─────────────────────────────────────────────────────────
    if args.reference_uni:
        ref = np.load(args.reference_uni)
        uni_embeds = torch.from_numpy(ref).view(1, 1, 1, 1536)
        print(f"Style conditioning: loaded UNI embedding from {args.reference_uni}")
    elif args.reference_he:
        from pipeline.extract_features import UNI2hExtractor
        uni_model_path = getattr(config, "uni_model_path",
                                 "./pretrained_models/uni-2h")
        extractor = UNI2hExtractor(model_path=uni_model_path, device=device)
        import cv2
        img = cv2.cvtColor(cv2.imread(args.reference_he), cv2.COLOR_BGR2RGB)
        feat = extractor.extract(img)
        uni_embeds = torch.from_numpy(feat).view(1, 1, 1, 1536)
        print(f"Style conditioning: extracted UNI from {args.reference_he}")
    else:
        uni_embeds = null_uni_embed(device='cpu', dtype=torch.float32)
        print("TME-only mode: using null UNI embedding")

    # ── Scheduler ─────────────────────────────────────────────────────────────
    scheduler = DDPMScheduler(
        num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02,
        beta_schedule="linear", prediction_type="epsilon", clip_sample=False,
    )
    scheduler.set_timesteps(args.num_steps, device=device)

    sim_channels_dir = Path(args.sim_channels_dir)

    # ── Single-tile mode ──────────────────────────────────────────────────────
    if args.sim_id is not None:
        print(f"Generating H&E for sim_id='{args.sim_id}'...")
        img = generate(
            sim_channels_dir=sim_channels_dir,
            sim_id=args.sim_id,
            models=models,
            config=config,
            uni_embeds=uni_embeds,
            scheduler=scheduler,
            guidance_scale=args.guidance_scale,
            device=device,
        )
        Image.fromarray(img).save(args.output)
        print(f"Saved → {args.output}")
        return

    # ── Batch mode ────────────────────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cell_mask_dir = sim_channels_dir / "cell_mask"
    sim_ids = sorted(p.stem for p in cell_mask_dir.glob("*.png"))
    if args.n_tiles:
        sim_ids = sim_ids[: args.n_tiles]

    if not sim_ids:
        raise RuntimeError(f"No sim channel PNGs found in {cell_mask_dir}")

    print(f"Batch generating {len(sim_ids)} tiles → {output_dir}")
    for sim_id in sim_ids:
        img = generate(
            sim_channels_dir=sim_channels_dir,
            sim_id=sim_id,
            models=models,
            config=config,
            uni_embeds=uni_embeds,
            scheduler=scheduler,
            guidance_scale=args.guidance_scale,
            device=device,
        )
        Image.fromarray(img).save(output_dir / f"{sim_id}_he.png")
        print(f"  {sim_id} → {output_dir / (sim_id + '_he.png')}")

    print(f"\nDone. Generated {len(sim_ids)} images → {output_dir}")


if __name__ == "__main__":
    main()
