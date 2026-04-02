"""
Verify pretrained PixCell model + ControlNet loading and inference.

Loads both the base transformer and ControlNet from safetensors using the
custom PixCellControlNet architecture with weight remapping, then runs
a single denoising pass on a test cell mask to confirm correctness.
"""
import argparse
import os
import sys
import torch
import numpy as np
from PIL import Image
from diffusers import DDPMScheduler

sys.path.insert(0, os.path.dirname(__file__))

from diffusion.model.builder import build_model
from diffusion.utils.misc import read_config
from diffusion.utils.checkpoint import load_checkpoint
from pipeline.extract_features import UNI2hExtractor
from train_scripts.inference_controlnet import load_vae, denoise
from train_scripts.mapping_weights_helper import map_sd_to_controlnet
from tools.pretrained_verify.cached_inference_features import load_or_compute_npy
from tools.pretrained_verify.visualize_pretrained_inference import save_comparison_figure


def load_base_model(config_path, safetensors_path, device):
    """Load the frozen PixArt base transformer from safetensors."""
    config = read_config(config_path)
    kv_compress_config = config.kv_compress_config if config.kv_compress else None
    latent_size = int(config.image_size) // 8

    base_model = build_model(
        config.base_model,
        False,
        config.get('fp32_attention', False),
        input_size=latent_size,
        learn_sigma=True,
        pred_sigma=True,
        pe_interpolation=config.pe_interpolation,
        config=config,
        model_max_length=config.model_max_length,
        qk_norm=config.qk_norm,
        kv_compress_config=kv_compress_config,
        micro_condition=config.micro_condition,
        **config.get('base_model_kwargs', {})
    )

    missing, unexpected = load_checkpoint(safetensors_path, model=base_model)
    print(f"\n[Base model] Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    if missing:
        print(f"  Missing examples: {missing[:5]}")
    if unexpected:
        print(f"  Unexpected examples: {unexpected[:5]}")

    base_model.to(device).eval()
    return base_model


def load_controlnet(config_path, safetensors_path, device):
    """Load the ControlNet from safetensors using key remapping."""
    config = read_config(config_path)
    kv_compress_config = config.kv_compress_config if config.kv_compress else None
    latent_size = int(config.image_size) // 8

    controlnet = build_model(
        config.controlnet_model,
        config.grad_checkpointing,
        config.get('fp32_attention', False),
        input_size=latent_size,
        learn_sigma=False,
        pred_sigma=False,
        pe_interpolation=config.pe_interpolation,
        config=None,
        model_max_length=config.model_max_length,
        qk_norm=config.qk_norm,
        kv_compress_config=kv_compress_config,
        conditioning_channels=config.controlnet_conditioning_channels,
        n_controlnet_blocks=getattr(config, 'n_controlnet_blocks', None),
        **config.get('controlnet_model_kwargs', {})
    )

    mapped = map_sd_to_controlnet(safetensors_path)
    missing, unexpected = controlnet.load_state_dict(mapped, strict=False)

    print(f"\n[ControlNet] load_state_dict results:")
    print(f"  Mapped keys:    {len(mapped)}")
    print(f"  Missing (rand): {len(missing)}")
    print(f"  Unexpected:     {len(unexpected)}")
    if missing:
        print(f"  Missing keys:")
        for k in missing:
            print(f"    {k}")
    if unexpected:
        print(f"  Unexpected keys:")
        for k in unexpected:
            print(f"    {k}")

    cn_w0 = controlnet.controlnet_blocks[0].weight
    print(f"\n  controlnet_blocks[0] weight max: {cn_w0.abs().max():.6f} "
          f"(should be non-zero if loaded correctly)")

    controlnet.to(device).eval()
    return controlnet


def encode_mask_as_conditioning(mask_path, vae, device, resolution=256):
    """VAE-encode a cell mask image to 16-channel latent conditioning."""
    mask_img = np.array(Image.open(mask_path).convert("RGB").resize(
        (resolution, resolution), Image.NEAREST
    ))
    mask_tensor = torch.from_numpy(mask_img.copy() / 255.0).float()
    mask_tensor = mask_tensor.permute(2, 0, 1).unsqueeze(0)
    mask_tensor = 2 * (mask_tensor - 0.5)

    dtype = next(vae.parameters()).dtype
    with torch.no_grad():
        latent = vae.encode(mask_tensor.to(device, dtype=dtype)).latent_dist.mean

    vae_scale = vae.config.scaling_factor
    vae_shift = getattr(vae.config, "shift_factor", 0)
    return (latent - vae_shift) * vae_scale


def load_or_cache_reference_uni(reference_he_path, reference_uni_path, uni_model_path, device):
    """Load a cached UNI embedding or extract it from the reference H&E image."""
    cache_exists = os.path.exists(reference_uni_path)
    status = "Loading cached UNI" if cache_exists else "Extracting UNI and saving cache"
    print(f"\n--- {status} ---")
    print(f"  Reference H&E: {reference_he_path}")
    print(f"  UNI cache:     {reference_uni_path}")

    extractor = None

    def compute_uni():
        nonlocal extractor
        if extractor is None:
            extractor = UNI2hExtractor(model_path=uni_model_path, device=device)
        image = Image.open(reference_he_path).convert("RGB")
        return extractor.extract(image).astype(np.float32)

    uni_np = load_or_compute_npy(reference_uni_path, compute_uni)
    uni_embeds = torch.from_numpy(uni_np).view(1, 1, 1, 1536)
    print(f"  UNI shape: {tuple(uni_embeds.shape)}")
    print(f"  UNI stats: mean={uni_embeds.mean().item():.4f}, std={uni_embeds.std().item():.4f}")
    return uni_embeds


def load_or_cache_mask_latent(mask_path, mask_latent_path, vae, device, resolution=256):
    """Load a cached VAE mask latent or encode it from the mask image."""
    cache_exists = os.path.exists(mask_latent_path)
    status = "Loading cached mask latent" if cache_exists else "Encoding mask latent and saving cache"
    print(f"\n--- {status} ---")
    print(f"  Mask image:  {mask_path}")
    print(f"  VAE cache:   {mask_latent_path}")

    def compute_mask_latent():
        latent = encode_mask_as_conditioning(mask_path, vae, device, resolution=resolution)
        return latent.squeeze(0).detach().cpu().float().numpy()

    mask_latent_np = load_or_compute_npy(mask_latent_path, compute_mask_latent)
    mask_latent = torch.from_numpy(mask_latent_np).unsqueeze(0)
    print(f"  Mask latent shape: {tuple(mask_latent.shape)}")
    print(f"  Mask latent stats: mean={mask_latent.mean().item():.4f}, std={mask_latent.std().item():.4f}")
    return mask_latent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Verify pretrained PixCell transformer + ControlNet loading."
    )
    parser.add_argument(
        "--config",
        default="configs/config_controlnet_exp.py",
        help="Config used to build the base model and ControlNet.",
    )
    parser.add_argument(
        "--base-safetensors",
        default="pretrained_models/pixcell-256/transformer/diffusion_pytorch_model.safetensors",
        help="Path to pretrained PixCell transformer safetensors.",
    )
    parser.add_argument(
        "--controlnet-safetensors",
        default="pretrained_models/pixcell-256-controlnet/controlnet/diffusion_pytorch_model.safetensors",
        help="Path to pretrained PixCell ControlNet safetensors.",
    )
    parser.add_argument(
        "--vae-path",
        default="pretrained_models/sd-3.5-vae/vae",
        help="Path to the VAE folder.",
    )
    parser.add_argument(
        "--mask-path",
        default="inference_data/test_mask.png",
        help="Mask image used for the verification run.",
    )
    parser.add_argument(
        "--output-path",
        default="inference_data/vis_pretrained_verification_test_mask.png",
        help="Where to save the comparison figure.",
    )
    parser.add_argument(
        "--generated-output",
        default="inference_data/generated_he_pretrained_test_mask.png",
        help="Where to save the generated H&E image.",
    )
    parser.add_argument(
        "--reference-he",
        default="inference_data/test_control_image.png",
        help="Reference H&E image used for UNI style conditioning and shown in ax[0].",
    )
    parser.add_argument(
        "--reference-uni",
        default="inference_data/test_control_image_uni.npy",
        help="Cached UNI embedding path for the reference H&E image.",
    )
    parser.add_argument(
        "--mask-latent",
        default="inference_data/test_mask_sd3_vae.npy",
        help="Cached VAE latent path for the mask image.",
    )
    parser.add_argument(
        "--uni-model-path",
        default="pretrained_models/uni-2h",
        help="Path to the UNI-2h model directory.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = args.config
    base_safetensors = args.base_safetensors
    cn_safetensors = args.controlnet_safetensors
    vae_path = args.vae_path
    mask_path = args.mask_path
    output_path = args.output_path
    generated_output = args.generated_output
    reference_he_path = args.reference_he
    reference_uni_path = args.reference_uni
    mask_latent_path = args.mask_latent
    uni_model_path = args.uni_model_path

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    seed = 42

    print("=" * 60)
    print("PixCell Pretrained Inference Verification")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Config: {config_path}")
    print(f"Base model: {base_safetensors}")
    print(f"ControlNet: {cn_safetensors}")
    print(f"Mask: {mask_path}")
    print(f"Reference H&E: {reference_he_path}")

    # 1. Load models
    print("\n--- Loading VAE ---")
    vae = load_vae(vae_path, device)
    vae.to(dtype=dtype)

    print("\n--- Loading Base Model (PixArt_XL_2_UNI) ---")
    base_model = load_base_model(config_path, base_safetensors, device)

    print("\n--- Loading ControlNet (PixCellControlNet, mapped weights) ---")
    controlnet = load_controlnet(config_path, cn_safetensors, device)

    # 2. Prepare cached conditioning inputs
    uni_embeds = load_or_cache_reference_uni(
        reference_he_path=reference_he_path,
        reference_uni_path=reference_uni_path,
        uni_model_path=uni_model_path,
        device=device,
    )
    cond_latent = load_or_cache_mask_latent(
        mask_path=mask_path,
        mask_latent_path=mask_latent_path,
        vae=vae,
        device=device,
        resolution=256,
    )

    # 3. Set up scheduler and run denoising
    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
        prediction_type="epsilon",
        clip_sample=False,
    )

    config = read_config(config_path)
    latent_size = config.image_size // 8
    latent_shape = (1, 16, latent_size, latent_size)

    torch.manual_seed(seed)
    latents = torch.randn(latent_shape, device=device, dtype=dtype)
    latents = latents * scheduler.init_noise_sigma

    print(f"\n--- Running denoising (20 steps, guidance=2.5) ---")
    denoised = denoise(
        latents=latents,
        uni_embeds=uni_embeds,
        controlnet_input_latent=cond_latent,
        scheduler=scheduler,
        controlnet_model=controlnet,
        pixcell_controlnet_model=base_model,
        guidance_scale=2.5,
        num_inference_steps=20,
        conditioning_scale=1.0,
        device=device,
    )

    # 4. Decode latents
    print("\n--- Decoding latents ---")
    vae_scale = vae.config.scaling_factor
    vae_shift = getattr(vae.config, "shift_factor", 0)

    vae_dtype = next(vae.parameters()).dtype
    with torch.no_grad():
        decoded = vae.decode(
            ((denoised / vae_scale) + vae_shift).to(dtype=vae_dtype),
            return_dict=False
        )[0]
    decoded = (decoded / 2 + 0.5).clamp(0, 1)
    gen_img = (decoded.cpu().permute(0, 2, 3, 1).numpy()[0] * 255).astype(np.uint8)

    Image.fromarray(gen_img).save(generated_output)
    print(f"  Image shape: {gen_img.shape}, range: [{gen_img.min()}, {gen_img.max()}]")
    print(f"  Generated H&E saved to: {generated_output}")

    # 5. Create comparison figure
    print("\n--- Creating visualization ---")
    save_comparison_figure(
        mask_path=mask_path,
        gen_img=gen_img,
        save_path=output_path,
        reference_he_path=reference_he_path,
    )

    print("\n" + "=" * 60)
    print("Verification complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
