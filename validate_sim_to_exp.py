"""
validate_sim_to_exp.py — Simulation-to-experiment validation pipeline.

For each simulation snapshot, generates an H&E tile in the experimental domain
using the trained PixCellControlNet, extracts UNI-2h features from the generated
image, and compares to precomputed experimental target UNI features.

Usage:
    python validate_sim_to_exp.py \\
        --config          configs/config_controlnet_exp.py \\
        --sim-root        /path/to/sim_data_root \\
        --exp-feat        /path/to/exp_features_dir \\
        --controlnet-ckpt /path/to/controlnet.pth \\
        --tme-ckpt        /path/to/tme_module.pth \\
        --uni-model       ./pretrained_models/uni-2h \\
        [--reference-uni  /path/to/reference_uni.npy]
        [--output-dir     ./validation_output]
        [--n-tiles        50]
        [--guidance-scale 2.5]
        [--device         cuda]
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import DDPMScheduler

from diffusion.utils.misc import read_config
from train_scripts.inference_controlnet import (
    load_vae,
    null_uni_embed,
    denoise,
    load_controlnet_model_from_checkpoint,
    load_pixcell_controlnet_model_from_checkpoint,
)
from diffusion.model.builder import build_model
from train_scripts.train_controlnet_sim import load_sim_checkpoint
from diffusion.data.datasets.sim_controlnet_dataset import _load_spatial_file, _find_file
from diffusion.data.datasets.paired_exp_controlnet_dataset import (
    _BINARY_CHANNELS as EXP_BINARY,
)
from extract_features import UNI2hExtractor


# ── Core metric ───────────────────────────────────────────────────────────────

def cosine_similarity_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Per-row cosine similarity between two [N, D] feature matrices.

    Returns:
        Tensor [N] of cosine similarities in [-1, 1].
    """
    a_norm = F.normalize(a, dim=1)
    b_norm = F.normalize(b, dim=1)
    return (a_norm * b_norm).sum(dim=1)


# ── Channel loader ────────────────────────────────────────────────────────────

def load_sim_ctrl_tensor(
    sim_root: Path,
    sim_id: str,
    active_channels: list[str],
    resolution: int = 256,
) -> torch.Tensor:
    """Load a single sim snapshot's TME channels → [C, H, W]."""
    from diffusion.data.datasets.sim_controlnet_dataset import _BINARY_CHANNELS as SIM_BINARY
    binary_set = SIM_BINARY | EXP_BINARY
    planes = []
    for ch in active_channels:
        ch_dir = sim_root / "sim_channels" / ch
        fpath  = _find_file(ch_dir, sim_id)
        arr    = _load_spatial_file(fpath, resolution=resolution, binary=(ch in binary_set))
        planes.append(arr)
    return torch.from_numpy(np.stack(planes, axis=0))


# ── Validation loop ───────────────────────────────────────────────────────────

def run_validation(
    config,
    sim_root: Path,
    sim_ids: list[str],
    exp_feat_dir: Path,
    controlnet,
    base_model,
    tme_module,
    vae,
    scheduler,
    uni_extractor: UNI2hExtractor,
    uni_embeds: torch.Tensor,      # [1,1,1,1536] — null for TME-only, or reference style
    guidance_scale: float,
    device: str,
    output_dir: Path | None,
) -> dict:
    """
    Generate H&E from sim TME channels, extract UNI features, compare to exp targets.

    Returns:
        dict with keys:
            cosine_similarities : list[float]  per-tile cosine similarity in UNI space
            mean_cosine_sim     : float
            std_cosine_sim      : float
    """
    active_channels = config.data.active_channels
    vae_scale = config.scale_factor
    vae_shift = config.shift_factor
    dtype = torch.float16 if device == 'cuda' else torch.float32

    cosine_sims = []
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    for sim_id in sim_ids:
        # 1. Load sim TME channels
        ctrl_full = load_sim_ctrl_tensor(
            sim_root, sim_id, active_channels, resolution=config.image_size
        )

        # 2. VAE-encode cell_mask (channel 0) for ControlNet conditioning
        cell_mask_img = ctrl_full[0:1].unsqueeze(0).repeat(1, 3, 1, 1)  # [1,3,H,W]
        cell_mask_img = 2 * (cell_mask_img - 0.5)
        with torch.no_grad():
            vae_mask = vae.encode(
                cell_mask_img.to(device, dtype=dtype)
            ).latent_dist.mean
            vae_mask = (vae_mask - vae_shift) * vae_scale

        # 3. Fuse TME channels (no weight attenuation at inference — sim channels are clean)
        tme_channels = ctrl_full[1:].unsqueeze(0).to(device, dtype=dtype)
        with torch.no_grad():
            fused_cond = tme_module(vae_mask.to(dtype), tme_channels)

        # 4. Denoise
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

        # 5. Decode to RGB image
        with torch.no_grad():
            gen_img = vae.decode(
                (denoised.float() / vae_scale) + vae_shift, return_dict=False
            )[0]
        gen_img = (gen_img / 2 + 0.5).clamp(0, 1)
        gen_np  = (gen_img.cpu().permute(0, 2, 3, 1).numpy()[0] * 255).astype(np.uint8)

        if output_dir:
            Image.fromarray(gen_np).save(output_dir / f"{sim_id}_generated.png")

        # 6. Extract UNI-2h features from generated H&E
        gen_feat_np = uni_extractor.extract(gen_np)               # [1536] float32
        gen_feat    = torch.from_numpy(gen_feat_np).unsqueeze(0)  # [1, 1536]

        # 7. Load precomputed exp target UNI features
        exp_feat_path = exp_feat_dir / f"{sim_id}_uni.npy"
        if not exp_feat_path.exists():
            print(f"[WARN] No exp feat for {sim_id}, skipping.")
            continue
        exp_feat = torch.from_numpy(np.load(exp_feat_path)).unsqueeze(0)  # [1, 1536]

        # 8. Cosine similarity in UNI feature space
        sim_val = cosine_similarity_matrix(gen_feat, exp_feat).item()
        cosine_sims.append(sim_val)
        print(f"  {sim_id}: cosine_sim={sim_val:.4f}")

    return {
        "cosine_similarities": cosine_sims,
        "mean_cosine_sim":     float(np.mean(cosine_sims)) if cosine_sims else float('nan'),
        "std_cosine_sim":      float(np.std(cosine_sims))  if cosine_sims else float('nan'),
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Sim-to-exp validation pipeline")
    parser.add_argument("--config",           required=True)
    parser.add_argument("--sim-root",         required=True)
    parser.add_argument("--exp-feat",         required=True,
                        help="Directory of *_uni.npy precomputed exp target features")
    parser.add_argument("--controlnet-ckpt",  required=True)
    parser.add_argument("--tme-ckpt",         required=True)
    parser.add_argument("--uni-model",        default="./pretrained_models/uni-2h",
                        help="Path to UNI-2h model directory")
    parser.add_argument("--reference-uni",    default=None,
                        help="Optional reference H&E UNI .npy for style-conditioned mode")
    parser.add_argument("--output-dir",       default=None,
                        help="Save generated H&E images here (optional)")
    parser.add_argument("--n-tiles",          type=int,   default=50)
    parser.add_argument("--guidance-scale",   type=float, default=2.5)
    parser.add_argument("--device",           default="cuda")
    args = parser.parse_args()

    config = read_config(args.config)
    device = args.device

    vae        = load_vae(config.vae_pretrained, device)
    controlnet = load_controlnet_model_from_checkpoint(args.config, args.controlnet_ckpt, device)
    base_model = load_pixcell_controlnet_model_from_checkpoint(args.config, args.controlnet_ckpt)
    base_model.to(device).eval()

    n_tme_channels = len(config.data.active_channels) - 1
    tme_module = build_model(
        "TMEConditioningModule", False, False,
        n_tme_channels=n_tme_channels,
        base_ch=getattr(config, "tme_base_ch", 32),
    )
    load_sim_checkpoint(args.tme_ckpt, tme_module, device=device)
    tme_module.to(device).eval()

    uni_extractor = UNI2hExtractor(model_path=args.uni_model, device=device)

    scheduler = DDPMScheduler(
        num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02,
        beta_schedule="linear", prediction_type="epsilon", clip_sample=False,
    )
    scheduler.set_timesteps(50, device=device)

    if args.reference_uni:
        ref = np.load(args.reference_uni)
        uni_embeds = torch.from_numpy(ref).view(1, 1, 1, 1536)
    else:
        uni_embeds = null_uni_embed(device=device, dtype=torch.float16)

    from diffusion.data.datasets.sim_controlnet_dataset import SimControlNetData
    ds      = SimControlNetData(root=args.sim_root, resolution=config.image_size,
                                active_channels=config.data.active_channels)
    sim_ids = ds.sim_ids[: args.n_tiles]

    results = run_validation(
        config=config,
        sim_root=Path(args.sim_root),
        sim_ids=sim_ids,
        exp_feat_dir=Path(args.exp_feat),
        controlnet=controlnet,
        base_model=base_model,
        tme_module=tme_module,
        vae=vae,
        scheduler=scheduler,
        uni_extractor=uni_extractor,
        uni_embeds=uni_embeds,
        guidance_scale=args.guidance_scale,
        device=device,
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )

    print("\n=== Validation Results ===")
    print(f"N tiles:          {len(results['cosine_similarities'])}")
    print(f"Mean cosine sim:  {results['mean_cosine_sim']:.4f}")
    print(f"Std cosine sim:   {results['std_cosine_sim']:.4f}")


if __name__ == "__main__":
    main()
