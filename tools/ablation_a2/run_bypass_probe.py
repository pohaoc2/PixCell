"""A2 bypass-probe inference with TME output forced to zero.

The meaningful bypass probe is run on a checkpoint trained with
``zero_mask_latent=False``. Under that additive design, zeroing the TME output
reduces the ControlNet conditioning to the VAE-encoded mask latent.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image


def compute_bypass_conditioning(
    *,
    mask_latent: torch.Tensor,
    tme_output: torch.Tensor,
) -> torch.Tensor:
    """Return additive ControlNet conditioning for the A2 bypass variant."""
    return mask_latent + tme_output


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - heavyweight path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--tile-ids", "--tile_ids", dest="tile_ids", required=True)
    parser.add_argument("--out-dir", "--out_dir", dest="out_dir", required=True)
    parser.add_argument("--exp-channels-dir", default=None)
    parser.add_argument("--features-dir", default=None)
    parser.add_argument("--num-steps", "--num_steps", dest="num_steps", type=int, default=30)
    parser.add_argument("--guidance-scale", "--guidance_scale", dest="guidance_scale", type=float, default=1.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    from diffusion.utils.misc import read_config
    from tools.stage3.common import make_inference_scheduler, resolve_uni_embedding
    from tools.stage3.tile_pipeline import (
        _decode_latents_to_image,
        _make_fixed_noise,
        load_all_models,
        prepare_tile_context,
        resolve_data_layout,
    )
    from train_scripts.inference_controlnet import denoise

    config = read_config(args.config)
    config._filename = args.config
    if getattr(config, "zero_mask_latent", None) is not False:
        raise ValueError("Bypass probe requires a config with zero_mask_latent=False")

    data_root = Path(getattr(config, "exp_data_root", "data/orion-crc33"))
    exp_channels_dir, features_dir, _ = resolve_data_layout(data_root)
    if args.exp_channels_dir:
        exp_channels_dir = Path(args.exp_channels_dir)
    if args.features_dir:
        features_dir = Path(args.features_dir)

    models = load_all_models(config, args.config, args.checkpoint, args.device)
    scheduler = make_inference_scheduler(num_steps=args.num_steps, device=args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tile_ids = [line.strip() for line in Path(args.tile_ids).read_text().splitlines() if line.strip()]
    for tile_id in tile_ids:
        uni_embeds = resolve_uni_embedding(tile_id, feat_dir=features_dir, null_uni=False)
        context = prepare_tile_context(
            tile_id=tile_id,
            models=models,
            config=config,
            uni_embeds=uni_embeds,
            device=args.device,
            exp_channels_dir=exp_channels_dir,
        )
        ctrl_input = compute_bypass_conditioning(
            mask_latent=context["vae_mask"],
            tme_output=torch.zeros_like(context["vae_mask"]),
        )
        fixed_noise = _make_fixed_noise(
            config=config,
            scheduler=scheduler,
            device=args.device,
            dtype=context["dtype"],
            seed=args.seed,
        )
        denoised = denoise(
            latents=fixed_noise,
            uni_embeds=context["uni_embeds"],
            controlnet_input_latent=ctrl_input,
            scheduler=scheduler,
            controlnet_model=context["controlnet"],
            pixcell_controlnet_model=context["base_model"],
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_steps,
            device=args.device,
        )
        image = _decode_latents_to_image(
            denoised,
            vae=context["vae"],
            vae_scale=context["vae_scale"],
            vae_shift=context["vae_shift"],
            dtype=context["dtype"],
        )
        out_path = out_dir / f"{tile_id}.png"
        Image.fromarray(image).save(out_path)
        print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
