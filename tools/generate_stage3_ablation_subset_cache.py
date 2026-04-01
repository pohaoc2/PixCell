#!/usr/bin/env python3
"""
Generate the 14 single/pair/triple Stage 3 ablation images once and cache them as PNGs.

This is intended for rapid iteration on the combined manuscript-style layout in
``tools/stage3_ablation_full_vis.py`` without rerunning diffusion every time.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from diffusers import DDPMScheduler

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate and cache single/pair/triple Stage 3 ablation PNGs for one tile",
    )
    parser.add_argument("--config", type=str, default=str(ROOT / "configs/config_controlnet_exp.py"))
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Folder with controlnet_*.pth and tme_module.pth. Default: latest under checkpoints/pixcell_controlnet_exp/checkpoints",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=str(ROOT / "inference_data/sample"),
        help="Dataset root: flat channel folders or ORION tree with exp_channels/, features/, he/",
    )
    parser.add_argument(
        "--exp-root",
        type=str,
        default=None,
        help="Deprecated alias for --data-root (overrides --data-root if set)",
    )
    parser.add_argument("--tile-id", type=str, required=True)
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Cache output directory (default: inference_output/test_combinations/{tile_id})",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--guidance-scale", type=float, default=2.5)
    parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--null-uni",
        action="store_true",
        help="Use null UNI embedding instead of paired features/{tile}_uni.npy",
    )
    parser.add_argument(
        "--uni-npy",
        type=str,
        default=None,
        help="Explicit path to UNI embedding .npy (overrides {feat_dir}/{tile_id}_uni.npy)",
    )
    args = parser.parse_args()

    os.chdir(ROOT)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from diffusion.utils.misc import read_config
    from train_scripts.inference_controlnet import null_uni_embed

    from tools.stage3_ablation import group_names_from_channel_groups
    from tools.stage3_ablation_cache import save_subset_condition_cache
    from tools.stage3_ablation_full_vis import build_subset_ablation_sections
    from tools.stage3_tile_pipeline import (
        find_latest_checkpoint_dir,
        generate_group_combination_ablation_images,
        load_all_models,
        load_exp_channels,
        resolve_data_layout,
    )

    data_root = Path(args.exp_root if args.exp_root is not None else args.data_root)
    exp_channels_dir, feat_dir, _ = resolve_data_layout(data_root)

    ckpt_parent = (
        Path(args.checkpoint_dir)
        if args.checkpoint_dir
        else ROOT / "checkpoints/pixcell_controlnet_exp/checkpoints"
    )
    ckpt_dir = find_latest_checkpoint_dir(ckpt_parent)
    print(f"Using checkpoint dir: {ckpt_dir}")

    config = read_config(args.config)
    config._filename = args.config
    device = args.device

    models = load_all_models(config, args.config, ckpt_dir, device)

    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
        prediction_type="epsilon",
        clip_sample=False,
    )
    scheduler.set_timesteps(args.num_steps, device=device)

    feat_path = Path(args.uni_npy) if args.uni_npy else feat_dir / f"{args.tile_id}_uni.npy"
    if args.null_uni or not feat_path.exists():
        uni_embeds = null_uni_embed(device="cpu", dtype=torch.float32)
        if not args.null_uni:
            print(f"Warning: missing {feat_path}, using null UNI")
    else:
        uni_embeds = torch.from_numpy(np.load(feat_path)).view(1, 1, 1, 1536)

    cache_dir = (
        Path(args.cache_dir)
        if args.cache_dir is not None
        else ROOT / "inference_output" / "test_combinations" / args.tile_id
    )
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("Generating single-group cache images...")
    single_group_imgs = generate_group_combination_ablation_images(
        tile_id=args.tile_id,
        models=models,
        config=config,
        scheduler=scheduler,
        uni_embeds=uni_embeds,
        device=device,
        exp_channels_dir=exp_channels_dir,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        subset_size=1,
    )

    print("Generating pair-group cache images...")
    pair_group_imgs = generate_group_combination_ablation_images(
        tile_id=args.tile_id,
        models=models,
        config=config,
        scheduler=scheduler,
        uni_embeds=uni_embeds,
        device=device,
        exp_channels_dir=exp_channels_dir,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        subset_size=2,
    )

    print("Generating triple-group cache images...")
    triple_group_imgs = generate_group_combination_ablation_images(
        tile_id=args.tile_id,
        models=models,
        config=config,
        scheduler=scheduler,
        uni_embeds=uni_embeds,
        device=device,
        exp_channels_dir=exp_channels_dir,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        subset_size=3,
    )

    group_names = group_names_from_channel_groups(config.channel_groups)
    subset_sections = build_subset_ablation_sections(
        group_names,
        single_images=single_group_imgs,
        pair_images=pair_group_imgs,
        triple_images=triple_group_imgs,
    )

    ctrl_full = load_exp_channels(
        args.tile_id,
        config.data.active_channels,
        config.image_size,
        exp_channels_dir,
    )
    cell_mask = None
    if "cell_masks" in config.data.active_channels:
        cell_mask = ctrl_full[config.data.active_channels.index("cell_masks")].numpy()

    manifest_path = save_subset_condition_cache(
        cache_dir,
        tile_id=args.tile_id,
        group_names=group_names,
        sections=subset_sections,
        cell_mask=cell_mask,
    )
    print(f"Saved subset ablation cache → {manifest_path}")


if __name__ == "__main__":
    main()
