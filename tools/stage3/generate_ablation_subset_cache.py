#!/usr/bin/env python3
"""
Generate single/pair/triple/all-four Stage 3 ablation images once and cache them as PNGs
(singles/, pairs/, triples/, all/).

This is intended for rapid iteration on the combined manuscript-style layout in
cache-backed figure scripts (for example ``tools.stage3.ablation_grid_figure.py``)
without rerunning diffusion every time.

Default output: ``inference_output/cache/{tile_id}``. Use ``--n-tiles`` to randomly
sample many tiles from ``--data-root`` (models load once).
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from diffusers import DDPMScheduler

ROOT = Path(__file__).resolve().parent.parent.parent


def generate_subset_cache_for_tile(
    tile_id: str,
    *,
    cache_dir: Path,
    models: dict,
    config,
    scheduler,
    exp_channels_dir: Path,
    feat_dir: Path,
    device: str,
    guidance_scale: float,
    seed: int,
    null_uni: bool,
    uni_npy: Path | None,
) -> Path:
    from train_scripts.inference_controlnet import null_uni_embed

    from tools.stage3.ablation import (
        build_subset_ablation_sections,
        group_names_from_channel_groups,
    )
    from tools.stage3.ablation_cache import save_subset_condition_cache
    from tools.stage3.tile_pipeline import generate_group_combination_ablation_images, load_exp_channels

    feat_path = Path(uni_npy) if uni_npy is not None else feat_dir / f"{tile_id}_uni.npy"
    if null_uni or not feat_path.exists():
        uni_embeds = null_uni_embed(device="cpu", dtype=torch.float32)
        if not null_uni:
            print(f"Warning: missing {feat_path}, using null UNI")
    else:
        uni_embeds = torch.from_numpy(np.load(feat_path)).view(1, 1, 1, 1536)

    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{tile_id}] Generating single-group cache images...")
    single_group_imgs = generate_group_combination_ablation_images(
        tile_id=tile_id,
        models=models,
        config=config,
        scheduler=scheduler,
        uni_embeds=uni_embeds,
        device=device,
        exp_channels_dir=exp_channels_dir,
        guidance_scale=guidance_scale,
        seed=seed,
        subset_size=1,
    )

    print(f"[{tile_id}] Generating pair-group cache images...")
    pair_group_imgs = generate_group_combination_ablation_images(
        tile_id=tile_id,
        models=models,
        config=config,
        scheduler=scheduler,
        uni_embeds=uni_embeds,
        device=device,
        exp_channels_dir=exp_channels_dir,
        guidance_scale=guidance_scale,
        seed=seed,
        subset_size=2,
    )

    print(f"[{tile_id}] Generating triple-group cache images...")
    triple_group_imgs = generate_group_combination_ablation_images(
        tile_id=tile_id,
        models=models,
        config=config,
        scheduler=scheduler,
        uni_embeds=uni_embeds,
        device=device,
        exp_channels_dir=exp_channels_dir,
        guidance_scale=guidance_scale,
        seed=seed,
        subset_size=3,
    )

    group_names = group_names_from_channel_groups(config.channel_groups)
    if len(group_names) >= 4:
        print(f"[{tile_id}] Generating all-four-groups cache image...")
        all_four_imgs = generate_group_combination_ablation_images(
            tile_id=tile_id,
            models=models,
            config=config,
            scheduler=scheduler,
            uni_embeds=uni_embeds,
            device=device,
            exp_channels_dir=exp_channels_dir,
            guidance_scale=guidance_scale,
            seed=seed,
            subset_size=4,
        )
        subset_sections = build_subset_ablation_sections(
            group_names,
            single_images=single_group_imgs,
            pair_images=pair_group_imgs,
            triple_images=triple_group_imgs,
            all_four_images=all_four_imgs,
        )
    else:
        subset_sections = build_subset_ablation_sections(
            group_names,
            single_images=single_group_imgs,
            pair_images=pair_group_imgs,
            triple_images=triple_group_imgs,
        )

    ctrl_full = load_exp_channels(
        tile_id,
        config.data.active_channels,
        config.image_size,
        exp_channels_dir,
    )
    cell_mask = None
    if "cell_masks" in config.data.active_channels:
        cell_mask = ctrl_full[config.data.active_channels.index("cell_masks")].numpy()

    manifest_path = save_subset_condition_cache(
        cache_dir,
        tile_id=tile_id,
        group_names=group_names,
        sections=subset_sections,
        cell_mask=cell_mask,
    )
    print(f"[{tile_id}] Saved subset ablation cache → {manifest_path}")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate and cache single/pair/triple Stage 3 ablation PNGs for one or many tiles",
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
    tile_group = parser.add_mutually_exclusive_group(required=True)
    tile_group.add_argument("--tile-id", type=str, default=None, help="One tile ID from exp_channels")
    tile_group.add_argument(
        "--n-tiles",
        "--n-tile",
        type=int,
        default=None,
        dest="n_tiles",
        metavar="N",
        help="Randomly sample N tiles from --data-root and write each under --cache-dir/{tile_id}",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="With --tile-id: output dir for that tile (default: inference_output/cache/{tile_id}). "
        "With --n-tiles: parent directory (default: inference_output/cache).",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--guidance-scale", type=float, default=2.5)
    parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42, help="Diffusion / generation seed (per tile)")
    parser.add_argument(
        "--tile-sample-seed",
        type=int,
        default=42,
        help="RNG seed when choosing tiles with --n-tiles (default: 42)",
    )
    parser.add_argument(
        "--null-uni",
        action="store_true",
        help="Use null UNI embedding instead of paired features/{tile}_uni.npy",
    )
    parser.add_argument(
        "--uni-npy",
        type=str,
        default=None,
        help="Explicit path to UNI embedding .npy (single-tile mode only; overrides {feat_dir}/{tile_id}_uni.npy)",
    )
    args = parser.parse_args()

    if args.n_tiles is not None and args.uni_npy is not None:
        parser.error("--uni-npy is only supported with --tile-id (single tile)")

    os.chdir(ROOT)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from diffusion.utils.misc import read_config

    from tools.stage3.tile_pipeline import (
        find_latest_checkpoint_dir,
        list_tile_ids_from_exp_channels,
        load_all_models,
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

    cache_parent_default = ROOT / "inference_output" / "cache"

    if args.tile_id is not None:
        cache_dir = (
            Path(args.cache_dir)
            if args.cache_dir is not None
            else cache_parent_default / args.tile_id
        )
        uni_override = Path(args.uni_npy) if args.uni_npy else None
        generate_subset_cache_for_tile(
            args.tile_id,
            cache_dir=cache_dir,
            models=models,
            config=config,
            scheduler=scheduler,
            exp_channels_dir=exp_channels_dir,
            feat_dir=feat_dir,
            device=device,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
            null_uni=args.null_uni,
            uni_npy=uni_override,
        )
        return

    # --n-tiles batch
    if args.n_tiles < 1:
        parser.error("--n-tiles must be >= 1")

    all_ids = list_tile_ids_from_exp_channels(exp_channels_dir)
    if len(all_ids) < args.n_tiles:
        parser.error(
            f"need at least {args.n_tiles} tiles under {exp_channels_dir}, found {len(all_ids)}"
        )

    random.seed(args.tile_sample_seed)
    selected = random.sample(all_ids, args.n_tiles)
    cache_parent = Path(args.cache_dir) if args.cache_dir is not None else cache_parent_default
    print(
        f"Sampled {args.n_tiles} tiles (tile_sample_seed={args.tile_sample_seed}): {selected}"
    )

    for tile_id in selected:
        cache_dir = cache_parent / tile_id
        generate_subset_cache_for_tile(
            tile_id,
            cache_dir=cache_dir,
            models=models,
            config=config,
            scheduler=scheduler,
            exp_channels_dir=exp_channels_dir,
            feat_dir=feat_dir,
            device=device,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
            null_uni=args.null_uni,
            uni_npy=None,
        )


if __name__ == "__main__":
    main()
