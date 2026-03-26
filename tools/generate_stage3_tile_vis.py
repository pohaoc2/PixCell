#!/usr/bin/env python3
"""
Generate the three Stage 3 visualization PNGs for one experimental tile:

  - overview.png
  - attention_heatmaps.png
  - ablation_grid.png

Uses the latest controlnet_*.pth (by mtime) under --checkpoint-dir by default.

Default input layout: ``inference_data/sample`` (flat sim-style: ``cell_mask/``, etc.).
ORION-style trees with ``exp_channels/``, ``features/``, ``he/`` are also supported;
see ``tools.stage3_tile_pipeline.resolve_data_layout``.

Run from repo root:

  python tools/generate_stage3_tile_vis.py \\
      --tile-id sim_0001 \\
      --output-dir inference_output/my_vis

Optional: UNI embedding at ``{data-root}/features/{tile_id}_uni.npy`` or next to channels;
use ``--null-uni`` for TME-only when no embedding exists.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from diffusers import DDPMScheduler
from PIL import Image

# Repo root (parent of tools/)
ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description="Stage 3: overview + attention + ablation PNGs for one tile")
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
        help="Dataset root: flat channel folders (default: inference_data/sample) "
        "or ORION tree with exp_channels/, features/, he/",
    )
    parser.add_argument(
        "--exp-root",
        type=str,
        default=None,
        help="Deprecated alias for --data-root (overrides --data-root if set)",
    )
    parser.add_argument("--tile-id", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
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
    parser.add_argument(
        "--reference-he",
        type=str,
        default=None,
        help="Explicit reference H&E PNG for overview/style column (default: {he_dir}/{tile_id}.png)",
    )
    args = parser.parse_args()

    os.chdir(ROOT)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from diffusion.utils.misc import read_config
    from train_scripts.inference_controlnet import null_uni_embed

    from tools.stage3_figures import (
        save_enhanced_ablation_grid,
        save_enhanced_attention_figure,
        save_overview_figure,
    )
    from tools.stage3_tile_pipeline import (
        find_latest_checkpoint_dir,
        generate_ablation_images,
        generate_tile,
        load_all_models,
        resolve_data_layout,
    )

    data_root = Path(args.exp_root if args.exp_root is not None else args.data_root)
    exp_channels_dir, feat_dir, he_dir = resolve_data_layout(data_root)

    ckpt_parent = Path(args.checkpoint_dir) if args.checkpoint_dir else ROOT / "checkpoints/pixcell_controlnet_exp/checkpoints"
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

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    gen_np, vis_data = generate_tile(
        tile_id=args.tile_id,
        models=models,
        config=config,
        scheduler=scheduler,
        uni_embeds=uni_embeds,
        device=device,
        exp_channels_dir=exp_channels_dir,
        guidance_scale=args.guidance_scale,
        return_vis_data=True,
    )
    if vis_data is None:
        raise RuntimeError("vis_data missing")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_he_path = Path(args.reference_he) if args.reference_he else he_dir / f"{args.tile_id}.png"
    ref_he = np.array(Image.open(ref_he_path).convert("RGB")) if ref_he_path.exists() else None
    style_inp = ([("H&E (style)", ref_he)] if ref_he is not None else [])

    sim = None
    if feat_path.exists():
        from pipeline.extract_features import UNI2hExtractor

        exp_feat = np.load(feat_path)
        uni_model_path = getattr(config, "uni_model_path", str(ROOT / "pretrained_models/uni-2h"))
        extractor = UNI2hExtractor(model_path=uni_model_path, device=device)
        gen_feat = extractor.extract(gen_np)
        a = gen_feat / (np.linalg.norm(gen_feat) + 1e-8)
        b = exp_feat / (np.linalg.norm(exp_feat) + 1e-8)
        sim = float(np.dot(a, b))

    ctrl_full_np = vis_data["ctrl_full"]
    active_channels = vis_data["active_channels"]

    save_overview_figure(
        ctrl_full=ctrl_full_np,
        active_channels=active_channels,
        gen_np=gen_np,
        save_path=out_dir / "overview.png",
        style_inputs=style_inp,
        cosine_sim_val=sim,
    )
    save_enhanced_attention_figure(
        ctrl_full=ctrl_full_np,
        active_channels=active_channels,
        gen_np=gen_np,
        attn_maps=vis_data["attn_maps"],
        save_path=out_dir / "attention_heatmaps.png",
        style_inputs=style_inp,
    )

    print("Generating ablation grid...")
    ablation_imgs = generate_ablation_images(
        tile_id=args.tile_id,
        models=models,
        config=config,
        scheduler=scheduler,
        uni_embeds=uni_embeds,
        device=device,
        exp_channels_dir=exp_channels_dir,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
    )
    save_enhanced_ablation_grid(
        ablation_images=ablation_imgs,
        refs=[("style_ref", "H&E (style)", ref_he)] if ref_he is not None else [],
        save_path=out_dir / "ablation_grid.png",
    )

    print(f"Done → {out_dir}")


if __name__ == "__main__":
    main()
