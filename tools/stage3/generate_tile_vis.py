#!/usr/bin/env python3
"""
Generate Stage 3 visualization PNGs for one experimental tile:

  - overview.png
  - ablation_grid.png
  - ablation_single_groups.png
  - ablation_group_pairs.png
  - ablation_group_triples.png
  - ablation_orders/*.png

Uses the latest controlnet_*.pth (by mtime) under --checkpoint-dir by default.

Default input layout: ``inference_data/sample`` (flat sim-style: ``cell_mask/``, etc.).
ORION-style trees with ``exp_channels/``, ``features/``, ``he/`` are also supported;
see ``tools.stage3.tile_pipeline.resolve_data_layout``.

**Reference H&E in figures:** The overview “style” column uses a PNG from
``--reference-he``, or else ``{he_dir}/{tile_id}.png`` when ``he/`` exists (ORION), or
``{data-root}/{tile_id}.png`` for flat layouts. If that file is missing, the style
column is omitted (common for ``inference_data/sample`` without paired H&E tiles).

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
ROOT = Path(__file__).resolve().parent.parent.parent


def run_vis_suite(
    layout_tile_id: str,
    models: dict,
    scheduler,
    config,
    device: str,
    exp_channels_dir: Path,
    feat_dir: Path,
    he_dir: Path,
    out_dir: Path,
    uni_embeds: torch.Tensor,
    *,
    guidance_scale: float,
    seed: int,
    style_reference_he_path: Path | None = None,
    cosine_compare_feat_path: Path | None = None,
    disable_cosine: bool = False,
    overview_style_label: str = "H&E (style)",
    ablation_ref_section: str = "style_ref",
    ablation_ref_label: str = "H&E (style)",
) -> None:
    """
    Run one generation + save overview / ablation outputs.

    - ``layout_tile_id``: TME channels loaded from ``exp_channels_dir`` for this tile.
    - ``uni_embeds``: Style conditioning (paired: same tile's UNI; unpaired: style tile B).
    - ``style_reference_he_path``: PNG shown in overview/attention style column; default
      ``he_dir / {layout_tile_id}.png`` when ``None``.
    - ``cosine_compare_feat_path``: UNI .npy to compare generated image against (cosine
      in overview title); default ``feat_dir / {layout_tile_id}_uni.npy`` when ``None``.
    """
    from pipeline.extract_features import UNI2hExtractor

    from tools.stage3.figures import (
        save_condition_ablation_grid,
        save_enhanced_ablation_grid,
        save_overview_figure,
    )
    from tools.stage3.ablation import order_slug, reorder_channel_groups
    from tools.stage3.tile_pipeline import (
        generate_all_progressive_order_ablation_images,
        generate_ablation_images,
        generate_group_combination_ablation_images,
        generate_tile,
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    ref_he_candidate = (
        style_reference_he_path
        if style_reference_he_path is not None
        else he_dir / f"{layout_tile_id}.png"
    )
    if ref_he_candidate.exists():
        ref_he = np.array(Image.open(ref_he_candidate).convert("RGB"))
        print(f"  Reference H&E panel: {ref_he_candidate}")
    else:
        ref_he = None
        print(
            f"  Reference H&E panel: SKIPPED (missing {ref_he_candidate}) — "
            "overview has no style column; use --reference-he or place PNG at default path."
        )

    torch.manual_seed(seed)
    np.random.seed(seed)

    gen_np, vis_data = generate_tile(
        tile_id=layout_tile_id,
        models=models,
        config=config,
        scheduler=scheduler,
        uni_embeds=uni_embeds,
        device=device,
        exp_channels_dir=exp_channels_dir,
        guidance_scale=guidance_scale,
        return_vis_data=True,
        seed=seed,
    )
    if vis_data is None:
        raise RuntimeError("vis_data missing")

    style_inp = ([(overview_style_label, ref_he)] if ref_he is not None else [])
    ablation_refs = ([(ablation_ref_section, ablation_ref_label, ref_he)] if ref_he is not None else [])

    sim = None
    if disable_cosine:
        print("  Cosine sim: skipped (disabled)")
    else:
        cos_feat = (
            cosine_compare_feat_path
            if cosine_compare_feat_path is not None
            else feat_dir / f"{layout_tile_id}_uni.npy"
        )
        if cos_feat.exists():
            exp_feat = np.load(cos_feat)
            uni_model_path = getattr(config, "uni_model_path", str(ROOT / "pretrained_models/uni-2h"))
            extractor = UNI2hExtractor(model_path=uni_model_path, device=device)
            gen_feat = extractor.extract(gen_np)
            a = gen_feat / (np.linalg.norm(gen_feat) + 1e-8)
            b = exp_feat / (np.linalg.norm(exp_feat) + 1e-8)
            sim = float(np.dot(a, b))
        else:
            print(f"  Cosine sim: skipped (missing {cos_feat})")

    ctrl_full_np = vis_data["ctrl_full"]
    active_channels = vis_data["active_channels"]

    # 1. Add green contour to overview.png — save_overview_figure already draws it when
    #    "cell_masks" is in active_channels; this is always true for the exp config.
    save_overview_figure(
        ctrl_full=ctrl_full_np,
        active_channels=active_channels,
        gen_np=gen_np,
        save_path=out_dir / "overview.png",
        style_inputs=style_inp,
        cosine_sim_val=sim,
    )
    print("  Generating ablation grid...")
    ablation_imgs = generate_ablation_images(
        tile_id=layout_tile_id,
        models=models,
        config=config,
        scheduler=scheduler,
        uni_embeds=uni_embeds,
        device=device,
        exp_channels_dir=exp_channels_dir,
        guidance_scale=guidance_scale,
        seed=seed,
    )
    # Pass ctrl_full so the ablation grid also gets the green cell-mask contour overlay.
    save_enhanced_ablation_grid(
        ablation_images=ablation_imgs,
        refs=ablation_refs,
        ctrl_full=ctrl_full_np,
        active_channels=active_channels,
        channel_groups=config.channel_groups,
        save_path=out_dir / "ablation_grid.png",
    )

    ablation_all_groups_np = ablation_imgs[-1][1]

    # MSE between overview H&E and ablation all-groups H&E
    mse = float(np.mean((gen_np.astype(np.float32) - ablation_all_groups_np.astype(np.float32)) ** 2))
    print(f"  MSE(overview_he, ablation_all_groups_he) = {mse:.6f}"
          + (" ✓ MATCH" if mse == 0.0 else " ✗ MISMATCH — seed/generation path differs"))

    print("  Generating single-group ablations...")
    single_group_imgs = generate_group_combination_ablation_images(
        tile_id=layout_tile_id,
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
    save_condition_ablation_grid(
        single_group_imgs,
        out_dir / "ablation_single_groups.png",
        refs=ablation_refs,
        ctrl_full=ctrl_full_np,
        active_channels=active_channels,
    )

    print("  Generating group-pair ablations...")
    pair_group_imgs = generate_group_combination_ablation_images(
        tile_id=layout_tile_id,
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
    save_condition_ablation_grid(
        pair_group_imgs,
        out_dir / "ablation_group_pairs.png",
        refs=ablation_refs,
        ctrl_full=ctrl_full_np,
        active_channels=active_channels,
    )

    print("  Generating group-triple ablations...")
    triple_group_imgs = generate_group_combination_ablation_images(
        tile_id=layout_tile_id,
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
    save_condition_ablation_grid(
        triple_group_imgs,
        out_dir / "ablation_group_triples.png",
        refs=ablation_refs,
        ctrl_full=ctrl_full_np,
        active_channels=active_channels,
    )

    print("  Generating progressive order ablations (24 orders)...")
    order_dir = out_dir / "ablation_orders"
    order_dir.mkdir(parents=True, exist_ok=True)
    progressive_orders = generate_all_progressive_order_ablation_images(
        tile_id=layout_tile_id,
        models=models,
        config=config,
        scheduler=scheduler,
        uni_embeds=uni_embeds,
        device=device,
        exp_channels_dir=exp_channels_dir,
        guidance_scale=guidance_scale,
        seed=seed,
    )
    for idx, (group_order, order_imgs) in enumerate(progressive_orders, start=1):
        save_enhanced_ablation_grid(
            ablation_images=order_imgs,
            refs=ablation_refs,
            ctrl_full=ctrl_full_np,
            active_channels=active_channels,
            channel_groups=reorder_channel_groups(config.channel_groups, group_order),
            save_path=order_dir / f"{idx:02d}_{order_slug(group_order)}.png",
        )

    Image.fromarray(gen_np).save(out_dir / "generated_he.png")
    print(f"  Done → {out_dir}")


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

    from tools.stage3.tile_pipeline import (
        find_latest_checkpoint_dir,
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
        cos_path = None
    else:
        uni_embeds = torch.from_numpy(np.load(feat_path)).view(1, 1, 1, 1536)
        cos_path = feat_path

    ref_arg = Path(args.reference_he) if args.reference_he else None

    run_vis_suite(
        layout_tile_id=args.tile_id,
        models=models,
        scheduler=scheduler,
        config=config,
        device=device,
        exp_channels_dir=exp_channels_dir,
        feat_dir=feat_dir,
        he_dir=he_dir,
        out_dir=Path(args.output_dir),
        uni_embeds=uni_embeds,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        style_reference_he_path=ref_arg,
        cosine_compare_feat_path=cos_path,
        disable_cosine=args.null_uni or not feat_path.exists(),
    )


if __name__ == "__main__":
    main()
