"""
tools/run_evaluation.py — Stage 3 batch inference + evaluation visualizations.

Output per tile:
    {output_dir}/{tile_id}/{paired,unpaired}/
        generated_he.png    generated H&E image
        overview.png        TME input channels → generated H&E
        ablation_grid.png   progressive group ablation (4-row)

Paired   = same tile's UNI embedding + TME channels.
Unpaired = next tile's UNI embedding + this tile's TME channels (cross-patch style).

Also writes metrics.json with per-tile UNI cosine similarity scores.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from diffusers import DDPMScheduler
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from tools.stage3.figures import (
    save_enhanced_ablation_grid,
    save_overview_figure,
)
from tools.stage3.tile_pipeline import (
    generate_ablation_images,
    generate_tile,
    list_tile_ids_from_exp_channels,
    load_all_models,
)
from tools.stage3.uni_cosine_similarity import cosine_similarity_uni


def _load_he(he_dir: Path, tile_id: str) -> np.ndarray | None:
    p = he_dir / f"{tile_id}.png"
    return np.array(Image.open(p).convert("RGB")) if p.exists() else None


def _load_uni(feat_dir: Path, tile_id: str) -> torch.Tensor | None:
    p = feat_dir / f"{tile_id}_uni.npy"
    return torch.from_numpy(np.load(p)).view(1, 1, 1, 1536) if p.exists() else None


def run_one(
    *,
    tile_id: str,
    style_tile_id: str,
    mode: str,                  # "paired" or "unpaired"
    out_dir: Path,
    models: dict,
    config,
    scheduler,
    feat_dir: Path,
    he_dir: Path,
    exp_ch_dir: Path,
    null_uni: torch.Tensor,
    device: str,
    guidance_scale: float,
    seed: int,
    uni_extractor=None,
) -> float | None:
    """Generate H&E + all visualizations for one tile/mode. Returns cosine sim or None."""
    tile_out = out_dir / tile_id / mode
    tile_out.mkdir(parents=True, exist_ok=True)

    # Style UNI from style_tile_id (same tile for paired, different for unpaired)
    _uni = _load_uni(feat_dir, style_tile_id)
    uni_embeds = _uni if _uni is not None else null_uni
    ref_he     = _load_he(he_dir, style_tile_id)      # H&E shown as style reference

    # Generate + collect vis data
    gen_np, vis_data = generate_tile(
        tile_id=tile_id,
        models=models,
        config=config,
        scheduler=scheduler,
        uni_embeds=uni_embeds,
        device=device,
        exp_channels_dir=exp_ch_dir,
        guidance_scale=guidance_scale,
        return_vis_data=True,
        seed=seed,
    )
    Image.fromarray(gen_np).save(tile_out / "generated_he.png")

    ctrl_full       = vis_data["ctrl_full"]
    active_channels = vis_data["active_channels"]
    style_inp       = [("H&E (style)", ref_he)] if ref_he is not None else []
    refs            = [("style_ref", "H&E (style)", ref_he)] if ref_he is not None else []

    # overview.png
    sim = None
    if uni_extractor is not None:
        paired_uni = _load_uni(feat_dir, tile_id)
        if paired_uni is not None:
            exp_feat = paired_uni.numpy().ravel()
            gen_feat = uni_extractor.extract(gen_np)
            sim = cosine_similarity_uni(gen_feat, exp_feat)
    save_overview_figure(
        ctrl_full=ctrl_full,
        active_channels=active_channels,
        gen_np=gen_np,
        style_inputs=style_inp,
        save_path=tile_out / "overview.png",
        cosine_sim_val=sim,
    )

    # ablation_grid.png
    ablation_imgs = generate_ablation_images(
        tile_id=tile_id,
        models=models,
        config=config,
        scheduler=scheduler,
        uni_embeds=uni_embeds,
        device=device,
        exp_channels_dir=exp_ch_dir,
        guidance_scale=guidance_scale,
        seed=seed,
    )
    save_enhanced_ablation_grid(
        ablation_images=ablation_imgs,
        refs=refs,
        ctrl_full=ctrl_full,
        active_channels=active_channels,
        channel_groups=config.channel_groups,
        save_path=tile_out / "ablation_grid.png",
    )

    print(f"    [{mode}] saved → {tile_out}"
          + (f"  cos_sim={sim:.4f}" if sim is not None else ""))
    return sim


def main():
    parser = argparse.ArgumentParser(
        description="Stage 3 batch inference + visualizations (paired + unpaired)"
    )
    parser.add_argument("--config",
        default="configs/config_controlnet_exp.py")
    parser.add_argument("--checkpoint-dir",
        default="checkpoints/pixcell_controlnet_exp/checkpoints/zero_out_mask_post",
        help="Directory containing controlnet_*.pth + tme_module.pth")
    parser.add_argument("--data-root",   default="data/orion-crc33")
    parser.add_argument("--output-dir",  default="inference_output/zero_out_mask_post")
    parser.add_argument("--n-tiles",     type=int,   default=3,
        help="Number of tiles to process")
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--guidance-scale", type=float, default=2.5)
    parser.add_argument("--num-steps",   type=int,   default=20)
    parser.add_argument("--device",
        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-metrics",  action="store_true",
        help="Skip UNI cosine similarity computation (faster)")
    args = parser.parse_args()

    from diffusion.utils.misc import read_config
    from train_scripts.inference_controlnet import null_uni_embed

    CONFIG_PATH = ROOT / args.config
    CKPT_DIR    = ROOT / args.checkpoint_dir
    EXP_ROOT    = ROOT / args.data_root
    EXP_CH_DIR  = EXP_ROOT / "exp_channels"
    FEAT_DIR    = EXP_ROOT / "features"
    HE_DIR      = EXP_ROOT / "he"
    OUT_DIR     = ROOT / args.output_dir
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    config = read_config(str(CONFIG_PATH))
    config._filename = str(CONFIG_PATH)

    all_ids = list_tile_ids_from_exp_channels(EXP_CH_DIR)
    print(f"Found {len(all_ids)} tiles.")

    tile_ids = random.sample(all_ids, min(args.n_tiles, len(all_ids)))
    N = len(tile_ids)
    print(f"Processing {N} tiles → {OUT_DIR}")

    models = load_all_models(config, CONFIG_PATH, CKPT_DIR, args.device)

    scheduler = DDPMScheduler(
        num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02,
        beta_schedule="linear", prediction_type="epsilon", clip_sample=False,
    )
    scheduler.set_timesteps(args.num_steps, device=args.device)

    null_uni = null_uni_embed(device="cpu", dtype=torch.float32)

    uni_extractor = None
    if not args.no_metrics:
        from pipeline.extract_features import UNI2hExtractor
        uni_model_path = getattr(config, "uni_model_path", "./pretrained_models/uni-2h")
        uni_extractor = UNI2hExtractor(model_path=uni_model_path, device=args.device)

    paired_sims, unpaired_sims = [], []

    for i, tid in enumerate(tile_ids):
        style_tid = tile_ids[(i + 1) % N]   # next tile in rotation for unpaired
        print(f"\n[{i+1}/{N}] {tid}  (unpaired style: {style_tid})")

        sim_p = run_one(
            tile_id=tid, style_tile_id=tid, mode="paired",
            out_dir=OUT_DIR, models=models, config=config, scheduler=scheduler,
            feat_dir=FEAT_DIR, he_dir=HE_DIR, exp_ch_dir=EXP_CH_DIR,
            null_uni=null_uni, device=args.device,
            guidance_scale=args.guidance_scale, seed=args.seed,
            uni_extractor=uni_extractor,
        )
        if sim_p is not None:
            paired_sims.append(sim_p)

        sim_u = run_one(
            tile_id=tid, style_tile_id=style_tid, mode="unpaired",
            out_dir=OUT_DIR, models=models, config=config, scheduler=scheduler,
            feat_dir=FEAT_DIR, he_dir=HE_DIR, exp_ch_dir=EXP_CH_DIR,
            null_uni=null_uni, device=args.device,
            guidance_scale=args.guidance_scale, seed=args.seed,
            uni_extractor=uni_extractor,
        )
        if sim_u is not None:
            unpaired_sims.append(sim_u)

    # Summary
    print(f"\n{'='*50}")
    def _fmt(sims):
        if not sims:
            return "n/a"
        return f"{np.mean(sims):.4f} ± {np.std(sims):.4f}  (n={len(sims)})"

    print(f"Paired   cosine sim: {_fmt(paired_sims)}")
    print(f"Unpaired cosine sim: {_fmt(unpaired_sims)}")

    metrics = {
        "checkpoint": str(CKPT_DIR),
        "n_tiles": N,
        "tile_ids": tile_ids,
        "guidance_scale": args.guidance_scale,
        "paired_cosine_sims":   paired_sims,
        "unpaired_cosine_sims": unpaired_sims,
        "paired_mean":   float(np.mean(paired_sims))   if paired_sims   else None,
        "unpaired_mean": float(np.mean(unpaired_sims)) if unpaired_sims else None,
    }
    metrics_path = OUT_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics  → {metrics_path}")
    print(f"Outputs  → {OUT_DIR}")


if __name__ == "__main__":
    main()
