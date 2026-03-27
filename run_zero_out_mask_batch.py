"""
run_zero_out_mask_batch.py — Stage 3 for 3 random ORION-CRC33 patches using the
zero_out_mask checkpoint.

Generates per patch, for both paired (UNI from reference H&E) and unpaired (null UNI):
  - generated_he.png
  - overview.png
  - attn_heatmaps.png
  - ablation_grid.png

Output: inference_output/zero_out_mask/<tile_id>/{paired,unpaired}/
"""
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from diffusers import DDPMScheduler
from PIL import Image

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

CONFIG_PATH = ROOT / "configs/config_controlnet_exp.py"
CKPT_DIR    = ROOT / "checkpoints/pixcell_controlnet_exp/zero_out_mask"
EXP_ROOT    = ROOT / "data/orion-crc33"
EXP_CH_DIR  = EXP_ROOT / "exp_channels"
FEAT_DIR    = EXP_ROOT / "features"
HE_DIR      = EXP_ROOT / "he"
OUT_DIR     = ROOT / "inference_output/zero_out_mask"

N_PATCHES      = 3
GUIDANCE_SCALE = 2.5
NUM_STEPS      = 20
SEED           = 42
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

from tools.stage3_figures import (
    save_enhanced_ablation_grid,
    save_enhanced_attention_figure,
    save_overview_figure,
)
from tools.stage3_tile_pipeline import (
    generate_ablation_images,
    generate_tile,
    load_all_models,
)


def run_one(tid, mode, uni_embeds, ref_he, models, config, scheduler, out_base):
    """Generate all vis for one tile in one mode (paired/unpaired)."""
    tile_out = out_base / mode
    tile_out.mkdir(parents=True, exist_ok=True)

    style_inp = [(f"H&E (style)", ref_he)] if ref_he is not None else []

    gen_np, vis_data = generate_tile(
        tile_id=tid,
        models=models,
        config=config,
        scheduler=scheduler,
        uni_embeds=uni_embeds,
        device=DEVICE,
        exp_channels_dir=EXP_CH_DIR,
        guidance_scale=GUIDANCE_SCALE,
        return_vis_data=True,
    )
    Image.fromarray(gen_np).save(tile_out / "generated_he.png")

    ctrl_np = vis_data["ctrl_full"]
    act_ch  = vis_data["active_channels"]

    save_overview_figure(
        ctrl_full=ctrl_np,
        active_channels=act_ch,
        gen_np=gen_np,
        style_inputs=style_inp,
        save_path=tile_out / "overview.png",
    )

    save_enhanced_attention_figure(
        ctrl_full=ctrl_np,
        active_channels=act_ch,
        gen_np=gen_np,
        attn_maps=vis_data["attn_maps"],
        residuals=vis_data["residuals"],
        style_inputs=style_inp,
        save_path=tile_out / "attn_heatmaps.png",
    )

    ablation_imgs = generate_ablation_images(
        tile_id=tid,
        models=models,
        config=config,
        scheduler=scheduler,
        uni_embeds=uni_embeds,
        device=DEVICE,
        exp_channels_dir=EXP_CH_DIR,
        guidance_scale=GUIDANCE_SCALE,
        seed=SEED,
    )
    refs = [("style_ref", "H&E (style)", ref_he)] if ref_he is not None else []
    save_enhanced_ablation_grid(
        ablation_images=ablation_imgs,
        refs=refs,
        save_path=tile_out / "ablation_grid.png",
    )
    print(f"    Saved {mode}/")


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    from diffusion.utils.misc import read_config
    from train_scripts.inference_controlnet import null_uni_embed

    print(f"Device: {DEVICE}")
    print(f"Checkpoint: {CKPT_DIR}")
    print(f"Output:     {OUT_DIR}")

    config = read_config(str(CONFIG_PATH))
    config._filename = str(CONFIG_PATH)

    mask_dir = EXP_CH_DIR / "cell_masks"
    all_ids = sorted(p.stem for p in mask_dir.glob("*.png"))
    print(f"Found {len(all_ids)} tiles")

    tile_ids = random.sample(all_ids, N_PATCHES)
    print(f"Selected: {tile_ids}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    models = load_all_models(config, CONFIG_PATH, CKPT_DIR, DEVICE)

    scheduler = DDPMScheduler(
        num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02,
        beta_schedule="linear", prediction_type="epsilon", clip_sample=False,
    )
    scheduler.set_timesteps(NUM_STEPS, device=DEVICE)

    def load_uni_and_he(tid):
        feat_path = FEAT_DIR / f"{tid}_uni.npy"
        he_path   = HE_DIR   / f"{tid}.png"
        uni = (torch.from_numpy(np.load(feat_path)).view(1, 1, 1, 1536)
               if feat_path.exists() else null_uni_embed(device="cpu", dtype=torch.float32))
        he  = np.array(Image.open(he_path).convert("RGB")) if he_path.exists() else None
        return uni, he

    for i, tid in enumerate(tile_ids):
        print(f"\n[{i+1}/{N_PATCHES}] {tid}  (TME/mask source)")
        tile_base = OUT_DIR / tid

        # Paired: style H&E from the same tile
        paired_uni, ref_he = load_uni_and_he(tid)

        # Unpaired: style H&E from a different tile (rotate through tile_ids)
        style_tid = tile_ids[(i + 1) % N_PATCHES]
        unpaired_uni, unpaired_he = load_uni_and_he(style_tid)
        print(f"  paired style: {tid} | unpaired style: {style_tid}")

        print("  Running paired...")
        run_one(tid, "paired", paired_uni, ref_he, models, config, scheduler, tile_base)

        print("  Running unpaired...")
        run_one(tid, "unpaired", unpaired_uni, unpaired_he, models, config, scheduler, tile_base)

    print(f"\nDone. Outputs in: {OUT_DIR}")
    for tid in tile_ids:
        print(f"  {tid}/paired/   {tid}/unpaired/")


if __name__ == "__main__":
    main()
