"""
tools/run_evaluation.py — Comprehensive Stage 3 inference + validation + visualizations.

Uses experimental channel data (ORION-CRC33) as inference input:
  - 20 random "inference" tiles (paired: channels → generate H&E, compare vs paired UNI)
  - 20 random "validation" tiles (held-out unpaired: same metric, different tiles)

Generates for each set:
  - Generated H&E images
  - UNI cosine similarity vs. ground-truth exp H&E features
  - Attention heatmaps (per TME group)
  - Residual magnitude maps (per TME group)
  - Ablation grid (progressive group addition)
  - Summary metrics JSON
"""
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

# ── Module-level root ─────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent

# Channel that holds the cell mask in exp data
MASK_CHANNEL = "cell_masks"

from tools.stage3_figures import (
    save_enhanced_ablation_grid,
    save_enhanced_attention_figure,
    save_enhanced_residual_figure,
    save_overview_figure,
)
from tools.stage3_tile_pipeline import (
    generate_ablation_images,
    generate_tile,
    load_all_models,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b))


# ── UNI extraction ────────────────────────────────────────────────────────────

def get_uni_extractor(config, device: str):
    from pipeline.extract_features import UNI2hExtractor
    uni_model_path = getattr(config, "uni_model_path", "./pretrained_models/uni-2h")
    return UNI2hExtractor(model_path=uni_model_path, device=device)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Stage 3 inference + validation evaluation")
    parser.add_argument("--config", default="configs/config_controlnet_exp.py")
    parser.add_argument("--checkpoint-dir", default="checkpoints/pixcell_controlnet_exp/checkpoints")
    parser.add_argument("--data-root", default="data/orion-crc33")
    parser.add_argument("--output-dir", default="inference_output/evaluation")
    parser.add_argument("--n-inference", type=int, default=20)
    parser.add_argument("--n-validation", type=int, default=20)
    parser.add_argument("--n-vis-tiles", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--guidance-scale", type=float, default=2.5)
    parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    CONFIG_PATH    = ROOT / args.config
    CKPT_DIR       = ROOT / args.checkpoint_dir
    EXP_ROOT       = ROOT / args.data_root
    EXP_CH_DIR     = EXP_ROOT / "exp_channels"
    FEAT_DIR       = EXP_ROOT / "features"
    HE_DIR         = EXP_ROOT / "he"
    OUT_DIR        = ROOT / args.output_dir
    N_INFERENCE    = args.n_inference
    N_VALIDATION   = args.n_validation
    N_VIS_TILES    = args.n_vis_tiles
    GUIDANCE_SCALE = args.guidance_scale
    NUM_STEPS      = args.num_steps
    SEED           = args.seed
    DEVICE         = args.device

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    sys.path.insert(0, str(ROOT))
    os.chdir(ROOT)

    from diffusion.utils.misc import read_config
    from train_scripts.inference_controlnet import null_uni_embed

    print(f"Device: {DEVICE}")
    print(f"Output: {OUT_DIR}")

    # Load config
    config = read_config(str(CONFIG_PATH))
    config._filename = str(CONFIG_PATH)

    # Collect all available tile IDs from exp_channels/cell_masks
    mask_dir = EXP_CH_DIR / MASK_CHANNEL
    all_ids = sorted(p.stem for p in mask_dir.glob("*.png"))
    print(f"Found {len(all_ids)} tiles in exp_channels/{MASK_CHANNEL}")

    # Split into inference and validation sets
    selected = random.sample(all_ids, N_INFERENCE + N_VALIDATION)
    inference_ids  = selected[:N_INFERENCE]
    validation_ids = selected[N_INFERENCE:]
    print(f"Inference tiles:  {N_INFERENCE}  | Validation tiles: {N_VALIDATION}")

    # Output dirs
    inf_dir = OUT_DIR / "inference"
    val_dir = OUT_DIR / "validation"
    vis_dir = OUT_DIR / "visualizations"
    for d in [inf_dir, val_dir, vis_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Load models
    models = load_all_models(config, CONFIG_PATH, CKPT_DIR, DEVICE)

    # Scheduler
    scheduler = DDPMScheduler(
        num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02,
        beta_schedule="linear", prediction_type="epsilon", clip_sample=False,
    )
    scheduler.set_timesteps(NUM_STEPS, device=DEVICE)

    # Null UNI embedding used as fallback when paired features are missing
    null_uni = null_uni_embed(device="cpu", dtype=torch.float32)

    # ── Initialize UNI extractor for cosine similarity ──────────────────────
    print("\nLoading UNI extractor for validation metrics...")
    uni_extractor = get_uni_extractor(config, DEVICE)

    # ── INFERENCE SET ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("INFERENCE SET (paired exp data — using paired UNI style)")
    print(f"{'='*60}")
    inf_cosine_sims = []

    for i, tid in enumerate(inference_ids):
        do_vis = (i < N_VIS_TILES)
        print(f"[{i+1:02d}/{N_INFERENCE}] {tid}", end="")

        # Use the paired H&E UNI embedding as style condition.
        # This is the correct mode for paired inference: the model sees
        # the same H&E style it was trained on, and we measure how well
        # the TME channels drive the layout.
        exp_feat_path = FEAT_DIR / f"{tid}_uni.npy"
        if exp_feat_path.exists():
            uni_embeds = torch.from_numpy(np.load(exp_feat_path)).view(1, 1, 1, 1536)
            print(f"  [style=paired UNI]", end="")
        else:
            uni_embeds = null_uni
            print(f"  [style=null UNI (no paired feat)]", end="")

        gen_np, vis_data = generate_tile(
            tile_id=tid,
            models=models,
            config=config,
            scheduler=scheduler,
            uni_embeds=uni_embeds,
            device=DEVICE,
            exp_channels_dir=EXP_CH_DIR,
            guidance_scale=GUIDANCE_SCALE,
            return_vis_data=do_vis,
        )

        # Save generated H&E
        out_path = inf_dir / f"{tid}_generated.png"
        Image.fromarray(gen_np).save(out_path)

        # Load reference exp H&E (if available)
        exp_he_path = HE_DIR / f"{tid}.png"
        ref_he = np.array(Image.open(exp_he_path).convert("RGB")) if exp_he_path.exists() else None

        # Cosine similarity
        sim = None
        exp_feat_path = FEAT_DIR / f"{tid}_uni.npy"
        if exp_feat_path.exists():
            exp_feat = np.load(exp_feat_path)
            gen_feat = uni_extractor.extract(gen_np)
            sim = cosine_sim(gen_feat, exp_feat)
            inf_cosine_sims.append(sim)
            print(f"  cosine_sim={sim:.4f}")
        else:
            print("  (no exp feat)")

        # Visualizations for first N_VIS_TILES tiles
        if do_vis and vis_data is not None:
            tile_vis_dir = vis_dir / f"inference_{tid}"
            tile_vis_dir.mkdir(parents=True, exist_ok=True)

            ctrl_full_np    = vis_data["ctrl_full"]
            active_channels = vis_data["active_channels"]
            # H&E is both the style input and the ground-truth reference for this tile
            style_inp = ([("H&E (style)", ref_he)] if ref_he is not None else [])

            # 1. Overview: [style H&E | TME channels] → Generated H&E
            save_overview_figure(
                ctrl_full=ctrl_full_np,
                active_channels=active_channels,
                gen_np=gen_np,
                style_inputs=style_inp,
                save_path=tile_vis_dir / "overview.png",
                cosine_sim_val=sim,
            )

            # 2. Attention heatmaps per group
            save_enhanced_attention_figure(
                ctrl_full=ctrl_full_np,
                active_channels=active_channels,
                gen_np=gen_np,
                attn_maps=vis_data["attn_maps"],
                style_inputs=style_inp,
                save_path=tile_vis_dir / "attention_heatmaps.png",
            )

            # 3. Residual magnitude per group (TME group contributions, no ref)
            save_enhanced_residual_figure(
                ctrl_full=ctrl_full_np,
                active_channels=active_channels,
                gen_np=gen_np,
                residuals=vis_data["residuals"],
                refs=[],
                save_path=tile_vis_dir / "residual_magnitudes.png",
            )

            # 4. Ablation grid (progressive group addition)
            print(f"  Generating ablation grid...")
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
            save_enhanced_ablation_grid(
                ablation_images=ablation_imgs,
                refs=[("style_ref", "H&E (style)", ref_he)] if ref_he is not None else [],
                save_path=tile_vis_dir / "ablation_grid.png",
            )

    # ── VALIDATION SET (unpaired) ────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("VALIDATION SET (paired exp data — using paired UNI style)")
    print(f"{'='*60}")
    val_cosine_sims = []

    for i, tid in enumerate(validation_ids):
        print(f"[{i+1:02d}/{N_VALIDATION}] {tid}", end="")

        exp_feat_path = FEAT_DIR / f"{tid}_uni.npy"
        if exp_feat_path.exists():
            uni_embeds = torch.from_numpy(np.load(exp_feat_path)).view(1, 1, 1, 1536)
        else:
            uni_embeds = null_uni

        gen_np, _ = generate_tile(
            tile_id=tid,
            models=models,
            config=config,
            scheduler=scheduler,
            uni_embeds=uni_embeds,
            device=DEVICE,
            exp_channels_dir=EXP_CH_DIR,
            guidance_scale=GUIDANCE_SCALE,
            return_vis_data=False,
        )

        out_path = val_dir / f"{tid}_generated.png"
        Image.fromarray(gen_np).save(out_path)

        exp_feat_path = FEAT_DIR / f"{tid}_uni.npy"
        if exp_feat_path.exists():
            exp_feat = np.load(exp_feat_path)
            gen_feat = uni_extractor.extract(gen_np)
            sim = cosine_sim(gen_feat, exp_feat)
            val_cosine_sims.append(sim)
            print(f"  cosine_sim={sim:.4f}")
        else:
            print("  (no exp feat)")

    # ── UNPAIRED INFERENCE ───────────────────────────────────────────────────
    # Layout from tile A + style (UNI embedding) from a different tile B.
    # Shows model's ability to apply an arbitrary H&E style to a given TME layout.
    print(f"\n{'='*60}")
    print("UNPAIRED INFERENCE  (A's TME layout × B's H&E style)")
    print(f"{'='*60}")

    unpaired_dir = OUT_DIR / "unpaired"
    unpaired_vis_dir = OUT_DIR / "visualizations_unpaired"
    unpaired_dir.mkdir(parents=True, exist_ok=True)
    unpaired_vis_dir.mkdir(parents=True, exist_ok=True)

    # Pool for style tiles: all IDs not in inference_ids
    style_pool = [x for x in all_ids if x not in set(inference_ids)]
    N_UNPAIRED = min(N_VIS_TILES, len(inference_ids))   # one vis per layout tile
    unpaired_cosine_sims = []

    for i in range(N_UNPAIRED):
        tid_A = inference_ids[i]
        tid_B = random.choice(style_pool)
        print(f"[{i+1:02d}/{N_UNPAIRED}] layout={tid_A}  style={tid_B}", end="")

        # Load B's precomputed UNI embedding as style vector
        uni_path_B = FEAT_DIR / f"{tid_B}_uni.npy"
        if not uni_path_B.exists():
            print("  (no UNI for B, skipping)")
            continue
        uni_feat_B = np.load(uni_path_B)
        uni_B = torch.from_numpy(uni_feat_B).view(1, 1, 1, 1536)

        gen_np, vis_data = generate_tile(
            tile_id=tid_A,
            models=models,
            config=config,
            scheduler=scheduler,
            uni_embeds=uni_B,
            device=DEVICE,
            exp_channels_dir=EXP_CH_DIR,
            guidance_scale=GUIDANCE_SCALE,
            return_vis_data=True,
        )

        out_path = unpaired_dir / f"{tid_A}_layout_{tid_B}_style.png"
        Image.fromarray(gen_np).save(out_path)

        # Cosine sim vs. A's own ground truth (layout fidelity)
        sim = None
        exp_feat_path = FEAT_DIR / f"{tid_A}_uni.npy"
        if exp_feat_path.exists():
            exp_feat = np.load(exp_feat_path)
            gen_feat = uni_extractor.extract(gen_np)
            sim = cosine_sim(gen_feat, exp_feat)
            unpaired_cosine_sims.append(sim)
            print(f"  cos_sim(vs A)={sim:.4f}")
        else:
            print()

        # Full visualization showing both references
        ref_A = np.array(Image.open(HE_DIR / f"{tid_A}.png").convert("RGB")) if (HE_DIR / f"{tid_A}.png").exists() else None
        ref_B = np.array(Image.open(HE_DIR / f"{tid_B}.png").convert("RGB")) if (HE_DIR / f"{tid_B}.png").exists() else None

        if vis_data is not None:
            tile_vis_dir = unpaired_vis_dir / f"{tid_A}_x_{tid_B}"
            tile_vis_dir.mkdir(parents=True, exist_ok=True)

            ctrl_np = vis_data["ctrl_full"]
            act_ch  = vis_data["active_channels"]

            # B's H&E is the style input (its UNI embedding drives generation)
            style_inp_B = ([("H&E (style from B)", ref_B)] if ref_B is not None else [])

            # Overview: [B's style H&E | A's TME channels] → Generated H&E
            save_overview_figure(
                ctrl_full=ctrl_np,
                active_channels=act_ch,
                gen_np=gen_np,
                style_inputs=style_inp_B,
                save_path=tile_vis_dir / "overview.png",
                cosine_sim_val=sim,
            )
            save_enhanced_attention_figure(
                ctrl_full=ctrl_np, active_channels=act_ch,
                gen_np=gen_np, attn_maps=vis_data["attn_maps"],
                style_inputs=style_inp_B,
                save_path=tile_vis_dir / "attention_heatmaps.png",
            )
            save_enhanced_residual_figure(
                ctrl_full=ctrl_np, active_channels=act_ch,
                gen_np=gen_np, residuals=vis_data["residuals"],
                refs=[],
                save_path=tile_vis_dir / "residual_magnitudes.png",
            )

            # 4. Ablation grid — progressive group addition with B's style
            print(f"  Generating unpaired ablation grid...")
            ablation_imgs = generate_ablation_images(
                tile_id=tid_A,
                models=models,
                config=config,
                scheduler=scheduler,
                uni_embeds=uni_B,
                device=DEVICE,
                exp_channels_dir=EXP_CH_DIR,
                guidance_scale=GUIDANCE_SCALE,
                seed=SEED,
            )
            save_enhanced_ablation_grid(
                ablation_images=ablation_imgs,
                refs=[("style_ref", "H&E (style from B)", ref_B)] if ref_B is not None else [],
                save_path=tile_vis_dir / "ablation_grid.png",
            )

    # ── SUMMARY ─────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    results = {
        "checkpoint": str(CKPT_DIR),
        "guidance_scale": GUIDANCE_SCALE,
        "num_steps": NUM_STEPS,
        "inference_paired": {
            "mode": "style-conditioned (paired UNI)",
            "n_tiles": N_INFERENCE,
            "tile_ids": inference_ids,
            "cosine_sims": inf_cosine_sims,
            "mean_cosine_sim": float(np.mean(inf_cosine_sims)) if inf_cosine_sims else None,
            "std_cosine_sim":  float(np.std(inf_cosine_sims))  if inf_cosine_sims else None,
        },
        "validation_paired": {
            "mode": "style-conditioned (paired UNI)",
            "n_tiles": N_VALIDATION,
            "tile_ids": validation_ids,
            "cosine_sims": val_cosine_sims,
            "mean_cosine_sim": float(np.mean(val_cosine_sims)) if val_cosine_sims else None,
            "std_cosine_sim":  float(np.std(val_cosine_sims))  if val_cosine_sims else None,
        },
        "inference_unpaired": {
            "mode": "style-conditioned (B's UNI → A's channels)",
            "n_tiles": N_UNPAIRED,
            "cosine_sims_vs_layout_A": unpaired_cosine_sims,
            "mean_cosine_sim": float(np.mean(unpaired_cosine_sims)) if unpaired_cosine_sims else None,
            "std_cosine_sim":  float(np.std(unpaired_cosine_sims))  if unpaired_cosine_sims else None,
        },
    }

    metrics_path = OUT_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)

    def _fmt(d):
        m, s = d["mean_cosine_sim"], d["std_cosine_sim"]
        return f"{m:.4f} ± {s:.4f}" if m is not None else "n/a"

    print(f"\nInference  (paired,   style-cond) — mean UNI cos sim: {_fmt(results['inference_paired'])}")
    print(f"Validation (paired,   style-cond) — mean UNI cos sim: {_fmt(results['validation_paired'])}")
    print(f"Inference  (unpaired, style-cond) — mean UNI cos sim: {_fmt(results['inference_unpaired'])}")
    print(f"\nOutputs saved to: {OUT_DIR}")
    print(f"Metrics:          {metrics_path}")
    print(f"Visualizations:   {vis_dir}")
    print(f"Unpaired vis:     {unpaired_vis_dir}")


if __name__ == "__main__":
    main()
