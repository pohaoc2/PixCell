#!/usr/bin/env python3
"""
Random tiles from ORION-CRC33: paired + unpaired Stage 3 visualizations.

**Paired** (per sampled tile ``t``): TME channels from ``t``, UNI from ``features/{t}_uni.npy``,
reference H&E from ``he/{t}.png``, outputs under ``{output}/paired/{t}/``.

**Unpaired** (per sampled layout tile ``A``): TME channels from ``A``, UNI from a different
tile ``B`` (``features/{B}_uni.npy``), reference H&E panel shows ``he/{B}.png``, cosine sim
vs layout ground truth uses ``features/{A}_uni.npy``, outputs under
``{output}/unpaired/{A}_layout_{B}_style/``.

Example::

    python tools/generate_orion_paired_unpaired_batch.py \\
        --n-tiles 5 \\
        --output-dir inference_output/orion_batch \\
        --seed 42
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

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description="ORION paired + unpaired batch vis")
    parser.add_argument("--data-root", type=str, default=str(ROOT / "data/orion-crc33"))
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(ROOT / "inference_output/orion_paired_unpaired"),
    )
    parser.add_argument("--n-tiles", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=str, default=str(ROOT / "configs/config_controlnet_exp.py"))
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--guidance-scale", type=float, default=2.5)
    parser.add_argument("--num-steps", type=int, default=20)
    args = parser.parse_args()

    os.chdir(ROOT)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    from diffusion.utils.misc import read_config

    from tools.generate_stage3_tile_vis import run_vis_suite
    from tools.stage3.tile_pipeline import (
        find_latest_checkpoint_dir,
        list_tile_ids_from_exp_channels,
        load_all_models,
        resolve_data_layout,
    )

    data_root = Path(args.data_root)
    exp_channels_dir, feat_dir, he_dir = resolve_data_layout(data_root)
    all_ids = list_tile_ids_from_exp_channels(exp_channels_dir)
    if len(all_ids) < max(args.n_tiles, 2):
        raise RuntimeError(
            f"Need at least max(n_tiles, 2) tiles; got {len(all_ids)} under {exp_channels_dir}"
        )

    sample = random.sample(all_ids, args.n_tiles)

    ckpt_parent = (
        Path(args.checkpoint_dir)
        if args.checkpoint_dir
        else ROOT / "checkpoints/pixcell_controlnet_exp/checkpoints"
    )
    ckpt_dir = find_latest_checkpoint_dir(ckpt_parent)
    print(f"Checkpoint: {ckpt_dir}")

    config = read_config(args.config)
    config._filename = args.config
    models = load_all_models(config, args.config, ckpt_dir, args.device)

    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
        prediction_type="epsilon",
        clip_sample=False,
    )
    scheduler.set_timesteps(args.num_steps, device=args.device)

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Sampled {args.n_tiles} tiles (seed={args.seed}): {sample}")

    # ── Paired ────────────────────────────────────────────────────────────
    for tid in sample:
        print(f"\n=== PAIRED  tile={tid} ===")
        feat = feat_dir / f"{tid}_uni.npy"
        if not feat.exists():
            print(f"  SKIP: missing {feat}")
            continue
        he_path = he_dir / f"{tid}.png"
        if not he_path.exists():
            print(f"  WARN: no reference H&E at {he_path} — overview style column omitted")
        uni = torch.from_numpy(np.load(feat)).view(1, 1, 1, 1536)
        run_vis_suite(
            layout_tile_id=tid,
            models=models,
            scheduler=scheduler,
            config=config,
            device=args.device,
            exp_channels_dir=exp_channels_dir,
            feat_dir=feat_dir,
            he_dir=he_dir,
            out_dir=out_root / "paired" / tid,
            uni_embeds=uni,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
            style_reference_he_path=None,
            cosine_compare_feat_path=feat,
            disable_cosine=False,
        )

    # ── Unpaired (layout A × style B) ───────────────────────────────────────
    for tid_A in sample:
        pool = [x for x in all_ids if x != tid_A]
        tid_B = random.choice(pool)
        print(f"\n=== UNPAIRED  layout={tid_A}  style={tid_B} ===")
        feat_A = feat_dir / f"{tid_A}_uni.npy"
        feat_B = feat_dir / f"{tid_B}_uni.npy"
        if not feat_B.exists():
            print(f"  SKIP: missing style UNI {feat_B}")
            continue
        if not feat_A.exists():
            print(f"  SKIP: missing layout cosine ref {feat_A}")
            continue
        he_B = he_dir / f"{tid_B}.png"
        if not he_B.exists():
            print(f"  WARN: no style H&E at {he_B} — overview style column omitted")
        uni_B = torch.from_numpy(np.load(feat_B)).view(1, 1, 1, 1536)
        run_vis_suite(
            layout_tile_id=tid_A,
            models=models,
            scheduler=scheduler,
            config=config,
            device=args.device,
            exp_channels_dir=exp_channels_dir,
            feat_dir=feat_dir,
            he_dir=he_dir,
            out_dir=out_root / "unpaired" / f"{tid_A}_layout_{tid_B}_style",
            uni_embeds=uni_B,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
            style_reference_he_path=he_B,
            cosine_compare_feat_path=feat_A,
            disable_cosine=False,
            overview_style_label="H&E (style from B)",
            ablation_ref_label="H&E (style from B)",
        )

    print(f"\nAll outputs under {out_root}")


if __name__ == "__main__":
    main()
