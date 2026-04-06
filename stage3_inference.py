"""
Stage 3: Inference
==================
Generate experimental-like H&E images from simulation channel inputs.

The trained model (Stage 2) learns sim/exp channel → H&E from PAIRED experimental
data. At inference, we pass UNPAIRED simulation channels to generate realistic H&E
images matching the simulated TME layout.

Two inference modes:

    Style-conditioned (recommended when a reference tissue is available):
        Pass a reference H&E image (--reference-he). Its UNI-2h embedding guides
        the global tissue appearance (staining style, cell density, collagen patterns).

    TME-only (when no reference is available):
        Omit --reference-he. A null UNI embedding is used, relying entirely on the
        TME channel layout for generation. Requires CFG dropout during training
        (cfg_dropout_prob > 0 in config, default 0.15).

Usage:
    # Style-conditioned generation
    python stage3_inference.py \\
        --config           configs/config_controlnet_exp.py \\
        --checkpoint-dir   checkpoints/pixcell_controlnet_exp/checkpoints/step_XXXXXXX \\
        --sim-channels-dir /path/to/sim_channels \\
        --sim-id           my_simulation_001 \\
        --reference-he     /path/to/reference.png \\
        --output           generated_he.png

    # TME-only generation (no reference H&E)
    python stage3_inference.py \\
        --config           configs/config_controlnet_exp.py \\
        --checkpoint-dir   checkpoints/pixcell_controlnet_exp/checkpoints/step_XXXXXXX \\
        --sim-channels-dir /path/to/sim_channels \\
        --sim-id           my_simulation_001 \\
        --output           generated_he.png

    # Batch: generate for all sims in a directory
    python stage3_inference.py \\
        --config           configs/config_controlnet_exp.py \\
        --checkpoint-dir   checkpoints/pixcell_controlnet_exp/checkpoints/step_XXXXXXX \\
        --sim-channels-dir /path/to/sim_channels \\
        --output-dir       ./inference_output \\
        --n-tiles          50

Simulation channel directory layout:
    sim_channels/
    ├── cell_mask/        {sim_id}.png     binary cell segmentation (required)
    ├── cell_type_healthy/{sim_id}.png     (optional, exp-compatible channels)
    ├── cell_type_cancer/ {sim_id}.png
    ├── cell_type_immune/ {sim_id}.png
    ├── cell_state_prolif/{sim_id}.png
    ├── cell_state_nonprolif/{sim_id}.png
    ├── cell_state_dead/  {sim_id}.png
    ├── oxygen/           {sim_id}.png/.npy
    ├── glucose/          {sim_id}.png/.npy
    └── vasculature/      {sim_id}.png     (optional)

Channel availability is controlled by --active-channels (default: all 10 exp channels).
Missing optional channels are silently skipped.
"""
import argparse
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from diffusion.utils.misc import read_config
from train_scripts.inference_controlnet import (
    null_uni_embed,
    denoise,
)
from tools.stage3.common import make_inference_scheduler
from tools.stage3.tile_pipeline import (
    generate_from_ctrl,
    load_all_models,
    load_sim_channels as load_sim_channels_shared,
)

# ── Channel loading ───────────────────────────────────────────────────────────


def load_sim_channels(sim_channels_dir: Path, sim_id: str,
                      active_channels: list[str], resolution: int) -> torch.Tensor:
    """Load simulation channel images for a single snapshot → [C, H, W] tensor."""
    return load_sim_channels_shared(
        sim_id,
        active_channels,
        resolution,
        sim_channels_dir,
    )


# ── Model loading ─────────────────────────────────────────────────────────────

def load_models(config, checkpoint_dir: str, device: str):
    """Load VAE, ControlNet, base model, and TME module from a training checkpoint."""
    return load_all_models(
        config,
        getattr(config, "_filename", None),
        checkpoint_dir,
        device,
    )


# ── Single-tile generation ────────────────────────────────────────────────────

def generate(
    sim_channels_dir: Path,
    sim_id: str,
    models: dict,
    config,
    uni_embeds: torch.Tensor,
    scheduler,
    guidance_scale: float,
    device: str,
    seed: int | None = None,
    active_groups: set | None = None,
) -> np.ndarray:
    """Generate a single experimental-like H&E image from simulation channels."""
    active_channels = config.data.active_channels
    ctrl_full = load_sim_channels(
        sim_channels_dir, sim_id, active_channels, resolution=config.image_size
    )
    gen_np, _ = generate_from_ctrl(
        ctrl_full,
        models=models,
        config=config,
        scheduler=scheduler,
        uni_embeds=uni_embeds,
        device=device,
        guidance_scale=guidance_scale,
        seed=seed,
        active_groups=active_groups,
        denoise_fn=denoise,
    )
    return gen_np


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Stage 3: Generate experimental H&E from simulation channels"
    )
    parser.add_argument("--config",           required=True,
                        help="Training config path (configs/config_controlnet_exp.py)")
    parser.add_argument("--checkpoint-dir",   required=True,
                        help="Stage 2 checkpoint directory (contains controlnet_*.pth + tme_module.pth)")
    parser.add_argument("--sim-channels-dir", required=True,
                        help="Root directory with per-channel subdirectories")

    # Single-tile mode
    parser.add_argument("--sim-id",           default=None,
                        help="Simulation ID to generate (file stem, e.g. 'sim_0001')")
    parser.add_argument("--output",           default=None,
                        help="Output PNG path for single-tile mode")

    # Batch mode
    parser.add_argument("--output-dir",       default=None,
                        help="Output directory for batch generation")
    parser.add_argument("--n-tiles",          type=int,   default=None,
                        help="Max number of tiles to generate in batch mode")

    # Style conditioning
    parser.add_argument("--reference-he",     default=None,
                        help="Reference H&E image (PNG) for style conditioning. "
                             "If omitted, uses null UNI embedding (TME-only mode).")
    parser.add_argument("--reference-uni",    default=None,
                        help="Precomputed reference UNI embedding (.npy). "
                             "Alternative to --reference-he (skips UNI extraction).")

    # Generation settings
    parser.add_argument("--guidance-scale",   type=float, default=2.5)
    parser.add_argument("--num-steps",        type=int,   default=20)
    parser.add_argument("--device",           default="cuda")
    parser.add_argument("--seed",             type=int,   default=42)
    parser.add_argument(
        "--active-groups",
        nargs="*",
        default=None,
        help="TME groups to include (default: all). "
        "e.g., --active-groups cell_types vasculature",
    )
    parser.add_argument(
        "--drop-groups",
        nargs="*",
        default=None,
        help="TME groups to exclude. e.g., --drop-groups microenv",
    )
    args = parser.parse_args()

    # ── Validate args ─────────────────────────────────────────────────────────
    if args.sim_id is None and args.output_dir is None:
        parser.error("Provide --sim-id (single tile) or --output-dir (batch).")
    if args.sim_id is not None and args.output is None:
        parser.error("--sim-id requires --output.")

    config = read_config(args.config)
    # Attach config filename for model loader
    config._filename = args.config
    device = args.device

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ── Load models ───────────────────────────────────────────────────────────
    print("Loading models...")
    models = load_models(config, args.checkpoint_dir, device)

    channel_groups_cfg = getattr(config, "channel_groups", None)
    if channel_groups_cfg is not None:
        all_group_names = {g["name"] for g in channel_groups_cfg}
        if args.active_groups is not None:
            unknown = set(args.active_groups) - all_group_names
            if unknown:
                parser.error(
                    f"Unknown groups: {unknown}. Valid: {all_group_names}"
                )
            active_groups = set(args.active_groups)
        elif args.drop_groups is not None:
            unknown = set(args.drop_groups) - all_group_names
            if unknown:
                parser.error(
                    f"Unknown groups: {unknown}. Valid: {all_group_names}"
                )
            active_groups = all_group_names - set(args.drop_groups)
        else:
            active_groups = None
    else:
        active_groups = None

    # ── UNI embedding ─────────────────────────────────────────────────────────
    if args.reference_uni:
        ref = np.load(args.reference_uni)
        uni_embeds = torch.from_numpy(ref).view(1, 1, 1, 1536)
        print(f"Style conditioning: loaded UNI embedding from {args.reference_uni}")
    elif args.reference_he:
        from pipeline.extract_features import UNI2hExtractor
        uni_model_path = getattr(config, "uni_model_path",
                                 "./pretrained_models/uni-2h")
        extractor = UNI2hExtractor(model_path=uni_model_path, device=device)
        import cv2
        img = cv2.cvtColor(cv2.imread(args.reference_he), cv2.COLOR_BGR2RGB)
        feat = extractor.extract(img)
        uni_embeds = torch.from_numpy(feat).view(1, 1, 1, 1536)
        print(f"Style conditioning: extracted UNI from {args.reference_he}")
    else:
        uni_embeds = null_uni_embed(device='cpu', dtype=torch.float32)
        print("TME-only mode: using null UNI embedding")

    # ── Scheduler ─────────────────────────────────────────────────────────────
    scheduler = make_inference_scheduler(num_steps=args.num_steps, device=device)

    sim_channels_dir = Path(args.sim_channels_dir)

    # ── Single-tile mode ──────────────────────────────────────────────────────
    if args.sim_id is not None:
        print(f"Generating H&E for sim_id='{args.sim_id}'...")
        img = generate(
            sim_channels_dir=sim_channels_dir,
            sim_id=args.sim_id,
            models=models,
            config=config,
            uni_embeds=uni_embeds,
            scheduler=scheduler,
            guidance_scale=args.guidance_scale,
            device=device,
            seed=args.seed,
            active_groups=active_groups,
        )
        Image.fromarray(img).save(args.output)
        print(f"Saved → {args.output}")
        return

    # ── Batch mode ────────────────────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cell_mask_dir = sim_channels_dir / "cell_mask"
    mask_pngs = list(cell_mask_dir.glob("*.png"))
    if not mask_pngs:
        alt = sim_channels_dir / "cell_masks"
        mask_pngs = list(alt.glob("*.png"))
        if mask_pngs:
            cell_mask_dir = alt
    sim_ids = sorted(p.stem for p in mask_pngs)
    if args.n_tiles:
        sim_ids = sim_ids[: args.n_tiles]

    if not sim_ids:
        raise RuntimeError(
            f"No sim channel PNGs in {sim_channels_dir / 'cell_mask'} "
            f"or {sim_channels_dir / 'cell_masks'}"
        )

    print(f"Batch generating {len(sim_ids)} tiles → {output_dir}")
    for sim_id in sim_ids:
        img = generate(
            sim_channels_dir=sim_channels_dir,
            sim_id=sim_id,
            models=models,
            config=config,
            uni_embeds=uni_embeds,
            scheduler=scheduler,
            guidance_scale=args.guidance_scale,
            device=device,
            seed=args.seed,
            active_groups=active_groups,
        )
        Image.fromarray(img).save(output_dir / f"{sim_id}_he.png")
        print(f"  {sim_id} → {output_dir / (sim_id + '_he.png')}")

    print(f"\nDone. Generated {len(sim_ids)} images → {output_dir}")


if __name__ == "__main__":
    main()
