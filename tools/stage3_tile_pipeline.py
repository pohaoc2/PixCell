"""
Shared Stage 3 inference helpers: load exp channels, models, generate one tile, ablation sweep.

Used by run_stage3_full.py and tools/generate_stage3_tile_vis.py.
"""
from __future__ import annotations

import glob as _glob
import os
from pathlib import Path

import numpy as np
import torch

# Canonical exp names → on-disk folder names (e.g. sim layouts use cell_mask/)
_CHANNEL_DIR_ALIASES: dict[str, str] = {
    "cell_masks": "cell_mask",
}

# Binary channels (thresholded to {0,1})
_BINARY: frozenset[str] = frozenset(
    {
        "cell_masks",
        "cell_type_healthy",
        "cell_type_cancer",
        "cell_type_immune",
        "cell_state_prolif",
        "cell_state_nonprolif",
        "cell_state_dead",
    }
)


_MIRROR_BORDER_PX: int = 8  # reflect-pad non-binary channels before resize


def load_channel(ch_dir: Path, tile_id: str, resolution: int, binary: bool) -> np.ndarray:
    """Load a single channel PNG → float32 [H, W] in [0, 1].

    Non-binary (continuous) channels are reflect-padded by _MIRROR_BORDER_PX pixels
    before resize to suppress simulation boundary artifacts.
    """
    fpath = ch_dir / f"{tile_id}.png"
    if not fpath.exists():
        fpath = ch_dir / f"{tile_id}.npy"
    if not fpath.exists():
        raise FileNotFoundError(f"Channel file not found: {ch_dir / tile_id}.*")
    if fpath.suffix == ".npy":
        arr = np.load(fpath).astype(np.float32)
    else:
        import cv2

        img = cv2.imread(str(fpath), cv2.IMREAD_GRAYSCALE)
        arr = img.astype(np.float32) / 255.0
    if not binary and _MIRROR_BORDER_PX > 0:
        arr = np.pad(arr, _MIRROR_BORDER_PX, mode="reflect")
    if arr.shape != (resolution, resolution):
        import cv2

        arr = cv2.resize(arr, (resolution, resolution), interpolation=cv2.INTER_LINEAR)
    if binary:
        arr = (arr > 0.5).astype(np.float32)
    return arr


def resolve_data_layout(data_root: Path) -> tuple[Path, Path, Path]:
    """
    Return (channels_dir, features_dir, he_dir).

    ORION-style: ``data_root/exp_channels``, ``data_root/features``, ``data_root/he``.
    Flat sim-style: channel subdirs live directly under ``data_root``; optional
    ``features/`` and ``he/`` otherwise fall back to ``data_root`` for loose files.
    """
    if (data_root / "exp_channels").is_dir():
        ch = data_root / "exp_channels"
    else:
        ch = data_root
    feat = data_root / "features" if (data_root / "features").is_dir() else data_root
    he = data_root / "he" if (data_root / "he").is_dir() else data_root
    return ch, feat, he


def resolve_channel_dir(exp_channels_dir: Path, channel_name: str) -> Path:
    """Prefer config name; fall back to alias dirs (cell_masks → cell_mask)."""
    d = exp_channels_dir / channel_name
    if d.is_dir():
        return d
    alt = _CHANNEL_DIR_ALIASES.get(channel_name)
    if alt:
        d2 = exp_channels_dir / alt
        if d2.is_dir():
            return d2
    return exp_channels_dir / channel_name


def load_exp_channels(
    tile_id: str,
    active_channels: list,
    resolution: int,
    exp_channels_dir: Path,
    binary_channels: frozenset[str] | None = None,
) -> torch.Tensor:
    """Load all active channels from exp_channels → [C, H, W]."""
    binary_channels = binary_channels or _BINARY
    planes = []
    for ch in active_channels:
        ch_dir = resolve_channel_dir(exp_channels_dir, ch)
        arr = load_channel(ch_dir, tile_id, resolution, binary=(ch in binary_channels))
        planes.append(arr)
    return torch.from_numpy(np.stack(planes, axis=0))


def load_all_models(
    config,
    config_path: str | Path,
    ckpt_dir: str | Path,
    device: str,
):
    """Load VAE, ControlNet, base transformer, TME from a training checkpoint directory."""
    from diffusion.model.builder import build_model
    from train_scripts.inference_controlnet import (
        load_controlnet_model_from_checkpoint,
        load_pixcell_controlnet_model_from_checkpoint,
        load_vae,
    )
    from train_scripts.training_utils import load_tme_checkpoint

    config_path = str(config_path)
    ckpt_dir = str(ckpt_dir)

    print("Loading VAE...")
    vae = load_vae(config.vae_pretrained, device)

    print("Loading TME module...")
    group_specs = [dict(name=g["name"], n_channels=len(g["channels"])) for g in config.channel_groups]
    tme_module = build_model(
        "MultiGroupTMEModule",
        False,
        False,
        channel_groups=group_specs,
        base_ch=getattr(config, "tme_base_ch", 32),
    )
    load_tme_checkpoint(ckpt_dir, tme_module, device=device)
    dtype = torch.float16 if device == "cuda" else torch.float32
    tme_module.to(device=device, dtype=dtype).eval()

    print("Loading ControlNet...")
    controlnet_pths = sorted(_glob.glob(os.path.join(ckpt_dir, "controlnet_*.pth")))
    if not controlnet_pths:
        raise FileNotFoundError(f"No controlnet_*.pth in {ckpt_dir}")
    controlnet_pth = controlnet_pths[-1]
    print(f"  → {controlnet_pth}")
    controlnet = load_controlnet_model_from_checkpoint(config_path, controlnet_pth, device)

    print("Loading base model...")
    base_model_path = getattr(config, "load_from", config.base_model_path)
    if os.path.isdir(base_model_path):
        candidates = _glob.glob(os.path.join(base_model_path, "*.safetensors")) + _glob.glob(
            os.path.join(base_model_path, "*.pth")
        )
        if not candidates:
            raise FileNotFoundError(f"No .safetensors or .pth found in base_model_path={base_model_path}")
        base_model_path = sorted(candidates)[0]
    print(f"  → {base_model_path}")
    base_model = load_pixcell_controlnet_model_from_checkpoint(config_path, base_model_path)
    base_model.to(device).eval()

    return dict(vae=vae, controlnet=controlnet, base_model=base_model, tme_module=tme_module)


def generate_tile(
    tile_id: str,
    models: dict,
    config,
    scheduler,
    uni_embeds: torch.Tensor,
    device: str,
    exp_channels_dir: Path,
    guidance_scale: float,
    return_vis_data: bool = False,
    seed: int | None = None,
):
    """
    Generate H&E for one tile from its exp channels.

    Returns:
        gen_np: uint8 [H, W, 3]
        vis_data: dict with residuals, attn_maps, ctrl_full, active_channels (if return_vis_data)
    """
    from tools.channel_group_utils import split_channels_to_groups
    from train_scripts.inference_controlnet import denoise, encode_ctrl_mask_latent

    active_channels = config.data.active_channels
    vae_scale = config.scale_factor
    vae_shift = config.shift_factor
    dtype = torch.float16 if device == "cuda" else torch.float32

    vae = models["vae"]
    controlnet = models["controlnet"]
    base_model = models["base_model"]
    tme_module = models["tme_module"]

    vae.to(device=device, dtype=dtype).eval()

    ctrl_full = load_exp_channels(tile_id, active_channels, config.image_size, exp_channels_dir)

    vae_mask = encode_ctrl_mask_latent(
        ctrl_full,
        vae,
        vae_shift=vae_shift,
        vae_scale=vae_scale,
        device=device,
        dtype=dtype,
    )
    tme_dict = split_channels_to_groups(
        ctrl_full.unsqueeze(0).to(device, dtype=dtype),
        active_channels,
        config.channel_groups,
    )

    with torch.no_grad():
        if return_vis_data:
            # Pass 1: xformers path (no return_attn_weights) — same path as generate_ablation_images
            fused, residuals = tme_module(
                vae_mask,
                tme_dict,
                return_residuals=True,
            )
            # Pass 2: manual path for attn_maps only (visualization, does not affect generation)
            _, _, attn_maps = tme_module(
                vae_mask,
                tme_dict,
                return_residuals=True,
                return_attn_weights=True,
            )
        else:
            fused = tme_module(vae_mask, tme_dict)
            residuals, attn_maps = {}, {}
    if getattr(config, "zero_mask_latent", False):
        fused = fused - vae_mask

    latent_shape = (1, 16, config.image_size // 8, config.image_size // 8)
    if seed is not None:
        torch.manual_seed(seed)
    latents = torch.randn(latent_shape, device=device, dtype=dtype)
    latents = latents * scheduler.init_noise_sigma
    if seed is not None:
        torch.manual_seed(seed)  # reset again so scheduler-step noise matches ablation
    denoised = denoise(
        latents=latents,
        uni_embeds=uni_embeds.to(device, dtype=dtype),
        controlnet_input_latent=fused,
        scheduler=scheduler,
        controlnet_model=controlnet,
        pixcell_controlnet_model=base_model,
        guidance_scale=guidance_scale,
        device=device,
    )

    with torch.no_grad():
        scaled = (denoised.to(dtype) / vae_scale) + vae_shift
        gen_img = vae.decode(scaled, return_dict=False)[0]
    gen_img = (gen_img / 2 + 0.5).clamp(0, 1)
    gen_np = (gen_img.cpu().permute(0, 2, 3, 1).numpy()[0] * 255).astype(np.uint8)

    vis_data = None
    if return_vis_data:
        vis_data = dict(
            residuals=residuals,
            attn_maps=attn_maps,
            ctrl_full=ctrl_full.numpy(),
            active_channels=active_channels,
        )

    return gen_np, vis_data


def generate_ablation_images(
    tile_id: str,
    models: dict,
    config,
    scheduler,
    uni_embeds: torch.Tensor,
    device: str,
    exp_channels_dir: Path,
    guidance_scale: float,
    seed: int,
) -> list[tuple[str, np.ndarray]]:
    """Return list of (label, gen_np) for progressive group addition."""
    from tools.channel_group_utils import split_channels_to_groups
    from train_scripts.inference_controlnet import denoise, encode_ctrl_mask_latent

    active_channels = config.data.active_channels
    vae_scale = config.scale_factor
    vae_shift = config.shift_factor
    dtype = torch.float16 if device == "cuda" else torch.float32

    vae = models["vae"]
    controlnet = models["controlnet"]
    base_model = models["base_model"]
    tme_module = models["tme_module"]

    vae.to(device=device, dtype=dtype).eval()

    ctrl_full = load_exp_channels(tile_id, active_channels, config.image_size, exp_channels_dir)
    vae_mask = encode_ctrl_mask_latent(
        ctrl_full,
        vae,
        vae_shift=vae_shift,
        vae_scale=vae_scale,
        device=device,
        dtype=dtype,
    )
    tme_dict = split_channels_to_groups(
        ctrl_full.unsqueeze(0).to(device, dtype=dtype),
        active_channels,
        config.channel_groups,
    )

    group_names = [g["name"] for g in config.channel_groups]
    ablation_images = []
    _zero_mask = getattr(config, "zero_mask_latent", False)

    torch.manual_seed(seed)
    latent_shape = (1, 16, config.image_size // 8, config.image_size // 8)
    fixed_noise = torch.randn(latent_shape, device=device, dtype=dtype)
    fixed_noise = fixed_noise * scheduler.init_noise_sigma

    for n in range(len(group_names) + 1):
        if n == 0:
            label = "No conditioning\n(zero TME)" if _zero_mask else "Mask only\n(no TME groups)"
            active = set()
        else:
            active = set(group_names[:n])
            label = "Groups:\n" + "\n".join(group_names[:n])

        with torch.no_grad():
            if active:
                fused = tme_module(vae_mask, tme_dict, active_groups=active)
                if _zero_mask:
                    fused = fused - vae_mask
            else:
                fused = torch.zeros_like(vae_mask) if _zero_mask else vae_mask.clone()

        torch.manual_seed(seed)  # identical scheduler-step noise across all ablation steps
        denoised = denoise(
            latents=fixed_noise.clone(),
            uni_embeds=uni_embeds.to(device, dtype=dtype),
            controlnet_input_latent=fused,
            scheduler=scheduler,
            controlnet_model=controlnet,
            pixcell_controlnet_model=base_model,
            guidance_scale=guidance_scale,
            device=device,
        )
        with torch.no_grad():
            scaled = (denoised.to(dtype) / vae_scale) + vae_shift
            gen = vae.decode(scaled, return_dict=False)[0]
        gen = (gen / 2 + 0.5).clamp(0, 1)
        gen_np = (gen.cpu().permute(0, 2, 3, 1).numpy()[0] * 255).astype(np.uint8)
        ablation_images.append((label, gen_np))

    return ablation_images


def generate_loo_ablation(
    tile_id: str,
    models: dict,
    config,
    scheduler,
    uni_embeds: torch.Tensor,
    device: str,
    exp_channels_dir: Path,
    guidance_scale: float,
    seed: int,
) -> list[tuple[str, np.ndarray]]:
    """Leave-one-out ablation: all_groups then all_minus_G for each group G.

    Returns [(label, gen_np), ...] with len = 1 + n_groups, fixed noise seed.
    """
    from tools.channel_group_utils import split_channels_to_groups
    from train_scripts.inference_controlnet import denoise, encode_ctrl_mask_latent

    active_channels = config.data.active_channels
    vae_scale = config.scale_factor
    vae_shift = config.shift_factor
    dtype = torch.float16 if device == "cuda" else torch.float32

    vae, controlnet, base_model, tme_module = (
        models["vae"], models["controlnet"], models["base_model"], models["tme_module"]
    )
    vae.to(device=device, dtype=dtype).eval()

    ctrl_full = load_exp_channels(tile_id, active_channels, config.image_size, exp_channels_dir)
    vae_mask = encode_ctrl_mask_latent(ctrl_full, vae, vae_shift=vae_shift, vae_scale=vae_scale,
                                       device=device, dtype=dtype)
    tme_dict = split_channels_to_groups(ctrl_full.unsqueeze(0).to(device, dtype=dtype),
                                        active_channels, config.channel_groups)
    group_names = [g["name"] for g in config.channel_groups]
    _zero_mask = getattr(config, "zero_mask_latent", False)

    torch.manual_seed(seed)
    latent_shape = (1, 16, config.image_size // 8, config.image_size // 8)
    fixed_noise = torch.randn(latent_shape, device=device, dtype=dtype) * scheduler.init_noise_sigma

    all_groups = set(group_names)
    conditions = [("All groups", all_groups)] + [
        (f"−{g}", all_groups - {g}) for g in group_names
    ]
    results = []
    for label, active in conditions:
        with torch.no_grad():
            fused = tme_module(vae_mask, tme_dict, active_groups=active)
            if _zero_mask:
                fused = fused - vae_mask
        denoised = denoise(latents=fixed_noise.clone(), uni_embeds=uni_embeds.to(device, dtype=dtype),
                           controlnet_input_latent=fused, scheduler=scheduler,
                           controlnet_model=controlnet, pixcell_controlnet_model=base_model,
                           guidance_scale=guidance_scale, device=device)
        with torch.no_grad():
            gen = vae.decode(((denoised.to(dtype) / vae_scale) + vae_shift), return_dict=False)[0]
        gen_np = ((gen / 2 + 0.5).clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()[0] * 255).astype(np.uint8)
        results.append((label, gen_np))
    return results


def generate_pairwise_ablation(
    tile_id: str,
    models: dict,
    config,
    scheduler,
    uni_embeds: torch.Tensor,
    device: str,
    exp_channels_dir: Path,
    guidance_scale: float,
    seed: int,
) -> list[tuple[str, np.ndarray]]:
    """Pairwise ablation: mask_only then mask+single_group for each group G.

    Returns [(label, gen_np), ...] with len = 1 + n_groups, fixed noise seed.
    """
    from tools.channel_group_utils import split_channels_to_groups
    from train_scripts.inference_controlnet import denoise, encode_ctrl_mask_latent

    active_channels = config.data.active_channels
    vae_scale = config.scale_factor
    vae_shift = config.shift_factor
    dtype = torch.float16 if device == "cuda" else torch.float32

    vae, controlnet, base_model, tme_module = (
        models["vae"], models["controlnet"], models["base_model"], models["tme_module"]
    )
    vae.to(device=device, dtype=dtype).eval()

    ctrl_full = load_exp_channels(tile_id, active_channels, config.image_size, exp_channels_dir)
    vae_mask = encode_ctrl_mask_latent(ctrl_full, vae, vae_shift=vae_shift, vae_scale=vae_scale,
                                       device=device, dtype=dtype)
    tme_dict = split_channels_to_groups(ctrl_full.unsqueeze(0).to(device, dtype=dtype),
                                        active_channels, config.channel_groups)
    group_names = [g["name"] for g in config.channel_groups]
    _zero_mask = getattr(config, "zero_mask_latent", False)

    torch.manual_seed(seed)
    latent_shape = (1, 16, config.image_size // 8, config.image_size // 8)
    fixed_noise = torch.randn(latent_shape, device=device, dtype=dtype) * scheduler.init_noise_sigma

    conditions = [("Mask only", None)] + [(f"+{g}", {g}) for g in group_names]
    results = []
    for label, active in conditions:
        with torch.no_grad():
            if active:
                fused = tme_module(vae_mask, tme_dict, active_groups=active)
                if _zero_mask:
                    fused = fused - vae_mask
            else:
                fused = torch.zeros_like(vae_mask) if _zero_mask else vae_mask.clone()
        denoised = denoise(latents=fixed_noise.clone(), uni_embeds=uni_embeds.to(device, dtype=dtype),
                           controlnet_input_latent=fused, scheduler=scheduler,
                           controlnet_model=controlnet, pixcell_controlnet_model=base_model,
                           guidance_scale=guidance_scale, device=device)
        with torch.no_grad():
            gen = vae.decode(((denoised.to(dtype) / vae_scale) + vae_shift), return_dict=False)[0]
        gen_np = ((gen / 2 + 0.5).clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()[0] * 255).astype(np.uint8)
        results.append((label, gen_np))
    return results


def find_latest_checkpoint_dir(checkpoints_parent: Path) -> Path:
    """Directory containing the newest controlnet_*.pth (by mtime)."""
    pths = sorted(
        checkpoints_parent.glob("controlnet_*.pth"),
        key=lambda p: p.stat().st_mtime,
    )
    if not pths:
        raise FileNotFoundError(f"No controlnet_*.pth under {checkpoints_parent}")
    return pths[-1].parent
