"""
Shared Stage 3 inference helpers: load exp channels, models, generate one tile, ablation sweep.

Used by tools/stage3/run_evaluation.py and tools/vis/generate_stage3_tile_vis.py.
"""
from __future__ import annotations

import glob as _glob
import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from diffusion.data.datasets.sim_controlnet_dataset import (
    _find_file,
    _load_spatial_file,
    get_channel_load_config,
    resolve_channel_dir as resolve_channel_dir_shared,
)
from tools.channel_group_utils import split_channels_to_groups
from tools.stage3.ablation import (
    AblationCondition,
    build_progressive_conditions,
    build_progressive_order_conditions,
    build_subset_conditions,
    group_names_from_channel_groups,
)
from tools.stage3.common import inference_dtype


_MIRROR_BORDER_PX: int = 8  # reflect-pad non-binary channels before resize
_TILE_ID_EXTS: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tif", ".npy")


def _prepare_generation_context(
    ctrl_full: torch.Tensor,
    *,
    models: dict,
    config,
    uni_embeds: torch.Tensor,
    device: str,
) -> dict:
    """Prepare reusable Stage 3 generation context from a full control tensor."""
    from train_scripts.inference_controlnet import encode_ctrl_mask_latent

    active_channels = config.data.active_channels
    vae_scale = config.scale_factor
    vae_shift = config.shift_factor
    dtype = inference_dtype(device)

    vae = models["vae"]
    controlnet = models["controlnet"]
    base_model = models["base_model"]
    tme_module = models["tme_module"]

    vae.to(device=device, dtype=dtype).eval()

    vae_mask = encode_ctrl_mask_latent(
        ctrl_full,
        vae,
        vae_shift=vae_shift,
        vae_scale=vae_scale,
        device=device,
        dtype=dtype,
    )
    channel_groups_cfg = getattr(config, "channel_groups", None)
    if channel_groups_cfg is not None:
        tme_inputs = split_channels_to_groups(
            ctrl_full.unsqueeze(0).to(device, dtype=dtype),
            active_channels,
            channel_groups_cfg,
        )
    else:
        tme_inputs = ctrl_full[1:].unsqueeze(0).to(device, dtype=dtype)

    return dict(
        active_channels=active_channels,
        vae_scale=vae_scale,
        vae_shift=vae_shift,
        dtype=dtype,
        vae=vae,
        controlnet=controlnet,
        base_model=base_model,
        tme_module=tme_module,
        ctrl_full=ctrl_full,
        vae_mask=vae_mask,
        tme_inputs=tme_inputs,
        uni_embeds=uni_embeds.to(device, dtype=dtype),
        zero_mask_latent=getattr(config, "zero_mask_latent", False),
        channel_groups_cfg=channel_groups_cfg,
    )


def _prepare_ablation_context(
    tile_id: str,
    models: dict,
    config,
    uni_embeds: torch.Tensor,
    device: str,
    exp_channels_dir: Path,
) -> dict:
    """Load inputs once for repeated ablation renders on the same tile."""
    ctrl_full = load_exp_channels(tile_id, config.data.active_channels, config.image_size, exp_channels_dir)
    return _prepare_generation_context(
        ctrl_full,
        models=models,
        config=config,
        uni_embeds=uni_embeds,
        device=device,
    )


def _make_fixed_noise(
    *,
    config,
    scheduler,
    device: str,
    dtype: torch.dtype,
    seed: int,
) -> torch.Tensor:
    """Fixed initial noise so every ablation step shares the same denoising start."""
    latent_shape = (1, 16, config.image_size // 8, config.image_size // 8)
    torch.manual_seed(seed)
    fixed_noise = torch.randn(latent_shape, device=device, dtype=dtype)
    return fixed_noise * scheduler.init_noise_sigma


def _decode_latents_to_image(
    denoised: torch.Tensor,
    *,
    vae,
    vae_scale: float,
    vae_shift: float,
    dtype: torch.dtype,
) -> np.ndarray:
    """Decode denoised latents into a uint8 RGB tile."""
    with torch.no_grad():
        scaled = (denoised.to(dtype) / vae_scale) + vae_shift
        gen = vae.decode(scaled, return_dict=False)[0]
    gen = (gen / 2 + 0.5).clamp(0, 1)
    return (gen.cpu().permute(0, 2, 3, 1).numpy()[0] * 255).astype(np.uint8)


def _render_fused_ablation_image(
    fused: torch.Tensor,
    *,
    context: dict,
    scheduler,
    guidance_scale: float,
    device: str,
    seed: int,
    fixed_noise: torch.Tensor,
) -> np.ndarray:
    """Run denoising + VAE decode for one already-fused conditioning latent."""
    from train_scripts.inference_controlnet import denoise

    torch.manual_seed(seed)  # identical scheduler-step noise across all ablation steps
    denoised = denoise(
        latents=fixed_noise.clone(),
        uni_embeds=context["uni_embeds"],
        controlnet_input_latent=fused,
        scheduler=scheduler,
        controlnet_model=context["controlnet"],
        pixcell_controlnet_model=context["base_model"],
        guidance_scale=guidance_scale,
        device=device,
    )
    return _decode_latents_to_image(
        denoised,
        vae=context["vae"],
        vae_scale=context["vae_scale"],
        vae_shift=context["vae_shift"],
        dtype=context["dtype"],
    )


def _fuse_active_groups(
    *,
    context: dict,
    active_groups: Sequence[str],
) -> torch.Tensor:
    """Create the ControlNet conditioning latent for one group subset."""
    vae_mask = context["vae_mask"]
    zero_mask_latent = context["zero_mask_latent"]

    with torch.no_grad():
        if active_groups:
            fused, _, _ = _fuse_conditioning_latent(
                context,
                active_groups=active_groups,
            )
        else:
            fused = torch.zeros_like(vae_mask) if zero_mask_latent else vae_mask.clone()
    return fused


def _fuse_conditioning_latent(
    context: dict,
    *,
    active_groups: Sequence[str] | set[str] | None = None,
    include_residuals: bool = False,
    include_attn_maps: bool = False,
) -> tuple[torch.Tensor, dict, dict]:
    """Fuse control channels into the conditioning latent for one generation pass."""
    fused: torch.Tensor
    residuals: dict = {}
    attn_maps: dict = {}
    vae_mask = context["vae_mask"]
    tme_module = context["tme_module"]
    tme_inputs = context["tme_inputs"]
    active_group_set = None if active_groups is None else set(active_groups)

    with torch.no_grad():
        if context["channel_groups_cfg"] is None:
            if include_residuals:
                fused, residuals = tme_module(vae_mask, tme_inputs, return_residuals=True)
            else:
                fused = tme_module(vae_mask, tme_inputs)
        elif include_attn_maps:
            fused, residuals = tme_module(
                vae_mask,
                tme_inputs,
                active_groups=active_group_set,
                return_residuals=True,
            )
            _, _, attn_maps = tme_module(
                vae_mask,
                tme_inputs,
                active_groups=active_group_set,
                return_residuals=True,
                return_attn_weights=True,
            )
        elif include_residuals:
            fused, residuals = tme_module(
                vae_mask,
                tme_inputs,
                active_groups=active_group_set,
                return_residuals=True,
            )
        else:
            fused = tme_module(
                vae_mask,
                tme_inputs,
                active_groups=active_group_set,
            )

    if context["zero_mask_latent"]:
        fused = fused - vae_mask
    return fused, residuals, attn_maps


def load_channel(
    ch_dir: Path,
    tile_id: str,
    resolution: int,
    binary: bool,
    *,
    channel_name: str | None = None,
    mirror_border_px_nonbinary: int = _MIRROR_BORDER_PX,
) -> np.ndarray:
    """Load a single channel PNG → float32 [H, W] in [0, 1].

    Non-binary (continuous) channels are reflect-padded by _MIRROR_BORDER_PX pixels
    before resize to suppress simulation boundary artifacts.
    """
    load_cfg = get_channel_load_config(channel_name or ch_dir.name)
    fpath = _find_file(ch_dir, tile_id, exts=load_cfg["preferred_exts"])
    normalization = str(load_cfg["normalization"])
    return _load_spatial_file(
        fpath,
        resolution=resolution,
        binary=binary,
        mirror_border_px=0 if binary or normalization == "clip01" else mirror_border_px_nonbinary,
        normalization=normalization,
    )


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


def list_tile_ids_from_exp_channels(exp_channels_dir: Path) -> list[str]:
    """Tile IDs discovered from the cell-mask channel directory."""
    mask_dir = resolve_channel_dir(exp_channels_dir, "cell_masks")
    if not mask_dir.is_dir():
        raise FileNotFoundError(
            f"No cell_masks/ or cell_mask/ directory found under {exp_channels_dir}"
        )
    tile_ids = sorted(
        p.stem for p in mask_dir.iterdir()
        if p.is_file() and p.suffix.lower() in _TILE_ID_EXTS
    )
    if not tile_ids:
        raise FileNotFoundError(f"No tile files found in {mask_dir}")
    return tile_ids


def resolve_channel_dir(exp_channels_dir: Path, channel_name: str) -> Path:
    """Prefer config name; fall back to alias dirs (cell_masks → cell_mask)."""
    return resolve_channel_dir_shared(exp_channels_dir, channel_name)


def load_channels(
    tile_id: str,
    active_channels: list[str],
    resolution: int,
    channels_dir: Path,
    *,
    mirror_border_px_nonbinary: int,
    binary_channels: frozenset[str] | None = None,
) -> torch.Tensor:
    """Load active channels from a per-channel directory tree → [C, H, W]."""
    planes = []
    for ch in active_channels:
        load_cfg = get_channel_load_config(ch)
        binary = bool(load_cfg["binary"]) if binary_channels is None else ch in binary_channels
        ch_dir = resolve_channel_dir(channels_dir, ch)
        arr = load_channel(
            ch_dir,
            tile_id,
            resolution,
            binary=binary,
            channel_name=ch,
            mirror_border_px_nonbinary=mirror_border_px_nonbinary,
        )
        planes.append(arr)
    return torch.from_numpy(np.stack(planes, axis=0))


def load_exp_channels(
    tile_id: str,
    active_channels: list[str],
    resolution: int,
    exp_channels_dir: Path,
    binary_channels: frozenset[str] | None = None,
) -> torch.Tensor:
    """Load active channels from paired experimental inputs → [C, H, W]."""
    return load_channels(
        tile_id,
        active_channels,
        resolution,
        exp_channels_dir,
        mirror_border_px_nonbinary=_MIRROR_BORDER_PX,
        binary_channels=binary_channels,
    )


def load_sim_channels(
    sim_id: str,
    active_channels: list[str],
    resolution: int,
    sim_channels_dir: Path,
    binary_channels: frozenset[str] | None = None,
) -> torch.Tensor:
    """Load active channels from sim-style inputs without non-binary mirror padding."""
    return load_channels(
        sim_id,
        active_channels,
        resolution,
        sim_channels_dir,
        mirror_border_px_nonbinary=0,
        binary_channels=binary_channels,
    )


def load_all_models(
    config,
    config_path: str | Path | None,
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

    config_path = None if config_path is None else str(config_path)
    ckpt_dir = str(ckpt_dir)

    print("Loading VAE...")
    vae = load_vae(config.vae_pretrained, device)

    print("Loading TME module...")
    channel_groups_cfg = getattr(config, "channel_groups", None)
    if channel_groups_cfg is not None:
        group_specs = [
            dict(name=g["name"], n_channels=len(g["channels"]))
            for g in channel_groups_cfg
        ]
        tme_module = build_model(
            "MultiGroupTMEModule",
            False,
            False,
            channel_groups=group_specs,
            base_ch=getattr(config, "tme_base_ch", 32),
        )
    else:
        n_tme_channels = len(config.data.active_channels) - 1
        tme_module = build_model(
            "TMEConditioningModule",
            False,
            False,
            n_tme_channels=n_tme_channels,
            base_ch=getattr(config, "tme_base_ch", 32),
        )
    load_tme_checkpoint(ckpt_dir, tme_module, device=device)
    dtype = inference_dtype(device)
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


def generate_from_ctrl(
    ctrl_full: torch.Tensor,
    *,
    models: dict,
    config,
    scheduler,
    uni_embeds: torch.Tensor,
    device: str,
    guidance_scale: float,
    return_vis_data: bool = False,
    include_residuals: bool = False,
    include_attn_maps: bool = False,
    seed: int | None = None,
    fixed_noise: torch.Tensor | None = None,
    active_groups: Sequence[str] | set[str] | None = None,
    denoise_fn=None,
):
    """Generate H&E from a prepared full control tensor."""
    from train_scripts.inference_controlnet import denoise as default_denoise

    if include_attn_maps:
        include_residuals = True

    context = _prepare_generation_context(
        ctrl_full,
        models=models,
        config=config,
        uni_embeds=uni_embeds,
        device=device,
    )
    fused, residuals, attn_maps = _fuse_conditioning_latent(
        context,
        active_groups=active_groups,
        include_residuals=include_residuals,
        include_attn_maps=include_attn_maps,
    )
    dtype = context["dtype"]

    if fixed_noise is None:
        latent_shape = (1, 16, config.image_size // 8, config.image_size // 8)
        if seed is not None:
            torch.manual_seed(seed)
        latents = torch.randn(latent_shape, device=device, dtype=dtype)
        latents = latents * scheduler.init_noise_sigma
    else:
        latents = fixed_noise.clone()

    if seed is not None:
        torch.manual_seed(seed)
    denoise_impl = default_denoise if denoise_fn is None else denoise_fn
    denoised = denoise_impl(
        latents=latents,
        uni_embeds=context["uni_embeds"],
        controlnet_input_latent=fused,
        scheduler=scheduler,
        controlnet_model=context["controlnet"],
        pixcell_controlnet_model=context["base_model"],
        guidance_scale=guidance_scale,
        device=device,
    )

    gen_np = _decode_latents_to_image(
        denoised,
        vae=context["vae"],
        vae_scale=context["vae_scale"],
        vae_shift=context["vae_shift"],
        dtype=dtype,
    )

    vis_data = None
    if return_vis_data:
        vis_data = {
            "ctrl_full": ctrl_full.numpy(),
            "active_channels": context["active_channels"],
        }
        if include_residuals:
            vis_data["residuals"] = residuals
        if include_attn_maps:
            vis_data["attn_maps"] = attn_maps

    return gen_np, vis_data


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
    include_residuals: bool = False,
    include_attn_maps: bool = False,
    seed: int | None = None,
):
    """
    Generate H&E for one tile from its exp channels.

    Returns:
        gen_np: uint8 [H, W, 3]
        vis_data: dict with ctrl_full/active_channels plus optional residuals/attn_maps.
    """
    ctrl_full = load_exp_channels(tile_id, config.data.active_channels, config.image_size, exp_channels_dir)
    return generate_from_ctrl(
        ctrl_full,
        models=models,
        config=config,
        scheduler=scheduler,
        uni_embeds=uni_embeds,
        device=device,
        guidance_scale=guidance_scale,
        return_vis_data=return_vis_data,
        include_residuals=include_residuals,
        include_attn_maps=include_attn_maps,
        seed=seed,
    )


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
    conditions: Sequence[AblationCondition] | None = None,
) -> list[tuple[str, np.ndarray]]:
    """Return list of (label, gen_np) for any requested group-ablation conditions."""
    context = _prepare_ablation_context(
        tile_id=tile_id,
        models=models,
        config=config,
        uni_embeds=uni_embeds,
        device=device,
        exp_channels_dir=exp_channels_dir,
    )
    if conditions is None:
        conditions = build_progressive_conditions(
            group_names_from_channel_groups(config.channel_groups),
            zero_mask_latent=context["zero_mask_latent"],
        )

    fixed_noise = _make_fixed_noise(
        config=config,
        scheduler=scheduler,
        device=device,
        dtype=context["dtype"],
        seed=seed,
    )

    ablation_images: list[tuple[str, np.ndarray]] = []
    for condition in conditions:
        fused = _fuse_active_groups(
            context=context,
            active_groups=condition.active_groups,
        )
        gen_np = _render_fused_ablation_image(
            fused,
            context=context,
            scheduler=scheduler,
            guidance_scale=guidance_scale,
            device=device,
            seed=seed,
            fixed_noise=fixed_noise,
        )
        ablation_images.append((condition.label, gen_np))

    return ablation_images


def generate_group_combination_ablation_images(
    tile_id: str,
    models: dict,
    config,
    scheduler,
    uni_embeds: torch.Tensor,
    device: str,
    exp_channels_dir: Path,
    guidance_scale: float,
    seed: int,
    subset_size: int,
) -> list[tuple[str, np.ndarray]]:
    """Render every size-k group combination as a standalone ablation condition."""
    conditions = build_subset_conditions(
        group_names_from_channel_groups(config.channel_groups),
        subset_size=subset_size,
    )
    return generate_ablation_images(
        tile_id=tile_id,
        models=models,
        config=config,
        scheduler=scheduler,
        uni_embeds=uni_embeds,
        device=device,
        exp_channels_dir=exp_channels_dir,
        guidance_scale=guidance_scale,
        seed=seed,
        conditions=conditions,
    )


def generate_progressive_order_ablation_images(
    tile_id: str,
    models: dict,
    config,
    scheduler,
    uni_embeds: torch.Tensor,
    device: str,
    exp_channels_dir: Path,
    guidance_scale: float,
    seed: int,
    group_order: Sequence[str],
) -> list[tuple[str, np.ndarray]]:
    """Render one progressive cumulative sweep for a specific group order."""
    conditions = build_progressive_conditions(
        group_order,
        zero_mask_latent=getattr(config, "zero_mask_latent", False),
    )
    return generate_ablation_images(
        tile_id=tile_id,
        models=models,
        config=config,
        scheduler=scheduler,
        uni_embeds=uni_embeds,
        device=device,
        exp_channels_dir=exp_channels_dir,
        guidance_scale=guidance_scale,
        seed=seed,
        conditions=conditions,
    )


def generate_all_progressive_order_ablation_images(
    tile_id: str,
    models: dict,
    config,
    scheduler,
    uni_embeds: torch.Tensor,
    device: str,
    exp_channels_dir: Path,
    guidance_scale: float,
    seed: int,
) -> list[tuple[tuple[str, ...], list[tuple[str, np.ndarray]]]]:
    """Render all 24 progressive cumulative sweeps across every group order."""
    order_conditions = build_progressive_order_conditions(
        group_names_from_channel_groups(config.channel_groups),
        zero_mask_latent=getattr(config, "zero_mask_latent", False),
    )
    return [
        (
            group_order,
            generate_ablation_images(
                tile_id=tile_id,
                models=models,
                config=config,
                scheduler=scheduler,
                uni_embeds=uni_embeds,
                device=device,
                exp_channels_dir=exp_channels_dir,
                guidance_scale=guidance_scale,
                seed=seed,
                conditions=conditions,
            ),
        )
        for group_order, conditions in order_conditions
    ]


def find_latest_checkpoint_dir(checkpoints_parent: Path) -> Path:
    """Directory containing the newest controlnet_*.pth (by mtime)."""
    pths = sorted(
        checkpoints_parent.glob("controlnet_*.pth"),
        key=lambda p: p.stat().st_mtime,
    )
    if not pths:
        raise FileNotFoundError(f"No controlnet_*.pth under {checkpoints_parent}")
    return pths[-1].parent
