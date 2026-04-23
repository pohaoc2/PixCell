"""Visualize trained TME CNN encoder features for one experimental tile."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np
from PIL import Image
import torch
from torchvision import transforms as T

from diffusion.data.datasets.paired_exp_controlnet_dataset import PairedExpControlNetData
from diffusion.iddpm import IDDPM
from diffusion.model.builder import build_model
from diffusion.utils.misc import read_config
from tools.channel_group_utils import split_channels_to_groups
from train_scripts.exp_config_utils import resolve_exp_active_channels, resolve_exp_dataset_kwargs
from train_scripts.inference_controlnet import load_vae
from train_scripts.training_utils import load_tme_checkpoint


LATENT_CMAP = LinearSegmentedColormap.from_list(
    "latent_blue_black_red",
    [
        (0.00, "#1f4ed8"),
        (0.18, "#285fdf"),
        (0.43, "#071329"),
        (0.50, "#000000"),
        (0.57, "#2c0708"),
        (0.82, "#b2172b"),
        (1.00, "#e34a33"),
    ],
)


def _build_tme_module(config, checkpoint_dir: Path, device: torch.device) -> torch.nn.Module:
    channel_groups = getattr(config, "channel_groups", None)
    if channel_groups is None:
        raise ValueError("Expected config.channel_groups for MultiGroupTMEModule.")

    group_specs = [
        {"name": group["name"], "n_channels": len(group["channels"])}
        for group in channel_groups
    ]
    module = build_model(
        getattr(config, "tme_model", "MultiGroupTMEModule"),
        False,
        False,
        channel_groups=group_specs,
        base_ch=getattr(config, "tme_base_ch", 32),
    )
    load_tme_checkpoint(str(checkpoint_dir), module, device=device)
    module.to(device=device, dtype=torch.float32).eval()
    return module


def _load_tile_inputs(config, tile_id: str) -> tuple[list[str], torch.Tensor]:
    dataset_kwargs = resolve_exp_dataset_kwargs(config)
    dataset = PairedExpControlNetData(
        root=str(config.exp_data_root),
        resolution=int(config.image_size),
        **dataset_kwargs,
    )
    ctrl_tensor = dataset._build_ctrl_tensor(tile_id)
    active_channels = resolve_exp_active_channels(config)
    return active_channels, ctrl_tensor.unsqueeze(0)


def _find_reference_he(exp_data_root: Path, tile_id: str) -> Path:
    for ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
        candidate = exp_data_root / "he" / f"{tile_id}{ext}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No reference H&E image found for tile {tile_id!r} under {exp_data_root / 'he'}")


def _load_reference_uni(exp_data_root: Path, tile_id: str) -> np.ndarray:
    path = exp_data_root / "features" / f"{tile_id}_uni.npy"
    if not path.exists():
        raise FileNotFoundError(f"No cached UNI-2h embedding found: {path}")
    return np.load(path).astype(np.float32)


def _latent_norm(arr: np.ndarray) -> TwoSlopeNorm:
    return TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)


def _features_to_mean_map(features: np.ndarray) -> np.ndarray:
    """Collapse [16, 32, 32] CNN features to one [32, 32] mean feature map."""
    if features.ndim != 3:
        raise ValueError(f"Expected [C, H, W] features, got {features.shape}")
    return features.mean(axis=0)


def _style_dark_axis(ax) -> None:
    ax.set_facecolor("black")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#2f2f2f")
        spine.set_linewidth(0.8)


def _plot_input_channels(
    ctrl_tensor: torch.Tensor,
    active_channels: list[str],
    out_path: Path,
) -> None:
    planes = ctrl_tensor.squeeze(0).detach().cpu().numpy()
    fig, axes = plt.subplots(
        2,
        5,
        figsize=(12, 5.3),
        constrained_layout=True,
        facecolor="black",
    )
    for ax, channel, plane in zip(axes.ravel(), active_channels, planes):
        ax.imshow(plane, cmap="gray", vmin=0.0, vmax=1.0)
        ax.set_title(channel, fontsize=9, color="white")
        _style_dark_axis(ax)
    for ax in axes.ravel()[len(active_channels):]:
        ax.axis("off")
    fig.suptitle("TME input channels", fontsize=13, color="white")
    fig.savefig(out_path, dpi=220, facecolor=fig.get_facecolor())
    plt.close(fig)


def _plot_group_features(group_name: str, features: np.ndarray, out_path: Path) -> None:
    mean_map = _features_to_mean_map(features)
    fig = plt.figure(figsize=(4.0, 4.0), facecolor="black")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(mean_map, cmap=LATENT_CMAP, norm=_latent_norm(mean_map), interpolation="nearest")
    ax.set_axis_off()
    fig.savefig(out_path, dpi=320, facecolor=fig.get_facecolor(), pad_inches=0)
    plt.close(fig)


def _style_latent_colorbar(cbar) -> None:
    cbar.outline.set_edgecolor("white")
    cbar.outline.set_linewidth(0.6)
    cbar.ax.tick_params(colors="white", labelsize=8, width=0.6, length=2.5)
    cbar.set_label("Latent value", color="white", fontsize=8)


def _plot_latent_colormap_legend(out_path: Path) -> None:
    fig = plt.figure(figsize=(4.8, 0.50), facecolor="none")
    ax = fig.add_axes([0.08, 0.48, 0.84, 0.28])
    ax.set_facecolor("none")
    norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=LATENT_CMAP)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=ax, orientation="horizontal", ticks=[-1.0, 0.0, 1.0])
    cbar.outline.set_edgecolor("black")
    cbar.outline.set_linewidth(0.6)
    cbar.ax.tick_params(colors="black", labelsize=8, width=0.6, length=2.5)
    fig.savefig(out_path, dpi=320, transparent=True, pad_inches=0)
    plt.close(fig)


def _plot_single_latent_image(
    latent: np.ndarray,
    out_path: Path,
    *,
    with_colorbar: bool = False,
) -> None:
    mean_map = _features_to_mean_map(latent)
    norm = _latent_norm(mean_map)
    if with_colorbar:
        fig = plt.figure(figsize=(4.7, 4.0), facecolor="black")
        ax = fig.add_axes([0.0, 0.0, 0.84, 1.0])
        cax = fig.add_axes([0.89, 0.10, 0.035, 0.80])
    else:
        fig = plt.figure(figsize=(4.0, 4.0), facecolor="black")
        ax = fig.add_axes([0, 0, 1, 1])
        cax = None
    image = ax.imshow(mean_map, cmap=LATENT_CMAP, norm=norm, interpolation="nearest")
    ax.set_axis_off()
    if cax is not None:
        cbar = fig.colorbar(image, cax=cax)
        _style_latent_colorbar(cbar)
    fig.savefig(out_path, dpi=320, facecolor=fig.get_facecolor(), pad_inches=0)
    plt.close(fig)


def _sample_visible_noisy_latent(
    clean_latent: torch.Tensor,
    *,
    train_sampling_steps: int,
    seed: int = 145925632,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Sample q(x_t | x_0) at a high random timestep for visualization."""
    diffusion = IDDPM(str(train_sampling_steps))
    generator = torch.Generator(device=clean_latent.device).manual_seed(seed)
    low = int(train_sampling_steps * 0.70)
    high = int(train_sampling_steps * 0.90)
    timestep = int(torch.randint(low, high + 1, (1,), generator=generator).item())
    noise = torch.randn(
        clean_latent.shape,
        generator=generator,
        device=clean_latent.device,
        dtype=clean_latent.dtype,
    )
    t = torch.tensor([timestep], device=clean_latent.device, dtype=torch.long)
    noisy = diffusion.q_sample(clean_latent, t, noise=noise)
    return noisy, noise, t, timestep


def _plot_rgb_image(image_path: Path, out_path: Path) -> None:
    image = Image.open(image_path).convert("RGB")
    fig = plt.figure(figsize=(4.0, 4.0), facecolor="black")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(image)
    ax.set_axis_off()
    fig.savefig(out_path, dpi=320, facecolor=fig.get_facecolor(), pad_inches=0)
    plt.close(fig)


def _plot_uni_embedding(embedding: np.ndarray, out_path: Path) -> None:
    if embedding.shape != (1536,):
        raise ValueError(f"Expected UNI-2h embedding shape [1536], got {embedding.shape}")
    matrix = embedding.reshape(32, 48)
    fig = plt.figure(figsize=(6.0, 4.0), facecolor="black")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(matrix, cmap=LATENT_CMAP, norm=_latent_norm(matrix), interpolation="nearest", aspect="auto")
    ax.set_axis_off()
    fig.savefig(out_path, dpi=320, facecolor=fig.get_facecolor(), pad_inches=0)
    plt.close(fig)


@torch.no_grad()
def _encode_reference_he_with_sd35_vae(
    image_path: Path,
    vae,
    *,
    device: torch.device,
    dtype: torch.dtype,
    size: int,
) -> torch.Tensor:
    transform = T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB")),
            T.Resize((size, size)),
            T.ToTensor(),
            T.Normalize([0.5], [0.5]),
        ]
    )
    image = Image.open(image_path)
    x = transform(image).unsqueeze(0).to(device=device, dtype=dtype)
    return vae.encode(x).latent_dist.mean


@torch.no_grad()
def _encode_cell_mask_with_sd35_vae(
    ctrl_tensor: torch.Tensor,
    vae,
    *,
    vae_shift: float,
    vae_scale: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    cell_mask_img = ctrl_tensor[:, 0:1].repeat(1, 3, 1, 1)
    cell_mask_img = 2 * (cell_mask_img - 0.5)
    raw_mean = vae.encode(cell_mask_img.to(device=device, dtype=dtype)).latent_dist.mean
    scaled_latent = (raw_mean - vae_shift) * vae_scale
    return raw_mean, scaled_latent


def _plot_all_features(feature_dict: dict[str, np.ndarray], out_path: Path) -> None:
    group_names = list(feature_dict)
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(8.2, 8.4),
        constrained_layout=True,
        facecolor="black",
    )
    for ax, group_name in zip(axes.ravel(), group_names):
        mean_map = _features_to_mean_map(feature_dict[group_name])
        ax.imshow(mean_map, cmap=LATENT_CMAP, norm=_latent_norm(mean_map), interpolation="nearest")
        ax.set_title(group_name, fontsize=11, color="white")
        _style_dark_axis(ax)
    for ax in axes.ravel()[len(group_names):]:
        ax.axis("off")
    fig.suptitle("Mean TME CNN latent features", fontsize=14, color="white")
    fig.savefig(out_path, dpi=280, facecolor=fig.get_facecolor())
    plt.close(fig)


def extract_features(
    config_path: Path,
    checkpoint_dir: Path,
    tile_id: str,
    output_dir: Path,
    device_name: str,
) -> dict[str, object]:
    config = read_config(str(config_path))
    device = torch.device(device_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    tme_module = _build_tme_module(config, checkpoint_dir, device)
    active_channels, ctrl_tensor = _load_tile_inputs(config, tile_id)
    tme_dict = split_channels_to_groups(
        ctrl_tensor.to(device=device, dtype=torch.float32),
        active_channels,
        config.channel_groups,
    )

    features: dict[str, np.ndarray] = {}
    stats: dict[str, dict[str, object]] = {}
    with torch.no_grad():
        for group_name in tme_module.group_names:
            encoded = tme_module.groups[group_name].encoder(tme_dict[group_name])
            encoded_cpu = encoded.detach().float().cpu().squeeze(0).numpy()
            features[group_name] = encoded_cpu
            stats[group_name] = {
                "input_shape": list(tme_dict[group_name].shape),
                "cnn_output_shape": list(encoded.shape),
                "min": float(encoded_cpu.min()),
                "max": float(encoded_cpu.max()),
                "mean": float(encoded_cpu.mean()),
                "std": float(encoded_cpu.std()),
            }

    prefix = f"tme_cnn_features_{tile_id}"
    _plot_input_channels(
        ctrl_tensor,
        active_channels,
        output_dir / f"tme_input_channels_{tile_id}.png",
    )
    _plot_all_features(features, output_dir / f"{prefix}_all_groups.png")
    for group_name, group_features in features.items():
        _plot_group_features(group_name, group_features, output_dir / f"{prefix}_{group_name}.png")

    vae_dtype = torch.float16 if device.type == "cuda" else torch.float32
    vae = load_vae(config.vae_pretrained, device=str(device))
    vae.to(dtype=vae_dtype).eval()
    raw_cell_mask_mean, cell_mask_latent = _encode_cell_mask_with_sd35_vae(
        ctrl_tensor,
        vae,
        vae_shift=float(config.shift_factor),
        vae_scale=float(config.scale_factor),
        device=device,
        dtype=vae_dtype,
    )
    raw_cell_mask_mean_np = raw_cell_mask_mean.detach().float().cpu().squeeze(0).numpy()
    cell_mask_latent_np = cell_mask_latent.detach().float().cpu().squeeze(0).numpy()
    _plot_single_latent_image(
        cell_mask_latent_np,
        output_dir / f"cell_mask_sd35_vae_latent_{tile_id}.png",
    )
    _plot_latent_colormap_legend(output_dir / "latent_colormap_legend_blue_black_red.png")

    exp_data_root = Path(config.exp_data_root)
    reference_he_path = _find_reference_he(exp_data_root, tile_id)
    _plot_rgb_image(reference_he_path, output_dir / f"reference_he_{tile_id}.png")
    reference_he_vae_mean = _encode_reference_he_with_sd35_vae(
        reference_he_path,
        vae,
        device=device,
        dtype=vae_dtype,
        size=int(config.image_size),
    )
    reference_he_vae_mean_np = reference_he_vae_mean.detach().float().cpu().squeeze(0).numpy()
    _plot_single_latent_image(
        reference_he_vae_mean_np,
        output_dir / f"reference_he_sd35_vae_latent_{tile_id}.png",
    )
    reference_he_scaled_latent = (
        reference_he_vae_mean.detach().float() - float(config.shift_factor)
    ) * float(config.scale_factor)
    noisy_reference_he_latent, reference_he_noise, noisy_t, noisy_timestep = _sample_visible_noisy_latent(
        reference_he_scaled_latent,
        train_sampling_steps=int(config.train_sampling_steps),
    )
    noisy_reference_he_latent_np = noisy_reference_he_latent.detach().float().cpu().squeeze(0).numpy()
    reference_he_noise_np = reference_he_noise.detach().float().cpu().squeeze(0).numpy()
    _plot_single_latent_image(
        noisy_reference_he_latent_np,
        output_dir / f"reference_he_sd35_vae_latent_noisy_{tile_id}.png",
    )

    reference_uni = _load_reference_uni(exp_data_root, tile_id)
    _plot_uni_embedding(reference_uni, output_dir / f"reference_he_uni2h_embedding_{tile_id}.png")
    np.savez_compressed(
        output_dir / f"reference_he_features_{tile_id}.npz",
        sd35_vae_mean=reference_he_vae_mean_np,
        sd35_scaled_latent=reference_he_scaled_latent.detach().float().cpu().squeeze(0).numpy(),
        sd35_noisy_latent=noisy_reference_he_latent_np,
        sd35_noise=reference_he_noise_np,
        sd35_noisy_timestep=np.asarray([noisy_timestep], dtype=np.int64),
        uni2h_embedding=reference_uni,
    )
    reference_he_stats = {
        "image_path": str(reference_he_path),
        "sd35_vae_mean_shape": list(reference_he_vae_mean.shape),
        "sd35_vae_mean_min": float(reference_he_vae_mean_np.min()),
        "sd35_vae_mean_max": float(reference_he_vae_mean_np.max()),
        "sd35_vae_mean_mean": float(reference_he_vae_mean_np.mean()),
        "sd35_vae_mean_std": float(reference_he_vae_mean_np.std()),
        "forward_diffusion_timestep": noisy_timestep,
        "forward_diffusion_num_timesteps": int(config.train_sampling_steps),
        "sd35_noisy_latent_shape": list(noisy_reference_he_latent.shape),
        "sd35_noisy_latent_min": float(noisy_reference_he_latent_np.min()),
        "sd35_noisy_latent_max": float(noisy_reference_he_latent_np.max()),
        "sd35_noisy_latent_mean": float(noisy_reference_he_latent_np.mean()),
        "sd35_noisy_latent_std": float(noisy_reference_he_latent_np.std()),
        "uni2h_embedding_shape": list(reference_uni.shape),
        "uni2h_embedding_min": float(reference_uni.min()),
        "uni2h_embedding_max": float(reference_uni.max()),
        "uni2h_embedding_mean": float(reference_uni.mean()),
        "uni2h_embedding_std": float(reference_uni.std()),
    }
    with (output_dir / f"reference_he_features_{tile_id}_summary.json").open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "tile_id": tile_id,
                "image_path": str(reference_he_path),
                "sd35_encoder": "SD3.5 AutoencoderKL latent_dist.mean",
                "uni_encoder": "cached UNI-2h embedding from data/orion-crc33/features",
                "summary": reference_he_stats,
            },
            f,
            indent=2,
        )

    np.savez_compressed(output_dir / f"{prefix}.npz", **features)
    np.savez_compressed(
        output_dir / f"cell_mask_sd35_vae_latent_{tile_id}.npz",
        raw_mean=raw_cell_mask_mean_np,
        scaled_latent=cell_mask_latent_np,
    )
    cell_mask_stats = {
        "input_shape": list(ctrl_tensor[:, 0:1].shape),
        "raw_vae_mean_shape": list(raw_cell_mask_mean.shape),
        "scaled_vae_latent_shape": list(cell_mask_latent.shape),
        "raw_mean_min": float(raw_cell_mask_mean_np.min()),
        "raw_mean_max": float(raw_cell_mask_mean_np.max()),
        "raw_mean_mean": float(raw_cell_mask_mean_np.mean()),
        "raw_mean_std": float(raw_cell_mask_mean_np.std()),
        "min": float(cell_mask_latent_np.min()),
        "max": float(cell_mask_latent_np.max()),
        "mean": float(cell_mask_latent_np.mean()),
        "std": float(cell_mask_latent_np.std()),
    }
    with (output_dir / f"cell_mask_sd35_vae_latent_{tile_id}_summary.json").open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "tile_id": tile_id,
                "encoder": "SD3.5 AutoencoderKL",
                "vae_pretrained": str(config.vae_pretrained),
                "input_channel": "cell_masks",
                "latent_convention": "(vae_mean - shift_factor) * scale_factor",
                "scale_factor": float(config.scale_factor),
                "shift_factor": float(config.shift_factor),
                "summary": cell_mask_stats,
            },
            f,
            indent=2,
        )
    payload = {
        "tile_id": tile_id,
        "config": str(config_path),
        "checkpoint_dir": str(checkpoint_dir),
        "active_channels": active_channels,
        "cnn_output_summary": stats,
        "cell_mask_sd35_vae_summary": cell_mask_stats,
        "reference_he_summary": reference_he_stats,
    }
    with (output_dir / f"{prefix}_summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config_controlnet_exp.py", type=Path)
    parser.add_argument(
        "--checkpoint-dir",
        default="checkpoints/pixcell_controlnet_exp/npy_inputs",
        type=Path,
    )
    parser.add_argument("--tile-id", default="14592_5632")
    parser.add_argument("--output-dir", default="paper/figures/stage2", type=Path)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device for feature extraction.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = extract_features(
        config_path=args.config,
        checkpoint_dir=args.checkpoint_dir,
        tile_id=args.tile_id,
        output_dir=args.output_dir,
        device_name=args.device,
    )
    print(json.dumps(summary["cnn_output_summary"], indent=2))


if __name__ == "__main__":
    main()
