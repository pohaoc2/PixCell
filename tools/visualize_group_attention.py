"""Per-group attention heatmap visualization for MultiGroupTMEModule."""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


def compute_attention_heatmaps(
    attn_maps: dict[str, torch.Tensor],
    spatial_size: tuple[int, int] = (32, 32),
    output_resolution: int = 256,
) -> dict[str, np.ndarray]:
    H, W = spatial_size
    heatmaps = {}
    for name, weights in attn_maps.items():
        avg = weights.mean(dim=(0, 1))
        importance = avg.sum(dim=0)
        hmap = importance.reshape(H, W).cpu().numpy()
        hmap = (hmap - hmap.min()) / (hmap.max() - hmap.min() + 1e-8)
        hmap_up = F.interpolate(
            torch.from_numpy(hmap).unsqueeze(0).unsqueeze(0).float(),
            size=(output_resolution, output_resolution),
            mode="bilinear",
            align_corners=False,
        ).squeeze().numpy()
        heatmaps[name] = hmap_up
    return heatmaps


def save_attention_heatmap_figure(
    mask_image: np.ndarray,
    gen_image: np.ndarray,
    attn_maps: dict[str, torch.Tensor],
    save_path: str | Path,
    spatial_size: tuple[int, int] = (32, 32),
    output_resolution: int = 256,
):
    heatmaps = compute_attention_heatmaps(attn_maps, spatial_size, output_resolution)
    n_panels = 2 + len(heatmaps)
    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4))

    axes[0].imshow(mask_image)
    axes[0].set_title("Cell Mask", fontsize=10)
    axes[0].axis("off")

    axes[1].imshow(gen_image)
    axes[1].set_title("Generated H&E", fontsize=10)
    axes[1].axis("off")

    for i, (name, hmap) in enumerate(heatmaps.items()):
        ax = axes[2 + i]
        ax.imshow(mask_image, alpha=0.3)
        im = ax.imshow(hmap, cmap="jet", alpha=0.7, vmin=0, vmax=1)
        ax.set_title(f"{name} attn", fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Attention heatmaps saved → {save_path}")
