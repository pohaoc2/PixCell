"""Per-group residual magnitude visualization for MultiGroupTMEModule."""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


def compute_residual_maps(
    residuals: dict[str, torch.Tensor],
    output_resolution: int = 256,
) -> dict[str, np.ndarray]:
    maps = {}
    for name, delta in residuals.items():
        norm_map = delta[0].norm(dim=0)
        norm_up = F.interpolate(
            norm_map.unsqueeze(0).unsqueeze(0).float(),
            size=(output_resolution, output_resolution),
            mode="bilinear",
            align_corners=False,
        ).squeeze().cpu().numpy()
        maps[name] = norm_up
    return maps


def save_residual_magnitude_figure(
    mask_image: np.ndarray,
    gen_image: np.ndarray,
    residuals: dict[str, torch.Tensor],
    save_path: str | Path,
    output_resolution: int = 256,
):
    res_maps = compute_residual_maps(residuals, output_resolution)
    global_max = max(m.max() for m in res_maps.values()) if res_maps else 1.0

    n_panels = 2 + len(res_maps)
    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4))

    axes[0].imshow(mask_image)
    axes[0].set_title("Cell Mask", fontsize=10)
    axes[0].axis("off")

    axes[1].imshow(gen_image)
    axes[1].set_title("Generated H&E", fontsize=10)
    axes[1].axis("off")

    for i, (name, rmap) in enumerate(res_maps.items()):
        ax = axes[2 + i]
        im = ax.imshow(rmap, cmap="inferno", vmin=0, vmax=global_max)
        ax.set_title(f"‖Δ_{name}‖", fontsize=10)
        ax.axis("off")

    fig.colorbar(im, ax=axes[-1], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Residual magnitude maps saved → {save_path}")
