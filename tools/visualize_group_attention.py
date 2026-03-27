"""Per-group attention heatmap visualization for MultiGroupTMEModule."""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


def _upsample(hmap: np.ndarray, output_resolution: int) -> np.ndarray:
    return F.interpolate(
        torch.from_numpy(hmap).unsqueeze(0).unsqueeze(0).float(),
        size=(output_resolution, output_resolution),
        mode="bilinear",
        align_corners=False,
    ).squeeze().numpy()


def compute_attention_heatmaps(
    attn_maps: dict[str, torch.Tensor],
    spatial_size: tuple[int, int] = (32, 32),
    output_resolution: int = 256,
) -> dict[str, np.ndarray]:
    """KV-space heatmap (sum over Q): which TME positions were consulted."""
    H, W = spatial_size
    heatmaps = {}
    for name, weights in attn_maps.items():
        avg = weights.mean(dim=(0, 1))          # [Q_len, KV_len]
        importance = avg.sum(dim=0)             # [KV_len] — sum over Q
        hmap = importance.reshape(H, W).cpu().numpy()
        hmap = (hmap - hmap.min()) / (hmap.max() - hmap.min() + 1e-8)
        heatmaps[name] = _upsample(hmap, output_resolution)
    return heatmaps


def compute_attention_heatmaps_dual(
    attn_maps: dict[str, torch.Tensor],
    spatial_size: tuple[int, int] = (32, 32),
    output_resolution: int = 256,
) -> dict[str, dict[str, np.ndarray]]:
    """Return both heatmap variants for each group.

    Returns:
        {group_name: {"tme_space": arr, "mask_space": arr}}

    tme_space  (sum over Q, dim=0): which TME/KV positions were consulted.
        → Heatmap lives in TME channel coordinates.
    mask_space (sum over KV, dim=1): which mask/query positions were seeking info.
        → Heatmap lives in cell mask coordinates; directly overlayable on H&E.
    """
    H, W = spatial_size
    result = {}
    for name, weights in attn_maps.items():
        avg = weights.mean(dim=(0, 1))          # [Q_len, KV_len]

        # TME space: which KV positions were most consulted
        tme = avg.sum(dim=0).reshape(H, W).cpu().numpy()
        tme = (tme - tme.min()) / (tme.max() - tme.min() + 1e-8)

        # Mask space: attention sharpness per query position.
        # avg.sum(dim=1) is trivially ~1 due to softmax normalisation, so we use
        # normalised entropy instead: high sharpness = mask token knows exactly
        # which TME region to attend to; low = diffuse / confused.
        eps = 1e-8
        H_q = -(avg * torch.log(avg.clamp(min=eps))).sum(dim=1)   # [Q_len] entropy
        H_max = torch.log(torch.tensor(float(H * W)))               # log(1024)
        sharpness = ((H_max - H_q) / (H_max + eps)).clamp(0, 1)    # [Q_len] in [0,1]
        mask = sharpness.reshape(H, W).cpu().float().numpy()
        mask = (mask - mask.min()) / (mask.max() - mask.min() + eps)

        result[name] = {
            "tme_space":  _upsample(tme,  output_resolution),
            "mask_space": _upsample(mask, output_resolution),
        }
    return result


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
