"""
Visualize histology and segmentation PNG images side by side.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2


def vis_hist_seg(
    hist_path: str,
    seg_path: str,
    *,
    alpha: float = 0.5,
    figsize: tuple[float, float] = (12, 6),
    save_path: str = None,
    dpi: int = 150,
) -> None:
    """
    Visualize histology and segmentation PNG images.

    Args:
        hist_path: Path to the histology PNG image.
        seg_path: Path to the segmentation PNG image.
        alpha: Transparency of segmentation overlay in the third panel (0=transparent, 1=opaque).
        figsize: Figure size (width, height) in inches.
        save_path: If set, save the figure to this path.
        dpi: DPI for saved figure when save_path is set.
    """
    hist_path = Path(hist_path)
    seg_path = Path(seg_path)

    if not hist_path.exists():
        raise FileNotFoundError(f"Histology image not found: {hist_path}")
    if not seg_path.exists():
        raise FileNotFoundError(f"Segmentation image not found: {seg_path}")
    hist_img = cv2.imread(str(hist_path))
    hist_img = cv2.cvtColor(hist_img, cv2.COLOR_BGR2RGB)
    seg_img = cv2.imread(str(seg_path))
    seg_img = cv2.cvtColor(seg_img, cv2.COLOR_BGR2RGB)

    # Normalize seg for display: if multi-channel or multi-class, use first channel or colormap
    if seg_img.ndim == 3:
        seg_display = seg_img[:, :, 0] if seg_img.shape[2] >= 1 else seg_img[:, :, 0]
    else:
        seg_display = seg_img

    # Use a discrete colormap for segmentation (e.g. viridis or tab20 for many classes)
    n_unique = len(np.unique(seg_display))
    cmap = "gray" #"tab20" if n_unique <= 20 else "viridis"

    # Three-panel layout: hist, seg, and overlay in ax[2]
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # ax[0]: histology
    axes[0].imshow(hist_img)
    axes[0].set_title("Histology")
    axes[0].axis("off")

    # ax[1]: segmentation
    im1 = axes[1].imshow(seg_display, cmap=cmap)
    axes[1].set_title("Segmentation")
    axes[1].axis("off")

    # ax[2]: histology with segmentation contours drawn on top
    overlay = hist_img.copy()
    unique_labels = [v for v in np.unique(seg_display) if v != 0]
    contour_colors = plt.cm.tab20(np.linspace(0, 1, max(len(unique_labels), 1)))
    for i, label in enumerate(unique_labels):
        binary = (seg_display == label).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        color_rgb = (255, 0, 0)#tuple(int(c * 255) for c in contour_colors[i][:3])
        cv2.drawContours(overlay, contours, -1, color_rgb, thickness=1)

    axes[2].imshow(overlay)
    axes[2].set_title("Histology + Segmentation contours")
    axes[2].axis("off")

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def main():
    for i in range(10):
        try:
            idx = np.random.randint(0, 3660)
            print(f"Processing image {idx}")
            hist_path = f"../data/tcga_subset_3660/png_patches/0_{idx}.png"
            seg_path = f"../data/tcga_subset_3660/0_tcga_subset_3660_output/0_{idx}/masks_output/type_mask_vis.png"
            vis_hist_seg(hist_path, seg_path)
        except Exception as e:
            print(f"Error processing image {idx}: {e}")
            continue
if __name__ == "__main__":
    main()


    