"""
Visualization helpers for pretrained PixCell inference checks.
"""

from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def load_rgb_image(image_path, resolution=256):
    """Load an RGB image and resize it for side-by-side comparison."""
    return np.array(
        Image.open(image_path)
        .convert("RGB")
        .resize((resolution, resolution), Image.Resampling.NEAREST)
    )


def make_contour_overlay(gen_img, mask_path, resolution=256, thickness=1):
    """Draw yellow cell contours from the mask onto the generated image."""
    mask = np.array(
        Image.open(mask_path)
        .convert("L")
        .resize((resolution, resolution), Image.Resampling.NEAREST)
    )
    mask_bin = (mask > 127).astype(np.uint8)

    num_labels, labeled = cv2.connectedComponents(mask_bin)
    overlay = gen_img.copy()
    for label_id in range(1, num_labels):
        cell_mask = (labeled == label_id).astype(np.uint8)
        contours, _ = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color=(255, 255, 0), thickness=thickness)
    return overlay


def save_comparison_figure(
    mask_path,
    gen_img,
    save_path,
    reference_he_path=None,
    resolution=256,
    contour_thickness=1,
):
    """
    Save a 3-panel or 4-panel comparison figure.

    If `reference_he_path` is provided, the layout is:
    Reference H&E | Mask | Generated H&E | Generated + Contours
    Otherwise, the layout is:
    Mask | Generated H&E | Generated + Contours
    """
    mask_display = load_rgb_image(mask_path, resolution=resolution)
    overlay = make_contour_overlay(
        gen_img, mask_path, resolution=resolution, thickness=contour_thickness
    )

    panels = []
    if reference_he_path:
        reference_path = Path(reference_he_path)
        if reference_path.exists():
            panels.append((load_rgb_image(reference_path, resolution=resolution), "Reference H&E"))

    panels.extend(
        [
            (mask_display, "Cell Mask (Input)"),
            (gen_img, "Generated H&E"),
            (overlay, "Generated + Cell Contours (Yellow)"),
        ]
    )

    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 5))
    if len(panels) == 1:
        axes = [axes]

    for ax, (image, title) in zip(axes, panels):
        ax.imshow(image)
        ax.set_title(title, fontsize=12)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nVisualization saved -> {save_path}")
