"""
Visualize histology and segmentation PNG images side by side.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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

    if seg_img.ndim == 3:
        seg_display = seg_img[:, :, 0]
    else:
        seg_display = seg_img

    cmap = "gray"

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    axes[0].imshow(hist_img)
    axes[0].set_title("Histology")
    axes[0].axis("off")

    im1 = axes[1].imshow(seg_display, cmap=cmap)
    axes[1].set_title("Segmentation")
    axes[1].axis("off")

    overlay = hist_img.copy()
    binary = (seg_display != 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 0, 0), thickness=1)

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


def _load_sample(
    idx: int,
    hist_dir: str,
    mask_dir: str,
    hist_pattern: str = "0_{idx}.png",
    mask_pattern: str = "0_{idx}/masks_output/type_mask_vis.png",
):
    """Load hist, seg_display, and overlay for a given index. Returns None if files missing."""
    hist_path = Path(hist_dir) / hist_pattern.format(idx=idx)
    seg_path  = Path(mask_dir) / mask_pattern.format(idx=idx)

    if not hist_path.exists() or not seg_path.exists():
        return None

    hist_img = cv2.cvtColor(cv2.imread(str(hist_path)), cv2.COLOR_BGR2RGB)
    seg_img  = cv2.cvtColor(cv2.imread(str(seg_path)),  cv2.COLOR_BGR2RGB)
    seg_display = seg_img[:, :, 0]

    overlay = hist_img.copy()
    binary = (seg_display != 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 0, 0), thickness=1)

    return hist_img, seg_display, overlay


def _sample_valid_indices(n: int, total: int, hist_dir: str, mask_dir: str, hist_pattern: str, mask_pattern: str) -> list:
    rng = np.random.default_rng()
    pool = list(rng.choice(total, size=total, replace=False))
    valid = []
    for idx in pool:
        if len(valid) == n:
            break
        if _load_sample(int(idx), hist_dir, mask_dir, hist_pattern, mask_pattern) is not None:
            valid.append(int(idx))
    if len(valid) < n:
        raise RuntimeError(f"Could only find {len(valid)} valid samples out of {total}")
    return valid


def vis_grid(
    hist_dir: str,
    mask_dir: str,
    hist_pattern: str = "0_{idx}.png",
    mask_pattern: str = "0_{idx}_mask.png",
    n_samples: int = 12,
    total: int = 3660,
    figsize: tuple = (12, 12),
    save_path: str = None,
    dpi: int = 150,
) -> None:
    """
    Visualize n_samples (default 12) random samples in a grid.

    Layout (for 12 samples):
      - 6 rows × 6 cols total
      - Left block  (cols 1-3): samples 1-6,  each row = [hist | seg | overlay]
      - Right block (cols 4-6): samples 7-12, each row = [hist | seg | overlay]

    Args:
        n_samples:  Number of samples to display (must be even).
        total:      Upper bound of sample indices to draw from.
        data_root:  Root directory containing the data folder.
        figsize:    Overall figure size (width, height).
        save_path:  If set, save figure here instead of showing.
        dpi:        DPI for saved figure.
    """
    assert n_samples % 2 == 0, "n_samples must be even (split equally into two blocks)"
    half = n_samples // 2       # samples per block (6)
    n_rows = half               # rows in the figure (6)
    n_cols = 6                  # 3 subplots × 2 blocks

    print(f"Sampling {n_samples} valid indices...")
    indices = _sample_valid_indices(n_samples, total, hist_dir, mask_dir, hist_pattern, mask_pattern)
    print(f"Selected indices: {indices}")

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    col_titles = ["Histology", "Segmentation", "Overlay"]

    for block in range(2):                      # 0 = left, 1 = right
        col_offset = block * 3                  # 0 or 3
        for row in range(n_rows):
            sample_idx = block * half + row     # 0-5 for left, 6-11 for right
            idx = indices[sample_idx]
            hist_img, seg_display, overlay = _load_sample(idx, hist_dir, mask_dir, hist_pattern, mask_pattern)

            panels = [hist_img, seg_display, overlay]
            cmaps  = [None,     "gray",      None]

            for col, (img, cm) in enumerate(zip(panels, cmaps)):
                ax = axes[row, col_offset + col]
                ax.imshow(img, cmap=cm)
                ax.axis("off")

                # Column titles on first row only
                if row == 0:
                    ax.set_title(col_titles[col], fontsize=10, fontweight="bold")

                # Sample index label on the hist panel (first col of each block)
                if col == 0:
                    ax.set_ylabel(f"idx {idx}", fontsize=8, rotation=0,
                                  labelpad=40, va="center")

    # Vertical separator between the two blocks
    fig.add_artist(
        plt.Line2D([0.5, 0.5], [0.02, 0.98],
                   transform=fig.transFigure,
                   color="gray", linewidth=1.5, linestyle="--")
    )

    plt.suptitle("Histology / Segmentation / Overlay — 12 random samples", fontsize=14, y=1.01)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved to {save_path}")
    else:
        plt.show()


def main():
    vis_grid(
        n_samples=12,
        total=3660,
        figsize=(24, 24),
        hist_dir="../data/tcga_subset_3660/tcga_subset_3660_png",
        mask_dir="../data/tcga_subset_3660/tcga_subset_3660_masks",
        save_path="grid_output.png",
    )


if __name__ == "__main__":
    main()