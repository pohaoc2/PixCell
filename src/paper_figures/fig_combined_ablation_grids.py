from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

_LABEL_STRIP_PX = 120
_SAVE_DPI = 220
_PANEL_TITLES = {
    "A": "Paired ablation grid",
    "B": "Unpaired ablation grid",
}


def _load_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def _scale_to_height(img: np.ndarray, target_h: int) -> np.ndarray:
    if img.shape[0] == target_h:
        return img
    w = round(img.shape[1] * target_h / img.shape[0])
    return np.array(Image.fromarray(img).resize((w, target_h), Image.LANCZOS))


def _add_label_strip(img: np.ndarray, px: int = _LABEL_STRIP_PX) -> np.ndarray:
    strip = np.full((px, img.shape[1], 3), 255, dtype=np.uint8)
    return np.concatenate([strip, img], axis=0)


def build_combined_ablation_grids_figure(path_a: Path, path_b: Path) -> plt.Figure:
    img_a = _load_rgb(path_a)
    img_b = _load_rgb(path_b)

    target_h = min(img_a.shape[0], img_b.shape[0])
    img_a = _add_label_strip(_scale_to_height(img_a, target_h))
    img_b = _add_label_strip(_scale_to_height(img_b, target_h))

    fig_h = img_a.shape[0]
    total_w = img_a.shape[1] + img_b.shape[1]
    fig = plt.figure(figsize=(total_w / _SAVE_DPI, fig_h / _SAVE_DPI))
    gs = fig.add_gridspec(1, 2, width_ratios=[img_a.shape[1], img_b.shape[1]], wspace=0)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    for i, (img, label) in enumerate(zip([img_a, img_b], ["A", "B"])):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(img, interpolation="none")
        ax.axis("off")
        ax.text(
            0.012,
            0.994,
            label,
            transform=ax.transAxes,
            # Overlay is drawn on 220 DPI raster panels, so this remains in raster-pixel scale.
            fontsize=28,
            fontweight="bold",
            va="top",
            ha="left",
            color="black",
        )
        ax.text(
            0.062,
            0.989,
            _PANEL_TITLES[label],
            transform=ax.transAxes,
            fontsize=18,
            fontweight="bold",
            va="top",
            ha="left",
            color="black",
        )

    # vertical dashed separator between A and B
    x_split = img_a.shape[1] / total_w
    fig.add_artist(
        plt.Line2D(
            [x_split, x_split], [0, 1],
            transform=fig.transFigure,
            color="black",
            linewidth=1.2,
            linestyle=(0, (6, 4)),
        )
    )

    return fig
