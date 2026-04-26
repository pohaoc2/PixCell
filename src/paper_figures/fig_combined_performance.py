import io

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from tools.ablation_report.data import DatasetSummary
from tools.ablation_report.figures import (
    build_channel_effect_heatmaps_figure,
    build_comparison_table_figure,
    build_metric_trends_figure,
)

_RENDER_DPI = 220
_LABEL_STRIP_PX = 80


def _fig_to_rgb(fig: plt.Figure) -> np.ndarray:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=_RENDER_DPI, bbox_inches="tight")
    buf.seek(0)
    arr = np.array(Image.open(buf).convert("RGB"))
    plt.close(fig)
    return arr


def _scale_to_width(img: np.ndarray, target_w: int) -> np.ndarray:
    if img.shape[1] == target_w:
        return img
    h = round(img.shape[0] * target_w / img.shape[1])
    return np.array(Image.fromarray(img).resize((target_w, h), Image.LANCZOS))


def _add_label_strip(img: np.ndarray, px: int = _LABEL_STRIP_PX) -> np.ndarray:
    strip = np.full((px, img.shape[1], 3), 255, dtype=np.uint8)
    return np.concatenate([strip, img], axis=0)


def build_combined_performance_figure(summaries: list[DatasetSummary]) -> plt.Figure:
    imgs = [
        _fig_to_rgb(build_metric_trends_figure(summaries)),
        _fig_to_rgb(build_comparison_table_figure(summaries)),
        _fig_to_rgb(build_channel_effect_heatmaps_figure(summaries)),
    ]
    max_w = max(img.shape[1] for img in imgs)
    imgs = [_add_label_strip(_scale_to_width(img, max_w)) for img in imgs]
    heights = [img.shape[0] for img in imgs]

    fig_w = max_w / _RENDER_DPI
    fig_h = sum(heights) / _RENDER_DPI
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(3, 1, height_ratios=heights, hspace=0)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    for i, (img, label) in enumerate(zip(imgs, ["A", "B", "C"])):
        ax = fig.add_subplot(gs[i, 0])
        ax.imshow(img, interpolation="none")
        ax.axis("off")
        ax.text(
            0.012,
            0.994,
            label,
            transform=ax.transAxes,
            fontsize=28,
            fontweight="bold",
            va="top",
            ha="left",
            color="black",
        )

    return fig
