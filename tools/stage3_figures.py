"""
Matplotlib figures for Stage 3 tile visualizations (overview, attention, residuals, ablation).

Used by run_stage3_full.py and tools/generate_stage3_tile_vis.py.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from tools.color_constants import (
    CHANNEL_CMAP,
    CHANNEL_LABEL,
    SECTION_BG,
    SECTION_TEXT,
)


def _header_ax(ax, label, section_key):
    """Render a section header on a thin axes row."""
    ax.set_facecolor(SECTION_BG[section_key])
    ax.text(
        0.5,
        0.5,
        label,
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        color=SECTION_TEXT[section_key],
        transform=ax.transAxes,
    )
    ax.axis("off")


def _image_ax(
    ax,
    img,
    label,
    section_key,
    cmap=None,
    fontsize=8,
    cosine_sim_val=None,
    channel_key: str | None = None,
):
    """Render a single image panel (row 1 of overview)."""
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    ax.set_facecolor(SECTION_BG[section_key])
    vmax = 1.0 if (img.ndim == 2 or img.dtype != np.uint8) else None
    im = ax.imshow(img, cmap=cmap, vmin=0, vmax=vmax)
    if channel_key in ("oxygen", "glucose") and cmap is not None:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="7%", pad=0.06)
        cbar = ax.figure.colorbar(im, cax=cax)
        cbar.ax.tick_params(labelsize=5)
        if channel_key == "oxygen":
            cbar.set_label("O\u2082 proxy", fontsize=6)
        else:
            cbar.set_label("Glucose proxy", fontsize=6)
    title = label
    if cosine_sim_val is not None:
        title += f"\ncos sim={cosine_sim_val:.3f}"
    ax.set_title(
        title,
        fontsize=fontsize,
        fontweight="bold",
        color=SECTION_TEXT[section_key],
        pad=3,
    )
    ax.axis("off")


def _titled_ax(ax, img, section_key, title, cmap=None, fontsize=9):
    """Image panel with plain title (no colored bbox)."""
    ax.set_facecolor(SECTION_BG[section_key])
    vmax = 1.0 if (img.ndim == 2 or img.dtype != np.uint8) else None
    ax.imshow(img, cmap=cmap, vmin=0, vmax=vmax)
    ax.set_title(
        title,
        fontsize=fontsize,
        fontweight="bold",
        color=SECTION_TEXT[section_key],
        pad=4,
    )
    ax.axis("off")


def save_overview_figure(
    ctrl_full: np.ndarray,
    active_channels: list,
    gen_np: np.ndarray,
    save_path: str | Path,
    style_inputs: list | None = None,
    cosine_sim_val: float | None = None,
):
    """
    3-row GridSpec overview (inputs split across 2 rows, output spans both).

    style_inputs: [(label, image), ...] prepended as style inputs (paired UNI mode).
    """
    style_inputs = style_inputs or []
    n_style = len(style_inputs)
    n_ch = len(active_channels)
    n_total = n_style + n_ch

    n_row1 = (n_total + 1) // 2
    n_ch_r1 = n_row1 - n_style
    n_ch_r2 = n_ch - n_ch_r1

    n_cols = n_row1 + 1 + 1
    ratios = [1.0] * n_row1 + [0.25] + [1.0]
    fig_w = max(n_cols * 2.2, 8)
    fig_h = 5.5

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
    gs = gridspec.GridSpec(
        3,
        n_cols,
        figure=fig,
        width_ratios=ratios,
        height_ratios=[0.07, 0.465, 0.465],
        wspace=0.05,
        hspace=0.08,
        left=0.01,
        right=0.99,
        top=0.97,
        bottom=0.02,
    )

    inp_hdr_text = (
        "INPUTS  (style H&E + TME layout channels)"
        if n_style > 0
        else "INPUTS  (TME channels)"
    )
    _header_ax(fig.add_subplot(gs[0, 0:n_row1]), inp_hdr_text, "input")
    fig.add_subplot(gs[0, n_row1]).axis("off")
    _header_ax(fig.add_subplot(gs[0, n_row1 + 1]), "OUTPUT", "output")

    for j, (lbl, img) in enumerate(style_inputs):
        _image_ax(fig.add_subplot(gs[1, j]), img, lbl, "style_ref", fontsize=7)
    for i in range(n_ch_r1):
        ch = active_channels[i]
        _image_ax(
            fig.add_subplot(gs[1, n_style + i]),
            ctrl_full[i],
            CHANNEL_LABEL.get(ch, ch),
            "input",
            cmap=CHANNEL_CMAP.get(ch),
            fontsize=7,
            channel_key=ch,
        )
    for i in range(n_ch_r2):
        ch = active_channels[n_ch_r1 + i]
        _image_ax(
            fig.add_subplot(gs[2, i]),
            ctrl_full[n_ch_r1 + i],
            CHANNEL_LABEL.get(ch, ch),
            "input",
            cmap=CHANNEL_CMAP.get(ch),
            fontsize=7,
            channel_key=ch,
        )

    ax_arr = fig.add_subplot(gs[1:3, n_row1])
    ax_arr.text(
        0.5,
        0.5,
        "▶",
        ha="center",
        va="center",
        fontsize=24,
        color="#555555",
        transform=ax_arr.transAxes,
    )
    ax_arr.axis("off")

    _image_ax(
        fig.add_subplot(gs[1:3, n_row1 + 1]),
        gen_np,
        "Generated H&E",
        "output",
        fontsize=9,
        cosine_sim_val=cosine_sim_val,
    )

    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Overview figure saved → {save_path}")


def save_enhanced_attention_figure(
    ctrl_full: np.ndarray,
    active_channels: list,
    gen_np: np.ndarray,
    attn_maps: dict,
    save_path: str | Path,
    style_inputs: list | None = None,
    spatial_size: tuple = (32, 32),
    output_resolution: int = 256,
):
    """
    Dual-row attention figure.

    Row 1 — TME space (sum over Q):  which TME positions were consulted.
    Row 2 — Mask space (sum over KV): which mask positions were seeking info.

    Layout per row: [style?] [cell mask] | [gen H&E] | ── | heatmaps…
    """
    from tools.visualize_group_attention import compute_attention_heatmaps_dual

    dual = compute_attention_heatmaps_dual(attn_maps, spatial_size, output_resolution)
    style_inputs = style_inputs or []

    n_style = len(style_inputs)
    n_left = n_style + 2      # style panels + cell mask + gen H&E
    group_names = list(dual.keys())
    n_attn = len(group_names)
    has_attn = n_attn > 0
    n_cols = n_left + 1 + n_attn + (1 if has_attn else 0)

    ratios = [1.0] * n_left + [0.08] + [1.0] * n_attn + ([0.08] if has_attn else [])
    # 5 rows: global header | tme sub-header | tme images | mask sub-header | mask images
    fig = plt.figure(figsize=(n_cols * 2.6, 9.0), facecolor="white")
    gs = gridspec.GridSpec(
        5,
        n_cols,
        figure=fig,
        width_ratios=ratios,
        height_ratios=[0.07, 0.06, 0.42, 0.06, 0.42],
        wspace=0.05,
        hspace=0.08,
        left=0.01,
        right=0.99,
        top=0.97,
        bottom=0.02,
    )

    mask_img = ctrl_full[active_channels.index("cell_masks")]
    inp_hdr_text = "INPUTS  (style H&E + cell mask)" if n_style > 0 else "INPUT"

    # ── Row 0: global section headers ──────────────────────────────────────────
    _header_ax(fig.add_subplot(gs[0, 0 : n_style + 1]), inp_hdr_text, "input")
    _header_ax(fig.add_subplot(gs[0, n_style + 1]), "OUTPUT", "output")
    fig.add_subplot(gs[0, n_left]).axis("off")
    if has_attn:
        _header_ax(
            fig.add_subplot(gs[0, n_left + 1 : n_left + 1 + n_attn]),
            "ATTENTION  (per TME group)",
            "analysis",
        )
        fig.add_subplot(gs[0, -1]).axis("off")

    def _render_attn_row(data_row, sub_row, maps_key, row_subtitle):
        # sub-header label
        ax_lbl = fig.add_subplot(gs[sub_row, n_left + 1 : n_left + 1 + n_attn])
        ax_lbl.set_facecolor(SECTION_BG["analysis"])
        ax_lbl.text(0.5, 0.5, row_subtitle, ha="center", va="center",
                    fontsize=8, style="italic", color=SECTION_TEXT["analysis"],
                    transform=ax_lbl.transAxes)
        ax_lbl.axis("off")
        fig.add_subplot(gs[sub_row, 0 : n_left + 1]).axis("off")
        if has_attn:
            fig.add_subplot(gs[sub_row, -1]).axis("off")

        # style + mask + gen panels
        for j, (lbl, img) in enumerate(style_inputs):
            _titled_ax(fig.add_subplot(gs[data_row, j]), img, "style_ref", lbl)
        _titled_ax(fig.add_subplot(gs[data_row, n_style]), mask_img, "input",
                   "Cell Mask", cmap="gray")
        _titled_ax(fig.add_subplot(gs[data_row, n_style + 1]), gen_np, "output",
                   "Generated H&E")
        fig.add_subplot(gs[data_row, n_left]).axis("off")

        # heatmap panels
        last_im = None
        for k, name in enumerate(group_names):
            hmap = dual[name][maps_key]
            ax = fig.add_subplot(gs[data_row, n_left + 1 + k])
            ax.set_facecolor(SECTION_BG["analysis"])
            last_im = ax.imshow(hmap, cmap="jet", vmin=0, vmax=1)
            ax.set_title(name, fontsize=8, fontweight="bold",
                         color=SECTION_TEXT["analysis"], pad=4)
            ax.axis("off")
        if last_im is not None and has_attn:
            cbar_ax = fig.add_subplot(gs[data_row, -1])
            cbar = fig.colorbar(last_im, cax=cbar_ax)
            cbar.set_label("Attn weight", fontsize=7)
            cbar.ax.tick_params(labelsize=7)

    _render_attn_row(2, 1, "tme_space",  "TME space — which TME positions were consulted")
    _render_attn_row(4, 3, "mask_space", "Mask space — which mask positions were seeking info")

    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Attention heatmaps saved → {save_path}")


def save_enhanced_residual_figure(
    ctrl_full: np.ndarray,
    active_channels: list,
    gen_np: np.ndarray,
    residuals: dict,
    save_path: str | Path,
    refs: list | None = None,
    output_resolution: int = 256,
):
    """
    Layout (2 rows):
      Row 0 headers: INPUT | OUTPUT | REF(s) | ── | RESIDUALS (per group)
      Row 1 images:  mask  | gen H&E| ref(s) | ── | ‖Δ_group‖ maps
    """
    from tools.visualize_group_residuals import compute_residual_maps

    res_maps = compute_residual_maps(residuals, output_resolution)
    global_max = max(m.max() for m in res_maps.values()) if res_maps else 1.0
    refs = refs or []

    n_fixed = 2 + len(refs)
    n_res = len(res_maps)
    n_cols = n_fixed + 1 + n_res

    ratios = [1.0] * n_fixed + [0.08] + [1.0] * n_res
    fig = plt.figure(figsize=(n_cols * 2.6, 5.0), facecolor="white")
    gs = gridspec.GridSpec(
        2,
        n_cols,
        figure=fig,
        width_ratios=ratios,
        height_ratios=[0.13, 0.87],
        wspace=0.05,
        hspace=0.08,
        left=0.01,
        right=0.99,
        top=0.97,
        bottom=0.02,
    )

    mask_img = ctrl_full[active_channels.index("cell_masks")]

    _header_ax(fig.add_subplot(gs[0, 0]), "INPUT", "input")
    _header_ax(fig.add_subplot(gs[0, 1]), "OUTPUT", "output")
    for j, (sk, lbl, _) in enumerate(refs):
        _header_ax(fig.add_subplot(gs[0, 2 + j]), lbl, sk)
    fig.add_subplot(gs[0, n_fixed]).axis("off")
    if n_res:
        _header_ax(fig.add_subplot(gs[0, n_fixed + 1 :]), "RESIDUALS  ‖Δ_group‖", "analysis")

    _titled_ax(fig.add_subplot(gs[1, 0]), mask_img, "input", "Cell Mask", cmap="gray")
    _titled_ax(fig.add_subplot(gs[1, 1]), gen_np, "output", "Generated H&E")
    for j, (sk, lbl, img) in enumerate(refs):
        _titled_ax(fig.add_subplot(gs[1, 2 + j]), img, sk, lbl)
    fig.add_subplot(gs[1, n_fixed]).axis("off")

    last_im = None
    for k, (name, rmap) in enumerate(res_maps.items()):
        ax = fig.add_subplot(gs[1, n_fixed + 1 + k])
        ax.set_facecolor(SECTION_BG["analysis"])
        last_im = ax.imshow(rmap, cmap="inferno", vmin=0, vmax=global_max)
        ax.set_title(
            f"‖Δ_{name}‖",
            fontsize=8,
            fontweight="bold",
            color=SECTION_TEXT["analysis"],
            pad=4,
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor=SECTION_BG["analysis"],
                edgecolor="none",
                alpha=0.9,
            ),
        )
        ax.axis("off")

    if last_im is not None:
        fig.colorbar(last_im, ax=fig.axes[-1], fraction=0.046, pad=0.04, label="L2 norm")

    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Residual magnitude maps saved → {save_path}")


def save_enhanced_ablation_grid(
    ablation_images: list,
    save_path: str | Path,
    refs: list | None = None,
):
    """
    2-row ablation grid. Row 0: section headers. Row 1: images.
    refs panels are prepended (e.g. reference H&E for comparison).
    Then: Mask only → +group1 → ... → all groups.
    """
    refs = refs or []

    panels = list(refs)
    for i, (label, img) in enumerate(ablation_images):
        sk = "input" if i == 0 else "output"
        panels.append((sk, label, img))

    n = len(panels)
    fig = plt.figure(figsize=(n * 2.5, 5.0), facecolor="white")
    gs = gridspec.GridSpec(
        2,
        n,
        figure=fig,
        height_ratios=[0.13, 0.87],
        wspace=0.05,
        hspace=0.08,
        left=0.01,
        right=0.99,
        top=0.97,
        bottom=0.02,
    )

    for j, (sk, lbl, img) in enumerate(panels):
        _header_ax(fig.add_subplot(gs[0, j]), lbl, sk)
        ax = fig.add_subplot(gs[1, j])
        ax.set_facecolor(SECTION_BG[sk])
        vmax = None if img.dtype == np.uint8 else 1.0
        ax.imshow(img, vmin=0, vmax=vmax)
        ax.axis("off")

    n_refs = len(refs)
    if n_refs > 0 and n_refs < n:
        ax_last_ref = fig.axes[n_refs * 2 - 1]
        ax_first_abl = fig.axes[n_refs * 2 + 1]
        x_sep = (ax_last_ref.get_position().x1 + ax_first_abl.get_position().x0) / 2
        fig.add_artist(
            plt.Line2D(
                [x_sep, x_sep],
                [0.02, 0.98],
                transform=fig.transFigure,
                color="#aaaaaa",
                linewidth=1.5,
                linestyle="--",
            )
        )

    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Ablation grid saved → {save_path}")


def save_loo_ablation_grid(
    ablation_images: list,
    save_path: str | Path,
    refs: list | None = None,
):
    """Leave-one-out ablation grid.

    ablation_images: [(label, gen_np), ...] where first item is 'All groups'
                     and remainder are 'minus_G' conditions.
    refs: optional list of (section_key, label, img) reference panels prepended.
    """
    refs = refs or []
    panels = list(refs)
    for i, (label, img) in enumerate(ablation_images):
        sk = "output" if i == 0 else "analysis"
        panels.append((sk, label, img))

    n = len(panels)
    fig = plt.figure(figsize=(n * 2.5, 5.0), facecolor="white")
    gs = gridspec.GridSpec(2, n, figure=fig, height_ratios=[0.13, 0.87],
                           wspace=0.05, hspace=0.08,
                           left=0.01, right=0.99, top=0.97, bottom=0.02)

    for j, (sk, lbl, img) in enumerate(panels):
        _header_ax(fig.add_subplot(gs[0, j]), lbl, sk)
        ax = fig.add_subplot(gs[1, j])
        ax.set_facecolor(SECTION_BG[sk])
        ax.imshow(img, vmin=0, vmax=None if img.dtype == np.uint8 else 1.0)
        ax.axis("off")

    # Separator between reference and first ablation
    n_refs = len(refs)
    if n_refs > 0 and n_refs < n:
        x_sep = (fig.axes[n_refs * 2 - 1].get_position().x1 +
                 fig.axes[n_refs * 2 + 1].get_position().x0) / 2
        fig.add_artist(plt.Line2D([x_sep, x_sep], [0.02, 0.98],
                                  transform=fig.transFigure, color="#aaaaaa",
                                  linewidth=1.5, linestyle="--"))

    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"LOO ablation grid saved → {save_path}")


def save_pairwise_ablation_grid(
    ablation_images: list,
    save_path: str | Path,
    refs: list | None = None,
):
    """Pairwise ablation grid: mask_only + mask+single_group for each group.

    ablation_images: [(label, gen_np), ...] first = 'Mask only', rest = '+G'.
    refs: optional list of (section_key, label, img) reference panels.
    """
    refs = refs or []
    panels = list(refs)
    for i, (label, img) in enumerate(ablation_images):
        sk = "input" if i == 0 else "output"
        panels.append((sk, label, img))

    n = len(panels)
    fig = plt.figure(figsize=(n * 2.5, 5.0), facecolor="white")
    gs = gridspec.GridSpec(2, n, figure=fig, height_ratios=[0.13, 0.87],
                           wspace=0.05, hspace=0.08,
                           left=0.01, right=0.99, top=0.97, bottom=0.02)

    for j, (sk, lbl, img) in enumerate(panels):
        _header_ax(fig.add_subplot(gs[0, j]), lbl, sk)
        ax = fig.add_subplot(gs[1, j])
        ax.set_facecolor(SECTION_BG[sk])
        ax.imshow(img, vmin=0, vmax=None if img.dtype == np.uint8 else 1.0)
        ax.axis("off")

    n_refs = len(refs)
    if n_refs > 0 and n_refs < n:
        x_sep = (fig.axes[n_refs * 2 - 1].get_position().x1 +
                 fig.axes[n_refs * 2 + 1].get_position().x0) / 2
        fig.add_artist(plt.Line2D([x_sep, x_sep], [0.02, 0.98],
                                  transform=fig.transFigure, color="#aaaaaa",
                                  linewidth=1.5, linestyle="--"))

    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Pairwise ablation grid saved → {save_path}")
