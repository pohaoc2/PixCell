"""
Matplotlib figures for Stage 3 tile visualizations (overview, attention, residuals, ablation).

Used by run_stage3_full.py and tools/generate_stage3_tile_vis.py.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

_RESIDUAL_CMAP = mcolors.LinearSegmentedColormap.from_list("black_red", ["black", "red"])

from tools.color_constants import (
    CELL_STATE_COLORS,
    CELL_TYPE_COLORS,
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

    ax_out = fig.add_subplot(gs[1:3, n_row1 + 1])
    _image_ax(
        ax_out,
        gen_np,
        "Generated H&E",
        "output",
        fontsize=9,
        cosine_sim_val=cosine_sim_val,
    )
    if "cell_masks" in active_channels:
        cell_mask = ctrl_full[active_channels.index("cell_masks")]
        ax_out.contour(cell_mask, levels=[0.5], colors=["lime"], linewidths=0.7, alpha=0.85)

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
    residuals: dict | None = None,
):
    """
    Residual figure: ‖Δ_group‖ in conditioning latent space (32×32 upsampled).
    Shows what conditioning positions were modified upstream of ControlNet.
    """
    from tools.visualize_group_attention import compute_attention_heatmaps
    from tools.visualize_group_residuals import compute_residual_maps

    tme_maps = compute_attention_heatmaps(attn_maps, spatial_size, output_resolution)
    res_maps = compute_residual_maps(residuals, output_resolution) if residuals else {}
    global_res_max = max(m.max() for m in res_maps.values()) if res_maps else 1.0

    style_inputs = style_inputs or []
    n_style = len(style_inputs)
    n_left = n_style + 2      # style panels + cell mask + gen H&E
    group_names = list(tme_maps.keys())
    n_maps = len(group_names)
    has_maps = n_maps > 0
    n_cols = n_left + 1 + n_maps + (1 if has_maps else 0)

    ratios = [1.0] * n_left + [0.08] + [1.0] * n_maps + ([0.08] if has_maps else [])

    # 3 rows: global header | sub-header | residual maps
    fig = plt.figure(figsize=(n_cols * 2.6, 5.5), facecolor="white")
    gs = gridspec.GridSpec(
        3, n_cols,
        figure=fig,
        width_ratios=ratios,
        height_ratios=[0.08, 0.07, 0.85],
        wspace=0.05, hspace=0.08,
        left=0.01, right=0.99, top=0.97, bottom=0.02,
    )

    mask_img = ctrl_full[active_channels.index("cell_masks")]
    inp_hdr_text = "INPUTS  (style H&E + cell mask)" if n_style > 0 else "INPUT"

    # ── Row 0: global section headers ──────────────────────────────────────────
    _header_ax(fig.add_subplot(gs[0, 0 : n_style + 1]), inp_hdr_text, "input")
    _header_ax(fig.add_subplot(gs[0, n_style + 1]), "OUTPUT", "output")
    fig.add_subplot(gs[0, n_left]).axis("off")
    if has_maps:
        _header_ax(
            fig.add_subplot(gs[0, n_left + 1 : n_left + 1 + n_maps]),
            "RESIDUALS  (per TME group)",
            "analysis",
        )
        fig.add_subplot(gs[0, -1]).axis("off")

    # ── Row 1: sub-header ───────────────────────────────────────────────────────
    ax_hdr = fig.add_subplot(gs[1, n_left + 1 : n_left + 1 + n_maps])
    ax_hdr.set_facecolor(SECTION_BG["analysis"])
    ax_hdr.text(0.5, 0.5,
                "Residual ‖Δ_group‖ — conditioning latent modification per group",
                ha="center", va="center", fontsize=8, style="italic",
                color=SECTION_TEXT["analysis"], transform=ax_hdr.transAxes)
    ax_hdr.axis("off")
    fig.add_subplot(gs[1, 0 : n_left + 1]).axis("off")
    if has_maps:
        fig.add_subplot(gs[1, -1]).axis("off")

    # ── Row 2: residual maps ────────────────────────────────────────────────────
    for j, (lbl, img) in enumerate(style_inputs):
        _titled_ax(fig.add_subplot(gs[2, j]), img, "style_ref", lbl)
    _titled_ax(fig.add_subplot(gs[2, n_style]), mask_img, "input", "Cell Mask", cmap="gray")
    _titled_ax(fig.add_subplot(gs[2, n_style + 1]), gen_np, "output", "Generated H&E")
    fig.add_subplot(gs[2, n_left]).axis("off")

    last_im = None
    for k, name in enumerate(group_names):
        hmap = res_maps.get(name)
        ax = fig.add_subplot(gs[2, n_left + 1 + k])
        ax.set_facecolor(SECTION_BG["analysis"])
        if hmap is not None:
            last_im = ax.imshow(hmap, cmap=_RESIDUAL_CMAP, vmin=0, vmax=global_res_max)
        ax.set_title(name, fontsize=8, fontweight="bold",
                     color=SECTION_TEXT["analysis"], pad=4)
        ax.axis("off")
    if last_im is not None and has_maps:
        cbar_ax = fig.add_subplot(gs[2, -1])
        cbar = fig.colorbar(last_im, cax=cbar_ax)
        cbar.set_label("L2 norm", fontsize=7)
        cbar.ax.tick_params(labelsize=7)

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
        last_im = ax.imshow(rmap, cmap=_RESIDUAL_CMAP, vmin=0, vmax=global_max)
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


def _group_rgb_composite(
    group_channels: list, ctrl_full: np.ndarray, active_channels: list
) -> np.ndarray:
    """
    Colored RGB composite for a TME group, using per-channel colors from color_constants.

    - cell types:  cancer=red, immune=blue, healthy=green  (matches CELL_TYPE_COLORS)
    - cell states: prolif=magenta, nonprolif=amber, dead=purple (matches CELL_STATE_COLORS)
    - vasculature: red
    - microenv:    oxygen=cyan, glucose=yellow — additive blend (both=white, absent=black)
    """
    # (R, G, B) in [0, 1] per channel name — derived from color_constants to stay in sync
    def _c(rgba): return (rgba[0]/255, rgba[1]/255, rgba[2]/255)
    _CH_COLOR: dict[str, tuple[float, float, float]] = {
        "cell_type_cancer":     _c(CELL_TYPE_COLORS["cancer"]),
        "cell_type_immune":     _c(CELL_TYPE_COLORS["immune"]),
        "cell_type_healthy":    _c(CELL_TYPE_COLORS["healthy"]),
        "cell_state_prolif":    _c(CELL_STATE_COLORS["proliferative"]),
        "cell_state_nonprolif": _c(CELL_STATE_COLORS["nonprolif"]),
        "cell_state_dead":      _c(CELL_STATE_COLORS["dead"]),
        "vasculature":          (200/255, 0.0,  0.0),
        "oxygen":               (0.0,     1.0,  1.0),   # cyan
        "glucose":              (1.0,     0.95, 0.12),  # yellow
    }
    H, W = ctrl_full.shape[1], ctrl_full.shape[2]
    rgb = np.zeros((H, W, 3), dtype=np.float32)
    for ch in group_channels:
        if ch not in active_channels:
            continue
        val = ctrl_full[active_channels.index(ch)]  # [H, W] in [0, 1]
        color = _CH_COLOR.get(ch, (1.0, 1.0, 1.0))
        rgb[..., 0] += val * color[0]
        rgb[..., 1] += val * color[1]
        rgb[..., 2] += val * color[2]
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def save_enhanced_ablation_grid(
    ablation_images: list,
    save_path: str | Path,
    refs: list | None = None,
    ctrl_full: np.ndarray | None = None,
    active_channels: list | None = None,
    channel_groups: list | None = None,
):
    """
    Ablation grid (3 or 4 rows).
      Row 0: section headers.
      Row 1: generated H&E with input cell mask contour overlay (green, no segmentation).
      Row 2: per-step diff maps (|gen_k − gen_{k−1}|, mean over RGB).
      Row 3: TME channel composite for each newly-added group (only when ctrl_full provided).
    refs panels are prepended for comparison (e.g. reference H&E).
    """
    refs = refs or []
    has_tme = ctrl_full is not None and active_channels is not None and channel_groups is not None

    # Cell mask for contour overlay (float32 [H, W])
    cell_mask = None
    if has_tme and "cell_masks" in active_channels:
        cell_mask = ctrl_full[active_channels.index("cell_masks")]

    # Per-ablation-step TME composites: baseline → blank, each added group → colored RGB composite
    tme_composites: list[np.ndarray | None] = []
    if has_tme:
        n_abl_total = len(ablation_images)
        for i in range(n_abl_total):
            if i == 0:
                tme_composites.append(None)  # baseline: no TME input, show blank
            else:
                g = channel_groups[i - 1]
                tme_composites.append(_group_rgb_composite(g["channels"], ctrl_full, active_channels))

    panels = list(refs)
    for i, (label, img) in enumerate(ablation_images):
        sk = "input" if i == 0 else "output"
        panels.append((sk, label, img))

    n = len(panels)
    n_refs = len(refs)
    n_abl = len(ablation_images)

    # Per-column diff maps: None for refs and baseline, computed for each added group
    diff_maps = [None] * n
    for i in range(1, n_abl):
        col = n_refs + i
        prev = ablation_images[i - 1][1].astype(np.float32)
        curr = ablation_images[i    ][1].astype(np.float32)
        diff_maps[col] = np.abs(curr - prev).mean(axis=-1)
    diff_vmax = max((d.max() for d in diff_maps if d is not None), default=1.0)

    n_rows = 4 if has_tme else 3
    height_ratios = [0.05, 0.42, 0.26, 0.27] if has_tme else [0.08, 0.54, 0.38]
    fig_h = 10.5 if has_tme else 7.5

    fig = plt.figure(figsize=(n * 2.5, fig_h), facecolor="white")
    gs = gridspec.GridSpec(
        n_rows, n,
        figure=fig,
        height_ratios=height_ratios,
        wspace=0.05, hspace=0.08,
        left=0.01, right=0.99, top=0.97, bottom=0.02,
    )

    # Row 0: headers
    for j, (sk, lbl, img) in enumerate(panels):
        _header_ax(fig.add_subplot(gs[0, j]), lbl, sk)

    # Row 1: generated H&E + cell mask contour overlay (matplotlib contour, no segmentation)
    img_axes = []
    for j, (sk, lbl, img) in enumerate(panels):
        ax = fig.add_subplot(gs[1, j])
        ax.set_facecolor(SECTION_BG[sk])
        vmax = None if img.dtype == np.uint8 else 1.0
        ax.imshow(img, vmin=0, vmax=vmax)
        if cell_mask is not None and j >= n_refs:
            ax.contour(cell_mask, levels=[0.5], colors=["lime"], linewidths=0.7, alpha=0.85)
        ax.axis("off")
        img_axes.append(ax)

    # Add legend note on first ablation column
    if cell_mask is not None and n_refs < n:
        img_axes[n_refs].set_title(
            "── input mask overlay", fontsize=6, color="lime",
            pad=2, loc="center",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#222222", alpha=0.6, edgecolor="none"),
        )

    # Row 2: diff maps (blank for refs and baseline)
    last_diff_im = None
    diff_axes = []
    for j, (sk, lbl, img) in enumerate(panels):
        ax = fig.add_subplot(gs[2, j])
        dmap = diff_maps[j]
        if dmap is not None:
            last_diff_im = ax.imshow(dmap, cmap="hot", vmin=0, vmax=diff_vmax)
            ax.set_title("Δ pixel", fontsize=7, color=SECTION_TEXT["analysis"], pad=2)
        else:
            ax.set_facecolor(SECTION_BG.get(sk, "#f0f0f0"))
        ax.axis("off")
        diff_axes.append(ax)

    if last_diff_im is not None:
        fig.colorbar(last_diff_im, ax=diff_axes[-1], fraction=0.046, pad=0.04, label="Mean |diff|")

    # Row 3: TME input composites (newly-added group channels per step; blank for baseline)
    if has_tme:
        for j, (sk, lbl, img) in enumerate(panels):
            ax = fig.add_subplot(gs[3, j])
            abl_idx = j - n_refs
            comp = None if (j < n_refs or abl_idx >= len(tme_composites)) else tme_composites[abl_idx]
            if comp is None:
                ax.set_facecolor(SECTION_BG.get(sk, "#f0f0f0"))
                ax.axis("off")
                continue
            ax.imshow(comp, vmin=0, vmax=255)
            g = channel_groups[abl_idx - 1]
            ch_names = [
                ch.replace("cell_type_", "").replace("cell_state_", "")
                for ch in g["channels"]
            ]
            # avoid "vasculature / vasculature" for single-channel groups
            if len(ch_names) == 1 and ch_names[0] == g["name"]:
                title = g["name"]
            else:
                title = g["name"] + "\n" + "  ".join(ch_names)
            ax.set_title(title, fontsize=6, color=SECTION_TEXT["input"], pad=2)
            ax.axis("off")

        # Row 3 section label (left margin annotation)
        fig.text(
            0.002, 0.12, "TME INPUT", va="center", ha="left",
            fontsize=7, fontweight="bold", color=SECTION_TEXT["input"],
            rotation=90,
        )

    # Separator line between refs and ablation columns
    if n_refs > 0 and n_refs < n:
        x_sep = (img_axes[n_refs - 1].get_position().x1 +
                 img_axes[n_refs    ].get_position().x0) / 2
        fig.add_artist(plt.Line2D(
            [x_sep, x_sep], [0.02, 0.98],
            transform=fig.transFigure,
            color="#aaaaaa", linewidth=1.5, linestyle="--",
        ))

    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Ablation grid saved → {save_path}")


def save_condition_ablation_grid(
    ablation_images: list,
    save_path: str | Path,
    refs: list | None = None,
    ctrl_full: np.ndarray | None = None,
    active_channels: list | None = None,
):
    """Simple grid for arbitrary standalone ablation conditions."""
    refs = refs or []

    cell_mask = None
    if ctrl_full is not None and active_channels is not None and "cell_masks" in active_channels:
        cell_mask = ctrl_full[active_channels.index("cell_masks")]

    panels = list(refs)
    for label, img in ablation_images:
        panels.append(("output", label, img))

    n = len(panels)
    fig = plt.figure(figsize=(n * 2.5, 5.0), facecolor="white")
    gs = gridspec.GridSpec(
        2, n, figure=fig,
        height_ratios=[0.13, 0.87],
        wspace=0.05, hspace=0.08,
        left=0.01, right=0.99, top=0.97, bottom=0.02,
    )

    n_refs = len(refs)
    for j, (section_key, label, img) in enumerate(panels):
        _header_ax(fig.add_subplot(gs[0, j]), label, section_key)
        ax = fig.add_subplot(gs[1, j])
        ax.set_facecolor(SECTION_BG[section_key])
        ax.imshow(img, vmin=0, vmax=None if img.dtype == np.uint8 else 1.0)
        if cell_mask is not None and j >= n_refs:
            ax.contour(cell_mask, levels=[0.5], colors=["lime"], linewidths=0.7, alpha=0.85)
        ax.axis("off")

    if n_refs > 0 and n_refs < n:
        x_sep = (fig.axes[n_refs * 2 - 1].get_position().x1 +
                 fig.axes[n_refs * 2 + 1].get_position().x0) / 2
        fig.add_artist(plt.Line2D([x_sep, x_sep], [0.02, 0.98],
                                  transform=fig.transFigure, color="#aaaaaa",
                                  linewidth=1.5, linestyle="--"))

    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Condition ablation grid saved → {save_path}")


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
