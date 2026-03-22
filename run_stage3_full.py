"""
run_stage3_full.py — Comprehensive Stage 3 inference + validation + visualizations.

Uses experimental channel data (ORION-CRC33) as inference input:
  - 20 random "inference" tiles (paired: channels → generate H&E, compare vs paired UNI)
  - 20 random "validation" tiles (held-out unpaired: same metric, different tiles)

Generates for each set:
  - Generated H&E images
  - UNI cosine similarity vs. ground-truth exp H&E features
  - Attention heatmaps (per TME group)
  - Residual magnitude maps (per TME group)
  - Ablation grid (progressive group addition)
  - Summary metrics JSON
"""
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler
from PIL import Image

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent
CONFIG_PATH = ROOT / "configs/config_controlnet_exp.py"
CKPT_DIR    = ROOT / "checkpoints/pixcell_controlnet_exp/checkpoints"
EXP_ROOT    = ROOT / "data/orion-crc33"
EXP_CH_DIR  = EXP_ROOT / "exp_channels"
FEAT_DIR    = EXP_ROOT / "features"
HE_DIR      = EXP_ROOT / "he"
OUT_DIR     = ROOT / "inference_output/stage3_full"

N_INFERENCE  = 20
N_VALIDATION = 20
N_VIS_TILES  =  5   # tiles with full visualization (attention + residuals + ablation)
GUIDANCE_SCALE = 2.5
NUM_STEPS      = 20
SEED           = 42
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

# Channel that holds the cell mask in exp data
MASK_CHANNEL = "cell_masks"

# Binary channels (thresholded to {0,1})
_BINARY = frozenset({
    "cell_masks",
    "cell_type_healthy", "cell_type_cancer", "cell_type_immune",
    "cell_state_prolif",  "cell_state_nonprolif", "cell_state_dead",
})

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_channel(ch_dir: Path, tile_id: str, resolution: int, binary: bool) -> np.ndarray:
    """Load a single channel PNG → float32 [H, W] in [0, 1]."""
    fpath = ch_dir / f"{tile_id}.png"
    if not fpath.exists():
        fpath = ch_dir / f"{tile_id}.npy"
    if not fpath.exists():
        raise FileNotFoundError(f"Channel file not found: {ch_dir / tile_id}.*")
    if fpath.suffix == ".npy":
        arr = np.load(fpath).astype(np.float32)
    else:
        import cv2
        img = cv2.imread(str(fpath), cv2.IMREAD_GRAYSCALE)
        arr = img.astype(np.float32) / 255.0
    if arr.shape != (resolution, resolution):
        import cv2
        arr = cv2.resize(arr, (resolution, resolution), interpolation=cv2.INTER_LINEAR)
    if binary:
        arr = (arr > 0.5).astype(np.float32)
    return arr


def load_exp_channels(tile_id: str, active_channels: list, resolution: int) -> torch.Tensor:
    """Load all active channels from exp_channels → [C, H, W]."""
    planes = []
    for ch in active_channels:
        ch_dir = EXP_CH_DIR / ch
        arr = load_channel(ch_dir, tile_id, resolution, binary=(ch in _BINARY))
        planes.append(arr)
    return torch.from_numpy(np.stack(planes, axis=0))  # [C, H, W]


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b))


# ── Model loading ─────────────────────────────────────────────────────────────

def load_all_models(config, device: str):
    import glob as _glob
    from diffusion.model.builder import build_model
    from diffusion.utils.misc import read_config
    from train_scripts.inference_controlnet import (
        load_vae, load_controlnet_model_from_checkpoint,
        load_pixcell_controlnet_model_from_checkpoint,
    )
    from train_scripts.training_utils import load_tme_checkpoint

    print("Loading VAE...")
    vae = load_vae(config.vae_pretrained, device)

    print("Loading TME module...")
    group_specs = [
        dict(name=g["name"], n_channels=len(g["channels"]))
        for g in config.channel_groups
    ]
    tme_module = build_model(
        "MultiGroupTMEModule", False, False,
        channel_groups=group_specs,
        base_ch=getattr(config, "tme_base_ch", 32),
    )
    load_tme_checkpoint(str(CKPT_DIR), tme_module, device=device)
    dtype = torch.float16 if device == "cuda" else torch.float32
    tme_module.to(device=device, dtype=dtype).eval()

    print("Loading ControlNet...")
    controlnet_pths = sorted(_glob.glob(str(CKPT_DIR / "controlnet_*.pth")))
    if not controlnet_pths:
        raise FileNotFoundError(f"No controlnet_*.pth in {CKPT_DIR}")
    controlnet_pth = controlnet_pths[-1]
    print(f"  → {controlnet_pth}")
    controlnet = load_controlnet_model_from_checkpoint(
        str(CONFIG_PATH), controlnet_pth, device
    )

    print("Loading base model...")
    base_model_path = getattr(config, "load_from", config.base_model_path)
    if os.path.isdir(base_model_path):
        candidates = (
            _glob.glob(os.path.join(base_model_path, "*.safetensors")) +
            _glob.glob(os.path.join(base_model_path, "*.pth"))
        )
        base_model_path = sorted(candidates)[0]
    print(f"  → {base_model_path}")
    base_model = load_pixcell_controlnet_model_from_checkpoint(
        str(CONFIG_PATH), base_model_path
    )
    base_model.to(device).eval()

    return dict(vae=vae, controlnet=controlnet, base_model=base_model, tme_module=tme_module)


# ── Single-tile generation ────────────────────────────────────────────────────

def generate_tile(
    tile_id: str,
    models: dict,
    config,
    scheduler,
    uni_embeds: torch.Tensor,
    device: str,
    return_vis_data: bool = False,
):
    """
    Generate H&E for one tile from its exp channels.

    Returns:
        gen_np: uint8 [H, W, 3]
        vis_data: dict with mask_rgb, residuals, attn_maps (if return_vis_data)
    """
    from tools.channel_group_utils import split_channels_to_groups

    active_channels = config.data.active_channels
    vae_scale = config.scale_factor
    vae_shift  = config.shift_factor
    dtype = torch.float16 if device == "cuda" else torch.float32

    vae        = models["vae"]
    controlnet = models["controlnet"]
    base_model = models["base_model"]
    tme_module = models["tme_module"]

    vae.to(device=device, dtype=dtype).eval()

    # 1. Load channels → [C, H, W]
    ctrl_full = load_exp_channels(tile_id, active_channels, config.image_size)

    # 2. VAE-encode cell_mask (channel 0)
    cell_mask_img = ctrl_full[0:1].unsqueeze(0).repeat(1, 3, 1, 1)  # [1,3,H,W]
    cell_mask_img = 2 * (cell_mask_img - 0.5)
    with torch.no_grad():
        vae_mask = vae.encode(
            cell_mask_img.to(device, dtype=dtype)
        ).latent_dist.mean
        vae_mask = (vae_mask - vae_shift) * vae_scale

    # 3. Split channels to groups
    tme_dict = split_channels_to_groups(
        ctrl_full.unsqueeze(0).to(device, dtype=dtype),
        active_channels,
        config.channel_groups,
    )

    # 4. TME module forward
    with torch.no_grad():
        if return_vis_data:
            fused, residuals, attn_maps = tme_module(
                vae_mask, tme_dict,
                return_residuals=True, return_attn_weights=True,
            )
        else:
            fused = tme_module(vae_mask, tme_dict)
            residuals, attn_maps = {}, {}

    # 5. Denoise
    from train_scripts.inference_controlnet import denoise
    latent_shape = (1, 16, config.image_size // 8, config.image_size // 8)
    latents = torch.randn(latent_shape, device=device, dtype=dtype)
    latents = latents * scheduler.init_noise_sigma
    denoised = denoise(
        latents=latents,
        uni_embeds=uni_embeds.to(device, dtype=dtype),
        controlnet_input_latent=fused,
        scheduler=scheduler,
        controlnet_model=controlnet,
        pixcell_controlnet_model=base_model,
        guidance_scale=GUIDANCE_SCALE,
        device=device,
    )

    # 6. Decode → RGB
    with torch.no_grad():
        scaled = (denoised.to(dtype) / vae_scale) + vae_shift
        gen_img = vae.decode(scaled, return_dict=False)[0]
    gen_img = (gen_img / 2 + 0.5).clamp(0, 1)
    gen_np = (gen_img.cpu().permute(0, 2, 3, 1).numpy()[0] * 255).astype(np.uint8)

    vis_data = None
    if return_vis_data:
        mask_ch = ctrl_full[0].numpy()
        mask_rgb = (np.stack([mask_ch] * 3, axis=-1) * 255).astype(np.uint8)
        vis_data = dict(
            mask_rgb=mask_rgb,
            residuals=residuals,
            attn_maps=attn_maps,
            ctrl_full=ctrl_full.numpy(),          # [C, H, W] float32
            active_channels=active_channels,
        )

    return gen_np, vis_data


# ── Ablation grid ─────────────────────────────────────────────────────────────

def _generate_ablation_images(
    tile_id: str,
    models: dict,
    config,
    scheduler,
    uni_embeds: torch.Tensor,
    device: str,
) -> list:
    """Return list of (label, gen_np) for progressive group addition."""
    from tools.channel_group_utils import split_channels_to_groups
    from train_scripts.inference_controlnet import denoise

    active_channels = config.data.active_channels
    vae_scale = config.scale_factor
    vae_shift  = config.shift_factor
    dtype = torch.float16 if device == "cuda" else torch.float32

    vae        = models["vae"]
    controlnet = models["controlnet"]
    base_model = models["base_model"]
    tme_module = models["tme_module"]

    vae.to(device=device, dtype=dtype).eval()

    ctrl_full = load_exp_channels(tile_id, active_channels, config.image_size)
    cell_mask_img = ctrl_full[0:1].unsqueeze(0).repeat(1, 3, 1, 1)
    cell_mask_img = 2 * (cell_mask_img - 0.5)
    with torch.no_grad():
        vae_mask = vae.encode(
            cell_mask_img.to(device, dtype=dtype)
        ).latent_dist.mean
        vae_mask = (vae_mask - vae_shift) * vae_scale

    tme_dict = split_channels_to_groups(
        ctrl_full.unsqueeze(0).to(device, dtype=dtype),
        active_channels,
        config.channel_groups,
    )

    group_names = [g["name"] for g in config.channel_groups]
    ablation_images = []

    torch.manual_seed(SEED)
    latent_shape = (1, 16, config.image_size // 8, config.image_size // 8)
    fixed_noise = torch.randn(latent_shape, device=device, dtype=dtype)
    fixed_noise = fixed_noise * scheduler.init_noise_sigma

    for n in range(len(group_names) + 1):
        if n == 0:
            label  = "Mask only\n(no TME groups)"
            active = set()
        else:
            active = set(group_names[:n])
            label  = "Groups:\n" + "\n".join(group_names[:n])

        with torch.no_grad():
            fused = (
                tme_module(vae_mask, tme_dict, active_groups=active)
                if active else vae_mask.clone()
            )

        denoised = denoise(
            latents=fixed_noise.clone(),
            uni_embeds=uni_embeds.to(device, dtype=dtype),
            controlnet_input_latent=fused,
            scheduler=scheduler,
            controlnet_model=controlnet,
            pixcell_controlnet_model=base_model,
            guidance_scale=GUIDANCE_SCALE,
            device=device,
        )
        with torch.no_grad():
            scaled = (denoised.to(dtype) / vae_scale) + vae_shift
            gen = vae.decode(scaled, return_dict=False)[0]
        gen = (gen / 2 + 0.5).clamp(0, 1)
        gen_np = (gen.cpu().permute(0, 2, 3, 1).numpy()[0] * 255).astype(np.uint8)
        ablation_images.append((label, gen_np))

    return ablation_images


# ── Visualization helpers ─────────────────────────────────────────────────────

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from tools.color_constants import (
    CHANNEL_CMAP as _CHANNEL_CMAP,
    CHANNEL_LABEL as _CHANNEL_LABEL,
    SECTION_BG, SECTION_TEXT,
)


def _header_ax(ax, label, section_key):
    """Render a section header on a thin axes row."""
    ax.set_facecolor(SECTION_BG[section_key])
    ax.text(0.5, 0.5, label,
            ha="center", va="center", fontsize=10, fontweight="bold",
            color=SECTION_TEXT[section_key], transform=ax.transAxes)
    ax.axis("off")


def _image_ax(ax, img, label, section_key, cmap=None, fontsize=8, cosine_sim_val=None):
    """Render a single image panel (row 1 of overview)."""
    ax.set_facecolor(SECTION_BG[section_key])
    vmax = 1.0 if (img.ndim == 2 or img.dtype != np.uint8) else None
    ax.imshow(img, cmap=cmap, vmin=0, vmax=vmax)
    title = label
    if cosine_sim_val is not None:
        title += f"\ncos sim={cosine_sim_val:.3f}"
    ax.set_title(title, fontsize=fontsize, fontweight="bold",
                 color=SECTION_TEXT[section_key], pad=3)
    ax.axis("off")


def save_overview_figure(
    ctrl_full: np.ndarray,          # [C, H, W] float32
    active_channels: list,
    gen_np: np.ndarray,             # uint8 [H, W, 3]
    save_path: Path,
    style_inputs: list | None = None,   # [(label, image), ...] prepended as style inputs
    cosine_sim_val: float | None = None,
):
    """
    3-row GridSpec overview (inputs split across 2 rows, output spans both).

    Row 0 (thin header): INPUTS header | ▶ | OUTPUT header
    Row 1 (images):  [style H&E purple] [TME ch 0..N/2] | ▶ | [Generated H&E  ↕ spans rows 1&2]
    Row 2 (images):  [TME ch N/2..end]                  | ↕ | [                               ]

    style_inputs: shown at the start of row 1 with style_ref (purple) coloring.
                  For paired inference the style tile = layout tile; for unpaired they differ.
    """
    style_inputs = style_inputs or []
    n_style   = len(style_inputs)
    n_ch      = len(active_channels)
    n_total   = n_style + n_ch

    # Split inputs across 2 rows: ceil(n_total/2) in row 1, rest in row 2.
    # Style panels always go into row 1.
    n_row1    = (n_total + 1) // 2
    n_ch_r1   = n_row1 - n_style   # TME channels in row 1
    n_ch_r2   = n_ch - n_ch_r1     # TME channels in row 2

    # Grid columns: n_row1 input cols + narrow arrow + output
    n_cols  = n_row1 + 1 + 1
    ratios  = [1.0] * n_row1 + [0.25] + [1.0]
    fig_w   = max(n_cols * 2.2, 8)
    fig_h   = 5.5

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
    gs = gridspec.GridSpec(
        3, n_cols, figure=fig,
        width_ratios=ratios, height_ratios=[0.07, 0.465, 0.465],
        wspace=0.05, hspace=0.08,
        left=0.01, right=0.99, top=0.97, bottom=0.02,
    )

    # ── Row 0: section headers ──────────────────────────────────────────────
    inp_hdr_text = (
        "INPUTS  (style H&E + TME layout channels)"
        if n_style > 0 else "INPUTS  (TME channels)"
    )
    _header_ax(fig.add_subplot(gs[0, 0:n_row1]), inp_hdr_text, "input")
    fig.add_subplot(gs[0, n_row1]).axis("off")
    _header_ax(fig.add_subplot(gs[0, n_row1 + 1]), "OUTPUT", "output")

    # ── Row 1: style panels + first half of TME channels ────────────────────
    for j, (lbl, img) in enumerate(style_inputs):
        _image_ax(fig.add_subplot(gs[1, j]), img, lbl, "style_ref", fontsize=7)
    for i in range(n_ch_r1):
        ch = active_channels[i]
        _image_ax(fig.add_subplot(gs[1, n_style + i]),
                  ctrl_full[i], _CHANNEL_LABEL.get(ch, ch), "input",
                  cmap=_CHANNEL_CMAP.get(ch), fontsize=7)

    # ── Row 2: second half of TME channels (left-aligned) ───────────────────
    for i in range(n_ch_r2):
        ch = active_channels[n_ch_r1 + i]
        _image_ax(fig.add_subplot(gs[2, i]),
                  ctrl_full[n_ch_r1 + i], _CHANNEL_LABEL.get(ch, ch), "input",
                  cmap=_CHANNEL_CMAP.get(ch), fontsize=7)

    # ── Arrow spanning both image rows ───────────────────────────────────────
    ax_arr = fig.add_subplot(gs[1:3, n_row1])
    ax_arr.text(0.5, 0.5, "▶", ha="center", va="center",
                fontsize=24, color="#555555", transform=ax_arr.transAxes)
    ax_arr.axis("off")

    # ── Output spanning both image rows ──────────────────────────────────────
    _image_ax(fig.add_subplot(gs[1:3, n_row1 + 1]),
              gen_np, "Generated H&E", "output",
              fontsize=9, cosine_sim_val=cosine_sim_val)

    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Overview figure saved → {save_path}")


def _titled_ax(ax, img, section_key, title, cmap=None, fontsize=9):
    """Image panel with plain title (no colored bbox)."""
    ax.set_facecolor(SECTION_BG[section_key])
    vmax = 1.0 if (img.ndim == 2 or img.dtype != np.uint8) else None
    ax.imshow(img, cmap=cmap, vmin=0, vmax=vmax)
    ax.set_title(title, fontsize=fontsize, fontweight="bold",
                 color=SECTION_TEXT[section_key], pad=4)
    ax.axis("off")


def save_enhanced_attention_figure(
    ctrl_full: np.ndarray,
    active_channels: list,
    gen_np: np.ndarray,
    attn_maps: dict,
    save_path: Path,
    style_inputs: list | None = None,   # [(label, image), ...] prepended as style inputs
    spatial_size: tuple = (32, 32),
    output_resolution: int = 256,
):
    """
    Layout (2 rows):
      Row 0 headers: INPUTS (style H&E + cell mask) | OUTPUT | ── | ATTENTION (per group)
      Row 1 images:  [style img purple] [cell mask blue] | gen H&E | ── | heatmaps on mask
    """
    from tools.visualize_group_attention import compute_attention_heatmaps
    heatmaps = compute_attention_heatmaps(attn_maps, spatial_size, output_resolution)
    style_inputs = style_inputs or []

    n_style  = len(style_inputs)
    # Left columns: style panels + cell mask + gen H&E
    n_left   = n_style + 2
    n_attn   = len(heatmaps)
    has_attn = n_attn > 0
    # +1 divider col; +1 dedicated colorbar col (keeps heatmap axes full-width)
    n_cols   = n_left + 1 + n_attn + (1 if has_attn else 0)

    ratios = [1.0] * n_left + [0.08] + [1.0] * n_attn + ([0.08] if has_attn else [])
    fig = plt.figure(figsize=(n_cols * 2.6, 5.0), facecolor="white")
    gs = gridspec.GridSpec(2, n_cols, figure=fig,
                           width_ratios=ratios, height_ratios=[0.13, 0.87],
                           wspace=0.05, hspace=0.08,
                           left=0.01, right=0.99, top=0.97, bottom=0.02)

    mask_img = ctrl_full[active_channels.index("cell_masks")]

    # ── Row 0: headers ──────────────────────────────────────────────────────
    inp_hdr_text = (
        "INPUTS  (style H&E + cell mask)"
        if n_style > 0 else "INPUT"
    )
    # INPUTS header spans style panels + cell mask
    _header_ax(fig.add_subplot(gs[0, 0:n_style + 1]), inp_hdr_text, "input")
    _header_ax(fig.add_subplot(gs[0, n_style + 1]), "OUTPUT", "output")
    fig.add_subplot(gs[0, n_left]).axis("off")   # divider
    if has_attn:
        _header_ax(fig.add_subplot(gs[0, n_left + 1: n_left + 1 + n_attn]),
                   "ATTENTION  (per TME group)", "analysis")
        fig.add_subplot(gs[0, -1]).axis("off")   # colorbar column header

    # ── Row 1: images ───────────────────────────────────────────────────────
    for j, (lbl, img) in enumerate(style_inputs):
        _titled_ax(fig.add_subplot(gs[1, j]), img, "style_ref", lbl)
    _titled_ax(fig.add_subplot(gs[1, n_style]), mask_img, "input", "Cell Mask", cmap="gray")
    _titled_ax(fig.add_subplot(gs[1, n_style + 1]), gen_np, "output", "Generated H&E")
    fig.add_subplot(gs[1, n_left]).axis("off")

    last_im = None
    for k, (name, hmap) in enumerate(heatmaps.items()):
        ax = fig.add_subplot(gs[1, n_left + 1 + k])
        ax.set_facecolor(SECTION_BG["analysis"])
        last_im = ax.imshow(hmap, cmap="jet", vmin=0, vmax=1)
        ax.set_title(name, fontsize=8, fontweight="bold",
                     color=SECTION_TEXT["analysis"], pad=4)
        ax.axis("off")
    if last_im is not None:
        cbar_ax = fig.add_subplot(gs[1, -1])
        cbar = fig.colorbar(last_im, cax=cbar_ax)
        cbar.set_label("Attention weight", fontsize=7)
        cbar.ax.tick_params(labelsize=7)

    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Attention heatmaps saved → {save_path}")


def save_enhanced_residual_figure(
    ctrl_full: np.ndarray,
    active_channels: list,
    gen_np: np.ndarray,
    residuals: dict,
    save_path: Path,
    refs: list | None = None,       # [(section_key, label, image), ...]
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
    n_res   = len(res_maps)
    n_cols  = n_fixed + 1 + n_res

    ratios = [1.0] * n_fixed + [0.08] + [1.0] * n_res
    fig = plt.figure(figsize=(n_cols * 2.6, 5.0), facecolor="white")
    gs = gridspec.GridSpec(2, n_cols, figure=fig,
                           width_ratios=ratios, height_ratios=[0.13, 0.87],
                           wspace=0.05, hspace=0.08,
                           left=0.01, right=0.99, top=0.97, bottom=0.02)

    mask_img = ctrl_full[active_channels.index("cell_masks")]

    _header_ax(fig.add_subplot(gs[0, 0]), "INPUT", "input")
    _header_ax(fig.add_subplot(gs[0, 1]), "OUTPUT", "output")
    for j, (sk, lbl, _) in enumerate(refs):
        _header_ax(fig.add_subplot(gs[0, 2 + j]), lbl, sk)
    fig.add_subplot(gs[0, n_fixed]).axis("off")
    if n_res:
        _header_ax(fig.add_subplot(gs[0, n_fixed + 1:]), "RESIDUALS  ‖Δ_group‖", "analysis")

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
        ax.set_title(f"‖Δ_{name}‖", fontsize=8, fontweight="bold",
                     color=SECTION_TEXT["analysis"], pad=4,
                     bbox=dict(boxstyle="round,pad=0.25",
                               facecolor=SECTION_BG["analysis"],
                               edgecolor="none", alpha=0.9))
        ax.axis("off")

    if last_im is not None:
        fig.colorbar(last_im, ax=fig.axes[-1], fraction=0.046, pad=0.04, label="L2 norm")

    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Residual magnitude maps saved → {save_path}")


def save_enhanced_ablation_grid(
    ablation_images: list,          # [(label, gen_np), ...]
    save_path: Path,
    refs: list | None = None,       # [(section_key, label, image), ...] prepended
):
    """
    2-row ablation grid. Row 0: section headers. Row 1: images.
    refs panels are prepended (e.g. reference H&E for comparison).
    Then: Mask only → +group1 → ... → all groups.
    """
    refs = refs or []

    # Build full panel list: (section_key, label, image)
    panels = list(refs)
    for i, (label, img) in enumerate(ablation_images):
        sk = "input" if i == 0 else "output"
        panels.append((sk, label, img))

    n = len(panels)
    fig = plt.figure(figsize=(n * 2.5, 5.0), facecolor="white")
    gs = gridspec.GridSpec(2, n, figure=fig,
                           height_ratios=[0.13, 0.87],
                           wspace=0.05, hspace=0.08,
                           left=0.01, right=0.99, top=0.97, bottom=0.02)

    for j, (sk, lbl, img) in enumerate(panels):
        _header_ax(fig.add_subplot(gs[0, j]), lbl, sk)
        _titled_ax(fig.add_subplot(gs[1, j]), img, sk, lbl, fontsize=7)

    # Dashed separator after ref panels
    n_refs = len(refs)
    if n_refs > 0 and n_refs < n:
        ax_last_ref = fig.axes[n_refs * 2 - 1]   # row-1 axis of last ref
        ax_first_abl = fig.axes[n_refs * 2 + 1]  # row-1 axis of first ablation
        x_sep = (ax_last_ref.get_position().x1 +
                 ax_first_abl.get_position().x0) / 2
        fig.add_artist(plt.Line2D([x_sep, x_sep], [0.02, 0.98],
                                  transform=fig.transFigure,
                                  color="#aaaaaa", linewidth=1.5, linestyle="--"))

    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Ablation grid saved → {save_path}")


# ── UNI extraction ────────────────────────────────────────────────────────────

def get_uni_extractor(config, device: str):
    from pipeline.extract_features import UNI2hExtractor
    uni_model_path = getattr(config, "uni_model_path", "./pretrained_models/uni-2h")
    return UNI2hExtractor(model_path=uni_model_path, device=device)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    sys.path.insert(0, str(ROOT))
    os.chdir(ROOT)

    from diffusion.utils.misc import read_config
    from train_scripts.inference_controlnet import null_uni_embed

    print(f"Device: {DEVICE}")
    print(f"Output: {OUT_DIR}")

    # Load config
    config = read_config(str(CONFIG_PATH))
    config._filename = str(CONFIG_PATH)

    # Collect all available tile IDs from exp_channels/cell_masks
    mask_dir = EXP_CH_DIR / MASK_CHANNEL
    all_ids = sorted(p.stem for p in mask_dir.glob("*.png"))
    print(f"Found {len(all_ids)} tiles in exp_channels/{MASK_CHANNEL}")

    # Split into inference and validation sets
    selected = random.sample(all_ids, N_INFERENCE + N_VALIDATION)
    inference_ids  = selected[:N_INFERENCE]
    validation_ids = selected[N_INFERENCE:]
    print(f"Inference tiles:  {N_INFERENCE}  | Validation tiles: {N_VALIDATION}")

    # Output dirs
    inf_dir = OUT_DIR / "inference"
    val_dir = OUT_DIR / "validation"
    vis_dir = OUT_DIR / "visualizations"
    for d in [inf_dir, val_dir, vis_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Load models
    models = load_all_models(config, DEVICE)

    # Scheduler
    scheduler = DDPMScheduler(
        num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02,
        beta_schedule="linear", prediction_type="epsilon", clip_sample=False,
    )
    scheduler.set_timesteps(NUM_STEPS, device=DEVICE)

    # Null UNI embedding (TME-only mode)
    uni_embeds = null_uni_embed(device="cpu", dtype=torch.float32)

    # ── Initialize UNI extractor for cosine similarity ──────────────────────
    print("\nLoading UNI extractor for validation metrics...")
    uni_extractor = get_uni_extractor(config, DEVICE)

    # ── INFERENCE SET ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("INFERENCE SET (paired exp data)")
    print(f"{'='*60}")
    inf_cosine_sims = []

    for i, tid in enumerate(inference_ids):
        do_vis = (i < N_VIS_TILES)
        print(f"[{i+1:02d}/{N_INFERENCE}] {tid}", end="")

        gen_np, vis_data = generate_tile(
            tile_id=tid,
            models=models,
            config=config,
            scheduler=scheduler,
            uni_embeds=uni_embeds,
            device=DEVICE,
            return_vis_data=do_vis,
        )

        # Save generated H&E
        out_path = inf_dir / f"{tid}_generated.png"
        Image.fromarray(gen_np).save(out_path)

        # Load reference exp H&E (if available)
        exp_he_path = HE_DIR / f"{tid}.png"
        ref_he = np.array(Image.open(exp_he_path).convert("RGB")) if exp_he_path.exists() else None

        # Cosine similarity
        sim = None
        exp_feat_path = FEAT_DIR / f"{tid}_uni.npy"
        if exp_feat_path.exists():
            exp_feat = np.load(exp_feat_path)
            gen_feat = uni_extractor.extract(gen_np)
            sim = cosine_sim(gen_feat, exp_feat)
            inf_cosine_sims.append(sim)
            print(f"  cosine_sim={sim:.4f}")
        else:
            print("  (no exp feat)")

        # Visualizations for first N_VIS_TILES tiles
        if do_vis and vis_data is not None:
            tile_vis_dir = vis_dir / f"inference_{tid}"
            tile_vis_dir.mkdir(parents=True, exist_ok=True)

            ctrl_full_np    = vis_data["ctrl_full"]
            active_channels = vis_data["active_channels"]
            # H&E is both the style input and the ground-truth reference for this tile
            style_inp = ([("H&E (style)", ref_he)] if ref_he is not None else [])

            # 1. Overview: [style H&E | TME channels] → Generated H&E
            save_overview_figure(
                ctrl_full=ctrl_full_np,
                active_channels=active_channels,
                gen_np=gen_np,
                style_inputs=style_inp,
                save_path=tile_vis_dir / "overview.png",
                cosine_sim_val=sim,
            )

            # 2. Attention heatmaps per group
            save_enhanced_attention_figure(
                ctrl_full=ctrl_full_np,
                active_channels=active_channels,
                gen_np=gen_np,
                attn_maps=vis_data["attn_maps"],
                style_inputs=style_inp,
                save_path=tile_vis_dir / "attention_heatmaps.png",
            )

            # 3. Residual magnitude per group (TME group contributions, no ref)
            save_enhanced_residual_figure(
                ctrl_full=ctrl_full_np,
                active_channels=active_channels,
                gen_np=gen_np,
                residuals=vis_data["residuals"],
                refs=[],
                save_path=tile_vis_dir / "residual_magnitudes.png",
            )

            # 4. Ablation grid (progressive group addition)
            print(f"  Generating ablation grid...")
            ablation_imgs = _generate_ablation_images(
                tile_id=tid, models=models, config=config,
                scheduler=scheduler, uni_embeds=uni_embeds, device=DEVICE,
            )
            save_enhanced_ablation_grid(
                ablation_images=ablation_imgs,
                refs=[("style_ref", "H&E (style)", ref_he)] if ref_he is not None else [],
                save_path=tile_vis_dir / "ablation_grid.png",
            )

    # ── VALIDATION SET (unpaired) ────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("VALIDATION SET (unpaired exp data)")
    print(f"{'='*60}")
    val_cosine_sims = []

    for i, tid in enumerate(validation_ids):
        print(f"[{i+1:02d}/{N_VALIDATION}] {tid}", end="")

        gen_np, _ = generate_tile(
            tile_id=tid,
            models=models,
            config=config,
            scheduler=scheduler,
            uni_embeds=uni_embeds,
            device=DEVICE,
            return_vis_data=False,
        )

        out_path = val_dir / f"{tid}_generated.png"
        Image.fromarray(gen_np).save(out_path)

        exp_feat_path = FEAT_DIR / f"{tid}_uni.npy"
        if exp_feat_path.exists():
            exp_feat = np.load(exp_feat_path)
            gen_feat = uni_extractor.extract(gen_np)
            sim = cosine_sim(gen_feat, exp_feat)
            val_cosine_sims.append(sim)
            print(f"  cosine_sim={sim:.4f}")
        else:
            print("  (no exp feat)")

    # ── UNPAIRED INFERENCE ───────────────────────────────────────────────────
    # Layout from tile A + style (UNI embedding) from a different tile B.
    # Shows model's ability to apply an arbitrary H&E style to a given TME layout.
    print(f"\n{'='*60}")
    print("UNPAIRED INFERENCE  (A's TME layout × B's H&E style)")
    print(f"{'='*60}")

    unpaired_dir = OUT_DIR / "unpaired"
    unpaired_vis_dir = OUT_DIR / "visualizations_unpaired"
    unpaired_dir.mkdir(parents=True, exist_ok=True)
    unpaired_vis_dir.mkdir(parents=True, exist_ok=True)

    # Pool for style tiles: all IDs not in inference_ids
    style_pool = [x for x in all_ids if x not in set(inference_ids)]
    N_UNPAIRED = min(N_VIS_TILES, len(inference_ids))   # one vis per layout tile
    unpaired_cosine_sims = []

    for i in range(N_UNPAIRED):
        tid_A = inference_ids[i]
        tid_B = random.choice(style_pool)
        print(f"[{i+1:02d}/{N_UNPAIRED}] layout={tid_A}  style={tid_B}", end="")

        # Load B's precomputed UNI embedding as style vector
        uni_path_B = FEAT_DIR / f"{tid_B}_uni.npy"
        if not uni_path_B.exists():
            print("  (no UNI for B, skipping)")
            continue
        uni_feat_B = np.load(uni_path_B)
        uni_B = torch.from_numpy(uni_feat_B).view(1, 1, 1, 1536)

        gen_np, vis_data = generate_tile(
            tile_id=tid_A,
            models=models,
            config=config,
            scheduler=scheduler,
            uni_embeds=uni_B,
            device=DEVICE,
            return_vis_data=True,
        )

        out_path = unpaired_dir / f"{tid_A}_layout_{tid_B}_style.png"
        Image.fromarray(gen_np).save(out_path)

        # Cosine sim vs. A's own ground truth (layout fidelity)
        sim = None
        exp_feat_path = FEAT_DIR / f"{tid_A}_uni.npy"
        if exp_feat_path.exists():
            exp_feat = np.load(exp_feat_path)
            gen_feat = uni_extractor.extract(gen_np)
            sim = cosine_sim(gen_feat, exp_feat)
            unpaired_cosine_sims.append(sim)
            print(f"  cos_sim(vs A)={sim:.4f}")
        else:
            print()

        # Full visualization showing both references
        ref_A = np.array(Image.open(HE_DIR / f"{tid_A}.png").convert("RGB")) if (HE_DIR / f"{tid_A}.png").exists() else None
        ref_B = np.array(Image.open(HE_DIR / f"{tid_B}.png").convert("RGB")) if (HE_DIR / f"{tid_B}.png").exists() else None

        if vis_data is not None:
            tile_vis_dir = unpaired_vis_dir / f"{tid_A}_x_{tid_B}"
            tile_vis_dir.mkdir(parents=True, exist_ok=True)

            ctrl_np = vis_data["ctrl_full"]
            act_ch  = vis_data["active_channels"]

            # B's H&E is the style input (its UNI embedding drives generation)
            style_inp_B = ([("H&E (style from B)", ref_B)] if ref_B is not None else [])

            # Overview: [B's style H&E | A's TME channels] → Generated H&E
            save_overview_figure(
                ctrl_full=ctrl_np,
                active_channels=act_ch,
                gen_np=gen_np,
                style_inputs=style_inp_B,
                save_path=tile_vis_dir / "overview.png",
                cosine_sim_val=sim,
            )
            save_enhanced_attention_figure(
                ctrl_full=ctrl_np, active_channels=act_ch,
                gen_np=gen_np, attn_maps=vis_data["attn_maps"],
                style_inputs=style_inp_B,
                save_path=tile_vis_dir / "attention_heatmaps.png",
            )
            save_enhanced_residual_figure(
                ctrl_full=ctrl_np, active_channels=act_ch,
                gen_np=gen_np, residuals=vis_data["residuals"],
                refs=[],
                save_path=tile_vis_dir / "residual_magnitudes.png",
            )

            # 4. Ablation grid — progressive group addition with B's style
            print(f"  Generating unpaired ablation grid...")
            ablation_imgs = _generate_ablation_images(
                tile_id=tid_A, models=models, config=config,
                scheduler=scheduler, uni_embeds=uni_B, device=DEVICE,
            )
            save_enhanced_ablation_grid(
                ablation_images=ablation_imgs,
                refs=[("style_ref", "H&E (style from B)", ref_B)] if ref_B is not None else [],
                save_path=tile_vis_dir / "ablation_grid.png",
            )

    # ── SUMMARY ─────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    results = {
        "checkpoint": str(CKPT_DIR),
        "guidance_scale": GUIDANCE_SCALE,
        "num_steps": NUM_STEPS,
        "inference_paired": {
            "mode": "TME-only (null UNI)",
            "n_tiles": N_INFERENCE,
            "tile_ids": inference_ids,
            "cosine_sims": inf_cosine_sims,
            "mean_cosine_sim": float(np.mean(inf_cosine_sims)) if inf_cosine_sims else None,
            "std_cosine_sim":  float(np.std(inf_cosine_sims))  if inf_cosine_sims else None,
        },
        "validation_paired": {
            "mode": "TME-only (null UNI)",
            "n_tiles": N_VALIDATION,
            "tile_ids": validation_ids,
            "cosine_sims": val_cosine_sims,
            "mean_cosine_sim": float(np.mean(val_cosine_sims)) if val_cosine_sims else None,
            "std_cosine_sim":  float(np.std(val_cosine_sims))  if val_cosine_sims else None,
        },
        "inference_unpaired": {
            "mode": "style-conditioned (B's UNI → A's channels)",
            "n_tiles": N_UNPAIRED,
            "cosine_sims_vs_layout_A": unpaired_cosine_sims,
            "mean_cosine_sim": float(np.mean(unpaired_cosine_sims)) if unpaired_cosine_sims else None,
            "std_cosine_sim":  float(np.std(unpaired_cosine_sims))  if unpaired_cosine_sims else None,
        },
    }

    metrics_path = OUT_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)

    def _fmt(d):
        m, s = d["mean_cosine_sim"], d["std_cosine_sim"]
        return f"{m:.4f} ± {s:.4f}" if m is not None else "n/a"

    print(f"\nInference  (paired,   TME-only)  — mean UNI cos sim: {_fmt(results['inference_paired'])}")
    print(f"Validation (paired,   TME-only)  — mean UNI cos sim: {_fmt(results['validation_paired'])}")
    print(f"Inference  (unpaired, style-cond) — mean UNI cos sim: {_fmt(results['inference_unpaired'])}")
    print(f"\nOutputs saved to: {OUT_DIR}")
    print(f"Metrics:          {metrics_path}")
    print(f"Visualizations:   {vis_dir}")
    print(f"Unpaired vis:     {unpaired_vis_dir}")


if __name__ == "__main__":
    main()
