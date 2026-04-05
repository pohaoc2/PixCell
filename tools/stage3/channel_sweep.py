"""Channel impact sweep experiments for experimental TME channels.

This script supports three analyses:
1. Microenvironment 2D sweeps over oxygen and glucose.
2. Cell-type relabeling on near-pure tiles.
3. Cell-state relabeling on near-pure tiles.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.channel_group_utils import channel_index_map
from tools.color_constants import CELL_STATE_COLORS, CELL_TYPE_COLORS
from tools.stage3.common import (
    inference_dtype,
    make_inference_scheduler,
    resolve_uni_embedding,
    to_uint8_rgb,
)

SWEEP_SCALES: list[float] = [0.0, 0.25, 0.5, 0.75, 1.0]
CACHE_VERSION = 1

_CELL_TYPE_CHANNELS = {
    "cancer": "cell_type_cancer",
    "immune": "cell_type_immune",
    "healthy": "cell_type_healthy",
}

_CELL_STATE_CHANNELS = {
    "prolif": "cell_state_prolif",
    "nonprolif": "cell_state_nonprolif",
    "dead": "cell_state_dead",
}

def _rgb(rgba: tuple[int, int, int, int]) -> tuple[int, int, int]:
    return rgba[:3]


_CELL_TYPE_THUMB_SPECS = {
    "cancer": ("cell_type_cancer", _rgb(CELL_TYPE_COLORS["cancer"])),
    "immune": ("cell_type_immune", _rgb(CELL_TYPE_COLORS["immune"])),
    "healthy": ("cell_type_healthy", _rgb(CELL_TYPE_COLORS["healthy"])),
}

_CELL_STATE_THUMB_SPECS = {
    "prolif": ("cell_state_prolif", _rgb(CELL_STATE_COLORS["proliferative"])),
    "nonprolif": ("cell_state_nonprolif", _rgb(CELL_STATE_COLORS["nonprolif"])),
    "dead": ("cell_state_dead", _rgb(CELL_STATE_COLORS["dead"])),
}


def _get_dtype(device: str) -> torch.dtype:
    return inference_dtype(device)


def _make_scheduler(*, num_steps: int, device: str):
    return make_inference_scheduler(num_steps=num_steps, device=device)


def _resolve_checkpoint_dir(checkpoint_dir: Path) -> Path:
    from tools.stage3.tile_pipeline import find_latest_checkpoint_dir

    if any(checkpoint_dir.glob("controlnet_*.pth")):
        return checkpoint_dir
    return find_latest_checkpoint_dir(checkpoint_dir)


def _source_labels_from_results(results: dict[str, dict[str, np.ndarray]]) -> list[str]:
    labels = list(results.keys())
    if labels:
        return labels
    return []


def _target_labels_from_results(results: dict[str, dict[str, np.ndarray]]) -> list[str]:
    for row in results.values():
        labels = list(row.keys())
        if labels:
            return labels
    return []


def _blank_rgb(size: int = 96, value: int = 35) -> np.ndarray:
    return np.full((size, size, 3), value, dtype=np.uint8)


def _save_rgb_png(image: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(to_uint8_rgb(image, value_range="byte")).save(path)


def _load_rgb_png(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _exp1_condition_slug(o2_scale: float, glucose_scale: float) -> str:
    return f"o2_{o2_scale:.2f}__glucose_{glucose_scale:.2f}"


def _render_label_badge(label: str, rgb: tuple[int, int, int], *, resolution: int = 96) -> np.ndarray:
    badge = np.zeros((resolution, resolution, 3), dtype=np.uint8)
    badge[...] = np.array(rgb, dtype=np.uint8)
    yy, xx = np.ogrid[:resolution, :resolution]
    dist = np.sqrt((xx - resolution / 2) ** 2 + (yy - resolution / 2) ** 2)
    vignette = np.clip(1.0 - dist / (resolution * 0.9), 0.35, 1.0)
    return np.clip(badge.astype(np.float32) * vignette[..., None], 0, 255).astype(np.uint8)


def _render_group_thumbnail(
    *,
    exp_channels_dir: Path,
    tile_id: str,
    label: str,
    thumb_specs: dict[str, tuple[str, tuple[int, int, int]]],
    resolution: int = 96,
) -> np.ndarray:
    from tools.stage3.ablation_vis_utils import load_exp_channel_plane

    channel_name, rgb = thumb_specs[label]
    try:
        plane = load_exp_channel_plane(
            exp_channels_dir,
            channel_name,
            tile_id,
            resolution=resolution,
        )
    except (FileNotFoundError, OSError, ValueError, KeyError):
        return _render_label_badge(label, rgb, resolution=resolution)

    out = np.zeros((resolution, resolution, 3), dtype=np.float32)
    color = np.array(rgb, dtype=np.float32) / 255.0
    out += plane[..., None] * color
    return (np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)


def _render_relabel_input_thumbnail(
    *,
    exp_channels_dir: Path,
    tile_id: str,
    source_label: str,
    target_label: str,
    thumb_specs: dict[str, tuple[str, tuple[int, int, int]]],
    resolution: int = 96,
) -> np.ndarray:
    from tools.stage3.ablation_vis_utils import load_exp_channel_plane

    src_channel, _ = thumb_specs[source_label]
    _, target_rgb = thumb_specs[target_label]
    try:
        plane = load_exp_channel_plane(
            exp_channels_dir,
            src_channel,
            tile_id,
            resolution=resolution,
        )
    except (FileNotFoundError, OSError, ValueError, KeyError):
        return _render_label_badge(target_label, target_rgb, resolution=resolution)

    out = np.zeros((resolution, resolution, 3), dtype=np.float32)
    out += plane[..., None] * (np.array(target_rgb, dtype=np.float32) / 255.0)
    return (np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)


def build_scaled_ctrl(ctrl_full: torch.Tensor, channel_idx: int, scale: float) -> torch.Tensor:
    """Return a cloned control tensor with one channel scaled."""
    ctrl = ctrl_full.clone()
    ctrl[channel_idx] = ctrl_full[channel_idx] * scale
    return ctrl


def build_2d_scaled_ctrl(
    ctrl_full: torch.Tensor,
    *,
    idx_o2: int,
    idx_glucose: int,
    o2_scale: float,
    glucose_scale: float,
) -> torch.Tensor:
    """Return a cloned control tensor with oxygen and glucose scaled independently."""
    ctrl = ctrl_full.clone()
    ctrl[idx_o2] = ctrl_full[idx_o2] * o2_scale
    ctrl[idx_glucose] = ctrl_full[idx_glucose] * glucose_scale
    return ctrl


def build_relabeled_ctrl(
    ctrl_full: torch.Tensor,
    *,
    idx_source: int,
    idx_target: int,
) -> torch.Tensor:
    """Move one channel's content into another and zero the source."""
    ctrl = ctrl_full.clone()
    ctrl[idx_target] = ctrl_full[idx_source].clone()
    ctrl[idx_source] = torch.zeros_like(ctrl_full[idx_source])
    return ctrl


def save_channel_sweep_manifest(cache_dir: Path, manifest: dict[str, Any]) -> Path:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def load_channel_sweep_manifest(cache_dir: Path) -> dict[str, Any]:
    cache_dir = Path(cache_dir)
    path = cache_dir / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


def save_channel_sweep_cache(
    *,
    cache_dir: Path,
    manifest: dict[str, Any] | None = None,
    tile_id: str | None = None,
    results: dict[str, Any] | None = None,
    images: dict[str, Any] | None = None,
    out_dir: Path | None = None,
) -> Path:
    del out_dir
    cache_dir = Path(cache_dir)
    payload = dict(manifest or {})
    payload.setdefault("version", CACHE_VERSION)
    if tile_id is not None:
        payload["tile_id"] = tile_id
    if "experiments" not in payload:
        payload["experiments"] = {}
        tree = results or images or {}
        for exp_name, exp_tree in tree.items():
            entries: list[dict[str, Any]] = []
            if isinstance(exp_tree, dict):
                for src_label, row in exp_tree.items():
                    if not isinstance(row, dict):
                        continue
                    for tgt_label, image in row.items():
                        rel = Path(str(exp_name)) / f"{src_label}__{tgt_label}.png"
                        _save_rgb_png(np.asarray(image), cache_dir / rel)
                        entries.append(
                            {
                                "source_label": str(src_label),
                                "target_label": str(tgt_label),
                                "image_path": rel.as_posix(),
                            }
                        )
            payload["experiments"][str(exp_name)] = {"entries": entries}
    return save_channel_sweep_manifest(cache_dir, payload)


def load_channel_sweep_cache(*, cache_dir: Path, cache_path: Path | None = None) -> dict[str, Any]:
    return load_channel_sweep_manifest(cache_path or cache_dir)


def save_exp1_microenv_cache(
    *,
    cache_dir: Path,
    tile_id: str,
    tile_class_label: str,
    images_grid: dict[tuple[float, float], np.ndarray],
) -> dict[str, Any]:
    exp_dir = Path(cache_dir) / "exp1_microenv" / f"{tile_class_label}_{tile_id}"
    items: list[dict[str, Any]] = []
    for (o2_scale, glucose_scale), image in sorted(images_grid.items()):
        rel = Path("exp1_microenv") / f"{tile_class_label}_{tile_id}" / f"{_exp1_condition_slug(o2_scale, glucose_scale)}.png"
        _save_rgb_png(image, Path(cache_dir) / rel)
        items.append(
            {
                "o2_scale": float(o2_scale),
                "glucose_scale": float(glucose_scale),
                "image_path": rel.as_posix(),
            }
        )
    return {
        "tile_id": tile_id,
        "tile_class_label": tile_class_label,
        "baseline": {"o2_scale": 1.0, "glucose_scale": 1.0},
        "items": items,
    }


def load_exp1_microenv_cache(
    cache_dir: Path,
    record: dict[str, Any],
) -> dict[tuple[float, float], np.ndarray]:
    grid: dict[tuple[float, float], np.ndarray] = {}
    for item in record.get("items", []):
        key = (float(item["o2_scale"]), float(item["glucose_scale"]))
        grid[key] = _load_rgb_png(Path(cache_dir) / item["image_path"])
    return grid


def save_relabeling_cache(
    *,
    cache_dir: Path,
    exp_name: str,
    results: dict[str, dict[str, np.ndarray]],
    tiles: dict[str, str],
    baseline_group_thumbs: dict[str, np.ndarray],
    input_thumbs: dict[str, dict[str, np.ndarray]],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    source_labels = list(results.keys())
    target_labels = _target_labels_from_results(results)
    for src_label in source_labels:
        images: dict[str, str] = {}
        input_thumb_paths: dict[str, str] = {}
        for tgt_label, image in results[src_label].items():
            rel = Path(exp_name) / src_label / f"{tgt_label}.png"
            _save_rgb_png(image, Path(cache_dir) / rel)
            images[tgt_label] = rel.as_posix()
            thumb = input_thumbs.get(src_label, {}).get(tgt_label)
            if thumb is not None:
                input_rel = Path(exp_name) / src_label / f"input__{tgt_label}.png"
                _save_rgb_png(thumb, Path(cache_dir) / input_rel)
                input_thumb_paths[tgt_label] = input_rel.as_posix()
        thumb_rel = Path(exp_name) / src_label / "baseline_group.png"
        _save_rgb_png(baseline_group_thumbs[src_label], Path(cache_dir) / thumb_rel)
        rows.append(
            {
                "source_label": src_label,
                "tile_id": tiles[src_label],
                "baseline_group_thumb_path": thumb_rel.as_posix(),
                "input_thumb_paths": input_thumb_paths,
                "images": images,
            }
        )
    return {
        "labels": source_labels,
        "source_labels": source_labels,
        "target_labels": target_labels,
        "rows": rows,
    }


def load_relabeling_cache(
    cache_dir: Path,
    record: dict[str, Any],
) -> dict[str, Any]:
    labels = [str(label) for label in record.get("labels", [])]
    source_labels = [str(label) for label in record.get("source_labels", labels)]
    target_labels = [str(label) for label in record.get("target_labels", labels)]
    results: dict[str, dict[str, np.ndarray]] = {}
    tiles: dict[str, str] = {}
    baseline_group_thumbs: dict[str, np.ndarray] = {}
    input_thumbs: dict[str, dict[str, np.ndarray]] = {}
    for row in record.get("rows", []):
        src_label = str(row["source_label"])
        tiles[src_label] = str(row["tile_id"])
        baseline_group_thumbs[src_label] = _load_rgb_png(
            Path(cache_dir) / row["baseline_group_thumb_path"]
        )
        input_thumbs[src_label] = {
            str(tgt_label): _load_rgb_png(Path(cache_dir) / rel_path)
            for tgt_label, rel_path in row.get("input_thumb_paths", {}).items()
        }
        results[src_label] = {
            str(tgt_label): _load_rgb_png(Path(cache_dir) / rel_path)
            for tgt_label, rel_path in row.get("images", {}).items()
        }
    return {
        "labels": labels,
        "source_labels": source_labels,
        "target_labels": target_labels,
        "tiles": tiles,
        "baseline_group_thumbs": baseline_group_thumbs,
        "input_thumbs": input_thumbs,
        "results": results,
    }


def load_sweep_models(
    config_path: str | Path,
    *,
    checkpoint_dir: Path,
    device: str,
    num_steps: int,
) -> tuple[dict[str, Any], Any, DDPMScheduler]:
    """Load config, models, and scheduler for channel sweep inference."""
    from diffusion.utils.misc import read_config
    from tools.stage3.tile_pipeline import load_all_models

    config_path = Path(config_path)
    ckpt_dir = _resolve_checkpoint_dir(Path(checkpoint_dir))

    os.chdir(ROOT)
    config = read_config(str(config_path))
    config._filename = str(config_path)
    models = load_all_models(config, str(config_path), ckpt_dir, device)
    scheduler = _make_scheduler(num_steps=num_steps, device=device)
    return models, config, scheduler


def generate_from_ctrl(
    ctrl_full: torch.Tensor,
    *,
    models: dict[str, Any],
    config: Any,
    scheduler: Any,
    uni_embeds: torch.Tensor,
    device: str,
    guidance_scale: float,
    fixed_noise: torch.Tensor,
    seed: int,
) -> np.ndarray:
    """Generate one H&E tile from a modified full control tensor."""
    from tools.stage3.tile_pipeline import generate_from_ctrl as generate_from_ctrl_shared

    gen_np, _ = generate_from_ctrl_shared(
        ctrl_full,
        models=models,
        config=config,
        scheduler=scheduler,
        uni_embeds=uni_embeds,
        device=device,
        guidance_scale=guidance_scale,
        seed=seed,
        fixed_noise=fixed_noise,
    )
    return gen_np


def run_exp1_microenv_grid(
    tile_id: str,
    *,
    exp_channels_dir: Path,
    feat_dir: Path,
    null_uni: bool,
    models: dict[str, Any],
    config: Any,
    scheduler: Any,
    device: str,
    guidance_scale: float,
    seed: int,
) -> dict[tuple[float, float], np.ndarray]:
    """Generate the full oxygen x glucose sweep for one tile."""
    from tools.stage3.tile_pipeline import _make_fixed_noise, load_exp_channels

    dtype = _get_dtype(device)
    ctrl_full = load_exp_channels(
        tile_id,
        config.data.active_channels,
        config.image_size,
        exp_channels_dir,
    )
    fixed_noise = _make_fixed_noise(
        config=config,
        scheduler=scheduler,
        device=device,
        dtype=dtype,
        seed=seed,
    )
    uni_embeds = resolve_uni_embedding(tile_id, feat_dir=feat_dir, null_uni=null_uni)

    results: dict[tuple[float, float], np.ndarray] = {}
    total = len(SWEEP_SCALES) ** 2
    channel_indices = channel_index_map(config.data.active_channels)
    idx_o2 = channel_indices["oxygen"]
    idx_glucose = channel_indices["glucose"]
    for i, o2_scale in enumerate(SWEEP_SCALES):
        for j, glucose_scale in enumerate(SWEEP_SCALES):
            step = i * len(SWEEP_SCALES) + j + 1
            print(f"  Exp1 [{step}/{total}] tile={tile_id} O2={o2_scale:.2f} glucose={glucose_scale:.2f}")
            ctrl = build_2d_scaled_ctrl(
                ctrl_full,
                idx_o2=idx_o2,
                idx_glucose=idx_glucose,
                o2_scale=o2_scale,
                glucose_scale=glucose_scale,
            )
            results[(o2_scale, glucose_scale)] = generate_from_ctrl(
                ctrl,
                models=models,
                config=config,
                scheduler=scheduler,
                uni_embeds=uni_embeds,
                device=device,
                guidance_scale=guidance_scale,
                fixed_noise=fixed_noise,
                seed=seed,
            )
    return results


def render_exp1_figure(
    images_grid: dict[tuple[float, float], np.ndarray],
    *,
    tile_id: str,
    tile_class_label: str,
    out_path: Path,
) -> None:
    """Render the 5x5 microenvironment sweep as separate generation and diff chunks."""
    n = len(SWEEP_SCALES)
    baseline = images_grid[(1.0, 1.0)].astype(np.float32)
    hot_cmap = mcolors.LinearSegmentedColormap.from_list(
        "hot4",
        ["#000000", "#ff4400", "#ffff00", "#ffffff"],
    )
    baseline_border = "#9B59B6"
    fig = plt.figure(figsize=(n * 2.25 + 1.6, n * 4.1 + 1.6))
    fig.suptitle(f"Microenv sweep - {tile_id} ({tile_class_label})", fontsize=11)
    gs = fig.add_gridspec(
        (n * 2) + 2,
        n + 1,
        hspace=0.12,
        wspace=0.04,
        height_ratios=[1] * n + [0.22] + [1] * n + [0.16],
        width_ratios=[0.2] + [1] * n,
    )

    ax_label_gen = fig.add_subplot(gs[:n, 0])
    ax_label_gen.axis("off")
    for i, scale in enumerate(SWEEP_SCALES):
        ax_label_gen.text(
            0.95,
            1 - (i + 0.5) / n,
            f"O2={scale:.2f}",
            ha="right",
            va="center",
            fontsize=7,
            transform=ax_label_gen.transAxes,
        )
    ax_label_gen.text(0.12, 1.02, "Generated H&E", fontsize=9, fontweight="bold", transform=ax_label_gen.transAxes)

    sep_ax = fig.add_subplot(gs[n, :])
    sep_ax.axis("off")

    ax_label_diff = fig.add_subplot(gs[n + 1 : (n * 2) + 1, 0])
    ax_label_diff.axis("off")
    for i, scale in enumerate(SWEEP_SCALES):
        ax_label_diff.text(
            0.95,
            1 - (i + 0.5) / n,
            f"O2={scale:.2f}",
            ha="right",
            va="center",
            fontsize=7,
            transform=ax_label_diff.transAxes,
        )
    ax_label_diff.text(0.12, 1.02, "Pixel diff vs baseline", fontsize=9, fontweight="bold", transform=ax_label_diff.transAxes)

    diff_max = 0.0
    diff_maps: dict[tuple[float, float], np.ndarray] = {}
    for o2_scale in SWEEP_SCALES:
        for glucose_scale in SWEEP_SCALES:
            img = images_grid[(o2_scale, glucose_scale)]
            diff = np.abs(img.astype(np.float32) - baseline).mean(axis=2)
            if o2_scale == 1.0 and glucose_scale == 1.0:
                diff = np.zeros_like(diff)
            diff_maps[(o2_scale, glucose_scale)] = diff
            diff_max = max(diff_max, float(diff.max()))
    diff_vmax = max(diff_max, 1.0)

    for i, o2_scale in enumerate(SWEEP_SCALES):
        for j, glucose_scale in enumerate(SWEEP_SCALES):
            ax = fig.add_subplot(gs[i, j + 1])
            img = images_grid[(o2_scale, glucose_scale)]
            ax.imshow(img)
            ax.axis("off")
            if i == 0:
                ax.set_title(f"Glucose={glucose_scale:.2f}", fontsize=7, pad=4)

            if o2_scale == 1.0 and glucose_scale == 1.0:
                ax.add_patch(
                    Rectangle(
                        (0, 0),
                        1,
                        1,
                        transform=ax.transAxes,
                        fill=False,
                        edgecolor=baseline_border,
                        linewidth=3.0,
                        linestyle="-",
                        clip_on=False,
                        zorder=10,
                    )
                )

            ax_diff = fig.add_subplot(gs[n + 1 + i, j + 1])
            diff = diff_maps[(o2_scale, glucose_scale)]
            im = ax_diff.imshow(diff, cmap=hot_cmap, vmin=0, vmax=diff_vmax)
            ax_diff.axis("off")
            if i == 0:
                ax_diff.set_title(f"Glucose={glucose_scale:.2f}", fontsize=7, pad=4)
            if o2_scale == 1.0 and glucose_scale == 1.0:
                ax_diff.add_patch(
                    Rectangle(
                        (0, 0),
                        1,
                        1,
                        transform=ax_diff.transAxes,
                        fill=False,
                        edgecolor=baseline_border,
                        linewidth=3.0,
                        linestyle="-",
                        clip_on=False,
                        zorder=10,
                    )
                )

    cbar_ax = fig.add_subplot(gs[(n * 2) + 1, 1:3])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
    cbar.ax.tick_params(labelsize=7, pad=1)
    cbar.set_label("Mean absolute pixel diff", fontsize=8, labelpad=2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.07, right=0.995, top=0.95, bottom=0.05)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _run_relabeling_experiment(
    tiles: dict[str, str],
    channel_map: dict[str, int],
    *,
    exp_channels_dir: Path,
    feat_dir: Path,
    null_uni: bool,
    models: dict[str, Any],
    config: Any,
    scheduler: Any,
    device: str,
    guidance_scale: float,
    seed: int,
) -> dict[str, dict[str, np.ndarray]]:
    """Run a generic source-label to target-label relabeling experiment."""
    from tools.stage3.tile_pipeline import _make_fixed_noise, load_exp_channels

    labels = list(channel_map.keys())
    dtype = _get_dtype(device)
    results: dict[str, dict[str, np.ndarray]] = {}
    channel_indices = channel_index_map(config.data.active_channels)
    resolved_channel_map = {
        label: channel_indices[channel_name]
        for label, channel_name in channel_map.items()
    }

    for src_label, tile_id in tiles.items():
        print(f"  Relabel tile {tile_id} (source={src_label})")
        ctrl_full = load_exp_channels(
            tile_id,
            config.data.active_channels,
            config.image_size,
            exp_channels_dir,
        )
        fixed_noise = _make_fixed_noise(
            config=config,
            scheduler=scheduler,
            device=device,
            dtype=dtype,
            seed=seed,
        )
        uni_embeds = resolve_uni_embedding(tile_id, feat_dir=feat_dir, null_uni=null_uni)
        row: dict[str, np.ndarray] = {}
        for tgt_label in labels:
            if src_label == tgt_label:
                ctrl = ctrl_full
            else:
                ctrl = build_relabeled_ctrl(
                    ctrl_full,
                    idx_source=resolved_channel_map[src_label],
                    idx_target=resolved_channel_map[tgt_label],
                )
            row[tgt_label] = generate_from_ctrl(
                ctrl,
                models=models,
                config=config,
                scheduler=scheduler,
                uni_embeds=uni_embeds,
                device=device,
                guidance_scale=guidance_scale,
                fixed_noise=fixed_noise,
                seed=seed,
            )
            print(f"    {src_label} -> {tgt_label}: done")
        results[src_label] = row
    return results


def run_exp2_cell_type_relabeling(
    tiles: dict[str, str],
    **kwargs: Any,
) -> dict[str, dict[str, np.ndarray]]:
    return _run_relabeling_experiment(tiles, _CELL_TYPE_CHANNELS, **kwargs)


def run_exp3_cell_state_relabeling(
    tiles: dict[str, str],
    **kwargs: Any,
) -> dict[str, dict[str, np.ndarray]]:
    return _run_relabeling_experiment(tiles, _CELL_STATE_CHANNELS, **kwargs)


def render_relabeling_figure(
    results: dict[str, dict[str, np.ndarray]],
    *,
    tiles: dict[str, str],
    thumb_specs: dict[str, tuple[str, tuple[int, int, int]]],
    exp_channels_dir: Path | None = None,
    baseline_group_thumbs: dict[str, np.ndarray] | None = None,
    input_thumbs: dict[str, dict[str, np.ndarray]] | None = None,
    labels: list[str] | None = None,
    source_labels: list[str] | None = None,
    target_labels: list[str] | None = None,
    exp_title: str,
    out_path: Path,
) -> None:
    """Render relabeling results as generated-H&E and diff chunks."""
    source_labels = list(source_labels) if source_labels is not None else list(labels) if labels is not None else _source_labels_from_results(results)
    target_labels = list(target_labels) if target_labels is not None else list(labels) if labels is not None else _target_labels_from_results(results)
    if not source_labels or not target_labels:
        raise ValueError("results must not be empty")

    hot_cmap = mcolors.LinearSegmentedColormap.from_list(
        "hot4",
        ["#000000", "#ff4400", "#ffff00", "#ffffff"],
    )
    baseline_color = "#9B59B6"
    n_source = len(source_labels)
    n_target = len(target_labels)
    n_rows = n_source * 2
    fig, axes = plt.subplots(n_rows, n_target, figsize=(3.1 * n_target, 2.55 * n_rows))
    if n_rows == 1 and n_target == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[None, :]
    elif n_target == 1:
        axes = axes[:, None]
    fig.suptitle(exp_title, fontsize=11)

    diff_maps: dict[tuple[str, str], np.ndarray] = {}
    diff_max = 0.0
    for src_label in source_labels:
        baseline = results[src_label][src_label].astype(np.float32)
        for tgt_label in target_labels:
            diff = np.abs(results[src_label][tgt_label].astype(np.float32) - baseline).mean(axis=2)
            if tgt_label == src_label:
                diff = np.zeros_like(diff)
            diff_maps[(src_label, tgt_label)] = diff
            diff_max = max(diff_max, float(diff.max()))
    diff_vmax = max(diff_max, 1.0)

    for i, src_label in enumerate(source_labels):
        row_he = i
        row_diff = i + n_source
        for j, tgt_label in enumerate(target_labels):
            ax_he = axes[row_he, j]
            img = results[src_label][tgt_label]
            ax_he.imshow(img)
            title = f"{src_label}->{tgt_label}"
            if tgt_label == src_label:
                title = f"all {src_label} (baseline)"
            elif src_label.startswith("non") and tgt_label.startswith("non"):
                title = f"all {tgt_label}"
            else:
                title = f"replace all {src_label} with {tgt_label}"
            ax_he.set_title(title, fontsize=8, pad=4)
            ax_he.axis("off")
            if tgt_label == src_label:
                ax_he.add_patch(
                    Rectangle(
                        (0.0, 0.0),
                        1.0,
                        1.0,
                        transform=ax_he.transAxes,
                        fill=False,
                        edgecolor=baseline_color,
                        linewidth=2.5,
                        linestyle="-",
                        zorder=10,
                        clip_on=False,
                    )
                )

            ax_diff = axes[row_diff, j]
            diff = diff_maps[(src_label, tgt_label)]
            ax_diff.imshow(diff, cmap=hot_cmap, vmin=0, vmax=diff_vmax)
            ax_diff.axis("off")

        axes[row_he, 0].set_ylabel(f"Generated H&E\nall {src_label}", fontsize=8, rotation=90, labelpad=18)
        axes[row_diff, 0].set_ylabel(f"Pixel diff\nvs all {src_label}", fontsize=8, rotation=90, labelpad=18)

    fig.subplots_adjust(left=0.11, right=0.99, top=0.95, bottom=0.04, wspace=0.05, hspace=0.16)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def build_baseline_group_thumbs(
    *,
    tiles: dict[str, str],
    exp_channels_dir: Path,
    thumb_specs: dict[str, tuple[str, tuple[int, int, int]]],
    resolution: int = 72,
) -> dict[str, np.ndarray]:
    return {
        label: _render_group_thumbnail(
            exp_channels_dir=exp_channels_dir,
            tile_id=tile_id,
            label=label,
            thumb_specs=thumb_specs,
            resolution=resolution,
        )
        for label, tile_id in tiles.items()
    }


def build_relabel_input_thumbs(
    *,
    tiles: dict[str, str],
    labels: list[str],
    exp_channels_dir: Path,
    thumb_specs: dict[str, tuple[str, tuple[int, int, int]]],
    resolution: int = 96,
) -> dict[str, dict[str, np.ndarray]]:
    out: dict[str, dict[str, np.ndarray]] = {}
    for src_label, tile_id in tiles.items():
        out[src_label] = {
            tgt_label: _render_relabel_input_thumbnail(
                exp_channels_dir=exp_channels_dir,
                tile_id=tile_id,
                source_label=src_label,
                target_label=tgt_label,
                thumb_specs=thumb_specs,
                resolution=resolution,
            )
            for tgt_label in labels
        }
    return out


def render_figures_from_cache(
    *,
    cache_dir: Path,
    out_dir: Path,
    experiments: list[str] | None = None,
) -> None:
    manifest = load_channel_sweep_manifest(cache_dir)
    selected = set(experiments or ["1", "2", "3"])
    exp_data = manifest.get("experiments", {})

    if "1" in selected and "exp1_microenv" in exp_data:
        for record in exp_data["exp1_microenv"].get("tiles", []):
            grid = load_exp1_microenv_cache(cache_dir, record)
            render_exp1_figure(
                grid,
                tile_id=str(record["tile_id"]),
                tile_class_label=str(record["tile_class_label"]),
                out_path=Path(out_dir) / "exp1_microenv" / f"{record['tile_class_label']}_{record['tile_id']}.png",
            )

    if "2" in selected and "exp2_cell_type_relabeling" in exp_data:
        loaded = load_relabeling_cache(cache_dir, exp_data["exp2_cell_type_relabeling"])
        render_relabeling_figure(
            loaded["results"],
            tiles=loaded["tiles"],
            thumb_specs=_CELL_TYPE_THUMB_SPECS,
            baseline_group_thumbs=loaded["baseline_group_thumbs"],
            input_thumbs=loaded.get("input_thumbs"),
            labels=loaded["labels"],
            source_labels=loaded.get("source_labels"),
            target_labels=loaded.get("target_labels"),
            exp_title="Exp 2: Cell type relabeling (given cell states + microenv)",
            out_path=Path(out_dir) / "exp2_cell_type_relabeling.png",
        )

    if "3" in selected and "exp3_cell_state_relabeling" in exp_data:
        loaded = load_relabeling_cache(cache_dir, exp_data["exp3_cell_state_relabeling"])
        render_relabeling_figure(
            loaded["results"],
            tiles=loaded["tiles"],
            thumb_specs=_CELL_STATE_THUMB_SPECS,
            baseline_group_thumbs=loaded["baseline_group_thumbs"],
            input_thumbs=loaded.get("input_thumbs"),
            labels=loaded["labels"],
            source_labels=loaded.get("source_labels"),
            target_labels=loaded.get("target_labels"),
            exp_title="Exp 3: Cell state relabeling (given cell types + microenv)",
            out_path=Path(out_dir) / "exp3_cell_state_relabeling.png",
        )

    rendered_known = {
        "exp1_microenv",
        "exp2_cell_type_relabeling",
        "exp3_cell_state_relabeling",
    }
    for exp_name, record in exp_data.items():
        if exp_name in rendered_known:
            continue
        entries = record.get("entries")
        if not isinstance(entries, list) or not entries:
            continue
        images: list[tuple[str, np.ndarray]] = []
        for entry in entries:
            rel = entry.get("image_path")
            if not rel:
                continue
            img_path = Path(cache_dir) / rel
            if not img_path.is_file():
                continue
            label = f"{entry.get('source_label', '')} -> {entry.get('target_label', '')}".strip()
            images.append((label, _load_rgb_png(img_path)))
        if not images:
            continue
        n = len(images)
        fig, axes = plt.subplots(1, n, figsize=(2.6 * n, 2.8))
        if n == 1:
            axes = np.array([axes])
        for ax, (label, image) in zip(axes, images, strict=True):
            ax.imshow(image)
            ax.set_title(label, fontsize=8)
            ax.axis("off")
        fig.suptitle(exp_name, fontsize=10)
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(out_dir) / f"{exp_name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def render_channel_sweep_figures(
    *,
    cache_dir: Path,
    out_dir: Path,
    output_dir: Path | None = None,
    cache_path: Path | None = None,
    cache_manifest: Path | None = None,
    experiments: list[str] | None = None,
) -> None:
    del cache_manifest
    render_figures_from_cache(
        cache_dir=Path(cache_path or cache_dir),
        out_dir=Path(output_dir or out_dir),
        experiments=experiments,
    )


def _pick_exp1_tiles(class_data: dict[str, Any]) -> dict[str, str]:
    reps = class_data.get("representatives", {})
    axis1_tiles: dict[str, str] = {}
    for axis1 in ("cancer", "immune", "healthy"):
        neutral_key = f"{axis1}+neutral"
        if neutral_key in reps:
            axis1_tiles[axis1] = reps[neutral_key]["tile_id"]
            continue
        for combo_key, rep in reps.items():
            if combo_key.startswith(f"{axis1}+"):
                axis1_tiles[axis1] = rep["tile_id"]
                break
    return axis1_tiles


def main() -> None:
    parser = argparse.ArgumentParser(description="Run channel sweep experiments (Exp1/Exp2/Exp3)")
    parser.add_argument("--class-json", required=True, help="tile_classes.json from classify_tiles.py")
    parser.add_argument("--data-root", required=True, help="Dataset root containing exp_channels/")
    parser.add_argument(
        "--checkpoint-dir",
        default=str(ROOT / "checkpoints/pixcell_controlnet_exp/checkpoints"),
        help="Checkpoint directory or parent containing controlnet_*.pth",
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs/config_controlnet_exp.py"),
        help="Config file used to load the checkpointed models",
    )
    parser.add_argument("--out", required=True, help="Output directory for figures")
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Directory to store/load cached generated images (default: <out>/cache)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--guidance-scale", type=float, default=2.5)
    parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=["1", "2", "3"],
        default=["1", "2", "3"],
        help="Subset of experiments to run",
    )
    parser.add_argument(
        "--null-uni",
        action="store_true",
        help="Use null UNI embeddings instead of paired features/<tile>_uni.npy",
    )
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="Run inference and write cache, but do not render figures.",
    )
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="Skip model loading and render figures from an existing cache.",
    )
    args = parser.parse_args()

    from tools.stage3.tile_pipeline import resolve_data_layout

    data_root = Path(args.data_root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir else out_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    if args.generate_only and args.render_only:
        raise SystemExit("--generate-only and --render-only are mutually exclusive")

    if args.render_only:
        print(f"Rendering figures from cache -> {cache_dir}")
        render_figures_from_cache(cache_dir=cache_dir, out_dir=out_dir, experiments=args.experiments)
        print(f"\nCached figure rendering complete. Outputs in {out_dir}")
        return

    exp_channels_dir, feat_dir, _ = resolve_data_layout(data_root)
    class_data = json.loads(Path(args.class_json).read_text(encoding="utf-8"))
    models, config, scheduler = load_sweep_models(
        args.config,
        checkpoint_dir=Path(args.checkpoint_dir),
        device=args.device,
        num_steps=args.num_steps,
    )

    shared_kwargs = dict(
        exp_channels_dir=exp_channels_dir,
        feat_dir=feat_dir,
        null_uni=args.null_uni,
        models=models,
        config=config,
        scheduler=scheduler,
        device=args.device,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
    )
    manifest: dict[str, Any] = {"version": CACHE_VERSION, "experiments": {}}

    if "1" in args.experiments:
        print("\n=== Experiment 1: Microenv 2D grid ===")
        tiles_records: list[dict[str, Any]] = []
        for label, tile_id in _pick_exp1_tiles(class_data).items():
            grid = run_exp1_microenv_grid(tile_id, **shared_kwargs)
            tiles_records.append(
                save_exp1_microenv_cache(
                    cache_dir=cache_dir,
                    tile_id=tile_id,
                    tile_class_label=label,
                    images_grid=grid,
                )
            )
        manifest["experiments"]["exp1_microenv"] = {"tiles": tiles_records}

    if "2" in args.experiments:
        print("\n=== Experiment 2: Cell type relabeling ===")
        exp2_tiles = {
            label: payload["tile_id"]
            for label, payload in class_data.get("exp2_tiles", {}).items()
            if isinstance(payload, dict) and "tile_id" in payload
        }
        if len(exp2_tiles) >= 2:
            results2 = run_exp2_cell_type_relabeling(exp2_tiles, **shared_kwargs)
            thumbs2 = build_baseline_group_thumbs(
                tiles=exp2_tiles,
                exp_channels_dir=exp_channels_dir,
                thumb_specs=_CELL_TYPE_THUMB_SPECS,
            )
            input_thumbs2 = build_relabel_input_thumbs(
                tiles=exp2_tiles,
                labels=_target_labels_from_results(results2),
                exp_channels_dir=exp_channels_dir,
                thumb_specs=_CELL_TYPE_THUMB_SPECS,
            )
            manifest["experiments"]["exp2_cell_type_relabeling"] = save_relabeling_cache(
                cache_dir=cache_dir,
                exp_name="exp2_cell_type_relabeling",
                results=results2,
                tiles=exp2_tiles,
                baseline_group_thumbs=thumbs2,
                input_thumbs=input_thumbs2,
            )
        else:
            print("  Skipping Exp 2: need at least 2 tiles in exp2_tiles")

    if "3" in args.experiments:
        print("\n=== Experiment 3: Cell state relabeling ===")
        exp3_tiles = {
            label: payload["tile_id"]
            for label, payload in class_data.get("exp3_tiles", {}).items()
            if isinstance(payload, dict) and "tile_id" in payload
        }
        if len(exp3_tiles) >= 2:
            results3 = run_exp3_cell_state_relabeling(exp3_tiles, **shared_kwargs)
            thumbs3 = build_baseline_group_thumbs(
                tiles=exp3_tiles,
                exp_channels_dir=exp_channels_dir,
                thumb_specs=_CELL_STATE_THUMB_SPECS,
            )
            input_thumbs3 = build_relabel_input_thumbs(
                tiles=exp3_tiles,
                labels=_target_labels_from_results(results3),
                exp_channels_dir=exp_channels_dir,
                thumb_specs=_CELL_STATE_THUMB_SPECS,
            )
            manifest["experiments"]["exp3_cell_state_relabeling"] = save_relabeling_cache(
                cache_dir=cache_dir,
                exp_name="exp3_cell_state_relabeling",
                results=results3,
                tiles=exp3_tiles,
                baseline_group_thumbs=thumbs3,
                input_thumbs=input_thumbs3,
            )
        else:
            print("  Skipping Exp 3: need at least 2 tiles in exp3_tiles")

    manifest_path = save_channel_sweep_manifest(cache_dir, manifest)
    print(f"\nSaved generation cache -> {manifest_path}")

    if not args.generate_only:
        render_figures_from_cache(cache_dir=cache_dir, out_dir=out_dir, experiments=args.experiments)

    print(f"\nAll requested experiments complete. Cache in {cache_dir} | figures in {out_dir}")


if __name__ == "__main__":
    main()
