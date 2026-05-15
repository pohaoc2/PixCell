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

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.channel_group_utils import channel_index_map
from tools.stage3.channel_sweep_cache import (
    CACHE_VERSION,
    load_channel_sweep_cache,
    load_channel_sweep_manifest,
    load_exp1_microenv_cache,
    load_relabeling_cache,
    save_channel_sweep_cache,
    save_channel_sweep_manifest,
    save_exp1_microenv_cache,
    save_relabeling_cache,
    source_labels_from_results as _source_labels_from_results,
    target_labels_from_results as _target_labels_from_results,
)
from tools.stage3.channel_sweep_figures import (
    CELL_STATE_THUMB_SPECS as _CELL_STATE_THUMB_SPECS,
    CELL_TYPE_THUMB_SPECS as _CELL_TYPE_THUMB_SPECS,
    SWEEP_SCALES,
    build_baseline_group_thumbs,
    build_relabel_input_thumbs,
    render_channel_sweep_figures,
    render_exp1_figure,
    render_figures_from_cache,
    render_relabeling_figure,
)
from tools.stage3.common import (
    inference_dtype,
    load_json,
    make_inference_scheduler,
    resolve_uni_embedding,
)
from tools.stage3.style_mapping import load_style_mapping, resolve_style_tile_id

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


def _get_dtype(device: str) -> torch.dtype:
    return inference_dtype(device)


def _make_scheduler(*, num_steps: int, device: str):
    return make_inference_scheduler(num_steps=num_steps, device=device)


def _resolve_checkpoint_dir(checkpoint_dir: Path) -> Path:
    from tools.stage3.tile_pipeline import find_latest_checkpoint_dir

    if any(checkpoint_dir.glob("controlnet_*.pth")):
        return checkpoint_dir
    return find_latest_checkpoint_dir(checkpoint_dir)


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
    style_mapping: dict[str, str] | None,
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
    mapped_tile_id = resolve_style_tile_id(tile_id, style_mapping=style_mapping)
    uni_embeds = resolve_uni_embedding(
        tile_id,
        feat_dir=feat_dir,
        null_uni=null_uni,
        uni_npy=feat_dir / f"{mapped_tile_id}_uni.npy" if style_mapping else None,
    )

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


def _run_relabeling_experiment(
    tiles: dict[str, str],
    channel_map: dict[str, int],
    *,
    exp_channels_dir: Path,
    feat_dir: Path,
    style_mapping: dict[str, str] | None,
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
        mapped_tile_id = resolve_style_tile_id(tile_id, style_mapping=style_mapping)
        uni_embeds = resolve_uni_embedding(
            tile_id,
            feat_dir=feat_dir,
            null_uni=null_uni,
            uni_npy=feat_dir / f"{mapped_tile_id}_uni.npy" if style_mapping else None,
        )
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
        default=str(ROOT / "checkpoints/concat_95470_0/checkpoints/step_0002600"),
        help="Checkpoint directory or parent containing controlnet_*.pth",
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs/config_controlnet_exp_a1_concat.py"),
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
        "--style-mapping-json",
        default=None,
        help="Optional layout->style mapping JSON for unpaired UNI feature lookup.",
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
    style_mapping = load_style_mapping(args.style_mapping_json)
    class_data = load_json(args.class_json)
    models, config, scheduler = load_sweep_models(
        args.config,
        checkpoint_dir=Path(args.checkpoint_dir),
        device=args.device,
        num_steps=args.num_steps,
    )

    shared_kwargs = dict(
        exp_channels_dir=exp_channels_dir,
        feat_dir=feat_dir,
        style_mapping=style_mapping,
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
