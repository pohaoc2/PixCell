#!/usr/bin/env python3
"""Per-sub-channel leave-one-out runner for the a1_concat ControlNet.

Generates one LOO image per (tile_id, sub_channel) by zeroing that single raw
input channel while keeping every other channel (including `cell_masks`) on.
Diffs against the existing Fig 3 baseline at
`<baseline-root>/<tile_id>/all/generated_he.png` to compute per-pixel ΔE.

Output layout:
  <out-dir>/<tile_id>/<sub_channel>/generated_he.png
  <out-dir>/<tile_id>/subchannel_loo_diff_stats.json   (one merged JSON per tile)

JSON schema matches `leave_one_out_diff_stats.json` keys (mean_diff,
delta_e_mean, delta_e_p99, ssim_loss_mean, pct_pixels_above_10), keyed by
sub-channel name.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CONFIG_PATH = ROOT / "configs" / "config_controlnet_exp_a1_concat.py"
DEFAULT_CHECKPOINT_DIR = ROOT / "checkpoints" / "concat_95470_0" / "checkpoints" / "step_0002600"
DEFAULT_DATA_ROOT = ROOT / "data" / "orion-crc33"
DEFAULT_BASELINE_ROOT = ROOT / "inference_output" / "concat_ablation_1000" / "paired_ablation" / "ablation_results"
DEFAULT_TILE_LIST = ROOT / "inference_output" / "concat_ablation_1000" / "tile_lists" / "paired_1000_tile_ids.txt"
DEFAULT_OUT_DIR = ROOT / "inference_output" / "subchannel_loo_n300"

DEFAULT_DEVICE = "cuda"
DEFAULT_GUIDANCE_SCALE = 2.5
DEFAULT_NUM_STEPS = 20
DEFAULT_SEED = 42

# 9 sub-channels for LOO (cell_masks is always-on, not droppable).
ALL_SUB_CHANNELS: tuple[str, ...] = (
    "cell_type_healthy",
    "cell_type_cancer",
    "cell_type_immune",
    "cell_state_prolif",
    "cell_state_nonprolif",
    "cell_state_dead",
    "vasculature",
    "oxygen",
    "glucose",
)


def _read_tile_ids(path: Path, n: int | None) -> list[str]:
    tiles = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if n is not None:
        tiles = tiles[:n]
    return tiles


def _load_rgb_uint8(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def _save_rgb(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.clip(arr, 0, 255).astype(np.uint8) if arr.dtype != np.uint8 else arr
    Image.fromarray(arr).save(path)


def _compute_stats(img_baseline: np.ndarray, img_loo: np.ndarray) -> dict[str, float]:
    from tools.vis.leave_one_out_diff import delta_e_lab_map

    diff = np.abs(img_baseline.astype(np.float32) - img_loo.astype(np.float32)).mean(axis=2)
    delta_e = delta_e_lab_map(img_baseline, img_loo)
    return {
        "mean_diff": round(float(diff.mean()), 4),
        "max_diff": round(float(diff.max()), 4),
        "pct_pixels_above_10": round(float((diff > 10).mean() * 100.0), 2),
        "delta_e_mean": round(float(delta_e.mean()), 4),
        "delta_e_p99": round(float(np.percentile(delta_e, 99)), 4),
    }


def _active_groups_for_loo(all_sub_channels: Iterable[str], drop: str) -> set[str]:
    """Channel-name set fed to generate_from_ctrl active_groups (concat path).

    The raw_group_channels fallback in tile_pipeline expands unknown names to
    themselves, so passing individual channel names works directly. cell_masks
    is always-on at the pipeline level — not included here.
    """
    keep = set(all_sub_channels) - {drop}
    return keep


def _build_full_active_groups(all_sub_channels: Iterable[str]) -> set[str]:
    return set(all_sub_channels)


def _load_runtime(
    *,
    config_path: Path,
    checkpoint_dir: Path,
    data_root: Path,
    device: str,
    num_steps: int,
):
    from tools.stage3.channel_sweep import load_sweep_models
    from tools.stage3.tile_pipeline import resolve_data_layout

    models, config, scheduler = load_sweep_models(
        config_path,
        checkpoint_dir=checkpoint_dir,
        device=device,
        num_steps=num_steps,
    )
    exp_channels_dir, feat_dir, _ = resolve_data_layout(data_root)
    return models, config, scheduler, exp_channels_dir, feat_dir


def _render(
    *,
    ctrl_full: torch.Tensor,
    uni_embeds: torch.Tensor,
    fixed_noise: torch.Tensor,
    active_groups: set[str],
    models,
    config,
    scheduler,
    device: str,
    guidance_scale: float,
    seed: int,
) -> np.ndarray:
    from tools.stage3.tile_pipeline import generate_from_ctrl

    gen_np, _ = generate_from_ctrl(
        ctrl_full,
        models=models,
        config=config,
        scheduler=scheduler,
        uni_embeds=uni_embeds,
        device=device,
        guidance_scale=guidance_scale,
        seed=seed,
        fixed_noise=fixed_noise,
        active_groups=active_groups,
    )
    return gen_np


def run(
    *,
    config_path: Path,
    checkpoint_dir: Path,
    data_root: Path,
    baseline_root: Path,
    tile_list_path: Path,
    out_dir: Path,
    n_tiles: int,
    sub_channels: tuple[str, ...],
    device: str,
    guidance_scale: float,
    num_steps: int,
    seed: int,
    regenerate_baseline: bool,
) -> None:
    from tools.stage3.common import inference_dtype, resolve_uni_embedding
    from tools.stage3.tile_pipeline import _make_fixed_noise, load_exp_channels

    tile_ids = _read_tile_ids(tile_list_path, n_tiles)
    print(f"[setup] {len(tile_ids)} tiles, {len(sub_channels)} sub-channels => {len(tile_ids) * len(sub_channels)} inferences", flush=True)
    print(f"[setup] sub_channels: {sub_channels}", flush=True)

    models, config, scheduler, exp_channels_dir, feat_dir = _load_runtime(
        config_path=config_path,
        checkpoint_dir=checkpoint_dir,
        data_root=data_root,
        device=device,
        num_steps=num_steps,
    )
    print(f"[setup] models loaded", flush=True)

    active_channels = list(config.data.active_channels)
    full_active_groups = _build_full_active_groups(ALL_SUB_CHANNELS)
    dtype = inference_dtype(device)
    fixed_noise = _make_fixed_noise(
        config=config,
        scheduler=scheduler,
        device=device,
        dtype=dtype,
        seed=seed,
    )

    total = 0
    for tile_idx, tile_id in enumerate(tile_ids, start=1):
        tile_out = out_dir / tile_id
        tile_out.mkdir(parents=True, exist_ok=True)

        baseline_path = baseline_root / tile_id / "all" / "generated_he.png"
        if not baseline_path.is_file() and not regenerate_baseline:
            print(f"[{tile_idx}/{len(tile_ids)}] {tile_id} SKIP — missing baseline {baseline_path}", flush=True)
            continue

        ctrl_full = load_exp_channels(tile_id, active_channels, config.image_size, exp_channels_dir)
        uni_embeds = resolve_uni_embedding(tile_id, feat_dir=feat_dir, null_uni=False).to(device, dtype=dtype)

        if regenerate_baseline:
            baseline_local = tile_out / "all_baseline.png"
            if not baseline_local.is_file():
                gen_all = _render(
                    ctrl_full=ctrl_full,
                    uni_embeds=uni_embeds,
                    fixed_noise=fixed_noise,
                    active_groups=full_active_groups,
                    models=models,
                    config=config,
                    scheduler=scheduler,
                    device=device,
                    guidance_scale=guidance_scale,
                    seed=seed,
                )
                _save_rgb(baseline_local, gen_all)
            img_baseline = _load_rgb_uint8(baseline_local)
        else:
            img_baseline = _load_rgb_uint8(baseline_path)

        stats_path = tile_out / "subchannel_loo_diff_stats.json"
        if stats_path.is_file():
            try:
                existing_stats: dict[str, dict[str, float]] = json.loads(stats_path.read_text())
            except json.JSONDecodeError:
                existing_stats = {}
        else:
            existing_stats = {}

        for sub in sub_channels:
            sub_dir = tile_out / sub
            sub_png = sub_dir / "generated_he.png"
            if sub_png.is_file() and sub in existing_stats:
                continue
            active_groups = _active_groups_for_loo(ALL_SUB_CHANNELS, drop=sub)
            gen_loo = _render(
                ctrl_full=ctrl_full,
                uni_embeds=uni_embeds,
                fixed_noise=fixed_noise,
                active_groups=active_groups,
                models=models,
                config=config,
                scheduler=scheduler,
                device=device,
                guidance_scale=guidance_scale,
                seed=seed,
            )
            _save_rgb(sub_png, gen_loo)
            existing_stats[sub] = _compute_stats(img_baseline, gen_loo)
            stats_path.write_text(json.dumps(existing_stats, indent=2) + "\n", encoding="utf-8")
            total += 1

        print(f"[{tile_idx}/{len(tile_ids)}] {tile_id} done ({len(sub_channels)} sub-channels)", flush=True)

    print(f"[done] generated {total} LOO images", flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Per-sub-channel LOO runner (a1_concat)")
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE_ROOT)
    parser.add_argument("--tile-list", type=Path, default=DEFAULT_TILE_LIST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--n-tiles", type=int, default=300)
    parser.add_argument(
        "--sub-channels",
        nargs="+",
        default=list(ALL_SUB_CHANNELS),
        help="Sub-channels to drop one at a time (default: all 9).",
    )
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--guidance-scale", type=float, default=DEFAULT_GUIDANCE_SCALE)
    parser.add_argument("--num-steps", type=int, default=DEFAULT_NUM_STEPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--regenerate-baseline",
        action="store_true",
        help="Re-render the all-channels-on baseline locally instead of using --baseline-root.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    unknown = [s for s in args.sub_channels if s not in ALL_SUB_CHANNELS]
    if unknown:
        raise SystemExit(f"unknown sub-channels: {unknown}; allowed: {ALL_SUB_CHANNELS}")
    run(
        config_path=args.config_path,
        checkpoint_dir=args.checkpoint_dir,
        data_root=args.data_root,
        baseline_root=args.baseline_root,
        tile_list_path=args.tile_list,
        out_dir=args.out_dir,
        n_tiles=args.n_tiles,
        sub_channels=tuple(args.sub_channels),
        device=args.device,
        guidance_scale=args.guidance_scale,
        num_steps=args.num_steps,
        seed=args.seed,
        regenerate_baseline=args.regenerate_baseline,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
