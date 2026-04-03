#!/usr/bin/env python3
"""
Generate and repair Stage 3 ablation caches.

Outputs per-tile cache folders with:
  - singles/
  - pairs/
  - triples/
  - all/
  - manifest.json

Optionally caches UNI features for every manifest image under:
  - features/<section>/<stem>_uni.npy
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from diffusers import DDPMScheduler
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent.parent


def _is_cuda_device(device: str) -> bool:
    return str(device).lower().startswith("cuda")


def _resolve_uni_embedding(
    tile_id: str,
    *,
    feat_dir: Path,
    null_uni: bool,
    uni_npy: Path | None,
) -> torch.Tensor:
    from train_scripts.inference_controlnet import null_uni_embed

    feat_path = Path(uni_npy) if uni_npy is not None else feat_dir / f"{tile_id}_uni.npy"
    if null_uni or not feat_path.exists():
        uni_embeds = null_uni_embed(device="cpu", dtype=torch.float32)
        if not null_uni:
            print(f"Warning: missing {feat_path}, using null UNI")
        return uni_embeds
    return torch.from_numpy(np.load(feat_path)).view(1, 1, 1, 1536)


def _to_uint8_rgb(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    clipped = np.clip(image, 0.0, 1.0)
    return (clipped * 255).astype(np.uint8)


def _subset_size(section: dict) -> int:
    try:
        return int(section.get("subset_size", 0))
    except (TypeError, ValueError):
        return 0


def _resolve_all_image_from_manifest(
    cache_dir: Path,
    manifest: dict,
    *,
    n_groups: int,
) -> Path | None:
    for section in manifest.get("sections", []):
        if _subset_size(section) != n_groups:
            continue
        entries = section.get("entries") or []
        if not entries:
            continue
        rel = Path(entries[0].get("image_path", ""))
        if rel and (cache_dir / rel).is_file():
            return cache_dir / rel

    canonical = cache_dir / "all" / "generated_he.png"
    if canonical.is_file():
        return canonical

    all_dir = cache_dir / "all"
    if all_dir.is_dir():
        pngs = sorted(all_dir.glob("*.png"))
        if len(pngs) == 1:
            return pngs[0]
    return None


def _upsert_all_section(
    cache_dir: Path,
    *,
    tile_id: str,
    group_names: tuple[str, ...],
    all_image: np.ndarray,
) -> Path:
    from tools.stage3.ablation import build_subset_conditions

    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not group_names:
        group_names = tuple(manifest.get("group_names") or ())
    if not group_names:
        raise ValueError(f"manifest has empty group_names: {manifest_path}")

    all_dir = cache_dir / "all"
    all_dir.mkdir(parents=True, exist_ok=True)
    canonical_rel = Path("all") / "generated_he.png"
    Image.fromarray(_to_uint8_rgb(all_image)).save(cache_dir / canonical_rel)

    cond = build_subset_conditions(group_names, subset_size=len(group_names))[0]
    all_section = {
        "title": f"{len(group_names)} active groups",
        "subset_size": len(group_names),
        "entries": [
            {
                "active_groups": list(group_names),
                "condition_label": cond.label,
                "image_label": cond.label,
                "image_path": canonical_rel.as_posix(),
            }
        ],
    }

    sections = [
        sec for sec in manifest.get("sections", [])
        if _subset_size(sec) != len(group_names)
    ]
    sections.append(all_section)
    sections.sort(key=_subset_size)

    manifest["tile_id"] = tile_id
    manifest["group_names"] = list(group_names)
    manifest["sections"] = sections
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def _cache_features_for_cache_dir(
    cache_dir: Path,
    *,
    uni_model: Path,
    device: str,
    force: bool,
) -> int:
    from tools.stage3.ablation_vis_utils import cache_manifest_uni_features

    try:
        return cache_manifest_uni_features(
            cache_dir,
            uni_model=uni_model,
            device=device,
            force=force,
        )
    except Exception as exc:
        if not _is_cuda_device(device):
            raise
        print(f"Note: feature caching on cuda failed ({exc}); retrying on cpu")
        return cache_manifest_uni_features(
            cache_dir,
            uni_model=uni_model,
            device="cpu",
            force=force,
        )


def _generate_all_groups_image(
    tile_id: str,
    *,
    models: dict,
    config,
    scheduler,
    exp_channels_dir: Path,
    uni_embeds: torch.Tensor,
    device: str,
    guidance_scale: float,
    seed: int,
    subset_size: int,
) -> np.ndarray:
    from tools.stage3.tile_pipeline import generate_group_combination_ablation_images

    images = generate_group_combination_ablation_images(
        tile_id=tile_id,
        models=models,
        config=config,
        scheduler=scheduler,
        uni_embeds=uni_embeds,
        device=device,
        exp_channels_dir=exp_channels_dir,
        guidance_scale=guidance_scale,
        seed=seed,
        subset_size=subset_size,
    )
    if not images:
        raise RuntimeError(f"[{tile_id}] no generated image for subset_size={subset_size}")
    return images[0][1]


def generate_subset_cache_for_tile(
    tile_id: str,
    *,
    cache_dir: Path,
    models: dict,
    config,
    scheduler,
    exp_channels_dir: Path,
    feat_dir: Path,
    device: str,
    guidance_scale: float,
    seed: int,
    null_uni: bool,
    uni_npy: Path | None,
) -> Path:
    from tools.stage3.ablation import (
        build_subset_ablation_sections,
        group_names_from_channel_groups,
    )
    from tools.stage3.ablation_cache import save_subset_condition_cache
    from tools.stage3.tile_pipeline import generate_group_combination_ablation_images, load_exp_channels

    uni_embeds = _resolve_uni_embedding(
        tile_id,
        feat_dir=feat_dir,
        null_uni=null_uni,
        uni_npy=uni_npy,
    )
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{tile_id}] Generating single-group cache images...")
    single_group_imgs = generate_group_combination_ablation_images(
        tile_id=tile_id,
        models=models,
        config=config,
        scheduler=scheduler,
        uni_embeds=uni_embeds,
        device=device,
        exp_channels_dir=exp_channels_dir,
        guidance_scale=guidance_scale,
        seed=seed,
        subset_size=1,
    )

    print(f"[{tile_id}] Generating pair-group cache images...")
    pair_group_imgs = generate_group_combination_ablation_images(
        tile_id=tile_id,
        models=models,
        config=config,
        scheduler=scheduler,
        uni_embeds=uni_embeds,
        device=device,
        exp_channels_dir=exp_channels_dir,
        guidance_scale=guidance_scale,
        seed=seed,
        subset_size=2,
    )

    print(f"[{tile_id}] Generating triple-group cache images...")
    triple_group_imgs = generate_group_combination_ablation_images(
        tile_id=tile_id,
        models=models,
        config=config,
        scheduler=scheduler,
        uni_embeds=uni_embeds,
        device=device,
        exp_channels_dir=exp_channels_dir,
        guidance_scale=guidance_scale,
        seed=seed,
        subset_size=3,
    )

    group_names = group_names_from_channel_groups(config.channel_groups)
    if len(group_names) >= 4:
        print(f"[{tile_id}] Generating all-groups cache image...")
        all_group_imgs = generate_group_combination_ablation_images(
            tile_id=tile_id,
            models=models,
            config=config,
            scheduler=scheduler,
            uni_embeds=uni_embeds,
            device=device,
            exp_channels_dir=exp_channels_dir,
            guidance_scale=guidance_scale,
            seed=seed,
            subset_size=len(group_names),
        )
        subset_sections = build_subset_ablation_sections(
            group_names,
            single_images=single_group_imgs,
            pair_images=pair_group_imgs,
            triple_images=triple_group_imgs,
            all_four_images=all_group_imgs,
        )
    else:
        subset_sections = build_subset_ablation_sections(
            group_names,
            single_images=single_group_imgs,
            pair_images=pair_group_imgs,
            triple_images=triple_group_imgs,
        )

    ctrl_full = load_exp_channels(
        tile_id,
        config.data.active_channels,
        config.image_size,
        exp_channels_dir,
    )
    cell_mask = None
    if "cell_masks" in config.data.active_channels:
        cell_mask = ctrl_full[config.data.active_channels.index("cell_masks")].numpy()

    manifest_path = save_subset_condition_cache(
        cache_dir,
        tile_id=tile_id,
        group_names=group_names,
        sections=subset_sections,
        cell_mask=cell_mask,
    )
    print(f"[{tile_id}] Saved subset ablation cache -> {manifest_path}")
    return manifest_path


def backfill_all_for_cache_dir(
    cache_dir: Path,
    *,
    models: dict,
    config,
    scheduler,
    exp_channels_dir: Path,
    feat_dir: Path,
    device: str,
    guidance_scale: float,
    seed: int,
    null_uni: bool,
) -> tuple[str, bool]:
    from tools.stage3.ablation import group_names_from_channel_groups

    cache_dir = Path(cache_dir)
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    tile_id = str(manifest.get("tile_id") or cache_dir.name)
    group_names = tuple(manifest.get("group_names") or group_names_from_channel_groups(config.channel_groups))
    if not group_names:
        raise ValueError(f"[{tile_id}] manifest has no group_names: {manifest_path}")

    n_groups = len(group_names)
    existing_all = _resolve_all_image_from_manifest(cache_dir, manifest, n_groups=n_groups)
    has_all_section = any(
        _subset_size(sec) == n_groups
        for sec in manifest.get("sections", [])
    )
    canonical_all = cache_dir / "all" / "generated_he.png"

    if existing_all is not None:
        need_upsert = (
            not has_all_section
            or not canonical_all.is_file()
            or existing_all.resolve() != canonical_all.resolve()
        )
        if not need_upsert:
            print(f"[{tile_id}] all/ already present, skipping generation")
            return tile_id, False
        all_img = np.asarray(Image.open(existing_all).convert("RGB"))
        _upsert_all_section(
            cache_dir,
            tile_id=tile_id,
            group_names=group_names,
            all_image=all_img,
        )
        print(f"[{tile_id}] Canonicalized all/ and manifest")
        return tile_id, True

    uni_embeds = _resolve_uni_embedding(
        tile_id,
        feat_dir=feat_dir,
        null_uni=null_uni,
        uni_npy=None,
    )
    print(f"[{tile_id}] Generating missing all/ cache image...")
    all_img = _generate_all_groups_image(
        tile_id,
        models=models,
        config=config,
        scheduler=scheduler,
        exp_channels_dir=exp_channels_dir,
        uni_embeds=uni_embeds,
        device=device,
        guidance_scale=guidance_scale,
        seed=seed,
        subset_size=n_groups,
    )
    _upsert_all_section(
        cache_dir,
        tile_id=tile_id,
        group_names=group_names,
        all_image=all_img,
    )
    print(f"[{tile_id}] Backfilled all/ and updated manifest")
    return tile_id, True


def _make_scheduler(*, num_steps: int, device: str) -> DDPMScheduler:
    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
        prediction_type="epsilon",
        clip_sample=False,
    )
    scheduler.set_timesteps(num_steps, device=device)
    return scheduler


def _load_runtime(
    *,
    config_path: str,
    checkpoint_dir: str | None,
    data_root: str,
    device: str,
    num_steps: int,
) -> dict:
    from diffusion.utils.misc import read_config
    from tools.stage3.tile_pipeline import (
        find_latest_checkpoint_dir,
        load_all_models,
        resolve_data_layout,
    )

    data_root_path = Path(data_root)
    exp_channels_dir, feat_dir, _ = resolve_data_layout(data_root_path)

    ckpt_parent = (
        Path(checkpoint_dir)
        if checkpoint_dir
        else ROOT / "checkpoints/pixcell_controlnet_exp/checkpoints"
    )
    ckpt_dir = find_latest_checkpoint_dir(ckpt_parent)

    config = read_config(config_path)
    config._filename = config_path
    models = load_all_models(config, config_path, ckpt_dir, device)
    scheduler = _make_scheduler(num_steps=num_steps, device=device)

    return {
        "config": config,
        "models": models,
        "scheduler": scheduler,
        "exp_channels_dir": exp_channels_dir,
        "feat_dir": feat_dir,
        "ckpt_dir": ckpt_dir,
    }


def _build_worker_common_args(args: argparse.Namespace, *, data_root: Path) -> dict:
    return {
        "config_path": args.config,
        "checkpoint_dir": args.checkpoint_dir,
        "data_root": str(data_root),
        "device": args.device,
        "guidance_scale": args.guidance_scale,
        "num_steps": args.num_steps,
        "seed": args.seed,
        "null_uni": args.null_uni,
        "cache_uni_features": args.cache_uni_features,
        "force_uni_features": args.force_uni_features,
        "uni_model": args.uni_model,
        "feature_device": args.feature_device or args.device,
    }


def _generate_tile_job(job: dict) -> tuple[str, int]:
    os.chdir(ROOT)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    runtime = _load_runtime(
        config_path=job["config_path"],
        checkpoint_dir=job["checkpoint_dir"],
        data_root=job["data_root"],
        device=job["device"],
        num_steps=job["num_steps"],
    )
    print(f"Using checkpoint dir: {runtime['ckpt_dir']}")

    tile_id = str(job["tile_id"])
    cache_dir = Path(job["cache_dir"])
    generate_subset_cache_for_tile(
        tile_id,
        cache_dir=cache_dir,
        models=runtime["models"],
        config=runtime["config"],
        scheduler=runtime["scheduler"],
        exp_channels_dir=runtime["exp_channels_dir"],
        feat_dir=runtime["feat_dir"],
        device=job["device"],
        guidance_scale=job["guidance_scale"],
        seed=job["seed"],
        null_uni=job["null_uni"],
        uni_npy=None,
    )

    written = 0
    if job["cache_uni_features"]:
        written = _cache_features_for_cache_dir(
            cache_dir,
            uni_model=Path(job["uni_model"]),
            device=job["feature_device"],
            force=job["force_uni_features"],
        )
        print(f"[{tile_id}] Cached {written} UNI feature files")
    return tile_id, written


def _backfill_cache_job(job: dict) -> tuple[str, bool, int]:
    os.chdir(ROOT)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    runtime = _load_runtime(
        config_path=job["config_path"],
        checkpoint_dir=job["checkpoint_dir"],
        data_root=job["data_root"],
        device=job["device"],
        num_steps=job["num_steps"],
    )
    print(f"Using checkpoint dir: {runtime['ckpt_dir']}")

    cache_dir = Path(job["cache_dir"])
    tile_id, changed = backfill_all_for_cache_dir(
        cache_dir,
        models=runtime["models"],
        config=runtime["config"],
        scheduler=runtime["scheduler"],
        exp_channels_dir=runtime["exp_channels_dir"],
        feat_dir=runtime["feat_dir"],
        device=job["device"],
        guidance_scale=job["guidance_scale"],
        seed=job["seed"],
        null_uni=job["null_uni"],
    )

    written = 0
    if job["cache_uni_features"]:
        written = _cache_features_for_cache_dir(
            cache_dir,
            uni_model=Path(job["uni_model"]),
            device=job["feature_device"],
            force=job["force_uni_features"],
        )
        print(f"[{tile_id}] Cached {written} UNI feature files")
    return tile_id, changed, written


def _run_parallel_jobs(
    *,
    jobs: list[dict],
    worker_fn,
    requested_jobs: int,
    label: str,
) -> list:
    worker_count = min(max(1, int(requested_jobs)), len(jobs), os.cpu_count() or max(1, int(requested_jobs)))
    print(f"Running {len(jobs)} {label} job(s) with {worker_count} worker process(es)")

    results: list = []
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(worker_fn, job) for job in jobs]
        total = len(futures)
        completed = 0
        _print_progress(0, total, prefix=label.title())
        for future in as_completed(futures):
            results.append(future.result())
            completed += 1
            _print_progress(completed, total, prefix=label.title())
    return results


def _print_progress(completed: int, total: int, *, prefix: str) -> None:
    """Write a simple in-place progress bar to stderr."""
    total = max(1, total)
    width = 28
    filled = int(width * completed / total)
    bar = "#" * filled + "-" * (width - filled)
    msg = f"\r{prefix} [{bar}] {completed}/{total}"
    if completed >= total:
        msg += "\n"
    print(msg, end="", file=sys.stderr, flush=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Stage 3 subset caches (tile-id / n-tiles) or backfill existing caches "
            "with missing all/ and optional UNI feature files."
        ),
    )
    parser.add_argument("--config", type=str, default=str(ROOT / "configs/config_controlnet_exp.py"))
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help=(
            "Folder with controlnet_*.pth and tme_module.pth. "
            "Default: latest under checkpoints/pixcell_controlnet_exp/checkpoints"
        ),
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=str(ROOT / "inference_data/sample"),
        help="Dataset root: flat channel folders or ORION tree with exp_channels/, features/, he/",
    )
    parser.add_argument(
        "--exp-root",
        type=str,
        default=None,
        help="Deprecated alias for --data-root (overrides --data-root if set)",
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--tile-id", type=str, default=None, help="One tile ID from exp_channels")
    mode.add_argument(
        "--n-tiles",
        "--n-tile",
        type=int,
        default=None,
        dest="n_tiles",
        metavar="N",
        help="Randomly sample N tiles from --data-root and write each under --cache-dir/{tile_id}",
    )
    mode.add_argument(
        "--existing-cache-parent",
        type=str,
        default=None,
        help=(
            "Parent directory with existing per-tile caches (each subdir has manifest.json). "
            "Backfills missing all/ and can cache UNI features."
        ),
    )

    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help=(
            "With --tile-id: output dir for that tile (default: inference_output/cache/{tile_id}). "
            "With --n-tiles: parent directory (default: inference_output/cache)."
        ),
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--guidance-scale", type=float, default=2.5)
    parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42, help="Diffusion / generation seed (per tile)")
    parser.add_argument(
        "--tile-sample-seed",
        type=int,
        default=42,
        help="RNG seed when choosing tiles with --n-tiles (default: 42)",
    )
    parser.add_argument(
        "--null-uni",
        action="store_true",
        help="Use null UNI embedding instead of paired features/{tile}_uni.npy",
    )
    parser.add_argument(
        "--uni-npy",
        type=str,
        default=None,
        help="Explicit path to UNI embedding .npy (single-tile mode only; overrides {feat_dir}/{tile_id}_uni.npy)",
    )

    parser.add_argument(
        "--cache-uni-features",
        action="store_true",
        help="Cache UNI features under features/<section>/<stem>_uni.npy for each manifest image.",
    )
    parser.add_argument(
        "--force-uni-features",
        action="store_true",
        help="Recompute UNI feature files even when they already exist.",
    )
    parser.add_argument(
        "--uni-model",
        type=str,
        default=str(ROOT / "pretrained_models/uni-2h"),
        help="UNI-2h model directory used for feature extraction.",
    )
    parser.add_argument(
        "--feature-device",
        type=str,
        default=None,
        help="Device for UNI feature extraction (default: same as --device).",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help=(
            "Worker processes for per-tile work in --n-tiles or --existing-cache-parent mode "
            "(default: 1). Each worker loads its own models."
        ),
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.jobs < 1:
        parser.error("--jobs must be >= 1")
    if args.n_tiles is not None and args.uni_npy is not None:
        parser.error("--uni-npy is only supported with --tile-id (single tile)")
    if args.existing_cache_parent is not None and args.uni_npy is not None:
        parser.error("--uni-npy is only supported with --tile-id (single tile)")

    os.chdir(ROOT)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from tools.stage3.ablation_cache import list_cached_tile_ids
    from tools.stage3.tile_pipeline import (
        list_tile_ids_from_exp_channels,
        resolve_data_layout,
    )

    data_root = Path(args.exp_root if args.exp_root is not None else args.data_root)
    exp_channels_dir, feat_dir, _ = resolve_data_layout(data_root)
    cache_parent_default = ROOT / "inference_output" / "cache"
    device = args.device
    feature_device = args.feature_device or device
    uni_model = Path(args.uni_model)
    worker_common = _build_worker_common_args(args, data_root=data_root)

    if args.tile_id is not None:
        runtime = _load_runtime(
            config_path=args.config,
            checkpoint_dir=args.checkpoint_dir,
            data_root=str(data_root),
            device=device,
            num_steps=args.num_steps,
        )
        print(f"Using checkpoint dir: {runtime['ckpt_dir']}")
        cache_dir = (
            Path(args.cache_dir)
            if args.cache_dir is not None
            else cache_parent_default / args.tile_id
        )
        uni_override = Path(args.uni_npy) if args.uni_npy else None
        generate_subset_cache_for_tile(
            args.tile_id,
            cache_dir=cache_dir,
            models=runtime["models"],
            config=runtime["config"],
            scheduler=runtime["scheduler"],
            exp_channels_dir=runtime["exp_channels_dir"],
            feat_dir=runtime["feat_dir"],
            device=device,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
            null_uni=args.null_uni,
            uni_npy=uni_override,
        )
        if args.cache_uni_features:
            written = _cache_features_for_cache_dir(
                cache_dir,
                uni_model=uni_model,
                device=feature_device,
                force=args.force_uni_features,
            )
            print(f"[{args.tile_id}] Cached {written} UNI feature files")
        return

    if args.n_tiles is not None:
        if args.n_tiles < 1:
            parser.error("--n-tiles must be >= 1")

        all_ids = list_tile_ids_from_exp_channels(exp_channels_dir)
        if len(all_ids) < args.n_tiles:
            parser.error(
                f"need at least {args.n_tiles} tiles under {exp_channels_dir}, found {len(all_ids)}"
            )

        random.seed(args.tile_sample_seed)
        selected = random.sample(all_ids, args.n_tiles)
        cache_parent = Path(args.cache_dir) if args.cache_dir is not None else cache_parent_default
        print(
            f"Sampled {args.n_tiles} tiles (tile_sample_seed={args.tile_sample_seed}): {selected}"
        )

        if args.jobs > 1 and _is_cuda_device(device):
            print(
                "Note: parallel workers on CUDA each load a full checkpoint and may contend for GPU "
                "memory; reduce --jobs or use --device cpu if needed."
            )
        if args.jobs > 1:
            jobs = [
                {
                    **worker_common,
                    "tile_id": tile_id,
                    "cache_dir": str(cache_parent / tile_id),
                }
                for tile_id in selected
            ]
            total_features = sum(
                written for _, written in _run_parallel_jobs(
                    jobs=jobs,
                    worker_fn=_generate_tile_job,
                    requested_jobs=args.jobs,
                    label="generation",
                )
            )
            if args.cache_uni_features:
                print(f"Done. Generated {len(selected)} tile caches; new UNI features: {total_features}")
            else:
                print(f"Done. Generated {len(selected)} tile caches")
            return

        runtime = _load_runtime(
            config_path=args.config,
            checkpoint_dir=args.checkpoint_dir,
            data_root=str(data_root),
            device=device,
            num_steps=args.num_steps,
        )
        print(f"Using checkpoint dir: {runtime['ckpt_dir']}")
        _print_progress(0, len(selected), prefix="Generation")
        for idx, tile_id in enumerate(selected, start=1):
            cache_dir = cache_parent / tile_id
            generate_subset_cache_for_tile(
                tile_id,
                cache_dir=cache_dir,
                models=runtime["models"],
                config=runtime["config"],
                scheduler=runtime["scheduler"],
                exp_channels_dir=runtime["exp_channels_dir"],
                feat_dir=runtime["feat_dir"],
                device=device,
                guidance_scale=args.guidance_scale,
                seed=args.seed,
                null_uni=args.null_uni,
                uni_npy=None,
            )
            if args.cache_uni_features:
                written = _cache_features_for_cache_dir(
                    cache_dir,
                    uni_model=uni_model,
                    device=feature_device,
                    force=args.force_uni_features,
                )
                print(f"[{tile_id}] Cached {written} UNI feature files")
            _print_progress(idx, len(selected), prefix="Generation")
        return

    cache_parent = Path(args.existing_cache_parent).resolve()
    try:
        cached_ids = list_cached_tile_ids(cache_parent)
    except FileNotFoundError as exc:
        parser.error(str(exc))
    if not cached_ids:
        parser.error(
            f"no per-tile caches under {cache_parent} "
            "(expected subdirs like <tile_id>/manifest.json)"
        )

    backfilled = 0
    total_features = 0
    print(f"Found {len(cached_ids)} existing cache dirs under {cache_parent}")

    if args.jobs > 1 and _is_cuda_device(device):
        print(
            "Note: parallel workers on CUDA each load a full checkpoint and may contend for GPU "
            "memory; reduce --jobs or use --device cpu if needed."
        )
    if args.jobs > 1:
        jobs = [
            {
                **worker_common,
                "cache_dir": str(cache_parent / tile_name),
            }
            for tile_name in cached_ids
        ]
        for _, changed, written in _run_parallel_jobs(
            jobs=jobs,
            worker_fn=_backfill_cache_job,
            requested_jobs=args.jobs,
            label="backfill",
        ):
            if changed:
                backfilled += 1
            total_features += written
        print(
            f"Done. Processed {len(cached_ids)} caches, backfilled all/ for {backfilled}, "
            f"new UNI features: {total_features}"
        )
        return

    runtime = _load_runtime(
        config_path=args.config,
        checkpoint_dir=args.checkpoint_dir,
        data_root=str(data_root),
        device=device,
        num_steps=args.num_steps,
    )
    print(f"Using checkpoint dir: {runtime['ckpt_dir']}")
    _print_progress(0, len(cached_ids), prefix="Backfill")
    for idx, tile_name in enumerate(cached_ids, start=1):
        tile_cache_dir = cache_parent / tile_name
        tile_id, changed = backfill_all_for_cache_dir(
            tile_cache_dir,
            models=runtime["models"],
            config=runtime["config"],
            scheduler=runtime["scheduler"],
            exp_channels_dir=runtime["exp_channels_dir"],
            feat_dir=runtime["feat_dir"],
            device=device,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
            null_uni=args.null_uni,
        )
        if changed:
            backfilled += 1

        if args.cache_uni_features:
            written = _cache_features_for_cache_dir(
                tile_cache_dir,
                uni_model=uni_model,
                device=feature_device,
                force=args.force_uni_features,
            )
            total_features += written
            print(f"[{tile_id}] Cached {written} UNI feature files")
        _print_progress(idx, len(cached_ids), prefix="Backfill")

    print(
        f"Done. Processed {len(cached_ids)} caches, backfilled all/ for {backfilled}, "
        f"new UNI features: {total_features}"
    )


if __name__ == "__main__":
    main()
