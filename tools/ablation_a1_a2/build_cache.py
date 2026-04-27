"""Build or incrementally update inference_output/si_a1_a2/cache.json.

Examples
--------
Update training curves only; no weights, GPU, or model imports needed:
    python tools/ablation_a1_a2/build_cache.py --update-curves --cache-dir inference_output/si_a1_a2

Merge pre-computed metrics:
    python tools/ablation_a1_a2/build_cache.py --merge-metrics-file metrics.json --cache-dir inference_output/si_a1_a2

Full cache build with qualitative tiles; needs model weights and usually CUDA:
    python tools/ablation_a1_a2/build_cache.py --cache-dir inference_output/si_a1_a2 \
        --tile-ids-file tools/ablation_a1_a2/qual_tile_ids.txt --device cuda
"""
from __future__ import annotations

import argparse
import gc
import json
import numpy as np
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.ablation_a1_a2.cache_io import load_cache, merge_curves, merge_metrics, merge_params, save_cache
from tools.ablation_a1_a2.log_utils import extract_all_curves


INFERENCE_VARIANTS: dict[str, dict] = {
    "production": {
        "config_path": "configs/config_controlnet_exp.py",
        "ckpt_dir": "checkpoints/pixcell_controlnet_exp/npy_inputs",
        "variant_type": "standard",
    },
    "a1_concat": {
        "config_path": "configs/config_controlnet_exp_a1_concat.py",
        "ckpt_dir": "checkpoints/a1_concat/full_seed_42/checkpoint/step_0002600",
        "variant_type": "standard",
    },
    "a1_per_channel": {
        "config_path": "configs/config_controlnet_exp_a1_per_channel.py",
        "ckpt_dir": "checkpoints/a1_per_channel/full_seed_42/checkpoint/step_0002600",
        "variant_type": "standard",
    },
    "a2_bypass": {
        "config_path": "configs/config_controlnet_exp_a2_bypass.py",
        "ckpt_dir": "checkpoints/a2_a3/a2_bypass/full_seed_42/checkpoint/step_0002600",
        "variant_type": "bypass",
    },
    "a2_bypass_full_tme": {
        "config_path": "configs/config_controlnet_exp_a2_bypass.py",
        "ckpt_dir": "checkpoints/a2_a3/a2_bypass/full_seed_42/checkpoint/step_0002600",
        "variant_type": "standard",
    },
    "a2_off_shelf": {
        "config_path": "configs/config_controlnet_exp.py",
        "controlnet_path": "pretrained_models/pixcell-256-controlnet/controlnet/diffusion_pytorch_model.safetensors",
        "base_model_path": "pretrained_models/pixcell-256/transformer",
        "vae_path": "pretrained_models/sd-3.5-vae/vae",
        "variant_type": "off_shelf",
    },
}


def _run_update_curves(cache_dir: Path) -> None:
    cache_path = cache_dir / "cache.json"
    cache = load_cache(cache_path)
    merge_curves(cache, extract_all_curves())
    save_cache(cache, cache_path)
    total_runs = sum(len(runs) for runs in cache["training_curves"].values())
    print(f"Curves updated: {len(cache['training_curves'])} variants, {total_runs} runs -> {cache_path}")


def _run_merge_metrics(cache_dir: Path, metrics_file: Path) -> None:
    payload = json.loads(metrics_file.read_text(encoding="utf-8"))
    cache_path = cache_dir / "cache.json"
    cache = load_cache(cache_path)

    if isinstance(payload, dict) and isinstance(payload.get("rows"), list):
        variants = []
        for row in payload["rows"]:
            row = dict(row)
            variant = row.pop("variant")
            merge_metrics(cache, variant, row)
            variants.append(variant)
        save_cache(cache, cache_path)
        print(f"Metrics merged for variants {variants} -> {cache_path}")
        return

    variant = payload.pop("variant")
    merge_metrics(cache, variant, payload)
    save_cache(cache, cache_path)
    print(f"Metrics merged for variant '{variant}' -> {cache_path}")


def _tiles_exist(tile_dir: Path, tile_ids: list[str]) -> bool:
    return all((tile_dir / f"{tile_id}.png").exists() for tile_id in tile_ids)


def _run_full(
    cache_dir: Path,
    tile_ids: list[str],
    device: str,
    *,
    tile_id_cache_key: str,
    variants: list[str] | None = None,
) -> None:
    """Generate qualitative tiles for each variant and update cache metadata."""
    _assert_requested_device(device)
    from diffusion.utils.misc import read_config
    from PIL import Image
    from tools.stage3.common import make_inference_scheduler, resolve_uni_embedding
    from tools.stage3.tile_pipeline import generate_tile, load_all_models, load_channel, resolve_channel_dir, resolve_data_layout

    data_root = Path("data/orion-crc33")
    exp_channels_dir, features_dir, he_dir = resolve_data_layout(data_root)

    cache_path = cache_dir / "cache.json"
    cache = load_cache(cache_path)
    merge_curves(cache, extract_all_curves())
    cache[tile_id_cache_key] = sorted(set(cache.get(tile_id_cache_key, []) + tile_ids))
    save_cache(cache, cache_path)
    param_counts: dict[str, int] = {}

    selected_variants = variants or list(INFERENCE_VARIANTS)
    for variant_key in selected_variants:
        variant_cfg = INFERENCE_VARIANTS[variant_key]
        tile_dir = cache_dir / "tiles" / variant_key
        tile_dir.mkdir(parents=True, exist_ok=True)
        if _tiles_exist(tile_dir, tile_ids):
            print(f"[{variant_key}] tiles already present; skipping inference")
            continue

        config_path = variant_cfg["config_path"]
        config = read_config(config_path)
        config._filename = config_path
        if getattr(config, "work_dir", None):
            Path(config.work_dir).mkdir(parents=True, exist_ok=True)
        models = None
        scheduler = None
        off_shelf_runner = None
        if variant_cfg["variant_type"] == "standard":
            print(f"[{variant_key}] loading models from {variant_cfg['ckpt_dir']}")
            models = load_all_models(config, config_path, variant_cfg["ckpt_dir"], device)
            scheduler = make_inference_scheduler(num_steps=30, device=device)
            if variant_key in ("production", "a1_concat", "a1_per_channel"):
                tme_params = sum(param.numel() for param in models["tme_module"].parameters())
                controlnet_params = sum(param.numel() for param in models["controlnet"].parameters())
                param_counts[variant_key] = tme_params + controlnet_params
                print(f"  {variant_key}: {param_counts[variant_key]:,} parameters")
        elif variant_cfg["variant_type"] == "bypass":
            print(f"[{variant_key}] loading models from {variant_cfg['ckpt_dir']}")
            models = load_all_models(config, config_path, variant_cfg["ckpt_dir"], device)
            scheduler = make_inference_scheduler(num_steps=30, device=device)
        elif variant_cfg["variant_type"] == "off_shelf":
            from tools.baselines.pixcell_offshelf_inference import OffShelfPixCellInference

            off_shelf_runner = OffShelfPixCellInference(
                controlnet_path=variant_cfg["controlnet_path"],
                base_model_path=variant_cfg["base_model_path"],
                vae_path=variant_cfg["vae_path"],
                uni_path=str(features_dir),
                device=device,
                config_path=config_path,
            )
        else:
            raise ValueError(f"unknown variant_type={variant_cfg['variant_type']!r}")

        for tile_id in tile_ids:
            out_path = tile_dir / f"{tile_id}.png"
            if out_path.exists():
                continue
            uni_embeds = resolve_uni_embedding(tile_id, feat_dir=features_dir, null_uni=False)
            if variant_cfg["variant_type"] == "standard":
                gen_np, _ = generate_tile(
                    tile_id,
                    models,
                    config,
                    scheduler,
                    uni_embeds,
                    device,
                    exp_channels_dir,
                    guidance_scale=1.5,
                    seed=42,
                )
            elif variant_cfg["variant_type"] == "bypass":
                gen_np = _generate_bypass_tile(
                    tile_id,
                    models,
                    config,
                    scheduler,
                    uni_embeds,
                    device,
                    exp_channels_dir,
                    guidance_scale=1.5,
                    seed=42,
                )
            elif variant_cfg["variant_type"] == "off_shelf":
                image_size = int(getattr(config.data, "image_size", 256))
                channel_dir = resolve_channel_dir(exp_channels_dir, "cell_masks")
                cell_mask = load_channel(channel_dir, tile_id, image_size, binary=True, channel_name="cell_masks")
                out_path = off_shelf_runner.run_on_tile(
                    tile_id=tile_id,
                    cell_mask=cell_mask,
                    uni_embedding=np.asarray(uni_embeds, dtype=np.float32).reshape(-1)[:1536],
                    out_dir=tile_dir,
                    num_steps=30,
                    guidance_scale=1.5,
                )
                print(f"  wrote {out_path}")
                continue
            else:
                raise ValueError(f"unknown variant_type={variant_cfg['variant_type']!r}")
            Image.fromarray(gen_np).save(out_path)
            print(f"  wrote {out_path}")

        gt_dir = cache_dir / "tiles" / "gt"
        gt_dir.mkdir(parents=True, exist_ok=True)
        for tile_id in tile_ids:
            gt_path = gt_dir / f"{tile_id}.png"
            src_path = he_dir / f"{tile_id}.png"
            if not gt_path.exists() and src_path.exists():
                shutil.copy2(src_path, gt_path)

        del models, scheduler, off_shelf_runner, config
        _release_accelerator_memory(device)

    merge_params(cache, param_counts)
    save_cache(cache, cache_path)
    print(f"Cache saved -> {cache_path}")


def _generate_bypass_tile(
    tile_id: str,
    models: dict,
    config,
    scheduler,
    uni_embeds,
    device: str,
    exp_channels_dir: Path,
    guidance_scale: float,
    seed: int,
) -> "np.ndarray":
    """A2 bypass: TME output zeroed, conditioning equals VAE mask only."""
    import torch
    from tools.stage3.tile_pipeline import _decode_latents_to_image, _make_fixed_noise, prepare_tile_context
    from train_scripts.inference_controlnet import denoise

    context = prepare_tile_context(
        tile_id=tile_id,
        models=models,
        config=config,
        uni_embeds=uni_embeds,
        device=device,
        exp_channels_dir=exp_channels_dir,
    )
    ctrl_input = context["vae_mask"] + torch.zeros_like(context["vae_mask"])
    fixed_noise = _make_fixed_noise(config=config, scheduler=scheduler, device=device, dtype=context["dtype"], seed=seed)
    denoised = denoise(
        latents=fixed_noise,
        uni_embeds=context["uni_embeds"],
        controlnet_input_latent=ctrl_input,
        scheduler=scheduler,
        controlnet_model=context["controlnet"],
        pixcell_controlnet_model=context["base_model"],
        guidance_scale=guidance_scale,
        num_inference_steps=30,
        device=device,
    )
    return _decode_latents_to_image(
        denoised,
        vae=context["vae"],
        vae_scale=context["vae_scale"],
        vae_shift=context["vae_shift"],
        dtype=context["dtype"],
    )


def _record_params(cache: dict, device: str) -> None:
    """Count A1-axis model parameters and store them under cache['params']."""
    from diffusion.utils.misc import read_config
    from tools.stage3.tile_pipeline import load_all_models

    params: dict[str, int] = {}
    for variant_key in ("production", "a1_concat", "a1_per_channel"):
        variant_cfg = INFERENCE_VARIANTS[variant_key]
        config_path = variant_cfg["config_path"]
        config = read_config(config_path)
        config._filename = config_path
        if getattr(config, "work_dir", None):
            Path(config.work_dir).mkdir(parents=True, exist_ok=True)
        models = load_all_models(config, config_path, variant_cfg["ckpt_dir"], device)
        tme_params = sum(param.numel() for param in models["tme_module"].parameters())
        controlnet_params = sum(param.numel() for param in models["controlnet"].parameters())
        params[variant_key] = tme_params + controlnet_params
        print(f"  {variant_key}: {params[variant_key]:,} parameters")
    merge_params(cache, params)


def _release_accelerator_memory(device: str) -> None:
    """Release cached CUDA allocations after a variant finishes."""
    gc.collect()
    if not str(device).lower().startswith("cuda"):
        return
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass


def _assert_requested_device(device: str) -> None:
    """Validate CUDA visibility before importing/loading the full model stack."""
    if not str(device).lower().startswith("cuda"):
        return
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "device='cuda' was requested, but torch.cuda.is_available() is False in this process"
        )
    torch.cuda.init()
    print(f"CUDA ready: {torch.cuda.get_device_name(0)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default="inference_output/si_a1_a2", help="Root dir for cache.json and tiles/")
    parser.add_argument("--update-curves", action="store_true", help="Only update training_curves; skip inference")
    parser.add_argument("--tile-ids-file", default=None, help="One tile ID per line; required for a full run")
    parser.add_argument(
        "--tile-id-cache-key",
        default="tile_ids",
        choices=["tile_ids", "metric_tile_ids"],
        help="Cache key to update with generated tile IDs. Use metric_tile_ids for large metric batches.",
    )
    parser.add_argument("--merge-metrics-file", default=None, help="Pre-computed metrics JSON to merge")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=sorted(INFERENCE_VARIANTS),
        default=None,
        help="Optional subset of variants to generate, e.g. a1_concat a1_per_channel.",
    )
    args = parser.parse_args(argv)

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if args.merge_metrics_file:
        _run_merge_metrics(cache_dir, Path(args.merge_metrics_file))
        return 0
    if args.update_curves:
        _run_update_curves(cache_dir)
        return 0
    if not args.tile_ids_file:
        parser.error("--tile-ids-file is required for a full run")

    tile_ids = [line.strip() for line in Path(args.tile_ids_file).read_text(encoding="utf-8").splitlines() if line.strip()]
    _run_full(
        cache_dir,
        tile_ids,
        args.device,
        tile_id_cache_key=args.tile_id_cache_key,
        variants=args.variants,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
