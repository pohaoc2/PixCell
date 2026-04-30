"""Build the SI A1/A2 cache from retrained debug checkpoints."""
from __future__ import annotations

import argparse
import gc
import shutil
import sys
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.ablation_a1_a2.cache_io import load_cache, merge_curves, merge_params, save_cache
from tools.ablation_a1_a2.log_utils import extract_run


DEBUG_VARIANTS: dict[str, dict[str, str]] = {
    "production": {
        "variant_type": "standard",
        "config_path": "checkpoints/debug/grouped_95470_2/config.py",
        "ckpt_dir": "checkpoints/debug/grouped_95470_2/checkpoints/step_0002600",
    },
    "a1_concat": {
        "variant_type": "standard",
        "config_path": "checkpoints/debug/concat_95470_0/config.py",
        "ckpt_dir": "checkpoints/debug/concat_95470_0/checkpoints/step_0002600",
    },
    "a1_per_channel": {
        "variant_type": "standard",
        "config_path": "checkpoints/debug/per_channel_95470_3/config.py",
        "ckpt_dir": "checkpoints/debug/per_channel_95470_3/checkpoints/step_0002600",
    },
    "a2_bypass_full_tme": {
        "variant_type": "standard",
        "config_path": "checkpoints/debug/additive_95470_1/config.py",
        "ckpt_dir": "checkpoints/debug/additive_95470_1/checkpoints/step_0002600",
    },
    "a2_off_shelf": {
        "variant_type": "off_shelf",
        "config_path": "configs/config_controlnet_exp.py",
        "controlnet_path": "pretrained_models/pixcell-256-controlnet/controlnet/diffusion_pytorch_model.safetensors",
        "base_model_path": "pretrained_models/pixcell-256/transformer",
        "vae_path": "pretrained_models/sd-3.5-vae/vae",
    },
}

DEBUG_CURVE_SOURCES: dict[str, Path] = {
    "production": ROOT / "checkpoints/debug/grouped_95470_2/train_log.jsonl",
    "a1_concat": ROOT / "checkpoints/debug/concat_95470_0/train_log.jsonl",
    "a1_per_channel": ROOT / "checkpoints/debug/per_channel_95470_3/train_log.jsonl",
    "a2_bypass_full_tme": ROOT / "checkpoints/debug/additive_95470_1/train_log.jsonl",
}


def _load_debug_curves() -> dict[str, dict[str, list[dict]]]:
    curves: dict[str, dict[str, list[dict]]] = {}
    for variant, path in DEBUG_CURVE_SOURCES.items():
        entries = extract_run(path)
        if entries:
            curves[variant] = {"full_seed_42": entries}
    return curves


def sync_debug_curves(*, cache_dir: Path, qualitative_cache_dir: Path | None = None) -> Path:
    cache_path = cache_dir / "cache.json"
    cache = load_cache(cache_path)
    if qualitative_cache_dir is not None:
        source_cache = load_cache(qualitative_cache_dir / "cache.json")
        cache["tile_ids"] = list(source_cache.get("tile_ids", []))
    merge_curves(cache, _load_debug_curves())
    save_cache(cache, cache_path)
    return cache_path


def select_metric_tile_ids(
    *,
    cache_dir: Path,
    paired_ablation_root: Path,
    n_tiles: int,
) -> Path:
    source_root = paired_ablation_root / "ablation_results"
    if not source_root.is_dir():
        raise FileNotFoundError(f"missing paired ablation results: {source_root}")

    eligible: list[str] = []
    for tile_dir in sorted(path for path in source_root.iterdir() if path.is_dir()):
        if (
            (tile_dir / "all" / "generated_he.png").is_file()
            and (tile_dir / "all" / "generated_he_cellvit_instances.json").is_file()
            and (tile_dir / "metrics.json").is_file()
        ):
            eligible.append(tile_dir.name)
    if len(eligible) < n_tiles:
        raise ValueError(f"requested {n_tiles} tiles but only found {len(eligible)} eligible paired-ablation tiles")

    selected = eligible[:n_tiles]
    cache_path = cache_dir / "cache.json"
    cache = load_cache(cache_path)
    cache["metric_tile_ids"] = selected
    save_cache(cache, cache_path)
    out_path = cache_dir / "metric_tile_ids.txt"
    out_path.write_text("\n".join(selected) + "\n", encoding="utf-8")
    return out_path


def _tiles_exist(tile_dir: Path, tile_ids: list[str]) -> bool:
    return all((tile_dir / f"{tile_id}.png").is_file() for tile_id in tile_ids)


def _runtime_config_path(cache_dir: Path, variant_key: str) -> Path:
    return cache_dir / "runtime_configs" / f"{variant_key}.py"


def _materialize_runtime_config(*, source_path: Path, cache_dir: Path, variant_key: str) -> Path:
    runtime_config_path = _runtime_config_path(cache_dir, variant_key)
    runtime_config_path.parent.mkdir(parents=True, exist_ok=True)
    runtime_work_dir = cache_dir / "runtime_work_dirs" / variant_key
    runtime_work_dir.mkdir(parents=True, exist_ok=True)

    lines = source_path.read_text(encoding="utf-8").splitlines()
    rewritten: list[str] = []
    replaced = False
    for line in lines:
        if line.strip().startswith("work_dir = "):
            rewritten.append(f"work_dir = {str(runtime_work_dir)!r}")
            replaced = True
        else:
            rewritten.append(line)
    if not replaced:
        rewritten.append(f"work_dir = {str(runtime_work_dir)!r}")
    runtime_config_path.write_text("\n".join(rewritten) + "\n", encoding="utf-8")
    return runtime_config_path


def _release_accelerator_memory(device: str) -> None:
    gc.collect()
    if not str(device).lower().startswith("cuda"):
        return
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass


def generate_tiles(
    *,
    cache_dir: Path,
    tile_ids: list[str],
    tile_id_cache_key: str,
    variants: list[str] | None,
    device: str,
) -> Path:
    from diffusion.utils.misc import read_config
    import numpy as np
    from tools.stage3.common import make_inference_scheduler, resolve_uni_embedding
    from tools.stage3.tile_pipeline import generate_tile, load_all_models, load_channel, resolve_channel_dir, resolve_data_layout

    selected_variants = variants or list(DEBUG_VARIANTS)
    data_root = ROOT / "data/orion-crc33"
    exp_channels_dir, features_dir, he_dir = resolve_data_layout(data_root)
    cache_path = sync_debug_curves(cache_dir=cache_dir, qualitative_cache_dir=ROOT / "inference_output/si_a1_a2")
    cache = load_cache(cache_path)
    cache[tile_id_cache_key] = sorted(set(cache.get(tile_id_cache_key, []) + tile_ids))
    save_cache(cache, cache_path)
    param_counts: dict[str, int] = {}

    for variant_key in selected_variants:
        variant_cfg = DEBUG_VARIANTS[variant_key]
        variant_type = variant_cfg.get("variant_type", "standard")
        source_config_path = ROOT / variant_cfg["config_path"]
        ckpt_dir = ROOT / variant_cfg.get("ckpt_dir", "") if variant_cfg.get("ckpt_dir") else None
        tile_dir = cache_dir / "tiles" / variant_key
        tile_dir.mkdir(parents=True, exist_ok=True)
        if _tiles_exist(tile_dir, tile_ids):
            print(f"[{variant_key}] tiles already present; skipping inference", flush=True)
            continue

        if variant_type == "off_shelf":
            config_path = source_config_path
        else:
            config_path = _materialize_runtime_config(
                source_path=source_config_path,
                cache_dir=cache_dir,
                variant_key=variant_key,
            )
        config = read_config(str(config_path))
        config._filename = str(config_path)
        models = None
        scheduler = None
        off_shelf_runner = None
        if variant_type == "standard":
            models = load_all_models(config, str(config_path), str(ckpt_dir), device)
            scheduler = make_inference_scheduler(num_steps=30, device=device)

            if variant_key in ("production", "a1_concat", "a1_per_channel"):
                tme_params = sum(param.numel() for param in models["tme_module"].parameters())
                controlnet_params = sum(param.numel() for param in models["controlnet"].parameters())
                param_counts[variant_key] = tme_params + controlnet_params
        elif variant_type == "off_shelf":
            from tools.baselines.pixcell_offshelf_inference import OffShelfPixCellInference

            off_shelf_runner = OffShelfPixCellInference(
                controlnet_path=str(ROOT / variant_cfg["controlnet_path"]),
                base_model_path=str(ROOT / variant_cfg["base_model_path"]),
                vae_path=str(ROOT / variant_cfg["vae_path"]),
                uni_path=str(features_dir),
                device=device,
                config_path=str(config_path),
            )
        else:
            raise ValueError(f"unsupported variant_type={variant_type!r}")

        for tile_id in tile_ids:
            out_path = tile_dir / f"{tile_id}.png"
            if out_path.exists():
                continue
            uni_embeds = resolve_uni_embedding(tile_id, feat_dir=features_dir, null_uni=False)
            if variant_type == "standard":
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
                Image.fromarray(gen_np).save(out_path)
            else:
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
            print(f"[{variant_key}] wrote {out_path}", flush=True)

        del models, scheduler, off_shelf_runner, config
        _release_accelerator_memory(device)

    gt_dir = cache_dir / "tiles" / "gt"
    gt_dir.mkdir(parents=True, exist_ok=True)
    for tile_id in tile_ids:
        gt_path = gt_dir / f"{tile_id}.png"
        src_path = he_dir / f"{tile_id}.png"
        if src_path.is_file() and not gt_path.exists():
            shutil.copy2(src_path, gt_path)

    cache = load_cache(cache_path)
    merge_params(cache, param_counts)
    save_cache(cache, cache_path)
    print(f"Cache saved -> {cache_path}", flush=True)
    return cache_path


def _debug_inference_variants(cache_dir: Path, variants: list[str] | None = None) -> dict[str, dict[str, str]]:
    selected = variants or list(DEBUG_VARIANTS)
    specs: dict[str, dict[str, str]] = {}
    for variant_key in selected:
        source_config_path = ROOT / DEBUG_VARIANTS[variant_key]["config_path"]
        if DEBUG_VARIANTS[variant_key].get("variant_type", "standard") == "off_shelf":
            runtime_config_path = source_config_path
        else:
            runtime_config_path = _materialize_runtime_config(
                source_path=source_config_path,
                cache_dir=cache_dir,
                variant_key=variant_key,
            )
        specs[variant_key] = {
            "config_path": str(runtime_config_path),
            "variant_type": DEBUG_VARIANTS[variant_key].get("variant_type", "standard"),
        }
        if DEBUG_VARIANTS[variant_key].get("ckpt_dir"):
            specs[variant_key]["ckpt_dir"] = str(ROOT / DEBUG_VARIANTS[variant_key]["ckpt_dir"])
    return specs


def run_sensitivity(
    *,
    cache_dir: Path,
    tile_ids: list[str],
    device: str,
    variants: list[str] | None,
    guidance_scale: float,
    seed: int,
    render_root: Path | None,
) -> dict[str, dict[str, object]]:
    from tools.ablation_a1_a2 import sensitivity_eval
    from tools.stage3.tile_pipeline import resolve_data_layout

    exp_channels_dir, features_dir, _ = resolve_data_layout(ROOT / "data/orion-crc33")
    original_variants = sensitivity_eval.INFERENCE_VARIANTS
    try:
        sensitivity_eval.INFERENCE_VARIANTS = _debug_inference_variants(cache_dir, variants)
        return sensitivity_eval.run_sensitivity(
            cache_dir=cache_dir,
            tile_ids=tile_ids,
            device=device,
            exp_channels_dir=exp_channels_dir,
            features_dir=features_dir,
            guidance_scale=guidance_scale,
            seed=seed,
            variants=variants,
            render_root=render_root,
        )
    finally:
        sensitivity_eval.INFERENCE_VARIANTS = original_variants


def _read_tile_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_sync = sub.add_parser("sync-curves", help="Write debug curves into cache.json")
    p_sync.add_argument("--cache-dir", type=Path, default=ROOT / "inference_output/debug")
    p_sync.add_argument("--qualitative-cache-dir", type=Path, default=ROOT / "inference_output/si_a1_a2")

    p_select = sub.add_parser("select-metric-tiles", help="Select metric tile IDs only")
    p_select.add_argument("--cache-dir", type=Path, default=ROOT / "inference_output/debug")
    p_select.add_argument("--paired-ablation-root", type=Path, default=ROOT / "inference_output/paired_ablation")
    p_select.add_argument("--n-tiles", type=int, default=300)

    p_generate = sub.add_parser("generate", help="Generate debug tiles for selected canonical variants")
    p_generate.add_argument("--cache-dir", type=Path, default=ROOT / "inference_output/debug")
    p_generate.add_argument("--tile-ids-file", type=Path, required=True)
    p_generate.add_argument("--tile-id-cache-key", choices=["tile_ids", "metric_tile_ids"], default="tile_ids")
    p_generate.add_argument("--device", default="cuda")
    p_generate.add_argument("--variants", nargs="+", choices=sorted(DEBUG_VARIANTS), default=None)

    p_sensitivity = sub.add_parser("sensitivity", help="Run section-4 sensitivity using the debug checkpoints")
    p_sensitivity.add_argument("--cache-dir", type=Path, default=ROOT / "inference_output/debug")
    p_sensitivity.add_argument("--tile-ids-file", type=Path, required=True)
    p_sensitivity.add_argument("--device", default="cuda")
    p_sensitivity.add_argument("--guidance-scale", type=float, default=1.5)
    p_sensitivity.add_argument("--seed", type=int, default=42)
    p_sensitivity.add_argument("--render-root", type=Path, default=None)
    p_sensitivity.add_argument("--variants", nargs="+", choices=sorted(DEBUG_VARIANTS), default=None)

    args = parser.parse_args(argv)
    if args.cmd == "sync-curves":
        path = sync_debug_curves(cache_dir=args.cache_dir, qualitative_cache_dir=args.qualitative_cache_dir)
        print(path)
        return 0
    if args.cmd == "select-metric-tiles":
        out = select_metric_tile_ids(
            cache_dir=args.cache_dir,
            paired_ablation_root=args.paired_ablation_root,
            n_tiles=args.n_tiles,
        )
        print(out)
        return 0
    if args.cmd == "generate":
        generate_tiles(
            cache_dir=args.cache_dir,
            tile_ids=_read_tile_ids(args.tile_ids_file),
            tile_id_cache_key=args.tile_id_cache_key,
            variants=args.variants,
            device=args.device,
        )
        return 0
    if args.cmd == "sensitivity":
        run_sensitivity(
            cache_dir=args.cache_dir,
            tile_ids=_read_tile_ids(args.tile_ids_file),
            device=args.device,
            variants=args.variants,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
            render_root=args.render_root,
        )
        return 0
    raise AssertionError(args.cmd)


if __name__ == "__main__":
    raise SystemExit(main())