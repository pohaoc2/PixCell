"""TME channel sensitivity evaluation for the SI A1/A2 ablation cache."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.ablation_a1_a2.build_cache import INFERENCE_VARIANTS, _release_accelerator_memory
from tools.ablation_a1_a2.cache_io import load_cache, merge_sensitivity, save_cache
from tools.stage3.common import make_inference_scheduler, resolve_uni_embedding

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - exercised in environments without torch
    torch = None  # type: ignore[assignment]

try:
    from tools.stage3.tile_pipeline import (
        _fuse_active_groups,
        _make_fixed_noise,
        _render_fused_ablation_image,
        load_all_models,
        prepare_tile_context,
        resolve_data_layout,
    )
except ModuleNotFoundError:  # pragma: no cover - exercised in lightweight test environments
    _fuse_active_groups = None  # type: ignore[assignment]
    _make_fixed_noise = None  # type: ignore[assignment]
    _render_fused_ablation_image = None  # type: ignore[assignment]
    load_all_models = None  # type: ignore[assignment]
    prepare_tile_context = None  # type: ignore[assignment]
    resolve_data_layout = None  # type: ignore[assignment]

read_config = None


def _ensure_generation_imports() -> None:
    global _fuse_active_groups, _make_fixed_noise, _render_fused_ablation_image
    global prepare_tile_context
    if all(
        value is not None
        for value in (
            _fuse_active_groups,
            _make_fixed_noise,
            _render_fused_ablation_image,
            prepare_tile_context,
        )
    ):
        return

    from tools.stage3.tile_pipeline import (
        _fuse_active_groups as fuse_active_groups,
        _make_fixed_noise as make_fixed_noise,
        _render_fused_ablation_image as render_fused_ablation_image,
        prepare_tile_context as prepare_tile_context_impl,
    )

    _fuse_active_groups = fuse_active_groups
    _make_fixed_noise = make_fixed_noise
    _render_fused_ablation_image = render_fused_ablation_image
    prepare_tile_context = prepare_tile_context_impl


def _ensure_model_imports() -> None:
    global load_all_models
    if load_all_models is None:
        from tools.stage3.tile_pipeline import load_all_models as load_all_models_impl

        load_all_models = load_all_models_impl


def _ensure_layout_imports() -> None:
    global resolve_data_layout
    if resolve_data_layout is None:
        from tools.stage3.tile_pipeline import resolve_data_layout as resolve_data_layout_impl

        resolve_data_layout = resolve_data_layout_impl


def _ensure_read_config() -> None:
    global read_config
    if read_config is None:
        from diffusion.utils.misc import read_config as read_config_impl

        read_config = read_config_impl


def _lpips_fn(device: str = "cpu"):
    if torch is None:
        raise ModuleNotFoundError("torch is required for sensitivity evaluation")

    import lpips

    resolved_device = str(device).lower()
    if resolved_device == "cuda" and not torch.cuda.is_available():
        resolved_device = "cpu"
    cache_key = f"_model_{resolved_device}"
    if not hasattr(_lpips_fn, cache_key):
        model = lpips.LPIPS(net="alex").to(resolved_device)
        model.eval()
        setattr(_lpips_fn, cache_key, model)
    return getattr(_lpips_fn, cache_key), resolved_device


def _to_lpips_tensor(rgb: np.ndarray, *, device: str):
    if torch is None:
        raise ModuleNotFoundError("torch is required for sensitivity evaluation")
    tensor = torch.from_numpy(np.asarray(rgb, dtype=np.float32))
    tensor = (tensor / 127.5) - 1.0
    return tensor.permute(2, 0, 1).unsqueeze(0).to(device)


def compute_lpips(img_a: np.ndarray, img_b: np.ndarray, *, device: str = "cpu") -> float:
    """LPIPS(AlexNet) between two uint8 RGB arrays."""
    if img_a.shape != img_b.shape:
        raise ValueError(f"image shape mismatch: {img_a.shape} != {img_b.shape}")

    model, resolved_device = _lpips_fn(device)
    with torch.no_grad():
        score = model(
            _to_lpips_tensor(img_a, device=resolved_device),
            _to_lpips_tensor(img_b, device=resolved_device),
        )
    return float(score.item())


def compute_sensitivity_scores(
    baseline: np.ndarray,
    group_images: dict[str, np.ndarray],
    *,
    device: str = "cpu",
) -> dict[str, float]:
    """Compute ΔLPIPS for each group-zeroed image against one baseline image."""
    return {
        group: compute_lpips(baseline, perturbed, device=device)
        for group, perturbed in group_images.items()
    }


def _semantic_groups(config) -> list[dict]:
    groups = getattr(config, "channel_groups", None)
    if groups:
        return [dict(group) for group in groups]

    _ensure_read_config()
    reference = read_config("configs/config_controlnet_exp.py")
    return [dict(group) for group in getattr(reference, "channel_groups", [])]


def _group_names_from_config(config) -> list[str]:
    return [str(group["name"]) for group in _semantic_groups(config)]


def _group_channel_indices(config) -> dict[str, list[int]]:
    active_channels = list(config.data.active_channels)
    indices = {name: idx for idx, name in enumerate(active_channels)}
    grouped_indices: dict[str, list[int]] = {}
    for group in _semantic_groups(config):
        members = [indices[channel] for channel in group.get("channels", []) if channel in indices]
        if members:
            grouped_indices[str(group["name"])] = members
    return grouped_indices


def summarize_variant_sensitivity(group_scores: dict[str, list[float]]) -> dict[str, object]:
    """Aggregate per-tile group LPIPS deltas into a per-variant summary."""
    per_group: dict[str, dict[str, float | list[float]]] = {}
    group_means: list[float] = []
    for group, values in group_scores.items():
        arr = np.asarray(values, dtype=np.float32)
        mean = float(arr.mean())
        per_group[group] = {
            "mean": mean,
            "std": float(arr.std()),
            "per_tile": [float(value) for value in values],
        }
        group_means.append(mean)

    if group_means:
        group_mean_arr = np.asarray(group_means, dtype=np.float32)
        return {
            "mean": float(group_mean_arr.mean()),
            "std": float(group_mean_arr.std()),
            "per_group": per_group,
        }
    return {"mean": 0.0, "std": 0.0, "per_group": {}}


def generate_group_ablations(
    tile_id: str,
    *,
    models: dict,
    config,
    scheduler,
    uni_embeds,
    device: str,
    exp_channels_dir: Path,
    guidance_scale: float = 1.5,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Generate one image per group with that group removed from the active set."""
    _ensure_generation_imports()
    context = prepare_tile_context(
        tile_id=tile_id,
        models=models,
        config=config,
        uni_embeds=uni_embeds,
        device=device,
        exp_channels_dir=exp_channels_dir,
    )
    fixed_noise = _make_fixed_noise(
        config=config,
        scheduler=scheduler,
        device=device,
        dtype=context["dtype"],
        seed=seed,
    )

    if context.get("conditioning_mode", "grouped") == "grouped":
        group_names = list(context["tme_module"].group_names)
        outputs: dict[str, np.ndarray] = {}
        for zeroed_group in group_names:
            active_groups = [group for group in group_names if group != zeroed_group]
            fused = _fuse_active_groups(context=context, active_groups=active_groups)
            outputs[zeroed_group] = _render_fused_ablation_image(
                fused,
                context=context,
                scheduler=scheduler,
                guidance_scale=guidance_scale,
                device=device,
                seed=seed,
                fixed_noise=fixed_noise.clone(),
            )
        return outputs

    grouped_indices = _group_channel_indices(config)
    outputs: dict[str, np.ndarray] = {}
    for zeroed_group, channel_indices in grouped_indices.items():
        ablation_context = dict(context)
        ablation_context["tme_inputs"] = context["tme_inputs"].clone()
        ablation_context["tme_inputs"][:, channel_indices] = 0
        fused = _fuse_active_groups(context=ablation_context, active_groups=None)
        outputs[zeroed_group] = _render_fused_ablation_image(
            fused,
            context=ablation_context,
            scheduler=scheduler,
            guidance_scale=guidance_scale,
            device=device,
            seed=seed,
            fixed_noise=fixed_noise.clone(),
        )
    return outputs


def run_sensitivity(
    *,
    cache_dir: Path,
    tile_ids: list[str],
    device: str,
    exp_channels_dir: Path,
    features_dir: Path,
    guidance_scale: float = 1.5,
    seed: int = 42,
    variants: list[str] | None = None,
) -> dict[str, dict[str, object]]:
    """Compute per-variant sensitivity summaries and write them into cache.json."""
    _ensure_model_imports()
    _ensure_read_config()
    sensitivity: dict[str, dict[str, object]] = {}
    selected_variants = variants or list(INFERENCE_VARIANTS)

    for variant_key in selected_variants:
        variant_cfg = INFERENCE_VARIANTS[variant_key]
        config_path = str(variant_cfg["config_path"])
        config = read_config(config_path)
        config._filename = config_path
        baseline_dir = cache_dir / "tiles" / variant_key
        per_group_scores: dict[str, list[float]] = {}

        models = None
        scheduler = None
        if variant_cfg.get("variant_type") not in {"bypass", "off_shelf"}:
            models = load_all_models(config, config_path, Path(variant_cfg["ckpt_dir"]), device)
            scheduler = make_inference_scheduler(num_steps=30, device=device)

        for tile_id in tile_ids:
            baseline_path = baseline_dir / f"{tile_id}.png"
            if not baseline_path.is_file():
                print(f"  [sensitivity] baseline tile missing: {baseline_path} - skip")
                continue

            baseline_rgb = np.asarray(Image.open(baseline_path).convert("RGB"))
            if variant_cfg.get("variant_type") in {"bypass", "off_shelf"}:
                group_images = {
                    group_name: baseline_rgb.copy()
                    for group_name in _group_names_from_config(config)
                }
            else:
                uni_embeds = resolve_uni_embedding(tile_id, feat_dir=features_dir, null_uni=False)
                group_images = generate_group_ablations(
                    tile_id,
                    models=models,
                    config=config,
                    scheduler=scheduler,
                    uni_embeds=uni_embeds,
                    device=device,
                    exp_channels_dir=exp_channels_dir,
                    guidance_scale=guidance_scale,
                    seed=seed,
                )
            scores = compute_sensitivity_scores(baseline_rgb, group_images, device=device)
            for group, score in scores.items():
                per_group_scores.setdefault(group, []).append(float(score))

        sensitivity[variant_key] = summarize_variant_sensitivity(per_group_scores)

        del models, scheduler, config
        _release_accelerator_memory(device)

    cache_path = cache_dir / "cache.json"
    cache = load_cache(cache_path)
    merge_sensitivity(cache, sensitivity)
    save_cache(cache, cache_path)
    return sensitivity


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default="inference_output/si_a1_a2")
    parser.add_argument("--tile-ids-file", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--guidance-scale", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=sorted(INFERENCE_VARIANTS),
        default=None,
    )
    args = parser.parse_args(argv)

    cache_dir = Path(args.cache_dir)
    tile_ids = [
        line.strip()
        for line in Path(args.tile_ids_file).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    _ensure_layout_imports()
    exp_channels_dir, features_dir, _ = resolve_data_layout(Path("data/orion-crc33"))
    run_sensitivity(
        cache_dir=cache_dir,
        tile_ids=tile_ids,
        device=args.device,
        exp_channels_dir=exp_channels_dir,
        features_dir=features_dir,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        variants=args.variants,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())