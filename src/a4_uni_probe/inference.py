"""Thin inference wrapper that overrides UNI embeddings for Stage 3 generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src._tasklib.io import ensure_directory


ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class GenSpec:
    tile_id: str
    uni: np.ndarray
    out_path: Path


@dataclass(frozen=True)
class InferenceBundle:
    config: Any
    models: dict[str, Any]
    scheduler: Any
    exp_channels_dir: Path
    device: str


def _default_device() -> str:
    try:
        import torch
    except Exception:  # pragma: no cover
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_config(config_path: Path):
    from diffusion.utils.misc import read_config

    config = read_config(str(config_path))
    config._filename = str(config_path)
    return config


def _as_uni_tensor(uni: np.ndarray, *, device: str):
    import torch

    array = np.asarray(uni, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, 1, 1, -1)
    elif array.ndim == 2:
        array = array.reshape(array.shape[0], 1, 1, array.shape[-1])
    return torch.from_numpy(array).to(device=device, dtype=torch.float32)


def load_inference_bundle(
    *,
    checkpoint_dir: str | Path,
    config_path: str | Path,
    data_root: str | Path | None = None,
    exp_channels_dir: str | Path | None = None,
    num_steps: int,
    device: str | None = None,
) -> InferenceBundle:
    from tools.stage3.common import make_inference_scheduler
    from tools.stage3.tile_pipeline import load_all_models, resolve_data_layout

    resolved_device = device or _default_device()
    config_path = Path(config_path)
    checkpoint_dir = Path(checkpoint_dir)
    config = _load_config(config_path)
    models = load_all_models(config, config_path, checkpoint_dir, resolved_device)
    scheduler = make_inference_scheduler(num_steps=num_steps, device=resolved_device)

    if exp_channels_dir is not None:
        channels_dir = Path(exp_channels_dir)
    else:
        data_root = Path(data_root) if data_root is not None else ROOT / "data" / "orion-crc33"
        channels_dir, _, _ = resolve_data_layout(data_root)
    return InferenceBundle(
        config=config,
        models=models,
        scheduler=scheduler,
        exp_channels_dir=channels_dir,
        device=resolved_device,
    )


def generate_with_uni_override(
    spec: GenSpec,
    *,
    checkpoint_dir: str | Path,
    config_path: str | Path,
    data_root: str | Path | None = None,
    exp_channels_dir: str | Path | None = None,
    num_steps: int,
    guidance_scale: float,
    seed: int,
    device: str | None = None,
    bundle: InferenceBundle | None = None,
) -> Path:
    from PIL import Image
    from tools.stage3.tile_pipeline import generate_tile

    resources = bundle or load_inference_bundle(
        checkpoint_dir=checkpoint_dir,
        config_path=config_path,
        data_root=data_root,
        exp_channels_dir=exp_channels_dir,
        num_steps=num_steps,
        device=device,
    )
    uni_tensor = _as_uni_tensor(spec.uni, device=resources.device)
    image, _ = generate_tile(
        spec.tile_id,
        resources.models,
        resources.config,
        resources.scheduler,
        uni_tensor,
        resources.device,
        resources.exp_channels_dir,
        guidance_scale,
        seed=seed,
    )
    out_path = Path(spec.out_path)
    ensure_directory(out_path.parent)
    Image.fromarray(image).save(out_path)
    return out_path
