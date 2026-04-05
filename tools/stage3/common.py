from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from diffusers import DDPMScheduler


def inference_dtype(device: str) -> torch.dtype:
    """Inference dtype for a requested device string."""
    return torch.float16 if str(device).lower().startswith("cuda") else torch.float32


def make_inference_scheduler(*, num_steps: int, device: str) -> DDPMScheduler:
    """Construct the shared DDPM inference scheduler used across Stage 3 tools."""
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


def resolve_uni_embedding(
    tile_id: str,
    *,
    feat_dir: Path,
    null_uni: bool,
    uni_npy: Path | None = None,
) -> torch.Tensor:
    """Load a cached UNI embedding or fall back to the canonical null embedding."""
    from train_scripts.inference_controlnet import null_uni_embed

    feat_path = Path(uni_npy) if uni_npy is not None else Path(feat_dir) / f"{tile_id}_uni.npy"
    if null_uni or not feat_path.exists():
        if not null_uni and not feat_path.exists():
            print(f"Warning: missing {feat_path}, using null UNI")
        return null_uni_embed(device="cpu", dtype=torch.float32)
    return torch.from_numpy(np.load(feat_path)).view(1, 1, 1, 1536)


def print_progress(completed: int, total: int, *, prefix: str) -> None:
    """Write a simple in-place progress bar to stderr."""
    total = max(1, total)
    width = 28
    filled = int(width * completed / total)
    bar = "#" * filled + "-" * (width - filled)
    msg = f"\r{prefix} [{bar}] {completed}/{total}"
    if completed >= total:
        msg += "\n"
    print(msg, end="", file=sys.stderr, flush=True)


def to_uint8_rgb(
    image: np.ndarray,
    *,
    value_range: str = "auto",
) -> np.ndarray:
    """Convert grayscale or RGB image data to uint8 RGB."""
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    elif arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[..., :3]
    elif arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"expected HxW, HxWx1, HxWx3, or HxWx4 image; got {arr.shape}")

    if arr.dtype == np.uint8:
        return arr

    arr = arr.astype(np.float32, copy=False)
    normalized_range = value_range
    if normalized_range == "auto":
        if arr.size == 0:
            normalized_range = "unit"
        else:
            min_val = float(np.nanmin(arr))
            max_val = float(np.nanmax(arr))
            normalized_range = "unit" if 0.0 <= min_val and max_val <= 1.0 else "byte"

    if normalized_range == "unit":
        return (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
    if normalized_range == "byte":
        return np.clip(arr, 0.0, 255.0).astype(np.uint8)
    raise ValueError(f"unsupported value_range={value_range!r}")
