"""Reusable patch-feature extractors for frozen image encoders."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as tv_transforms


_DEFAULT_CNN_TRANSFORM = tv_transforms.Compose(
    [
        tv_transforms.Resize((224, 224)),
        tv_transforms.ToTensor(),
        tv_transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


def _to_pil(image: Any) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, np.ndarray):
        return Image.fromarray(image).convert("RGB")
    raise TypeError(f"unsupported image type: {type(image)!r}")


def _resolve_model(model_or_wrapper: Any) -> Any:
    return getattr(model_or_wrapper, "model", model_or_wrapper)


def _resolve_transform(model_or_wrapper: Any, default: Any) -> Any:
    return getattr(model_or_wrapper, "transform", None) or default


def _model_device(model: Any) -> torch.device:
    try:
        return next(model.parameters()).device
    except (StopIteration, AttributeError, TypeError):
        return torch.device("cpu")


def _default_vit_transform(model: Any) -> Any:
    try:
        from timm.data import resolve_data_config
        from timm.data.transforms_factory import create_transform

        return create_transform(**resolve_data_config(getattr(model, "pretrained_cfg", {}), model=model))
    except Exception:
        return tv_transforms.Compose(
            [
                tv_transforms.Resize((224, 224)),
                tv_transforms.ToTensor(),
            ]
        )


def _stack_inputs(images: Sequence[Any], *, transform: Any, device: torch.device) -> torch.Tensor:
    tensors = [transform(_to_pil(image)) for image in images]
    return torch.stack(tensors, dim=0).to(device)


def _normalize_spatial_output(features: torch.Tensor) -> torch.Tensor:
    if features.ndim == 4 and features.shape[1] > features.shape[-1]:
        batch, channels, height, width = features.shape
        return features.permute(0, 2, 3, 1).reshape(batch, height * width, channels)
    if features.ndim == 4:
        batch, height, width, channels = features.shape
        return features.reshape(batch, height * width, channels)
    if features.ndim == 3:
        return features
    raise RuntimeError(f"unexpected feature map shape {tuple(features.shape)}")


@torch.no_grad()
def extract_vit_patches(
    model: Any,
    images: Sequence[Any],
    *,
    transform: Any | None = None,
) -> np.ndarray:
    """Return patch tokens from a ViT-like encoder as (N, P, D) float16."""
    model = _resolve_model(model)
    transform = transform or _default_vit_transform(model)
    batch = _stack_inputs(images, transform=transform, device=_model_device(model))

    tokens = model.forward_features(batch)
    if isinstance(tokens, (tuple, list)):
        tokens = tokens[0]
    if tokens.ndim != 3:
        raise RuntimeError(f"expected (B, T, D) tokens; got {tuple(tokens.shape)}")

    prefix_tokens = int(getattr(model, "num_prefix_tokens", 1))
    return tokens[:, prefix_tokens:, :].to(torch.float16).cpu().numpy()


@torch.no_grad()
def extract_hooked_patches(
    model: Any,
    target_layer: Any,
    images: Sequence[Any],
    *,
    transform: Any | None = None,
) -> np.ndarray:
    """Capture a spatial activation tensor and return it as (N, P, D) float16."""
    model = _resolve_model(model)
    transform = transform or _DEFAULT_CNN_TRANSFORM
    batch = _stack_inputs(images, transform=transform, device=_model_device(model))

    captured: list[torch.Tensor] = []

    def _hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
        tensor = output[0] if isinstance(output, (tuple, list)) else output
        captured.append(tensor.detach())

    handle = target_layer.register_forward_hook(_hook)
    try:
        _ = model(batch)
    finally:
        handle.remove()

    if not captured:
        raise RuntimeError("hooked layer produced no output")
    return _normalize_spatial_output(captured[-1]).to(torch.float16).cpu().numpy()


def extract_uni_patches(model_or_extractor: Any, images: Sequence[Any]) -> np.ndarray:
    if hasattr(model_or_extractor, "extract_patch_tokens_batch"):
        return np.asarray(model_or_extractor.extract_patch_tokens_batch(images), dtype=np.float16)
    model = _resolve_model(model_or_extractor)
    transform = _resolve_transform(model_or_extractor, _default_vit_transform(model))
    return extract_vit_patches(model, images, transform=transform)


def extract_virchow_patches(model_or_extractor: Any, images: Sequence[Any]) -> np.ndarray:
    model = _resolve_model(model_or_extractor)
    transform = _resolve_transform(model_or_extractor, _default_vit_transform(model))
    return extract_vit_patches(model, images, transform=transform)


def extract_ctranspath_patches(model_or_extractor: Any, images: Sequence[Any]) -> np.ndarray:
    model = _resolve_model(model_or_extractor)
    target_layer = getattr(model, "norm", None)
    if target_layer is None:
        raise RuntimeError("CTransPath model is missing a .norm layer for patch extraction")
    transform = _resolve_transform(model_or_extractor, _DEFAULT_CNN_TRANSFORM)
    return extract_hooked_patches(model, target_layer, images, transform=transform)


def extract_resnet50_patches(model_or_extractor: Any, images: Sequence[Any]) -> np.ndarray:
    model = _resolve_model(model_or_extractor)
    target_layer = getattr(model, "layer4", None)
    if target_layer is None:
        raise RuntimeError("ResNet-50 model is missing .layer4 for patch extraction")
    transform = _resolve_transform(model_or_extractor, _DEFAULT_CNN_TRANSFORM)
    return extract_hooked_patches(model, target_layer, images, transform=transform)


__all__ = [
    "extract_ctranspath_patches",
    "extract_hooked_patches",
    "extract_resnet50_patches",
    "extract_uni_patches",
    "extract_virchow_patches",
    "extract_vit_patches",
]