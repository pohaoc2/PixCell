from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _to_tensor(image):
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(np.transpose(array, (2, 0, 1)))


class _FakeViT(nn.Module):
    num_prefix_tokens = 9

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.proj = nn.Linear(3, embed_dim)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        return torch.zeros(batch, 265, self.embed_dim, device=x.device)


class _FakeWrapper:
    def __init__(self, model: nn.Module):
        self.model = model
        self.transform = _to_tensor


class _FakeNorm(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        return torch.zeros(batch, 7, 7, self.embed_dim, device=x.device)


class _FakeCTransPath(nn.Module):
    def __init__(self, embed_dim: int = 768):
        super().__init__()
        self.norm = _FakeNorm(embed_dim)
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(x).permute(0, 2, 3, 1))


class _FakeResNetLayer4(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(x.shape[0], 2048, 7, 7, device=x.device)


class _FakeResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer4 = _FakeResNetLayer4()
        self.proj = nn.Conv2d(3, 64, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _ = self.proj(x)
        return self.layer4(x).mean(dim=(2, 3))


def test_extract_vit_patches_shape():
    from pipeline.patch_extractors import extract_vit_patches

    model = _FakeViT(embed_dim=1536).eval()
    images = [np.zeros((224, 224, 3), dtype=np.uint8)]
    out = extract_vit_patches(model, images, transform=_to_tensor)

    assert out.shape == (1, 256, 1536)
    assert out.dtype == np.float16


def test_extract_uni_and_virchow_patches_shape():
    from pipeline.patch_extractors import extract_uni_patches, extract_virchow_patches

    wrapper = _FakeWrapper(_FakeViT(embed_dim=1280).eval())
    images = [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(2)]

    uni = extract_uni_patches(wrapper, images)
    virchow = extract_virchow_patches(wrapper, images)

    assert uni.shape == (2, 256, 1280)
    assert virchow.shape == (2, 256, 1280)
    assert uni.dtype == np.float16
    assert virchow.dtype == np.float16


def test_extract_ctranspath_patches_shape():
    from pipeline.patch_extractors import extract_ctranspath_patches

    wrapper = _FakeWrapper(_FakeCTransPath().eval())
    images = [np.zeros((224, 224, 3), dtype=np.uint8)]
    out = extract_ctranspath_patches(wrapper, images)

    assert out.shape == (1, 49, 768)
    assert out.dtype == np.float16


def test_extract_resnet50_patches_shape():
    from pipeline.patch_extractors import extract_resnet50_patches

    wrapper = _FakeWrapper(_FakeResNet().eval())
    images = [np.zeros((224, 224, 3), dtype=np.uint8), np.zeros((224, 224, 3), dtype=np.uint8)]
    out = extract_resnet50_patches(wrapper, images)

    assert out.shape == (2, 49, 2048)
    assert out.dtype == np.float16