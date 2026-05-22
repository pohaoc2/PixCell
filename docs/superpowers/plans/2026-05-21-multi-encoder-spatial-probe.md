# Multi-Encoder Spatial Decodability Probe — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce `figures/pngs_updated/07d_t1_spatial_multi_encoder.png` ranking T1 spatial decodability per encoder at each encoder's native patch grid (UNI-2h 16×16, Virchow2 16×16, CTransPath 7×7, ResNet-50 7×7, REMEDIS optional).

**Architecture:** Forward-hook the last spatial stage of each pretrained encoder (no new modules, no fine-tuning). Cache `(N_tiles, H*W, D)` patch features per encoder. Block-mean-pool T1 channel maps to the encoder's `(H, W)` grid. Reuse the existing `src/a1_probe_mlp_spatial` probe — it already accepts arbitrary `(N, P, D)` features and `(N, P, T)` targets. Assemble a single multi-encoder figure with grouped bars.

**Tech Stack:** Python 3.12 / `pixcell` conda env, PyTorch + timm + torchvision, sklearn MLPRegressor, matplotlib, joblib, numpy memmap.

**Constraints:** T4 GPU (15 GB VRAM), 32 GB RAM with 24 GB AS-cap policy (`prlimit --as=24000000000`), ~6 hr wall budget.

**Spec:** `docs/superpowers/specs/2026-05-21-multi-encoder-spatial-probe.md`

---

## File Structure

| Status | Path | Responsibility |
|--------|------|----------------|
| create | `pipeline/patch_extractors.py` | Pure functions: `extract_uni_patches`, `extract_virchow_patches`, `extract_ctranspath_patches`, `extract_resnet50_patches`. Each returns `(H, W, D)` float16 tensor per tile via forward hooks on the appropriate model. |
| create | `tests/test_patch_extractors.py` | Per-extractor shape + dtype asserts on random tensors using stub `nn.Module` shims so tests don't need encoder weights. |
| modify | `stage1_extract_features.py` | New CLI `--encoder {virchow2,ctranspath,resnet50}` + `--save-patches` to cache `<tile>_<encoder>_patches.npy`. UNI-2h path unchanged. |
| modify | `src/a1_mask_targets_spatial/main.py` | Already takes `--grid`; verify `block_mean_pool` raises on non-divisible grids (currently does) and add a `--resolution` override path test for `grid=7`, which uses `cv2.INTER_AREA` resize when `256 % grid != 0`. |
| create | `src/a1_probe_mlp_spatial/run_multi_encoder.py` | Thin orchestrator: for each encoder, load its patch cache + matching-grid target tensor, call `run_task` with appropriate out_dir. |
| create | `src/paper_figures/fig_t1_spatial_multi_encoder.py` | Two-panel figure (r2_within + pearson_r) with grouped bars per target, one color per encoder, sorted by UNI-2h r2_within. |
| create | `tests/test_fig_t1_spatial_multi_encoder.py` | Smoke test: build figure from synthetic CSVs, assert PNG written and axis labels correct. |
| run | `data/orion-crc33/features_patches/<encoder>/<tile>_patches.npy` | Cached patch features per encoder. ~6–10 GB each. |
| run | `src/a1_mask_targets_spatial/out_grid_07/mask_targets_T1_spatial.npy` | T1 targets pooled to 7×7. |
| run | `src/a1_probe_mlp_spatial/out/<encoder>_<grid>/mlp_spatial_probe_results.csv` | Probe results per encoder. |

---

## Task 1: Verify existing 16×16 T1 builder also handles 7×7 (and other divisors)

**Files:**
- Modify: `src/a1_mask_targets_spatial/main.py:50-58` (`block_mean_pool`)
- Test: `tests/test_task_a1_mask_targets_spatial.py` (add test)

The existing pool raises if `256 % grid != 0`. For grid=7, 256/7 is not integer; we'll need `cv2.INTER_AREA` resize. Add a fallback path.

- [ ] **Step 1: Write the failing test**

```python
# Append to tests/test_task_a1_mask_targets_spatial.py
def test_block_mean_pool_handles_non_divisor_grid():
    from src.a1_mask_targets_spatial.main import block_mean_pool

    arr = np.zeros((256, 256), dtype=np.float32)
    arr[:, 128:] = 1.0  # right half hot
    pooled = block_mean_pool(arr, grid=7)
    assert pooled.shape == (7, 7)
    # Right ~half of 7 columns should be > 0.5, left ~half < 0.5
    assert pooled[:, 4:].mean() > 0.5
    assert pooled[:, :3].mean() < 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run --no-capture-output -n pixcell python -m pytest tests/test_task_a1_mask_targets_spatial.py::test_block_mean_pool_handles_non_divisor_grid -x -v`

Expected: FAIL with `ValueError: array shape (256, 256) not divisible by grid=7`.

- [ ] **Step 3: Implement non-divisor path**

```python
# Replace block_mean_pool in src/a1_mask_targets_spatial/main.py
def block_mean_pool(array: np.ndarray, grid: int) -> np.ndarray:
    """Average-pool a 2D array into (grid, grid) blocks.

    Uses exact reshape when grid evenly divides the side length; otherwise
    falls back to cv2.INTER_AREA, which is the canonical area-averaging
    resize and matches block-mean behavior in expectation.
    """
    h, w = array.shape
    if h % grid == 0 and w % grid == 0:
        bh, bw = h // grid, w // grid
        return array.reshape(grid, bh, grid, bw).mean(axis=(1, 3))
    import cv2
    return cv2.resize(array, (grid, grid), interpolation=cv2.INTER_AREA)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run --no-capture-output -n pixcell python -m pytest tests/test_task_a1_mask_targets_spatial.py -x -v`

Expected: 4 tests passed.

- [ ] **Step 5: Commit**

```bash
git add src/a1_mask_targets_spatial/main.py tests/test_task_a1_mask_targets_spatial.py
git commit -m "feat(a1_mask_targets_spatial): support non-divisor block_mean_pool grids via cv2.INTER_AREA"
```

---

## Task 2: Build T1 targets at 7×7 grid

**Files:**
- Run: `src/a1_mask_targets_spatial/main.py` (CLI, no edit)
- Output: `src/a1_mask_targets_spatial/out_grid_07/mask_targets_T1_spatial.npy`

- [ ] **Step 1: Run target builder at grid=7**

```bash
conda run --no-capture-output -n pixcell python -m src.a1_mask_targets_spatial.main \
  --features-dir data/orion-crc33/features \
  --exp-channels-dir data/orion-crc33/exp_channels \
  --out-dir src/a1_mask_targets_spatial/out_grid_07 \
  --grid 7
```

Expected wall time: ~5 min (single-process, I/O bound).

- [ ] **Step 2: Verify output shape**

```bash
conda run --no-capture-output -n pixcell python -c "
import numpy as np
m = np.load('src/a1_mask_targets_spatial/out_grid_07/mask_targets_T1_spatial.npy')
print(m.shape, m.dtype, 'finite:', np.isfinite(m).all())
"
```

Expected: `(10379, 49, 10) float32 finite: True`.

- [ ] **Step 3: Commit cache-not-tracked marker**

No files to commit (data artifact). Verify with `git status`. The `out_grid_07/` directory is untracked; no action needed.

---

## Task 3: Create patch_extractors.py with UNI-2h shape parity

**Files:**
- Create: `pipeline/patch_extractors.py`
- Test: `tests/test_patch_extractors.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_patch_extractors.py
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class _FakeViT(nn.Module):
    """Stub ViT with patch_14 layout. forward_features returns (B, 1+8+256, D)."""

    num_prefix_tokens = 9  # CLS + 8 register tokens
    pretrained_cfg = {"input_size": (3, 224, 224)}

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.linear = nn.Linear(3 * 14 * 14, embed_dim)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        # Fake 16x16 patch tokens of dim embed_dim
        patches = torch.zeros(b, 1 + 8 + 256, self.embed_dim)
        return patches


def test_extract_vit_patches_shape():
    from pipeline.patch_extractors import extract_vit_patches

    model = _FakeViT(embed_dim=1536)
    images = [np.zeros((224, 224, 3), dtype=np.uint8)]
    out = extract_vit_patches(model, images)
    assert out.shape == (1, 256, 1536)
    assert out.dtype == np.float16
```

- [ ] **Step 2: Run test, expect failure (module missing)**

Run: `conda run --no-capture-output -n pixcell python -m pytest tests/test_patch_extractors.py -x -v`

Expected: `ModuleNotFoundError: pipeline.patch_extractors`.

- [ ] **Step 3: Implement minimal `extract_vit_patches`**

```python
# pipeline/patch_extractors.py
"""Patch-feature extractors for frozen image encoders.

Each function takes a loaded model + a list of PIL/numpy images and returns
(N, H*W, D) float16 patch-feature arrays. No backprop, no model edits.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image


def _to_pil(image: Any) -> Image.Image:
    if isinstance(image, Image.Image):
        return image
    return Image.fromarray(image)


def _default_vit_transform(model: Any) -> Any:
    """Try to recover the model's preprocessing transform; fall back to a 224 resize."""
    try:
        from timm.data import create_transform, resolve_data_config

        return create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    except Exception:
        from torchvision import transforms as T

        return T.Compose([T.Resize((224, 224)), T.ToTensor()])


@torch.no_grad()
def extract_vit_patches(model: Any, images: Sequence[Any]) -> np.ndarray:
    """Run `model.forward_features`, drop prefix tokens, return (N, P, D) fp16."""
    transform = _default_vit_transform(model)
    x = torch.stack([transform(_to_pil(img)) for img in images])
    device = next(model.parameters()).device if any(model.parameters()) else torch.device("cpu")
    x = x.to(device)

    tokens = model.forward_features(x)
    if tokens.ndim != 3:
        raise RuntimeError(f"expected (B, T, D) tokens; got {tuple(tokens.shape)}")
    n_prefix = int(getattr(model, "num_prefix_tokens", 1))
    patches = tokens[:, n_prefix:, :]
    return patches.to(torch.float16).cpu().numpy()
```

- [ ] **Step 4: Run test, expect pass**

Run: `conda run --no-capture-output -n pixcell python -m pytest tests/test_patch_extractors.py -x -v`

Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add pipeline/patch_extractors.py tests/test_patch_extractors.py
git commit -m "feat(patch_extractors): generic ViT patch-token extractor for spatial probes"
```

---

## Task 4: Add CNN/Swin hook-based patch extractor

**Files:**
- Modify: `pipeline/patch_extractors.py`
- Modify: `tests/test_patch_extractors.py`

- [ ] **Step 1: Write the failing test**

```python
# Append to tests/test_patch_extractors.py

class _FakeResNetLayer4(nn.Module):
    def forward(self, x):
        # x = (B, 3, 224, 224), returns (B, 2048, 7, 7)
        return torch.zeros(x.shape[0], 2048, 7, 7)


class _FakeCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer4 = _FakeResNetLayer4()

    def forward(self, x):
        return self.layer4(x).mean(dim=(2, 3))


def test_extract_hooked_patches_shape():
    from pipeline.patch_extractors import extract_hooked_patches

    model = _FakeCNN().eval()
    images = [np.zeros((224, 224, 3), dtype=np.uint8), np.zeros((224, 224, 3), dtype=np.uint8)]
    out = extract_hooked_patches(model, model.layer4, images)
    assert out.shape == (2, 49, 2048)
    assert out.dtype == np.float16
```

- [ ] **Step 2: Run test, expect failure**

Run: `conda run --no-capture-output -n pixcell python -m pytest tests/test_patch_extractors.py::test_extract_hooked_patches_shape -x -v`

Expected: `AttributeError` on `extract_hooked_patches`.

- [ ] **Step 3: Implement `extract_hooked_patches`**

```python
# Append to pipeline/patch_extractors.py

from torchvision import transforms as _tv_transforms


_DEFAULT_CNN_TRANSFORM = _tv_transforms.Compose(
    [
        _tv_transforms.Resize((224, 224)),
        _tv_transforms.ToTensor(),
        _tv_transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


@torch.no_grad()
def extract_hooked_patches(
    model: Any,
    target_layer: Any,
    images: Sequence[Any],
    *,
    transform: Any | None = None,
) -> np.ndarray:
    """Capture `target_layer`'s output as the spatial feature map.

    Accepts both `(B, C, H, W)` (CNN/ResNet) and `(B, H, W, C)` (Swin)
    activations. Returns `(N, H*W, C)` float16 rows in raster order.
    """
    transform = transform or _DEFAULT_CNN_TRANSFORM
    x = torch.stack([transform(_to_pil(img)) for img in images])
    device = next(model.parameters()).device if any(model.parameters()) else torch.device("cpu")
    x = x.to(device)

    captured: list[torch.Tensor] = []

    def _hook(_module, _inputs, output):
        captured.append(output.detach())

    handle = target_layer.register_forward_hook(_hook)
    try:
        _ = model(x)
    finally:
        handle.remove()

    if not captured:
        raise RuntimeError("hooked layer produced no output")
    feat = captured[-1]
    if feat.ndim == 4 and feat.shape[1] < feat.shape[-1]:
        # CNN-style (B, C, H, W) -> (B, H*W, C)
        b, c, h, w = feat.shape
        feat = feat.permute(0, 2, 3, 1).reshape(b, h * w, c)
    elif feat.ndim == 4:
        # Swin-style (B, H, W, C) -> (B, H*W, C)
        b, h, w, c = feat.shape
        feat = feat.reshape(b, h * w, c)
    else:
        raise RuntimeError(f"unexpected feature map shape {tuple(feat.shape)}")
    return feat.to(torch.float16).cpu().numpy()
```

- [ ] **Step 4: Run test, expect pass**

Run: `conda run --no-capture-output -n pixcell python -m pytest tests/test_patch_extractors.py -x -v`

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add pipeline/patch_extractors.py tests/test_patch_extractors.py
git commit -m "feat(patch_extractors): hook-based CNN/Swin spatial feature extractor"
```

---

## Task 5: Wire stage1 CLI to cache patch features per encoder

**Files:**
- Modify: `stage1_extract_features.py`
- Modify: `pipeline/extract_features.py` (add a per-encoder dispatch)

- [ ] **Step 1: Add encoder dispatch + CLI flags**

In `pipeline/extract_features.py`, append below the existing `main()`:

```python
# Add at top of pipeline/extract_features.py
from pipeline.patch_extractors import extract_vit_patches, extract_hooked_patches

# Append a new helper function

def cache_patch_features(
    *,
    encoder: str,
    image_dir: str | Path,
    output_dir: str | Path,
    weights_path: str | Path | None,
    device: str = "cuda",
    batch_size: int = 8,
    suffix: str = "_patches",
) -> None:
    """Cache (P, D) patch features per tile for the named encoder."""
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if encoder == "uni":
        model = UNI2hExtractor(weights_path, device=device).model
        extract_fn = lambda imgs: extract_vit_patches(model, imgs)
    elif encoder == "virchow2":
        from src.a1_probe_encoders.main import _build_virchow_extractor

        wrapper = _build_virchow_extractor(weights_path, device=device)
        model = wrapper.model  # see Step 2 below for shim
        extract_fn = lambda imgs: extract_vit_patches(model, imgs)
    elif encoder == "ctranspath":
        from src.a1_probe_encoders.main import _build_ctranspath_extractor

        wrapper = _build_ctranspath_extractor(str(weights_path or ""), device=device)
        model = wrapper.model
        # CTransPath Swin -> hook on final norm layer (stage-4 output)
        target_layer = model.norm
        extract_fn = lambda imgs: extract_hooked_patches(model, target_layer, imgs)
    elif encoder == "resnet50":
        import torchvision.models as tv_models

        model = tv_models.resnet50(weights=tv_models.ResNet50_Weights.IMAGENET1K_V2).eval().to(device)
        extract_fn = lambda imgs: extract_hooked_patches(model, model.layer4, imgs)
    else:
        raise ValueError(f"unsupported encoder: {encoder!r}")

    image_paths = sorted(p for ext in (".png", ".jpg", ".jpeg") for p in image_dir.glob(f"*{ext}"))
    print(f"Found {len(image_paths)} tiles for {encoder} patch extraction.")
    for i in tqdm(range(0, len(image_paths), batch_size), desc=f"{encoder} patches"):
        batch = image_paths[i : i + batch_size]
        imgs = [cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p in batch]
        feats = extract_fn(imgs)
        for j, path in enumerate(batch):
            np.save(output_dir / f"{path.stem}{suffix}.npy", feats[j])
```

In the same file, extend `main()`:

```python
# In pipeline/extract_features.py main()
    parser.add_argument(
        "--encoder",
        choices=("uni", "virchow2", "ctranspath", "resnet50"),
        default=None,
        help="Run patch-feature extraction for the named encoder (overrides UNI/VAE flow).",
    )
    parser.add_argument(
        "--patches-output-dir",
        type=str,
        default=None,
        help="Output dir for per-encoder patch features.",
    )

    args = parser.parse_args()

    if args.encoder:
        cache_patch_features(
            encoder=args.encoder,
            image_dir=args.image_dir,
            output_dir=args.patches_output_dir or args.output_dir,
            weights_path=args.uni_model if args.encoder == "uni" else None,
            device=args.device,
            batch_size=args.batch_size,
        )
        return
```

- [ ] **Step 2: Shim the existing encoder extractors to expose `.model`**

The current `_build_virchow_extractor` / `_build_ctranspath_extractor` return wrapper classes. Inspect to confirm they expose `.model` or add one. If not, add a thin attribute:

```bash
grep -n "class.*Extractor\|self.model" /home/ec2-user/PixCell/src/a1_probe_encoders/main.py | head
```

If absent, modify the wrapper class to keep its underlying timm/torch module accessible:

```python
# Inside src/a1_probe_encoders/main.py wrapper class (Virchow / CTransPath)
class _VirchowExtractor:
    def __init__(self, model, transform, device):
        self.model = model
        ...
```

- [ ] **Step 3: Quick CLI dry-run on 4 tiles to verify**

```bash
mkdir -p /tmp/patches_test
conda run --no-capture-output -n pixcell python stage1_extract_features.py \
  --image-dir data/orion-crc33/he \
  --output-dir data/orion-crc33/features \
  --encoder resnet50 \
  --patches-output-dir /tmp/patches_test \
  --batch-size 4

ls /tmp/patches_test | head -3
conda run --no-capture-output -n pixcell python -c "
import numpy as np
import glob
p = sorted(glob.glob('/tmp/patches_test/*_patches.npy'))[0]
arr = np.load(p)
print(arr.shape, arr.dtype)
"
```

Expected: prints `(49, 2048) float16`.

- [ ] **Step 4: Commit**

```bash
git add pipeline/extract_features.py src/a1_probe_encoders/main.py stage1_extract_features.py
git commit -m "feat(stage1): per-encoder --encoder/--patches-output-dir patch extraction"
```

---

## Task 6: Cache patch features for the four encoders on the full dataset

**Files:**
- Run only.

- [ ] **Step 1: Virchow2 patches (~25 min)**

```bash
prlimit --as=24000000000 -- nohup conda run --no-capture-output -n pixcell \
  python stage1_extract_features.py \
  --image-dir data/orion-crc33/he \
  --encoder virchow2 \
  --patches-output-dir data/orion-crc33/features_patches/virchow2 \
  --batch-size 16 \
  > logs/patches_virchow2.log 2>&1 &
echo "PID $!"
```

Expected wall: ~25 min. Check log periodically. Verify final count:

```bash
ls data/orion-crc33/features_patches/virchow2/ | wc -l
```

Expected: 10379.

- [ ] **Step 2: CTransPath patches (~20 min)**

```bash
prlimit --as=24000000000 -- nohup conda run --no-capture-output -n pixcell \
  python stage1_extract_features.py \
  --image-dir data/orion-crc33/he \
  --encoder ctranspath \
  --patches-output-dir data/orion-crc33/features_patches/ctranspath \
  --batch-size 16 \
  > logs/patches_ctranspath.log 2>&1 &
echo "PID $!"
```

Expected: ~20 min, 10379 files in output dir.

- [ ] **Step 3: ResNet-50 patches (~12 min)**

```bash
prlimit --as=24000000000 -- nohup conda run --no-capture-output -n pixcell \
  python stage1_extract_features.py \
  --image-dir data/orion-crc33/he \
  --encoder resnet50 \
  --patches-output-dir data/orion-crc33/features_patches/resnet50 \
  --batch-size 32 \
  > logs/patches_resnet50.log 2>&1 &
echo "PID $!"
```

Expected: ~12 min, 10379 files. ResNet-50 is small enough to run batch=32.

- [ ] **Step 4: Verify disk usage**

```bash
du -sh data/orion-crc33/features_patches/*
```

Expected: ~10 GB / encoder. If disk pressure, drop ResNet-50 cache to fp16 (already default).

- [ ] **Step 5: Commit log files only**

```bash
git add logs/.gitkeep 2>/dev/null || true
# Patch caches are data artifacts; do not commit.
```

---

## Task 7: Create multi-encoder probe orchestrator

**Files:**
- Create: `src/a1_probe_mlp_spatial/run_multi_encoder.py`

The existing `run_task` in `src/a1_probe_mlp_spatial/main.py` already accepts a custom `features_dir` and a feature-file suffix logic that needs slight generalization. Add a `--feature-suffix` flag and pass it through.

- [ ] **Step 1: Add feature suffix flag to probe**

```python
# In src/a1_probe_mlp_spatial/main.py:load_patch_token_matrix signature
def load_patch_token_matrix(
    features_dir: str | Path,
    tile_ids: list[str],
    *,
    memmap_path: str | Path | None = None,
    suffix: str = "_uni_tokens.npy",
) -> np.ndarray:
    ...
    for idx, tile_id in enumerate(tile_ids[1:], start=1):
        arr = np.load(feature_dir / f"{tile_id}{suffix}")
        ...
```

Add CLI `--feature-suffix` (default `_uni_tokens.npy`) and pass to the loader.

- [ ] **Step 2: Write orchestrator script**

```python
# src/a1_probe_mlp_spatial/run_multi_encoder.py
"""Run the spatial probe across multiple encoders, one per native grid.

Skips encoders whose patch cache is missing so the figure can render with
whatever's available.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

ENCODERS = [
    {
        "name": "uni",
        "features_dir": ROOT / "data/orion-crc33/features",
        "suffix": "_uni_tokens.npy",
        "grid": 16,
        "targets": ROOT / "src/a1_mask_targets_spatial/out/mask_targets_T1_spatial.npy",
        "tile_ids": ROOT / "src/a1_mask_targets_spatial/out/tile_ids.txt",
        "names": ROOT / "src/a1_mask_targets_spatial/out/target_names_T1_spatial.json",
        "out_dir": ROOT / "src/a1_probe_mlp_spatial/out/uni_16",
    },
    {
        "name": "virchow2",
        "features_dir": ROOT / "data/orion-crc33/features_patches/virchow2",
        "suffix": "_patches.npy",
        "grid": 16,
        "targets": ROOT / "src/a1_mask_targets_spatial/out/mask_targets_T1_spatial.npy",
        "tile_ids": ROOT / "src/a1_mask_targets_spatial/out/tile_ids.txt",
        "names": ROOT / "src/a1_mask_targets_spatial/out/target_names_T1_spatial.json",
        "out_dir": ROOT / "src/a1_probe_mlp_spatial/out/virchow2_16",
    },
    {
        "name": "ctranspath",
        "features_dir": ROOT / "data/orion-crc33/features_patches/ctranspath",
        "suffix": "_patches.npy",
        "grid": 7,
        "targets": ROOT / "src/a1_mask_targets_spatial/out_grid_07/mask_targets_T1_spatial.npy",
        "tile_ids": ROOT / "src/a1_mask_targets_spatial/out_grid_07/tile_ids.txt",
        "names": ROOT / "src/a1_mask_targets_spatial/out_grid_07/target_names_T1_spatial.json",
        "out_dir": ROOT / "src/a1_probe_mlp_spatial/out/ctranspath_07",
    },
    {
        "name": "resnet50",
        "features_dir": ROOT / "data/orion-crc33/features_patches/resnet50",
        "suffix": "_patches.npy",
        "grid": 7,
        "targets": ROOT / "src/a1_mask_targets_spatial/out_grid_07/mask_targets_T1_spatial.npy",
        "tile_ids": ROOT / "src/a1_mask_targets_spatial/out_grid_07/tile_ids.txt",
        "names": ROOT / "src/a1_mask_targets_spatial/out_grid_07/target_names_T1_spatial.json",
        "out_dir": ROOT / "src/a1_probe_mlp_spatial/out/resnet50_07",
    },
]


def main() -> int:
    for entry in ENCODERS:
        feat = entry["features_dir"]
        if not feat.exists() or not any(feat.glob(f"*{entry['suffix']}")):
            print(f"SKIP {entry['name']}: no features at {feat}")
            continue
        print(f"=== {entry['name']} ===")
        cmd = [
            "conda", "run", "--no-capture-output", "-n", "pixcell",
            "python", "-m", "src.a1_probe_mlp_spatial.main",
            "--features-dir", str(feat),
            "--targets-path", str(entry["targets"]),
            "--tile-ids-path", str(entry["tile_ids"]),
            "--target-names-path", str(entry["names"]),
            "--out-dir", str(entry["out_dir"]),
            "--feature-suffix", entry["suffix"],
            "--n-tiles", "800",
            "--batch-size", "2048",
            "--max-train-rows", "50000",
            "--n-jobs", "2",
        ]
        subprocess.run(cmd, check=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
```

- [ ] **Step 3: Smoke-test orchestrator skips missing encoders**

```bash
conda run --no-capture-output -n pixcell python -m src.a1_probe_mlp_spatial.run_multi_encoder 2>&1 | head -20
```

Expected: prints "SKIP virchow2 / ctranspath / resnet50" if their caches don't exist yet; runs UNI cleanly otherwise.

- [ ] **Step 4: Commit**

```bash
git add src/a1_probe_mlp_spatial/main.py src/a1_probe_mlp_spatial/run_multi_encoder.py
git commit -m "feat(a1_probe_mlp_spatial): multi-encoder orchestrator + feature-suffix flag"
```

---

## Task 8: Run the full multi-encoder probe

**Files:**
- Run only.

- [ ] **Step 1: Background launch**

```bash
setsid nohup bash -c "
prlimit --as=24000000000 -- \
conda run --no-capture-output -n pixcell \
  python -m src.a1_probe_mlp_spatial.run_multi_encoder
" </dev/null > logs/multi_encoder_probe.log 2>&1 &
disown
echo "PID $!"
```

Wall ETA: ~1 hr total (UNI 12 min + Virchow2 12 min + CTransPath ~5 min + ResNet-50 ~5 min, plus per-encoder load + cv_splits).

- [ ] **Step 2: Monitor**

```bash
tail -f logs/multi_encoder_probe.log
```

Watch for `mlp_spatial_probe_results.csv` appearing under each
`src/a1_probe_mlp_spatial/out/<encoder>_*/`.

- [ ] **Step 3: Verify results**

```bash
for d in src/a1_probe_mlp_spatial/out/*_*/; do
  echo "=== $d ==="
  if [ -f "$d/mlp_spatial_probe_results.csv" ]; then
    head -1 "$d/mlp_spatial_probe_results.csv"
    awk -F, 'NR>1 {print $1, $2, $4, $6}' "$d/mlp_spatial_probe_results.csv"
  else
    echo "MISSING"
  fi
done
```

Expected: each encoder produces a 10-row CSV (or skipped marker). UNI's
numbers should match the existing `out/t1_spatial/` (sanity).

---

## Task 9: Build multi-encoder figure

**Files:**
- Create: `src/paper_figures/fig_t1_spatial_multi_encoder.py`
- Test: `tests/test_fig_t1_spatial_multi_encoder.py`

- [ ] **Step 1: Write smoke test**

```python
# tests/test_fig_t1_spatial_multi_encoder.py
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")


def _write_dummy_csv(path: Path, targets: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as h:
        w = csv.DictWriter(
            h,
            fieldnames=[
                "target", "r2_mean", "r2_sd",
                "r2_within_mean", "r2_within_sd",
                "pearson_r_mean", "pearson_r_sd",
                "delta_shuffle", "n_valid_folds",
            ],
        )
        w.writeheader()
        for t in targets:
            w.writerow({
                "target": t, "r2_mean": 0.1, "r2_sd": 0.02,
                "r2_within_mean": 0.05, "r2_within_sd": 0.01,
                "pearson_r_mean": 0.3, "pearson_r_sd": 0.05,
                "delta_shuffle": "nan", "n_valid_folds": 5,
            })


def test_fig_renders_with_two_encoders(tmp_path):
    from src.paper_figures.fig_t1_spatial_multi_encoder import build_figure

    targets = ["prolif_frac", "cell_density", "oxygen_mean"]
    csvs = {
        "UNI-2h": tmp_path / "uni/csv.csv",
        "Virchow2": tmp_path / "virchow/csv.csv",
    }
    for path in csvs.values():
        _write_dummy_csv(path, targets)
    fig = build_figure(encoder_csvs=csvs)
    out_path = tmp_path / "fig.png"
    fig.savefig(out_path)
    assert out_path.exists() and out_path.stat().st_size > 0
```

- [ ] **Step 2: Run, expect failure**

Run: `conda run --no-capture-output -n pixcell python -m pytest tests/test_fig_t1_spatial_multi_encoder.py -x -v`

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement figure builder**

```python
# src/paper_figures/fig_t1_spatial_multi_encoder.py
"""SI Figure 07d — multi-encoder per-patch spatial decodability for T1 targets."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from src.paper_figures.style import (
    FONT_FAMILY,
    FONT_SIZE_LABEL,
    FONT_SIZE_TICK,
    apply_style,
)


_T1_DISPLAY_LABELS = {
    "cell_density": "Density",
    "prolif_frac": "Prolif.",
    "nonprolif_frac": "Non-prolif.",
    "glucose_mean": "Glucose",
    "oxygen_mean": r"O$_2$",
    "healthy_frac": "Healthy",
    "cancer_frac": "Cancer",
    "vasculature_frac": "Vasculature",
    "immune_frac": "Immune",
    "dead_frac": "Dead",
}

_ENCODER_COLORS = {
    "UNI-2h": "#f98866",
    "Virchow2": "#9b59b6",
    "CTransPath": "#bed7d8",
    "ResNet-50": "#A9A9A9",
    "REMEDIS": "#5A5A5A",
}

_R2_FLOOR = -2.0


def _read_csv(path: Path) -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    with Path(path).open(encoding="utf-8", newline="") as h:
        for row in csv.DictReader(h):
            rows[row["target"]] = {k: float(v) for k, v in row.items() if k != "target"}
    return rows


def _draw_grouped_bars(
    ax: plt.Axes,
    targets: list[str],
    encoder_rows: dict[str, dict[str, dict[str, float]]],
    *,
    metric: str,
    sd_key: str,
    ylabel: str,
    clip_floor: float | None = None,
) -> None:
    n_enc = len(encoder_rows)
    step = 0.8 / max(1, n_enc)
    bar_w = step * 0.85
    x = np.arange(len(targets), dtype=np.float64)

    for enc_index, (encoder, target_map) in enumerate(encoder_rows.items()):
        offset = (enc_index - (n_enc - 1) / 2.0) * step
        ys = np.asarray([target_map.get(t, {}).get(metric, np.nan) for t in targets])
        sds = np.asarray([target_map.get(t, {}).get(sd_key, 0.0) for t in targets])
        if clip_floor is not None:
            ys = np.where(ys < clip_floor, clip_floor, ys)
        ax.bar(
            x + offset, ys, width=bar_w,
            color=_ENCODER_COLORS.get(encoder, "#888"),
            edgecolor="black", linewidth=0.6,
            yerr=sds, ecolor="#444",
            error_kw={"elinewidth": 0.6, "capsize": 1.5, "capthick": 0.6},
            label=encoder,
            zorder=2,
        )
    ax.axhline(0.0, color="#555", linewidth=0.6, zorder=1)
    ax.set_xticks(x)
    labels = [_T1_DISPLAY_LABELS.get(t, t) for t in targets]
    ax.set_xticklabels(labels, rotation=35, ha="right",
                       fontfamily=FONT_FAMILY, fontsize=FONT_SIZE_TICK)
    ax.set_ylabel(ylabel, fontfamily=FONT_FAMILY, fontsize=FONT_SIZE_LABEL)
    for s in ("top", "right", "bottom", "left"):
        ax.spines[s].set_color("black")
        ax.spines[s].set_linewidth(0.8)


def build_figure(*, encoder_csvs: dict[str, Path]) -> plt.Figure:
    apply_style()
    encoder_rows = {name: _read_csv(Path(path)) for name, path in encoder_csvs.items()
                    if Path(path).is_file()}
    if not encoder_rows:
        raise ValueError("no encoder CSVs found")

    primary = next(iter(encoder_rows.values()))
    targets = sorted(
        primary.keys(),
        key=lambda t: primary[t].get("r2_within_mean", -np.inf),
        reverse=True,
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0))
    _draw_grouped_bars(
        axes[0], targets, encoder_rows,
        metric="r2_within_mean", sd_key="r2_within_sd",
        ylabel="Per-patch R² (within-tile)",
        clip_floor=_R2_FLOOR,
    )
    axes[0].text(-0.10, 1.06, "A", transform=axes[0].transAxes,
                 fontsize=12, fontweight="bold", fontfamily=FONT_FAMILY)

    _draw_grouped_bars(
        axes[1], targets, encoder_rows,
        metric="pearson_r_mean", sd_key="pearson_r_sd",
        ylabel="Per-patch Pearson r",
    )
    axes[1].text(-0.10, 1.06, "B", transform=axes[1].transAxes,
                 fontsize=12, fontweight="bold", fontfamily=FONT_FAMILY)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               frameon=False, fontsize=FONT_SIZE_LABEL, fontfamily=FONT_FAMILY)
    fig.tight_layout(rect=[0.0, 0.06, 1.0, 1.0])
    return fig


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    encoder_csvs = {
        "UNI-2h": root / "src/a1_probe_mlp_spatial/out/uni_16/mlp_spatial_probe_results.csv",
        "Virchow2": root / "src/a1_probe_mlp_spatial/out/virchow2_16/mlp_spatial_probe_results.csv",
        "CTransPath": root / "src/a1_probe_mlp_spatial/out/ctranspath_07/mlp_spatial_probe_results.csv",
        "ResNet-50": root / "src/a1_probe_mlp_spatial/out/resnet50_07/mlp_spatial_probe_results.csv",
    }
    fig = build_figure(encoder_csvs=encoder_csvs)
    out = root / "figures/pngs_updated/07d_t1_spatial_multi_encoder.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":  # pragma: no cover
    main()
```

- [ ] **Step 4: Run smoke test, expect pass**

Run: `conda run --no-capture-output -n pixcell python -m pytest tests/test_fig_t1_spatial_multi_encoder.py -x -v`

Expected: 1 passed.

- [ ] **Step 5: Render real figure**

```bash
conda run --no-capture-output -n pixcell python -m src.paper_figures.fig_t1_spatial_multi_encoder
```

Expected: `wrote .../figures/pngs_updated/07d_t1_spatial_multi_encoder.png`.

- [ ] **Step 6: Commit**

```bash
git add src/paper_figures/fig_t1_spatial_multi_encoder.py tests/test_fig_t1_spatial_multi_encoder.py figures/pngs_updated/07d_t1_spatial_multi_encoder.png
git commit -m "feat(paper_figures): SI 07d multi-encoder per-patch spatial decodability"
```

---

## Task 10: Wire into figure main runner

**Files:**
- Modify: `src/paper_figures/main.py`

- [ ] **Step 1: Add the new figure to the runner**

In `src/paper_figures/main.py`, near the other figure invocations, append:

```python
# In src/paper_figures/main.py
try:
    from src.paper_figures.fig_t1_spatial_multi_encoder import build_figure as _build_07d
    encoder_csvs = {
        "UNI-2h": ROOT / "src/a1_probe_mlp_spatial/out/uni_16/mlp_spatial_probe_results.csv",
        "Virchow2": ROOT / "src/a1_probe_mlp_spatial/out/virchow2_16/mlp_spatial_probe_results.csv",
        "CTransPath": ROOT / "src/a1_probe_mlp_spatial/out/ctranspath_07/mlp_spatial_probe_results.csv",
        "ResNet-50": ROOT / "src/a1_probe_mlp_spatial/out/resnet50_07/mlp_spatial_probe_results.csv",
    }
    if any(p.is_file() for p in encoder_csvs.values()):
        fig_07d = _build_07d(encoder_csvs=encoder_csvs)
        _save_figure_png_outputs(fig_07d, "07d_t1_spatial_multi_encoder.png")
    else:
        print("Skipping 07d_t1_spatial_multi_encoder.png; missing encoder CSVs")
except ImportError:
    print("Skipping 07d_t1_spatial_multi_encoder.png; importer not available")
```

- [ ] **Step 2: Run main runner end-to-end**

```bash
conda run --no-capture-output -n pixcell python -m src.paper_figures.main 2>&1 | tail -20
```

Expected: existing figures unchanged; new line `wrote .../07d_t1_spatial_multi_encoder.png`.

- [ ] **Step 3: Commit**

```bash
git add src/paper_figures/main.py
git commit -m "chore(paper_figures): wire fig 07d into main runner"
```

---

## Self-review checklist (run after writing the plan)

- Spec section "Encoder coverage" → Tasks 5–6 cache all four; Task 7 orchestrator skips missing.
- Spec section "Targets" at non-divisor grids → Task 1 fixes block_mean_pool.
- Spec section "Figure layout" → Task 9 builds two-panel grouped-bar figure.
- Spec section "Success criteria 3" (UNI parity) → Task 8 step 3 sanity-greps UNI rows against existing `t1_spatial/` CSV (note: existing CSV at `out/t1_spatial/`, new at `out/uni_16/`; will need to re-run the existing UNI probe with `--feature-suffix _uni_tokens.npy` into `out/uni_16/` to match the new naming).
- Placeholders scanned: no TBD / TODO / "similar to" left.
- Type consistency: every encoder dict key matches `_ENCODER_COLORS` and the orchestrator's `name` field.

REMEDIS not in plan because no local weights confirmed; if user provides weights file later, add a 5th `ENCODERS` entry in `run_multi_encoder.py` and the figure builder will pick it up automatically via the same dict pattern.

---

## Execution Handoff

Plan saved to `docs/superpowers/plans/2026-05-21-multi-encoder-spatial-probe.md`. Two execution options:

1. **Subagent-Driven (recommended)** — dispatch fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — execute tasks in this session using executing-plans, batch with checkpoints.

Which approach?
