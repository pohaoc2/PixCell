# TME Evaluation Metrics: Sensitivity & TSC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two TME-conditional evaluation metrics — TME Channel Sensitivity (ΔLPIPS per group when zeroed) and TSC (Dice between detected H&E nuclei and CODEX cell masks) — to the SI A1/A2 ablation pipeline and figure.

**Architecture:** Phase 1 (Sensitivity) reuses `_fuse_active_groups()` from `tile_pipeline.py` to zero individual TME groups at inference time, measuring LPIPS delta against the existing production baseline tiles; results go into `cache.json["sensitivity"]` and a new bar-chart panel in the unified figure. Phase 2 (TSC) is CPU-only: applies scikit-image stain deconvolution to existing generated tiles, thresholds the hematoxylin channel to get a nuclei map, and computes Dice against the CODEX `cell_masks` channel; results merge into `cache.json["metrics"][variant]["tsc"]` and appear as a new column in the metrics table.

**Tech Stack:** PyTorch, lpips (already used in `tools/compute_ablation_metrics.py`), scikit-image (`rgb2hed`, `threshold_otsu`), numpy, PIL, matplotlib

---

## File Map

**Phase 1 — Sensitivity**
- Create: `tools/ablation_a1_a2/sensitivity_eval.py`
- Create: `tests/test_ablation_a1_a2_sensitivity_eval.py`
- Modify: `tools/ablation_a1_a2/cache_io.py` — add `merge_sensitivity`
- Modify: `src/paper_figures/fig_si_a1_a2_unified.py` — add Section 4 sensitivity bar chart

**Phase 2 — TSC**
- Create: `tools/ablation_a1_a2/tsc_eval.py`
- Create: `tests/test_ablation_a1_a2_tsc_eval.py`
- Modify: `tools/ablation_a1_a2/cache_io.py` — add `merge_tsc` (same file as Phase 1)
- Modify: `src/paper_figures/fig_si_a1_a2_unified.py` — add TSC column to table, update col_x

**These phases are independently deployable.** Phase 2 requires only existing generated tiles and CODEX data (no GPU).

---

## Key APIs (reference throughout)

```python
# tile_pipeline.py
prepare_tile_context(tile_id, *, models, config, uni_embeds, device, exp_channels_dir) -> dict
_fuse_active_groups(*, context: dict, active_groups: Sequence[str]) -> torch.Tensor
_render_fused_ablation_image(fused, *, context, scheduler, guidance_scale, device, seed, fixed_noise) -> np.ndarray  # uint8 [H,W,3]
_make_fixed_noise(*, config, scheduler, device, dtype, seed) -> torch.Tensor
load_channel(ch_dir, tile_id, resolution, binary, *, channel_name=None) -> np.ndarray  # float32 [H,W] in [0,1]
resolve_channel_dir(exp_channels_dir, channel_name) -> Path

# multi_group_tme.py
MultiGroupTMEModule.group_names: list[str]  # ['cell_types', 'cell_state', 'vasculature', 'microenv']

# cache_io.py
load_cache(path) -> dict
save_cache(cache, path) -> None
merge_metrics(cache, variant, metrics_dict) -> dict  # writes into cache["metrics"][variant]
```

---

## Phase 1 — TME Channel Sensitivity

### Task 1: `merge_sensitivity` in cache_io.py

**Files:**
- Modify: `tools/ablation_a1_a2/cache_io.py`
- Test: `tests/test_ablation_a1_a2_cache_io.py` (already exists — add to it)

- [ ] **Step 1: Write failing test**

Open `tests/test_ablation_a1_a2_cache_io.py` and add:

```python
from tools.ablation_a1_a2.cache_io import load_cache, merge_sensitivity, merge_tsc


def test_merge_sensitivity_creates_key():
    cache = load_cache(Path("/nonexistent"))  # returns empty cache
    scores = {
        "cell_types":  {"mean": 0.12, "std": 0.03, "per_tile": [0.10, 0.14]},
        "cell_state":  {"mean": 0.05, "std": 0.01, "per_tile": [0.04, 0.06]},
        "vasculature": {"mean": 0.08, "std": 0.02, "per_tile": [0.07, 0.09]},
        "microenv":    {"mean": 0.02, "std": 0.01, "per_tile": [0.01, 0.03]},
    }
    merge_sensitivity(cache, scores)
    assert cache["sensitivity"] == scores


def test_merge_sensitivity_overwrites():
    cache = {"sensitivity": {"cell_types": {"mean": 0.99}}}
    merge_sensitivity(cache, {"cell_types": {"mean": 0.01}})
    assert cache["sensitivity"]["cell_types"]["mean"] == 0.01
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/ec2-user/PixCell && python -m pytest tests/test_ablation_a1_a2_cache_io.py::test_merge_sensitivity_creates_key -xvs 2>&1 | tail -10
```
Expected: `ImportError: cannot import name 'merge_sensitivity'`

- [ ] **Step 3: Implement `merge_sensitivity` in cache_io.py**

Add after `merge_params`:

```python
def merge_sensitivity(cache: dict, group_scores: dict) -> dict:
    """Write or overwrite sensitivity scores for all groups."""
    cache["sensitivity"] = dict(group_scores)
    return cache


def merge_tsc(cache: dict, variant: str, tsc_score: float) -> dict:
    """Write TSC score into cache['metrics'][variant]['tsc']."""
    cache.setdefault("metrics", {}).setdefault(variant, {})["tsc"] = tsc_score
    return cache
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/ec2-user/PixCell && python -m pytest tests/test_ablation_a1_a2_cache_io.py -xvs 2>&1 | tail -15
```
Expected: all tests PASS

- [ ] **Step 5: Commit**

```bash
git add tools/ablation_a1_a2/cache_io.py tests/test_ablation_a1_a2_cache_io.py
git commit -m "feat(cache): add merge_sensitivity and merge_tsc helpers"
```

---

### Task 2: Pure metric helpers in sensitivity_eval.py

**Files:**
- Create: `tools/ablation_a1_a2/sensitivity_eval.py`
- Test: `tests/test_ablation_a1_a2_sensitivity_eval.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_ablation_a1_a2_sensitivity_eval.py`:

```python
import numpy as np
import pytest
from tools.ablation_a1_a2.sensitivity_eval import compute_sensitivity_scores


def _rand_rgb(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)


def test_compute_sensitivity_scores_keys():
    baseline = _rand_rgb(0)
    group_images = {
        "cell_types": _rand_rgb(1),
        "cell_state": _rand_rgb(2),
        "vasculature": _rand_rgb(3),
        "microenv": _rand_rgb(4),
    }
    scores = compute_sensitivity_scores(baseline, group_images)
    assert set(scores.keys()) == set(group_images.keys())
    for v in scores.values():
        assert isinstance(v, float)
        assert v >= 0.0


def test_compute_sensitivity_scores_identical_is_zero():
    img = _rand_rgb(0)
    group_images = {"cell_types": img.copy()}
    scores = compute_sensitivity_scores(img, group_images)
    # LPIPS of identical images should be very small (≤ 0.01)
    assert scores["cell_types"] < 0.01
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/ec2-user/PixCell && python -m pytest tests/test_ablation_a1_a2_sensitivity_eval.py -xvs 2>&1 | tail -10
```
Expected: `ModuleNotFoundError` or `ImportError`

- [ ] **Step 3: Implement compute_sensitivity_scores**

Create `tools/ablation_a1_a2/sensitivity_eval.py`:

```python
"""TME Channel Sensitivity evaluation.

For each TME group, measures LPIPS delta between the production baseline
and a generation with that group zeroed out (all other groups active).

Usage (CLI):
    python tools/ablation_a1_a2/sensitivity_eval.py \\
        --cache-dir inference_output/si_a1_a2 \\
        --tile-ids-file tools/ablation_a1_a2/qual_tile_ids.txt \\
        --device cuda
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# --- LPIPS helper -----------------------------------------------------------

def _lpips_fn():
    if not hasattr(_lpips_fn, "_model"):
        import lpips
        _lpips_fn._model = lpips.LPIPS(net="alex").eval()
    return _lpips_fn._model


def _to_lpips_tensor(rgb: np.ndarray) -> torch.Tensor:
    """uint8 [H,W,3] → float32 [1,3,H,W] in [-1, 1]."""
    t = torch.from_numpy(rgb).float() / 127.5 - 1.0
    return t.permute(2, 0, 1).unsqueeze(0)


def compute_lpips(img_a: np.ndarray, img_b: np.ndarray) -> float:
    """LPIPS between two uint8 RGB [H,W,3] images."""
    fn = _lpips_fn()
    with torch.no_grad():
        return float(fn(_to_lpips_tensor(img_a), _to_lpips_tensor(img_b)).item())


# --- Core metric ------------------------------------------------------------

def compute_sensitivity_scores(
    baseline: np.ndarray,
    group_images: dict[str, np.ndarray],
) -> dict[str, float]:
    """Return LPIPS delta for each group: higher = group has more impact.

    Args:
        baseline: uint8 [H,W,3] generated with all groups active.
        group_images: {group_name: uint8 [H,W,3]} generated with that group zeroed.
    """
    return {
        group: compute_lpips(baseline, perturbed)
        for group, perturbed in group_images.items()
    }
```

- [ ] **Step 4: Run tests**

```bash
cd /home/ec2-user/PixCell && python -m pytest tests/test_ablation_a1_a2_sensitivity_eval.py -xvs 2>&1 | tail -15
```
Expected: PASS (note: first run downloads lpips weights ~50 MB)

- [ ] **Step 5: Commit**

```bash
git add tools/ablation_a1_a2/sensitivity_eval.py tests/test_ablation_a1_a2_sensitivity_eval.py
git commit -m "feat(sensitivity): add compute_sensitivity_scores with LPIPS"
```

---

### Task 3: Per-tile group ablation inference

**Files:**
- Modify: `tools/ablation_a1_a2/sensitivity_eval.py`
- Test: `tests/test_ablation_a1_a2_sensitivity_eval.py`

- [ ] **Step 1: Write failing test**

Add to `tests/test_ablation_a1_a2_sensitivity_eval.py`:

```python
from unittest.mock import MagicMock, patch
from tools.ablation_a1_a2.sensitivity_eval import generate_group_ablations


def test_generate_group_ablations_returns_one_image_per_group():
    """generate_group_ablations calls _fuse_active_groups once per group."""
    dummy_rgb = np.zeros((256, 256, 3), dtype=np.uint8)

    mock_context = {
        "tme_module": MagicMock(group_names=["cell_types", "cell_state"]),
        "dtype": torch.float32,
    }
    mock_scheduler = MagicMock()
    mock_models = {}
    mock_config = MagicMock()

    with (
        patch("tools.ablation_a1_a2.sensitivity_eval.prepare_tile_context", return_value=mock_context),
        patch("tools.ablation_a1_a2.sensitivity_eval._fuse_active_groups", return_value=torch.zeros(1, 16, 32, 32)),
        patch("tools.ablation_a1_a2.sensitivity_eval._render_fused_ablation_image", return_value=dummy_rgb),
        patch("tools.ablation_a1_a2.sensitivity_eval._make_fixed_noise", return_value=torch.zeros(1, 16, 32, 32)),
    ):
        result = generate_group_ablations(
            tile_id="tile_0",
            models=mock_models,
            config=mock_config,
            scheduler=mock_scheduler,
            uni_embeds=torch.zeros(1, 1, 1024),
            device="cpu",
            exp_channels_dir=Path("/tmp"),
            guidance_scale=1.5,
            seed=42,
        )

    assert set(result.keys()) == {"cell_types", "cell_state"}
    for arr in result.values():
        assert arr.shape == (256, 256, 3)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/ec2-user/PixCell && python -m pytest tests/test_ablation_a1_a2_sensitivity_eval.py::test_generate_group_ablations_returns_one_image_per_group -xvs 2>&1 | tail -10
```
Expected: `ImportError: cannot import name 'generate_group_ablations'`

- [ ] **Step 3: Implement generate_group_ablations**

Add to `tools/ablation_a1_a2/sensitivity_eval.py` (after the imports, add the new imports; then the function):

```python
# Add to the imports block at the top:
from tools.stage3.tile_pipeline import (
    _fuse_active_groups,
    _make_fixed_noise,
    _render_fused_ablation_image,
    prepare_tile_context,
)


def generate_group_ablations(
    tile_id: str,
    *,
    models: dict,
    config,
    scheduler,
    uni_embeds: torch.Tensor,
    device: str,
    exp_channels_dir: Path,
    guidance_scale: float = 1.5,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Generate one image per group with that group zeroed; return {group: uint8 rgb}.

    For each group G, runs inference with active_groups = all_groups - {G}.
    Uses a fixed noise seed so differences are purely due to conditioning change.
    """
    context = prepare_tile_context(
        tile_id=tile_id,
        models=models,
        config=config,
        uni_embeds=uni_embeds,
        device=device,
        exp_channels_dir=exp_channels_dir,
    )
    group_names: list[str] = context["tme_module"].group_names
    fixed_noise = _make_fixed_noise(
        config=config,
        scheduler=scheduler,
        device=device,
        dtype=context["dtype"],
        seed=seed,
    )

    result: dict[str, np.ndarray] = {}
    for zeroed_group in group_names:
        active = [g for g in group_names if g != zeroed_group]
        fused = _fuse_active_groups(context=context, active_groups=active)
        result[zeroed_group] = _render_fused_ablation_image(
            fused,
            context=context,
            scheduler=scheduler,
            guidance_scale=guidance_scale,
            device=device,
            seed=seed,
            fixed_noise=fixed_noise.clone(),
        )
    return result
```

- [ ] **Step 4: Run tests**

```bash
cd /home/ec2-user/PixCell && python -m pytest tests/test_ablation_a1_a2_sensitivity_eval.py -xvs 2>&1 | tail -15
```
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add tools/ablation_a1_a2/sensitivity_eval.py tests/test_ablation_a1_a2_sensitivity_eval.py
git commit -m "feat(sensitivity): add generate_group_ablations using _fuse_active_groups"
```

---

### Task 4: `run_sensitivity` orchestrator + CLI

**Files:**
- Modify: `tools/ablation_a1_a2/sensitivity_eval.py`

- [ ] **Step 1: Write failing test**

Add to `tests/test_ablation_a1_a2_sensitivity_eval.py`:

```python
import json
import tempfile
from tools.ablation_a1_a2.sensitivity_eval import run_sensitivity


def test_run_sensitivity_writes_cache(tmp_path):
    cache_dir = tmp_path / "si_a1_a2"
    cache_dir.mkdir()
    tile_dir = cache_dir / "tiles" / "production"
    tile_dir.mkdir(parents=True)

    # Write fake baseline tile
    fake_rgb = np.zeros((256, 256, 3), dtype=np.uint8)
    Image.fromarray(fake_rgb).save(tile_dir / "tile_0.png")

    with (
        patch("tools.ablation_a1_a2.sensitivity_eval.load_all_models", return_value={}),
        patch("tools.ablation_a1_a2.sensitivity_eval.make_inference_scheduler", return_value=MagicMock()),
        patch("tools.ablation_a1_a2.sensitivity_eval.read_config", return_value=MagicMock()),
        patch("tools.ablation_a1_a2.sensitivity_eval.resolve_uni_embedding", return_value=torch.zeros(1, 1, 1024)),
        patch(
            "tools.ablation_a1_a2.sensitivity_eval.generate_group_ablations",
            return_value={"cell_types": fake_rgb, "cell_state": fake_rgb},
        ),
    ):
        run_sensitivity(
            cache_dir=cache_dir,
            tile_ids=["tile_0"],
            device="cpu",
            ckpt_dir=Path("checkpoints/fake"),
            config_path="configs/config_controlnet_exp.py",
            exp_channels_dir=Path("/tmp"),
            features_dir=Path("/tmp"),
        )

    cache = json.loads((cache_dir / "cache.json").read_text())
    assert "sensitivity" in cache
    assert "cell_types" in cache["sensitivity"]
    assert "mean" in cache["sensitivity"]["cell_types"]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/ec2-user/PixCell && python -m pytest tests/test_ablation_a1_a2_sensitivity_eval.py::test_run_sensitivity_writes_cache -xvs 2>&1 | tail -10
```
Expected: `ImportError: cannot import name 'run_sensitivity'`

- [ ] **Step 3: Implement run_sensitivity and CLI main**

Append to `tools/ablation_a1_a2/sensitivity_eval.py`:

```python
# Add to imports block at top:
from diffusion.utils.misc import read_config
from tools.ablation_a1_a2.cache_io import load_cache, merge_sensitivity, save_cache
from tools.stage3.common import make_inference_scheduler, resolve_uni_embedding
from tools.stage3.tile_pipeline import load_all_models


def run_sensitivity(
    *,
    cache_dir: Path,
    tile_ids: list[str],
    device: str,
    ckpt_dir: Path,
    config_path: str,
    exp_channels_dir: Path,
    features_dir: Path,
    guidance_scale: float = 1.5,
    seed: int = 42,
) -> dict:
    """Run group-ablation inference for all tiles; merge results into cache.json.

    Baseline images are loaded from cache_dir/tiles/production/{tile_id}.png
    (generated during the main ablation run). Each group is zeroed in turn.

    Returns the per-group sensitivity dict written to cache.
    """
    config = read_config(config_path)
    config._filename = config_path
    models = load_all_models(config, config_path, str(ckpt_dir), device)
    scheduler = make_inference_scheduler(num_steps=30, device=device)

    baseline_dir = cache_dir / "tiles" / "production"
    per_tile: dict[str, list[float]] = {}

    for tile_id in tile_ids:
        baseline_path = baseline_dir / f"{tile_id}.png"
        if not baseline_path.exists():
            print(f"  [sensitivity] baseline tile missing: {baseline_path} — skip")
            continue
        baseline_rgb = np.asarray(Image.open(baseline_path).convert("RGB"))
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
        scores = compute_sensitivity_scores(baseline_rgb, group_images)
        for group, score in scores.items():
            per_tile.setdefault(group, []).append(score)
        print(f"  [sensitivity] {tile_id}: {scores}")

    group_scores: dict[str, dict] = {}
    for group, values in per_tile.items():
        arr = np.array(values)
        group_scores[group] = {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "per_tile": [float(v) for v in values],
        }

    cache_path = cache_dir / "cache.json"
    cache = load_cache(cache_path)
    merge_sensitivity(cache, group_scores)
    save_cache(cache, cache_path)
    print(f"Sensitivity saved -> {cache_path}")
    return group_scores


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default="inference_output/si_a1_a2")
    parser.add_argument("--tile-ids-file", required=True)
    parser.add_argument("--ckpt-dir", default="checkpoints/pixcell_controlnet_exp/npy_inputs")
    parser.add_argument("--config-path", default="configs/config_controlnet_exp.py")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    cache_dir = Path(args.cache_dir)
    tile_ids = [
        line.strip()
        for line in Path(args.tile_ids_file).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    from tools.stage3.tile_pipeline import resolve_data_layout
    exp_channels_dir, features_dir, _ = resolve_data_layout(Path("data/orion-crc33"))

    run_sensitivity(
        cache_dir=cache_dir,
        tile_ids=tile_ids,
        device=args.device,
        ckpt_dir=Path(args.ckpt_dir),
        config_path=args.config_path,
        exp_channels_dir=exp_channels_dir,
        features_dir=features_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run all sensitivity tests**

```bash
cd /home/ec2-user/PixCell && python -m pytest tests/test_ablation_a1_a2_sensitivity_eval.py -xvs 2>&1 | tail -20
```
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add tools/ablation_a1_a2/sensitivity_eval.py tests/test_ablation_a1_a2_sensitivity_eval.py
git commit -m "feat(sensitivity): add run_sensitivity orchestrator and CLI"
```

---

### Task 5: Sensitivity bar chart panel in unified figure

**Files:**
- Modify: `src/paper_figures/fig_si_a1_a2_unified.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_fig_si_a1_a2_unified.py` (or append if exists):

```python
import json
import tempfile
from pathlib import Path
import matplotlib.pyplot as plt
from src.paper_figures.fig_si_a1_a2_unified import build_section4_figure


def _fake_cache(tmp_path: Path) -> Path:
    cache = {
        "version": 1, "generated": "2026-04-27",
        "tile_ids": [], "training_curves": {}, "metrics": {}, "params": {},
        "sensitivity": {
            "cell_types":  {"mean": 0.12, "std": 0.03, "per_tile": [0.10, 0.14]},
            "cell_state":  {"mean": 0.05, "std": 0.01, "per_tile": [0.04, 0.06]},
            "vasculature": {"mean": 0.08, "std": 0.02, "per_tile": [0.07, 0.09]},
            "microenv":    {"mean": 0.02, "std": 0.01, "per_tile": [0.01, 0.03]},
        },
    }
    p = tmp_path / "cache.json"
    p.write_text(json.dumps(cache))
    return p


def test_build_section4_figure_returns_figure(tmp_path):
    cache_path = _fake_cache(tmp_path)
    fig = build_section4_figure(cache_path=cache_path)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_build_section4_figure_missing_sensitivity(tmp_path):
    cache = {"version": 1, "generated": "", "tile_ids": [], "training_curves": {},
             "metrics": {}, "params": {}}
    p = tmp_path / "cache.json"
    p.write_text(json.dumps(cache))
    fig = build_section4_figure(cache_path=p)
    assert isinstance(fig, plt.Figure)  # gracefully handles missing key
    plt.close(fig)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/ec2-user/PixCell && python -m pytest tests/test_fig_si_a1_a2_unified.py::test_build_section4_figure_returns_figure -xvs 2>&1 | tail -10
```
Expected: `ImportError: cannot import name 'build_section4_figure'`

- [ ] **Step 3: Implement `_draw_section4_sensitivity` and `build_section4_figure`**

In `src/paper_figures/fig_si_a1_a2_unified.py`, add after `_draw_section3_tiles`:

```python
_GROUP_DISPLAY_NAMES = {
    "cell_types": "Cell types",
    "cell_state": "Cell state",
    "vasculature": "Vasculature",
    "microenv": "Microenv.",
}
_SENSITIVITY_COLOR = "#2e7d32"  # production model green


def _draw_section4_sensitivity(ax: plt.Axes, cache: dict) -> None:
    sensitivity = cache.get("sensitivity", {})
    ax.axis("on")
    ax.set_xlabel("Sensitivity (ΔLPIPS↑)", fontsize=FONT_SIZE_LABEL)
    ax.set_title("TME channel sensitivity", fontsize=FONT_SIZE_LABEL, loc="left", pad=3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=FONT_SIZE_TICK, width=0.7, length=3)

    if not sensitivity:
        ax.text(0.5, 0.5, "No sensitivity data", ha="center", va="center",
                fontsize=FONT_SIZE_ANNOTATION, transform=ax.transAxes)
        return

    group_order = ["cell_types", "cell_state", "vasculature", "microenv"]
    present = [g for g in group_order if g in sensitivity]
    labels = [_GROUP_DISPLAY_NAMES.get(g, g) for g in present]
    means = [sensitivity[g]["mean"] for g in present]
    stds = [sensitivity[g]["std"] for g in present]

    y_pos = range(len(present))
    ax.barh(list(y_pos), means, xerr=stds, color=_SENSITIVITY_COLOR,
            alpha=0.80, height=0.55, error_kw={"linewidth": 0.8, "capsize": 3})
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels, fontsize=FONT_SIZE_TICK)
    ax.set_xlim(left=0)
    ax.grid(axis="x", color="#d9d9d9", linewidth=0.45)
    ax.invert_yaxis()
```

Also add `build_section4_figure` alongside the other section-build functions:

```python
def build_section4_figure(*, cache_path: Path) -> plt.Figure:
    apply_style()
    cache = _load_cache(cache_path)
    fig = plt.figure(figsize=(4.5, 2.2), constrained_layout=False)
    fig.subplots_adjust(left=0.22, right=0.97, top=0.88, bottom=0.22)
    _draw_section4_sensitivity(fig.add_subplot(111), cache)
    return fig
```

Update `build_section_figures` to include section4:

```python
def build_section_figures(*, cache_path: Path, tile_dir: Path) -> dict[str, plt.Figure]:
    return {
        "section1_curves": build_section1_figure(cache_path=cache_path),
        "section2_metrics": build_section2_figure(cache_path=cache_path),
        "section3_tiles": build_section3_figure(cache_path=cache_path, tile_dir=tile_dir),
        "section4_sensitivity": build_section4_figure(cache_path=cache_path),
    }
```

Update `_split_output_paths` to include section4:

```python
def _split_output_paths(out: Path) -> dict[str, Path]:
    stem = out.stem
    prefix = stem.replace("_unified", "")
    return {
        "section1_curves": out.with_name(f"{prefix}_section1_curves{out.suffix}"),
        "section2_metrics": out.with_name(f"{prefix}_section2_metrics{out.suffix}"),
        "section3_tiles": out.with_name(f"{prefix}_section3_tiles{out.suffix}"),
        "section4_sensitivity": out.with_name(f"{prefix}_section4_sensitivity{out.suffix}"),
    }
```

Update `build_figure` to include Section 4:

```python
def build_figure(*, cache_path: Path, tile_dir: Path) -> plt.Figure:
    apply_style()
    cache = _load_cache(cache_path)
    fig = plt.figure(figsize=(12.0, 12.2), constrained_layout=False)
    fig.subplots_adjust(left=0.11, right=0.985, top=0.985, bottom=0.045, hspace=0.25, wspace=0.18)

    outer = fig.add_gridspec(4, 1, height_ratios=[1.7, 0.85, 3.0, 0.75], hspace=0.22)
    _draw_section1_curves(fig, outer[0], cache)
    _draw_section2_table(fig.add_subplot(outer[1]), cache)
    _draw_section3_tiles(fig, outer[2], cache, tile_dir)
    _draw_section4_sensitivity(fig.add_subplot(outer[3]), cache)

    return fig
```

- [ ] **Step 4: Run tests**

```bash
cd /home/ec2-user/PixCell && python -m pytest tests/test_fig_si_a1_a2_unified.py -xvs 2>&1 | tail -15
```
Expected: all PASS

- [ ] **Step 5: Regenerate unified figure to verify layout**

```bash
cd /home/ec2-user/PixCell && python -m src.paper_figures.fig_si_a1_a2_unified \
    --cache-dir inference_output/si_a1_a2 \
    --out figures/pngs/SI_A1_A2_unified.png --dpi 150
```
Expected: exits 0, `figures/pngs/SI_A1_A2_unified.png` updated (Section 4 shows "No sensitivity data" until `run_sensitivity` is executed).

- [ ] **Step 6: Commit**

```bash
git add src/paper_figures/fig_si_a1_a2_unified.py tests/test_fig_si_a1_a2_unified.py
git commit -m "feat(figure): add Section 4 sensitivity bar chart panel"
```

---

## Phase 2 — TSC (TME Spatial Concordance)

### Task 6: Nuclei extraction and Dice in tsc_eval.py

**Files:**
- Create: `tools/ablation_a1_a2/tsc_eval.py`
- Create: `tests/test_ablation_a1_a2_tsc_eval.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_ablation_a1_a2_tsc_eval.py`:

```python
import numpy as np
import pytest
from tools.ablation_a1_a2.tsc_eval import dice_score, extract_nuclei_map, compute_tsc_tile


# --- dice_score ---

def test_dice_score_perfect():
    a = np.array([[1, 0], [0, 1]], dtype=bool)
    assert dice_score(a, a) == pytest.approx(1.0)


def test_dice_score_no_overlap():
    a = np.array([[1, 0], [0, 0]], dtype=bool)
    b = np.array([[0, 0], [0, 1]], dtype=bool)
    assert dice_score(a, b) == pytest.approx(0.0)


def test_dice_score_both_empty():
    a = np.zeros((4, 4), dtype=bool)
    # Both empty → 1.0 by convention (no cells to detect = no disagreement)
    assert dice_score(a, a) == pytest.approx(1.0)


# --- extract_nuclei_map ---

def test_extract_nuclei_map_shape():
    rng = np.random.default_rng(0)
    he_rgb = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
    nmap = extract_nuclei_map(he_rgb)
    assert nmap.shape == (256, 256)
    assert nmap.dtype == bool


def test_extract_nuclei_map_all_dark_image():
    """Dark purple image (all nuclei) should produce mostly-True map."""
    dark_purple = np.full((64, 64, 3), [80, 40, 100], dtype=np.uint8)
    nmap = extract_nuclei_map(dark_purple)
    assert nmap.mean() > 0.3  # at least 30% classified as nuclei


# --- compute_tsc_tile ---

def test_compute_tsc_tile_range():
    rng = np.random.default_rng(1)
    he_rgb = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
    cell_mask = rng.integers(0, 2, (256, 256), dtype=np.uint8).astype(bool)
    score = compute_tsc_tile(he_rgb, cell_mask)
    assert 0.0 <= score <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/ec2-user/PixCell && python -m pytest tests/test_ablation_a1_a2_tsc_eval.py -xvs 2>&1 | tail -10
```
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement core functions**

Create `tools/ablation_a1_a2/tsc_eval.py`:

```python
"""TSC (TME Spatial Concordance): Dice between H&E nuclei map and CODEX cell_masks.

Applies stain deconvolution (rgb2hed) to the generated H&E to extract the
hematoxylin channel, thresholds it via Otsu, and computes Dice overlap with
the binary CODEX cell_masks channel.

Usage (CLI):
    python tools/ablation_a1_a2/tsc_eval.py \\
        --cache-dir inference_output/si_a1_a2 \\
        --tile-ids-file tools/ablation_a1_a2/qual_tile_ids.txt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.color import rgb2hed
from skimage.filters import threshold_otsu

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """Dice coefficient for two boolean arrays. Both-empty case returns 1.0."""
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    if not pred.any() and not gt.any():
        return 1.0
    intersection = float((pred & gt).sum())
    return 2.0 * intersection / (float(pred.sum()) + float(gt.sum()) + 1e-8)


def extract_nuclei_map(he_rgb: np.ndarray) -> np.ndarray:
    """Return binary nuclei map from uint8 [H,W,3] H&E image.

    Uses stain deconvolution (rgb2hed) on the hematoxylin channel
    followed by Otsu thresholding. High hematoxylin = nuclei present.
    """
    hed = rgb2hed(he_rgb.astype(np.float32) / 255.0)
    h_channel = hed[:, :, 0]
    thresh = threshold_otsu(h_channel)
    return h_channel > thresh


def compute_tsc_tile(he_rgb: np.ndarray, codex_cell_mask: np.ndarray) -> float:
    """Dice between detected H&E nuclei and CODEX binary cell mask."""
    nuclei = extract_nuclei_map(he_rgb)
    gt = codex_cell_mask.astype(bool)
    return dice_score(nuclei, gt)
```

- [ ] **Step 4: Run tests**

```bash
cd /home/ec2-user/PixCell && python -m pytest tests/test_ablation_a1_a2_tsc_eval.py -xvs 2>&1 | tail -20
```
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add tools/ablation_a1_a2/tsc_eval.py tests/test_ablation_a1_a2_tsc_eval.py
git commit -m "feat(tsc): add extract_nuclei_map, dice_score, compute_tsc_tile"
```

---

### Task 7: `run_tsc` orchestrator + CLI

**Files:**
- Modify: `tools/ablation_a1_a2/tsc_eval.py`
- Test: `tests/test_ablation_a1_a2_tsc_eval.py`

- [ ] **Step 1: Write failing test**

Add to `tests/test_ablation_a1_a2_tsc_eval.py`:

```python
import json
import tempfile
from unittest.mock import patch
from tools.ablation_a1_a2.tsc_eval import run_tsc


def test_run_tsc_writes_cache(tmp_path):
    cache_dir = tmp_path / "si_a1_a2"
    cache_dir.mkdir()

    # Create fake tile PNG for variant "production"
    fake_rgb = np.full((256, 256, 3), 80, dtype=np.uint8)  # dark = nuclei
    tile_dir = cache_dir / "tiles" / "production"
    tile_dir.mkdir(parents=True)
    Image.fromarray(fake_rgb).save(tile_dir / "tile_0.png")

    # Fake cell_mask returns half-ones
    fake_mask = np.ones((256, 256), dtype=bool)

    with patch("tools.ablation_a1_a2.tsc_eval._load_codex_cell_mask", return_value=fake_mask):
        run_tsc(
            cache_dir=cache_dir,
            tile_ids=["tile_0"],
            variants=["production"],
            exp_channels_dir=Path("/tmp"),
            image_size=256,
        )

    cache = json.loads((cache_dir / "cache.json").read_text())
    assert "tsc" in cache["metrics"]["production"]
    assert 0.0 <= cache["metrics"]["production"]["tsc"] <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/ec2-user/PixCell && python -m pytest tests/test_ablation_a1_a2_tsc_eval.py::test_run_tsc_writes_cache -xvs 2>&1 | tail -10
```
Expected: `ImportError: cannot import name 'run_tsc'`

- [ ] **Step 3: Implement `_load_codex_cell_mask`, `run_tsc`, and CLI main**

Append to `tools/ablation_a1_a2/tsc_eval.py`:

```python
# Additional import at top:
from tools.ablation_a1_a2.cache_io import load_cache, merge_tsc, save_cache
from tools.stage3.tile_pipeline import load_channel, resolve_channel_dir


def _load_codex_cell_mask(tile_id: str, exp_channels_dir: Path, image_size: int) -> np.ndarray:
    """Load CODEX cell_masks channel as boolean [H, W] array."""
    ch_dir = resolve_channel_dir(exp_channels_dir, "cell_masks")
    arr = load_channel(ch_dir, tile_id, image_size, binary=True, channel_name="cell_masks")
    return arr > 0.5


def run_tsc(
    *,
    cache_dir: Path,
    tile_ids: list[str],
    variants: list[str],
    exp_channels_dir: Path,
    image_size: int = 256,
) -> dict[str, float]:
    """Compute mean TSC for each variant and merge into cache.json.

    Loads generated tiles from cache_dir/tiles/{variant}/{tile_id}.png.
    Compares nuclei map against CODEX cell_masks.
    Returns {variant: mean_tsc}.
    """
    cache_path = cache_dir / "cache.json"
    cache = load_cache(cache_path)
    variant_scores: dict[str, float] = {}

    for variant in variants:
        tile_dir = cache_dir / "tiles" / variant
        scores: list[float] = []
        for tile_id in tile_ids:
            tile_path = tile_dir / f"{tile_id}.png"
            if not tile_path.exists():
                print(f"  [tsc] tile missing: {tile_path} — skip")
                continue
            he_rgb = np.asarray(Image.open(tile_path).convert("RGB"))
            cell_mask = _load_codex_cell_mask(tile_id, exp_channels_dir, image_size)
            scores.append(compute_tsc_tile(he_rgb, cell_mask))
        if scores:
            mean_tsc = float(np.mean(scores))
            merge_tsc(cache, variant, mean_tsc)
            variant_scores[variant] = mean_tsc
            print(f"  [tsc] {variant}: mean TSC = {mean_tsc:.4f} over {len(scores)} tiles")

    save_cache(cache, cache_path)
    print(f"TSC saved -> {cache_path}")
    return variant_scores


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default="inference_output/si_a1_a2")
    parser.add_argument("--tile-ids-file", required=True)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["production", "a1_concat", "a1_per_channel", "a2_bypass"],
    )
    parser.add_argument("--image-size", type=int, default=256)
    args = parser.parse_args(argv)

    cache_dir = Path(args.cache_dir)
    tile_ids = [
        line.strip()
        for line in Path(args.tile_ids_file).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    from tools.stage3.tile_pipeline import resolve_data_layout
    exp_channels_dir, _, _ = resolve_data_layout(Path("data/orion-crc33"))

    run_tsc(
        cache_dir=cache_dir,
        tile_ids=tile_ids,
        variants=args.variants,
        exp_channels_dir=exp_channels_dir,
        image_size=args.image_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run all TSC tests**

```bash
cd /home/ec2-user/PixCell && python -m pytest tests/test_ablation_a1_a2_tsc_eval.py -xvs 2>&1 | tail -20
```
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add tools/ablation_a1_a2/tsc_eval.py tests/test_ablation_a1_a2_tsc_eval.py
git commit -m "feat(tsc): add run_tsc orchestrator and CLI"
```

---

### Task 8: Add TSC column to metrics table

**Files:**
- Modify: `src/paper_figures/fig_si_a1_a2_unified.py`

- [ ] **Step 1: Write failing test**

Add to `tests/test_fig_si_a1_a2_unified.py`:

```python
from src.paper_figures.fig_si_a1_a2_unified import build_section2_figure


def test_section2_figure_with_tsc(tmp_path):
    cache = {
        "version": 1, "generated": "", "tile_ids": [], "training_curves": {},
        "metrics": {
            "production":     {"fud": 187.86, "dice": 0.791, "pq": 0.569, "lpips": 0.385, "style_hed": 0.041, "tsc": 0.72},
            "a1_concat":      {"fud": 187.84, "dice": 0.883, "pq": 0.782, "lpips": 0.344, "style_hed": 0.034, "tsc": 0.68},
            "a1_per_channel": {"fud": 289.07, "dice": 0.054, "pq": 0.011, "lpips": 0.501, "style_hed": 0.143, "tsc": 0.21},
            "a2_bypass":      {"fud": 184.42, "dice": 0.897, "pq": 0.786, "lpips": 0.369, "style_hed": 0.043, "tsc": 0.41},
        },
        "params": {},
    }
    p = tmp_path / "cache.json"
    p.write_text(json.dumps(cache))
    fig = build_section2_figure(cache_path=p)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
```

- [ ] **Step 2: Run test (should pass — rendering doesn't fail on extra metric key)**

```bash
cd /home/ec2-user/PixCell && python -m pytest tests/test_fig_si_a1_a2_unified.py::test_section2_figure_with_tsc -xvs 2>&1 | tail -10
```
This test should already pass (TSC just shows "pending" since key not in METRIC_COLS yet). This is the baseline.

- [ ] **Step 3: Add TSC to METRIC_COLS and expand col_x**

In `src/paper_figures/fig_si_a1_a2_unified.py`:

```python
# before
METRIC_COLS = (
    ("FUD↓", "fud", "{:.2f}"),
    ("DICE↑", "dice", "{:.3f}"),
    ("PQ↑", "pq", "{:.3f}"),
    ("LPIPS↓", "lpips", "{:.3f}"),
    ("HED↓", "style_hed", "{:.3f}"),
)
# after
METRIC_COLS = (
    ("FUD↓", "fud", "{:.2f}"),
    ("DICE↑", "dice", "{:.3f}"),
    ("PQ↑", "pq", "{:.3f}"),
    ("LPIPS↓", "lpips", "{:.3f}"),
    ("HED↓", "style_hed", "{:.3f}"),
    ("TSC↑", "tsc", "{:.3f}"),
)
METRIC_DIRECTIONS = {
    "fud": "min",
    "dice": "max",
    "pq": "max",
    "lpips": "min",
    "style_hed": "min",
    "tsc": "max",  # add this line
}
```

Update `col_x` to fit 8 columns (Config + 6 metrics + Params). Also widen section2 standalone slightly:

```python
# In _draw_section2_table, replace col_x:
# before: col_x = [0.01, 0.27, 0.39, 0.51, 0.63, 0.75, 0.88]
# after (8 columns):
col_x = [0.01, 0.22, 0.32, 0.42, 0.52, 0.62, 0.72, 0.87]
```

Update `build_section2_figure` to widen the standalone figure:
```python
# before: figsize=(5.25, 1.28)
# after:
fig = plt.figure(figsize=(6.2, 1.28), constrained_layout=False)
```

- [ ] **Step 4: Run all figure tests**

```bash
cd /home/ec2-user/PixCell && python -m pytest tests/test_fig_si_a1_a2_unified.py -xvs 2>&1 | tail -15
```
Expected: all PASS

- [ ] **Step 5: Regenerate and verify no layout overflow**

```bash
cd /home/ec2-user/PixCell && python -m src.paper_figures.fig_si_a1_a2_unified \
    --cache-dir inference_output/si_a1_a2 \
    --out figures/pngs/SI_A1_A2_unified.png --dpi 150
```
Inspect `figures/pngs/SI_A1_A2_section2_metrics.png` — TSC column should show "pending" (no data yet) without layout overflow.

- [ ] **Step 6: Commit**

```bash
git add src/paper_figures/fig_si_a1_a2_unified.py tests/test_fig_si_a1_a2_unified.py
git commit -m "feat(figure): add TSC column to metrics table"
```

---

## Running order for real data

```bash
# 1. Compute sensitivity (needs GPU + production checkpoint)
python tools/ablation_a1_a2/sensitivity_eval.py \
    --cache-dir inference_output/si_a1_a2 \
    --tile-ids-file tools/ablation_a1_a2/qual_tile_ids.txt \
    --device cuda

# 2. Compute TSC (CPU-only, runs on existing tiles)
python tools/ablation_a1_a2/tsc_eval.py \
    --cache-dir inference_output/si_a1_a2 \
    --tile-ids-file tools/ablation_a1_a2/qual_tile_ids.txt

# 3. Regenerate figure with both metrics
python -m src.paper_figures.fig_si_a1_a2_unified \
    --cache-dir inference_output/si_a1_a2 \
    --out figures/pngs/SI_A1_A2_unified.png --dpi 300
```
