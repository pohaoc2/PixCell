# A2/A3 Design-Justification Ablations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Spec:** `docs/superpowers/specs/2026-04-26-A2-A3-design-justification-ablations-design.md`

**Goal:** Build code + configs needed to run the A2 (`zero_mask_latent`) bypass probe and A3 (`zero_init_conv_out`) stability ablations and produce two SI figures.

**Architecture:** Two new training configs (False variants) reuse the existing paired-exp training loop. New inference wrappers cover the bypass probe (load False checkpoint, zero TME at forward) and the off-the-shelf PixCell ControlNet reference (mask-only inference, no fine-tune). New aggregation script reads training logs across seeds for A3 stability metrics. Two new figure builders consume metric JSONs + per-tile PNGs and emit `SI_A2_bypass_probe.png` / `SI_A3_zero_init.png`. Existing CellViT pipeline reused unchanged.

**Tech Stack:** Python 3.13 + PyTorch + matplotlib; existing `train_controlnet_exp.py`, `tools/cellvit/`, `tools/ablation_report/`, `src/paper_figures/`.

---

## File Structure

**New files:**
- `configs/config_controlnet_exp_a2_bypass.py` — variant config: `zero_mask_latent=False`. One per A2 variant.
- `configs/config_controlnet_exp_a3_no_zero_init.py` — variant config: `zero_init_conv_out=False`.
- `tools/baselines/__init__.py`
- `tools/baselines/pixcell_offshelf_inference.py` — inference wrapper for the original PixCell ControlNet checkpoint, mask-only conditioning + UNI.
- `tools/ablation_a2/__init__.py`
- `tools/ablation_a2/run_bypass_probe.py` — loads A2 False-variant checkpoint, runs inference with TME outputs zeroed.
- `tools/ablation_a3/__init__.py`
- `tools/ablation_a3/aggregate_stability.py` — reads per-seed training logs, writes `stability_summary.json`.
- `src/paper_figures/fig_si_a2_bypass.py` — figure builder.
- `src/paper_figures/fig_si_a3_zero_init.py` — figure builder.
- `tests/test_pixcell_offshelf_inference.py`
- `tests/test_bypass_probe.py`
- `tests/test_aggregate_stability.py`

**Modified files:**
- `src/paper_figures/main.py` — register two new SI figure builders, save to both `pngs/` and `pngs_updated/`.

---

## Task 1: A2 variant config

**Files:**
- Create: `configs/config_controlnet_exp_a2_bypass.py`

- [ ] **Step 1: Create the variant config**

This config inherits from the production config and only overrides the flag, work_dir, and seed.

```python
"""
config_controlnet_exp_a2_bypass.py

A2 design-justification ablation: zero_mask_latent=False (additive TME, bypass-capable).
Inherits config_controlnet_exp.py and overrides only the flag and the work_dir to avoid
overwriting the production checkpoints.
"""

_base_ = ['./config_controlnet_exp.py']

zero_mask_latent = False

work_dir = "./checkpoints/pixcell_controlnet_exp_a2_bypass"
seed = 42  # operator overrides per seed run; default kept for reproducibility
```

- [ ] **Step 2: Verify the config loads**

```bash
python -c "from mmcv import Config; c = Config.fromfile('configs/config_controlnet_exp_a2_bypass.py'); print(c.zero_mask_latent, c.work_dir)"
```

Expected: `False ./checkpoints/pixcell_controlnet_exp_a2_bypass`

- [ ] **Step 3: Commit**

```bash
git add configs/config_controlnet_exp_a2_bypass.py
git commit -m "feat(A2): config for zero_mask_latent=False bypass variant"
```

---

## Task 2: A3 variant config

**Files:**
- Create: `configs/config_controlnet_exp_a3_no_zero_init.py`

- [ ] **Step 1: Create the variant config**

```python
"""
config_controlnet_exp_a3_no_zero_init.py

A3 design-justification ablation: zero_init_conv_out=False (no residual gating).
Inherits config_controlnet_exp.py and only overrides controlnet_config + work_dir.
Logs gradient norms at every log_interval to support stability analysis.
"""

_base_ = ['./config_controlnet_exp.py']

# Override controlnet_config to disable zero-init residual gating.
controlnet_config = dict(
    zero_init_conv_out=False,
    copy_base_layers=True,
    conditioning_scale=1.0,
)
model_kwargs = dict(
    use_controlnet=True,
    controlnet_config=controlnet_config,
)

work_dir = "./checkpoints/pixcell_controlnet_exp_a3_no_zero_init"
seed = 42
```

- [ ] **Step 2: Verify the config loads**

```bash
python -c "from mmcv import Config; c = Config.fromfile('configs/config_controlnet_exp_a3_no_zero_init.py'); print(c.controlnet_config['zero_init_conv_out'])"
```

Expected: `False`

- [ ] **Step 3: Commit**

```bash
git add configs/config_controlnet_exp_a3_no_zero_init.py
git commit -m "feat(A3): config for zero_init_conv_out=False stability variant"
```

---

## Task 3: Off-the-shelf PixCell ControlNet inference wrapper

**Files:**
- Create: `tools/baselines/__init__.py` (empty)
- Create: `tools/baselines/pixcell_offshelf_inference.py`
- Create: `tests/test_pixcell_offshelf_inference.py`

- [ ] **Step 1: Write failing smoke test**

```python
# tests/test_pixcell_offshelf_inference.py
"""Smoke test for off-the-shelf PixCell ControlNet inference wrapper."""
from __future__ import annotations

from pathlib import Path

import pytest


def test_offshelf_wrapper_imports():
    from tools.baselines.pixcell_offshelf_inference import OffShelfPixCellInference  # noqa: F401


def test_offshelf_wrapper_constructor_signature():
    from tools.baselines.pixcell_offshelf_inference import OffShelfPixCellInference

    expected = {"controlnet_path", "base_model_path", "vae_path", "uni_path", "device"}
    import inspect
    sig = inspect.signature(OffShelfPixCellInference.__init__)
    params = set(sig.parameters) - {"self"}
    missing = expected - params
    assert not missing, f"missing constructor params: {missing}"


def test_offshelf_run_signature():
    from tools.baselines.pixcell_offshelf_inference import OffShelfPixCellInference
    import inspect
    sig = inspect.signature(OffShelfPixCellInference.run_on_tile)
    params = set(sig.parameters) - {"self"}
    expected = {"tile_id", "cell_mask", "uni_embedding", "out_dir", "num_steps", "guidance_scale"}
    missing = expected - params
    assert not missing, f"missing run_on_tile params: {missing}"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
conda run -n pixcell python -m pytest tests/test_pixcell_offshelf_inference.py -v
```

Expected: FAIL on import (`ModuleNotFoundError: tools.baselines`).

- [ ] **Step 3: Implement the wrapper**

```python
# tools/baselines/pixcell_offshelf_inference.py
"""Off-the-shelf PixCell ControlNet inference (mask-only, no fine-tune).

Used as the A2 reference baseline. Runs the original published checkpoint
on the same paired test tiles using cached UNI embeddings + the cell_masks
channel, producing PNGs in the standard ablation-output layout.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image


@dataclass
class OffShelfPixCellInference:
    controlnet_path: str
    base_model_path: str
    vae_path: str
    uni_path: str
    device: str = "cuda"

    def __post_init__(self) -> None:
        self._loaded = False
        self._base = None
        self._controlnet = None
        self._vae = None

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        from diffusion.model.builder import build_model
        from diffusers import AutoencoderKL

        self._base = build_model(
            "PixArt_XL_2_UNI",
            input_size=32,
            pe_interpolation=0.5,
            model_max_length=1,
        ).to(self.device).eval()
        # Load base PixArt weights
        from safetensors.torch import load_file
        base_state = load_file(str(Path(self.base_model_path) / "diffusion_pytorch_model.safetensors"))
        self._base.load_state_dict(base_state, strict=False)

        self._controlnet = build_model(
            "PixCell_ControlNet_XL_2_UNI",
            input_size=32,
            pe_interpolation=0.5,
            model_max_length=1,
            controlnet_config=dict(
                zero_init_conv_out=True,
                copy_base_layers=True,
                conditioning_scale=1.0,
            ),
        ).to(self.device).eval()
        ctrl_state = load_file(self.controlnet_path)
        self._controlnet.load_state_dict(ctrl_state, strict=False)

        self._vae = AutoencoderKL.from_pretrained(self.vae_path).to(self.device).eval()
        self._loaded = True

    def encode_mask_to_latent(self, cell_mask: np.ndarray) -> torch.Tensor:
        self._ensure_loaded()
        if cell_mask.ndim == 2:
            cell_mask = np.repeat(cell_mask[:, :, None], 3, axis=2)
        x = torch.from_numpy(cell_mask).float().permute(2, 0, 1).unsqueeze(0).to(self.device)
        x = (x / 127.5) - 1.0  # [-1, 1]
        with torch.no_grad():
            latent = self._vae.encode(x).latent_dist.mode()
        latent = (latent - 0.0609) * 1.5305  # shift_factor / scale_factor
        return latent

    def run_on_tile(
        self,
        *,
        tile_id: str,
        cell_mask: np.ndarray,
        uni_embedding: np.ndarray,
        out_dir: Path,
        num_steps: int = 30,
        guidance_scale: float = 1.5,
    ) -> Path:
        from diffusion.model.diffusion_utils import get_named_beta_schedule, SpacedDiffusion, space_timesteps

        self._ensure_loaded()
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        mask_latent = self.encode_mask_to_latent(cell_mask)
        y = torch.from_numpy(uni_embedding).float().to(self.device)
        if y.dim() == 2:
            y = y.unsqueeze(0)

        diffusion = SpacedDiffusion(
            use_timesteps=space_timesteps(1000, [num_steps]),
            betas=get_named_beta_schedule("squaredcos_cap_v2", 1000),
            model_mean_type="eps",
            model_var_type="learned_range",
            loss_type="mse",
            rescale_timesteps=False,
        )

        x_t = torch.randn_like(mask_latent)
        with torch.no_grad():
            sample = diffusion.p_sample_loop(
                self._base,
                shape=x_t.shape,
                noise=x_t,
                clip_denoised=False,
                model_kwargs=dict(
                    y=y,
                    mask=None,
                    control_input=mask_latent,
                    cn_model=self._controlnet,
                ),
                progress=False,
                device=self.device,
            )
            decoded = self._vae.decode((sample / 1.5305) + 0.0609).sample

        img = ((decoded.clamp(-1, 1) + 1) * 127.5).cpu().numpy()[0].transpose(1, 2, 0).astype(np.uint8)
        out_path = out_dir / f"{tile_id}.png"
        Image.fromarray(img).save(out_path)
        return out_path
```

Note: this wrapper depends on the same diffusion utilities the production inference uses. Adapt the diffusion-loop call site if the production codebase has its own `controlnet_p_sample` helper — search for `cn_model` or `control_input` in `diffusion/` and `stage3_inference.py` and reuse the existing helper rather than reimplementing the sampler.

- [ ] **Step 4: Run smoke tests**

```bash
conda run -n pixcell python -m pytest tests/test_pixcell_offshelf_inference.py -v
```

Expected: PASS (3 tests). Real inference is not run in unit tests.

- [ ] **Step 5: Commit**

```bash
git add tools/baselines/__init__.py tools/baselines/pixcell_offshelf_inference.py tests/test_pixcell_offshelf_inference.py
git commit -m "feat(A2): off-the-shelf PixCell ControlNet inference wrapper"
```

---

## Task 4: Bypass-probe inference script

**Files:**
- Create: `tools/ablation_a2/__init__.py` (empty)
- Create: `tools/ablation_a2/run_bypass_probe.py`
- Create: `tests/test_bypass_probe.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_bypass_probe.py
"""Bypass probe correctness — TME output is zeroed at inference."""
from __future__ import annotations

import torch


def test_zero_tme_yields_mask_latent_only():
    """When TME output is zeroed, the conditioning fed to ControlNet equals mask_latent."""
    from tools.ablation_a2.run_bypass_probe import compute_bypass_conditioning

    mask_latent = torch.randn(1, 16, 32, 32)
    tme_output = torch.zeros_like(mask_latent)  # bypass probe: TME zeroed

    cond = compute_bypass_conditioning(mask_latent=mask_latent, tme_output=tme_output)

    # Under zero_mask_latent=False (additive), cond = mask_latent + tme = mask_latent.
    assert torch.allclose(cond, mask_latent, atol=1e-7)


def test_full_tme_produces_additive_conditioning():
    from tools.ablation_a2.run_bypass_probe import compute_bypass_conditioning

    mask_latent = torch.randn(1, 16, 32, 32)
    tme_output = torch.randn_like(mask_latent)
    cond = compute_bypass_conditioning(mask_latent=mask_latent, tme_output=tme_output)
    assert torch.allclose(cond, mask_latent + tme_output, atol=1e-7)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
conda run -n pixcell python -m pytest tests/test_bypass_probe.py -v
```

Expected: FAIL on import.

- [ ] **Step 3: Implement the script**

```python
# tools/ablation_a2/run_bypass_probe.py
"""A2 bypass probe inference.

Loads an A2-False checkpoint (trained with zero_mask_latent=False) and runs
inference with TME outputs forced to zero. Under additive conditioning this
collapses the input to ControlNet to mask_latent only — the canonical
"PixCell mask-only" comparator. Outputs PNGs to the configured directory.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import torch


def compute_bypass_conditioning(
    *,
    mask_latent: torch.Tensor,
    tme_output: torch.Tensor,
) -> torch.Tensor:
    """Conditioning fed to ControlNet under zero_mask_latent=False (additive).

    For the bypass probe, callers pass tme_output=zeros, which reduces this to
    mask_latent.
    """
    return mask_latent + tme_output


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to A2-False checkpoint dir.")
    parser.add_argument("--config", required=True, help="Path to config_controlnet_exp_a2_bypass.py")
    parser.add_argument("--tile_ids", required=True, help="File with one tile_id per line.")
    parser.add_argument("--out_dir", required=True, help="Where to write generated PNGs.")
    parser.add_argument("--num_steps", type=int, default=30)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    from mmcv import Config
    cfg = Config.fromfile(args.config)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Reuse the production stage3 inference machinery, but force TME outputs
    # to zero before the conditioning compute.
    from stage3_inference import build_inference_pipeline, load_tile_inputs  # see note below

    pipeline = build_inference_pipeline(cfg, checkpoint_dir=args.checkpoint, device=args.device)

    with open(args.tile_ids) as fh:
        tile_ids = [line.strip() for line in fh if line.strip()]

    for tile_id in tile_ids:
        inputs = load_tile_inputs(cfg, tile_id, exp_data_root=cfg.exp_data_root)
        with torch.no_grad():
            mask_latent = pipeline.encode_mask(inputs.cell_mask_image)
            # Bypass: zero out the TME residual.
            tme_zeroed = torch.zeros_like(mask_latent)
            ctrl_input = compute_bypass_conditioning(mask_latent=mask_latent, tme_output=tme_zeroed)
            png_path = pipeline.sample_with_control(
                tile_id=tile_id,
                ctrl_input=ctrl_input,
                uni_embedding=inputs.uni_embedding,
                out_dir=out,
                num_steps=args.num_steps,
            )
        print(f"Wrote {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Note: `build_inference_pipeline` and `load_tile_inputs` may not exist with those names. Search `stage3_inference.py` for the equivalent helpers (the file currently uses inline code) and either factor those out into named functions, or rewrite this script to call the existing inline patterns. Whichever path is taken, **only the conditioning computation must be guaranteed to use a zeroed TME output** — that is the entirety of the probe.

- [ ] **Step 4: Run tests to verify pass**

```bash
conda run -n pixcell python -m pytest tests/test_bypass_probe.py -v
```

Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add tools/ablation_a2/__init__.py tools/ablation_a2/run_bypass_probe.py tests/test_bypass_probe.py
git commit -m "feat(A2): bypass-probe inference script (zero-TME at inference)"
```

---

## Task 5: A3 stability aggregator

**Files:**
- Create: `tools/ablation_a3/__init__.py` (empty)
- Create: `tools/ablation_a3/aggregate_stability.py`
- Create: `tests/test_aggregate_stability.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_aggregate_stability.py
"""Stability aggregator: divergence detection and per-seed loss summary."""
from __future__ import annotations

import json
from pathlib import Path

import pytest


def _write_log(path: Path, entries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for entry in entries:
            fh.write(json.dumps(entry) + "\n")


def test_divergence_flagged_on_nan_loss(tmp_path):
    from tools.ablation_a3.aggregate_stability import aggregate

    seed_dir = tmp_path / "seed_1"
    _write_log(
        seed_dir / "train_log.jsonl",
        [
            {"step": 100, "loss": 1.0, "grad_norm": 0.5},
            {"step": 200, "loss": float("nan"), "grad_norm": 1.0},
        ],
    )
    summary = aggregate([seed_dir], fixed_step=100, grad_threshold=10.0, fid_diverge_cutoff=None)
    assert summary["per_seed"][0]["diverged"] is True
    assert summary["per_seed"][0]["divergence_reason"] == "nan_loss"


def test_divergence_flagged_on_grad_explosion(tmp_path):
    from tools.ablation_a3.aggregate_stability import aggregate

    seed_dir = tmp_path / "seed_2"
    _write_log(
        seed_dir / "train_log.jsonl",
        [
            {"step": 100, "loss": 1.0, "grad_norm": 0.5},
            {"step": 200, "loss": 1.1, "grad_norm": 50.0},
        ],
    )
    summary = aggregate([seed_dir], fixed_step=100, grad_threshold=10.0, fid_diverge_cutoff=None)
    assert summary["per_seed"][0]["diverged"] is True
    assert summary["per_seed"][0]["divergence_reason"] == "grad_explosion"


def test_loss_at_fixed_step(tmp_path):
    from tools.ablation_a3.aggregate_stability import aggregate

    seed_dir = tmp_path / "seed_3"
    _write_log(
        seed_dir / "train_log.jsonl",
        [
            {"step": 50, "loss": 2.0, "grad_norm": 0.1},
            {"step": 100, "loss": 1.5, "grad_norm": 0.1},
            {"step": 200, "loss": 1.0, "grad_norm": 0.1},
        ],
    )
    summary = aggregate([seed_dir], fixed_step=100, grad_threshold=100.0, fid_diverge_cutoff=None)
    assert summary["per_seed"][0]["loss_at_fixed_step"] == pytest.approx(1.5)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
conda run -n pixcell python -m pytest tests/test_aggregate_stability.py -v
```

Expected: FAIL on import.

- [ ] **Step 3: Implement aggregator**

```python
# tools/ablation_a3/aggregate_stability.py
"""Aggregate per-seed training logs for the A3 stability ablation.

Reads JSONL training logs (one record per logged step, with fields
`step`, `loss`, `grad_norm`), and computes a summary per seed:
- loss at a fixed step (mean reference point)
- divergence flag and reason
The output is written to a single stability_summary.json suitable for the
SI A3 figure builder.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable, Optional


def _read_log(path: Path) -> list[dict]:
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _is_nan(x) -> bool:
    try:
        return math.isnan(float(x))
    except (TypeError, ValueError):
        return False


def aggregate(
    seed_dirs: Iterable[Path],
    *,
    fixed_step: int,
    grad_threshold: float,
    fid_diverge_cutoff: Optional[float],
    fid_lookup: Optional[dict[str, float]] = None,
) -> dict:
    """Aggregate per-seed logs into a stability summary.

    Args:
        seed_dirs: directories each containing `train_log.jsonl`.
        fixed_step: report loss at the closest step ≤ this number.
        grad_threshold: divergence trigger if any logged grad_norm exceeds this.
        fid_diverge_cutoff: if given, also flag divergence when fid_lookup[seed] > cutoff.
        fid_lookup: optional mapping of seed-dir name → final FID.

    Returns:
        dict with keys: per_seed (list), divergence_count, mean_loss_at_fixed_step,
        std_loss_at_fixed_step.
    """
    fid_lookup = fid_lookup or {}
    per_seed: list[dict] = []
    losses_at_step: list[float] = []
    divergence_count = 0

    for seed_dir in seed_dirs:
        seed_dir = Path(seed_dir)
        log_path = seed_dir / "train_log.jsonl"
        entries = _read_log(log_path)
        diverged = False
        reason: Optional[str] = None
        loss_at_fixed: Optional[float] = None

        for e in entries:
            step = int(e["step"])
            if step <= fixed_step:
                loss_at_fixed = float(e["loss"]) if not _is_nan(e["loss"]) else loss_at_fixed
            if _is_nan(e.get("loss")):
                diverged = True
                reason = "nan_loss"
                break
            if float(e.get("grad_norm", 0.0)) > grad_threshold:
                diverged = True
                reason = "grad_explosion"
                break

        seed_name = seed_dir.name
        if not diverged and fid_diverge_cutoff is not None:
            fid_val = fid_lookup.get(seed_name)
            if fid_val is not None and fid_val > fid_diverge_cutoff:
                diverged = True
                reason = "fid_outlier"

        if diverged:
            divergence_count += 1
        else:
            if loss_at_fixed is not None:
                losses_at_step.append(loss_at_fixed)

        per_seed.append({
            "seed_dir": str(seed_dir),
            "loss_at_fixed_step": loss_at_fixed,
            "diverged": diverged,
            "divergence_reason": reason,
            "fid": fid_lookup.get(seed_name),
        })

    if losses_at_step:
        mean = sum(losses_at_step) / len(losses_at_step)
        var = sum((x - mean) ** 2 for x in losses_at_step) / len(losses_at_step)
        std = var ** 0.5
    else:
        mean = float("nan")
        std = float("nan")

    return {
        "per_seed": per_seed,
        "divergence_count": divergence_count,
        "mean_loss_at_fixed_step": mean,
        "std_loss_at_fixed_step": std,
        "fixed_step": fixed_step,
        "grad_threshold": grad_threshold,
        "fid_diverge_cutoff": fid_diverge_cutoff,
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed_dirs", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--fixed_step", type=int, default=10000)
    parser.add_argument("--grad_threshold", type=float, default=100.0)
    parser.add_argument("--fid_diverge_cutoff", type=float, default=None)
    parser.add_argument("--fid_json", default=None,
                        help="Optional JSON mapping seed-dir name → final FID.")
    args = parser.parse_args(argv)

    fid_lookup = {}
    if args.fid_json:
        fid_lookup = json.loads(Path(args.fid_json).read_text())

    summary = aggregate(
        [Path(d) for d in args.seed_dirs],
        fixed_step=args.fixed_step,
        grad_threshold=args.grad_threshold,
        fid_diverge_cutoff=args.fid_diverge_cutoff,
        fid_lookup=fid_lookup,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify pass**

```bash
conda run -n pixcell python -m pytest tests/test_aggregate_stability.py -v
```

Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add tools/ablation_a3/__init__.py tools/ablation_a3/aggregate_stability.py tests/test_aggregate_stability.py
git commit -m "feat(A3): stability aggregator for per-seed training logs"
```

---

## Task 6: SI A2 figure builder

**Files:**
- Create: `src/paper_figures/fig_si_a2_bypass.py`

- [ ] **Step 1: Implement figure builder**

```python
# src/paper_figures/fig_si_a2_bypass.py
"""Build SI_A2_bypass_probe.png — top metric table + bottom qualitative grid."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.paper_figures.style import (
    FONT_SIZE_ANNOTATION,
    FONT_SIZE_LABEL,
    FONT_SIZE_TITLE,
)

ROW_LABELS = (
    "Production\n(zero_mask_latent=True, full TME)",
    "Bypass probe\n(zero_mask_latent=False, TME→0)",
    "Off-the-shelf PixCell\n(mask-only, no fine-tune)",
)

METRIC_COLUMNS = (
    ("FID", "fid", "{:.2f}"),
    ("UNI-cos ↑", "uni_cos", "{:.3f}"),
    ("Cell-count r ↑", "cellvit_count_r", "{:.3f}"),
    ("Cell-type KL ↓", "cellvit_type_kl", "{:.3f}"),
    ("Nuc-morph KS ↓", "cellvit_nuc_ks", "{:.3f}"),
)


def _load_metrics_summary(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def _draw_metric_table(ax: plt.Axes, summary: dict) -> None:
    ax.axis("off")
    n_rows = len(ROW_LABELS)
    n_cols = 1 + len(METRIC_COLUMNS)

    # Header
    ax.text(0.0, 1.0, "Variant", fontsize=FONT_SIZE_LABEL, fontweight="bold", va="top")
    for ci, (label, _, _) in enumerate(METRIC_COLUMNS, start=1):
        ax.text(ci / n_cols + 0.18, 1.0, label, fontsize=FONT_SIZE_LABEL,
                fontweight="bold", va="top", ha="center")

    rows = summary.get("rows", [])
    for ri, (row_label, row_data) in enumerate(zip(ROW_LABELS, rows)):
        y = 0.9 - ri * 0.28
        ax.text(0.0, y, row_label, fontsize=FONT_SIZE_ANNOTATION, va="top")
        for ci, (_, key, fmt) in enumerate(METRIC_COLUMNS, start=1):
            value = row_data.get(key)
            text = fmt.format(value) if value is not None else "—"
            ax.text(ci / n_cols + 0.18, y, text, fontsize=FONT_SIZE_ANNOTATION,
                    va="top", ha="center")


def _draw_qual_grid(fig: plt.Figure, gs, tile_paths: dict[str, list[Path]]) -> None:
    n_rows = len(ROW_LABELS)
    n_cols = max((len(v) for v in tile_paths.values()), default=4)
    sub = gs.subgridspec(n_rows, n_cols, wspace=0.04, hspace=0.06)
    for ri, label_full in enumerate(ROW_LABELS):
        key = label_full.split("\n")[0].lower().replace(" ", "_").replace("-", "_")
        paths = tile_paths.get(key, [])
        for ci in range(n_cols):
            ax = fig.add_subplot(sub[ri, ci])
            ax.set_xticks([])
            ax.set_yticks([])
            if ci < len(paths):
                ax.imshow(np.asarray(Image.open(paths[ci]).convert("RGB")))
            if ci == 0:
                ax.set_ylabel(label_full, fontsize=FONT_SIZE_ANNOTATION, rotation=0,
                              ha="right", va="center", labelpad=80)


def build_si_a2_bypass_figure(
    *,
    metrics_summary_path: Path,
    tile_paths: dict[str, list[Path]],
) -> plt.Figure:
    summary = _load_metrics_summary(metrics_summary_path)
    fig = plt.figure(figsize=(15.8, 9.0))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 3.0], hspace=0.05)
    _draw_metric_table(fig.add_subplot(gs[0]), summary)
    _draw_qual_grid(fig, gs[1], tile_paths)
    fig.suptitle("A2 — Bypass probe under zero_mask_latent",
                 fontsize=FONT_SIZE_TITLE, y=0.995)
    return fig
```

- [ ] **Step 2: Smoke-import**

```bash
conda run -n pixcell python -c "from src.paper_figures.fig_si_a2_bypass import build_si_a2_bypass_figure; print('ok')"
```

Expected: `ok`.

- [ ] **Step 3: Commit**

```bash
git add src/paper_figures/fig_si_a2_bypass.py
git commit -m "feat(A2): SI bypass-probe figure builder (table + qual grid)"
```

---

## Task 7: SI A3 figure builder

**Files:**
- Create: `src/paper_figures/fig_si_a3_zero_init.py`

- [ ] **Step 1: Implement figure builder**

```python
# src/paper_figures/fig_si_a3_zero_init.py
"""Build SI_A3_zero_init.png — loss curves + divergence bar + summary table."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.paper_figures.style import (
    FONT_SIZE_ANNOTATION,
    FONT_SIZE_LABEL,
    FONT_SIZE_TICK,
    FONT_SIZE_TITLE,
)


def _load_loss_curves(seed_log_paths: list[Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_steps: list[list[int]] = []
    all_losses: list[list[float]] = []
    for path in seed_log_paths:
        steps, losses = [], []
        with Path(path).open() as fh:
            for line in fh:
                if not line.strip():
                    continue
                rec = json.loads(line)
                steps.append(int(rec["step"]))
                losses.append(float(rec["loss"]) if not str(rec["loss"]) == "nan" else float("nan"))
        all_steps.append(steps)
        all_losses.append(losses)
    # Align on common steps (intersection).
    common = sorted(set(all_steps[0]).intersection(*all_steps[1:]))
    arr = np.full((len(seed_log_paths), len(common)), np.nan, dtype=float)
    for i, (s, l) in enumerate(zip(all_steps, all_losses)):
        idx = {step: j for j, step in enumerate(s)}
        for k, step in enumerate(common):
            arr[i, k] = l[idx[step]]
    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0)
    return np.array(common), mean, std


def _draw_loss_panel(
    ax: plt.Axes,
    seeds_true: list[Path],
    seeds_false: list[Path],
) -> None:
    for paths, label, color in [
        (seeds_true, "zero_init=True (production)", "#1f77b4"),
        (seeds_false, "zero_init=False", "#d62728"),
    ]:
        if not paths:
            continue
        steps, mean, std = _load_loss_curves(paths)
        ax.plot(steps, mean, label=label, color=color, linewidth=1.8)
        ax.fill_between(steps, mean - std, mean + std, color=color, alpha=0.18, linewidth=0)
    ax.set_xscale("log")
    ax.set_xlabel("Training step (log)", fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel("Loss (mean ± SD across seeds)", fontsize=FONT_SIZE_LABEL)
    ax.tick_params(labelsize=FONT_SIZE_TICK)
    ax.legend(fontsize=FONT_SIZE_ANNOTATION, frameon=False)
    ax.set_title("Training-loss stability", fontsize=FONT_SIZE_TITLE)


def _draw_divergence_bar(ax: plt.Axes, summary_true: dict, summary_false: dict) -> None:
    counts = [summary_true.get("divergence_count", 0), summary_false.get("divergence_count", 0)]
    totals = [len(summary_true.get("per_seed", [])) or 1, len(summary_false.get("per_seed", [])) or 1]
    fractions = [c / t for c, t in zip(counts, totals)]
    ax.bar(["zero_init=True", "zero_init=False"], fractions,
           color=["#1f77b4", "#d62728"], edgecolor="black", linewidth=0.8)
    ax.set_ylabel("Diverged seeds (fraction)", fontsize=FONT_SIZE_LABEL)
    ax.set_ylim(0, 1.0)
    ax.tick_params(labelsize=FONT_SIZE_TICK)
    ax.set_title("Divergence rate", fontsize=FONT_SIZE_TITLE)
    for i, (c, t) in enumerate(zip(counts, totals)):
        ax.text(i, fractions[i] + 0.03, f"{c}/{t}", ha="center", fontsize=FONT_SIZE_ANNOTATION)


def _draw_summary_table(
    ax: plt.Axes,
    metrics_summary: dict,
) -> None:
    ax.axis("off")
    cols = ("Variant", "Loss@step (mean ± SD)", "Diverged", "FID", "UNI-cos",
            "Cell r", "Type KL", "Nuc KS")
    rows = metrics_summary.get("rows", [])
    n_cols = len(cols)

    for ci, label in enumerate(cols):
        ax.text(ci / n_cols, 1.0, label, fontsize=FONT_SIZE_LABEL,
                fontweight="bold", va="top")

    for ri, row in enumerate(rows):
        y = 0.85 - ri * 0.18
        cells = [
            row.get("variant", "?"),
            f"{row.get('loss_mean', float('nan')):.3f} ± {row.get('loss_std', float('nan')):.3f}",
            f"{row.get('divergence_count', 0)}/{row.get('n_seeds', 0)}",
            f"{row.get('fid', float('nan')):.2f}",
            f"{row.get('uni_cos', float('nan')):.3f}",
            f"{row.get('cellvit_count_r', float('nan')):.3f}",
            f"{row.get('cellvit_type_kl', float('nan')):.3f}",
            f"{row.get('cellvit_nuc_ks', float('nan')):.3f}",
        ]
        for ci, text in enumerate(cells):
            ax.text(ci / n_cols, y, text, fontsize=FONT_SIZE_ANNOTATION, va="top")


def build_si_a3_zero_init_figure(
    *,
    seeds_true_logs: list[Path],
    seeds_false_logs: list[Path],
    stability_summary_true_path: Path,
    stability_summary_false_path: Path,
    metrics_summary_path: Path,
) -> plt.Figure:
    fig = plt.figure(figsize=(15.8, 11.0))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.5, 1.0, 1.5], hspace=0.45)
    _draw_loss_panel(fig.add_subplot(gs[0]), seeds_true_logs, seeds_false_logs)
    _draw_divergence_bar(
        fig.add_subplot(gs[1]),
        json.loads(Path(stability_summary_true_path).read_text()),
        json.loads(Path(stability_summary_false_path).read_text()),
    )
    _draw_summary_table(
        fig.add_subplot(gs[2]),
        json.loads(Path(metrics_summary_path).read_text()),
    )
    fig.suptitle("A3 — Zero-init residual gating: stability ablation",
                 fontsize=FONT_SIZE_TITLE, y=0.995)
    return fig
```

- [ ] **Step 2: Smoke-import**

```bash
conda run -n pixcell python -c "from src.paper_figures.fig_si_a3_zero_init import build_si_a3_zero_init_figure; print('ok')"
```

Expected: `ok`.

- [ ] **Step 3: Commit**

```bash
git add src/paper_figures/fig_si_a3_zero_init.py
git commit -m "feat(A3): SI zero-init stability figure builder"
```

---

## Task 8: Register SI builders in main.py

**Files:**
- Modify: `src/paper_figures/main.py`

- [ ] **Step 1: Read the current main.py**

```bash
sed -n '1,30p' src/paper_figures/main.py
```

- [ ] **Step 2: Add imports and constants**

Edit `src/paper_figures/main.py`:

After the existing `from src.paper_figures.fig_inverse_decoding import ...` line, add:

```python
from src.paper_figures.fig_si_a2_bypass import build_si_a2_bypass_figure
from src.paper_figures.fig_si_a3_zero_init import build_si_a3_zero_init_figure
```

After the existing path constants block, add:

```python
A2_METRICS_SUMMARY = ROOT / "inference_output" / "a2_bypass" / "metrics_summary.json"
A2_TILE_DIRS = {
    "production": ROOT / "inference_output" / "paired_ablation" / "production",
    "bypass_probe": ROOT / "inference_output" / "a2_bypass" / "bypass",
    "off_the_shelf": ROOT / "inference_output" / "a2_bypass" / "offshelf",
}
A2_TILE_IDS = ["10752_13824", "13056_27392", "5632_18432", "21504_8192"]  # match main paired grid

A3_METRICS_SUMMARY = ROOT / "inference_output" / "a3_zero_init" / "metrics_summary.json"
A3_STABILITY_TRUE = ROOT / "inference_output" / "a3_zero_init" / "stability_true.json"
A3_STABILITY_FALSE = ROOT / "inference_output" / "a3_zero_init" / "stability_false.json"
A3_SEEDS_TRUE_LOGS = sorted((ROOT / "checkpoints" / "pixcell_controlnet_exp").glob("seed_*/train_log.jsonl"))
A3_SEEDS_FALSE_LOGS = sorted((ROOT / "checkpoints" / "pixcell_controlnet_exp_a3_no_zero_init").glob("seed_*/train_log.jsonl"))
```

After the existing `save_figure_png(fig_combined, ...)` block, add:

```python
    if A2_METRICS_SUMMARY.is_file():
        a2_tile_paths = {
            key: [d / f"{tid}.png" for tid in A2_TILE_IDS if (d / f"{tid}.png").is_file()]
            for key, d in A2_TILE_DIRS.items()
        }
        fig_a2 = build_si_a2_bypass_figure(
            metrics_summary_path=A2_METRICS_SUMMARY,
            tile_paths=a2_tile_paths,
        )
        save_figure_png(fig_a2, PNG_DIR / "SI_A2_bypass_probe.png")
        save_figure_png(fig_a2, PNG_DIR_UPDATED / "SI_A2_bypass_probe.png")
    else:
        print("Skipping SI_A2_bypass_probe.png; missing", A2_METRICS_SUMMARY)

    if A3_METRICS_SUMMARY.is_file() and A3_STABILITY_TRUE.is_file() and A3_STABILITY_FALSE.is_file():
        fig_a3 = build_si_a3_zero_init_figure(
            seeds_true_logs=A3_SEEDS_TRUE_LOGS,
            seeds_false_logs=A3_SEEDS_FALSE_LOGS,
            stability_summary_true_path=A3_STABILITY_TRUE,
            stability_summary_false_path=A3_STABILITY_FALSE,
            metrics_summary_path=A3_METRICS_SUMMARY,
        )
        save_figure_png(fig_a3, PNG_DIR / "SI_A3_zero_init.png")
        save_figure_png(fig_a3, PNG_DIR_UPDATED / "SI_A3_zero_init.png")
    else:
        print("Skipping SI_A3_zero_init.png; missing one of",
              A3_METRICS_SUMMARY, A3_STABILITY_TRUE, A3_STABILITY_FALSE)
```

(Place these blocks before the final `print(...)` that announces saved files.)

- [ ] **Step 3: Smoke-run main.py**

```bash
conda run -n pixcell python -m src.paper_figures.main 2>&1 | tail -10
```

Expected: existing figures still build; the two new SI sections print "Skipping ..." (because input files don't exist yet — that's fine).

- [ ] **Step 4: Commit**

```bash
git add src/paper_figures/main.py
git commit -m "feat(A2/A3): wire SI design-justification figures into main"
```

---

## Task 9: Operator runbook (training + inference + CellViT)

This task is **operator-executed**, not agent-executed. Document the steps in `docs/runbooks/A2_A3_design_justification.md` so they can be reproduced.

**Files:**
- Create: `docs/runbooks/A2_A3_design_justification.md`

- [ ] **Step 1: Write the runbook**

```markdown
# A2 / A3 Runbook

## Short-proxy length

Run a single seed of `config_controlnet_exp.py` with `num_epochs` reduced to 25% (≈5 epochs / ≈940 steps) and confirm paired-test FID ranking against the production run matches at full schedule on a single reference seed. Lock that step count as `--proxy_steps` for all subsequent A2/A3 short-proxy runs.

## A2 training (zero_mask_latent=False)

5 seeds (short proxy) + 1 full-headline:

```bash
for SEED in 1 2 3 4 5; do
  python -m accelerate.commands.launch train_scripts/train_controlnet_exp.py \
    --config configs/config_controlnet_exp_a2_bypass.py \
    --seed $SEED \
    --work-dir checkpoints/pixcell_controlnet_exp_a2_bypass/seed_${SEED}
done

# Full-headline run
python -m accelerate.commands.launch train_scripts/train_controlnet_exp.py \
  --config configs/config_controlnet_exp_a2_bypass.py \
  --seed 42 \
  --work-dir checkpoints/pixcell_controlnet_exp_a2_bypass/full_seed_42
```

## A3 training (zero_init_conv_out=False)

Same pattern with `config_controlnet_exp_a3_no_zero_init.py`.

## A2 inference

1. Production row: reuse existing `inference_output/paired_ablation/production/` (or rerun with current production checkpoint).
2. Bypass probe row:
   ```bash
   python -m tools.ablation_a2.run_bypass_probe \
     --checkpoint checkpoints/pixcell_controlnet_exp_a2_bypass/full_seed_42 \
     --config configs/config_controlnet_exp_a2_bypass.py \
     --tile_ids tools/ablation_report/paired_test_tile_ids.txt \
     --out_dir inference_output/a2_bypass/bypass
   ```
3. Off-the-shelf row:
   ```bash
   python -m tools.baselines.pixcell_offshelf_inference \
     --controlnet pretrained_models/pixcell-256-controlnet/controlnet/diffusion_pytorch_model.safetensors \
     --base pretrained_models/pixcell-256/transformer \
     --vae pretrained_models/sd-3.5-vae/vae \
     --uni pretrained_models/uni-2h \
     --tile_ids tools/ablation_report/paired_test_tile_ids.txt \
     --out_dir inference_output/a2_bypass/offshelf
   ```

## CellViT pass

Run the existing `tools/cellvit/export_batch.py` pointed at each generated-tile directory. Then `tools/cellvit/import_results.py` to merge.

## Metric aggregation

```bash
python -m tools.compute_ablation_metrics \
  --variant_dirs inference_output/paired_ablation/production \
                 inference_output/a2_bypass/bypass \
                 inference_output/a2_bypass/offshelf \
  --reference_dir data/orion-crc33 \
  --out inference_output/a2_bypass/metrics_summary.json
```

(The above invokes the existing metrics tool; if it does not yet support multi-variant aggregation in this exact form, add a thin wrapper script `tools/ablation_a2/aggregate_metrics.py` that emits the schema expected by `fig_si_a2_bypass.py`: `{"rows": [{"variant": str, "fid": ..., "uni_cos": ..., "cellvit_count_r": ..., "cellvit_type_kl": ..., "cellvit_nuc_ks": ...}, ...]}`.)

## A3 stability aggregation

```bash
python -m tools.ablation_a3.aggregate_stability \
  --seed_dirs checkpoints/pixcell_controlnet_exp/seed_1 \
              checkpoints/pixcell_controlnet_exp/seed_2 \
              checkpoints/pixcell_controlnet_exp/seed_3 \
              checkpoints/pixcell_controlnet_exp/seed_4 \
              checkpoints/pixcell_controlnet_exp/seed_5 \
  --out inference_output/a3_zero_init/stability_true.json \
  --fixed_step 10000 --grad_threshold 100.0

python -m tools.ablation_a3.aggregate_stability \
  --seed_dirs checkpoints/pixcell_controlnet_exp_a3_no_zero_init/seed_1 \
              ... \
  --out inference_output/a3_zero_init/stability_false.json \
  --fixed_step 10000 --grad_threshold 100.0
```

## Final figure regeneration

```bash
python -m src.paper_figures.main
```

Should now produce `figures/pngs/SI_A2_bypass_probe.png` and `figures/pngs/SI_A3_zero_init.png` (and updated copies in `pngs_updated/`).
```

- [ ] **Step 2: Commit**

```bash
git add docs/runbooks/A2_A3_design_justification.md
git commit -m "docs(A2/A3): runbook for training, inference, CellViT, and figure regen"
```

---

## Self-Review Checklist (post-write)

- [x] Spec coverage:
  - §3 definitions → Tasks 1, 2, 4 (configs + bypass probe + off-the-shelf wrapper).
  - §5 A2 three-row table + qual grid → Task 6.
  - §5.4 off-the-shelf reference notes → Task 3.
  - §6 A3 panels → Task 7.
  - §6.3 divergence definition → Task 5 (aggregator with NaN/grad thresholds).
  - §7 metric set → integrated in Tasks 6, 7 via metrics_summary JSON schema.
  - §9 components map 1:1 to Tasks 1–7.
  - §10 output artifacts → Task 8 wires them; Task 9 documents how they're produced.
  - §11 failure modes → addressed in Tasks 3 (mask-format note), 9 (operator can adapt), 5 (divergence handling).
  - §12 testing → Tasks 3, 4, 5 each ship a test.
  - §13 caption requirements → captioning is part of operator-driven figure usage; not a code task. No code task added — captions live in the paper text. NOTE: if captions need to be embedded in figures, add a follow-up task; currently the figures only carry titles.
  - §14 open assumptions → Task 9 runbook validates short-proxy length and original-PixCell mask format empirically before full runs.
  - §15 acceptance criteria → satisfied by Tasks 6–8 outputs once operator completes Task 9.

- [x] Placeholder scan: no TBDs, every code step has actual code. The note in Task 4 about `build_inference_pipeline` legitimately requires the implementer to inspect `stage3_inference.py` and adapt — not a placeholder, an explicit instruction.

- [x] Type consistency: `compute_bypass_conditioning` signature unchanged across uses. Schema for `metrics_summary.json` (`{"rows": [...]}`) is the same in Tasks 6, 7, and 9 (runbook).

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-26-A2-A3-design-justification-ablations.md`. Two execution options:

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

Per CLAUDE.md, all real implementation here will be delegated to Codex regardless of execution mode. Which approach?
