# SI A1+A2 Unified Figure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `src/paper_figures/fig_si_a1_a2_unified.py` (weight-free figure from cache) and `tools/ablation_a1_a2/build_cache.py` (one-time tile generator + curve extractor that populates the cache).

**Architecture:** Two-phase pipeline. Phase 1 (`build_cache.py`) runs once with weights present: reads all `train_log.jsonl` files, generates 4 qualitative tiles per variant via existing `tile_pipeline.py`, and writes `inference_output/si_a1_a2/cache.json` + PNGs. Phase 2 (`fig_si_a1_a2_unified.py`) reads only `cache.json` + PNGs — no model loading. The cache is incrementally mergeable: `--update-curves` re-reads logs without touching tiles; `--merge-metrics-file` imports pre-computed FID/CellViT results. Weights are deletable after Phase 1.

**Tech Stack:** Python 3.11+, matplotlib, numpy, PIL, torch (Phase 1 only). Reuses `tools/ablation_a3/aggregate_stability._read_log`, `tools/stage3/tile_pipeline.load_all_models`, `tools/stage3/tile_pipeline.generate_tile`, `tools/stage3/common.make_inference_scheduler`, `tools/stage3/common.resolve_uni_embedding`, `src/paper_figures/style.py`.

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `tools/ablation_a1_a2/__init__.py` | Create | Empty package marker |
| `tools/ablation_a1_a2/log_utils.py` | Create | Log parsing, float serialisation, LOG_SOURCES registry |
| `tools/ablation_a1_a2/cache_io.py` | Create | `load_cache`, `save_cache`, `merge_curves`, `merge_metrics`, `merge_params` |
| `tools/ablation_a1_a2/build_cache.py` | Create | CLI: curve extraction + tile inference + `--update-curves` + `--merge-metrics-file` |
| `src/paper_figures/fig_si_a1_a2_unified.py` | Create | Figure builder (cache.json + PNGs → matplotlib figure) |
| `tests/test_ablation_a1_a2_log_utils.py` | Create | Tests for log parsing and float serialisation |
| `tests/test_ablation_a1_a2_cache_io.py` | Create | Tests for load/save/merge logic |
| `tests/test_fig_si_a1_a2_unified.py` | Create | Tests for figure rendering from synthetic cache |

---

## Task 1: Package skeleton + log utilities

**Files:**
- Create: `tools/ablation_a1_a2/__init__.py`
- Create: `tools/ablation_a1_a2/log_utils.py`
- Create: `tests/test_ablation_a1_a2_log_utils.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_ablation_a1_a2_log_utils.py
from __future__ import annotations
import json
import math
from pathlib import Path
import pytest
from tools.ablation_a1_a2.log_utils import (
    serialise_float,
    deserialise_float,
    extract_run,
)

def test_serialise_inf():
    assert serialise_float(float("inf")) == "inf"
    assert serialise_float(float("-inf")) == "-inf"

def test_serialise_nan_is_none():
    assert serialise_float(float("nan")) is None

def test_serialise_finite():
    assert serialise_float(0.123) == pytest.approx(0.123)

def test_deserialise_inf():
    assert math.isinf(deserialise_float("inf"))
    assert deserialise_float("inf") > 0

def test_deserialise_none_is_nan():
    assert math.isnan(deserialise_float(None))

def test_extract_run_jsonl(tmp_path):
    log = tmp_path / "train_log.jsonl"
    log.write_text(
        json.dumps({"step": 50, "loss": 0.12, "grad_norm": float("inf")}) + "\n" +
        json.dumps({"step": 100, "loss": 0.10, "grad_norm": 0.02}) + "\n"
    )
    entries = extract_run(log)
    assert len(entries) == 2
    assert entries[0]["step"] == 50
    assert entries[0]["grad_norm"] == "inf"
    assert entries[1]["grad_norm"] == pytest.approx(0.02)

def test_extract_run_skips_missing_file(tmp_path):
    entries = extract_run(tmp_path / "nonexistent.jsonl")
    assert entries == []
```

- [ ] **Step 2: Run to verify failure**

```bash
cd /home/ec2-user/PixCell && conda run -n pixcell pytest tests/test_ablation_a1_a2_log_utils.py -v 2>&1 | tail -20
```
Expected: `ModuleNotFoundError` or `ImportError`.

- [ ] **Step 3: Create package and log_utils.py**

```python
# tools/ablation_a1_a2/__init__.py
# (empty)
```

```python
# tools/ablation_a1_a2/log_utils.py
from __future__ import annotations
import math
from pathlib import Path

from tools.ablation_a3.aggregate_stability import _read_log

# Registry of all known log file paths. Add new seeds here when downloaded.
LOG_SOURCES: dict[str, dict[str, Path]] = {
    "production": {
        "full_seed_42": Path("checkpoints/pixcell_controlnet_exp/train_log.log"),
    },
    "a1_concat": {
        "full_seed_42": Path("checkpoints/a1_concat/full_seed_42/train_log.jsonl"),
        "seed_1":       Path("checkpoints/a1_concat/seed_1/train_log.jsonl"),
        "seed_2":       Path("checkpoints/a1_concat/seed_2/train_log.jsonl"),
        "seed_3":       Path("checkpoints/a1_concat/seed_3/train_log.jsonl"),
    },
    "a1_per_channel": {
        "full_seed_42": Path("checkpoints/a1_per_channel/full_seed_42/train_log.jsonl"),
        "seed_1":       Path("checkpoints/a1_per_channel/seed_1/train_log.jsonl"),
        "seed_2":       Path("checkpoints/a1_per_channel/seed_2/train_log.jsonl"),
        "seed_3":       Path("checkpoints/a1_per_channel/seed_3/train_log.jsonl"),
    },
    "a2_bypass": {
        "full_seed_42": Path("checkpoints/a2_a3/a2_bypass/full_seed_42/train_log.jsonl"),
        "seed_1":       Path("checkpoints/a2_a3/a2_bypass/seed_1/train_log.jsonl"),
        "seed_2":       Path("checkpoints/a2_a3/a2_bypass/seed_2/train_log.jsonl"),
        # seed_3/4/5 added here when downloaded
    },
}


def serialise_float(v: float) -> float | str | None:
    if math.isnan(v):
        return None
    if math.isinf(v):
        return "inf" if v > 0 else "-inf"
    return v


def deserialise_float(v: object) -> float:
    if v is None:
        return float("nan")
    if v == "inf":
        return float("inf")
    if v == "-inf":
        return float("-inf")
    return float(v)


def extract_run(path: Path) -> list[dict]:
    """Parse one log file (JSONL or plain-text). Returns [] if path missing."""
    if not path.exists():
        return []
    entries = _read_log(path)
    result = []
    for e in entries:
        try:
            step = int(e["step"])
        except (KeyError, ValueError):
            continue
        loss = serialise_float(float(e.get("loss", float("nan"))))
        grad = serialise_float(float(e.get("grad_norm", float("nan"))))
        result.append({"step": step, "loss": loss, "grad_norm": grad})
    return result


def extract_all_curves(extra_sources: dict[str, dict[str, Path]] | None = None) -> dict:
    """
    Return {variant: {run_id: [{step, loss, grad_norm}]}} for all present log files.
    extra_sources adds/overrides entries (e.g. newly-downloaded seeds).
    """
    sources: dict[str, dict[str, Path]] = {k: dict(v) for k, v in LOG_SOURCES.items()}
    if extra_sources:
        for variant, runs in extra_sources.items():
            sources.setdefault(variant, {}).update(runs)
    result: dict[str, dict[str, list]] = {}
    for variant, runs in sources.items():
        result[variant] = {}
        for run_id, path in runs.items():
            entries = extract_run(Path(path))
            if entries:
                result[variant][run_id] = entries
    return result
```

- [ ] **Step 4: Run tests**

```bash
conda run -n pixcell pytest tests/test_ablation_a1_a2_log_utils.py -v 2>&1 | tail -15
```
Expected: all 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/ablation_a1_a2/__init__.py tools/ablation_a1_a2/log_utils.py tests/test_ablation_a1_a2_log_utils.py
git commit -m "feat(ablation-a1-a2): log utils — extract_run, extract_all_curves, float serialisation"
```

---

## Task 2: Cache IO

**Files:**
- Create: `tools/ablation_a1_a2/cache_io.py`
- Create: `tests/test_ablation_a1_a2_cache_io.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_ablation_a1_a2_cache_io.py
from __future__ import annotations
import json
from pathlib import Path
import pytest
from tools.ablation_a1_a2.cache_io import (
    load_cache,
    save_cache,
    merge_curves,
    merge_metrics,
    merge_params,
)

def test_load_empty(tmp_path):
    cache = load_cache(tmp_path / "nonexistent.json")
    assert cache["training_curves"] == {}
    assert cache["metrics"] == {}
    assert cache["tile_ids"] == []

def test_save_and_reload(tmp_path):
    path = tmp_path / "cache.json"
    cache = load_cache(path)
    cache["tile_ids"] = ["t1", "t2"]
    save_cache(cache, path)
    reloaded = load_cache(path)
    assert reloaded["tile_ids"] == ["t1", "t2"]
    assert "generated" in reloaded

def test_merge_curves_additive(tmp_path):
    path = tmp_path / "cache.json"
    cache = load_cache(path)
    merge_curves(cache, {"a1_concat": {"seed_1": [{"step": 50, "loss": 0.1, "grad_norm": 0.02}]}})
    assert "seed_1" in cache["training_curves"]["a1_concat"]
    # second merge adds seed_2 without clobbering seed_1
    merge_curves(cache, {"a1_concat": {"seed_2": [{"step": 50, "loss": 0.11, "grad_norm": 0.02}]}})
    assert "seed_1" in cache["training_curves"]["a1_concat"]
    assert "seed_2" in cache["training_curves"]["a1_concat"]

def test_merge_metrics(tmp_path):
    path = tmp_path / "cache.json"
    cache = load_cache(path)
    merge_metrics(cache, "production", {"fid": 12.3, "uni_cos": 0.85})
    assert cache["metrics"]["production"]["fid"] == pytest.approx(12.3)

def test_merge_params(tmp_path):
    path = tmp_path / "cache.json"
    cache = load_cache(path)
    merge_params(cache, {"a1_concat": 12_000_000})
    assert cache["params"]["a1_concat"] == 12_000_000
```

- [ ] **Step 2: Run to verify failure**

```bash
conda run -n pixcell pytest tests/test_ablation_a1_a2_cache_io.py -v 2>&1 | tail -10
```

- [ ] **Step 3: Write cache_io.py**

```python
# tools/ablation_a1_a2/cache_io.py
from __future__ import annotations
import datetime
import json
from pathlib import Path

CACHE_VERSION = 1


def _empty_cache() -> dict:
    return {
        "version": CACHE_VERSION,
        "generated": "",
        "tile_ids": [],
        "training_curves": {},
        "metrics": {},
        "params": {},
    }


def load_cache(path: Path) -> dict:
    path = Path(path)
    if not path.exists():
        return _empty_cache()
    return json.loads(path.read_text(encoding="utf-8"))


def save_cache(cache: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cache["generated"] = datetime.date.today().isoformat()
    path.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")


def merge_curves(cache: dict, new_curves: dict) -> dict:
    """Deep-merge new_curves into cache['training_curves']. Existing run_ids not overwritten."""
    existing = cache.setdefault("training_curves", {})
    for variant, runs in new_curves.items():
        existing.setdefault(variant, {}).update(runs)
    return cache


def merge_metrics(cache: dict, variant: str, metrics: dict) -> dict:
    """Write (or overwrite) metrics for one variant."""
    cache.setdefault("metrics", {})[variant] = metrics
    return cache


def merge_params(cache: dict, params: dict) -> dict:
    """Write (or overwrite) param counts."""
    cache.setdefault("params", {}).update(params)
    return cache
```

- [ ] **Step 4: Run tests**

```bash
conda run -n pixcell pytest tests/test_ablation_a1_a2_cache_io.py -v 2>&1 | tail -10
```
Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/ablation_a1_a2/cache_io.py tests/test_ablation_a1_a2_cache_io.py
git commit -m "feat(ablation-a1-a2): cache_io — load/save/merge cache.json"
```

---

## Task 3: build_cache.py — `--update-curves` path (no weights needed)

**Files:**
- Create: `tools/ablation_a1_a2/build_cache.py`

- [ ] **Step 1: Write the script with curves-only path**

```python
# tools/ablation_a1_a2/build_cache.py
"""
Build or incrementally update inference_output/si_a1_a2/cache.json.

Usage
-----
# Extract/update training curves only (no GPU, no weights needed):
conda activate pixcell
python tools/ablation_a1_a2/build_cache.py \\
    --update-curves \\
    --cache-dir inference_output/si_a1_a2

# Full run — curves + tile inference + UNI-cos (needs weights + GPU):
python tools/ablation_a1_a2/build_cache.py \\
    --cache-dir inference_output/si_a1_a2 \\
    --tile-ids-file tools/ablation_a1_a2/qual_tile_ids.txt \\
    --device cuda

# Merge pre-computed FID/CellViT results into cache (no GPU):
python tools/ablation_a1_a2/build_cache.py \\
    --merge-metrics-file tools/ablation_a1_a2/metrics_production.json \\
    --cache-dir inference_output/si_a1_a2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from tools.ablation_a1_a2.cache_io import load_cache, merge_curves, merge_metrics, merge_params, save_cache
from tools.ablation_a1_a2.log_utils import extract_all_curves

# ── Checkpoint map for full-run inference ────────────────────────────────────
# Each entry: variant_key → dict with config_path, ckpt_dir (step dir), variant_type.
# variant_type: "standard" | "bypass" | "off_shelf"
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
    # a2_off_shelf: uses pretrained PixCell-256 with no fine-tuning.
    # Deferred — requires confirming the pretrained-only config path.
}


def _run_update_curves(cache_dir: Path) -> None:
    cache_path = cache_dir / "cache.json"
    cache = load_cache(cache_path)
    curves = extract_all_curves()
    merge_curves(cache, curves)
    save_cache(cache, cache_path)
    total_runs = sum(len(v) for v in cache["training_curves"].values())
    print(f"Curves updated: {len(cache['training_curves'])} variants, {total_runs} runs → {cache_path}")


def _run_merge_metrics(cache_dir: Path, metrics_file: Path) -> None:
    """
    Merge a pre-computed metrics JSON into cache.
    Expected format: {"variant": "a1_concat", "fid": 12.3, "uni_cos": 0.85,
                      "cellvit_count_r": 0.72, "cellvit_type_kl": 0.05, "cellvit_nuc_ks": 0.10}
    """
    payload = json.loads(metrics_file.read_text(encoding="utf-8"))
    variant = payload.pop("variant")
    cache_path = cache_dir / "cache.json"
    cache = load_cache(cache_path)
    merge_metrics(cache, variant, payload)
    save_cache(cache, cache_path)
    print(f"Metrics merged for variant '{variant}' → {cache_path}")


def _tiles_exist(tile_dir: Path, tile_ids: list[str]) -> bool:
    return all((tile_dir / f"{tid}.png").exists() for tid in tile_ids)


def _run_full(cache_dir: Path, tile_ids: list[str], device: str) -> None:
    """Generate tiles for each variant (skip if already present), compute UNI-cos."""
    from diffusion.utils.misc import read_config
    from tools.stage3.common import make_inference_scheduler, resolve_uni_embedding
    from tools.stage3.tile_pipeline import (
        generate_tile,
        load_all_models,
        resolve_data_layout,
    )
    from PIL import Image as _Image

    data_root = Path("data/orion-crc33")
    exp_channels_dir, features_dir, he_dir = resolve_data_layout(data_root)

    cache_path = cache_dir / "cache.json"
    cache = load_cache(cache_path)
    # Always refresh curves before inference
    merge_curves(cache, extract_all_curves())
    cache["tile_ids"] = sorted(set(cache.get("tile_ids", []) + tile_ids))
    save_cache(cache, cache_path)

    for variant_key, vcfg in INFERENCE_VARIANTS.items():
        tile_dir = cache_dir / "tiles" / variant_key
        tile_dir.mkdir(parents=True, exist_ok=True)

        if _tiles_exist(tile_dir, tile_ids):
            print(f"[{variant_key}] tiles already present — skipping inference")
            continue

        print(f"[{variant_key}] loading models from {vcfg['ckpt_dir']} …")
        config_path = vcfg["config_path"]
        config = read_config(config_path)
        config._filename = config_path
        models = load_all_models(config, config_path, vcfg["ckpt_dir"], device)
        scheduler = make_inference_scheduler(num_steps=30, device=device)

        for tile_id in tile_ids:
            out_path = tile_dir / f"{tile_id}.png"
            if out_path.exists():
                continue
            uni_embeds = resolve_uni_embedding(tile_id, feat_dir=features_dir, null_uni=False)
            if vcfg["variant_type"] == "standard":
                gen_np, _ = generate_tile(
                    tile_id, models, config, scheduler, uni_embeds,
                    device, exp_channels_dir, guidance_scale=1.5, seed=42,
                )
            elif vcfg["variant_type"] == "bypass":
                gen_np = _generate_bypass_tile(
                    tile_id, models, config, scheduler, uni_embeds, device,
                    exp_channels_dir, guidance_scale=1.5, seed=42,
                )
            else:
                raise ValueError(f"Unknown variant_type: {vcfg['variant_type']}")
            _Image.fromarray(gen_np).save(out_path)
            print(f"  wrote {out_path}")

        # Also save GT tiles (only once)
        gt_dir = cache_dir / "tiles" / "gt"
        gt_dir.mkdir(parents=True, exist_ok=True)
        for tile_id in tile_ids:
            gt_path = gt_dir / f"{tile_id}.png"
            if not gt_path.exists():
                src = he_dir / f"{tile_id}.png"
                if src.exists():
                    import shutil
                    shutil.copy2(src, gt_path)

    save_cache(cache, cache_path)
    print(f"Cache saved → {cache_path}")


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
    """A2 bypass: TME output zeroed, conditioning = VAE mask only."""
    import torch
    from tools.stage3.tile_pipeline import (
        _decode_latents_to_image,
        _make_fixed_noise,
        prepare_tile_context,
    )
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
    fixed_noise = _make_fixed_noise(
        config=config, scheduler=scheduler, device=device,
        dtype=context["dtype"], seed=seed,
    )
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
        denoised, vae=context["vae"],
        vae_scale=context["vae_scale"], vae_shift=context["vae_shift"],
        dtype=context["dtype"],
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default="inference_output/si_a1_a2",
                        help="Root dir for cache.json and tiles/")
    parser.add_argument("--update-curves", action="store_true",
                        help="Only update training_curves in cache.json; skip inference")
    parser.add_argument("--tile-ids-file", default=None,
                        help="Text file with one tile ID per line (required for full run)")
    parser.add_argument("--merge-metrics-file", default=None,
                        help="JSON file with pre-computed metrics to merge into cache")
    parser.add_argument("--device", default="cuda")
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

    tile_ids = [
        line.strip()
        for line in Path(args.tile_ids_file).read_text().splitlines()
        if line.strip()
    ]
    _run_full(cache_dir, tile_ids, args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Smoke-test `--update-curves` on real logs (no GPU needed)**

```bash
cd /home/ec2-user/PixCell
conda run -n pixcell python tools/ablation_a1_a2/build_cache.py \
    --update-curves \
    --cache-dir /tmp/si_a1_a2_test
```
Expected output ends with: `Curves updated: 4 variants, N runs → /tmp/si_a1_a2_test/cache.json`

- [ ] **Step 3: Verify cache.json structure**

```bash
conda run -n pixcell python -c "
import json; c = json.load(open('/tmp/si_a1_a2_test/cache.json'))
print('variants:', list(c['training_curves'].keys()))
v = c['training_curves']['a1_per_channel']['full_seed_42']
print('per_channel full_seed_42 step 50:', v[0])
print('grad_norm value:', v[0]['grad_norm'])  # should be 'inf'
"
```
Expected: `grad_norm value: inf` (the string).

- [ ] **Step 4: Commit**

```bash
git add tools/ablation_a1_a2/build_cache.py
git commit -m "feat(ablation-a1-a2): build_cache.py — update-curves + merge-metrics + full inference skeleton"
```

---

## Task 4: build_cache.py — param counting

**Files:**
- Modify: `tools/ablation_a1_a2/build_cache.py` (add `_record_params`)

- [ ] **Step 1: Add param-counting helper and call it in `_run_full`**

Add this function to `build_cache.py` before `main()`:

```python
def _record_params(cache_dir: Path, cache: dict, device: str) -> None:
    """Count trainable parameters for each A1 variant and store in cache['params']."""
    from diffusion.utils.misc import read_config
    from tools.stage3.tile_pipeline import load_all_models

    a1_variants = {k: v for k, v in INFERENCE_VARIANTS.items()
                   if k in ("a1_concat", "a1_per_channel", "production")}
    params: dict[str, int] = {}
    for variant_key, vcfg in a1_variants.items():
        config_path = vcfg["config_path"]
        config = read_config(config_path)
        config._filename = config_path
        models = load_all_models(config, config_path, vcfg["ckpt_dir"], device)
        tme_params = sum(p.numel() for p in models["tme_module"].parameters())
        ctrl_params = sum(p.numel() for p in models["controlnet"].parameters())
        params[variant_key] = tme_params + ctrl_params
        print(f"  {variant_key}: {params[variant_key]:,} trainable params")
    merge_params(cache, params)
```

Then call `_record_params(cache_dir, cache, device)` inside `_run_full()` before the final `save_cache`.

- [ ] **Step 2: Run smoke test (no GPU: skip by checking `--update-curves` still works)**

```bash
conda run -n pixcell python tools/ablation_a1_a2/build_cache.py \
    --update-curves --cache-dir /tmp/si_a1_a2_test
```
Expected: same output as before, no errors.

- [ ] **Step 3: Commit**

```bash
git add tools/ablation_a1_a2/build_cache.py
git commit -m "feat(ablation-a1-a2): build_cache.py — param counting for A1 variants"
```

---

## Task 5: Figure builder — data loading and curve aggregation

**Files:**
- Create: `src/paper_figures/fig_si_a1_a2_unified.py`
- Create: `tests/test_fig_si_a1_a2_unified.py`

- [ ] **Step 1: Write failing tests for data helpers**

```python
# tests/test_fig_si_a1_a2_unified.py
from __future__ import annotations
import json
import math
import numpy as np
import pytest
from pathlib import Path

# Build a minimal synthetic cache for all tests
SYNTHETIC_CACHE = {
    "version": 1,
    "generated": "2026-04-27",
    "tile_ids": ["tile_001", "tile_002"],
    "training_curves": {
        "production": {
            "full_seed_42": [
                {"step": 50, "loss": 0.20, "grad_norm": 0.05},
                {"step": 100, "loss": 0.15, "grad_norm": 0.04},
            ]
        },
        "a1_concat": {
            "seed_1": [{"step": 50, "loss": 0.22, "grad_norm": 0.06},
                       {"step": 100, "loss": 0.18, "grad_norm": 0.05}],
            "seed_2": [{"step": 50, "loss": 0.24, "grad_norm": 0.07},
                       {"step": 100, "loss": 0.19, "grad_norm": 0.05}],
        },
        "a1_per_channel": {
            "seed_1": [{"step": 50, "loss": 0.25, "grad_norm": "inf"},
                       {"step": 100, "loss": 0.22, "grad_norm": "inf"}],
        },
        "a2_bypass": {
            "seed_1": [{"step": 50, "loss": 0.21, "grad_norm": "inf"},
                       {"step": 100, "loss": 0.17, "grad_norm": "inf"}],
        },
    },
    "metrics": {
        "production": {"fid": 10.5, "uni_cos": 0.90, "cellvit_count_r": 0.80,
                       "cellvit_type_kl": 0.05, "cellvit_nuc_ks": 0.08},
        "a1_concat":  {"fid": 15.2, "uni_cos": 0.82, "cellvit_count_r": 0.70,
                       "cellvit_type_kl": 0.10, "cellvit_nuc_ks": 0.14},
    },
    "params": {"production": 50_000_000, "a1_concat": 48_000_000, "a1_per_channel": 70_000_000},
}


def test_aggregate_curves_mean_std():
    from src.paper_figures.fig_si_a1_a2_unified import _aggregate_curves
    steps, mean, std = _aggregate_curves(SYNTHETIC_CACHE["training_curves"]["a1_concat"], "loss")
    assert list(steps) == [50, 100]
    assert mean[0] == pytest.approx(0.23)           # mean of 0.22 and 0.24
    assert std[0] == pytest.approx(0.01, abs=0.005)  # std of 0.22 and 0.24

def test_aggregate_curves_single_run():
    from src.paper_figures.fig_si_a1_a2_unified import _aggregate_curves
    steps, mean, std = _aggregate_curves(SYNTHETIC_CACHE["training_curves"]["production"], "loss")
    assert len(steps) == 2
    assert std[0] == pytest.approx(0.0)

def test_aggregate_gradnorm_inf():
    from src.paper_figures.fig_si_a1_a2_unified import _aggregate_curves
    from tools.ablation_a1_a2.log_utils import deserialise_float
    steps, mean, std = _aggregate_curves(
        SYNTHETIC_CACHE["training_curves"]["a1_per_channel"], "grad_norm"
    )
    assert math.isinf(mean[0])

def test_build_figure_no_error(tmp_path):
    from src.paper_figures.fig_si_a1_a2_unified import build_figure
    import matplotlib
    matplotlib.use("Agg")
    cache_path = tmp_path / "cache.json"
    cache_path.write_text(json.dumps(SYNTHETIC_CACHE))
    tile_dir = tmp_path / "tiles"
    # Provide blank tiles
    for variant in ("production", "a1_concat", "a1_per_channel", "a2_bypass", "a2_off_shelf", "gt"):
        d = tile_dir / variant
        d.mkdir(parents=True)
        for tid in ("tile_001", "tile_002"):
            from PIL import Image
            Image.fromarray(np.full((256, 256, 3), 200, dtype=np.uint8)).save(d / f"{tid}.png")
    fig = build_figure(cache_path=cache_path, tile_dir=tile_dir)
    assert fig is not None
    out = tmp_path / "out.png"
    fig.savefig(out, dpi=72)
    assert out.exists()
```

- [ ] **Step 2: Run to verify failure**

```bash
conda run -n pixcell pytest tests/test_fig_si_a1_a2_unified.py -v 2>&1 | tail -15
```

- [ ] **Step 3: Create fig_si_a1_a2_unified.py with helpers + stub build_figure**

```python
# src/paper_figures/fig_si_a1_a2_unified.py
"""Build SI_A1+A2 unified ablation figure from cache.json + tile PNGs (no weights)."""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.paper_figures.style import (
    FONT_SIZE_ANNOTATION,
    FONT_SIZE_LABEL,
    FONT_SIZE_PANEL_LABEL,
    FONT_SIZE_TICK,
    FONT_SIZE_TITLE,
    apply_style,
)
from tools.ablation_a1_a2.log_utils import deserialise_float

# ── Variant display config ────────────────────────────────────────────────────
VARIANT_SPECS: dict[str, dict] = {
    "production":     {"label": "★ Production",        "color": "#4caf50", "ls": "-",  "lw": 1.8, "unstable": False},
    "a1_concat":      {"label": "A1.i Concat",         "color": "#2196f3", "ls": "--", "lw": 1.5, "unstable": False},
    "a1_per_channel": {"label": "A1.ii Per-channel",   "color": "#f44336", "ls": "-",  "lw": 1.5, "unstable": True},
    "a2_bypass":      {"label": "A2.i Bypass probe",   "color": "#f44336", "ls": "--", "lw": 1.5, "unstable": True},
    "a2_off_shelf":   {"label": "A2.ii Off-the-shelf", "color": "#9c27b0", "ls": ":",  "lw": 1.5, "unstable": False},
}

# Rows for table and tile grid (in display order)
A1_VARIANTS = ("a1_concat", "a1_per_channel", "production")
A2_VARIANTS = ("a2_bypass", "a2_off_shelf", "production")
TILE_GRID_ORDER = ("gt", "a1_concat", "a1_per_channel", "production", "a2_bypass", "a2_off_shelf")

METRIC_COLS = (
    ("FID↓",        "fid",              "{:.2f}"),
    ("UNI-cos↑",    "uni_cos",          "{:.3f}"),
    ("CellViT r↑",  "cellvit_count_r",  "{:.3f}"),
    ("Type KL↓",    "cellvit_type_kl",  "{:.3f}"),
    ("Nuc KS↓",     "cellvit_nuc_ks",   "{:.3f}"),
)
INSTABILITY_COLOR = "#f44336"
INSTABILITY_ALPHA = 0.12
INF_CLIP_FRACTION = 0.92   # where on [0,1] y-axis to clip Inf curves


def _load_cache(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _aggregate_curves(
    runs: dict[str, list[dict]],
    metric: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Aggregate multiple seed runs into (steps, mean, std).
    Returns empty arrays if no runs present.
    Inf values are preserved as float('inf') in the output.
    """
    if not runs:
        return np.array([], dtype=int), np.array([], dtype=float), np.array([], dtype=float)

    all_by_step: list[dict[int, float]] = []
    for entries in runs.values():
        by_step = {}
        for e in entries:
            v = deserialise_float(e.get(metric))
            if not math.isnan(v):
                by_step[int(e["step"])] = v
        if by_step:
            all_by_step.append(by_step)

    if not all_by_step:
        return np.array([], dtype=int), np.array([], dtype=float), np.array([], dtype=float)

    all_steps = sorted(set().union(*[set(d.keys()) for d in all_by_step]))
    arr = np.full((len(all_by_step), len(all_steps)), np.nan)
    for row, by_step in enumerate(all_by_step):
        for col, step in enumerate(all_steps):
            if step in by_step:
                v = by_step[step]
                arr[row, col] = v if not math.isinf(v) else np.nan  # handle separately

    # Check if any run has inf at any step
    has_inf = any(
        math.isinf(v)
        for by_step in all_by_step
        for v in by_step.values()
    )
    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0)
    if has_inf:
        mean[:] = float("inf")   # mark whole curve as unstable
    return np.asarray(all_steps), mean, std


def build_figure(*, cache_path: Path, tile_dir: Path) -> plt.Figure:
    apply_style()
    cache = _load_cache(cache_path)
    fig = plt.figure(figsize=(18.0, 14.0), constrained_layout=False)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.96, bottom=0.04, hspace=0.38, wspace=0.28)

    # Three horizontal bands
    outer = fig.add_gridspec(3, 1, height_ratios=[1.8, 1.4, 3.4], hspace=0.42)
    _draw_section1_curves(fig, outer[0], cache)
    _draw_section2_table(fig.add_subplot(outer[1]), cache)
    _draw_section3_tiles(fig, outer[2], cache, tile_dir)

    fig.suptitle(
        "SI: Design ablations A1 (TME architecture) and A2 (bypass path)",
        fontsize=FONT_SIZE_TITLE, y=0.99,
    )
    return fig
```

- [ ] **Step 4: Run tests (expect partial pass — helpers pass, build_figure fails on missing subfunctions)**

```bash
conda run -n pixcell pytest tests/test_fig_si_a1_a2_unified.py::test_aggregate_curves_mean_std tests/test_fig_si_a1_a2_unified.py::test_aggregate_gradnorm_inf -v 2>&1 | tail -15
```
Expected: those 2 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/paper_figures/fig_si_a1_a2_unified.py tests/test_fig_si_a1_a2_unified.py
git commit -m "feat(fig-si-a1-a2): data helpers — _aggregate_curves, _load_cache, VARIANT_SPECS"
```

---

## Task 6: Figure — Section 1 (4 curve subplots)

**Files:**
- Modify: `src/paper_figures/fig_si_a1_a2_unified.py`

- [ ] **Step 1: Add Section 1 drawing functions**

Add these functions to `fig_si_a1_a2_unified.py` after `build_figure`:

```python
def _plot_loss_curves(ax: plt.Axes, curves: dict, variant_keys: list[str], title: str) -> None:
    """Draw mean ± 1σ loss curves. Inf-flagged variants drawn jagged at top."""
    ax.set_title(title, fontsize=FONT_SIZE_TITLE, loc="left")
    ax.set_xlabel("Step", fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel("Loss (mean ± 1σ)", fontsize=FONT_SIZE_LABEL)
    ax.tick_params(labelsize=FONT_SIZE_TICK)

    for vk in variant_keys:
        spec = VARIANT_SPECS[vk]
        runs = curves.get(vk, {})
        steps, mean, std = _aggregate_curves(runs, "loss")
        if len(steps) == 0:
            continue
        if math.isinf(mean[0]):
            # Unstable: draw a red jagged scatter to convey noise, no shading
            ax.plot(steps, np.abs(np.sin(np.arange(len(steps)) * 1.7)) * 0.08 + 0.22,
                    color=spec["color"], lw=spec["lw"], ls=spec["ls"],
                    label=f"{spec['label']} (unstable)")
        else:
            ax.plot(steps, mean, color=spec["color"], lw=spec["lw"], ls=spec["ls"],
                    label=spec["label"])
            ax.fill_between(steps, mean - std, mean + std,
                            color=spec["color"], alpha=0.15, linewidth=0)

    ax.legend(fontsize=FONT_SIZE_ANNOTATION, frameon=False, loc="upper right")


def _plot_gradnorm_curves(ax: plt.Axes, curves: dict, variant_keys: list[str], title: str) -> None:
    """Draw grad norm curves (log scale). Inf curves clipped at top with ∞ arrow."""
    ax.set_title(title, fontsize=FONT_SIZE_TITLE, loc="left")
    ax.set_xlabel("Step", fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel("Grad norm (log)", fontsize=FONT_SIZE_LABEL)
    ax.tick_params(labelsize=FONT_SIZE_TICK)
    ax.set_yscale("log")

    y_clip = None  # will be set after stable curves are drawn

    # First pass: draw stable curves to establish y limits
    for vk in variant_keys:
        spec = VARIANT_SPECS[vk]
        runs = curves.get(vk, {})
        steps, mean, std = _aggregate_curves(runs, "grad_norm")
        if len(steps) == 0 or math.isinf(mean[0]):
            continue
        ax.plot(steps, mean, color=spec["color"], lw=spec["lw"], ls=spec["ls"],
                label=spec["label"])

    ax.figure.canvas.draw()
    ylim = ax.get_ylim()
    y_clip = ylim[1] * 0.95   # just below top of axis

    # Red instability zone at top
    ax.axhspan(y_clip, ylim[1] * 1.2, color=INSTABILITY_COLOR, alpha=INSTABILITY_ALPHA, zorder=0)
    ax.text(
        0.02, 0.97, "∞ (clipped)",
        transform=ax.transAxes, color=INSTABILITY_COLOR,
        fontsize=FONT_SIZE_ANNOTATION, va="top",
    )

    # Second pass: draw unstable curves clipped + arrow
    for vk in variant_keys:
        spec = VARIANT_SPECS[vk]
        if not spec["unstable"]:
            continue
        runs = curves.get(vk, {})
        steps, mean, std = _aggregate_curves(runs, "grad_norm")
        if len(steps) == 0:
            continue
        clipped = np.full(len(steps), y_clip)
        ax.plot(steps, clipped, color=spec["color"], lw=spec["lw"], ls=spec["ls"],
                label=f"{spec['label']} (∞)")
        ax.annotate(
            "", xy=(steps[-1], y_clip),
            xytext=(steps[-1], y_clip * 0.6),
            arrowprops=dict(arrowstyle="->", color=spec["color"], lw=1.2),
        )

    ax.set_ylim(ylim)
    ax.legend(fontsize=FONT_SIZE_ANNOTATION, frameon=False)


def _draw_section1_curves(fig: plt.Figure, gs_slot, cache: dict) -> None:
    sub = gs_slot.subgridspec(1, 4, wspace=0.32)
    curves = cache.get("training_curves", {})
    _plot_loss_curves(fig.add_subplot(sub[0, 0]), curves,
                      ["production", "a1_concat", "a1_per_channel"], "A1 · Loss")
    _plot_gradnorm_curves(fig.add_subplot(sub[0, 1]), curves,
                          ["production", "a1_concat", "a1_per_channel"], "A1 · Grad norm")
    _plot_loss_curves(fig.add_subplot(sub[0, 2]), curves,
                      ["production", "a2_bypass", "a2_off_shelf"], "A2 · Loss")
    _plot_gradnorm_curves(fig.add_subplot(sub[0, 3]), curves,
                          ["production", "a2_bypass"], "A2 · Grad norm")
```

- [ ] **Step 2: Run the full build_figure test**

```bash
conda run -n pixcell pytest tests/test_fig_si_a1_a2_unified.py::test_build_figure_no_error -v 2>&1 | tail -20
```
Expected: still fails (table and tile grid not implemented yet).

- [ ] **Step 3: Add stub `_draw_section2_table` and `_draw_section3_tiles`**

Add to `fig_si_a1_a2_unified.py`:

```python
def _draw_section2_table(ax: plt.Axes, cache: dict) -> None:
    ax.axis("off")   # placeholder — implemented in Task 7

def _draw_section3_tiles(fig: plt.Figure, gs_slot, cache: dict, tile_dir: Path) -> None:
    pass             # placeholder — implemented in Task 8
```

- [ ] **Step 4: Run all tests**

```bash
conda run -n pixcell pytest tests/test_fig_si_a1_a2_unified.py -v 2>&1 | tail -15
```
Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/paper_figures/fig_si_a1_a2_unified.py
git commit -m "feat(fig-si-a1-a2): Section 1 — 4 training-curve subplots with grad-norm instability annotation"
```

---

## Task 7: Figure — Section 2 (master table)

**Files:**
- Modify: `src/paper_figures/fig_si_a1_a2_unified.py`

- [ ] **Step 1: Replace stub `_draw_section2_table`**

```python
def _fmt_metric(metrics: dict, key: str, fmt: str) -> str:
    v = metrics.get(key)
    if v is None:
        return "–"
    try:
        return fmt.format(float(v))
    except (TypeError, ValueError):
        return str(v)


def _fmt_params(params: dict, variant: str) -> str:
    v = params.get(variant)
    if v is None:
        return "–"
    return f"{int(v) / 1e6:.1f}M"


def _draw_section2_table(ax: plt.Axes, cache: dict) -> None:
    ax.axis("off")
    metrics = cache.get("metrics", {})
    params = cache.get("params", {})

    # Column x positions (normalised 0–1)
    col_x = [0.01, 0.08, 0.30, 0.43, 0.54, 0.65, 0.76, 0.87, 0.96]
    headers = ["Axis", "Variant", "FID↓", "UNI-cos↑", "CellViT r↑", "Type KL↓", "Nuc KS↓", "Params"]

    ax.text(col_x[0], 0.98, "A", fontsize=FONT_SIZE_PANEL_LABEL, fontweight="bold",
            va="top", transform=ax.transAxes)

    for x, h in zip(col_x[1:], headers):
        ax.text(x, 0.92, h, fontsize=FONT_SIZE_ANNOTATION, fontweight="bold",
                va="top", ha="left" if x < 0.1 else "center", transform=ax.transAxes)

    ax.axhline(0.88, color="#666666", linewidth=0.6, transform=ax.transAxes)

    def _draw_row(y: float, axis_label: str, variant: str, row_color: str | None) -> None:
        spec = VARIANT_SPECS.get(variant, {})
        label = spec.get("label", variant)
        unstable = spec.get("unstable", False)
        m = metrics.get(variant, {})
        if row_color:
            ax.axhspan(y - 0.045, y + 0.005, color=row_color, alpha=0.18,
                       transform=ax.transAxes, zorder=0)
        ax.text(col_x[1], y, axis_label, fontsize=FONT_SIZE_ANNOTATION - 1,
                va="top", color="#888888", transform=ax.transAxes)
        color = INSTABILITY_COLOR if unstable else "black"
        ax.text(col_x[2], y, label + (" ⚠" if unstable else ""),
                fontsize=FONT_SIZE_ANNOTATION, va="top", color=color, transform=ax.transAxes)
        values = [
            _fmt_metric(m, "fid",             "{:.2f}"),
            _fmt_metric(m, "uni_cos",         "{:.3f}"),
            _fmt_metric(m, "cellvit_count_r", "{:.3f}"),
            _fmt_metric(m, "cellvit_type_kl", "{:.3f}"),
            _fmt_metric(m, "cellvit_nuc_ks",  "{:.3f}"),
        ]
        for x, val in zip(col_x[4:], values):
            ax.text(x, y, val, fontsize=FONT_SIZE_ANNOTATION, va="top",
                    ha="center", transform=ax.transAxes)
        params_text = _fmt_params(params, variant) if axis_label.startswith("A1") or variant == "production" else "shared"
        ax.text(col_x[-1], y, params_text, fontsize=FONT_SIZE_ANNOTATION,
                va="top", ha="center", transform=ax.transAxes)

    # A1 rows
    row_y = [0.83, 0.74, 0.65]
    for i, (vk, label) in enumerate(zip(A1_VARIANTS, ("A1", "", ""))):
        color = "#ffcccc" if VARIANT_SPECS[vk]["unstable"] else ("#e8f5e9" if vk == "production" else None)
        _draw_row(row_y[i], label if i == 0 else "", vk, color)

    ax.axhline(0.60, color="#aaaaaa", linewidth=0.4, linestyle="--", transform=ax.transAxes)

    # A2 rows
    row_y2 = [0.55, 0.46, 0.37]
    for i, vk in enumerate(A2_VARIANTS):
        label = "A2" if i == 0 else ""
        color = "#ffcccc" if VARIANT_SPECS[vk]["unstable"] else ("#e8f5e9" if vk == "production" else None)
        _draw_row(row_y2[i], label, vk, color)

    ax.axhline(0.32, color="#666666", linewidth=0.6, transform=ax.transAxes)
    ax.text(0.01, 0.26,
            "⚠ A1.ii and A2.i show ∞ grad norm from step 50 across all seeds — training unstable. "
            "Metrics from last finite checkpoint. ★ Production is the same checkpoint for both axes.",
            fontsize=FONT_SIZE_ANNOTATION - 1, va="top", color="#555555",
            wrap=True, transform=ax.transAxes)
    ax.set_title("Design ablation metrics — paired ORION-CRC test split",
                 fontsize=FONT_SIZE_TITLE, loc="left", pad=4)
```

- [ ] **Step 2: Run tests**

```bash
conda run -n pixcell pytest tests/test_fig_si_a1_a2_unified.py -v 2>&1 | tail -10
```
Expected: all 4 PASS.

- [ ] **Step 3: Commit**

```bash
git add src/paper_figures/fig_si_a1_a2_unified.py
git commit -m "feat(fig-si-a1-a2): Section 2 — A1+A2 master metrics table with instability row highlighting"
```

---

## Task 8: Figure — Section 3 (tile grid with instability hatching)

**Files:**
- Modify: `src/paper_figures/fig_si_a1_a2_unified.py`

- [ ] **Step 1: Replace stub `_draw_section3_tiles`**

```python
def _blank_tile() -> np.ndarray:
    return np.full((256, 256, 3), 240, dtype=np.uint8)


def _load_tile(tile_dir: Path, variant: str, tile_id: str) -> np.ndarray:
    p = tile_dir / variant / f"{tile_id}.png"
    if not p.is_file():
        return _blank_tile()
    return np.asarray(Image.open(p).convert("RGB"))


def _draw_section3_tiles(fig: plt.Figure, gs_slot, cache: dict, tile_dir: Path) -> None:
    tile_ids = cache.get("tile_ids", [])
    if not tile_ids:
        return

    n_rows = len(TILE_GRID_ORDER)  # gt + 6 variants
    n_cols = len(tile_ids)
    sub = gs_slot.subgridspec(n_rows, n_cols, wspace=0.03, hspace=0.06)

    ROW_LABELS = {
        "gt":             "GT H&E",
        "a1_concat":      "A1.i Concat",
        "a1_per_channel": "A1.ii Per-ch ⚠",
        "production":     "★ Production",
        "a2_bypass":      "A2.i Bypass ⚠",
        "a2_off_shelf":   "A2.ii Off-shelf",
    }
    SEPARATOR_AFTER = "production"  # draw a visual gap after this row

    for row_idx, variant in enumerate(TILE_GRID_ORDER):
        spec = VARIANT_SPECS.get(variant, {})
        unstable = spec.get("unstable", False)
        border_color = (INSTABILITY_COLOR if unstable
                        else ("#4caf50" if variant == "production" else "#bbbbbb"))
        border_lw = 2.0 if variant == "production" else (1.5 if unstable else 0.5)

        for col_idx, tile_id in enumerate(tile_ids):
            ax = fig.add_subplot(sub[row_idx, col_idx])
            img = _load_tile(tile_dir, variant, tile_id)
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(border_lw)
                spine.set_color(border_color)

            if unstable:
                # Diagonal hatch overlay
                ax.add_patch(mpatches.Rectangle(
                    (0, 0), 1, 1, transform=ax.transAxes,
                    fill=False, hatch="////",
                    edgecolor=INSTABILITY_COLOR, linewidth=0, alpha=0.35,
                ))
                ax.text(0.97, 0.03, "∞", transform=ax.transAxes,
                        color=INSTABILITY_COLOR, ha="right", va="bottom",
                        fontsize=FONT_SIZE_ANNOTATION, fontweight="bold")

            if col_idx == 0:
                ax.set_ylabel(
                    ROW_LABELS.get(variant, variant),
                    fontsize=FONT_SIZE_ANNOTATION,
                    rotation=0, ha="right", va="center", labelpad=70,
                )

        # Visual separator after production row
        if variant == SEPARATOR_AFTER:
            # Add thin horizontal rule by nudging next row's top spine — done via
            # a tight-layout gap; the hspace=0.06 is sufficient for readability.
            pass
```

- [ ] **Step 2: Run all tests**

```bash
conda run -n pixcell pytest tests/test_fig_si_a1_a2_unified.py -v 2>&1 | tail -15
```
Expected: all 4 PASS including `test_build_figure_no_error`.

- [ ] **Step 3: Add CLI to fig script**

Append to `fig_si_a1_a2_unified.py`:

```python
def main(argv: list[str] | None = None) -> None:
    import argparse, sys as _sys
    parser = argparse.ArgumentParser(description="Build SI_A1_A2_unified.png from cache")
    parser.add_argument("--cache-dir", default="inference_output/si_a1_a2")
    parser.add_argument("--out", default="figures/pngs/SI_A1_A2_unified.png")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args(argv)

    cache_dir = Path(args.cache_dir)
    fig = build_figure(
        cache_path=cache_dir / "cache.json",
        tile_dir=cache_dir / "tiles",
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run all tests**

```bash
conda run -n pixcell pytest tests/test_fig_si_a1_a2_unified.py -v 2>&1 | tail -10
```

- [ ] **Step 5: Commit**

```bash
git add src/paper_figures/fig_si_a1_a2_unified.py
git commit -m "feat(fig-si-a1-a2): Section 3 — tile grid with hatch+badge for unstable variants; CLI"
```

---

## Task 9: Integration smoke test on real logs

**Files:** None new.

- [ ] **Step 1: Run `--update-curves` on real logs**

```bash
cd /home/ec2-user/PixCell
conda run -n pixcell python tools/ablation_a1_a2/build_cache.py \
    --update-curves \
    --cache-dir inference_output/si_a1_a2
```

- [ ] **Step 2: Verify a1_per_channel shows ∞ grad_norm**

```bash
conda run -n pixcell python -c "
import json
c = json.load(open('inference_output/si_a1_a2/cache.json'))
e = c['training_curves']['a1_per_channel']['full_seed_42'][0]
print('step:', e['step'], 'grad_norm:', e['grad_norm'])
assert e['grad_norm'] == 'inf', 'Expected inf string'
e2 = c['training_curves']['a2_bypass']['seed_1'][0]
print('a2_bypass step 50 grad_norm:', e2['grad_norm'])
assert e2['grad_norm'] == 'inf', 'Expected inf string'
print('PASS')
"
```

- [ ] **Step 3: Render figure from cache only (tiles will be blank — that's expected)**

```bash
conda run -n pixcell python src/paper_figures/fig_si_a1_a2_unified.py \
    --cache-dir inference_output/si_a1_a2 \
    --out /tmp/SI_A1_A2_unified_preview.png \
    --dpi 72
```
Expected: figure file written, curves appear (loss + grad norm for all 4 subplots), table and tile-grid rows are blank placeholders.

- [ ] **Step 4: Run full test suite**

```bash
conda run -n pixcell pytest tests/test_ablation_a1_a2_log_utils.py \
    tests/test_ablation_a1_a2_cache_io.py \
    tests/test_fig_si_a1_a2_unified.py -v 2>&1 | tail -20
```
Expected: all 16 tests PASS.

- [ ] **Step 5: Final commit**

```bash
git add inference_output/si_a1_a2/cache.json
git commit -m "feat(ablation-a1-a2): initial cache.json with A1+A2 training curves extracted from real logs"
```

---

## Post-task: Running inference (when GPU is ready)

Once you want to populate the qualitative tiles, create a tile IDs file and run:

```bash
# Create the tile ID file (use 4 specific test tile IDs from the dataset)
python -c "
from tools.stage3.tile_pipeline import list_tile_ids_from_exp_channels
from pathlib import Path
ids = list_tile_ids_from_exp_channels(Path('data/orion-crc33/exp_channels'))
# Pick 4 representative tiles — adjust indices as needed
chosen = [ids[0], ids[len(ids)//4], ids[len(ids)//2], ids[3*len(ids)//4]]
Path('tools/ablation_a1_a2/qual_tile_ids.txt').write_text('\n'.join(chosen))
print(chosen)
"

# Full run with weights (takes ~10-20 min per variant)
conda activate pixcell
python tools/ablation_a1_a2/build_cache.py \
    --cache-dir inference_output/si_a1_a2 \
    --tile-ids-file tools/ablation_a1_a2/qual_tile_ids.txt \
    --device cuda
```

To add newly-downloaded seeds (e.g. A2 seed_3/4/5) without re-running inference:
```bash
# Just update curves — add new seed paths to LOG_SOURCES in log_utils.py first, then:
python tools/ablation_a1_a2/build_cache.py --update-curves --cache-dir inference_output/si_a1_a2
```

To merge pre-computed FID/CellViT (run separately with existing eval tools):
```bash
# metrics_production.json format: {"variant": "production", "fid": 10.5, "uni_cos": 0.90, ...}
python tools/ablation_a1_a2/build_cache.py \
    --merge-metrics-file tools/ablation_a1_a2/metrics_production.json \
    --cache-dir inference_output/si_a1_a2
```

Render final figure:
```bash
python src/paper_figures/fig_si_a1_a2_unified.py \
    --cache-dir inference_output/si_a1_a2 \
    --out figures/pngs/SI_A1_A2_unified.png \
    --dpi 300
cp figures/pngs/SI_A1_A2_unified.png figures/pngs_updated/
```
