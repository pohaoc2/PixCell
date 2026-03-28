# PixCell Codebase Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove 9 dead functions from `inference_controlnet.py`, delete/refactor two batch scripts into a proper CLI tool, and add test coverage for uncovered inference paths.

**Architecture:** Four independent workstreams — dead code deletion (`inference_controlnet.py`), batch script cleanup (`run_zero_out_mask_batch.py` deleted, `run_stage3_full.py` → `tools/run_evaluation.py`), and two new test files validating existing zero_mask_latent behavior.

**Tech Stack:** Python 3.12, pytest, unittest.mock, PIL, NumPy, PyTorch (CPU only in tests), argparse.

---

## File Map

| Action | File |
|---|---|
| Modify | `train_scripts/inference_controlnet.py` — delete 9 dead functions |
| Delete | `run_zero_out_mask_batch.py` |
| Delete | `run_stage3_full.py` |
| Create | `tools/run_evaluation.py` |
| Create | `tests/test_inference_core.py` |
| Create | `tests/test_stage3_tile_pipeline.py` |

---

## Task 1: Remove dead functions from `train_scripts/inference_controlnet.py`

**Files:**
- Modify: `train_scripts/inference_controlnet.py`

Nine functions have zero callers outside this file. The functions to keep are: `null_uni_embed`, `encode_ctrl_mask_latent`, `load_pixcell_controlnet_model_from_checkpoint`, `load_controlnet_model_from_checkpoint`, `load_vae`, `denoise`.

- [ ] **Step 1: Confirm zero external callers**

```bash
grep -rn \
  "load_base_model_checkpoint\|save_keys_comparison_controlnet\|test_load_controlnet\|initialize_pixcell_controlnet_model\|initialize_controlnet_model\|prepare_controlnet_input\|decode_latents" \
  /home/ec2-user/PixCell --include="*.py" | \
  grep -v __pycache__ | \
  grep -v "train_scripts/inference_controlnet.py"
```

Expected: zero lines of output. If any appear, stop and investigate before proceeding.

Also check the two non-`_from_checkpoint` loaders:
```bash
grep -rn "load_pixcell_controlnet_model\b\|load_controlnet_model\b" \
  /home/ec2-user/PixCell --include="*.py" | \
  grep -v __pycache__ | \
  grep -v "train_scripts/inference_controlnet.py"
```

Expected: zero lines.

- [ ] **Step 2: Delete `load_controlnet_model` (lines 58–71)**

In `train_scripts/inference_controlnet.py`, delete the entire function body. The old string to remove:

```python
def load_controlnet_model(module_name, file_path, checkpoints_folder, device="cuda"):
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    controlnet_mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = controlnet_mod
    spec.loader.exec_module(controlnet_mod)
    PixCellControlNet = controlnet_mod.PixCellControlNet
    model = PixCellControlNet.from_pretrained(checkpoints_folder)
    model.to(device)
    model.eval()
    return model


```

Replace with nothing (empty string).

- [ ] **Step 3: Delete `load_base_model_checkpoint` (lines 110–123)**

```python
def load_base_model_checkpoint(base_model, checkpoint_path):
    finetuned = torch.load(checkpoint_path, map_location="cpu")
    finetuned_sd = finetuned["state_dict"] if "state_dict" in finetuned else finetuned

    # NO remapping needed — both checkpoint and model use 'blocks.' naming
    missing, unexpected = base_model.load_state_dict(finetuned_sd, strict=False)

    print(f"Total missing: {len(missing)}, Total unexpected: {len(unexpected)}")
    if missing:
        print(f"Missing examples: {missing[:3]}")
    if unexpected:
        print(f"Unexpected examples: {unexpected[:3]}")
    return base_model


```

Replace with nothing.

- [ ] **Step 4: Delete `save_keys_comparison_controlnet` (lines 161–204)**

```python
def save_keys_comparison_controlnet(controlnet, state_file_path, device="cuda"):
    from safetensors.torch import load_file
    import pandas as pd

    sd = load_file(state_file_path)
    controlnet_sd = controlnet.state_dict()
    sd_keys = list(sd.keys())
    controlnet_keys = list(controlnet_sd.keys())

    # Check counts first
    print(f"Pretrained keys: {len(sd_keys)}")
    print(f"ControlNet keys: {len(controlnet_keys)}")

    # Build comparison df by position
    max_len = max(len(sd_keys), len(controlnet_keys))
    df = pd.DataFrame(
        {
            "sd_key": sd_keys + [None] * (max_len - len(sd_keys)),
            "controlnet_key": controlnet_keys + [None] * (max_len - len(controlnet_keys)),
            "sd_shape": [
                str(sd[k].shape) if k else None for k in sd_keys + [None] * (max_len - len(sd_keys))
            ],
            "controlnet_shape": [
                str(controlnet_sd[k].shape) if k else None
                for k in controlnet_keys + [None] * (max_len - len(controlnet_keys))
            ],
            "shape_match": [
                (
                    str(sd[sd_keys[i]].shape) == str(controlnet_sd[controlnet_keys[i]].shape)
                    if i < len(sd_keys) and i < len(controlnet_keys)
                    else False
                )
                for i in range(max_len)
            ],
        }
    )

    df.to_csv("keys.csv", index=False)

    # Quick summary
    print(f"Shape mismatches: {(~df['shape_match']).sum()}")
    print(f"Missing in sd: {df['sd_key'].isna().sum()}")
    print(f"Missing in controlnet: {df['controlnet_key'].isna().sum()}")


```

Replace with nothing.

- [ ] **Step 5: Delete `test_load_controlnet` (lines 206–220)**

```python
def test_load_controlnet(controlnet, state_file_path, device="cuda"):
    mapped = torch.load(state_file_path)
    controlnet_sd = controlnet.state_dict()
    not_loaded = [k for k in controlnet_sd.keys() if k not in mapped]
    print(f"Not loaded ({len(not_loaded)} keys):")
    for k in not_loaded:
        print(f"  {k:60s} {tuple(controlnet_sd[k].shape)}")
    missing, unexpected = controlnet.load_state_dict(mapped, strict=False)

    print(f"MISSING (random init): {len(missing)}")
    print(f"UNEXPECTED (dropped): {len(unexpected)}")
    controlnet.to(device)
    controlnet.eval()
    return controlnet


```

Replace with nothing.

- [ ] **Step 6: Delete `load_pixcell_controlnet_model` (lines 222–238)**

```python
def load_pixcell_controlnet_model(module_name, file_path, checkpoints_folder, device="cuda"):
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    pixcell_mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = pixcell_mod
    spec.loader.exec_module(pixcell_mod)
    PixCellTransformer2DModelControlNet = pixcell_mod.PixCellTransformer2DModelControlNet
    model = PixCellTransformer2DModelControlNet.from_pretrained(
        checkpoints_folder,
        # subfolder="transformer"
    )
    model.to(device)
    model.eval()
    return model


```

Replace with nothing.

- [ ] **Step 7: Delete `initialize_pixcell_controlnet_model` (lines 254–275)**

```python
def initialize_pixcell_controlnet_model(module_name, file_path, checkpoints_folder, device="cuda"):
    import importlib.util
    import sys

    # Standard dynamic import logic
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    pixcell_mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = pixcell_mod
    spec.loader.exec_module(pixcell_mod)

    PixCellTransformer2DModelControlNet = pixcell_mod.PixCellTransformer2DModelControlNet

    # 1. Load only the configuration dictionary
    config = PixCellTransformer2DModelControlNet.load_config(checkpoints_folder)

    # 2. Initialize the model with random weights based on that config
    model = PixCellTransformer2DModelControlNet.from_config(config)

    model.to(device)
    model.eval()
    return model


```

Replace with nothing.

- [ ] **Step 8: Delete `initialize_controlnet_model` (lines 277–291)**

```python
def initialize_controlnet_model(module_name, file_path, checkpoints_folder, device="cuda"):
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    controlnet_mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = controlnet_mod
    spec.loader.exec_module(controlnet_mod)
    PixCellControlNet = controlnet_mod.PixCellControlNet
    config = PixCellControlNet.load_config(checkpoints_folder)
    model = PixCellControlNet.from_config(config)
    model.to(device)
    model.eval()
    return model


```

Replace with nothing.

- [ ] **Step 9: Delete `prepare_controlnet_input` (lines 387–421)**

```python
def prepare_controlnet_input(idx):

    latent_shape = (1, 16, 32, 32)
    latents = torch.randn(latent_shape, device=device, dtype=torch.float32).to(device)
    latents = latents * scheduler.init_noise_sigma

    uni_embeds = torch.from_numpy(np.load(f"../features_consep/sample_{idx}_uni.npy"))
    # uni_embeds = torch.from_numpy(np.load(f"uni_emb_control.npy"))
    uni_embeds = torch.from_numpy(np.load(f"../data/features_tcga_3660/0_{idx}_uni.npy"))
    uni_embeds = uni_embeds.view(1, 1, 1, 1536).to(device)
    mask_path = "../test_mask.png"
    mask_path = f"../consep_masks/sample_{idx}_mask.png"
    # mask_path = f"../data/tcga_3660_masks/0_{idx}_mask.png"
    controlnet_input = np.asarray(Image.open(mask_path).convert("RGB").resize((256, 256)))
    # controlnet_input = Image.open(mask_path).convert('L')
    # controlnet_input = np.array(controlnet_input)
    # controlnet_input = torch.from_numpy(controlnet_input > 0).float()
    # resize to 256x256
    # import torchvision.transforms as T
    # controlnet_input = T.Resize((256, 256))(controlnet_input)
    # controlnet_input = np.array(Image.open(f"../masks/sample_{idx}_mask.png"))
    # controlnet_input = np.repeat(controlnet_input[..., None], 3, axis=-1)

    controlnet_input_torch = torch.from_numpy(controlnet_input.copy() / 255.0).float().to(device)
    controlnet_input_torch = controlnet_input_torch.permute(2, 0, 1).unsqueeze(0)
    controlnet_input_torch = 2 * (controlnet_input_torch - 0.5)
    vae_scale = vae.config.scaling_factor
    vae_shift = getattr(vae.config, "shift_factor", 0)
    controlnet_input_latent = vae.encode(controlnet_input_torch).latent_dist.mean
    controlnet_input_latent = (controlnet_input_latent - vae_shift) * vae_scale
    # controlnet_input_latent, _ = torch.from_numpy(np.load(f"../features_consep_masks/sample_{idx}_mask_sd3_vae.npy")).chunk(2)
    # controlnet_input_latent = (controlnet_input_latent-vae_shift)*vae_scale

    return latents, uni_embeds, controlnet_input_latent, controlnet_input


```

Replace with nothing.

- [ ] **Step 10: Delete `decode_latents` (lines 423–693, to end of file)**

```python
def decode_latents(vae, latents, hist_image, mask_image, save_path):
```

Delete from this line to the end of file (everything after `denoise`).

- [ ] **Step 11: Run existing tests to confirm no regressions**

```bash
cd /home/ec2-user/PixCell && python -m pytest tests/ -x -q 2>&1 | tail -20
```

Expected: all tests pass (same count as before). If any test imports a now-deleted function, fix the test import first.

- [ ] **Step 12: Commit**

```bash
cd /home/ec2-user/PixCell
git add train_scripts/inference_controlnet.py
git commit -m "refactor: remove 9 dead functions from inference_controlnet.py"
```

---

## Task 2: Delete `run_zero_out_mask_batch.py`

**Files:**
- Delete: `run_zero_out_mask_batch.py`

`tools/generate_orion_paired_unpaired_batch.py` covers the same functionality with full argparse. This file is a hardcoded one-off with no unique logic.

- [ ] **Step 1: Delete the file**

```bash
cd /home/ec2-user/PixCell && git rm run_zero_out_mask_batch.py
```

- [ ] **Step 2: Commit**

```bash
git commit -m "chore: remove run_zero_out_mask_batch.py (superseded by tools/generate_orion_paired_unpaired_batch.py)"
```

---

## Task 3: Create `tools/run_evaluation.py`

**Files:**
- Create: `tools/run_evaluation.py`
- Delete: `run_stage3_full.py`

Refactor `run_stage3_full.py` into a proper CLI tool. All hardcoded path/parameter constants become argparse arguments. Internal tile loops, cosine-similarity metric, and summary JSON are preserved verbatim.

- [ ] **Step 1: Create `tools/run_evaluation.py`**

```python
"""
run_evaluation.py — Stage 3 inference + validation + visualizations.

Uses experimental channel data (ORION-CRC33) as inference input:
  - N inference tiles (paired: channels → generate H&E, compare vs paired UNI)
  - N validation tiles (held-out unpaired: same metric, different tiles)

Generates for each set:
  - Generated H&E images
  - UNI cosine similarity vs. ground-truth exp H&E features
  - Attention heatmaps (per TME group)
  - Residual magnitude maps (per TME group)
  - Ablation grid (progressive group addition)
  - Summary metrics JSON

Usage::

    python tools/run_evaluation.py \\
        --config           configs/config_controlnet_exp.py \\
        --checkpoint-dir   checkpoints/pixcell_controlnet_exp/checkpoints \\
        --data-root        data/orion-crc33 \\
        --output-dir       inference_output/evaluation
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from diffusers import DDPMScheduler
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from tools.stage3_figures import (
    save_enhanced_ablation_grid,
    save_enhanced_attention_figure,
    save_enhanced_residual_figure,
    save_overview_figure,
)
from tools.stage3_tile_pipeline import (
    generate_ablation_images,
    generate_tile,
    load_all_models,
)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b))


def get_uni_extractor(config, device: str):
    from pipeline.extract_features import UNI2hExtractor
    uni_model_path = getattr(config, "uni_model_path", "./pretrained_models/uni-2h")
    return UNI2hExtractor(model_path=uni_model_path, device=device)


def main():
    parser = argparse.ArgumentParser(
        description="Stage 3 inference + validation evaluation"
    )
    parser.add_argument(
        "--config",
        default="configs/config_controlnet_exp.py",
        help="Path to training config (default: configs/config_controlnet_exp.py)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="checkpoints/pixcell_controlnet_exp/checkpoints",
        help="Parent dir containing step_XXXXXXX checkpoint subdirs",
    )
    parser.add_argument(
        "--data-root",
        default="data/orion-crc33",
        help="Root of ORION-CRC33 dataset (default: data/orion-crc33)",
    )
    parser.add_argument(
        "--output-dir",
        default="inference_output/evaluation",
        help="Where to write outputs (default: inference_output/evaluation)",
    )
    parser.add_argument("--n-inference", type=int, default=20)
    parser.add_argument("--n-validation", type=int, default=20)
    parser.add_argument("--n-vis-tiles", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--guidance-scale", type=float, default=2.5)
    parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    CONFIG_PATH    = ROOT / args.config
    CKPT_DIR       = ROOT / args.checkpoint_dir
    EXP_ROOT       = ROOT / args.data_root
    EXP_CH_DIR     = EXP_ROOT / "exp_channels"
    FEAT_DIR       = EXP_ROOT / "features"
    HE_DIR         = EXP_ROOT / "he"
    OUT_DIR        = ROOT / args.output_dir
    N_INFERENCE    = args.n_inference
    N_VALIDATION   = args.n_validation
    N_VIS_TILES    = args.n_vis_tiles
    GUIDANCE_SCALE = args.guidance_scale
    NUM_STEPS      = args.num_steps
    SEED           = args.seed
    DEVICE         = args.device
    MASK_CHANNEL   = "cell_masks"

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    from diffusion.utils.misc import read_config
    from train_scripts.inference_controlnet import null_uni_embed

    print(f"Device: {DEVICE}")
    print(f"Output: {OUT_DIR}")

    config = read_config(str(CONFIG_PATH))
    config._filename = str(CONFIG_PATH)

    mask_dir = EXP_CH_DIR / MASK_CHANNEL
    all_ids = sorted(p.stem for p in mask_dir.glob("*.png"))
    print(f"Found {len(all_ids)} tiles in exp_channels/{MASK_CHANNEL}")

    selected = random.sample(all_ids, N_INFERENCE + N_VALIDATION)
    inference_ids  = selected[:N_INFERENCE]
    validation_ids = selected[N_INFERENCE:]
    print(f"Inference tiles:  {N_INFERENCE}  | Validation tiles: {N_VALIDATION}")

    inf_dir = OUT_DIR / "inference"
    val_dir = OUT_DIR / "validation"
    vis_dir = OUT_DIR / "visualizations"
    for d in [inf_dir, val_dir, vis_dir]:
        d.mkdir(parents=True, exist_ok=True)

    models = load_all_models(config, CONFIG_PATH, CKPT_DIR, DEVICE)

    scheduler = DDPMScheduler(
        num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02,
        beta_schedule="linear", prediction_type="epsilon", clip_sample=False,
    )
    scheduler.set_timesteps(NUM_STEPS, device=DEVICE)

    null_uni = null_uni_embed(device="cpu", dtype=torch.float32)

    print("\nLoading UNI extractor for validation metrics...")
    uni_extractor = get_uni_extractor(config, DEVICE)

    # ── INFERENCE SET ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("INFERENCE SET (paired exp data — using paired UNI style)")
    print(f"{'='*60}")
    inf_cosine_sims = []

    for i, tid in enumerate(inference_ids):
        do_vis = (i < N_VIS_TILES)
        print(f"[{i+1:02d}/{N_INFERENCE}] {tid}", end="")

        exp_feat_path = FEAT_DIR / f"{tid}_uni.npy"
        if exp_feat_path.exists():
            uni_embeds = torch.from_numpy(np.load(exp_feat_path)).view(1, 1, 1, 1536)
            print(f"  [style=paired UNI]", end="")
        else:
            uni_embeds = null_uni
            print(f"  [style=null UNI (no paired feat)]", end="")

        gen_np, vis_data = generate_tile(
            tile_id=tid,
            models=models,
            config=config,
            scheduler=scheduler,
            uni_embeds=uni_embeds,
            device=DEVICE,
            exp_channels_dir=EXP_CH_DIR,
            guidance_scale=GUIDANCE_SCALE,
            return_vis_data=do_vis,
        )

        out_path = inf_dir / f"{tid}_generated.png"
        Image.fromarray(gen_np).save(out_path)

        exp_he_path = HE_DIR / f"{tid}.png"
        ref_he = np.array(Image.open(exp_he_path).convert("RGB")) if exp_he_path.exists() else None

        sim = None
        exp_feat_path = FEAT_DIR / f"{tid}_uni.npy"
        if exp_feat_path.exists():
            exp_feat = np.load(exp_feat_path)
            gen_feat = uni_extractor.extract(gen_np)
            sim = cosine_sim(gen_feat, exp_feat)
            inf_cosine_sims.append(sim)
            print(f"  cosine_sim={sim:.4f}")
        else:
            print("  (no exp feat)")

        if do_vis and vis_data is not None:
            tile_vis_dir = vis_dir / f"inference_{tid}"
            tile_vis_dir.mkdir(parents=True, exist_ok=True)

            ctrl_full_np    = vis_data["ctrl_full"]
            active_channels = vis_data["active_channels"]
            style_inp = ([("H&E (style)", ref_he)] if ref_he is not None else [])

            save_overview_figure(
                ctrl_full=ctrl_full_np,
                active_channels=active_channels,
                gen_np=gen_np,
                style_inputs=style_inp,
                save_path=tile_vis_dir / "overview.png",
                cosine_sim_val=sim,
            )
            save_enhanced_attention_figure(
                ctrl_full=ctrl_full_np,
                active_channels=active_channels,
                gen_np=gen_np,
                attn_maps=vis_data["attn_maps"],
                style_inputs=style_inp,
                save_path=tile_vis_dir / "attention_heatmaps.png",
            )
            save_enhanced_residual_figure(
                ctrl_full=ctrl_full_np,
                active_channels=active_channels,
                gen_np=gen_np,
                residuals=vis_data["residuals"],
                refs=[],
                save_path=tile_vis_dir / "residual_magnitudes.png",
            )
            print(f"  Generating ablation grid...")
            ablation_imgs = generate_ablation_images(
                tile_id=tid,
                models=models,
                config=config,
                scheduler=scheduler,
                uni_embeds=uni_embeds,
                device=DEVICE,
                exp_channels_dir=EXP_CH_DIR,
                guidance_scale=GUIDANCE_SCALE,
                seed=SEED,
            )
            save_enhanced_ablation_grid(
                ablation_images=ablation_imgs,
                refs=[("style_ref", "H&E (style)", ref_he)] if ref_he is not None else [],
                save_path=tile_vis_dir / "ablation_grid.png",
            )

    # ── VALIDATION SET ───────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("VALIDATION SET (paired exp data — using paired UNI style)")
    print(f"{'='*60}")
    val_cosine_sims = []

    for i, tid in enumerate(validation_ids):
        print(f"[{i+1:02d}/{N_VALIDATION}] {tid}", end="")

        exp_feat_path = FEAT_DIR / f"{tid}_uni.npy"
        if exp_feat_path.exists():
            uni_embeds = torch.from_numpy(np.load(exp_feat_path)).view(1, 1, 1, 1536)
        else:
            uni_embeds = null_uni

        gen_np, _ = generate_tile(
            tile_id=tid,
            models=models,
            config=config,
            scheduler=scheduler,
            uni_embeds=uni_embeds,
            device=DEVICE,
            exp_channels_dir=EXP_CH_DIR,
            guidance_scale=GUIDANCE_SCALE,
            return_vis_data=False,
        )

        out_path = val_dir / f"{tid}_generated.png"
        Image.fromarray(gen_np).save(out_path)

        exp_feat_path = FEAT_DIR / f"{tid}_uni.npy"
        if exp_feat_path.exists():
            exp_feat = np.load(exp_feat_path)
            gen_feat = uni_extractor.extract(gen_np)
            sim = cosine_sim(gen_feat, exp_feat)
            val_cosine_sims.append(sim)
            print(f"  cosine_sim={sim:.4f}")
        else:
            print("  (no exp feat)")

    # ── UNPAIRED INFERENCE ───────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("UNPAIRED INFERENCE  (A's TME layout × B's H&E style)")
    print(f"{'='*60}")

    unpaired_dir     = OUT_DIR / "unpaired"
    unpaired_vis_dir = OUT_DIR / "visualizations_unpaired"
    unpaired_dir.mkdir(parents=True, exist_ok=True)
    unpaired_vis_dir.mkdir(parents=True, exist_ok=True)

    style_pool = [x for x in all_ids if x not in set(inference_ids)]
    N_UNPAIRED = min(N_VIS_TILES, len(inference_ids))
    unpaired_cosine_sims = []

    for i in range(N_UNPAIRED):
        tid_A = inference_ids[i]
        tid_B = random.choice(style_pool)
        print(f"[{i+1:02d}/{N_UNPAIRED}] layout={tid_A}  style={tid_B}", end="")

        uni_path_B = FEAT_DIR / f"{tid_B}_uni.npy"
        if not uni_path_B.exists():
            print("  (no UNI for B, skipping)")
            continue
        uni_B = torch.from_numpy(np.load(uni_path_B)).view(1, 1, 1, 1536)

        gen_np, vis_data = generate_tile(
            tile_id=tid_A,
            models=models,
            config=config,
            scheduler=scheduler,
            uni_embeds=uni_B,
            device=DEVICE,
            exp_channels_dir=EXP_CH_DIR,
            guidance_scale=GUIDANCE_SCALE,
            return_vis_data=True,
        )

        out_path = unpaired_dir / f"{tid_A}_layout_{tid_B}_style.png"
        Image.fromarray(gen_np).save(out_path)

        sim = None
        exp_feat_path = FEAT_DIR / f"{tid_A}_uni.npy"
        if exp_feat_path.exists():
            exp_feat = np.load(exp_feat_path)
            gen_feat = uni_extractor.extract(gen_np)
            sim = cosine_sim(gen_feat, exp_feat)
            unpaired_cosine_sims.append(sim)
            print(f"  cos_sim(vs A)={sim:.4f}")
        else:
            print()

        if vis_data is not None:
            tile_vis_dir = unpaired_vis_dir / f"{tid_A}_x_{tid_B}"
            tile_vis_dir.mkdir(parents=True, exist_ok=True)

            ctrl_np = vis_data["ctrl_full"]
            act_ch  = vis_data["active_channels"]
            ref_A = np.array(Image.open(HE_DIR / f"{tid_A}.png").convert("RGB")) if (HE_DIR / f"{tid_A}.png").exists() else None
            ref_B = np.array(Image.open(HE_DIR / f"{tid_B}.png").convert("RGB")) if (HE_DIR / f"{tid_B}.png").exists() else None
            style_inp_B = ([("H&E (style from B)", ref_B)] if ref_B is not None else [])

            save_overview_figure(
                ctrl_full=ctrl_np, active_channels=act_ch,
                gen_np=gen_np, style_inputs=style_inp_B,
                save_path=tile_vis_dir / "overview.png",
                cosine_sim_val=sim,
            )
            save_enhanced_attention_figure(
                ctrl_full=ctrl_np, active_channels=act_ch,
                gen_np=gen_np, attn_maps=vis_data["attn_maps"],
                style_inputs=style_inp_B,
                save_path=tile_vis_dir / "attention_heatmaps.png",
            )
            save_enhanced_residual_figure(
                ctrl_full=ctrl_np, active_channels=act_ch,
                gen_np=gen_np, residuals=vis_data["residuals"],
                refs=[],
                save_path=tile_vis_dir / "residual_magnitudes.png",
            )
            print(f"  Generating unpaired ablation grid...")
            ablation_imgs = generate_ablation_images(
                tile_id=tid_A,
                models=models,
                config=config,
                scheduler=scheduler,
                uni_embeds=uni_B,
                device=DEVICE,
                exp_channels_dir=EXP_CH_DIR,
                guidance_scale=GUIDANCE_SCALE,
                seed=SEED,
            )
            save_enhanced_ablation_grid(
                ablation_images=ablation_imgs,
                refs=[("style_ref", "H&E (style from B)", ref_B)] if ref_B is not None else [],
                save_path=tile_vis_dir / "ablation_grid.png",
            )

    # ── SUMMARY ─────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    results = {
        "checkpoint": str(CKPT_DIR),
        "guidance_scale": GUIDANCE_SCALE,
        "num_steps": NUM_STEPS,
        "inference_paired": {
            "mode": "style-conditioned (paired UNI)",
            "n_tiles": N_INFERENCE,
            "tile_ids": inference_ids,
            "cosine_sims": inf_cosine_sims,
            "mean_cosine_sim": float(np.mean(inf_cosine_sims)) if inf_cosine_sims else None,
            "std_cosine_sim":  float(np.std(inf_cosine_sims))  if inf_cosine_sims else None,
        },
        "validation_paired": {
            "mode": "style-conditioned (paired UNI)",
            "n_tiles": N_VALIDATION,
            "tile_ids": validation_ids,
            "cosine_sims": val_cosine_sims,
            "mean_cosine_sim": float(np.mean(val_cosine_sims)) if val_cosine_sims else None,
            "std_cosine_sim":  float(np.std(val_cosine_sims))  if val_cosine_sims else None,
        },
        "inference_unpaired": {
            "mode": "style-conditioned (B's UNI → A's channels)",
            "n_tiles": N_UNPAIRED,
            "cosine_sims_vs_layout_A": unpaired_cosine_sims,
            "mean_cosine_sim": float(np.mean(unpaired_cosine_sims)) if unpaired_cosine_sims else None,
            "std_cosine_sim":  float(np.std(unpaired_cosine_sims))  if unpaired_cosine_sims else None,
        },
    }

    metrics_path = OUT_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)

    def _fmt(d):
        m, s = d["mean_cosine_sim"], d["std_cosine_sim"]
        return f"{m:.4f} ± {s:.4f}" if m is not None else "n/a"

    print(f"\nInference  (paired,   style-cond) — mean UNI cos sim: {_fmt(results['inference_paired'])}")
    print(f"Validation (paired,   style-cond) — mean UNI cos sim: {_fmt(results['validation_paired'])}")
    print(f"Inference  (unpaired, style-cond) — mean UNI cos sim: {_fmt(results['inference_unpaired'])}")
    print(f"\nOutputs saved to: {OUT_DIR}")
    print(f"Metrics:          {metrics_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the new script parses args without error**

```bash
cd /home/ec2-user/PixCell && python tools/run_evaluation.py --help
```

Expected: prints argparse help with all flags listed, exits 0.

- [ ] **Step 3: Delete the old script**

```bash
git rm run_stage3_full.py
```

- [ ] **Step 4: Run existing tests to confirm no regressions**

```bash
python -m pytest tests/ -x -q 2>&1 | tail -10
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add tools/run_evaluation.py
git commit -m "refactor: replace run_stage3_full.py with tools/run_evaluation.py (adds argparse)"
```

---

## Task 4: Write `tests/test_inference_core.py`

**Files:**
- Create: `tests/test_inference_core.py`
- Test: `stage3_inference.py`, `train_scripts/inference_controlnet.py`

Four tests. All run on CPU with no real model weights.

- [ ] **Step 1: Write the test file**

```python
"""
Tests for stage3_inference channel loading and zero_mask_latent logic.

All tests run on CPU; no real model weights needed.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from PIL import Image


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_gray_png(path, value: int, size: int = 32):
    """Write a (size x size) single-value grayscale PNG."""
    arr = np.full((size, size), value, dtype=np.uint8)
    Image.fromarray(arr, mode="L").save(path)


def _make_sim_channels_dir(tmp_path, channels: list[str], pixel_value: int = 200, size: int = 32):
    """Create per-channel subdirs under tmp_path with a single tile PNG."""
    for ch in channels:
        d = tmp_path / ch
        d.mkdir(parents=True, exist_ok=True)
        _write_gray_png(d / "t.png", pixel_value, size)
    return tmp_path


# ---------------------------------------------------------------------------
# load_sim_channels
# ---------------------------------------------------------------------------

def test_load_sim_channels_binary_thresholding(tmp_path):
    """cell_masks channel is thresholded to {0.0, 1.0}; vasculature is not."""
    _make_sim_channels_dir(tmp_path, ["cell_masks", "vasculature"], pixel_value=200)

    from stage3_inference import load_sim_channels

    result = load_sim_channels(
        sim_channels_dir=tmp_path,
        sim_id="t",
        active_channels=["cell_masks", "vasculature"],
        resolution=32,
    )

    assert result.shape == (2, 32, 32)
    assert result.dtype == torch.float32

    # Binary channel: all values must be exactly 0.0 or 1.0
    binary_vals = result[0].unique().tolist()
    assert all(v in (0.0, 1.0) for v in binary_vals), f"Binary channel has non-binary values: {binary_vals}"

    # Continuous channel (vasculature): pixel value 200 → ~0.784, definitely not {0,1}-only
    cont_vals = result[1].unique()
    assert not all(v.item() in (0.0, 1.0) for v in cont_vals), "Continuous channel was unexpectedly binarized"


def test_load_sim_channels_cell_mask_alias(tmp_path):
    """load_sim_channels falls back to cell_mask/ dir when cell_masks/ is absent."""
    # Create the alias directory name (singular)
    _make_sim_channels_dir(tmp_path, ["cell_mask"], pixel_value=255)

    from stage3_inference import load_sim_channels

    result = load_sim_channels(
        sim_channels_dir=tmp_path,
        sim_id="t",
        active_channels=["cell_masks"],
        resolution=32,
    )
    assert result.shape == (1, 32, 32)
    # pixel 255 → 1.0 after binary threshold
    assert result.max().item() == 1.0


# ---------------------------------------------------------------------------
# encode_ctrl_mask_latent
# ---------------------------------------------------------------------------

def test_encode_ctrl_mask_latent_shape():
    """Output shape is [1, 16, H/8, W/8] regardless of number of input channels."""
    from train_scripts.inference_controlnet import encode_ctrl_mask_latent

    H, W = 32, 32
    ctrl_full = torch.rand(4, H, W)  # 4 channels; only channel 0 (mask) is used

    # Fake VAE: encode returns a latent_dist whose mean has the expected shape
    fake_vae = MagicMock()
    lat_mean = torch.zeros(1, 16, H // 8, W // 8)
    fake_vae.encode.return_value.latent_dist.mean = lat_mean

    result = encode_ctrl_mask_latent(
        ctrl_full,
        fake_vae,
        vae_shift=0.0,
        vae_scale=1.0,
        device="cpu",
        dtype=torch.float32,
    )

    assert result.shape == (1, 16, H // 8, W // 8), f"Expected (1,16,4,4), got {result.shape}"


def test_encode_ctrl_mask_latent_scaling():
    """Output equals (mean - vae_shift) * vae_scale."""
    from train_scripts.inference_controlnet import encode_ctrl_mask_latent

    ctrl_full = torch.rand(2, 16, 16)
    raw_mean = torch.ones(1, 16, 2, 2) * 5.0

    fake_vae = MagicMock()
    fake_vae.encode.return_value.latent_dist.mean = raw_mean

    result = encode_ctrl_mask_latent(
        ctrl_full, fake_vae, vae_shift=1.0, vae_scale=2.0, device="cpu", dtype=torch.float32
    )
    expected = (raw_mean - 1.0) * 2.0
    assert torch.allclose(result, expected), f"Scaling mismatch: {result} vs {expected}"


# ---------------------------------------------------------------------------
# generate — zero_mask_latent subtraction
# ---------------------------------------------------------------------------

def _make_fake_models(vae_mean: torch.Tensor, tme_out: torch.Tensor) -> dict:
    fake_vae = MagicMock()
    fake_vae.encode.return_value.latent_dist.mean = vae_mean
    fake_vae.decode = MagicMock(return_value=[torch.zeros(1, 3, 32, 32)])

    fake_tme = MagicMock(return_value=tme_out)

    return dict(
        vae=fake_vae,
        controlnet=MagicMock(),
        base_model=MagicMock(),
        tme_module=fake_tme,
    )


def _make_config(zero_mask_latent: bool):
    return SimpleNamespace(
        data=SimpleNamespace(active_channels=["cell_masks", "vasculature"]),
        scale_factor=1.0,
        shift_factor=0.0,
        image_size=32,
        channel_groups=None,   # use flat TME path (no split_channels_to_groups)
        zero_mask_latent=zero_mask_latent,
    )


def _make_scheduler():
    sched = MagicMock()
    sched.init_noise_sigma = 1.0
    sched.timesteps = []
    return sched


def test_generate_zero_mask_latent_applied(tmp_path):
    """When zero_mask_latent=True, controlnet receives tme_out - vae_mask."""
    from unittest.mock import patch
    from stage3_inference import generate

    _make_sim_channels_dir(tmp_path, ["cell_masks", "vasculature"], pixel_value=200)

    vae_mean = torch.ones(1, 16, 4, 4)     # encode returns ones
    tme_out  = torch.full((1, 16, 4, 4), 3.0)  # TME returns 3.0

    models  = _make_fake_models(vae_mean, tme_out)
    config  = _make_config(zero_mask_latent=True)
    sched   = _make_scheduler()
    captured = {}

    def fake_denoise(**kwargs):
        captured["cil"] = kwargs["controlnet_input_latent"].clone()
        return torch.zeros(1, 16, 4, 4)

    with patch("stage3_inference.denoise", side_effect=fake_denoise):
        generate(
            sim_channels_dir=tmp_path,
            sim_id="t",
            models=models,
            config=config,
            uni_embeds=torch.zeros(1, 1, 1, 1536),
            scheduler=sched,
            guidance_scale=1.0,
            device="cpu",
        )

    # fused = tme_out - vae_mask = 3.0 - (ones * scale - shift) = 3.0 - 1.0 = 2.0
    # vae_mask = (vae_mean - shift) * scale = (1.0 - 0.0) * 1.0 = 1.0
    expected = tme_out - vae_mean
    assert torch.allclose(captured["cil"].float(), expected.float()), (
        f"Expected fused={expected.flatten()[:4]}, got {captured['cil'].flatten()[:4]}"
    )


def test_generate_zero_mask_latent_off(tmp_path):
    """When zero_mask_latent=False, controlnet receives raw tme_out (no subtraction)."""
    from unittest.mock import patch
    from stage3_inference import generate

    _make_sim_channels_dir(tmp_path, ["cell_masks", "vasculature"], pixel_value=200)

    vae_mean = torch.ones(1, 16, 4, 4)
    tme_out  = torch.full((1, 16, 4, 4), 3.0)

    models  = _make_fake_models(vae_mean, tme_out)
    config  = _make_config(zero_mask_latent=False)
    sched   = _make_scheduler()
    captured = {}

    def fake_denoise(**kwargs):
        captured["cil"] = kwargs["controlnet_input_latent"].clone()
        return torch.zeros(1, 16, 4, 4)

    with patch("stage3_inference.denoise", side_effect=fake_denoise):
        generate(
            sim_channels_dir=tmp_path,
            sim_id="t",
            models=models,
            config=config,
            uni_embeds=torch.zeros(1, 1, 1, 1536),
            scheduler=sched,
            guidance_scale=1.0,
            device="cpu",
        )

    # No subtraction: fused = tme_out = 3.0
    assert torch.allclose(captured["cil"].float(), tme_out.float()), (
        f"Expected raw tme_out={tme_out.flatten()[:4]}, got {captured['cil'].flatten()[:4]}"
    )
```

- [ ] **Step 2: Run the tests**

```bash
cd /home/ec2-user/PixCell && python -m pytest tests/test_inference_core.py -v 2>&1 | tail -20
```

Expected: 6 tests collected, all PASS. If a test FAILs, read the traceback — the most common issues are:
- Import path: make sure `sys.path` includes the repo root (the conftest.py handles this).
- `fake_vae.decode.return_value` type: `generate` calls `vae.decode(...)[0]` so the return value must be subscriptable.

- [ ] **Step 3: Commit**

```bash
git add tests/test_inference_core.py
git commit -m "test: add test_inference_core for load_sim_channels and zero_mask_latent"
```

---

## Task 5: Write `tests/test_stage3_tile_pipeline.py`

**Files:**
- Create: `tests/test_stage3_tile_pipeline.py`
- Test: `tools/stage3_tile_pipeline.py`

Four tests covering channel loading, data layout resolution, and zero_mask_latent in `generate_tile`.

- [ ] **Step 1: Write the test file**

```python
"""
Tests for tools/stage3_tile_pipeline.py channel loading and generate_tile logic.

All tests run on CPU; no real model weights needed.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from PIL import Image


def _write_gray_png(path, value: int, size: int = 32):
    arr = np.full((size, size), value, dtype=np.uint8)
    Image.fromarray(arr, mode="L").save(path)


# ---------------------------------------------------------------------------
# load_channel
# ---------------------------------------------------------------------------

def test_load_channel_binary_thresholding(tmp_path):
    """Binary channel: pixel value 200 → 1.0; pixel value 50 → 0.0."""
    from tools.stage3_tile_pipeline import load_channel

    high_png = tmp_path / "high.png"
    low_png  = tmp_path / "low.png"
    _write_gray_png(high_png, 200)  # 200/255 ≈ 0.784 → > 0.5 → 1.0
    _write_gray_png(low_png,   50)  # 50/255  ≈ 0.196 → < 0.5 → 0.0

    high = load_channel(tmp_path, "high", resolution=32, binary=True)
    low  = load_channel(tmp_path, "low",  resolution=32, binary=True)

    assert high.dtype == np.float32
    unique_high = set(np.unique(high).tolist())
    unique_low  = set(np.unique(low).tolist())
    assert unique_high == {1.0}, f"Expected {{1.0}}, got {unique_high}"
    assert unique_low  == {0.0}, f"Expected {{0.0}}, got {unique_low}"


def test_load_channel_continuous_not_binarized(tmp_path):
    """Non-binary channel: pixel value 128 → ~0.502, not snapped to 0 or 1."""
    from tools.stage3_tile_pipeline import load_channel

    _write_gray_png(tmp_path / "cont.png", 128)

    result = load_channel(tmp_path, "cont", resolution=32, binary=False)

    unique = set(np.unique(result).tolist())
    assert unique != {0.0, 1.0}, "Continuous channel was unexpectedly binarized"
    # 128/255 ≈ 0.502; allow for reflect-pad + resize rounding
    assert 0.4 < result.mean() < 0.6, f"Unexpected mean: {result.mean()}"


def test_load_channel_reflect_pad_output_size(tmp_path):
    """Non-binary channel is reflect-padded then resized back to requested resolution."""
    from tools.stage3_tile_pipeline import load_channel

    # 32x32 source; with _MIRROR_BORDER_PX=8 it becomes 48x48, then resized to 32
    _write_gray_png(tmp_path / "pad.png", 100, size=32)

    result = load_channel(tmp_path, "pad", resolution=32, binary=False)

    assert result.shape == (32, 32), f"Expected (32,32), got {result.shape}"
    assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# resolve_data_layout
# ---------------------------------------------------------------------------

def test_resolve_data_layout_orion(tmp_path):
    """ORION-style root (has exp_channels/) resolves all three dirs correctly."""
    from tools.stage3_tile_pipeline import resolve_data_layout

    (tmp_path / "exp_channels").mkdir()
    (tmp_path / "features").mkdir()
    (tmp_path / "he").mkdir()

    ch_dir, feat_dir, he_dir = resolve_data_layout(tmp_path)

    assert ch_dir   == tmp_path / "exp_channels"
    assert feat_dir == tmp_path / "features"
    assert he_dir   == tmp_path / "he"


def test_resolve_data_layout_flat(tmp_path):
    """Flat sim-style root (no exp_channels/) falls back to data_root for channels."""
    from tools.stage3_tile_pipeline import resolve_data_layout

    # No exp_channels/ — channels are directly under root
    ch_dir, feat_dir, he_dir = resolve_data_layout(tmp_path)

    assert ch_dir == tmp_path   # flat: channels live directly under root
    # features/ and he/ don't exist → both fall back to root
    assert feat_dir == tmp_path
    assert he_dir   == tmp_path


# ---------------------------------------------------------------------------
# generate_tile — zero_mask_latent
# ---------------------------------------------------------------------------

def test_generate_tile_zero_mask_latent_applied(tmp_path):
    """generate_tile() subtracts vae_mask from fused when zero_mask_latent=True."""
    from unittest.mock import patch
    from tools.stage3_tile_pipeline import generate_tile

    # Exp channel dirs
    exp_ch_dir = tmp_path / "exp_channels"
    for ch in ("cell_masks", "vasculature"):
        (exp_ch_dir / ch).mkdir(parents=True)
        _write_gray_png(exp_ch_dir / ch / "t.png", 200)

    vae_mean = torch.ones(1, 16, 4, 4)
    tme_out  = torch.full((1, 16, 4, 4), 3.0)

    fake_tme = MagicMock(return_value=tme_out)
    fake_vae = MagicMock()
    fake_vae.decode = MagicMock(return_value=[torch.zeros(1, 3, 32, 32)])

    models = dict(
        vae=fake_vae,
        controlnet=MagicMock(),
        base_model=MagicMock(),
        tme_module=fake_tme,
    )

    config = SimpleNamespace(
        data=SimpleNamespace(active_channels=["cell_masks", "vasculature"]),
        scale_factor=1.0,
        shift_factor=0.0,
        image_size=32,
        channel_groups=[{"name": "g1", "channels": ["vasculature"]}],
        zero_mask_latent=True,
    )

    scheduler = MagicMock()
    scheduler.init_noise_sigma = 1.0
    scheduler.timesteps = []

    captured = {}

    def fake_denoise(**kwargs):
        captured["cil"] = kwargs["controlnet_input_latent"].clone()
        return torch.zeros(1, 16, 4, 4)

    with patch("train_scripts.inference_controlnet.encode_ctrl_mask_latent", return_value=vae_mean), \
         patch("tools.channel_group_utils.split_channels_to_groups", return_value={}), \
         patch("train_scripts.inference_controlnet.denoise", side_effect=fake_denoise):
        generate_tile(
            tile_id="t",
            models=models,
            config=config,
            scheduler=scheduler,
            uni_embeds=torch.zeros(1, 1, 1, 1536),
            device="cpu",
            exp_channels_dir=exp_ch_dir,
            guidance_scale=1.0,
        )

    # zero_mask_latent=True: fused = tme_out - vae_mask = 3.0 - 1.0 = 2.0
    expected = tme_out - vae_mean
    assert torch.allclose(captured["cil"].float(), expected.float()), (
        f"Expected {expected.flatten()[:4]}, got {captured['cil'].flatten()[:4]}"
    )


def test_generate_tile_zero_mask_latent_off(tmp_path):
    """generate_tile() passes raw tme_out when zero_mask_latent=False."""
    from unittest.mock import patch
    from tools.stage3_tile_pipeline import generate_tile

    exp_ch_dir = tmp_path / "exp_channels"
    for ch in ("cell_masks", "vasculature"):
        (exp_ch_dir / ch).mkdir(parents=True)
        _write_gray_png(exp_ch_dir / ch / "t.png", 200)

    vae_mean = torch.ones(1, 16, 4, 4)
    tme_out  = torch.full((1, 16, 4, 4), 3.0)

    fake_tme = MagicMock(return_value=tme_out)
    fake_vae = MagicMock()
    fake_vae.decode = MagicMock(return_value=[torch.zeros(1, 3, 32, 32)])

    models = dict(
        vae=fake_vae,
        controlnet=MagicMock(),
        base_model=MagicMock(),
        tme_module=fake_tme,
    )

    config = SimpleNamespace(
        data=SimpleNamespace(active_channels=["cell_masks", "vasculature"]),
        scale_factor=1.0,
        shift_factor=0.0,
        image_size=32,
        channel_groups=[{"name": "g1", "channels": ["vasculature"]}],
        zero_mask_latent=False,
    )

    scheduler = MagicMock()
    scheduler.init_noise_sigma = 1.0
    scheduler.timesteps = []

    captured = {}

    def fake_denoise(**kwargs):
        captured["cil"] = kwargs["controlnet_input_latent"].clone()
        return torch.zeros(1, 16, 4, 4)

    with patch("train_scripts.inference_controlnet.encode_ctrl_mask_latent", return_value=vae_mean), \
         patch("tools.channel_group_utils.split_channels_to_groups", return_value={}), \
         patch("train_scripts.inference_controlnet.denoise", side_effect=fake_denoise):
        generate_tile(
            tile_id="t",
            models=models,
            config=config,
            scheduler=scheduler,
            uni_embeds=torch.zeros(1, 1, 1, 1536),
            device="cpu",
            exp_channels_dir=exp_ch_dir,
            guidance_scale=1.0,
        )

    # No subtraction: fused = tme_out
    assert torch.allclose(captured["cil"].float(), tme_out.float()), (
        f"Expected {tme_out.flatten()[:4]}, got {captured['cil'].flatten()[:4]}"
    )
```

- [ ] **Step 2: Run the tests**

```bash
cd /home/ec2-user/PixCell && python -m pytest tests/test_stage3_tile_pipeline.py -v 2>&1 | tail -20
```

Expected: 7 tests collected, all PASS.

Common failure modes:
- If `resolve_data_layout` flat-layout test fails, check whether the function returns `data_root` or `data_root / "features"` when `features/` doesn't exist — read `tools/stage3_tile_pipeline.py:66-80` to verify.
- If the `generate_tile` zero_mask_latent tests fail with "captured is empty", the patch path for `denoise` may need to change. Check the import inside `generate_tile` and update the patch target accordingly.

- [ ] **Step 3: Run full test suite to confirm no regressions**

```bash
python -m pytest tests/ -q 2>&1 | tail -10
```

Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add tests/test_stage3_tile_pipeline.py
git commit -m "test: add test_stage3_tile_pipeline for channel loading and zero_mask_latent"
```
