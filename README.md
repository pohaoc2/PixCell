# PixCell-ControlNet: Simulation-to-Experiment TME Mapping

A multi-channel ControlNet for mapping tumor microenvironment (TME) simulation outputs to experimental H&E histology, using paired ORION-CRC multiplexed imaging data.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pohaoc2/PixCell/blob/main/notebook/multichannel_controlnet.ipynb)
[![CI](https://github.com/pohaoc2/PixCell/actions/workflows/ci.yml/badge.svg)](https://github.com/pohaoc2/PixCell/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/pohaoc2/PixCell/branch/main/graph/badge.svg)](https://codecov.io/gh/pohaoc2/PixCell)

---

## Overview

This project fine-tunes [PixCell-256](https://huggingface.co/StonyBrook-CVLab/PixCell-256) with a ControlNet that accepts multi-channel spatial TME maps (cell type, cell state, vasculature, oxygen, glucose) and generates realistic H&E patches that match the simulated spatial layout.

**Training data:** Paired H&E + CODEX multiplexed protein imaging from the [ORION-CRC dataset](https://github.com/labsyspharm/orion-crc/tree/main).

**Two inference modes:**
- **Style-conditioned** — reference H&E UNI embedding + TME channels
- **TME-only** — null UNI embedding (CFG dropout trained at 15%)

**Key papers:**
- PixCell: [arXiv:2506.05127](https://arxiv.org/abs/2506.05127)
- ORION-CRC: [labsyspharm/orion-crc](https://github.com/labsyspharm/orion-crc/tree/main)

---

## Contents

- [🔧 Installation](#-installation)
- [🗺️ Pipeline Overview](#️-pipeline-overview)
- [Stage 0 — Model Setup](#stage-0--model-setup)
- [Stage 1 — Feature Extraction](#stage-1--feature-extraction)
- [Stage 2 — Training](#stage-2--training)
- [Stage 3 — Inference](#stage-3--inference)
- [📊 Monitoring](#-monitoring)
- [✅ Validation](#-validation)
- [📂 Data Reference](#-data-reference)
- [🔍 Analysis Tools](#-analysis-tools)
- [📦 Pretrained Weights](#-pretrained-weights)
- [Pretrained inference check](#pretrained-inference-check)

---

## 🔧 Installation

Python >= 3.9, PyTorch >= 2.0.1, CUDA 11.7+

```bash
bash setup.sh
```

Dependency note:

- Primary dependency file is `requirements.txt`.
- `requirements_nommcv.txt` is now a compatibility alias to `requirements.txt`.
- `mmcv==1.7.0` is installed separately (not via requirements) because it needs:
  `pip install mmcv==1.7.0 --no-build-isolation`

---

## 🗺️ Pipeline Overview

The pipeline is divided into four sequential stages:

```
Stage 0: Model Setup       stage0_setup.py
         ↓ pretrained weights ready
Stage 1: Feature Extraction  stage1_extract_features.py
         ↓ UNI embeddings + VAE latents cached
Stage 2: Training            stage2_train.py
         ↓ ControlNet + TME module checkpoint
Stage 3: Inference           stage3_inference.py
         → experimental-like H&E from simulation channels
```

**Training** uses paired experimental data (ORION-CRC H&E + CODEX multichannel).
**Inference** accepts unpaired simulation channels and generates realistic H&E.

---

## Stage 0 — Model Setup

Download all pretrained models:

```bash
python stage0_setup.py
```

This downloads and organizes under `pretrained_models/`:

```
pretrained_models/
├── pixcell-256/transformer/          # frozen base diffusion transformer
├── pixcell-256-controlnet/controlnet/ # ControlNet initialization
├── sd-3.5-vae/vae/                   # image encoder / decoder
└── uni-2h/                           # histopathology feature extractor
```

Selective download:

```bash
python stage0_setup.py --model pixcell          # base transformer only
python stage0_setup.py --model pixcell-controlnet
python stage0_setup.py --model uni2h
python stage0_setup.py --model sd3_vae
```

Requires a HuggingFace token with access to gated models:

```bash
export HF_TOKEN=hf_...
```

---

## Stage 1 — Feature Extraction

Extract UNI-2h embeddings and SD3.5 VAE latents from your paired experimental H&E tiles. These are cached once and loaded at every training step.

**Pass 1 — H&E images** (produces UNI embeddings + VAE latents):

```bash
python stage1_extract_features.py \
    --image-dir  ./data/exp_paired/he_images \
    --output-dir ./data/exp_paired/features
```

**Pass 2 — cell_mask images** (produces VAE latents used for conditioning):

```bash
python stage1_extract_features.py \
    --image-dir   ./data/exp_paired/exp_channels/cell_mask \
    --output-dir  ./data/exp_paired/vae_features \
    --vae-prefix  mask_sd3_vae
```

Output per image:

| File | Shape | Contents |
|------|-------|----------|
| `{stem}_uni.npy` | `[1536]` | UNI-2h embedding |
| `{stem}_sd3_vae.npy` | `[2, 16, H/8, W/8]` | VAE latent mean + std |
| `{stem}_mask_sd3_vae.npy` | `[2, 16, H/8, W/8]` | cell_mask VAE latent |

### Build HDF5 index

```python
from diffusion.data.datasets.paired_exp_controlnet_dataset import build_exp_index
build_exp_index(
    exp_channels_dir="data/exp_paired/exp_channels",
    output_path="data/exp_paired/metadata/exp_index.hdf5",
)
```

---

## Stage 2 — Training

Train ControlNet + Multi-Group TME module on paired experimental data.

### TME Architecture

The TME conditioning uses a **Multi-Group architecture** where each channel group gets its own CNN encoder and cross-attention module. Groups produce additive, zero-initialized residuals, enabling:

- **Disentangled control** — independently include/exclude channel groups at inference
- **Interpretability** — per-group residual magnitude maps and ablation diff maps
- **Graceful degradation** — missing groups contribute zero (no special handling needed)

**`zero_mask_latent=True`** (enabled by default): the TME module receives the real VAE mask latent as spatial query keys, then subtracts it from its output — `fused = tme(vae_mask) − vae_mask`. This closes the direct mask→ControlNet bypass path and forces the ControlNet to rely on TME residuals, while preserving the spatial structure needed for cell-layout-aware cross-attention.

| Group | Channels | Nature |
|-------|----------|--------|
| `cell_types` | `cell_type_healthy`, `cell_type_cancer`, `cell_type_immune` | One-hot (CODEX) |
| `cell_state` | `cell_state_prolif`, `cell_state_nonprolif`, `cell_state_dead` | One-hot (CODEX) |
| `vasculature` | `vasculature` | Continuous (CD31) |
| `microenv` | `oxygen`, `glucose` | Continuous (PDE-derived) |

Each group is independently droppable during both training (per-group dropout) and inference (`--active-groups` / `--drop-groups`).

### Single GPU

```bash
python stage2_train.py configs/config_controlnet_exp.py
```

### Multi-GPU (recommended)

```bash
accelerate config   # configure once
accelerate launch --num_processes 4 stage2_train.py \
    configs/config_controlnet_exp.py
```

### Config: `configs/config_controlnet_exp.py`

Key fields to set before training:

```python
exp_data_root = "./data/exp_paired"   # path to your paired dataset
```

Key training knobs:

| Field | Default | Description |
|-------|---------|-------------|
| `cfg_dropout_prob` | `0.15` | Fraction of steps where UNI embedding is zeroed (enables TME-only inference) |
| `group_dropout_probs` | `{cell_types: 0.10, cell_state: 0.10, vasculature: 0.15, microenv: 0.20}` | Per-group dropout rates during training |
| `tme_lr` | `1e-5` | TME module learning rate |
| `num_epochs` | `200` | Training epochs |
| `save_model_steps` | `10000` | Checkpoint every N steps |

### CLI options

| Flag | Description |
|------|-------------|
| `--work-dir PATH` | Output directory for checkpoints and logs |
| `--resume-from PATH` | Resume ControlNet from checkpoint directory |
| `--load-from PATH` | Load specific checkpoint file |
| `--batch-size N` | Override `train_batch_size` in config |
| `--debug` | Minimal steps for debugging |

### Checkpoint layout

```
checkpoints/pixcell_controlnet_exp/checkpoints/step_XXXXXXX/
├── epoch_X_step_XXXXXXX.pth    # ControlNet weights
└── tme_module.pth              # Multi-Group TME module weights
```

### Training-time visualizations

At every `save_model_steps` checkpoint, the training loop generates validation visualizations:

```
checkpoints/pixcell_controlnet_exp/vis/step_XXXXXXX/
├── overview.png               # input channels and generated H&E
└── attention_heatmaps.png     # per-group attention maps
```

---

## Stage 3 — Inference

Generate experimental-like H&E from simulation channel images. The trained model maps CODEX-compatible multichannel layout → realistic H&E. At inference, CODEX channels are replaced with simulation outputs of the same spatial format.

### Style-conditioned (recommended)

Pass a reference H&E image to set the tissue appearance (staining, cell density):

```bash
python stage3_inference.py \
    --config           configs/config_controlnet_exp.py \
    --checkpoint-dir   checkpoints/pixcell_controlnet_exp/checkpoints/step_0000004 \
    --sim-channels-dir inference_data \
    --sim-id           sim_0001 \
    --reference-uni    data/orion-crc/features/0_256_uni.npy \
    --output           inference_data/generated_he.png
```

### TME-only (no reference H&E)

Generate purely from TME layout. Requires `cfg_dropout_prob > 0` at training time:

```bash
python stage3_inference.py \
    --config           configs/config_controlnet_exp.py \
    --checkpoint-dir   checkpoints/pixcell_controlnet_exp/checkpoints/step_XXXXXXX \
    --sim-channels-dir /path/to/sim_channels \
    --sim-id           sim_0001 \
    --output           generated_he.png
```

### Batch generation with visualizations

Generate H&E plus the two Stage 3 visuals used in evaluation (paired + unpaired style conditioning):

```bash
python tools/run_evaluation.py \
    --checkpoint-dir checkpoints/pixcell_controlnet_exp/checkpoints/zero_out_mask_post \
    --output-dir     inference_output/zero_out_mask_post \
    --n-tiles        3
```

Per patch, outputs under `{output_dir}/{tile_id}/{paired,unpaired}/`:

| File | Contents |
|------|----------|
| `generated_he.png` | Generated H&E image |
| `overview.png` | TME input channels → generated H&E |
| `ablation_grid.png` | 4-row: H&E+mask overlay \| Δpixel diff \| TME channel composites |

Paired = same tile's UNI + TME; unpaired = next tile's UNI + this tile's TME (cross-patch style test).
Also writes `metrics.json` with per-tile UNI cosine similarity scores.

Add `--no-metrics` to skip cosine similarity computation (faster, no UNI extractor needed).

### Single-tile visualization + ablation test suite

Generate the full Stage 3 visualization bundle for one tile, including the exhaustive group ablation tests:

```bash
python tools/generate_stage3_tile_vis.py \
    --config         configs/config_controlnet_exp.py \
    --checkpoint-dir checkpoints/pixcell_controlnet_exp/checkpoints/zero_out_mask_post \
    --data-root      data/orion-crc33 \
    --tile-id        YOUR_TILE_ID \
    --output-dir     inference_output/YOUR_TILE_ID
```

Add `--null-uni` for TME-only generation, or pass `--uni-npy /path/to/{tile_id}_uni.npy` and `--reference-he /path/to/reference.png` to override the default style inputs.

Outputs under `{output_dir}/`:

| File | Contents |
|------|----------|
| `overview.png` | Input channels, reference style H&E, and generated H&E |
| `ablation_grid.png` | Default progressive group-addition sweep |
| `ablation_single_groups.png` | 4 standalone single-group tests: cell types, cell state, vasculature, nutrient |
| `ablation_group_pairs.png` | All 6 two-group combinations |
| `ablation_group_triples.png` | All 4 three-group combinations |
| `ablation_orders/` | 24 progressive addition orders; each figure contains baseline + cumulative additions for one group order |

The exhaustive ablation suite is built from the four Stage 3 groups: `cell_types`, `cell_state`, `vasculature`, and `microenv` (`nutrient` in figure labels).

For full subset-combination testing (singles/pairs/triples/all), use the cache-based workflow.

### Ablation + Metrics CLI Summary

| Script | Purpose | Typical command |
|--------|---------|-----------------|
| `tools/generate_stage3_ablation_subset_cache.py` | Generate cached single/pair/triple/all H&E PNGs plus `manifest.json` | `python tools/generate_stage3_ablation_subset_cache.py --config configs/config_controlnet_exp.py --checkpoint-dir checkpoints/pixcell_controlnet_exp/checkpoints/zero_out_mask_post --data-root data/orion-crc33 --tile-id YOUR_TILE_ID` |
| `tools/export_cellvit_batch.py` | Flatten cached generated H&E PNGs into one folder for external CellViT processing | `python tools/export_cellvit_batch.py --cache-root inference_output/cache --output-dir inference_output/cellvit_batch --zip` |
| `tools/import_cellvit_results.py` | Copy flat CellViT JSON results back beside each cached generated H&E image | `python tools/import_cellvit_results.py --manifest inference_output/cellvit_batch/manifest.json --results-dir inference_output/cellvit` |
| `tools/compute_ablation_metrics.py` | Write `<cache-dir>/metrics.json` with cosine / LPIPS / AJI / PQ | `conda run -n pixcell python tools/compute_ablation_metrics.py --cache-dir inference_output/cache --orion-root data/orion-crc33 --metrics lpips aji pq` |
| `tools/stage3_ablation_grid_figure.py` | Render the static ranked 4×4 matplotlib figure from cached PNGs + `metrics.json` for one tile or all tiles | `python tools/stage3_ablation_grid_figure.py --cache-dir inference_output/cache --orion-root data/orion-crc33 --sort-by pq --no-auto-cosine --jobs 8` |
| `tools/stage3_ablation_grid_webvis.py` | Render the self-contained interactive HTML ablation grid | `python tools/stage3_ablation_grid_webvis.py --cache-dir inference_output/cache/YOUR_TILE_ID --orion-root data/orion-crc33 --all4ch-image inference_output/cache/YOUR_TILE_ID/all/generated_he.png` |

Recommended end-to-end sequence:

1. Generate or refresh the ablation cache.
2. Export flat PNGs for CellViT.
3. Run CellViT externally.
4. Import CellViT JSON back into the cache tree.
5. Compute `metrics.json` across all tiles.
6. Render the static PNG and/or interactive HTML view.

```bash
python tools/generate_stage3_ablation_subset_cache.py \
    --config         configs/config_controlnet_exp.py \
    --checkpoint-dir checkpoints/pixcell_controlnet_exp/checkpoints/zero_out_mask_post \
    --data-root      data/orion-crc33 \
    --tile-id        YOUR_TILE_ID

# Existing cache repair: add missing all/ and cache UNI features for each condition.
python tools/generate_stage3_ablation_subset_cache.py \
    --config                configs/config_controlnet_exp.py \
    --checkpoint-dir        checkpoints/pixcell_controlnet_exp/checkpoints/zero_out_mask_post \
    --data-root             data/orion-crc33 \
    --existing-cache-parent inference_output/cache \
    --cache-uni-features

python tools/stage3_ablation_grid_figure.py \
    --cache-dir inference_output/cache/YOUR_TILE_ID \
    --orion-root data/orion-crc33 \
    --sort-by pq \
    --no-auto-cosine
```

The first command writes `singles/`, `pairs/`, `triples/`, `all/`, and `manifest.json` under the tile cache directory.  
The repair command backfills missing `all/generated_he.png`, updates each manifest to include the all-groups condition, and writes UNI embeddings under `features/`.
The second command renders `<cache-dir>/ablation_grid.png` from cached images without rerunning diffusion.

To render static figures for every tile in a parent cache directory, use:

```bash
python tools/stage3_ablation_grid_figure.py \
    --cache-dir inference_output/cache \
    --orion-root data/orion-crc33 \
    --sort-by pq \
    --no-auto-cosine \
    --jobs 8
```

Useful figure flags:

- `--sort-by {cosine,lpips,aji,pq}` chooses the primary ranking metric.
- `--no-auto-cosine` keeps the renderer from trying to recompute cosine values.
- `--jobs N` parallelizes parent-directory rendering across tiles.
- `--debug-cellvit-overlay` overlays imported CellViT contours in yellow on generated H&E panels for PQ/AJI debugging.

If CellViT results have already been imported as `*_cellvit_instances.json`, compute metrics for every tile in the cache root with:

```bash
conda run --no-capture-output -n pixcell \
    python -u tools/compute_ablation_metrics.py \
    --cache-dir inference_output/cache \
    --orion-root data/orion-crc33 \
    --metrics lpips aji pq \
    --lpips-batch-size 8
```

Notes:

- Use `--metrics aji pq` if you want to skip LPIPS.
- `lpips` is installed via `requirements.txt` / `environment.yml`, but the command should be run from the `pixcell` conda env because `base` does not include PyTorch.
- `compute_ablation_metrics.py` uses precomputed reference UNI features under `data/orion-crc33/features/`; it only needs the UNI model if cosine is requested and per-condition cosine scores are missing.

For raw batch generation without visualizations:

```bash
python stage3_inference.py \
    --config           configs/config_controlnet_exp.py \
    --checkpoint-dir   checkpoints/pixcell_controlnet_exp/checkpoints/step_XXXXXXX \
    --sim-channels-dir /path/to/sim_channels \
    --output-dir       ./inference_output \
    --n-tiles          50
```

### Group control at inference

Selectively include or exclude TME channel groups:

```bash
# Only use cell types and vasculature (drop cell state and microenvironment)
python stage3_inference.py \
    --config configs/config_controlnet_exp.py \
    --checkpoint-dir checkpoints/pixcell_controlnet_exp/checkpoints/step_XXXXXXX \
    --sim-channels-dir /path/to/sim_channels \
    --sim-id sim_0001 \
    --active-groups cell_types vasculature \
    --output generated_he.png

# Exclude microenvironment channels (O₂/glucose)
python stage3_inference.py \
    ... \
    --drop-groups microenv \
    --output generated_he.png
```

### All inference flags

| Flag | Default | Description |
|------|---------|-------------|
| `--sim-channels-dir` | required | Root dir with per-channel subdirectories |
| `--sim-id` | — | Single snapshot ID (file stem) |
| `--output` | — | Output PNG for single-tile mode |
| `--output-dir` | — | Output directory for batch mode |
| `--n-tiles` | all | Max tiles in batch mode |
| `--reference-he` | — | Reference H&E image for style conditioning |
| `--reference-uni` | — | Precomputed UNI `.npy` (skips extraction) |
| `--active-groups` | all | TME groups to include (e.g., `cell_types vasculature`) |
| `--drop-groups` | none | TME groups to exclude (e.g., `microenv`) |
| `--guidance-scale` | `2.5` | CFG guidance scale |
| `--num-steps` | `20` | Denoising steps |
| `--device` | `cuda` | Device |

### Simulation channel directory layout

```
sim_channels/
├── cell_mask/              {sim_id}.png   binary (required)
├── cell_type_healthy/      {sim_id}.png   (optional)
├── cell_type_cancer/       {sim_id}.png   (optional)
├── cell_type_immune/       {sim_id}.png   (optional)
├── cell_state_prolif/      {sim_id}.png   (optional)
├── cell_state_nonprolif/   {sim_id}.png   (optional)
├── cell_state_dead/        {sim_id}.png   (optional)
├── vasculature/            {sim_id}.png   (optional)
├── oxygen/                 {sim_id}.png or .npy
└── glucose/                {sim_id}.png or .npy
```

Only channels listed in `configs/config_controlnet_exp.py → data.active_channels` are loaded.

---

## 📊 Monitoring

Training logs are written to TensorBoard:

```bash
tensorboard --logdir checkpoints/pixcell_controlnet_exp/logs
```

Key metrics:

| Metric | Description |
|--------|-------------|
| `loss` | Diffusion MSE loss — should decrease steadily |
| `lr_ctrl` | ControlNet learning rate |
| `lr_tme` | TME encoder learning rate |
| `samples_per_sec` | Training throughput |

---

## ✅ Validation

Evaluate sim→exp domain alignment: generate H&E from simulation TME channels and measure cosine similarity against precomputed experimental UNI features.

```bash
python pipeline/validate_sim_to_exp.py \
    --config          configs/config_controlnet_exp.py \
    --sim-root        /path/to/sim_data_root \
    --exp-feat        /path/to/exp_data_root/features \
    --controlnet-ckpt checkpoints/pixcell_controlnet_exp/checkpoints/step_XXXXXXX \
    --tme-ckpt        checkpoints/pixcell_controlnet_exp/checkpoints/step_XXXXXXX \
    --uni-model       ./pretrained_models/uni-2h \
    --n-tiles         50 \
    --guidance-scale  2.5 \
    --output-dir      ./validation_output
```

| Flag | Description |
|------|-------------|
| `--sim-root` | Sim data root (contains `sim_channels/`) |
| `--exp-feat` | Directory of `{tile_id}_uni.npy` exp target features |
| `--controlnet-ckpt` | Checkpoint directory |
| `--tme-ckpt` | Checkpoint directory (usually same as above) |
| `--reference-uni` | Optional `.npy` for style-conditioned mode |
| `--n-tiles` | Number of sim snapshots to evaluate (default 50) |
| `--guidance-scale` | CFG guidance scale (default 2.5) |
| `--output-dir` | Save generated H&E tiles here (optional) |

Example output:

```
  snap_0001: cosine_sim=0.7821
  snap_0002: cosine_sim=0.7543
  ...
=== Validation Results ===
N tiles:          50
Mean cosine sim:  0.771
Std cosine sim:   0.032
```

---

## 📂 Data Reference

### Experimental channel layout (ORION-CRC)

| Channel | Group | Source | Type |
|---------|-------|--------|------|
| `cell_mask` | *(always present)* | Cell segmentation | Binary |
| `cell_type_healthy` | `cell_types` | CODEX multi-protein panel | Binary one-hot |
| `cell_type_cancer` | `cell_types` | CODEX multi-protein panel | Binary one-hot |
| `cell_type_immune` | `cell_types` | CODEX multi-protein panel | Binary one-hot |
| `cell_state_prolif` | `cell_state` | CODEX multi-protein panel | Binary one-hot |
| `cell_state_nonprolif` | `cell_state` | CODEX multi-protein panel | Binary one-hot |
| `cell_state_dead` | `cell_state` | CODEX multi-protein panel | Binary one-hot |
| `vasculature` | `vasculature` | CD31 marker (CODEX) | Float [0,1] |
| `oxygen` | `microenv` | PDE model (distance to vasculature) | Float [0,1] |
| `glucose` | `microenv` | PDE model (distance to vasculature) | Float [0,1] |

Each group has independent dropout during training (higher for PDE-derived `microenv`). Groups can be selectively included/excluded at inference for counterfactual analysis.

### Experimental dataset directory layout

```
exp_data_root/
├── metadata/
│   └── exp_index.hdf5              # HDF5: "exp_256" → list of tile_ids
├── exp_channels/                   # one sub-folder per channel
│   ├── cell_mask/
│   │   └── {tile_id}.png           # binary PNG {0, 255}
│   ├── cell_type_healthy/
│   │   └── {tile_id}.png
│   ├── cell_type_cancer/
│   ├── cell_type_immune/
│   ├── cell_state_prolif/
│   ├── cell_state_nonprolif/
│   ├── cell_state_dead/
│   ├── vasculature/
│   │   └── {tile_id}.png           # grayscale float PNG
│   ├── oxygen/
│   └── glucose/
├── features/
│   └── {tile_id}_uni.npy           # UNI-2h embedding [1536]  ← Stage 1 output
└── vae_features/
    ├── {tile_id}_sd3_vae.npy       # VAE latent [2, 16, 32, 32]  ← Stage 1 output
    └── {tile_id}_mask_sd3_vae.npy  # cell_mask VAE latent  ← Stage 1 output
```

---

## 🔍 Analysis Tools

Most visualization functions live in `tools/stage3/figures.py`. Cache helpers for subset-combination ablations live in `tools/stage3/ablation_cache.py`, and the publication/evaluation grid renderer is `tools/stage3/ablation_grid_figure.py`. Inference helpers (channel loading, model generation, ablation sweeps) are in `tools/stage3/tile_pipeline.py`. Channel colors are centralized in `tools/color_constants.py`.

### Dataset metrics figure renderer

Use `tools/render_dataset_metrics_option_a.py` to export the standalone five-metric "Option A" summary figure as a transparent PNG. The current script renders the curated layout in `dataset_metrics_option_a.html` using the script's built-in example statistics and writes the PNG to the repo root by default.

```bash
python tools/render_dataset_metrics_option_a.py
```

Optional flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--output PATH` | `dataset_metrics_option_a.png` | Output PNG path |
| `--dpi N` | `300` | Export resolution |

Examples:

```bash
python tools/render_dataset_metrics_option_a.py \
    --output figures/dataset_metrics_option_a.png \
    --dpi 400
```

Notes:

- The export uses a transparent background.
- The renderer requests `Helvetica` first and falls back to `Arial` / `DejaVu Sans` if needed.
- If you want to iterate on the browser version, update `dataset_metrics_option_a.html` and rerun the script to refresh the PNG.

### Ablation tests

4-row figure: generated H&E with input cell mask contour overlay | per-step Δpixel diff maps | TME channel composites for each newly-added group.

```python
from tools.stage3.figures import save_enhanced_ablation_grid
from tools.stage3.tile_pipeline import generate_ablation_images

ablation_imgs = generate_ablation_images(tile_id, models, config, scheduler,
                                          uni_embeds, device, exp_channels_dir,
                                          guidance_scale, seed)
save_enhanced_ablation_grid(
    ablation_images=ablation_imgs,
    refs=[("style_ref", "H&E (style)", ref_he)],
    ctrl_full=vis_data["ctrl_full"],
    active_channels=vis_data["active_channels"],
    channel_groups=config.channel_groups,
    save_path="ablation_grid.png",
)
```

TME composites use semantic colors matching `CELL_TYPE_COLORS` / `CELL_STATE_COLORS`: cancer=red, immune=blue, healthy=green; prolif=yellow, nonprolif=grey, dead=brown; microenv uses additive cyan (O₂) + yellow (glucose) blend.

For the full ablation test suite, the shared generator accepts arbitrary group-condition plans and is reused for single-group, pair, triple, and all-order sweeps. During figure-design iteration, cache the 14 single/pair/triple PNGs once and then rebuild the combined layout from disk:

```python
from tools.stage3.ablation import (
    build_subset_ablation_sections,
    order_slug,
    reorder_channel_groups,
)
from tools.stage3.ablation_cache import save_subset_condition_cache
from tools.stage3.figures import save_condition_ablation_grid, save_enhanced_ablation_grid
from tools.stage3.tile_pipeline import (
    generate_all_progressive_order_ablation_images,
    generate_group_combination_ablation_images,
)

# 4 single-group conditions
single_group_imgs = generate_group_combination_ablation_images(
    tile_id, models, config, scheduler, uni_embeds, device, exp_channels_dir,
    guidance_scale, seed, subset_size=1,
)
save_condition_ablation_grid(single_group_imgs, "ablation_single_groups.png")

pair_group_imgs = generate_group_combination_ablation_images(
    tile_id, models, config, scheduler, uni_embeds, device, exp_channels_dir,
    guidance_scale, seed, subset_size=2,
)
triple_group_imgs = generate_group_combination_ablation_images(
    tile_id, models, config, scheduler, uni_embeds, device, exp_channels_dir,
    guidance_scale, seed, subset_size=3,
)

# Cache the 14 subset images for quick layout iteration
subset_sections = build_subset_ablation_sections(
    tuple(group["name"] for group in config.channel_groups),
    single_images=single_group_imgs,
    pair_images=pair_group_imgs,
    triple_images=triple_group_imgs,
)
save_subset_condition_cache(
    "inference_output/test_combinations/YOUR_TILE_ID",
    tile_id=tile_id,
    group_names=tuple(group["name"] for group in config.channel_groups),
    sections=subset_sections,
)
# Then render the 4x4 ranked grid from cache:
# python tools/stage3_ablation_grid_figure.py \
#   --cache-dir inference_output/test_combinations/YOUR_TILE_ID \
#   --orion-root data/orion-crc33

# 24 progressive addition orders
for idx, (group_order, order_imgs) in enumerate(
    generate_all_progressive_order_ablation_images(
        tile_id, models, config, scheduler, uni_embeds, device, exp_channels_dir,
        guidance_scale, seed,
    ),
    start=1,
):
    save_enhanced_ablation_grid(
        ablation_images=order_imgs,
        channel_groups=reorder_channel_groups(config.channel_groups, group_order),
        save_path=f"ablation_orders/{idx:02d}_{order_slug(group_order)}.png",
    )
```

Available exhaustive group tests:

| Output | Count | Description |
|--------|-------|-------------|
| `ablation_single_groups.png` | `4` | Only one group active at a time |
| `ablation_group_pairs.png` | `6` | All `4 choose 2` two-group combinations |
| `ablation_group_triples.png` | `4` | All `4 choose 3` three-group combinations |
| `ablation_orders/*.png` | `24` | All `4!` progressive addition orders |
| `test_combinations/{tile_id}/ablation_grid.png` | `16` | Ranked 4×4 grid (single/pair/triple/all + real H&E reference) rendered from cache |

### Legacy residual figure

`residuals.png` generation is no longer part of the default Stage 3 evaluation workflow.
Legacy residual visualization helpers remain in `tools/stage3/figures.py` if needed for debugging.

---

## 📦 Pretrained Weights

| Model | Purpose | Source |
|-------|---------|--------|
| PixCell-256 transformer | Frozen base model | [HuggingFace](https://huggingface.co/StonyBrook-CVLab/PixCell-256) |
| PixCell-256 ControlNet | Initial ControlNet weights | [HuggingFace](https://huggingface.co/StonyBrook-CVLab/PixCell-256) |
| SD3.5 VAE | Image encoder/decoder | [HuggingFace](https://huggingface.co/stabilityai/stable-diffusion-3.5-large) |
| UNI-2h | Feature extractor | [HuggingFace](https://huggingface.co/MahmoodLab/UNI2-h) |

Download automatically with `python stage0_setup.py`.

### Pretrained inference check

[`verify_pretrained_inference.py`](verify_pretrained_inference.py) loads the **public** PixCell-256 transformer and ControlNet weights (not your Stage 2 fine-tuned checkpoint), runs a short denoising pass on a mask + reference H&E, and writes a comparison figure—useful to confirm `stage0_setup.py` and remapping work before training or inference.

**Prerequisites:** `python stage0_setup.py` (and `inference_data/` sample mask + reference images, or pass your own paths).

**Minimal run** (uses defaults in the script):

```bash
python verify_pretrained_inference.py
```

**Explicit paths** (matches the script defaults; override as needed):

```bash
python verify_pretrained_inference.py \
    --config                 configs/config_controlnet_exp.py \
    --base-safetensors       pretrained_models/pixcell-256/transformer/diffusion_pytorch_model.safetensors \
    --controlnet-safetensors pretrained_models/pixcell-256-controlnet/controlnet/diffusion_pytorch_model.safetensors \
    --vae-path               pretrained_models/sd-3.5-vae/vae \
    --uni-model-path         pretrained_models/uni-2h \
    --mask-path              inference_data/sample/test_mask.png \
    --reference-he           inference_data/sample/test_control_image.png \
    --reference-uni          inference_data/sample/test_control_image_uni.npy \
    --mask-latent            inference_data/sample/test_mask_sd3_vae.npy \
    --generated-output       inference_data/sample/generated_he_pretrained_test_mask.png \
    --output-path            inference_data/sample/vis_pretrained_verification_test_mask.png
```

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | `configs/config_controlnet_exp.py` | Builds base + ControlNet architecture |
| `--base-safetensors` | `pretrained_models/pixcell-256/transformer/diffusion_pytorch_model.safetensors` | Frozen PixCell transformer |
| `--controlnet-safetensors` | `pretrained_models/pixcell-256-controlnet/controlnet/diffusion_pytorch_model.safetensors` | Init ControlNet (key-remapped on load) |
| `--vae-path` | `pretrained_models/sd-3.5-vae/vae` | SD3.5 VAE directory |
| `--uni-model-path` | `pretrained_models/uni-2h` | UNI-2h for reference style embedding |
| `--mask-path` | `inference_data/test_mask.png` | Cell mask image (RGB PNG) |
| `--reference-he` | `inference_data/test_control_image.png` | Reference H&E for UNI conditioning / plot |
| `--reference-uni` | `inference_data/test_control_image_uni.npy` | Cached UNI vector (created if missing) |
| `--mask-latent` | `inference_data/test_mask_sd3_vae.npy` | Cached VAE latent for mask (created if missing) |
| `--generated-output` | `inference_data/generated_he_pretrained_test_mask.png` | Saved generated H&E |
| `--output-path` | `inference_data/vis_pretrained_verification_test_mask.png` | Side-by-side comparison figure |

---

## 📄 Citation

If you use this work, please cite:

```bibtex
@article{yellapragada2025pixcell,
  title={PixCell: A generative foundation model for digital histopathology images},
  author={Yellapragada, Srikar and Graikos, Alexandros and Li, Zilinghan and Triaridis, Kostas
          and Belagali, Varun and Kapse, Saarthak and Nandi, Tarak Nath and Madduri, Ravi K
          and Prasanna, Prateek and Kurc, Tahsin and others},
  journal={arXiv preprint arXiv:2506.05127},
  year={2025}
}
```

---

## 🤗 Acknowledgements

Built on [PixArt-Sigma](https://github.com/PixArt-alpha/PixArt-sigma), [HuggingFace Diffusers](https://github.com/huggingface/diffusers), and the [ORION-CRC](https://github.com/labsyspharm/orion-crc/tree/main) dataset.
