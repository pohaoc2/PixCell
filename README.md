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

The TME conditioning uses a **Multi-Group architecture** where each channel group gets its own CNN encoder and cross-attention module. Groups produce additive, zero-initialized residuals to the VAE mask latent, enabling:

- **Disentangled control** — independently include/exclude channel groups at inference
- **Interpretability** — per-group attention heatmaps and residual magnitude maps
- **Graceful degradation** — missing groups contribute zero (no special handling needed)

| Group | Channels | Nature |
|-------|----------|--------|
| `cell_identity` | `cell_type_healthy`, `cell_type_cancer`, `cell_type_immune` | One-hot (CODEX) |
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
| `group_dropout_probs` | `{cell_identity: 0.10, cell_state: 0.10, vasculature: 0.15, microenv: 0.20}` | Per-group dropout rates during training |
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
├── attention_heatmaps.png     # per-group attention maps over cell mask
├── residual_magnitudes.png    # per-group ‖Δ‖ spatial magnitude maps
└── ablation_grid.png          # progressive composition showing group contributions
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

### Batch generation

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
# Only use cell identity and vasculature (drop cell state and microenvironment)
python stage3_inference.py \
    --config configs/config_controlnet_exp.py \
    --checkpoint-dir checkpoints/pixcell_controlnet_exp/checkpoints/step_XXXXXXX \
    --sim-channels-dir /path/to/sim_channels \
    --sim-id sim_0001 \
    --active-groups cell_identity vasculature \
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
| `--active-groups` | all | TME groups to include (e.g., `cell_identity vasculature`) |
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
| `cell_type_healthy` | `cell_identity` | CODEX multi-protein panel | Binary one-hot |
| `cell_type_cancer` | `cell_identity` | CODEX multi-protein panel | Binary one-hot |
| `cell_type_immune` | `cell_identity` | CODEX multi-protein panel | Binary one-hot |
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

Standalone tools for interpreting how each channel group influences H&E generation.

### Per-group attention heatmaps

Visualize where each group's encoder output attends on the cell mask:

```python
from tools.visualize_group_attention import save_attention_heatmap_figure

# attn_maps from module(..., return_attn_weights=True)
save_attention_heatmap_figure(mask_image, gen_image, attn_maps, "attention.png")
```

### Per-group residual magnitudes

Spatial L2 norm of each group's additive residual — "how much does each group change the conditioning at each location?"

```python
from tools.visualize_group_residuals import save_residual_magnitude_figure

# residuals from module(..., return_residuals=True)
save_residual_magnitude_figure(mask_image, gen_image, residuals, "residuals.png")
```

### Ablation grid

Progressive composition showing incremental group contributions:

```python
from tools.visualize_ablation_grid import save_ablation_grid

save_ablation_grid(
    [("Mask only", img0), ("+ Cell ID", img1), ("+ State", img2), ("All", img3)],
    "ablation.png",
)
```

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
