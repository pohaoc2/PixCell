# PixCell-ControlNet: Simulation-to-Experiment TME Mapping

A multi-channel ControlNet for mapping tumor microenvironment (TME) simulation outputs to experimental H&E histology, using paired ORION-CRC multiplexed imaging data.

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
- [📦 Pretrained Weights](#-pretrained-weights)

---

## 🔧 Installation

Python >= 3.9, PyTorch >= 2.0.1, CUDA 11.7+

```bash
bash setup.sh
```

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

Train ControlNet + TMEConditioningModule on paired experimental data.

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
| `channel_reliability_weights` | `[1.0]*6 + [0.5]*3` | Per-channel attenuation; 0.5× for approximate CODEX channels |
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
└── tme_module.pth              # TME encoder weights
```

---

## Stage 3 — Inference

Generate experimental-like H&E from simulation channel images. The trained model maps CODEX-compatible multichannel layout → realistic H&E. At inference, CODEX channels are replaced with simulation outputs of the same spatial format.

### Style-conditioned (recommended)

Pass a reference H&E image to set the tissue appearance (staining, cell density):

```bash
python stage3_inference.py \
    --config           configs/config_controlnet_exp.py \
    --checkpoint-dir   checkpoints/pixcell_controlnet_exp/checkpoints/step_XXXXXXX \
    --sim-channels-dir /path/to/sim_channels \
    --sim-id           sim_0001 \
    --reference-he     /path/to/reference.png \
    --output           generated_he.png
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

| Channel | Source | Type | Train weight |
|---------|--------|------|-------------|
| `cell_mask` | Cell segmentation | Binary | — |
| `cell_type_healthy` | CODEX cell typing | Binary one-hot | 1.0 |
| `cell_type_cancer` | CODEX cell typing | Binary one-hot | 1.0 |
| `cell_type_immune` | CODEX cell typing | Binary one-hot | 1.0 |
| `cell_state_prolif` | CODEX cell state | Binary one-hot | 1.0 |
| `cell_state_nonprolif` | CODEX cell state | Binary one-hot | 1.0 |
| `cell_state_dead` | CODEX cell state | Binary one-hot | 1.0 |
| `vasculature` | CD31 (CODEX) | Float [0,1] | 0.5 |
| `oxygen` | Metabolic model | Float [0,1] | 0.5 |
| `glucose` | Metabolic model | Float [0,1] | 0.5 |

Approximate channels (vasculature, oxygen, glucose) are attenuated by 0.5× during training to account for registration uncertainty.

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

## 📦 Pretrained Weights

| Model | Purpose | Source |
|-------|---------|--------|
| PixCell-256 transformer | Frozen base model | [HuggingFace](https://huggingface.co/StonyBrook-CVLab/PixCell-256) |
| PixCell-256 ControlNet | Initial ControlNet weights | [HuggingFace](https://huggingface.co/StonyBrook-CVLab/PixCell-256) |
| SD3.5 VAE | Image encoder/decoder | [HuggingFace](https://huggingface.co/stabilityai/stable-diffusion-3.5-large) |
| UNI-2h | Feature extractor | [HuggingFace](https://huggingface.co/MahmoodLab/UNI2-h) |

Download automatically with `python stage0_setup.py`.

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
