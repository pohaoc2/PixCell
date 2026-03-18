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
- [📂 Data Preparation](#-data-preparation)
- [🚀 Training](#-training)
- [📊 Monitoring](#-monitoring)
- [🔬 Inference](#-inference)
- [✅ Validation](#-validation)
- [📦 Pretrained Weights](#-pretrained-weights)

---

## 🔧 Installation

Python >= 3.9, PyTorch >= 2.0.1, CUDA 11.7+

```bash
bash setup.sh
```

---

## 📂 Data Preparation

### ORION-CRC data

Download the ORION-CRC dataset from [labsyspharm/orion-crc](https://github.com/labsyspharm/orion-crc/tree/main). The pipeline expects each tile to have:
- An H&E image (or pre-extracted VAE latent)
- Registered CODEX-derived TME channels at the same spatial coordinates

### Channel layout

| Channel | Source | Type | Weight |
|---------|--------|------|--------|
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

### Directory layout

```
exp_data_root/
├── metadata/
│   └── exp_index.hdf5              # HDF5: "exp_256" → list of tile_ids
├── exp_channels/                   # one sub-folder per channel
│   ├── cell_mask/
│   │   └── {tile_id}.png           # binary PNG {0, 255}
│   ├── cell_type_healthy/
│   │   └── {tile_id}.png           # binary PNG {0, 255}
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
│   └── {tile_id}_uni.npy           # UNI-2h embedding, shape [1536]
└── vae_features/
    ├── {tile_id}_sd3_vae.npy       # SD3-VAE latent, shape [32, 32, 32] (mean+std)
    └── {tile_id}_mask_sd3_vae.npy  # cell_mask VAE latent
```

### Build HDF5 index

```python
from diffusion.data.datasets.paired_exp_controlnet_dataset import build_exp_index
build_exp_index(
    exp_channels_dir="exp_data_root/exp_channels",
    output_path="exp_data_root/metadata/exp_index.hdf5",
)
```

### Feature extraction

Extract UNI-2h embeddings and SD3-VAE latents:

```bash
python extract_features.py \
    --data-root /path/to/exp_data_root \
    --output-dir /path/to/exp_data_root
```

---

## 🚀 Training

### Stage 1 — Simulation pre-training (optional)

If you have agent-based simulation outputs (ABM/Physicell), pre-train on unpaired sim snapshots + real H&E:

**Simulation data layout:**

```
sim_data_root/
├── metadata/
│   ├── sim_index.hdf5          # "sim_256" → sim snapshot IDs
│   └── real_index.hdf5         # "real_256" → real tile IDs
├── sim_channels/
│   ├── cell_mask/              # binary PNG (required)
│   ├── oxygen/                 # float PNG or NPY (required)
│   ├── glucose/                # float PNG or NPY (optional)
│   └── tgf/                    # float PNG or NPY (optional)
├── features/
│   └── {tile_id}_uni.npy
└── vae_features/
    └── {tile_id}_sd3_vae.npy
```

```bash
accelerate launch train_scripts/train_controlnet_sim.py \
    configs/config_controlnet_sim.py \
    --work-dir checkpoints/pixcell_controlnet_sim
```

### Stage 2 — Experimental fine-tuning

Fine-tune on paired ORION-CRC data (required for sim→exp mapping):

```bash
accelerate launch train_scripts/train_controlnet_exp.py \
    configs/config_controlnet_exp.py \
    --work-dir checkpoints/pixcell_controlnet_exp
```

To start from a sim checkpoint, set in `configs/config_controlnet_exp.py`:

```python
resume_from           = "./checkpoints/pixcell_controlnet_sim/checkpoints/step_0050000"
resume_tme_checkpoint = "./checkpoints/pixcell_controlnet_sim/checkpoints/step_0050000"
```

### CLI options

| Flag | Description |
|------|-------------|
| `--work-dir PATH` | Output directory for checkpoints and logs |
| `--resume-from PATH` | Resume ControlNet from checkpoint directory |
| `--load-from PATH` | Load specific checkpoint file |
| `--batch-size N` | Override `train_batch_size` in config |
| `--report-to tensorboard` | Logging backend (default: `tensorboard`) |
| `--tracker-project-name NAME` | Project name for the tracker |
| `--debug` | Run with minimal steps for debugging |

### Multi-GPU

```bash
accelerate config   # configure once
accelerate launch --num_processes 4 train_scripts/train_controlnet_exp.py \
    configs/config_controlnet_exp.py \
    --work-dir checkpoints/pixcell_controlnet_exp
```

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

Checkpoints are saved every `save_model_steps` steps (default 10,000) under:

```
checkpoints/pixcell_controlnet_exp/checkpoints/step_XXXXXXX/
├── epoch_X_step_XXXXXXX.pth    # ControlNet weights
└── tme_module.pth              # TME encoder weights
```

---

## 🔬 Inference

### Style-conditioned (reference H&E + TME channels)

Use a reference H&E tile's UNI embedding to set the staining style:

```python
import torch
import numpy as np
from diffusers import DDPMScheduler
from train_scripts.inference_controlnet import (
    load_vae, load_controlnet_model_from_checkpoint,
    load_pixcell_controlnet_model_from_checkpoint, denoise,
)
from train_scripts.train_controlnet_sim import load_sim_checkpoint
from diffusion.model.builder import build_model
from diffusion.utils.misc import read_config

config = read_config("configs/config_controlnet_exp.py")
device = "cuda"

vae        = load_vae(config.vae_pretrained, device)
controlnet = load_controlnet_model_from_checkpoint(
    "configs/config_controlnet_exp.py",
    "checkpoints/pixcell_controlnet_exp/checkpoints/step_0010000",
    device,
)
base_model = load_pixcell_controlnet_model_from_checkpoint(
    "configs/config_controlnet_exp.py",
    "checkpoints/pixcell_controlnet_exp/checkpoints/step_0010000",
)
base_model.to(device).eval()

tme_module = build_model("TMEConditioningModule", False, False,
                          n_tme_channels=9, base_ch=32)
load_sim_checkpoint(
    "checkpoints/pixcell_controlnet_exp/checkpoints/step_0010000",
    tme_module, device=device,
)
tme_module.to(device).eval()

# Load reference UNI embedding (from a real H&E tile)
ref_uni = np.load("path/to/reference_uni.npy")        # shape [1536]
uni_embeds = torch.from_numpy(ref_uni).view(1, 1, 1, 1536).to(device, torch.float16)
```

### TME-only (no style reference)

Generate purely from TME layout, without a reference H&E:

```python
from train_scripts.inference_controlnet import null_uni_embed

uni_embeds = null_uni_embed(device='cuda', dtype=torch.float16)  # shape [1, 1, 1, 1536]
```

---

## ✅ Validation

Evaluate sim→exp domain alignment: generate H&E from simulation TME channels and measure cosine similarity against precomputed experimental UNI features.

```bash
python validate_sim_to_exp.py \
    --config          configs/config_controlnet_exp.py \
    --sim-root        /path/to/sim_data_root \
    --exp-feat        /path/to/exp_data_root/features \
    --controlnet-ckpt checkpoints/pixcell_controlnet_exp/checkpoints/step_0010000 \
    --tme-ckpt        checkpoints/pixcell_controlnet_exp/checkpoints/step_0010000 \
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

## 📦 Pretrained Weights

| Model | Purpose | Source |
|-------|---------|--------|
| PixCell-256 transformer | Frozen base model | [HuggingFace](https://huggingface.co/StonyBrook-CVLab/PixCell-256) |
| PixCell-256 ControlNet | Initial ControlNet weights | [HuggingFace](https://huggingface.co/StonyBrook-CVLab/PixCell-256) |
| SD3.5 VAE | Image encoder/decoder | [HuggingFace](https://huggingface.co/stabilityai/stable-diffusion-3.5-large) |
| UNI-2h | Feature extractor | [HuggingFace](https://huggingface.co/MahmoodLab/UNI2-h) |

Download and organize under `pretrained_models/`:

```
pretrained_models/
├── pixcell-256/transformer/
├── pixcell-256-controlnet/controlnet/
├── sd-3.5-vae/vae/
└── uni-2h/
```

Run `python setup_pretrained_model.py` to download automatically.

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
