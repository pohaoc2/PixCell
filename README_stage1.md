# Stage 1: Model Setup and Feature Extraction

This guide covers Stage 0 and Stage 1 of the PixCell pipeline:

1. Download pretrained models with `stage0_setup.py`.
2. Cache UNI-2h embeddings and SD3.5 VAE latents with `stage1_extract_features.py`.
3. Build the HDF5 metadata index used during training.

For the project overview and installation, see [`README.md`](README.md).

---

## Stage 0: Model Setup

Download all pretrained models:

```bash
python stage0_setup.py
```

This downloads and organizes under `pretrained_models/`:

```text
pretrained_models/
├── pixcell-256/transformer/           # frozen base diffusion transformer
├── pixcell-256-controlnet/controlnet/ # ControlNet initialization
├── sd-3.5-vae/vae/                    # image encoder / decoder
└── uni-2h/                            # histopathology feature extractor
```

Selective download:

```bash
python stage0_setup.py --model pixcell
python stage0_setup.py --model pixcell-controlnet
python stage0_setup.py --model uni2h
python stage0_setup.py --model sd3_vae
```

Requires a HuggingFace token with access to gated models:

```bash
export HF_TOKEN=hf_...
```

---

## Pretrained Weights

| Model | Purpose | Source |
|-------|---------|--------|
| PixCell-256 transformer | Frozen base model | [HuggingFace](https://huggingface.co/StonyBrook-CVLab/PixCell-256) |
| PixCell-256 ControlNet | Initial ControlNet weights | [HuggingFace](https://huggingface.co/StonyBrook-CVLab/PixCell-256) |
| SD3.5 VAE | Image encoder/decoder | [HuggingFace](https://huggingface.co/stabilityai/stable-diffusion-3.5-large) |
| UNI-2h | Feature extractor | [HuggingFace](https://huggingface.co/MahmoodLab/UNI2-h) |

Download automatically with `python stage0_setup.py`.

---

## Pretrained Inference Check

[`verify_pretrained_inference.py`](verify_pretrained_inference.py) loads the public PixCell-256 transformer and ControlNet weights, runs a short denoising pass on a mask + reference H&E, and writes a comparison figure. This is a quick sanity check before training or fine-tuned inference.

**Prerequisites:** `python stage0_setup.py` and sample inputs under `inference_data/`, or your own paths.

Minimal run:

```bash
python verify_pretrained_inference.py
```

Explicit paths:

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
| `--controlnet-safetensors` | `pretrained_models/pixcell-256-controlnet/controlnet/diffusion_pytorch_model.safetensors` | Init ControlNet weights |
| `--vae-path` | `pretrained_models/sd-3.5-vae/vae` | SD3.5 VAE directory |
| `--uni-model-path` | `pretrained_models/uni-2h` | UNI-2h for reference style embedding |
| `--mask-path` | `inference_data/test_mask.png` | Cell mask image (RGB PNG) |
| `--reference-he` | `inference_data/test_control_image.png` | Reference H&E for UNI conditioning / plot |
| `--reference-uni` | `inference_data/test_control_image_uni.npy` | Cached UNI vector (created if missing) |
| `--mask-latent` | `inference_data/test_mask_sd3_vae.npy` | Cached VAE latent for mask (created if missing) |
| `--generated-output` | `inference_data/generated_he_pretrained_test_mask.png` | Saved generated H&E |
| `--output-path` | `inference_data/vis_pretrained_verification_test_mask.png` | Side-by-side comparison figure |

---

## Stage 1: Feature Extraction

Extract UNI-2h embeddings and SD3.5 VAE latents from paired experimental H&E tiles. These features are cached once and reused during training.

**Pass 1: H&E images** (produces UNI embeddings + VAE latents):

```bash
python stage1_extract_features.py \
    --image-dir  ./data/exp_paired/he_images \
    --output-dir ./data/exp_paired/features
```

**Pass 2: `cell_mask` images** (produces VAE latents used for conditioning):

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
| `{stem}_mask_sd3_vae.npy` | `[2, 16, H/8, W/8]` | `cell_mask` VAE latent |

---

## Build HDF5 Index

```python
from diffusion.data.datasets.paired_exp_controlnet_dataset import build_exp_index

build_exp_index(
    exp_channels_dir="data/exp_paired/exp_channels",
    output_path="data/exp_paired/metadata/exp_index.hdf5",
)
```

---

## Experimental Channel Layout

| Channel | Group | Source | Type |
|---------|-------|--------|------|
| `cell_mask` | *(always present)* | Cell segmentation | Binary |
| `cell_type_healthy` | `cell_types` | CODEX multi-protein panel | Binary one-hot |
| `cell_type_cancer` | `cell_types` | CODEX multi-protein panel | Binary one-hot |
| `cell_type_immune` | `cell_types` | CODEX multi-protein panel | Binary one-hot |
| `cell_state_prolif` | `cell_state` | CODEX multi-protein panel | Binary one-hot |
| `cell_state_nonprolif` | `cell_state` | CODEX multi-protein panel | Binary one-hot |
| `cell_state_dead` | `cell_state` | CODEX multi-protein panel | Binary one-hot |
| `vasculature` | `vasculature` | CD31 marker (CODEX) | Float `[0,1]` |
| `oxygen` | `microenv` | PDE model (distance to vasculature) | Float `[0,1]` |
| `glucose` | `microenv` | PDE model (distance to vasculature) | Float `[0,1]` |

Each group has independent dropout during training. Groups can also be selectively included or excluded later during Stage 3 inference.

---

## Experimental Dataset Directory Layout

```text
exp_data_root/
├── metadata/
│   └── exp_index.hdf5              # HDF5: "exp_256" -> list of tile_ids
├── exp_channels/
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
│   └── {tile_id}_uni.npy           # UNI-2h embedding [1536]
└── vae_features/
    ├── {tile_id}_sd3_vae.npy       # VAE latent [2, 16, 32, 32]
    └── {tile_id}_mask_sd3_vae.npy  # cell_mask VAE latent
```

Next step: move to [`stage2.md`](stage2.md) to train the ControlNet + TME module.
