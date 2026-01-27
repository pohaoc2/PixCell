# PixCell-256 ControlNet Training

This implementation adapts the PixCell training pipeline for ControlNet-based conditional generation with cell segmentation masks.

## Overview

Based on *Yellapragada et al. - 2025 - PixCell: A Generative Foundation Model for Digital Histopathology Images*, this training script implements a ControlNet for PixCell-256 to guide generation with cell layout masks.

### Key Features from Paper

1. **Cell Mask Extraction**: Uses pre-trained CellViT-SAM-H model trained on 20× (0.5 μm/px) pathology images
2. **Training Data**: 10,000 images randomly sampled from PanCan-30M with their UNI-2h embeddings
3. **Training Triplets**: (image, UNI embedding, cell mask)
4. **Architecture**: ControlNet copies each layer of the base transformer with zero-initialized intermediate output layers
5. **Training Specs**: 
   - 25,000 iterations
   - Batch size: 4
   - Optimizer: AdamW with lr=1×10⁻⁵

## File Structure

```
.
├── train_controlnet.py              # Main training script
├── pan_cancer_controlnet.py         # Dataset class with mask loading
├── pixart_20x_256_controlnet.py     # Configuration file
└── README_controlnet.md             # This file
```

## Key Modifications from Base Training

### 1. Training Script (`train_controlnet.py`)

**Cell Mask Processing** (lines 91-103):
```python
# Process cell mask for ControlNet
if cell_mask.shape[-1] != clean_images.shape[-1]:
    kernel_size = cell_mask.shape[-1] // clean_images.shape[-1]
    cell_mask = nn.functional.max_pool2d(
        cell_mask.float(), 
        kernel_size=kernel_size, 
        stride=kernel_size
    )
```
- Downsamples binary cell masks from image resolution (256×256) to latent resolution (32×32)
- Uses max pooling to preserve cell presence information

**ControlNet Conditioning** (lines 105-119):
```python
model_kwargs = dict(
    y=y,  # UNI embeddings
    mask=None,  # No attention mask needed (model_max_length=1)
    data_info=data_info,
    controlnet_cond=cell_mask  # Cell mask for ControlNet
)

loss_term = train_diffusion.training_losses(
    model, 
    clean_images, 
    timesteps, 
    model_kwargs=model_kwargs
)
```
- Passes cell mask as `controlnet_cond` to the model
- Maintains UNI embeddings for style conditioning

### 2. Dataset Class (`pan_cancer_controlnet.py`)

**New Features**:
- `_mask_loader()`: Loads binary cell masks from .npy or .png files
- `mask_prefix`: Configuration for cell mask file naming (default: "cellvit_mask")
- `masks_dir`: Separate directory for storing extracted cell masks
- Returns 4-tuple: `(vae_feat, ssl_feat, cell_mask, data_info)`

**Data Format**:
```python
vae_feat:   [16, 32, 32]      # SD3 VAE latents
ssl_feat:   [256]             # UNI-2h embeddings (or appropriate dim)
cell_mask:  [1, 256, 256]     # Binary cell segmentation mask
data_info:  dict              # Metadata
```

### 3. Configuration (`pixart_20x_256_controlnet.py`)

**Paper-Specified Hyperparameters**:
```python
train_batch_size = 4        # Paper: batch size of 4
num_epochs = 25             # ~25,000 iterations with 10k dataset
optimizer = dict(
    type='AdamW',           # Paper: AdamW optimizer
    lr=1e-5,                # Paper: 1×10⁻⁵
)
```

**ControlNet Configuration**:
```python
controlnet_config = dict(
    zero_init_conv_out=True,     # Zero-initialize output layers
    copy_base_layers=True,        # Copy from base transformer
    conditioning_scale=1.0,       # Conditioning strength
)
```

## Usage

### Prerequisites

1. **Pretrained Base Model**: Train PixCell-256 base model first
2. **Cell Masks**: Extract cell masks using CellViT-SAM-H:
   ```bash
   # Example mask extraction (you'll need CellViT-SAM-H)
   python extract_cellvit_masks.py \
       --input_dir patches/ \
       --output_dir masks/ \
       --model cellvit_sam_h_20x
   ```

3. **Dataset Preparation**:
   - Create `patch_names_controlnet.hdf5` with 10k sampled images
   - Organize masks in `masks/` directory
   - Ensure naming convention: `{image_name}_cellvit_mask.png`

### Training

```bash
# Single GPU
python train_controlnet.py pixart_20x_256_controlnet.py

# Multi-GPU (DDP)
accelerate launch --config_file accelerate_config.yaml \
    train_controlnet.py pixart_20x_256_controlnet.py

# With custom work directory
python train_controlnet.py \
    pixart_20x_256_controlnet.py \
    --work-dir ./experiments/controlnet_run1

# Resume training
python train_controlnet.py \
    pixart_20x_256_controlnet.py \
    --resume-from ./checkpoints/pixart_20x_256_controlnet/checkpoints/
```

### Key Arguments

- `--load-from`: Path to pretrained base model checkpoint
- `--resume-from`: Resume from checkpoint (auto-finds latest if directory)
- `--batch-size`: Override config batch size
- `--slurm-time-limit`: Save checkpoint before time limit (minutes)
- `--debug`: Enable debug mode (smaller batch, frequent logging)
- `--report-to`: Logging backend (tensorboard/wandb)

## Architecture Details

### ControlNet Integration

The ControlNet architecture follows the paper's design:

1. **Layer Copying**: Each transformer block is copied from the base model
2. **Zero Initialization**: Output projection layers are zero-initialized
3. **Feature Addition**: ControlNet features are added to base features via:
   ```python
   base_output + zero_conv(controlnet_output)
   ```

### Training Flow

```
Input Image (256×256)
    ↓
VAE Encode → Latent (32×32)
    ↓
Add Noise (Diffusion)
    ↓
┌─────────────────┬──────────────────┐
│  Base Transformer  │  ControlNet Transformer │
│  + UNI Embedding  │  + Cell Mask        │
└─────────────────┴──────────────────┘
    ↓         ↓
    └─────── + ────────┐
             ↓
        Predict Noise
             ↓
          Loss (MSE)
```

### Conditioning Mechanism

**Dual Conditioning**:
1. **Style**: UNI embeddings control tissue appearance and staining
2. **Layout**: Cell masks guide spatial arrangement of cells

This allows "fine control over the generated image: the UNI embedding dictates the style of the generated image while the cell mask guides the cell layout" (paper).

## Model Architecture Notes

### Required Model Implementation

You'll need to implement or modify the base model to support ControlNet:

```python
class PixArt_XL_2_UNI_ControlNet(nn.Module):
    def __init__(self, ...):
        super().__init__()
        # Base transformer
        self.base_model = PixArt_XL_2_UNI(...)
        
        # ControlNet transformer (copy of base)
        self.controlnet = deepcopy(self.base_model.blocks)
        
        # Zero-initialized output convolutions
        self.controlnet_outputs = nn.ModuleList([
            zero_module(nn.Linear(hidden_dim, hidden_dim))
            for _ in range(len(self.base_model.blocks))
        ])
        
    def forward(self, x, timesteps, y, controlnet_cond=None, **kwargs):
        # Encode cell mask through ControlNet path
        if controlnet_cond is not None:
            controlnet_feats = self.encode_controlnet(controlnet_cond)
        
        # Base path with UNI conditioning
        for i, block in enumerate(self.base_model.blocks):
            x = block(x, timesteps, y, **kwargs)
            
            # Add ControlNet features
            if controlnet_cond is not None:
                cn_feat = self.controlnet[i](
                    controlnet_feats, timesteps, y, **kwargs
                )
                x = x + self.controlnet_outputs[i](cn_feat)
        
        return x
```

## Expected Results

Following the paper's training setup should yield:
- **Controllable Generation**: Cell positions follow the input mask
- **Style Consistency**: UNI embeddings maintain tissue appearance
- **Fine-grained Control**: Individual cell placement in mask is respected

## Troubleshooting

### Common Issues

1. **Mask Resolution Mismatch**:
   - Ensure masks are 256×256 at image resolution
   - Script auto-downsamples to 32×32 latent resolution

2. **Memory Issues**:
   - Reduce batch size (paper uses 4)
   - Enable gradient checkpointing (default: enabled)
   - Use mixed precision training (default: fp16)

3. **Convergence Issues**:
   - Verify base model is properly loaded
   - Check cell mask quality (should be binary)
   - Ensure UNI embeddings are correct dimension

### Validation

To validate training:
```python
# Check mask loading
dataset = PanCancerControlNetData(root="./", resolution=256)
vae_feat, ssl_feat, mask, info = dataset[0]
print(f"VAE: {vae_feat.shape}, SSL: {ssl_feat.shape}, Mask: {mask.shape}")

# Expected output:
# VAE: torch.Size([16, 32, 32]), SSL: torch.Size([256]), Mask: torch.Size([1, 256, 256])
```

## Citation

If you use this implementation, please cite:

```bibtex
@article{yellapragada2025pixcell,
  title={PixCell: A Generative Foundation Model for Digital Histopathology Images},
  author={Yellapragada, et al.},
  year={2025}
}
```

## Additional Notes

### Data Preparation Pipeline

1. **Sample Images**: Randomly select 10k images from PanCan-30M
2. **Extract Features**:
   - VAE: Use SD3.5 VAE encoder
   - SSL: Extract UNI-2h embeddings
3. **Generate Masks**: Run CellViT-SAM-H on 20× images
4. **Create Metadata**: Build HDF5 file with image lists

### Inference with ControlNet

After training, use the ControlNet for controlled generation:

```python
# Load models
model = load_trained_controlnet(checkpoint_path)
cell_mask = load_mask("path/to/mask.png")
uni_embedding = extract_uni_embedding(reference_image)

# Generate with control
synthetic_image = model.sample(
    mask=cell_mask,
    embedding=uni_embedding,
    num_inference_steps=14,
    guidance_scale=4.5
)
```

### Scaling to Higher Resolutions

For PixCell-512 or PixCell-1024 ControlNet:
- Adjust `image_size` in config
- Retrain CellViT or use multi-scale masks
- May need to adjust ControlNet architecture layers
