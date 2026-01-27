"""
Configuration for PixCell-256 ControlNet Training

Based on the paper:
- CellViT-SAM-H for cell mask extraction (trained on 0.5 μm/px images)
- 10,000 images from PanCan-30M
- Training with (image, UNI embedding, mask) triplets
- 25,000 iterations
- Batch size: 4
- AdamW optimizer with learning rate 1e-5
- ControlNet copies each layer of base transformer
- Zero-initialized intermediate output linear layer
"""

_base_ = ['../PixArt_xl2_internal.py']

image_size = 256
root = "./"

# Dataset configuration - use ControlNet-specific dataset
data = dict(
    type="PanCancerControlNetData",
    root=root,
    resolution=image_size,
    vae_prefix="sd3_vae",
    ssl_prefix="uni",
    mask_prefix="cellvit_mask",  # Cell masks from CellViT-SAM-H
    patch_names_file="patch_names_controlnet.hdf5",  # Subset of 10k images
)

# Model setting - ControlNet variant
model = "PixArt_XL_2_UNI_ControlNet"  # Need to implement this
model_max_length = 1

mixed_precision = 'fp16'  
fp32_attention = True

# Load pretrained base model
load_from = f"{root}/checkpoints/pixart_20x_256_sd3_vae/checkpoints/epoch_10_step_XXXX.pth"
resume_from = None

vae_pretrained = f"{root}/pretrained_models/sd-3.5-vae"
pe_interpolation = 0.5

# Training setting - following paper specifications
num_workers = 16
train_batch_size = 4  # Paper uses batch size of 4
num_epochs = 25  # ~25,000 iterations with 10k images and batch size 4
gradient_accumulation_steps = 1
grad_checkpointing = True
gradient_clip = 0.01

# AdamW optimizer with lr=1e-5 as specified in paper
optimizer = dict(
    type='AdamW',  # Paper uses AdamW
    lr=1e-5,  # Paper specifies 1e-5
    weight_decay=0.0, 
    betas=(0.9, 0.999),
    eps=1e-8
)

lr_schedule_args = dict(num_warmup_steps=500)
auto_lr = None

log_interval = 20
save_model_epochs = 1
save_model_steps = 1000
work_dir = f"{root}/checkpoints/pixart_20x_256_controlnet"

# VAE configuration (SD3 VAE)
scale_factor = 1.5305
shift_factor = 0.0609

# ControlNet specific settings
controlnet_config = dict(
    # Zero-initialize the intermediate output layers
    zero_init_conv_out=True,
    # Copy each layer from base transformer
    copy_base_layers=True,
    # Conditioning scale (can be adjusted during inference)
    conditioning_scale=1.0,
)

# Class dropout for classifier-free guidance
class_dropout_prob = 0.1

# EMA configuration
ema_rate = 0.9999

# Training sampling steps
train_sampling_steps = 1000
snr_loss = True

seed = 42

# Additional model kwargs
model_kwargs = dict(
    use_controlnet=True,
    controlnet_config=controlnet_config,
)
