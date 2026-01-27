"""
Configuration for PixCell-256 ControlNet Training

Uses pretrained PixCell-256 base model from:
https://huggingface.co/StonyBrook-CVLab/PixCell-256

Assumes you have:
1. Binary cell masks already extracted
2. VAE features pre-computed (sd3_vae)
3. UNI embeddings pre-computed
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
    mask_prefix="cellvit_mask",  # Your binary masks
    patch_names_file="patch_names_controlnet.hdf5",  # Your image list
)

# Model setting - ControlNet variant
model = "PixArt_XL_2_UNI_ControlNet"
model_max_length = 1

mixed_precision = 'fp16'  
fp32_attention = True

# Load pretrained PixCell-256 base model from HuggingFace
# Download first using: python setup_pretrained_model.py
load_from = f"{root}/pretrained_models/pixcell-256/model.pth"
resume_from = None

vae_pretrained = f"{root}/pretrained_models/sd-3.5-vae"
pe_interpolation = 0.5

# Training setting - following paper specifications
num_workers = 16
train_batch_size = 4  # Paper uses batch size of 4
num_epochs = 25  # ~25,000 iterations with 10k dataset
gradient_accumulation_steps = 1
grad_checkpointing = True
gradient_clip = 0.01

# AdamW optimizer with lr=1e-5 as specified in paper
optimizer = dict(
    type='AdamW',
    lr=1e-5,
    weight_decay=0.0, 
    betas=(0.9, 0.999),
    eps=1e-8
)

lr_schedule_args = dict(num_warmup_steps=500)
auto_lr = None

log_interval = 20
save_model_epochs = 1
save_model_steps = 1000
work_dir = f"{root}/checkpoints/pixcell_controlnet"

# VAE configuration (SD3 VAE)
scale_factor = 1.5305
shift_factor = 0.0609

# ControlNet specific settings
controlnet_config = dict(
    zero_init_conv_out=True,
    copy_base_layers=True,
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

# Data root
data_root = "./"
