"""
Configuration for PixCell-256 ControlNet with Adversarial Training

Adds a discriminator to distinguish between real and synthetic images,
improving generation quality through adversarial loss.
"""

_base_ = ['../PixArt_xl2_internal.py']

image_size = 256
root = "./"

# Dataset configuration
data = dict(
    type="PanCancerControlNetData",
    root=root,
    resolution=image_size,
    vae_prefix="sd3_vae",
    ssl_prefix="uni",
    mask_prefix="cellvit_mask",
    patch_names_file="patch_names_controlnet.hdf5",
)

# Model setting
model = "PixArt_XL_2_UNI_ControlNet"
model_max_length = 1

mixed_precision = 'fp16'  
fp32_attention = True

# Load pretrained PixCell-256 base model
load_from = f"{root}/pretrained_models/pixcell-256/model.pth"
resume_from = None

vae_pretrained = f"{root}/pretrained_models/sd-3.5-vae"
pe_interpolation = 0.5

# Training setting
num_workers = 16
train_batch_size = 4  # Paper uses batch size of 4
num_epochs = 25
gradient_accumulation_steps = 1
grad_checkpointing = True
gradient_clip = 0.01

# AdamW optimizer for generator
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
work_dir = f"{root}/checkpoints/pixcell_controlnet_gan"

# VAE configuration
scale_factor = 1.5305
shift_factor = 0.0609

# ControlNet specific settings
controlnet_config = dict(
    zero_init_conv_out=True,
    copy_base_layers=True,
    conditioning_scale=1.0,
)

# =====================================================================
# Discriminator Configuration (NEW!)
# =====================================================================
use_discriminator = True  # Enable adversarial training

discriminator = dict(
    # Options: 'patchgan', 'latent', 'conditional'
    type='latent',  # Works in VAE latent space (more efficient)
    
    # Latent discriminator settings
    latent_channels=16,  # SD3 VAE has 16 latent channels
    num_filters=64,
    use_spectral_norm=True,  # Stabilizes training
    
    # Loss type: 'hinge', 'vanilla', 'lsgan'
    loss_type='hinge',  # Hinge loss (used in StyleGAN2)
)

# Discriminator optimizer (can be different from generator)
discriminator_optimizer = dict(
    type='AdamW',
    lr=4e-5,  # Typically 2-4x generator lr
    weight_decay=0.0,
    betas=(0.5, 0.999),  # Different beta1 for discriminator
    eps=1e-8
)

# Adversarial loss weight (balance with diffusion loss)
adversarial_weight = 0.1  # Start small, can increase to 0.5

# =====================================================================

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