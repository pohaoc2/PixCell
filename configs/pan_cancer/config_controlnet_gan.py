"""
Full Configuration for PixCell-256 ControlNet with ALL enhancements:
1. ControlNet conditioning (UNI embeddings + cell masks)
2. Adversarial training (GAN discriminator)
3. Cell segmentation consistency (ensures mask adherence)

This configuration provides the highest quality generation with strict
adherence to provided cell masks.
"""

_base_ = ['../PixArt_xl2_internal.py']

image_size = 256
#root = "/home/pohaoc2/UW/bagherilab/PixCell"
root = "./"

# Dataset configuration
data = dict(
    type="PanCancerControlNetData",
    root=root,
    resolution=image_size,
    vae_prefix="sd3_vae",
    ssl_prefix="uni",
    mask_prefix="mask",
    patch_names_file="patch_names_controlnet.hdf5",
)

# Model setting
model = "PixArt_XL_2_UNI_ControlNet"
#model = "PixArt_XL_2_UNI"

model_path = f"{root}/pretrained_models/pixcell-256/transformer"
model_max_length = 1

mixed_precision = 'bf16'  
fp32_attention = True

# Load pretrained PixCell-256 base model
load_from = f"{root}/pretrained_models/pixcell-256/transformer"
resume_from = None

vae_pretrained = f"{root}/pretrained_models/sd-3.5-vae/vae"
pe_interpolation = 0.5

# Training setting
num_workers = 2
train_batch_size = 64
num_epochs = 3
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

log_interval = 100
save_model_epochs = 3
save_model_steps = 5000
work_dir = f"{root}/checkpoints/pixcell_controlnet_full"

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
# Discriminator Configuration (for perceptual quality)
# =====================================================================
use_discriminator = False #True

discriminator = dict(
    type='latent',  # Efficient latent-space discrimination
    latent_channels=16,
    num_filters=64,
    use_spectral_norm=True,
    loss_type='hinge',
)

discriminator_optimizer = dict(
    type='AdamW',
    lr=4e-5,  # 4x generator lr
    weight_decay=0.0,
    betas=(0.5, 0.999),
    eps=1e-8
)

adversarial_weight = 0.1  # Balance with diffusion loss

# =====================================================================
# Cell Segmentation Consistency (for mask adherence)
# =====================================================================
use_segmentation_consistency = False #True

# Cellpose configuration
# Model types: 'cyto', 'cyto2', 'cyto3', 'nuclei'
# - 'cyto2': General purpose cytoplasm model (recommended for histology)
# - 'nuclei': Specialized for nucleus segmentation
# - 'cyto3': Latest cytoplasm model with improved performance
cellpose_model_type = 'cyto2'  

# Expected cell diameter in pixels (for 20x magnification, ~30 pixels is typical)
# Set to None for automatic diameter estimation
cell_diameter = 30

# Consistency loss configuration
consistency_loss_type = 'combined'  # 'bce', 'dice', 'focal', or 'combined'
consistency_weight = 0.5  # Weight for mask consistency loss
consistency_check_interval = 10  # Compute every N steps (saves computation)

# =====================================================================

# Class dropout for classifier-free guidance
class_dropout_prob = 0.1

# EMA configuration
ema_rate = 0.9999

# Training sampling steps
train_sampling_steps = 500
snr_loss = True

seed = 42

# Additional model kwargs
model_kwargs = dict(
    use_controlnet=True,
    controlnet_config=controlnet_config,
)

# Data root
data_root = "./"

# =====================================================================
# Loss Weights Summary
# =====================================================================
# Total Loss = loss_diffusion + λ_adv * loss_adv + λ_cons * loss_consistency
#
# loss_diffusion:     Main diffusion loss (MSE on noise prediction)
# loss_adv:          Adversarial loss (fool discriminator)           λ_adv = 0.1
# loss_consistency:  Mask consistency (segmentation match)           λ_cons = 0.5
#
# Recommended ranges:
# - adversarial_weight: 0.05 - 0.2 (higher = more realistic texture)
# - consistency_weight: 0.3 - 1.0 (higher = stricter mask adherence)
#
# Start with these values and tune based on:
# - If masks not respected → increase consistency_weight
# - If images blurry → increase adversarial_weight  
# - If training unstable → decrease both weights
# =====================================================================