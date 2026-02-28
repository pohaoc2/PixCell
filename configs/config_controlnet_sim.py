"""
Configuration for PixCell-256 ControlNet with TME conditioning (unpaired sim input).
Uses SimControlNetData with dummy dataset for testing.
"""

_base_ = ['./PixArt_xl2_internal.py']
image_size = 256
root = "/home/ec2-user/PixCell"
#root = "./"
# =====================================================================
# Dataset — SimControlNetData
# =====================================================================
data = dict(
    type="SimControlNetData",
    resolution=image_size,
    # Paths (all relative to sim_data_root)
    sim_channels_dir="sim_channels",
    features_dir="features",
    vae_features_dir="vae_features",
    sim_index_h5="metadata/sim_index.hdf5",
    real_index_h5="metadata/real_index.hdf5",
    # Feature file suffixes
    vae_prefix="sd3_vae",
    ssl_prefix="uni",
    # Active simulation channels — cell_mask and oxygen are required
    active_channels=["cell_mask", "oxygen", "glucose", "tgf"],
)

# Root of the dummy dataset (created by dummy_sim_generator.py)
sim_data_root = f"{root}/dummy_sim_data"

# =====================================================================
# TME Encoder
# =====================================================================
# n_tme_channels is derived automatically as len(active_channels) - 1
tme_model = "TMEConditioningModule"
tme_base_ch = 32       # CNN base channel width; increase to 64 if GPU allows
tme_lr      = 1e-4     # LR for TME module; can match controlnet LR

# =====================================================================
# Model
# =====================================================================
base_model       = "PixArt_XL_2_UNI"
base_model_path  = f"{root}/pretrained_models/pixcell-256/transformer"
model_max_length = 1

controlnet_model                    = "PixCell_ControlNet_XL_2_UNI"
controlnet_depth                    = 27
controlnet_conditioning_channels    = 16
controlnet_conditioning_scale       = 1.0
pixcell_controlnet_module_name      = "pixcell_controlnet_transformer"
pixcell_controlnet_file_path        = f"{root}/pretrained_models/pixcell-256-controlnet/transformer/pixcell_controlnet_transformer.py"
pixcell_controlnet_checkpoints_folder = f"{root}/pretrained_models/pixcell-256-controlnet/transformer/"
controlnet_module_name              = "pixcell_controlnet"
controlnet_file_path                = f"{root}/pretrained_models/pixcell-256-controlnet/controlnet/pixcell_controlnet.py"
controlnet_checkpoints_folder       = f"{root}/pretrained_models/pixcell-256-controlnet/controlnet/"
controlnet_load_from                = f"{root}/train_scripts/controlnet_mapped_weights.pt"
load_from   = f"{root}/pretrained_models/pixcell-256/transformer"
resume_from = None
# To resume TME module from a checkpoint:
# resume_tme_checkpoint = f"{root}/checkpoints/pixcell_controlnet_sim/checkpoints/step_0025000"

vae_pretrained   = f"{root}/pretrained_models/sd-3.5-vae/vae"
pe_interpolation = 0.5

mixed_precision  = 'no'
fp32_attention   = True

# =====================================================================
# Training
# =====================================================================
num_workers                 = 2
train_batch_size            = 4
num_epochs                  = 1
gradient_accumulation_steps = 1
grad_checkpointing          = True
gradient_clip               = 1.0

optimizer = dict(
    type='AdamW',
    lr=1e-5,
    weight_decay=0.0,
    betas=(0.9, 0.999),
    eps=1e-8,
)

lr_schedule_args = dict(num_warmup_steps=1000)
auto_lr          = None

log_interval       = 100
save_model_epochs  = 500
save_model_steps   = 50000
work_dir           = f"{root}/checkpoints/pixcell_controlnet_sim"

# =====================================================================
# VAE
# =====================================================================
scale_factor = 1.5305
shift_factor = 0.0609

# =====================================================================
# Misc
# =====================================================================
class_dropout_prob   = 0.1
ema_rate             = 0.9999
train_sampling_steps = 500
snr_loss             = True
seed                 = 42
data_root            = "./"

controlnet_config = dict(
    zero_init_conv_out=True,
    copy_base_layers=True,
    conditioning_scale=1.0,
)

model_kwargs = dict(
    use_controlnet=True,
    controlnet_config=controlnet_config,
)