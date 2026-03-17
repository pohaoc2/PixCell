"""
config_controlnet_exp.py

Configuration for PixCell-256 ControlNet fine-tuning on paired experimental data
(H&E + CODEX-derived TME channels).

Inherits base PixArt settings; overrides dataset, channel layout, and training knobs.
Set exp_data_root to your actual paired dataset path before running.
"""

_base_ = ['./PixArt_xl2_internal.py']
image_size = 256
root = "./"

# =====================================================================
# Dataset — PairedExpControlNetData
# =====================================================================
data = dict(
    type="PairedExpControlNetData",
    resolution=image_size,
    exp_channels_dir="exp_channels",
    features_dir="features",
    vae_features_dir="vae_features",
    exp_index_h5="metadata/exp_index.hdf5",
    vae_prefix="sd3_vae",
    ssl_prefix="uni",
    active_channels=[
        "cell_mask",
        # One-hot cell type (healthy=0, cancer=1, immune=2)
        "cell_type_healthy", "cell_type_cancer", "cell_type_immune",
        # One-hot cell state (prolif=0, nonprolif=1, dead=2)
        "cell_state_prolif",  "cell_state_nonprolif", "cell_state_dead",
        # Approximate channels derived from CODEX (CD31, Ki67+dist, metabolic model)
        "vasculature", "oxygen", "glucose",
    ],
)

# Root of the paired experimental dataset — set before training
exp_data_root = f"{root}/data/exp_paired"

# =====================================================================
# TME Encoder
# =====================================================================
# n_tme_channels = 9  (all active_channels except cell_mask)
tme_model   = "TMEConditioningModule"
tme_base_ch = 32
tme_lr      = 1e-5

# =====================================================================
# Experimental training knobs
# =====================================================================
# CFG dropout: fraction of steps where the UNI embedding is zeroed.
# Enables TME-only inference (null_uni_embed) at no extra training cost.
cfg_dropout_prob = 0.15

# Per-tme-channel reliability weights (9 values, matching tme_channels order):
#   [cell_type_healthy, cell_type_cancer, cell_type_immune,        → 1.0 (pixel-perfect)
#    cell_state_prolif, cell_state_nonprolif, cell_state_dead,     → 1.0 (pixel-perfect)
#    vasculature, oxygen, glucose]                                 → 0.5 (CODEX approximation)
channel_reliability_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5]

# =====================================================================
# Model
# =====================================================================
base_model       = "PixArt_XL_2_UNI"
base_model_path  = f"{root}/pretrained_models/pixcell-256/transformer"
model_max_length = 1

controlnet_model                 = "PixCell_ControlNet_XL_2_UNI"
controlnet_depth                 = 27
controlnet_conditioning_channels = 16
controlnet_conditioning_scale    = 1.0
controlnet_load_from             = f"{root}/pretrained_models/pixcell-256-controlnet/controlnet/diffusion_pytorch_model.safetensors"
load_from   = f"{root}/pretrained_models/pixcell-256/transformer"

# To resume from a sim ControlNet checkpoint:
# resume_from = f"{root}/checkpoints/pixcell_controlnet_sim/checkpoints/step_XXXXXXX"
# resume_tme_checkpoint = f"{root}/checkpoints/pixcell_controlnet_sim/checkpoints/step_XXXXXXX"
resume_from = None

vae_pretrained   = f"{root}/pretrained_models/sd-3.5-vae/vae"
pe_interpolation = 0.5

mixed_precision  = 'no'
fp32_attention   = True

# =====================================================================
# Training
# =====================================================================
num_workers                 = 4
train_batch_size            = 4
num_epochs                  = 200
gradient_accumulation_steps = 1
grad_checkpointing          = True
gradient_clip               = 1.0

optimizer = dict(
    type='AdamW',
    lr=5e-6,            # lower than sim (1e-5) — fine-tuning from sim checkpoint
    weight_decay=0.0,
    betas=(0.9, 0.999),
    eps=1e-8,
)

lr_schedule_args = dict(num_warmup_steps=500)
auto_lr          = None

log_interval       = 100
save_model_epochs  = 50
save_model_steps   = 10000
work_dir           = f"{root}/checkpoints/pixcell_controlnet_exp"

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
