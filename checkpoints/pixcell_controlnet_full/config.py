data_root = './'
data = dict(
    type='PanCancerControlNetData',
    root='./',
    image_list_json=['data_info.json'],
    transform='default_train',
    load_vae_feat=True,
    load_t5_feat=True,
    resolution=256,
    vae_prefix='sd3_vae',
    ssl_prefix='uni',
    mask_prefix='mask',
    patch_names_file='patch_names_controlnet.hdf5')
image_size = 256
train_batch_size = 16
eval_batch_size = 16
use_fsdp = False
valid_num = 0
fp32_attention = True
model = 'PixArt_XL_2_UNI'
aspect_ratio_type = None
multi_scale = False
pe_interpolation = 0.5
qk_norm = False
kv_compress = False
kv_compress_config = dict(sampling=None, scale_factor=1, kv_compress_layer=[])
num_workers = 16
train_sampling_steps = 500
visualize = False
deterministic_validation = False
eval_sampling_steps = 250
model_max_length = 1
lora_rank = 4
num_epochs = 25
gradient_accumulation_steps = 1
grad_checkpointing = True
gradient_clip = 0.01
gc_step = 1
auto_lr = None
validation_prompts = [
    'dog',
    'portrait photo of a girl, photograph, highly detailed face, depth of field',
    'Self-portrait oil painting, a beautiful cyborg with golden hair, 8k',
    'Astronaut in a jungle, cold color palette, muted colors, detailed, 8k',
    'A photo of beautiful mountain with realistic sunset and blue lake, highly detailed, masterpiece'
]
optimizer = dict(
    type='AdamW', lr=1e-05, weight_decay=0.0, eps=1e-08, betas=(0.9, 0.999))
lr_schedule = 'constant'
lr_schedule_args = dict(num_warmup_steps=500)
save_image_epochs = 1
save_model_epochs = 5
save_model_steps = 500
sample_posterior = True
mixed_precision = 'fp16'
scale_factor = 1.5305
ema_rate = 0.9999
tensorboard_mox_interval = 50
log_interval = 20
cfg_scale = 4
mask_type = 'null'
num_group_tokens = 0
mask_loss_coef = 0.0
load_mask_index = False
vae_pretrained = './/pretrained_models/sd-3.5-vae/vae'
load_from = './/pretrained_models/pixcell-256/transformer'
resume_from = None
snr_loss = True
real_prompt_ratio = 1.0
class_dropout_prob = 0.1
work_dir = './/checkpoints/pixcell_controlnet_full'
s3_work_dir = None
micro_condition = False
seed = 42
skip_step = 0
loss_type = 'huber'
huber_c = 0.001
num_ddim_timesteps = 50
w_max = 15.0
w_min = 3.0
ema_decay = 0.95
root = './'
model_path = './/pretrained_models/pixcell-256/transformer'
shift_factor = 0.0609
controlnet_config = dict(
    zero_init_conv_out=True, copy_base_layers=True, conditioning_scale=1.0)
use_discriminator = False
discriminator = dict(
    type='latent',
    latent_channels=16,
    num_filters=64,
    use_spectral_norm=True,
    loss_type='hinge')
discriminator_optimizer = dict(
    type='AdamW', lr=4e-05, weight_decay=0.0, betas=(0.5, 0.999), eps=1e-08)
adversarial_weight = 0.1
use_segmentation_consistency = False
cellpose_model_type = 'cyto2'
cell_diameter = 30
consistency_loss_type = 'combined'
consistency_weight = 0.5
consistency_check_interval = 10
model_kwargs = dict(
    use_controlnet=True,
    controlnet_config=dict(
        zero_init_conv_out=True, copy_base_layers=True,
        conditioning_scale=1.0))
