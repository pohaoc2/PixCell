"""
config_controlnet_exp_a1_concat.py

A1 design-justification ablation: raw 10-channel concat ControlNet.
The ControlNet conditioning path consumes the full-resolution control tensor
directly; a pass-through shim preserves the existing training/inference API.
"""

_base_ = ["./config_controlnet_exp.py"]

channel_groups = None
zero_mask_latent = False

tme_input_mode = "all_channels"
tme_model = "RawConditioningPassthrough"

controlnet_model = "PixCell_ControlNet_XL_2_UNI_Concat"
controlnet_conditioning_channels = 10
controlnet_config = dict(
    zero_init_conv_out=True,
    copy_base_layers=True,
    conditioning_scale=1.0,
)
model_kwargs = dict(
    use_controlnet=True,
    controlnet_config=controlnet_config,
)
controlnet_model_kwargs = dict(
    conditioning_input_size=256,
    conditioning_patch_size=16,
)

work_dir = "./checkpoints/concat"
seed = 42