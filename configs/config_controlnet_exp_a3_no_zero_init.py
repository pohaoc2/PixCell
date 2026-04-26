"""
config_controlnet_exp_a3_no_zero_init.py

A3 design-justification ablation: zero_init_conv_out=False (no residual
gating). Inherits config_controlnet_exp.py and only overrides the ControlNet
initialization config and work_dir.
"""

_base_ = ["./config_controlnet_exp.py"]

controlnet_config = dict(
    zero_init_conv_out=False,
    copy_base_layers=True,
    conditioning_scale=1.0,
)
model_kwargs = dict(
    use_controlnet=True,
    controlnet_config=controlnet_config,
)

work_dir = "./checkpoints/pixcell_controlnet_exp_a3_no_zero_init"
seed = 42
