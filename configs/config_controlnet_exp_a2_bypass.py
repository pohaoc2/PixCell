"""
config_controlnet_exp_a2_bypass.py

A2 design-justification ablation: zero_mask_latent=False (additive TME,
bypass-capable). Inherits config_controlnet_exp.py and overrides only the flag
and the work_dir to avoid overwriting the production checkpoints.
"""

_base_ = ["./config_controlnet_exp.py"]

zero_mask_latent = False

work_dir = "./checkpoints/pixcell_controlnet_exp_a2_bypass"
seed = 42
