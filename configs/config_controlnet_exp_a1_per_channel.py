"""
config_controlnet_exp_a1_per_channel.py

A1 design-justification ablation: one encoder per individual control channel,
without higher-level grouping.
"""

_base_ = ["./config_controlnet_exp.py"]

channel_groups = None
group_dropout_probs = {}
tme_input_mode = "all_channels"
tme_model = "PerChannelTMEModule"
tme_base_ch = 16

work_dir = "./checkpoints/pixcell_controlnet_exp_a1_per_channel"
seed = 42