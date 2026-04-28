"""Depth-18 smoke config for validating finite-safe TME gradient clipping."""

_base_ = ["./config_controlnet_exp.py"]

controlnet_depth = 18
num_epochs = 1
train_batch_size = 1
num_workers = 0
log_interval = 1
save_model_steps = 100000
save_model_epochs = 1000

max_train_samples = 5
data = dict(max_train_samples=5)

work_dir = "./checkpoints/pixcell_controlnet_exp_smoke_depth18_safeclip_post_fix"
seed = 42
