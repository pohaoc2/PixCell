"""Depth-10 post-fix smoke config for local gradient-stability checks."""

_base_ = ["./config_controlnet_exp.py"]

controlnet_depth = 10
num_epochs = 2
train_batch_size = 1
num_workers = 0
log_interval = 5
save_model_steps = 100000
save_model_epochs = 1000

max_train_samples = 20
data = dict(max_train_samples=20)

work_dir = "./checkpoints/pixcell_controlnet_exp_smoke_depth10_post_fix"
seed = 42
