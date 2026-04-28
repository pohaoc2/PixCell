"""Depth-18 concat-TME smoke config for post-fix gradient checks."""

_base_ = ["./config_controlnet_exp_a1_concat.py"]

controlnet_depth = 18
num_epochs = 1
train_batch_size = 10
num_workers = 0
log_interval = 1
save_model_steps = 100000
save_model_epochs = 1000

max_train_samples = 20
data = dict(max_train_samples=20)

work_dir = "./checkpoints/smoke_arch_depth18_bs10_concat_fixed"
seed = 42
