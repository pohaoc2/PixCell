"""Ultra-light post-fix smoke config for validating training on small GPUs."""

_base_ = ["./config_controlnet_exp.py"]

controlnet_depth = 2
num_epochs = 3
train_batch_size = 1
log_interval = 10
save_model_steps = 100000
save_model_epochs = 1000

max_train_samples = 20
data = dict(max_train_samples=20)

work_dir = "./checkpoints/pixcell_controlnet_exp_smoke_depth2_post_fix"
seed = 42