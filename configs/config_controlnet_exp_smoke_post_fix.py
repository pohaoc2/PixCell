"""Short post-fix smoke config for paired-exp ControlNet retraining."""

_base_ = ["./config_controlnet_exp.py"]

num_epochs = 1
train_batch_size = 1
log_interval = 10
save_model_steps = 100000
save_model_epochs = 1000

max_train_samples = 100
data = dict(max_train_samples=100)

work_dir = "./checkpoints/pixcell_controlnet_exp_smoke_post_fix"
seed = 42