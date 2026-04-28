"""Depth-18 additive mask+TME smoke config for post-fix gradient checks."""

_base_ = ["./config_controlnet_exp_a2_bypass.py"]

controlnet_depth = 18
num_epochs = 1
train_batch_size = 2
num_workers = 0
log_interval = 1
save_model_steps = 100000
save_model_epochs = 1000
save_final_checkpoint = False
debug_tme_probe = True

max_train_samples = 10
data = dict(max_train_samples=10)

work_dir = "./checkpoints/smoke_arch_depth18_bs2_additive_fixed"
seed = 42
