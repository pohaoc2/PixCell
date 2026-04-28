"""Depth-27 grouped-TME mid-length smoke for post-fix gradient checks."""

_base_ = ["./config_controlnet_exp.py"]

controlnet_depth = 27
num_epochs = 1
train_batch_size = 2
num_workers = 0
log_interval = 10
save_model_steps = 100000
save_model_epochs = 1000
save_final_checkpoint = False
debug_tme_probe = True

# bs=2 over 1000 samples gives 500 optimizer steps.
max_train_samples = 1000
data = dict(max_train_samples=1000)

work_dir = "./checkpoints/tme_midlength_depth27_bs2_grouped"
seed = 42
