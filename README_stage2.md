# Stage 2: Training

This guide covers ControlNet + TME training on paired experimental data.

Prerequisites:

1. Project dependencies installed from [`README.md`](README.md).
2. Pretrained models downloaded and features cached from [`stage1.md`](stage1.md).

---

## Stage 2: Training

Train ControlNet + Multi-Group TME module on paired experimental data.

### TME Architecture

The TME conditioning uses a **Multi-Group architecture** where each channel group gets its own CNN encoder and cross-attention module. Groups produce additive, zero-initialized residuals, enabling:

- **Disentangled control**: independently include or exclude channel groups at inference
- **Interpretability**: per-group residual magnitude maps and ablation diff maps
- **Graceful degradation**: missing groups contribute zero without special handling

**`zero_mask_latent=True`** (enabled by default): the TME module receives the real VAE mask latent as spatial query keys, then subtracts it from its output:

```text
fused = tme(vae_mask) - vae_mask
```

This closes the direct mask -> ControlNet bypass path and forces the ControlNet to rely on TME residuals while preserving the spatial structure needed for cell-layout-aware cross-attention.

| Group | Channels | Nature |
|-------|----------|--------|
| `cell_types` | `cell_type_healthy`, `cell_type_cancer`, `cell_type_immune` | One-hot (CODEX) |
| `cell_state` | `cell_state_prolif`, `cell_state_nonprolif`, `cell_state_dead` | One-hot (CODEX) |
| `vasculature` | `vasculature` | Continuous (CD31) |
| `microenv` | `oxygen`, `glucose` | Continuous (PDE-derived) |

Each group is independently droppable during both training (per-group dropout) and inference (`--active-groups` / `--drop-groups`).

---

## Launch Training

### Single GPU

```bash
python stage2_train.py configs/config_controlnet_exp.py
```

### Multi-GPU (recommended)

```bash
accelerate config
accelerate launch --num_processes 4 stage2_train.py \
    configs/config_controlnet_exp.py
```

---

## Config: `configs/config_controlnet_exp.py`

Key field to set before training:

```python
exp_data_root = "./data/exp_paired"
```

Key training knobs:

| Field | Default | Description |
|-------|---------|-------------|
| `cfg_dropout_prob` | `0.15` | Fraction of steps where UNI embedding is zeroed |
| `tme_lr` | `1e-5` | TME module learning rate |
| `num_epochs` | `200` | Training epochs |
| `save_model_steps` | `10000` | Checkpoint every N steps |

---

## CLI Options

| Flag | Description |
|------|-------------|
| `--work-dir PATH` | Output directory for checkpoints and logs |
| `--resume-from PATH` | Resume ControlNet from checkpoint directory |
| `--load-from PATH` | Load specific checkpoint file |
| `--batch-size N` | Override `train_batch_size` in config |
| `--debug` | Minimal steps for debugging |

---

## Checkpoint Layout

```text
checkpoints/pixcell_controlnet_exp/checkpoints/step_XXXXXXX/
├── epoch_X_step_XXXXXXX.pth    # ControlNet weights
└── tme_module.pth              # Multi-Group TME module weights
```

---

## Training-Time Visualizations

At every `save_model_steps` checkpoint, the training loop generates validation visualizations:

```text
checkpoints/pixcell_controlnet_exp/vis/step_XXXXXXX/
├── overview.png
└── attention_heatmaps.png
```

---

## Monitoring

Training logs are written to TensorBoard:

```bash
tensorboard --logdir checkpoints/pixcell_controlnet_exp/logs
```

Key metrics:

| Metric | Description |
|--------|-------------|
| `loss` | Diffusion MSE loss; should decrease steadily |
| `lr_ctrl` | ControlNet learning rate |
| `lr_tme` | TME encoder learning rate |
| `samples_per_sec` | Training throughput |

Next step: move to [`stage3.md`](stage3.md) for inference, validation, and ablation workflows.
