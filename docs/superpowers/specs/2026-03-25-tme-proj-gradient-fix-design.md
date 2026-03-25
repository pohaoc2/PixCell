# Design: TME Proj Gradient Starvation Fix

**Date:** 2026-03-25
**Status:** Approved, pending implementation
**Related issue:** `docs/superpowers/specs/issue-tme-proj-gradient-starvation.md`

## Problem

Each `_GroupBlock` in `MultiGroupTMEModule` zero-initializes its `cross_attn.proj` layer.
After 30 epochs (`tme_lr=1e-5`), `proj.weight` remains ~1e-6 — too small for TME residuals to
influence the generated H&E image. The ControlNet already produces acceptable output from
`mask_latent` alone, so the gradient signal through the TME pathway is not amplified by task
pressure.

Two compounding risks with a naive blanket `tme_lr=3e-4` increase:
1. The encoder CNN and Q/K/V weights are already healthy (0.1–0.5). A 30× LR jump could
   destabilize them.
2. When resuming from the epoch-30 checkpoint, the restored optimizer state carries decayed
   momentum and a potentially low effective LR regardless of the config value.

## Solution: Split-LR + Diagnostics (Approaches A + B)

### A. Split TME optimizer into two param groups

`proj` layers (zero-init, stuck) get a high LR; all other TME parameters (already healthy)
keep their original rate.

**Config changes** (`configs/config_controlnet_exp.py`):
```python
tme_lr              = 1e-5   # encoder CNN + Q/K/V — stable, already learning
tme_proj_lr         = 3e-4   # cross_attn.proj only — zero-init, needs boost
reset_tme_optimizer = True   # load model weights only on resume; fresh optim state
```

**Optimizer split** (`train_scripts/training_utils.py`, `_build_tme_module_and_optimizers`):

Replace the single `build_optimizer(tme_module, ...)` call with two param groups when
`tme_proj_lr` is present in config:

```python
tme_proj_lr = getattr(config, "tme_proj_lr", None)
if tme_proj_lr is not None:
    proj_params  = [p for n, p in tme_module.named_parameters() if "cross_attn.proj" in n]
    other_params = [p for n, p in tme_module.named_parameters() if "cross_attn.proj" not in n]
    base_lr = getattr(config, "tme_lr", 1e-5)
    optimizer_tme = torch.optim.AdamW(
        [{"params": proj_params,  "lr": tme_proj_lr},
         {"params": other_params, "lr": base_lr}],
        weight_decay=config.optimizer.get("weight_decay", 0.0),
        betas=tuple(config.optimizer.get("betas", (0.9, 0.999))),
        eps=config.optimizer.get("eps", 1e-8),
    )
else:
    # fallback: single LR (legacy path)
    tme_optimizer_cfg = deepcopy(config.optimizer)
    tme_optimizer_cfg["lr"] = getattr(config, "tme_lr", 1e-4)
    optimizer_tme = build_optimizer(tme_module, tme_optimizer_cfg)
```

The LR scheduler wraps the multi-group optimizer unchanged — PyTorch schedulers respect
per-group base LRs automatically.

### B. Optimizer reset on resume

**`train_scripts/train_controlnet_exp.py`** — resume block:

```python
tme_ckpt = getattr(config, "resume_tme_checkpoint", None)
if tme_ckpt:
    reset_opt = getattr(config, "reset_tme_optimizer", False)
    step = load_tme_checkpoint(
        tme_ckpt, tme_module,
        optimizer_tme=None if reset_opt else optimizer_tme,
        lr_scheduler_tme=None if reset_opt else lr_scheduler_tme,
        device=accelerator.device,
    )
```

With `reset_tme_optimizer=True`: model weights from checkpoint, fresh optimizer/scheduler
from new config. Also avoids a param-group shape mismatch when the old checkpoint had a
single-group optimizer.

### C. Diagnostics

Two logging additions in the `use_multi_group` branch of
`train_scripts/train_controlnet_exp.py`:

**Residual magnitudes** — logged at every `log_interval` steps (no extra forward pass
overhead; uses `return_residuals=True`):

```python
log_now = (global_step % config.log_interval == 0 and accelerator.is_main_process)
if log_now:
    fused, residuals = tme_module(vae_mask.to(dtype=tme_dtype), tme_channel_dict,
                                  return_residuals=True)
    for gname, delta in residuals.items():
        logger.info(f"  delta_mean[{gname}]={delta.abs().mean():.3e}")
else:
    fused = tme_module(vae_mask.to(dtype=tme_dtype), tme_channel_dict)
vae_mask = fused
```

**Proj gradient norms** — inside `if accelerator.sync_gradients`, before gradient clipping:

```python
if log_now:
    for gname, gblock in accelerator.unwrap_model(tme_module).groups.items():
        g = gblock.cross_attn.proj.weight.grad
        if g is not None:
            logger.info(
                f"  proj_grad[{gname}]={g.norm():.3e}"
                f"  proj_wmax={gblock.cross_attn.proj.weight.abs().max():.3e}"
            )
```

**Interpretation guide:**

| Signal | Healthy | Problem |
|--------|---------|---------|
| `delta_mean[*]` | grows ~1e-5 → 1e-3 within 5k steps | flat or oscillating near 1e-5 |
| `proj_grad[*]` | > 1e-4 | < 1e-5 → task-easiness; escalate to auxiliary TME loss |
| `proj_wmax` | growing each log interval | stuck at 1e-6 despite LR boost |

## Files Changed

| File | Change |
|------|--------|
| `configs/config_controlnet_exp.py` | Add `tme_proj_lr`, `reset_tme_optimizer`; revert `tme_lr=1e-5` |
| `train_scripts/training_utils.py` | Split TME optimizer into two param groups |
| `train_scripts/train_controlnet_exp.py` | Optimizer reset on resume; residual + grad norm logging |

## Success Criteria

- `proj_grad[*]` consistently > 1e-4 at first log interval after resume
- `delta_mean[*]` reaches ~1e-3 within ~5k steps
- Ablation grid shows visible per-group differences by epoch 35–40
