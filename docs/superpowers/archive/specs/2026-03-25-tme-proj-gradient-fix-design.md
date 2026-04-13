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

Two compounding risks with a naive blanket `tme_lr=3e-4` increase (already applied to the
config but not yet tested):
1. The encoder CNN and Q/K/V weights are already healthy (0.1–0.5). A 30× LR jump could
   destabilize them.
2. When resuming from the epoch-30 checkpoint, the restored optimizer state carries decayed
   momentum. Loading it into a differently-structured (two-group) optimizer will hard-crash
   with `ValueError: loaded state dict has a different number of parameter groups`.

## Solution: Split-LR + Diagnostics (Approaches A + B)

### A. Split TME optimizer into two param groups

`proj` layers (zero-init, stuck) get a high LR; all other TME parameters (already healthy)
keep their original rate.

**Config changes** (`configs/config_controlnet_exp.py`):

The current config has `tme_lr = 3e-4` (blanket increase — the naive approach). Replace with:

```python
tme_lr              = 1e-5   # encoder CNN + Q/K/V — stable, already learning
tme_proj_lr         = 3e-4   # cross_attn.proj only — zero-init, needs boost
reset_tme_optimizer = True   # REQUIRED for first resume after optimizer-split is activated
                             # (see Section B — hard crash if False with old single-group ckpt)
```

`reset_tme_optimizer=True` is **mandatory** for the first resume after this change. Attempting
to load the old single-group optimizer state into the new two-group optimizer will raise a
`ValueError`. Once training has saved a new checkpoint with the two-group optimizer, this
can optionally be set back to `False`.

**Optimizer split** (`train_scripts/training_utils.py`, `_build_tme_module_and_optimizers`):

Replace the single `build_optimizer(tme_module, ...)` call with two param groups when
`tme_proj_lr` is present in config. `"cross_attn.proj" in n` correctly matches
`groups.<name>.cross_attn.proj.weight` and `.bias` from `named_parameters()`:

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

The LR scheduler wraps the multi-group optimizer unchanged — PyTorch's `LambdaLR` applies the
same multiplier to each group's `base_lr` independently. `save_checkpoint_with_tme` is also
unchanged: `optimizer.state_dict()` is self-describing for any number of param groups.

### B. Optimizer reset on resume

**`train_scripts/train_controlnet_exp.py`** — replace the existing unconditional resume block
(lines 421–427) with the following. Do not add a second block; replace the existing one:

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

With `reset_tme_optimizer=True`: loads model weights only; optimizer and scheduler start fresh
from the new config. This is the only safe path when the checkpoint was saved with the old
single-group optimizer.

Note: `load_tme_checkpoint` is called after `accelerator.prepare()`. For the `reset_opt=True`
path this is harmless (no optimizer state loaded). For future runs where `reset_opt=False`
and the checkpoint also has two groups, this ordering is also fine for the default Accelerate
(DDP) backend.

### C. Diagnostics

Two logging additions in the `use_multi_group` branch of
`train_scripts/train_controlnet_exp.py`. Both gates on `accelerator.sync_gradients` to avoid
logging on accumulation micro-steps (where `.grad` is `None`).

`log_now` must be computed **once per step** before both diagnostic blocks — hoist it above
the residuals branch and above `if accelerator.sync_gradients` so the grad-norm block can
reference it without a `NameError`.

**Residual magnitudes** — single forward pass, `return_residuals=True` on log steps only.
The output `fused` is the one used for training — there is no second forward pass:

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

**Proj gradient norms** — inside `if accelerator.sync_gradients`, before gradient clipping.
Grads are only non-None at sync steps; the `if g is not None` guard is defensive, not a
substitute for being inside `sync_gradients`:

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
| `configs/config_controlnet_exp.py` | Revert `tme_lr=1e-5`; add `tme_proj_lr=3e-4`, `reset_tme_optimizer=True` |
| `train_scripts/training_utils.py` | Split TME optimizer into two param groups |
| `train_scripts/train_controlnet_exp.py` | Replace resume block (not add); residual + grad norm logging |

## Success Criteria

- `proj_grad[*]` consistently > 1e-4 at first log interval after resume
- `delta_mean[*]` reaches ~1e-3 within ~5k steps
- Ablation grid shows visible per-group differences by epoch 35–40
