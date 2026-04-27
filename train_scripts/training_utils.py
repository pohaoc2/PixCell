"""
training_utils.py

Shared utilities for PixCell ControlNet training:
    - TMEConditioningModule construction + optimizer/scheduler setup
    - Diffusion loss computation (controlnet forward + base model)
    - Checkpoint save/load helpers for controlnet + TME module

Used by stage2_train.py (paired experimental training).
"""
from copy import deepcopy

import torch
from diffusion.utils.tme_checkpoint_key_remap import (
    remap_tme_state_dict_cell_identity_to_cell_types,
)
from diffusion.model.builder import build_model
from diffusion.utils.checkpoint import save_checkpoint
from diffusion.utils.lr_scheduler import build_lr_scheduler
from diffusion.utils.optimizer import build_optimizer
import os


# ── TME module + optimizer construction ──────────────────────────────────────

def _build_tme_module_and_optimizers(config, controlnet, train_dataloader,
                                     active_channels, logger):
    """
    Build TMEConditioningModule + all four optimizers/schedulers.

    Args:
        config:           Training config.
        controlnet:       The trainable ControlNet model.
        train_dataloader: DataLoader (used for scheduler step count).
        active_channels:  List of channel names including cell_mask.
        logger:           Logger instance.

    Returns dict:
        tme_module, optimizer, optimizer_tme, lr_scheduler, lr_scheduler_tme
    """
    channel_groups_cfg = getattr(config, "channel_groups", None)
    tme_input_mode = getattr(config, "tme_input_mode", None)

    if channel_groups_cfg is not None:
        group_specs = []
        for g in channel_groups_cfg:
            group_specs.append(dict(name=g["name"], n_channels=len(g["channels"])))
        tme_module = build_model(
            getattr(config, "tme_model", "MultiGroupTMEModule"),
            False,
            False,
            channel_groups=group_specs,
            base_ch=getattr(config, "tme_base_ch", 32),
        )
    elif tme_input_mode == "all_channels":
        tme_module = build_model(
            getattr(config, "tme_model", "RawConditioningPassthrough"),
            False,
            False,
            active_channels=active_channels,
            base_ch=getattr(config, "tme_base_ch", 32),
        )
    else:
        n_tme_channels = len(active_channels) - 1   # all channels except cell_mask
        tme_module = build_model(
            getattr(config, "tme_model", "TMEConditioningModule"),
            False,
            False,
            n_tme_channels=n_tme_channels,
            base_ch=getattr(config, "tme_base_ch", 32),
        )
    logger.info(
        f"[TME Module: {type(tme_module).__name__}] "
        f"trainable params="
        f"{sum(p.numel() for p in tme_module.parameters() if p.requires_grad):,}"
    )

    optimizer    = build_optimizer(controlnet, config.optimizer)
    lr_scheduler = build_lr_scheduler(config, optimizer, train_dataloader, lr_scale_ratio=1)

    tme_proj_lr = getattr(config, "tme_proj_lr", None)
    has_proj_split = any("cross_attn.proj" in n for n, _ in tme_module.named_parameters())
    if tme_proj_lr is not None and has_proj_split:
        proj_params  = [p for n, p in tme_module.named_parameters() if "cross_attn.proj" in n]
        other_params = [p for n, p in tme_module.named_parameters() if "cross_attn.proj" not in n]
        base_tme_lr  = getattr(config, "tme_lr", 1e-5)
        optimizer_tme = torch.optim.AdamW(
            [{"params": proj_params,  "lr": tme_proj_lr},
             {"params": other_params, "lr": base_tme_lr}],
            weight_decay=config.optimizer.get("weight_decay", 0.0),
            betas=tuple(config.optimizer.get("betas", (0.9, 0.999))),
            eps=config.optimizer.get("eps", 1e-8),
        )
    else:
        tme_optimizer_cfg       = deepcopy(config.optimizer)
        tme_optimizer_cfg['lr'] = getattr(config, "tme_lr", config.optimizer.get('lr', 1e-4))
        optimizer_tme = build_optimizer(tme_module, tme_optimizer_cfg)
    lr_scheduler_tme = build_lr_scheduler(config, optimizer_tme, train_dataloader, lr_scale_ratio=1)

    return {
        "tme_module":        tme_module,
        "optimizer":         optimizer,
        "optimizer_tme":     optimizer_tme,
        "lr_scheduler":      lr_scheduler,
        "lr_scheduler_tme":  lr_scheduler_tme,
    }


# ── Diffusion loss ────────────────────────────────────────────────────────────

def training_losses_controlnet(diffusion, controlnet, base_model, x_start,
                                timesteps, model_kwargs=None, config=None):
    """
    Compute ControlNet training loss (MSE in latent space).

    Args:
        diffusion:      IDDPM diffusion process.
        controlnet:     Trainable ControlNet model.
        base_model:     Frozen PixCell transformer.
        x_start:        Clean latents [B, 16, H, W].
        timesteps:      Sampled timesteps [B].
        model_kwargs:   Dict containing: y (UNI embeds), mask, data_info,
                        control_input (fused TME conditioning).
        config:         Training config.

    Returns:
        Dict: loss, pred, target, var_values
    """
    if model_kwargs is None:
        model_kwargs = {}
    noise           = torch.randn_like(x_start).float()
    x_t             = diffusion.q_sample(x_start, timesteps, noise=noise)
    map_tensor      = torch.tensor(diffusion.timestep_map,
                                   device=timesteps.device, dtype=timesteps.dtype)
    model_timesteps = map_tensor[timesteps]
    control_input   = model_kwargs.pop('control_input', None)
    if control_input is None:
        raise ValueError("control_input must be provided in model_kwargs!")
    conditioning_scale = getattr(config, 'controlnet_conditioning_scale', 1.0)
    pred_sigma         = getattr(config, 'pred_sigma',  True)
    learn_sigma        = getattr(config, 'learn_sigma', True) and pred_sigma

    controlnet_outputs = controlnet(
        hidden_states=x_t,
        conditioning=control_input,
        encoder_hidden_states=model_kwargs['y'],
        timestep=model_timesteps,
        conditioning_scale=conditioning_scale,
        mask=model_kwargs.get('mask', None),
        data_info=model_kwargs.get('data_info', None),
    )
    if isinstance(controlnet_outputs, tuple):
        controlnet_residuals = controlnet_outputs[0]
    else:
        controlnet_residuals = controlnet_outputs['controlnet_block_samples']
    if config.mixed_precision in ['fp16', 'bf16']:
        controlnet_residuals = [r.float() for r in controlnet_residuals]

    model_output = base_model(
        x=x_t, y=model_kwargs['y'], timestep=model_timesteps,
        controlnet_outputs=controlnet_residuals,
        attention_mask=model_kwargs.get('mask', None), return_dict=True,
    )
    model_pred = model_output.sample if hasattr(model_output, 'sample') else model_output
    model_var_values = None
    if learn_sigma and model_pred.shape[1] == x_start.shape[1] * 2:
        model_pred, model_var_values = model_pred.chunk(2, dim=1)
    loss = torch.nn.functional.mse_loss(model_pred, noise, reduction='none')
    loss = loss.mean(dim=list(range(1, len(loss.shape))))
    return {
        'loss': loss.mean(), 'pred': model_pred, 'target': noise,
        'var_values': model_var_values if learn_sigma else None,
    }


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def save_checkpoint_with_tme(
    accelerator, controlnet, tme_module, model_ema,
    optimizer, optimizer_tme, lr_scheduler, lr_scheduler_tme,
    step, epoch, config, logger,
):
    """Save controlnet + tme_module checkpoints side-by-side."""
    ckpt_dir = os.path.join(config.work_dir, "checkpoints", f"step_{step:07d}")
    os.makedirs(ckpt_dir, exist_ok=True)
    if accelerator.is_main_process:
        save_checkpoint(
            work_dir=ckpt_dir, epoch=epoch,
            model=accelerator.unwrap_model(controlnet),
            model_ema=model_ema, optimizer=optimizer,
            lr_scheduler=lr_scheduler, step=step,
            keep_last=False, model_type="controlnet",
        )
        torch.save(
            {
                "step":        step,
                "epoch":       epoch,
                "model_state": accelerator.unwrap_model(tme_module).state_dict(),
                "optim_state": optimizer_tme.state_dict(),
                "sched_state": lr_scheduler_tme.state_dict(),
            },
            os.path.join(ckpt_dir, "tme_module.pth"),
        )
        logger.info(f"Saved checkpoint step={step} → {ckpt_dir}")


def load_tme_checkpoint(ckpt_dir, tme_module, optimizer_tme=None,
                        lr_scheduler_tme=None, device="cpu"):
    """Load TME module weights (+ optionally optimizer/scheduler) from checkpoint."""
    ckpt = torch.load(os.path.join(ckpt_dir, "tme_module.pth"), map_location=device)
    model_state = remap_tme_state_dict_cell_identity_to_cell_types(ckpt["model_state"])
    tme_module.load_state_dict(model_state)
    if optimizer_tme is not None:
        optimizer_tme.load_state_dict(ckpt["optim_state"])
    if lr_scheduler_tme is not None:
        lr_scheduler_tme.load_state_dict(ckpt["sched_state"])
    return ckpt["step"]
