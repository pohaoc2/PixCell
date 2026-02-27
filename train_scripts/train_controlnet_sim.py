"""
train_controlnet_sim.py

PixCell ControlNet + TME conditioning training (unpaired sim input).

All shared infrastructure is imported from initialize_models — nothing duplicated.
Only sim-specific code lives here:

    initialize_sim_training()   — SimControlNetData + TMEConditioningModule +
                                   all optimizers/schedulers in one place
    train_controlnet_sim()      — training loop (4 lines added vs train_controlnet)
    _save_sim_checkpoint()      — saves controlnet + tme_module together
    load_sim_checkpoint()       — restores tme_module from checkpoint

Usage:
    python  train_controlnet_sim.py <config_path> [--work-dir ...] [--debug ...]
    accelerate launch train_controlnet_sim.py <config_path>
"""

import os
import time
from copy import deepcopy

import torch
import torch.nn as nn

# ── All shared infrastructure — imported, not duplicated ─────────────────────
from train_scripts.initialize_models import (
    # config / accelerator / logging
    initialize_config_and_accelerator,
    # model construction
    initialize_models,
    # training state (resume, epoch counters)
    setup_training_state,
    # helpers used inside this file
    ema_update,
    _resume_from_checkpoint,
)

# ── Diffusion utils (same ones initialize_models uses) ───────────────────────
from diffusion.data.builder import build_dataloader
from diffusion.model.builder import build_model          # ← used for tme_module
from diffusion.utils.checkpoint import save_checkpoint
from diffusion.utils.optimizer import build_optimizer    # ← used for both optimizers
from diffusion.utils.lr_scheduler import build_lr_scheduler
from diffusion.utils.dist_utils import get_world_size

# ── Sim-specific dataset ──────────────────────────────────────────────────────
from diffusion.data.datasets.sim_controlnet_dataset import SimControlNetData


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Sim-specific initialization
# ─────────────────────────────────────────────────────────────────────────────

def initialize_sim_training(config, accelerator, logger, controlnet):
    """
    Build everything needed to train on sim data, in a single call:
        - SimControlNetData + dataloader
        - TMEConditioningModule  (via build_model, same pattern as controlnet)
        - ControlNet optimizer   (via build_optimizer, replaces the
                                  initialize_dataset_and_optimizer call)
        - TME optimizer          (via build_optimizer, with tme_lr override)
        - Both lr schedulers     (via build_lr_scheduler)

    Args:
        config:      Training config. Required fields:
                         sim_data_root, active_channels, image_size,
                         train_batch_size, num_workers, optimizer
                     Optional fields:
                         tme_model    (default "TMEConditioningModule")
                         tme_base_ch  (default 32)
                         tme_lr       (default: same as config.optimizer.lr)
        accelerator: HuggingFace Accelerator.
        logger:      Logger instance.
        controlnet:  The already-built trainable ControlNet model.
                     Passed in so its optimizer is built here in one place,
                     avoiding a second initialize_dataset_and_optimizer call.

    Returns dict with:
        train_dataloader,
        tme_module,
        optimizer,         ← controlnet optimizer
        optimizer_tme,
        lr_scheduler,      ← controlnet scheduler
        lr_scheduler_tme,
    """
    active_channels = getattr(config, "active_channels",
                              ["cell_mask", "oxygen", "glucose", "tgf"])

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset = SimControlNetData(
        root=config.sim_data_root,
        resolution=config.image_size,
        active_channels=active_channels,
        vae_prefix=getattr(config, "vae_prefix", "sd3_vae"),
        ssl_prefix=getattr(config, "ssl_prefix", "uni"),
    )
    train_dataloader = build_dataloader(
        dataset,
        num_workers=config.num_workers,
        batch_size=config.train_batch_size,
        shuffle=True,
    )

    # ── TME module via build_model ────────────────────────────────────────────
    # TMEConditioningModule must be registered in diffusion/model/builder.py
    # under the key config.tme_model (default "TMEConditioningModule").
    n_tme_channels = len(active_channels) - 1   # all channels except cell_mask
    tme_module = build_model(
        getattr(config, "tme_model", "TMEConditioningModule"),
        False,   # no grad checkpointing for a lightweight CNN
        False,   # no fp32 attention override
        n_tme_channels=n_tme_channels,
        base_ch=getattr(config, "tme_base_ch", 32),
    )
    logger.info(
        f"[TMEConditioningModule] n_tme_channels={n_tme_channels}  "
        f"trainable params="
        f"{sum(p.numel() for p in tme_module.parameters() if p.requires_grad):,}"
    )

    # ── ControlNet optimizer (same logic as initialize_dataset_and_optimizer) ─
    optimizer    = build_optimizer(controlnet, config.optimizer)
    lr_scheduler = build_lr_scheduler(config, optimizer, train_dataloader, lr_scale_ratio=1)

    # ── TME optimizer — same type as controlnet, optionally different lr ──────
    tme_optimizer_cfg        = deepcopy(config.optimizer)
    tme_optimizer_cfg['lr']  = getattr(config, "tme_lr", config.optimizer.get('lr', 1e-4))
    optimizer_tme    = build_optimizer(tme_module, tme_optimizer_cfg)
    lr_scheduler_tme = build_lr_scheduler(config, optimizer_tme, train_dataloader, lr_scale_ratio=1)

    return {
        "train_dataloader": train_dataloader,
        "tme_module":        tme_module,
        "optimizer":         optimizer,
        "optimizer_tme":     optimizer_tme,
        "lr_scheduler":      lr_scheduler,
        "lr_scheduler_tme":  lr_scheduler_tme,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_controlnet_sim(models_dict):
    """
    ControlNet + TME training loop.

    Identical to train_controlnet() in initialize_models.py with exactly
    4 additions for TME fusion, all marked  # <- NEW  or  # <- CHANGED.
    """
    # Unpack — same as train_controlnet
    base_model        = models_dict['base_model']
    controlnet        = models_dict['controlnet']
    model_ema         = models_dict['model_ema']
    vae               = models_dict['vae']
    train_diffusion   = models_dict['train_diffusion']
    optimizer         = models_dict['optimizer']
    lr_scheduler      = models_dict['lr_scheduler']
    train_dataloader  = models_dict['train_dataloader']
    accelerator       = models_dict['accelerator']
    config            = models_dict['config']
    logger            = models_dict['logger']
    args              = models_dict['args']
    start_epoch       = models_dict['start_epoch']
    start_step        = models_dict['start_step']
    skip_step         = models_dict['skip_step']
    total_steps       = models_dict['total_steps']

    # <- NEW
    tme_module        = models_dict['tme_module']
    optimizer_tme     = models_dict['optimizer_tme']
    lr_scheduler_tme  = models_dict['lr_scheduler_tme']

    controlnet.train()
    for param in controlnet.parameters():
        param.requires_grad = True
    tme_module.train()   # <- NEW

    time_start, last_tic = time.time(), time.time()
    global_step   = start_step + 1
    load_vae_feat = getattr(train_dataloader.dataset, 'load_vae_feat', False)
    vae_scale     = config.scale_factor
    vae_shift     = config.shift_factor

    logger.info("=" * 80)
    logger.info("Starting ControlNet + TME Training (unpaired sim input)")
    logger.info(f"start_epoch={start_epoch}  start_step={start_step}  total_steps={total_steps}")
    logger.info("=" * 80)

    for epoch in range(start_epoch + 1, config.num_epochs + 1):
        data_time_start = time.time()

        for step, batch in enumerate(train_dataloader):
            if step < skip_step:
                if (step + 1) % 50 == 0 and accelerator.is_main_process:
                    logger.info(
                        f"Skipping [{global_step}/{epoch}][{step+1}/{len(train_dataloader)}]"
                    )
                continue

            # 1. Encode images to latent space — unchanged
            if load_vae_feat:
                z = batch[0]
            else:
                with torch.no_grad():
                    with torch.cuda.amp.autocast(
                        enabled=(config.mixed_precision in ['fp16', 'bf16'])
                    ):
                        x_in = batch[0].to(dtype=next(vae.parameters()).dtype)
                        posterior = vae.encode(x_in).latent_dist
                        z = (posterior.sample()
                             if config.sample_posterior else posterior.mode())
            clean_images = (z.float() - config.shift_factor) * config.scale_factor

            # 2. Unpack batch — unchanged
            y             = batch[1]   # UNI embeddings      [B, 1, 1, 1152]
            control_input = batch[2]   # All sim channels    [B, C, 256, 256]
            vae_mask      = batch[3]   # VAE cell mask       [B, 16, 32, 32]
            data_info     = batch[4]

            bs        = clean_images.shape[0]
            timesteps = torch.randint(
                0, config.train_sampling_steps, (bs,), device=clean_images.device
            ).long()

            # Scale VAE mask — unchanged
            vae_mask = (vae_mask - vae_shift) * vae_scale

            # <- NEW: fuse TME channels into the conditioning signal
            # control_input[:, 0]  = cell_mask (already encoded in vae_mask via SD3 VAE)
            # control_input[:, 1:] = TME channels (oxygen, glucose, tgf, ...)
            tme_dtype    = next(tme_module.parameters()).dtype   # matches fp16/bf16/fp32
            tme_channels = control_input[:, 1:, :, :].to(dtype=tme_dtype)
            vae_mask     = tme_module(vae_mask.to(dtype=tme_dtype), tme_channels)
            vae_mask     = vae_mask.float()                      # back to fp32 for loss
            # <- END NEW

            # 3. Training step
            with accelerator.accumulate(controlnet, tme_module):   # <- CHANGED
                optimizer.zero_grad()
                optimizer_tme.zero_grad()   # <- NEW

                model_kwargs = dict(
                    y=y, mask=None, data_info=data_info, control_input=vae_mask,
                )
                loss_term = training_losses_controlnet(
                    diffusion=train_diffusion,
                    controlnet=controlnet,
                    base_model=base_model,
                    x_start=clean_images,
                    timesteps=timesteps,
                    model_kwargs=model_kwargs,
                    config=config,
                )
                loss = loss_term['loss']
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(controlnet.parameters(), config.gradient_clip)
                    optimizer.step()
                    lr_scheduler.step()
                    # <- NEW
                    accelerator.clip_grad_norm_(tme_module.parameters(), config.gradient_clip)
                    optimizer_tme.step()
                    lr_scheduler_tme.step()

                if accelerator.is_main_process:
                    ema_update(model_ema, controlnet, config.ema_rate)

            # 4. Logging — unchanged + tme LR
            if accelerator.sync_gradients:
                global_step += 1
                if global_step % config.log_interval == 0:
                    time_cost       = time.time() - last_tic
                    samples_per_sec = (
                        config.log_interval * config.train_batch_size / time_cost
                    )
                    logger.info(
                        f"Epoch [{epoch}/{config.num_epochs}] "
                        f"Step [{global_step}/{total_steps}] "
                        f"Loss: {loss.item():.4f} "
                        f"LR_ctrl: {optimizer.param_groups[0]['lr']:.2e} "
                        f"LR_tme:  {optimizer_tme.param_groups[0]['lr']:.2e} "  # <- NEW
                        f"Samples/s: {samples_per_sec:.2f}"
                    )
                    last_tic = time.time()

                if global_step % config.save_model_steps == 0:
                    _save_sim_checkpoint(
                        accelerator, controlnet, tme_module, model_ema,
                        optimizer, optimizer_tme, lr_scheduler, lr_scheduler_tme,
                        global_step, epoch, config, logger,
                    )

            if global_step >= total_steps:
                logger.info(f"Reached max steps ({total_steps}). Stopping.")
                break

        if epoch % config.save_model_epochs == 0 or epoch == config.num_epochs:
            _save_sim_checkpoint(
                accelerator, controlnet, tme_module, model_ema,
                optimizer, optimizer_tme, lr_scheduler, lr_scheduler_tme,
                global_step, epoch, config, logger,
            )

        if global_step >= total_steps:
            break

    logger.info("=" * 80)
    logger.info("Training Complete!")
    logger.info("=" * 80)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def _save_sim_checkpoint(
    accelerator, controlnet, tme_module, model_ema,
    optimizer, optimizer_tme, lr_scheduler, lr_scheduler_tme,
    step, epoch, config, logger,
):
    """Save controlnet (existing format) + tme_module side-by-side."""
    ckpt_dir = os.path.join(config.work_dir, "checkpoints", f"step_{step:07d}")
    os.makedirs(ckpt_dir, exist_ok=True)
    if accelerator.is_main_process:
        # ControlNet — identical to original save_checkpoint call
        save_checkpoint(
            work_dir=ckpt_dir, epoch=epoch,
            model=accelerator.unwrap_model(controlnet),
            model_ema=model_ema, optimizer=optimizer,
            lr_scheduler=lr_scheduler, step=step,
            keep_last=False, model_type="controlnet",
        )
        # TME module
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


def load_sim_checkpoint(ckpt_dir, tme_module, optimizer_tme=None,
                        lr_scheduler_tme=None, device="cpu"):
    """Load TME module weights (+ optionally optimizer/scheduler) from checkpoint."""
    ckpt = torch.load(os.path.join(ckpt_dir, "tme_module.pth"), map_location=device)
    tme_module.load_state_dict(ckpt["model_state"])
    if optimizer_tme is not None:
        optimizer_tme.load_state_dict(ckpt["optim_state"])
    if lr_scheduler_tme is not None:
        lr_scheduler_tme.load_state_dict(ckpt["sched_state"])
    return ckpt["step"]


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Loss function — unchanged from initialize_models.py / train_controlnet
# ─────────────────────────────────────────────────────────────────────────────

def training_losses_controlnet(diffusion, controlnet, base_model, x_start,
                                timesteps, model_kwargs=None, config=None):
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
        hidden_states=x_t, conditioning=control_input,
        encoder_hidden_states=model_kwargs['y'], timestep=model_timesteps,
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


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Entrypoint
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # parse_args() in initialize_models expects a positional 'config' argument:
    #   python train_controlnet_sim.py <config_path> [optional flags]
    init_data   = initialize_config_and_accelerator()
    config      = init_data['config']
    accelerator = init_data['accelerator']
    logger      = init_data['logger']
    args        = init_data['args']

    # Build frozen base model, trainable controlnet, EMA, VAE, diffusion
    model_data      = initialize_models(config, accelerator, logger)
    base_model      = model_data['base_model']
    controlnet      = model_data['controlnet']
    model_ema       = model_data['model_ema']
    vae             = model_data['vae']
    train_diffusion = model_data['train_diffusion']

    # Build sim dataset + TME module + all optimizers in one call.
    # controlnet is passed in so its optimizer is built here — no second
    # initialize_dataset_and_optimizer call needed.
    sim_data         = initialize_sim_training(config, accelerator, logger, controlnet)
    train_dataloader = sim_data['train_dataloader']
    tme_module       = sim_data['tme_module']
    optimizer        = sim_data['optimizer']
    optimizer_tme    = sim_data['optimizer_tme']
    lr_scheduler     = sim_data['lr_scheduler']
    lr_scheduler_tme = sim_data['lr_scheduler_tme']

    # Prepare everything with accelerator
    (
        base_model, controlnet, model_ema,
        optimizer, train_dataloader, lr_scheduler,
        tme_module, optimizer_tme,
    ) = accelerator.prepare(
        base_model, controlnet, model_ema,
        optimizer, train_dataloader, lr_scheduler,
        tme_module, optimizer_tme,
    )

    # Setup epoch/step counters; optionally resume controlnet checkpoint
    state_data = setup_training_state(
        config, accelerator, logger, args, train_dataloader,
        base_model, controlnet, model_ema, optimizer, lr_scheduler,
    )

    # Optionally resume TME module from a separate checkpoint
    tme_ckpt = getattr(config, "resume_tme_checkpoint", None)
    if tme_ckpt:
        step = load_sim_checkpoint(
            tme_ckpt, tme_module, optimizer_tme, lr_scheduler_tme,
            device=accelerator.device,
        )
        logger.info(f"Resumed TME module from step {step} ({tme_ckpt})")

    models = {
        'base_model':        base_model,
        'controlnet':        controlnet,
        'model_ema':         model_ema,
        'vae':               vae,
        'train_diffusion':   train_diffusion,
        'optimizer':         optimizer,
        'optimizer_d':       None,
        'lr_scheduler':      lr_scheduler,
        'train_dataloader':  train_dataloader,
        'accelerator':       accelerator,
        'config':            config,
        'logger':            logger,
        'args':              args,
        'tme_module':        tme_module,
        'optimizer_tme':     optimizer_tme,
        'lr_scheduler_tme':  lr_scheduler_tme,
        **state_data,
    }
    train_controlnet_sim(models)


if __name__ == "__main__":
    main()