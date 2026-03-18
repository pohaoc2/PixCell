"""
train_controlnet_exp.py

PixCell ControlNet fine-tuning on PAIRED experimental H&E + CODEX-derived TME channels.

Three additions vs train_controlnet_sim.py (all marked # <- EXP):
    1. PairedExpControlNetData  — paired dataset (single index, no random cross-sampling)
    2. CFG dropout              — zero UNI embedding with probability cfg_dropout_prob
    3. Channel reliability weights — attenuate approximate channels before TMEEncoder

Usage:
    python  train_scripts/train_controlnet_exp.py <config_path> [--work-dir ...] [--debug ...]
    accelerate launch train_scripts/train_controlnet_exp.py <config_path>
"""
import os
import time

import torch

from train_scripts.initialize_models import (
    initialize_config_and_accelerator,
    initialize_models,
    setup_training_state,
    ema_update,
)
from diffusion.data.builder import build_dataloader

from diffusion.data.datasets.paired_exp_controlnet_dataset import PairedExpControlNetData
from train_scripts.train_controlnet_sim import (
    training_losses_controlnet,
    _save_sim_checkpoint,
    load_sim_checkpoint,
    _build_tme_module_and_optimizers,
)


# ── Initialization ────────────────────────────────────────────────────────────

def initialize_exp_training(config, accelerator, logger, controlnet):
    """
    Build everything for paired-exp training:
        PairedExpControlNetData + TMEConditioningModule + both optimizers/schedulers.

    Required config fields (beyond base controlnet fields):
        exp_data_root               (str)         root of exp dataset
        active_channels             (list[str])   channel list for dataset
    Optional config fields:
        cfg_dropout_prob            (float)        default 0.15
        channel_reliability_weights (list[float])  one per tme channel (excl. cell_mask)
        tme_model, tme_base_ch, tme_lr             (same as sim config)
    """
    active_channels = getattr(config, "active_channels", [
        "cell_mask",
        "cell_type_healthy", "cell_type_cancer", "cell_type_immune",
        "cell_state_prolif",  "cell_state_nonprolif", "cell_state_dead",
        "vasculature", "oxygen", "glucose",
    ])

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset = PairedExpControlNetData(
        root=config.exp_data_root,
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

    tme_and_opts = _build_tme_module_and_optimizers(
        config, controlnet, train_dataloader, active_channels, logger
    )
    return {"train_dataloader": train_dataloader, **tme_and_opts}


# ── Training loop ─────────────────────────────────────────────────────────────

def train_controlnet_exp(models_dict):
    """
    Paired-exp ControlNet training loop.

    Identical to train_controlnet_sim() with 3 additions (marked # <- EXP).
    """
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
    tme_module        = models_dict['tme_module']
    optimizer_tme     = models_dict['optimizer_tme']
    lr_scheduler_tme  = models_dict['lr_scheduler_tme']

    # <- EXP: read training knobs from config
    cfg_dropout_prob = getattr(config, "cfg_dropout_prob", 0.15)
    channel_weights  = getattr(config, "channel_reliability_weights", None)

    controlnet.train()
    for param in controlnet.parameters():
        param.requires_grad = True
    tme_module.train()

    time_start, last_tic = time.time(), time.time()
    global_step   = start_step + 1
    load_vae_feat = getattr(train_dataloader.dataset, 'load_vae_feat', False)
    vae_scale     = config.scale_factor
    vae_shift     = config.shift_factor

    logger.info("=" * 80)
    logger.info("Starting ControlNet + TME Fine-tuning (paired experimental data)")
    logger.info(f"cfg_dropout_prob={cfg_dropout_prob}  channel_weights={channel_weights}")
    logger.info(f"start_epoch={start_epoch}  start_step={start_step}  total_steps={total_steps}")
    logger.info("=" * 80)

    for epoch in range(start_epoch + 1, config.num_epochs + 1):
        for step, batch in enumerate(train_dataloader):
            if step < skip_step:
                if (step + 1) % 50 == 0 and accelerator.is_main_process:
                    logger.info(
                        f"Skipping [{global_step}/{epoch}][{step+1}/{len(train_dataloader)}]"
                    )
                continue

            # 1. Encode images to latent space
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

            # 2. Unpack batch
            y             = batch[1]           # UNI embeddings  [B, 1, 1, 1536]
            control_input = batch[2]           # TME channels    [B, C, 256, 256]
            vae_mask      = batch[3]           # VAE cell mask   [B, 16, 32, 32]
            data_info     = batch[4]

            bs        = clean_images.shape[0]
            timesteps = torch.randint(
                0, config.train_sampling_steps, (bs,), device=clean_images.device
            ).long()

            vae_mask = (vae_mask - vae_shift) * vae_scale

            # <- EXP 1: CFG dropout — zero the UNI embedding for a fraction of samples
            for b in range(bs):
                if torch.rand(1).item() < cfg_dropout_prob:
                    y[b] = torch.zeros_like(y[b])

            tme_dtype    = next(tme_module.parameters()).dtype
            tme_channels = control_input[:, 1:, :, :].to(dtype=tme_dtype)

            # <- EXP 2: Channel reliability weighting (training only — not applied at inference)
            if channel_weights is not None:
                w = torch.tensor(
                    channel_weights, device=tme_channels.device, dtype=tme_channels.dtype
                ).view(1, -1, 1, 1)
                tme_channels = tme_channels * w

            # <- EXP 3: Fuse TME into conditioning (same as sim, but with EXP additions above)
            vae_mask = tme_module(vae_mask.to(dtype=tme_dtype), tme_channels)
            vae_mask = vae_mask.float()

            # 3. Training step
            with accelerator.accumulate(controlnet, tme_module):
                optimizer.zero_grad()
                optimizer_tme.zero_grad()

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
                    accelerator.clip_grad_norm_(tme_module.parameters(), config.gradient_clip)
                    optimizer_tme.step()
                    lr_scheduler_tme.step()

                if accelerator.is_main_process:
                    ema_update(model_ema, controlnet, config.ema_rate)

            # 4. Logging
            if accelerator.sync_gradients:
                global_step += 1
                if global_step % config.log_interval == 0:
                    time_cost       = time.time() - last_tic
                    samples_per_sec = config.log_interval * config.train_batch_size / time_cost
                    logger.info(
                        f"Epoch [{epoch}/{config.num_epochs}] "
                        f"Step [{global_step}/{total_steps}] "
                        f"Loss: {loss.item():.4f} "
                        f"LR_ctrl: {optimizer.param_groups[0]['lr']:.2e} "
                        f"LR_tme:  {optimizer_tme.param_groups[0]['lr']:.2e} "
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
    logger.info("Fine-tuning Complete!")
    logger.info("=" * 80)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    config_path = "./configs/config_controlnet_exp.py"
    init_data   = initialize_config_and_accelerator([config_path])
    config      = init_data['config']
    accelerator = init_data['accelerator']
    logger      = init_data['logger']
    args        = init_data['args']

    model_data      = initialize_models(config, accelerator, logger)
    base_model      = model_data['base_model']
    controlnet      = model_data['controlnet']
    model_ema       = model_data['model_ema']
    vae             = model_data['vae']
    train_diffusion = model_data['train_diffusion']

    exp_data         = initialize_exp_training(config, accelerator, logger, controlnet)
    train_dataloader = exp_data['train_dataloader']
    tme_module       = exp_data['tme_module']
    optimizer        = exp_data['optimizer']
    optimizer_tme    = exp_data['optimizer_tme']
    lr_scheduler     = exp_data['lr_scheduler']
    lr_scheduler_tme = exp_data['lr_scheduler_tme']

    (
        base_model, controlnet, model_ema,
        optimizer, train_dataloader, lr_scheduler,
        tme_module, optimizer_tme,
    ) = accelerator.prepare(
        base_model, controlnet, model_ema,
        optimizer, train_dataloader, lr_scheduler,
        tme_module, optimizer_tme,
    )

    state_data = setup_training_state(
        config, accelerator, logger, args, train_dataloader,
        base_model, controlnet, model_ema, optimizer, lr_scheduler,
    )

    tme_ckpt = getattr(config, "resume_tme_checkpoint", None)
    if tme_ckpt:
        step = load_sim_checkpoint(
            tme_ckpt, tme_module, optimizer_tme, lr_scheduler_tme,
            device=accelerator.device,
        )
        logger.info(f"Resumed TME module from step {step} ({tme_ckpt})")

    models = {
        'base_model':       base_model,
        'controlnet':       controlnet,
        'model_ema':        model_ema,
        'vae':              vae,
        'train_diffusion':  train_diffusion,
        'optimizer':        optimizer,
        'optimizer_d':      None,
        'lr_scheduler':     lr_scheduler,
        'train_dataloader': train_dataloader,
        'accelerator':      accelerator,
        'config':           config,
        'logger':           logger,
        'args':             args,
        'tme_module':       tme_module,
        'optimizer_tme':    optimizer_tme,
        'lr_scheduler_tme': lr_scheduler_tme,
        **state_data,
    }
    train_controlnet_exp(models)


if __name__ == "__main__":
    main()
