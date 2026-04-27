"""
train_controlnet_exp.py

PixCell ControlNet training on PAIRED experimental H&E + CODEX-derived TME channels.

Design:
    1. PairedExpControlNetData  — paired dataset (H&E + multichannel, same tile_id)
    2. CFG dropout              — zero UNI embedding with probability cfg_dropout_prob
                                  to enable TME-only inference via null_uni_embed()
    3. Channel reliability weights — attenuate approximate CODEX channels before TMEEncoder

Entry point: use stage2_train.py (calls main() here).
"""
import os
import json
import time
from copy import deepcopy
from pathlib import Path

import torch
from diffusion.utils.lr_scheduler import build_lr_scheduler
from diffusion.utils.optimizer import build_optimizer

from tools.stage3.common import make_inference_scheduler
from train_scripts.initialize_models import (
    initialize_config_and_accelerator,
    initialize_models,
    setup_training_state,
    ema_update,
)
from diffusion.data.builder import build_dataloader

from diffusion.data.datasets.paired_exp_controlnet_dataset import (
    PairedExpControlNetData,
    build_exp_index,
)
from train_scripts.training_utils import (
    training_losses_controlnet,
    save_checkpoint_with_tme,
    load_tme_checkpoint,
    _build_tme_module_and_optimizers,
)
from train_scripts.exp_config_utils import (
    resolve_exp_active_channels,
    resolve_exp_dataset_kwargs,
)
from tools.channel_group_utils import split_channels_to_groups, apply_group_dropout


def _conditioning_mode(config) -> str:
    mode = getattr(config, "tme_input_mode", None)
    if mode is not None:
        return mode
    if getattr(config, "channel_groups", None) is not None:
        return "grouped"
    return "non_mask_channels"


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
    dataset_kwargs = resolve_exp_dataset_kwargs(config)
    active_channels = dataset_kwargs["active_channels"]
    exp_root = Path(config.exp_data_root)
    exp_index_path = exp_root / dataset_kwargs["exp_index_h5"]
    if not exp_index_path.exists():
        exp_channels_path = exp_root / dataset_kwargs["exp_channels_dir"]
        logger.info(
            f"Paired-exp index not found at {exp_index_path}; "
            f"building from {exp_channels_path}."
        )
        build_exp_index(
            exp_channels_dir=exp_channels_path,
            output_path=exp_index_path,
            resolution=config.image_size,
        )

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset = PairedExpControlNetData(
        root=config.exp_data_root,
        resolution=config.image_size,
        **dataset_kwargs,
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

    Paired-exp training loop with CFG dropout and channel reliability weighting.
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
    channel_groups = getattr(config, "channel_groups", None)
    group_dropout_probs = getattr(config, "group_dropout_probs", {})
    use_multi_group = channel_groups is not None
    conditioning_mode = _conditioning_mode(config)
    channel_weights = getattr(config, "channel_reliability_weights", None)

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

    train_log_fh = None
    if accelerator.is_main_process:
        train_log_path = Path(config.work_dir) / "train_log.jsonl"
        train_log_path.parent.mkdir(parents=True, exist_ok=True)
        train_log_fh = train_log_path.open("a", buffering=1, encoding="utf-8")

    _proj_grad_norms: dict = {}
    _tme_residuals:   dict = {}
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

            tme_dtype = next(tme_module.parameters()).dtype

            # Prepare TME inputs (no learnable ops here — pure data shaping).
            if conditioning_mode == "grouped":
                active_channels = resolve_exp_active_channels(config)
                tme_channel_dict = split_channels_to_groups(
                    control_input.to(dtype=tme_dtype), active_channels, channel_groups,
                )
                active_groups_per_sample = apply_group_dropout(
                    [g["name"] for g in channel_groups], group_dropout_probs, batch_size=bs,
                )
                # Per-sample dropout: zero out channels for groups dropped in each sample
                for b_idx in range(bs):
                    for g in channel_groups:
                        gname = g["name"]
                        if gname not in active_groups_per_sample[b_idx] and gname in tme_channel_dict:
                            tme_channel_dict[gname][b_idx] = 0.0
            elif conditioning_mode == "all_channels":
                tme_inputs = control_input.to(dtype=tme_dtype)
            else:
                tme_channels = control_input[:, 1:, :, :].to(dtype=tme_dtype)
                if channel_weights is not None:
                    w = torch.tensor(
                        channel_weights, device=tme_channels.device, dtype=tme_channels.dtype
                    ).view(1, -1, 1, 1)
                    tme_channels = tme_channels * w

            # 3. Training step — TME forward is inside accumulate so its gradient
            # graph is always built after zero_grad(), guaranteeing that
            # optimizer_tme.step() sees non-None gradients on every step.
            grad_norm = None
            with accelerator.accumulate(controlnet, tme_module):
                optimizer.zero_grad()
                optimizer_tme.zero_grad()

                if conditioning_mode == "grouped":
                    fused, _tme_residuals = tme_module(
                        vae_mask.to(dtype=tme_dtype), tme_channel_dict, return_residuals=True,
                    )
                    if getattr(config, "zero_mask_latent", False):
                        fused = fused - vae_mask.to(dtype=tme_dtype)
                    ctrl_latent = fused.float()
                elif conditioning_mode == "all_channels":
                    ctrl_latent = tme_module(vae_mask.to(dtype=tme_dtype), tme_inputs).float()
                else:
                    ctrl_latent = tme_module(vae_mask.to(dtype=tme_dtype), tme_channels).float()

                model_kwargs = dict(
                    y=y, mask=None, data_info=data_info, control_input=ctrl_latent,
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
                    _proj_grad_norms = {}
                    if conditioning_mode == "grouped":
                        for _gname, _gblock in accelerator.unwrap_model(tme_module).groups.items():
                            _g = _gblock.cross_attn.proj.weight.grad
                            if _g is not None:
                                _proj_grad_norms[_gname] = (
                                    _g.norm().item(),
                                    _gblock.cross_attn.proj.weight.abs().max().item(),
                                )
                    grad_norm_ctrl = accelerator.clip_grad_norm_(controlnet.parameters(), config.gradient_clip)
                    optimizer.step()
                    lr_scheduler.step()
                    grad_norm_tme = accelerator.clip_grad_norm_(tme_module.parameters(), config.gradient_clip)
                    optimizer_tme.step()
                    lr_scheduler_tme.step()
                    grad_norm = max(float(grad_norm_ctrl), float(grad_norm_tme))

                if accelerator.is_main_process:
                    ema_update(model_ema, controlnet, config.ema_rate)

            # 4. Logging
            if accelerator.sync_gradients:
                global_step += 1
                if (global_step % config.log_interval == 0
                        and accelerator.is_main_process
                        and use_multi_group):
                    for _gname, _delta in _tme_residuals.items():
                        logger.info(f"  delta_mean[{_gname}]={_delta.abs().mean():.3e}")
                    for _gname, (_gnorm, _wmax) in _proj_grad_norms.items():
                        logger.info(f"  proj_grad[{_gname}]={_gnorm:.3e}  proj_wmax={_wmax:.3e}")

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
                    if train_log_fh is not None:
                        train_log_fh.write(
                            json.dumps(
                                {
                                    "step": int(global_step),
                                    "loss": float(loss.detach().item()),
                                    "grad_norm": grad_norm,
                                }
                            )
                            + "\n"
                        )
                    last_tic = time.time()

                if global_step % config.save_model_steps == 0:
                    save_checkpoint_with_tme(
                        accelerator, controlnet, tme_module, model_ema,
                        optimizer, optimizer_tme, lr_scheduler, lr_scheduler_tme,
                        global_step, epoch, config, logger,
                    )
                    if accelerator.is_main_process and use_multi_group:
                        try:
                            generate_validation_visualizations(
                                tme_module=tme_module,
                                controlnet=controlnet,
                                base_model=base_model,
                                vae=vae,
                                train_diffusion=train_diffusion,
                                val_control_input=batch[2][:1],
                                val_vae_mask=batch[3][:1],
                                val_uni_embeds=batch[1][:1],
                                config=config,
                                save_dir=os.path.join(config.work_dir, f"vis/step_{global_step}"),
                                device=accelerator.device,
                            )
                        except Exception as e:
                            logger.warning(f"Validation vis failed at step {global_step}: {e}")

            if global_step >= total_steps:
                logger.info(f"Reached max steps ({total_steps}). Stopping.")
                break

        if epoch % config.save_model_epochs == 0 or epoch == config.num_epochs:
            save_checkpoint_with_tme(
                accelerator, controlnet, tme_module, model_ema,
                optimizer, optimizer_tme, lr_scheduler, lr_scheduler_tme,
                global_step, epoch, config, logger,
            )
        if global_step >= total_steps:
            break

    if train_log_fh is not None:
        train_log_fh.close()

    logger.info("=" * 80)
    logger.info("Fine-tuning Complete!")
    logger.info("=" * 80)


@torch.no_grad()
def generate_validation_visualizations(
    tme_module, controlnet, base_model, vae, train_diffusion,
    val_control_input, val_vae_mask, val_uni_embeds,
    config, save_dir, device,
):
    """Generate attention heatmaps and overview visualization for one fixed sample."""
    from pathlib import Path
    from tools.channel_group_utils import split_channels_to_groups
    from tools.stage3.figures import save_attention_heatmap_figure, save_overview_figure
    from train_scripts.inference_controlnet import denoise
    import numpy as np

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    channel_groups = config.channel_groups
    conditioning_mode = _conditioning_mode(config)
    active_channels = resolve_exp_active_channels(config)
    dtype = next(tme_module.parameters()).dtype
    vae_scale, vae_shift = config.scale_factor, config.shift_factor
    mask_latent = val_vae_mask.to(device, dtype=dtype)
    mask_latent_scaled = (mask_latent - vae_shift) * vae_scale

    tme_module.eval()
    if conditioning_mode == "grouped":
        tme_inputs = split_channels_to_groups(
            val_control_input.to(device, dtype=dtype), active_channels, channel_groups,
        )
        fused, _residuals, attn_maps = tme_module(
            mask_latent_scaled, tme_inputs,
            return_residuals=True, return_attn_weights=True,
        )
    elif conditioning_mode == "all_channels":
        fused, _residuals = tme_module(
            mask_latent_scaled,
            val_control_input.to(device, dtype=dtype),
            return_residuals=True,
        )
        attn_maps = {}
    else:
        fused, _residuals = tme_module(
            mask_latent_scaled,
            val_control_input[:, 1:, :, :].to(device, dtype=dtype),
            return_residuals=True,
        )
        attn_maps = {}
    tme_module.train()

    scheduler = make_inference_scheduler(num_steps=20, device=device)
    latent_shape = (1, 16, config.image_size // 8, config.image_size // 8)
    latents = torch.randn(latent_shape, device=device, dtype=dtype)
    latents = latents * scheduler.init_noise_sigma

    controlnet.eval()
    denoised = denoise(
        latents=latents,
        uni_embeds=val_uni_embeds.to(device, dtype=dtype),
        controlnet_input_latent=fused,
        scheduler=scheduler,
        controlnet_model=controlnet,
        pixcell_controlnet_model=base_model,
        guidance_scale=2.5,
        device=device,
    )
    controlnet.train()

    scaled_latents = (denoised.to(dtype) / vae_scale) + vae_shift
    gen_img = vae.decode(scaled_latents, return_dict=False)[0]
    gen_img = (gen_img / 2 + 0.5).clamp(0, 1)
    gen_np = (gen_img.cpu().permute(0, 2, 3, 1).numpy()[0] * 255).astype(np.uint8)

    mask_ch = val_control_input[0, 0].cpu().numpy()
    mask_rgb = np.stack([mask_ch] * 3, axis=-1)
    mask_rgb = (mask_rgb * 255).astype(np.uint8)

    ctrl_full_np = val_control_input[0].detach().cpu().numpy()
    save_overview_figure(
        ctrl_full=ctrl_full_np,
        active_channels=active_channels,
        gen_np=gen_np,
        save_path=save_dir / "overview.png",
    )
    save_attention_heatmap_figure(mask_rgb, gen_np, attn_maps, save_dir / "attention_heatmaps.png")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    init_data   = initialize_config_and_accelerator()
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

    # Rebuild optimizer_tme after prepare: accelerator.prepare() may replace
    # nn.Parameter objects during dtype conversion (PyTorch _apply creates new
    # Parameter instances), leaving optimizer_tme with stale param references
    # that are no longer in the computation graph.  Collect params from the
    # prepared model so the optimizer always steps on live parameters.
    _tme = accelerator.unwrap_model(tme_module)
    _tme_proj_lr = getattr(config, "tme_proj_lr", None)
    if _tme_proj_lr is not None and any("cross_attn.proj" in n for n, _ in _tme.named_parameters()):
        _proj  = [p for n, p in _tme.named_parameters() if "cross_attn.proj" in n]
        _other = [p for n, p in _tme.named_parameters() if "cross_attn.proj" not in n]
        optimizer_tme = torch.optim.AdamW(
            [{"params": _proj,  "lr": _tme_proj_lr},
             {"params": _other, "lr": getattr(config, "tme_lr", 1e-5)}],
            weight_decay=config.optimizer.get("weight_decay", 0.0),
            betas=tuple(config.optimizer.get("betas", (0.9, 0.999))),
            eps=config.optimizer.get("eps", 1e-8),
        )
    else:
        _tme_optcfg = deepcopy(config.optimizer)
        _tme_optcfg["lr"] = getattr(config, "tme_lr", config.optimizer.get("lr", 1e-4))
        optimizer_tme = build_optimizer(_tme, _tme_optcfg)
    # Re-prepare the rebuilt optimizer so Accelerate tracks it correctly
    # (handles bf16/fp16 gradient scaling and distributed gradient sync).
    optimizer_tme = accelerator.prepare(optimizer_tme)
    lr_scheduler_tme = build_lr_scheduler(config, optimizer_tme, train_dataloader, lr_scale_ratio=1)

    state_data = setup_training_state(
        config, accelerator, logger, args, train_dataloader,
        base_model, controlnet, model_ema, optimizer, lr_scheduler,
    )

    tme_ckpt = getattr(config, "resume_tme_checkpoint", None)
    if tme_ckpt:
        reset_opt = getattr(config, "reset_tme_optimizer", False)
        step = load_tme_checkpoint(
            tme_ckpt, tme_module,
            optimizer_tme=None if reset_opt else optimizer_tme,
            lr_scheduler_tme=None if reset_opt else lr_scheduler_tme,
            device=accelerator.device,
        )
        logger.info(
            f"Resumed TME module from step {step} ({tme_ckpt})"
            + (" [optimizer reset]" if reset_opt else "")
        )

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
