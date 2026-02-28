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
"""
# %%
import os
import time
from copy import deepcopy
import glob
from PIL import Image
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import numpy as np
import cv2
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
import train_scripts.inference_controlnet as inference_controlnet
# ── Diffusion utils (same ones initialize_models uses) ───────────────────────
from diffusion.data.builder import build_dataloader
from diffusion.model.builder import build_model          # ← used for tme_module
from diffusion.utils.checkpoint import save_checkpoint
from diffusion.utils.optimizer import build_optimizer    # ← used for both optimizers
from diffusion.utils.lr_scheduler import build_lr_scheduler
from diffusion.utils.dist_utils import get_world_size

# ── Sim-specific dataset ──────────────────────────────────────────────────────
from diffusion.data.datasets.sim_controlnet_dataset import SimControlNetData

# ── Helper functions ───────────────────────────────────────────────────────────
def prepare_controlnet_input(idx, vae=None, scheduler=None, device='cpu'):
    
    latent_shape = (1, 16, 32, 32)
    latents = torch.randn(latent_shape, device=device, dtype=torch.float32).to(device)
    latents = latents * scheduler.init_noise_sigma
    
    uni_embeds = torch.from_numpy(np.load(f"../dummy_sim_data/features/TCGA_dummy_{idx:04d}_uni.npy"))
    #uni_embeds = torch.from_numpy(np.load(f"../data/features_tcga_3660/0_{idx}_uni.npy"))
    uni_embeds = uni_embeds.view(1, 1, 1, 1536).to(device)
    mask_path = "../test_mask.png"
    controlnet_input = np.asarray(Image.open(mask_path).convert("RGB").resize((256, 256)))
    
    controlnet_input_torch = torch.from_numpy(controlnet_input.copy()/255.).float().to(device)
    controlnet_input_torch = controlnet_input_torch.permute(2, 0, 1).unsqueeze(0)
    controlnet_input_torch = 2 * (controlnet_input_torch - 0.5)
    vae_scale = vae.config.scaling_factor
    vae_shift = getattr(vae.config, "shift_factor", 0)
    vae.to(device, dtype=controlnet_input_torch.dtype)
    controlnet_input_latent = vae.encode(controlnet_input_torch.float()).latent_dist.mean
    controlnet_input_latent = (controlnet_input_latent-vae_shift)*vae_scale
    controlnet_input = torch.from_numpy(controlnet_input).float().to(device)
    return latents, uni_embeds, controlnet_input_latent, controlnet_input

def inference_pretrained_controlnet(controlnet,
                                    base_model,
                                    accelerator,
                                    vae,
                                    device='cpu'):
    idx = 3
    scheduler_folder = "../pretrained_models/pixcell-256/scheduler/"
    from diffusers import DPMSolverMultistepScheduler
    scheduler = DPMSolverMultistepScheduler.from_pretrained(
        scheduler_folder,
    )
    from diffusers import DDPMScheduler

    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
        prediction_type="epsilon",
        clip_sample=False,
    )
    scheduler.set_timesteps(20, device=device)
    print(type(scheduler))
    print(scheduler.config)

    latents, uni_embeds, controlnet_input_latent, controlnet_input = prepare_controlnet_input(idx, vae, scheduler, device=accelerator.device)
    controlnet.eval()
    base_model.eval()
    denoised_latents = inference_controlnet.denoise(latents,
            uni_embeds,
            controlnet_input_latent,
            scheduler,
            controlnet,
            pixcell_controlnet_model=base_model,
            guidance_scale=2.5,
            num_inference_steps=50,
            conditioning_scale=1.0,
            device=accelerator.device)
    hist_image = cv2.imread(f"../data/tcga_3660/0_{idx}.png")
    hist_image = cv2.cvtColor(hist_image, cv2.COLOR_BGR2RGB)
    hist_image = Image.fromarray(hist_image)
    #hist_image = np.zeros_like(hist_image)
    mask_image = controlnet_input.cpu().numpy()
    generated_image = inference_controlnet.decode_latents(vae, denoised_latents, hist_image, mask_image, "generated_image.png")
    return generated_image

def _plot_control_input(config, train_dataloader):
    for step, batch in enumerate(train_dataloader):
        clean_images = batch[0]
        y = batch[1]
        control_input = batch[2].to('cpu')
        vae_mask = batch[3]
        data_info = batch[4]
        print(f"clean_images.shape: {clean_images.shape}")
        print(f"y.shape: {y.shape}")
        print(f"control_input.shape: {control_input.shape}")
        print(f"vae_mask.shape: {vae_mask.shape}")
        print(f"sim_idx = {data_info['sim_idx'].to('cpu')}")
        print('--------------------------------')
        
        fig, ax = plt.subplots(2, control_input.shape[1], figsize=(10, 5))
        channel_names = config.data.active_channels
        
        for i in range(control_input.shape[1]):
            print(f"{channel_names[i]}: range = {control_input[0, i, :, :].min()}, {control_input[0, i, :, :].max()}")
            ax[0, i].imshow(control_input[0, i, :, :], cmap='gray')
            ax[0, i].set_title(channel_names[i])
            if i == 0:
                ax[0, i].set_ylabel("from dataloader", fontsize=12)
            image_file_name = f"sim_{data_info['sim_idx'].to('cpu')[0]:04d}_*.png"
            file = glob.glob(f"{config.sim_data_root}/sim_channels/{channel_names[i]}/{image_file_name}")
            image = f"{file[0]}"
            image = Image.open(image)
            image = image.resize((256, 256))
            image = np.array(image)
            # if range is not [0, 1], normalize image to [0, 1]
            if image.min() != 0 or image.max() != 1:
                image = image / 255.0
            print(f"{channel_names[i]}: range = {image.min()}, {image.max()}")
            ax[1, i].imshow(image, cmap='gray')
            if i == 0:
                ax[1, i].set_ylabel("direct load", fontsize=12)
            ax[0, i].set_xticks([])
            ax[0, i].set_yticks([])

            ax[1, i].set_xticks([])
            ax[1, i].set_yticks([])
        plt.tight_layout()
        plt.show()
        break


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
        optimizer,         ← combined optimizer (2 param groups: controlnet + tme_module)
        lr_scheduler,
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
    tme_module.to(accelerator.device)
    logger.info(
        f"[TMEConditioningModule] n_tme_channels={n_tme_channels}  "
        f"trainable params="
        f"{sum(p.numel() for p in tme_module.parameters() if p.requires_grad):,}"
    )
    # ── Combined optimizer: two param groups, one optimizer, one scheduler ──────
    # param_groups in config lets each model have its own lr (and any other
    # per-group overrides). Falls back to a single default lr if not specified.
    #
    # Config example:
    #   optimizer = dict(
    #       type='AdamW', lr=1e-5, weight_decay=0.0, betas=(0.9,0.999), eps=1e-8,
    #       param_groups=[
    #           {"name": "controlnet", "lr": 1e-5},
    #           {"name": "tme_module",  "lr": 1e-4},
    #       ],
    #   )
    opt_cfg   = config.optimizer
    default_lr = opt_cfg.get('lr', 1e-5)
    models_by_name = {"controlnet": controlnet, "tme_module": tme_module}

    raw_groups = opt_cfg.get('param_groups', None)
    if raw_groups:
        # Config-driven: one entry per named model, inherits base lr if not overridden
        param_groups = [
            {
                "params": list(models_by_name[g['name']].parameters()),
                **{k: v for k, v in g.items() if k != 'name'},   # lr + any overrides
                "name": g['name'],
            }
            for g in raw_groups
        ]
    else:
        # Fallback: both groups use the same lr (old behaviour)
        tme_lr = getattr(config, "tme_lr", default_lr)
        param_groups = [
            {"params": list(controlnet.parameters()), "lr": default_lr, "name": "controlnet"},
            {"params": list(tme_module.parameters()),  "lr": tme_lr,    "name": "tme_module"},
        ]

    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=opt_cfg.get("weight_decay", 0.0),
        betas=opt_cfg.get("betas", (0.9, 0.999)),
        eps=opt_cfg.get("eps", 1e-8),
    )
    lr_scheduler = build_lr_scheduler(config, optimizer, train_dataloader, lr_scale_ratio=1)

    return {
        "train_dataloader": train_dataloader,
        "tme_module":        tme_module,
        "optimizer":         optimizer,
        "lr_scheduler":      lr_scheduler,
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

    tme_module        = models_dict['tme_module']   # param group 1 of optimizer

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
            

            # 3. Training step
            with accelerator.accumulate(controlnet, tme_module):   # <- CHANGED
                optimizer.zero_grad()

                model_kwargs = dict(
                    y=y, mask=None, data_info=data_info, control_input=vae_mask,
                )
                if 0:
                    print(f"clean_images.shape: {clean_images.shape}")
                    print(f"clean_images.mean(): {clean_images.mean()}")
                    print(f"clean_images.std(): {clean_images.std()}")
                    print(f"clean_images.min(): {clean_images.min()}")
                    print(f"clean_images.max(): {clean_images.max()}")
                    print(f"clean_images.norm(): {torch.norm(clean_images, p=2).item()}")
                    print('--------------------------------')
                    print(f"timesteps.shape: {timesteps.shape}")
                    print(f"y.shape: {y.shape}")
                    print(f"y.mean(): {y.mean()}")
                    print(f"y.std(): {y.std()}")
                    print(f"y.min(): {y.min()}")
                    print(f"y.max(): {y.max()}")
                    print(f"y.norm(): {torch.norm(y, p=2).item()}")
                    print('--------------------------------')
                    print(f"vae_mask.shape: {vae_mask.shape}")
                    print(f"vae_mask.mean(): {vae_mask.mean()}")
                    print(f"vae_mask.std(): {vae_mask.std()}")
                    print(f"vae_mask.min(): {vae_mask.min()}")
                    print(f"vae_mask.max(): {vae_mask.max()}")
                    print(f"vae_mask.norm(): {torch.norm(vae_mask, p=2).item()}")
                    print('--------------------------------')
                    asd()
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
                    accelerator.clip_grad_norm_(
                        list(controlnet.parameters()) + list(tme_module.parameters()),
                        config.gradient_clip,
                    )
                    optimizer.step()
                    lr_scheduler.step()

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
                        f"LR_tme:  {optimizer.param_groups[1]['lr']:.2e} "
                        f"Samples/s: {samples_per_sec:.2f}"
                    )
                    last_tic = time.time()

                if global_step % config.save_model_steps == 0:
                    _save_sim_checkpoint(
                        accelerator, controlnet, tme_module, model_ema,
                        optimizer, lr_scheduler,
                        global_step, epoch, config, logger,
                    )

            if global_step >= total_steps:
                logger.info(f"Reached max steps ({total_steps}). Stopping.")
                break

        if epoch % config.save_model_epochs == 0 or epoch == config.num_epochs:
            _save_sim_checkpoint(
                accelerator, controlnet, tme_module, model_ema,
                optimizer, lr_scheduler,
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
    optimizer, lr_scheduler,
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
                "optim_state": optimizer.state_dict(),      # shared optimizer
                "sched_state": lr_scheduler.state_dict(),   # shared scheduler
            },
            os.path.join(ckpt_dir, "tme_module.pth"),
        )
        logger.info(f"Saved checkpoint step={step} → {ckpt_dir}")

def load_sim_checkpoint(ckpt_dir, tme_module, optimizer=None,
                        lr_scheduler=None, device="cpu"):
    """Load TME module weights (+ optionally shared optimizer/scheduler) from checkpoint."""
    ckpt = torch.load(os.path.join(ckpt_dir, "tme_module.pth"), map_location=device)
    tme_module.load_state_dict(ckpt["model_state"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optim_state"])
    if lr_scheduler is not None:
        lr_scheduler.load_state_dict(ckpt["sched_state"])
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

# %%
def main():
    # %%
    config_path = "../configs/config_controlnet_sim.py"
    init_data   = initialize_config_and_accelerator([config_path])
    config      = init_data['config']
    accelerator = init_data['accelerator']
    logger      = init_data['logger']
    args        = init_data['args']
    # %%
    # Build frozen base model, trainable controlnet, EMA, VAE, diffusion
    model_data      = initialize_models(config, accelerator, logger)
    base_model      = model_data['base_model']
    controlnet      = model_data['controlnet']
    model_ema       = model_data['model_ema']
    vae             = model_data['vae']
    train_diffusion = model_data['train_diffusion']
    # %%
    # Build sim dataset + TME module + all optimizers in one call.
    # controlnet is passed in so its optimizer is built here — no second
    # initialize_dataset_and_optimizer call needed.
    sim_data         = initialize_sim_training(config, accelerator, logger, controlnet)
    train_dataloader = sim_data['train_dataloader']
    tme_module       = sim_data['tme_module']
    optimizer        = sim_data['optimizer']        # combined: controlnet + tme param groups
    lr_scheduler     = sim_data['lr_scheduler']
    # %%
    # Prepare everything with accelerator
    (
        base_model, controlnet, model_ema,
        optimizer, train_dataloader, lr_scheduler,
    ) = accelerator.prepare(
        base_model, controlnet, model_ema,
        optimizer, train_dataloader, lr_scheduler,
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
            tme_ckpt, tme_module, optimizer, lr_scheduler,
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
        **state_data,
    }
    # %%
    print(f"controlnet_blocks[0].weight max: {controlnet.controlnet_blocks[0].weight.abs().max():.6f}")
    # If this is 0.0, the model was re-initialized after loading
    with torch.no_grad():
        weight_0 = controlnet.controlnet_blocks[0].weight
        has_changed = (weight_0 != 0).any().item()
        max_val = weight_0.abs().max().item()
        
        print(f"Did weights move from zero? {has_changed}")
        print(f"Absolute max weight value: {max_val:.20f}")
    # %%
    _plot_control_input(config, train_dataloader)
    generated_image = inference_pretrained_controlnet(controlnet, base_model, accelerator, vae)
    # %%
    train_controlnet_sim(models)

# %%

if __name__ == "__main__":
    main()
# %%
