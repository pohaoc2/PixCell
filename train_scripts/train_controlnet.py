
import os
import time
import datetime
import torch
import torch.nn as nn
from diffusion.utils.mmcv_compat import LogBuffer
import numpy as np
from diffusion.utils.misc import DebugUnderflowOverflow
from diffusion.utils.checkpoint import save_checkpoint
from diffusion.utils.dist_utils import get_world_size
from train_scripts.initialize_models import initialize_models, ema_update

def train(models):
    """
    Main training loop for PixCell ControlNet
    
    Args:
        models: Dictionary containing all training components from initialize_models()
    """
    # Unpack all components
    base_model = models['base_model']
    model_ema = models['model_ema']
    vae = models['vae']
    train_diffusion = models['train_diffusion']
    
    optimizer = models['optimizer']
    optimizer_d = models['optimizer_d']
    lr_scheduler = models['lr_scheduler']
    train_dataloader = models['train_dataloader']
    
    discriminator = models['discriminator']
    segmentation_checker = models['segmentation_checker']
    
    accelerator = models['accelerator']
    config = models['config']
    logger = models['logger']
    
    start_epoch = models['start_epoch']
    start_step = models['start_step']
    skip_step = models['skip_step']
    total_steps = models['total_steps']
    
    args = models['args']
    
    # Training flags
    use_discriminator = config.get('use_discriminator', False)
    use_segmentation = config.get('use_segmentation_consistency', False)
    
    # Loss weights
    lambda_adv = config.get('lambda_adv', 0.1) if use_discriminator else 0.0
    lambda_seg = config.get('lambda_seg', 0.1) if use_segmentation else 0.0
    
    logger.info("=" * 50)
    logger.info("Starting Training Loop")
    logger.info(f"Total epochs: {config.num_epochs}")
    logger.info(f"Total steps: {total_steps}")
    logger.info(f"Starting from epoch {start_epoch}, step {start_step}")
    logger.info(f"Batch size: {config.train_batch_size}")
    logger.info(f"Gradient accumulation steps: {config.gradient_accumulation_steps}")
    if use_discriminator:
        logger.info(f"Adversarial loss weight: {lambda_adv}")
    if use_segmentation:
        logger.info(f"Segmentation consistency weight: {lambda_seg}")
    logger.info("=" * 50)
    
    global_step = start_step
    load_vae_feat = getattr(train_dataloader.dataset, 'load_vae_feat', False)
    log_buffer = LogBuffer()
    time_start, last_tic = time.time(), time.time()
    # Training loop
    # Now you train the model
    for epoch in range(start_epoch + 1, config.num_epochs + 1):
        data_time_start= time.time()
        data_time_all = 0
        for step, batch in enumerate(train_dataloader):

            if step < skip_step:
                if (step + 1) % 50 == 0 and accelerator.is_main_process:
                    info = f"Skipping Step/Epoch [{global_step}/{epoch}][{step + 1}/{len(train_dataloader)}] "
                    logger.info(info)
                continue    # skip training computations

            # Unpack batch
            # batch contains: [vae_feat, ssl_feat, cell_mask, data_info]
            if load_vae_feat:
                z = batch[0]
            else:
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=(config.mixed_precision == 'fp16' or config.mixed_precision == 'bf16')):
                        posterior = vae.encode(batch[0]).latent_dist
                        if config.sample_posterior:
                            z = posterior.sample()
                        else:
                            z = posterior.mode()

            if hasattr(config, 'shift_factor'):
                z = z - config.shift_factor

            clean_images = z * config.scale_factor
            y = batch[1]   # SSL embeddings (UNI)
            y = y / np.sqrt(y.shape[1])
            cell_mask = batch[2]  # Cell segmentation mask
            data_info = batch[3]
            if 0:
                print(f"y.shape: {y.shape}")
                print(f"np.unique(cell_mask.cpu().numpy()): {np.unique(cell_mask.cpu().numpy())}")
                #dtype of cell_mask
                print(f"cell_mask.dtype: {cell_mask.dtype}")
                print(f"cell_mask.shape: {cell_mask.shape}")
                print(f"clean_images.shape: {clean_images.shape}")
                #print(data_info)
                asd()
            # Process cell mask for ControlNet
            # Assuming cell_mask is binary [B, 1, H, W] at image resolution
            # Need to downsample to latent resolution if necessary
            cell_mask_latent = cell_mask
            if cell_mask.shape[-1] != clean_images.shape[-1]:
                cell_mask_latent = nn.functional.interpolate(
                    cell_mask.float(), 
                    size=(clean_images.shape[-2], clean_images.shape[-1]), 
                    mode='nearest'
                )
            #print(f"cell_mask_latent.shape: {cell_mask_latent.shape}")
            # Sample a random timestep for each image
            bs = clean_images.shape[0]
            timesteps = torch.randint(0, config.train_sampling_steps, (bs,), device=clean_images.device).long()
            grad_norm = None
            grad_norm_d = None
            data_time_all += time.time() - data_time_start
            
            # =====================================================================
            # Train Discriminator (if enabled)
            # =====================================================================
            if use_discriminator and discriminator is not None:
                discriminator.train()
                
                with accelerator.accumulate(discriminator):
                    optimizer_d.zero_grad()
                    
                    # Generate fake samples
                    with torch.no_grad():
                        # Add noise to clean images
                        noise = torch.randn_like(clean_images)
                        noisy_images = train_diffusion.q_sample(clean_images, timesteps, noise=noise)
                        
                        # Denoise (predict x0 from noisy)
                        model_kwargs_disc = dict(
                            y=y,
                            mask=None,
                            data_info=data_info,
                            controlnet_cond=cell_mask_latent
                        )
                        model_output = base_model(noisy_images, timesteps, **model_kwargs_disc)
                        
                        # Get predicted clean image (x0)
                        if hasattr(config, 'pred_sigma') and config.pred_sigma:
                            pred_noise, _ = model_output.chunk(2, dim=1)
                        else:
                            pred_noise = model_output
                        
                        # Predict x0 from noise prediction
                        fake_images = train_diffusion.predict_start_from_noise(
                            noisy_images, timesteps, pred_noise
                        )
                        
                        # Clamp to valid range
                        fake_images = torch.clamp(fake_images, -1, 1)
                    
                    # Discriminator forward pass
                    disc_config = config.discriminator
                    if disc_config['type'] == 'latent':
                        # Work in latent space
                        real_pred = discriminator(clean_images)
                        fake_pred = discriminator(fake_images.detach())
                    elif disc_config['type'] == 'conditional':
                        # Use conditioning (mask + embedding)
                        # Note: For image-space discriminator, need to decode latents first
                        # For simplicity, we work in latent space here
                        real_pred = discriminator(clean_images, condition=cell_mask_latent, embedding=y.squeeze(1))
                        fake_pred = discriminator(fake_images.detach(), condition=cell_mask_latent, embedding=y.squeeze(1))
                    else:  # patchgan
                        real_pred = discriminator(clean_images)
                        fake_pred = discriminator(fake_images.detach())
                    
                    # Discriminator loss
                    from discriminator import discriminator_loss
                    loss_d = discriminator_loss(
                        real_pred, fake_pred, 
                        loss_type=disc_config.get('loss_type', 'hinge')
                    )
                    
                    accelerator.backward(loss_d)
                    if accelerator.sync_gradients:
                        grad_norm_d = accelerator.clip_grad_norm_(
                            discriminator.parameters(), 
                            config.gradient_clip
                        )
                    optimizer_d.step()
            
            # =====================================================================
            # Train Generator (Diffusion Model + Adversarial Loss + Consistency)
            # =====================================================================
            with accelerator.accumulate(base_model):
                # Predict the noise residual with ControlNet conditioning
                optimizer.zero_grad()
                
                # For ControlNet, we need to pass the cell mask as additional conditioning
                # The mask is provided through model_kwargs
                model_kwargs = dict(
                    y=y,
                    control_input=cell_mask_latent,  # CRITICAL: Cell mask for ControlNet
                    mask=None,  # No attention mask for embeddings
                    data_info=data_info,
                )
                
                # Diffusion loss
                loss_term = train_diffusion.training_losses(
                    base_model, 
                    clean_images, 
                    timesteps, 
                    model_kwargs=model_kwargs
                )
                loss_diffusion = loss_term['loss'].mean()
                loss = loss_diffusion
                
                # Adversarial loss (if discriminator is used)
                loss_adv = torch.tensor(0.0).to(loss.device)
                if use_discriminator and discriminator is not None:
                    # Generate prediction for adversarial loss
                    noise = torch.randn_like(clean_images)
                    noisy_images = train_diffusion.q_sample(clean_images, timesteps, noise=noise)
                    
                    model_output = base_model(
                        noisy_images, 
                        timesteps, 
                        y=y,
                        control_input=cell_mask_latent,
                        mask=None,
                        data_info=data_info
                    )
                    
                    if hasattr(config, 'pred_sigma') and config.pred_sigma:
                        pred_noise, _ = model_output.chunk(2, dim=1)
                    else:
                        pred_noise = model_output
                    
                    fake_images = train_diffusion.predict_start_from_noise(
                        noisy_images, timesteps, pred_noise
                    )
                    fake_images = torch.clamp(fake_images, -1, 1)
                    
                    # Get discriminator prediction on generated images
                    disc_config = config.discriminator
                    if disc_config['type'] == 'latent':
                        fake_pred = discriminator(fake_images)
                    elif disc_config['type'] == 'conditional':
                        fake_pred = discriminator(fake_images, condition=cell_mask_latent, embedding=y.squeeze(1))
                    else:
                        fake_pred = discriminator(fake_images)
                    
                    # Generator adversarial loss
                    from discriminator import generator_loss
                    loss_adv = generator_loss(fake_pred, loss_type=disc_config.get('loss_type', 'hinge'))
                    
                    # Combine losses
                    adv_weight = config.get('adversarial_weight', 0.1)
                    loss = loss_diffusion + adv_weight * loss_adv
                
                # Cell segmentation consistency loss (if enabled)
                loss_consistency = torch.tensor(0.0).to(loss.device)
                if config.get('use_segmentation_consistency', False) and segmentation_checker is not None:
                    # We need to decode latents to images for segmentation
                    # Only compute this loss periodically to save computation
                    if global_step % config.get('consistency_check_interval', 10) == 0:
                        with torch.no_grad():
                            # Decode fake_images if available, otherwise generate new ones
                            if 'fake_images' not in locals():
                                noise = torch.randn_like(clean_images)
                                noisy_images = train_diffusion.q_sample(clean_images, timesteps, noise=noise)
                                model_output = base_model(noisy_images, timesteps, **model_kwargs)
                                
                                if hasattr(config, 'pred_sigma') and config.pred_sigma:
                                    pred_noise, _ = model_output.chunk(2, dim=1)
                                else:
                                    pred_noise = model_output
                                
                                fake_images = train_diffusion.predict_start_from_noise(
                                    noisy_images, timesteps, pred_noise
                                )
                                fake_images = torch.clamp(fake_images, -1, 1)
                            
                            # Decode latents to images for segmentation
                            if vae is not None:
                                # Scale and shift back
                                fake_images_unscaled = fake_images / config.scale_factor
                                if hasattr(config, 'shift_factor'):
                                    fake_images_unscaled = fake_images_unscaled + config.shift_factor
                                
                                # Decode
                                decoded_images = vae.decode(fake_images_unscaled.to(torch.float16)).sample
                                decoded_images = torch.clamp(decoded_images, -1, 1)
                            else:
                                # If no VAE, work in latent space (less accurate)
                                decoded_images = fake_images
                        
                        # Segment the generated images
                        pred_masks = segmentation_checker.segment_image(decoded_images.float())
                        
                        # Compute consistency loss
                        from cell_segmentation_consistency import mask_consistency_loss
                        loss_consistency = mask_consistency_loss(
                            pred_masks, 
                            cell_mask,  # Original full-resolution mask
                            loss_type=config.get('consistency_loss_type', 'combined')
                        )
                        
                        # Add to total loss
                        consistency_weight = config.get('consistency_weight', 0.5)
                        loss = loss + consistency_weight * loss_consistency
                
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(base_model.parameters(), config.gradient_clip)
                optimizer.step()
                lr_scheduler.step()
                if accelerator.sync_gradients:
                    ema_update(model_ema, base_model, config.ema_rate)

            lr = lr_scheduler.get_last_lr()[0]
            logs = {args.loss_report_name: accelerator.gather(loss).mean().item()}
            
            # Add individual loss components
            logs['loss_diffusion'] = accelerator.gather(loss_diffusion).mean().item()
            if use_discriminator and discriminator is not None:
                logs['loss_adv'] = accelerator.gather(loss_adv).mean().item()
                if 'loss_d' in locals():
                    logs['loss_d'] = accelerator.gather(loss_d).mean().item()
                if grad_norm_d is not None:
                    logs['grad_norm_d'] = accelerator.gather(grad_norm_d).mean().item()
            
            if config.get('use_segmentation_consistency', False):
                logs['loss_consistency'] = accelerator.gather(loss_consistency).mean().item()
            
            if grad_norm is not None:
                logs.update(grad_norm=accelerator.gather(grad_norm).mean().item())
            log_buffer.update(logs)
            
            if (step + 1) % config.log_interval == 0 or (step + 1) == 1:
                t = (time.time() - last_tic) / config.log_interval
                t_d = data_time_all / config.log_interval
                avg_time = (time.time() - time_start) / (global_step + 1)
                eta = str(datetime.timedelta(seconds=int(avg_time * (total_steps - global_step - 1))))
                eta_epoch = str(datetime.timedelta(seconds=int(avg_time * (len(train_dataloader) - step - 1))))
                log_buffer.average()

                info = f"Step/Epoch [{global_step}/{epoch}][{step + 1}/{len(train_dataloader)}]:total_eta: {eta}, " \
                    f"epoch_eta:{eta_epoch}, time_all:{t:.3f}, time_data:{t_d:.3f}, lr:{lr:.3e}, "
                info += f's:({base_model.module.h}, {base_model.module.w}), ' if hasattr(base_model, 'module') else f's:({base_model.h}, {base_model.w}), '

                info += ', '.join([f"{k}:{v:.4f}" for k, v in log_buffer.output.items()])
                logger.info(info)
                last_tic = time.time()
                log_buffer.clear()
                data_time_all = 0
            logs.update(lr=lr)
            accelerator.log(logs, step=global_step)

            global_step += 1
            data_time_start = time.time()

            def save_checkpoint_fn():
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    os.umask(0o000)
                    save_checkpoint(os.path.join(config.work_dir, 'checkpoints'),
                                    epoch=epoch,
                                    step=global_step,
                                    model=accelerator.unwrap_model(base_model),
                                    model_ema=accelerator.unwrap_model(model_ema),
                                    optimizer=optimizer,
                                    lr_scheduler=lr_scheduler)
                    
            # save checkpoint when time limit has reached
            time_elapsed_minutes = (time.time() - time_start) / 60
            time_to_save = (time_elapsed_minutes >= args.slurm_time_limit) and (ckpt_time_limit_saved == False)

            if (config.save_model_steps and global_step % config.save_model_steps == 0) or time_to_save:
                if not ckpt_time_limit_saved:
                    ckpt_time_limit_saved = True
                save_checkpoint_fn()

        if epoch % config.save_model_epochs == 0 or epoch == config.num_epochs:
            save_checkpoint_fn()
        accelerator.wait_for_everyone()


def generate_validation_samples(model, vae, train_diffusion, batch, config, device):
    """Generate validation samples for logging"""
    # Simple sampling function - you can make this more sophisticated
    masks = batch.get('mask', None)
    uni_features = batch.get('uni_features', None)
    
    # Start from random noise
    latents = torch.randn(
        min(4, batch['img'].shape[0]),  # Generate up to 4 samples
        4,  # Latent channels
        config.image_size // 8,
        config.image_size // 8,
        device=device
    )
    
    # Denoise
    for t in reversed(range(train_diffusion.num_timesteps)):
        timesteps = torch.full((latents.shape[0],), t, device=device, dtype=torch.long)
        
        model_output = model(
            latents,
            timesteps,
            y=uni_features[:latents.shape[0]] if uni_features is not None else None,
            mask=masks[:latents.shape[0]] if masks is not None else None,
        )
        
        if config.pred_sigma:
            model_pred, _ = model_output.chunk(2, dim=1)
        else:
            model_pred = model_output
        
        # Simple DDPM step
        latents = train_diffusion.p_sample(latents, model_pred, timesteps)
    
    # Decode to images
    latents = latents / config.scale_factor
    images = vae.decode(latents).sample
    
    # Convert to [0, 1] range
    images = (images + 1.0) / 2.0
    images = torch.clamp(images, 0.0, 1.0)
    
    return images


if __name__ == "__main__":
    models = initialize_models([
        'configs/pan_cancer/config_controlnet_gan.py',
    ])
    train(models)

    model = models['base_model']
    model.eval()
    model.to(device)
    model.load_state_dict(torch.load(os.path.join(config.work_dir, 'checkpoints', 'model_ema.pth')))
    vae = models['vae']
    vae.eval()
    vae.to(device)
    vae.load_state_dict(torch.load(os.path.join(config.work_dir, 'checkpoints', 'vae.pth')))
    train_diffusion = models['train_diffusion']
    train_diffusion.eval()
    train_diffusion.to(device)
    batch = next(iter(models['train_dataloader']))
    config = models['config']
    device = models['device']
    images = generate_validation_samples(model, vae, train_diffusion, batch, config, device)
    images.save(os.path.join(config.work_dir, 'checkpoints', 'images.png'))
