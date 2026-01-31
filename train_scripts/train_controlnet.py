from initialize_models import initialize_models
import os
import time
import datetime
import torch
import torch.nn as nn
from mmcv.runner import LogBuffer
from diffusion.utils.misc import DebugUnderflowOverflow
from diffusion.utils.checkpoint import save_checkpoint
from diffusion.utils.dist_utils import ema_update, get_world_size

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
    
    # Training loop
    for epoch in range(start_epoch, config.num_epochs):
        base_model.train()
        if discriminator is not None:
            discriminator.train()
            
        logger.info(f"Epoch {epoch + 1}/{config.num_epochs}")
        
        for step, batch in enumerate(train_dataloader):
            # Skip steps if resuming
            if step < skip_step:
                continue
                
            with accelerator.accumulate(base_model):
                # Get batch data
                images = batch['img'].to(accelerator.device)  # [B, C, H, W]
                prompts = batch.get('prompt', None)
                masks = batch.get('mask', None)  # Control signal (cell segmentation masks)
                uni_features = batch.get('uni_features', None)  # UNI embeddings
                
                # Encode images to latent space
                with torch.no_grad():
                    latents = vae.encode(images).latent_dist.sample()
                    latents = latents * config.scale_factor
                
                # Sample timesteps
                batch_size = latents.shape[0]
                timesteps = torch.randint(
                    0, 
                    train_diffusion.num_timesteps, 
                    (batch_size,), 
                    device=accelerator.device
                )
                
                # Add noise to latents
                noise = torch.randn_like(latents)
                noisy_latents = train_diffusion.q_sample(latents, timesteps, noise=noise)
                
                # Forward pass through ControlNet
                model_output = base_model(
                    noisy_latents,
                    timesteps,
                    y=uni_features,  # UNI embeddings as condition
                    mask=masks,  # Cell segmentation masks as control
                )
                
                # Compute diffusion loss
                if config.pred_sigma:
                    model_pred, model_var_values = model_output.chunk(2, dim=1)
                else:
                    model_pred = model_output
                
                # Compute target based on prediction type
                if config.get('prediction_type', 'epsilon') == 'epsilon':
                    target = noise
                elif config.get('prediction_type', 'epsilon') == 'v_prediction':
                    target = train_diffusion.get_v(latents, noise, timesteps)
                else:
                    target = latents
                
                # Main diffusion loss (MSE)
                loss_diffusion = torch.nn.functional.mse_loss(
                    model_pred.float(), 
                    target.float(), 
                    reduction='mean'
                )
                
                # Total loss starts with diffusion loss
                loss = loss_diffusion
                loss_dict = {'loss_diffusion': loss_diffusion.item()}
                
                # Adversarial loss (if discriminator enabled)
                if use_discriminator and discriminator is not None:
                    # Generate fake images from model predictions
                    with torch.no_grad():
                        # Predict x0 from model output
                        if config.get('prediction_type', 'epsilon') == 'epsilon':
                            pred_x0 = train_diffusion.predict_start_from_noise(
                                noisy_latents, timesteps, model_pred
                            )
                        else:
                            pred_x0 = model_pred
                        
                        # Decode to image space
                        pred_x0_scaled = pred_x0 / config.scale_factor
                        fake_images = vae.decode(pred_x0_scaled).sample
                    
                    # Discriminator thinks these are real
                    fake_pred = discriminator(fake_images, masks)
                    loss_adv = torch.nn.functional.binary_cross_entropy_with_logits(
                        fake_pred,
                        torch.ones_like(fake_pred)
                    )
                    
                    loss = loss + lambda_adv * loss_adv
                    loss_dict['loss_adv'] = loss_adv.item()
                
                # Segmentation consistency loss (if enabled)
                if use_segmentation and segmentation_checker is not None:
                    with torch.no_grad():
                        # Predict x0 and decode
                        if config.get('prediction_type', 'epsilon') == 'epsilon':
                            pred_x0 = train_diffusion.predict_start_from_noise(
                                noisy_latents, timesteps, model_pred
                            )
                        else:
                            pred_x0 = model_pred
                        
                        pred_x0_scaled = pred_x0 / config.scale_factor
                        fake_images = vae.decode(pred_x0_scaled).sample
                    
                    # Compute segmentation consistency
                    loss_seg = segmentation_checker.compute_consistency_loss(
                        fake_images, 
                        masks
                    )
                    
                    loss = loss + lambda_seg * loss_seg
                    loss_dict['loss_seg'] = loss_seg.item()
                
                # Backward pass for generator
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(base_model.parameters(), config.gradient_clip)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                # Train discriminator (if enabled)
                if use_discriminator and discriminator is not None and optimizer_d is not None:
                    # Real images
                    real_pred = discriminator(images, masks)
                    loss_d_real = torch.nn.functional.binary_cross_entropy_with_logits(
                        real_pred,
                        torch.ones_like(real_pred)
                    )
                    
                    # Fake images (detached)
                    fake_pred = discriminator(fake_images.detach(), masks)
                    loss_d_fake = torch.nn.functional.binary_cross_entropy_with_logits(
                        fake_pred,
                        torch.zeros_like(fake_pred)
                    )
                    
                    loss_d = (loss_d_real + loss_d_fake) * 0.5
                    loss_dict['loss_d'] = loss_d.item()
                    loss_dict['loss_d_real'] = loss_d_real.item()
                    loss_dict['loss_d_fake'] = loss_d_fake.item()
                    
                    # Backward for discriminator
                    accelerator.backward(loss_d)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(discriminator.parameters(), config.gradient_clip)
                    optimizer_d.step()
                    optimizer_d.zero_grad()
            
            # Update EMA model
            if accelerator.sync_gradients:
                ema_update(model_ema, base_model, config.ema_rate)
                global_step += 1
            
            # Logging
            if global_step % config.log_interval == 0:
                loss_dict['loss_total'] = loss.item()
                loss_dict['lr'] = optimizer.param_groups[0]['lr']
                loss_dict['epoch'] = epoch
                loss_dict['step'] = global_step
                
                if accelerator.is_main_process:
                    log_str = f"Epoch [{epoch+1}/{config.num_epochs}] Step [{global_step}/{total_steps}] "
                    log_str += " | ".join([f"{k}: {v:.6f}" for k, v in loss_dict.items()])
                    logger.info(log_str)
                    
                    # Log to tracker (tensorboard/wandb)
                    accelerator.log(loss_dict, step=global_step)
            
            # Save checkpoint
            if global_step % config.save_model_steps == 0 or global_step == total_steps:
                if accelerator.is_main_process:
                    save_path = os.path.join(
                        config.work_dir, 
                        f"epoch_{epoch+1}_step_{global_step}.pth"
                    )
                    
                    save_dict = {
                        'state_dict': accelerator.unwrap_model(base_model).state_dict(),
                        'state_dict_ema': accelerator.unwrap_model(model_ema).state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch + 1,
                        'step': global_step,
                        'config': config,
                    }
                    
                    if discriminator is not None and optimizer_d is not None:
                        save_dict['discriminator'] = accelerator.unwrap_model(discriminator).state_dict()
                        save_dict['optimizer_d'] = optimizer_d.state_dict()
                    
                    torch.save(save_dict, save_path)
                    logger.info(f"Saved checkpoint to {save_path}")
                
                accelerator.wait_for_everyone()
            
            # Validation/sampling (optional)
            if global_step % config.get('eval_steps', 5000) == 0:
                if accelerator.is_main_process:
                    logger.info("Generating validation samples...")
                    base_model.eval()
                    
                    with torch.no_grad():
                        # Generate sample images
                        sample_images = generate_validation_samples(
                            model_ema, 
                            vae, 
                            train_diffusion,
                            batch,
                            config,
                            accelerator.device
                        )
                        
                        # Log images
                        if hasattr(accelerator, 'get_tracker'):
                            for tracker in accelerator.trackers:
                                if tracker.name == "tensorboard":
                                    tracker.writer.add_images(
                                        "validation", 
                                        sample_images, 
                                        global_step
                                    )
                                elif tracker.name == "wandb":
                                    import wandb
                                    tracker.log({
                                        "validation": [wandb.Image(img) for img in sample_images]
                                    })
                    
                    base_model.train()
                
                accelerator.wait_for_everyone()
    
    # Final save
    if accelerator.is_main_process:
        final_path = os.path.join(config.work_dir, "final_model.pth")
        save_dict = {
            'state_dict': accelerator.unwrap_model(base_model).state_dict(),
            'state_dict_ema': accelerator.unwrap_model(model_ema).state_dict(),
            'config': config,
        }
        torch.save(save_dict, final_path)
        logger.info(f"Training complete! Final model saved to {final_path}")
    
    accelerator.end_training()


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
    models = initialize_models()
    train(models)