import os 
import argparse
import datetime
import sys
import time
import types
import warnings
from pathlib import Path
from copy import deepcopy

current_file_path = Path(__file__).resolve()
sys.path.insert(0, str(current_file_path.parent.parent))

import numpy as np
import torch
import torch.nn as nn
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import DistributedType
from diffusers.models import AutoencoderKL
from mmcv.runner import LogBuffer
from torch.utils.data import RandomSampler

from diffusion import IDDPM
from diffusion.data.builder import build_dataset, build_dataloader, set_data_root
from diffusion.model.builder import build_model
from diffusion.utils.checkpoint import save_checkpoint, load_checkpoint
from diffusion.utils.data_sampler import AspectRatioBatchSampler
from diffusion.utils.dist_utils import synchronize, get_world_size, clip_grad_norm_, flush
from diffusion.utils.logger import get_root_logger, rename_file_with_creation_time
from diffusion.utils.lr_scheduler import build_lr_scheduler
from diffusion.utils.misc import set_random_seed, read_config, init_random_seed, DebugUnderflowOverflow
from diffusion.utils.optimizer import build_optimizer, auto_scale_lr

warnings.filterwarnings("ignore")

def _initialize_controlnet_from_base(model):
    """
    Initialize ControlNet weights by copying from the pretrained base model.
    This ensures ControlNet starts with the same understanding as the base model.
    
    Args:
        model: PixArt_UNI_ControlNet instance
    """
    if not hasattr(model, 'controlnet'):
        return
    
    logger = get_root_logger()
    logger.info("Copying base model weights to ControlNet...")
    logger.info(f"ControlNet has {len(model.controlnet.control_blocks)} blocks, "
                f"Base model has {len(model.blocks)} blocks")
    
    with torch.no_grad():
        # Initialize control_x_embedder
        nn.init.xavier_uniform_(model.controlnet.control_x_embedder.proj.weight.view(
            [model.controlnet.control_x_embedder.proj.weight.shape[0], -1]
        ))
        
        # Copy positional embeddings
        model.controlnet.pos_embed.data.copy_(model.pos_embed.data)
        
        # Copy transformer block weights from base model to ControlNet
        # Use the block mapping to determine which base blocks to copy from
        for control_idx, control_block in enumerate(model.controlnet.control_blocks):
            # Find which base block this controlnet block should copy from
            # For even distribution: base_idx = control_idx * (base_depth / control_depth)
            base_depth = len(model.blocks)
            control_depth = len(model.controlnet.control_blocks)
            base_idx = int(control_idx * base_depth / control_depth)
            base_idx = min(base_idx, base_depth - 1)  # Ensure we don't go out of bounds
            
            base_block = model.blocks[base_idx]
            
            logger.info(f"  Copying base block {base_idx} -> control block {control_idx}")
            
            # Copy attention weights
            control_block.attn.qkv.weight.data.copy_(base_block.attn.qkv.weight.data)
            if base_block.attn.qkv.bias is not None:
                control_block.attn.qkv.bias.data.copy_(base_block.attn.qkv.bias.data)
            control_block.attn.proj.weight.data.copy_(base_block.attn.proj.weight.data)
            if base_block.attn.proj.bias is not None:
                control_block.attn.proj.bias.data.copy_(base_block.attn.proj.bias.data)
            
            # Copy Q/K norm if present
            if hasattr(control_block.attn, 'q_norm') and not isinstance(control_block.attn.q_norm, nn.Identity):
                if hasattr(base_block.attn.q_norm, 'weight') and base_block.attn.q_norm.weight is not None:
                    control_block.attn.q_norm.weight.data.copy_(base_block.attn.q_norm.weight.data)
                if hasattr(base_block.attn.q_norm, 'bias') and base_block.attn.q_norm.bias is not None:
                    control_block.attn.q_norm.bias.data.copy_(base_block.attn.q_norm.bias.data)
                if hasattr(base_block.attn.k_norm, 'weight') and base_block.attn.k_norm.weight is not None:
                    control_block.attn.k_norm.weight.data.copy_(base_block.attn.k_norm.weight.data)
                if hasattr(base_block.attn.k_norm, 'bias') and base_block.attn.k_norm.bias is not None:
                    control_block.attn.k_norm.bias.data.copy_(base_block.attn.k_norm.bias.data)
            
            # Copy KV compression if present
            if hasattr(control_block.attn, 'sr') and control_block.attn.sr_ratio > 1:
                control_block.attn.sr.weight.data.copy_(base_block.attn.sr.weight.data)
                control_block.attn.sr.bias.data.copy_(base_block.attn.sr.bias.data)
                if hasattr(control_block.attn, 'norm'):
                    control_block.attn.norm.weight.data.copy_(base_block.attn.norm.weight.data)
                    control_block.attn.norm.bias.data.copy_(base_block.attn.norm.bias.data)
            
            # Copy cross-attention weights
            control_block.cross_attn.q_linear.weight.data.copy_(base_block.cross_attn.q_linear.weight.data)
            if base_block.cross_attn.q_linear.bias is not None:
                control_block.cross_attn.q_linear.bias.data.copy_(base_block.cross_attn.q_linear.bias.data)
            control_block.cross_attn.kv_linear.weight.data.copy_(base_block.cross_attn.kv_linear.weight.data)
            if base_block.cross_attn.kv_linear.bias is not None:
                control_block.cross_attn.kv_linear.bias.data.copy_(base_block.cross_attn.kv_linear.bias.data)
            control_block.cross_attn.proj.weight.data.copy_(base_block.cross_attn.proj.weight.data)
            if base_block.cross_attn.proj.bias is not None:
                control_block.cross_attn.proj.bias.data.copy_(base_block.cross_attn.proj.bias.data)
            
            # Copy MLP weights
            control_block.mlp.fc1.weight.data.copy_(base_block.mlp.fc1.weight.data)
            if base_block.mlp.fc1.bias is not None:
                control_block.mlp.fc1.bias.data.copy_(base_block.mlp.fc1.bias.data)
            control_block.mlp.fc2.weight.data.copy_(base_block.mlp.fc2.weight.data)
            if base_block.mlp.fc2.bias is not None:
                control_block.mlp.fc2.bias.data.copy_(base_block.mlp.fc2.bias.data)
            
            # Copy scale_shift_table
            control_block.scale_shift_table.data.copy_(base_block.scale_shift_table.data)
        
        # Verify zero convs remain zero
        for i, zero_conv in enumerate(model.controlnet.zero_convs):
            if not torch.allclose(zero_conv.weight, torch.zeros_like(zero_conv.weight)):
                logger.info(f"Re-initializing zero conv {i} to zero")
                nn.init.constant_(zero_conv.weight, 0)
                nn.init.constant_(zero_conv.bias, 0)
    
    logger.info("ControlNet initialization complete!")
    logger.info(f"Copied {len(model.controlnet.control_blocks)} transformer blocks from base model to ControlNet")

def _print_trainable_parameters(model, logger):
    """
    Print statistics about frozen vs trainable parameters.
    
    Args:
        model: PixArt_UNI_ControlNet instance
        logger: Logger instance
    """
    total_params = 0
    trainable_params = 0
    frozen_params = 0
    
    param_groups = {
        'base_embedders': 0,
        'base_blocks': 0,
        'base_final': 0,
        'controlnet_embedder': 0,
        'controlnet_blocks': 0,
        'controlnet_zero_convs': 0,
    }
    
    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        
        if param.requires_grad:
            trainable_params += num_params
            
            # Categorize trainable parameters
            if 'controlnet.control_x_embedder' in name:
                param_groups['controlnet_embedder'] += num_params
            elif 'controlnet.control_blocks' in name:
                param_groups['controlnet_blocks'] += num_params
            elif 'controlnet.zero_convs' in name:
                param_groups['controlnet_zero_convs'] += num_params
        else:
            frozen_params += num_params
            
            # Categorize frozen parameters
            if any(x in name for x in ['x_embedder', 't_embedder', 't_block', 'y_embedder', 'pos_embed']):
                param_groups['base_embedders'] += num_params
            elif 'blocks.' in name and 'controlnet' not in name:
                param_groups['base_blocks'] += num_params
            elif 'final_layer' in name:
                param_groups['base_final'] += num_params
    
    logger.info("=" * 80)
    logger.info("Parameter Statistics:")
    logger.info(f"Total Parameters: {total_params:,}")
    logger.info(f"Trainable Parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    logger.info(f"Frozen Parameters: {frozen_params:,} ({100*frozen_params/total_params:.2f}%)")
    logger.info("-" * 80)
    logger.info("Parameter Breakdown:")
    logger.info(f"  Base Embedders (frozen): {param_groups['base_embedders']:,}")
    logger.info(f"  Base Blocks (frozen): {param_groups['base_blocks']:,}")
    logger.info(f"  Base Final Layer (frozen): {param_groups['base_final']:,}")
    logger.info(f"  ControlNet Embedder (trainable): {param_groups['controlnet_embedder']:,}")
    logger.info(f"  ControlNet Blocks (trainable): {param_groups['controlnet_blocks']:,}")
    logger.info(f"  ControlNet Zero Convs (trainable): {param_groups['controlnet_zero_convs']:,}")
    logger.info("=" * 80)


def set_fsdp_env():
    os.environ["ACCELERATE_USE_FSDP"] = 'true'
    os.environ["FSDP_AUTO_WRAP_POLICY"] = 'TRANSFORMER_BASED_WRAP'
    os.environ["FSDP_BACKWARD_PREFETCH"] = 'BACKWARD_PRE'
    os.environ["FSDP_TRANSFORMER_CLS_TO_WRAP"] = 'PixArtBlock'


def ema_update(model_dest: nn.Module, model_src: nn.Module, rate):
    """Update EMA model parameters."""
    param_dict_src = dict(model_src.named_parameters())
    for p_name, p_dest in model_dest.named_parameters():
        p_src = param_dict_src[p_name]
        assert p_src is not p_dest
        p_dest.data.mul_(rate).add_((1 - rate) * p_src.data)


def train():
    if config.get('debug_nan', False):
        DebugUnderflowOverflow(base_model)
        logger.info('NaN debugger registered. Start to detect overflow during training.')
    time_start, last_tic = time.time(), time.time()
    log_buffer = LogBuffer()

    ckpt_time_limit_saved = False

    global_step = start_step + 1

    load_vae_feat = getattr(train_dataloader.dataset, 'load_vae_feat', False)
    use_discriminator = config.get('use_discriminator', False)
    
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
            y = batch[1]  # SSL embeddings (UNI)
            cell_mask = batch[2]  # Cell segmentation mask
            data_info = batch[3]

            # Process cell mask for ControlNet
            # Assuming cell_mask is binary [B, 1, H, W] at image resolution
            # Need to downsample to latent resolution if necessary
            cell_mask_latent = cell_mask
            if cell_mask.shape[-1] != clean_images.shape[-1]:
                # Downsample mask to latent resolution using max pooling to preserve cell presence
                # From 256x256 to 32x32 (256/8 = 32)
                kernel_size = cell_mask.shape[-1] // clean_images.shape[-1]
                cell_mask_latent = nn.functional.max_pool2d(
                    cell_mask.float(), 
                    kernel_size=kernel_size, 
                    stride=kernel_size
                )

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
                    y=y,  # UNI embeddings
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
                                    model=accelerator.unwrap_model(model),
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


def parse_args():
    parser = argparse.ArgumentParser(description="Train ControlNet for PixCell-256")
    parser.add_argument("config", type=str, help="config")
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--resume-from', help='the dir to resume the training')
    parser.add_argument('--local-rank', type=int, default=-1)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument(
        "--report-to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--tracker-project-name",
        type=str,
        default="pixcell_controlnet",
        help="The `project_name` argument passed to Accelerator.init_trackers for more information",
    )
    parser.add_argument(
        "--slurm-time-limit", 
        type=float, 
        default=float('inf'),
        help="slurm time limit in minutes to save checkpoint before job ends"
    )
    parser.add_argument("--loss-report-name", type=str, default="loss")
    parser.add_argument("--skip-step", type=int, default=0, help="number of steps to skip when resuming")
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    config = read_config(args.config)
    if args.work_dir is not None:
        config.work_dir = args.work_dir

    if args.resume_from is not None:
        # if resume from is a dir, we will find the latest checkpoint in the dir
        if os.path.isdir(args.resume_from):
            # checkpoints are in the form of epoch_{epoch}_step_{step}.pth
            checkpoints = [ckpt for ckpt in os.listdir(args.resume_from) if ckpt.endswith('.pth')]
            if len(checkpoints) == 0:
                raise ValueError(f"No checkpoint found in {args.resume_from}")

            checkpoints = sorted(checkpoints, key=lambda x: int(x.split('_')[-1].replace('.pth', '')), reverse=True)
            resume_from = os.path.join(args.resume_from, checkpoints[0])
        else:
            resume_from = args.resume_from

        config.load_from = None
        config.resume_from = dict(
            checkpoint=resume_from,
            load_ema=False,
            resume_optimizer=True,
            resume_lr_scheduler=True)
            
    if args.debug:
        config.log_interval = 1
        config.train_batch_size = 4

    if args.batch_size is not None:
        config.train_batch_size = args.batch_size

    os.umask(0o000)
    os.makedirs(config.work_dir, exist_ok=True)

    init_handler = InitProcessGroupKwargs()
    init_handler.timeout = datetime.timedelta(seconds=5400)
    
    # Initialize accelerator
    if config.use_fsdp:
        init_train = 'FSDP'
        from accelerate import FullyShardedDataParallelPlugin
        from torch.distributed.fsdp.fully_sharded_data_parallel import FullStateDictConfig
        set_fsdp_env()
        fsdp_plugin = FullyShardedDataParallelPlugin(state_dict_config=FullStateDictConfig(offload_to_cpu=False, rank0_only=False),)
    else:
        init_train = 'DDP'
        fsdp_plugin = None

    from accelerate import Accelerator, DataLoaderConfiguration
    dataloader_config = DataLoaderConfiguration(dispatch_batches=True)
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with=args.report_to,
        project_dir=os.path.join(config.work_dir, "logs"),
        fsdp_plugin=fsdp_plugin,
        dataloader_config=dataloader_config,
        kwargs_handlers=[init_handler]
    )

    log_name = 'train_log.log'
    if accelerator.is_main_process:
        if os.path.exists(os.path.join(config.work_dir, log_name)):
            rename_file_with_creation_time(os.path.join(config.work_dir, log_name))
    logger = get_root_logger(os.path.join(config.work_dir, log_name))

    logger.info(accelerator.state)
    config.seed = init_random_seed(config.get('seed', None))
    set_random_seed(config.seed)

    if accelerator.is_main_process:
        config.dump(os.path.join(config.work_dir, 'config.py'))

    logger.info(f"Config: \n{config.pretty_text}")
    logger.info(f"World_size: {get_world_size()}, seed: {config.seed}")
    logger.info(f"Initializing: {init_train} for ControlNet training")
    
    image_size = config.image_size  
    latent_size = int(image_size) // 8
    pred_sigma = getattr(config, 'pred_sigma', True)
    learn_sigma = getattr(config, 'learn_sigma', True) and pred_sigma
    max_length = config.model_max_length
    kv_compress_config = config.kv_compress_config if config.kv_compress else None
    vae = None

    if not config.data.load_vae_feat:
        vae = AutoencoderKL.from_pretrained(config.vae_pretrained, torch_dtype=torch.float16).to(accelerator.device)
        config.scale_factor = vae.config.scaling_factor

    logger.info(f"vae scale factor: {config.scale_factor}")

    model_kwargs = {
        "pe_interpolation": config.pe_interpolation, 
        "config": config,
        "model_max_length": max_length, 
        "qk_norm": config.qk_norm,
        "kv_compress_config": kv_compress_config, 
        "micro_condition": config.micro_condition,
        "add_pos_embed_to_cond": getattr(config, 'add_pos_embed_to_cond', False),
        **config.get('model_kwargs', {})
    }

    # Build models - base model + ControlNet
    train_diffusion = IDDPM(str(config.train_sampling_steps), learn_sigma=learn_sigma, pred_sigma=pred_sigma, snr=config.snr_loss)
    # IMPORTANT: Ensure the transformer directory is in the path BEFORE importing
    model_dir = os.path.join(os.getcwd(), "pretrained_models/pixcell-256/transformer")
    if model_dir not in sys.path:
        sys.path.append(model_dir)
    from pixcell_transformer_2d import PixCellTransformer2DModel

    logger.info("Building PixCell model architecture...")
    # Use config-based building instead of just from_pretrained to ensure ControlNet blocks are created
    from diffusion.model.builder import build_model
    # Build the combined model (base + controlnet)
    base_model = build_model(config.model, **model_kwargs).to(accelerator.device)

    # 2. Load Pretrained Base Model Weights and Initialize ControlNet
    if config.load_from is not None:
        load_path = Path(config.load_from)
        if load_path.is_dir():
            # Automatically find the safetensors file in the directory
            st_files = list(load_path.glob("**/diffusion_pytorch_model.safetensors"))
            load_file = str(st_files[0]) if st_files else str(config.load_from)
        else:
            load_file = str(config.load_from)

        logger.info(f"Loading pretrained base model weights from {load_file}")
        
        # Load pretrained weights into base model components
        missing, unexpect = load_checkpoint(
            load_file,
            base_model,  # This loads into the PixArt_UNI_ControlNet
            load_ema=config.get('load_ema', False),
            max_length=max_length
        )

        logger.info(f"Missing keys: {len(missing)}")
        logger.info(f"Unexpected keys: {len(unexpect)}")

        # Log details if there are concerning missing/unexpected keys
        if missing:
            logger.warning(f"Missing keys (first 10): {missing[:10]}")
        if unexpect:
            logger.warning(f"Unexpected keys (first 10): {unexpect[:10]}")
        
        # 3. Initialize ControlNet by copying base model weights
        logger.info("Initializing ControlNet from base model weights...")
        _initialize_controlnet_from_base(base_model)
        
        # Verify frozen/trainable parameters
        _print_trainable_parameters(base_model, logger)
    if hasattr(base_model, 'controlnet'):
        # Enable gradient checkpointing for ControlNet blocks
        for block in base_model.controlnet.control_blocks:
            block.gradient_checkpointing = True
    logger.info(f"{base_model.__class__.__name__} Model Parameters: {sum(p.numel() for p in base_model.parameters()):,}")
    logger.info(f"Trainable Parameters: {sum(p.numel() for p in base_model.parameters() if p.requires_grad):,}")

    # Create EMA model
    # 1. Re-build the same architecture for EMA
    logger.info("Initializing EMA model architecture...")
    model_ema = build_model(config.model, **model_kwargs).to(accelerator.device)

    # 2. Copy the current weights from the base_model to the EMA model
    model_ema.load_state_dict(base_model.state_dict())

    # 3. Set to eval mode and freeze parameters
    model_ema.eval()
    for param in model_ema.parameters():
        param.requires_grad = False

    logger.info("✓ EMA model initialized via state_dict copy.")
    for param in model_ema.parameters():
        param.requires_grad = False

    # Build discriminator if enabled
    discriminator = None
    optimizer_d = None
    if config.get('use_discriminator', False):
        logger.info("Building discriminator for adversarial training...")
        
        from discriminator import build_discriminator
        discriminator = build_discriminator(config.discriminator).train()
        
        logger.info(f"Discriminator Parameters: {sum(p.numel() for p in discriminator.parameters()):,}")
        
        # Build discriminator optimizer
        disc_optimizer_config = config.get('discriminator_optimizer', config.optimizer)
        optimizer_d = build_optimizer(discriminator, disc_optimizer_config)
    
    # Build segmentation consistency checker if enabled
    segmentation_checker = None
    if config.get('use_segmentation_consistency', False):
        logger.info("Building cell segmentation consistency checker with Cellpose...")
        
        from cell_segmentation_consistency import CellSegmentationConsistency
        segmentation_checker = CellSegmentationConsistency(
            model_type=config.get('cellpose_model_type', 'cyto2'),
            device=accelerator.device,
            image_size=image_size,
            use_gpu=True,
            diameter=config.get('cell_diameter', 30),
        )
        
        if segmentation_checker.model is not None:
            if hasattr(segmentation_checker.model, 'eval'):
                logger.info(f"✓ Using Cellpose model: {config.get('cellpose_model_type', 'cyto2')}")
            else:
                logger.info(f"Segmentation Model Parameters: {sum(p.numel() for p in segmentation_checker.model.parameters()):,}")
                # Freeze segmentation model
                for param in segmentation_checker.model.parameters():
                    param.requires_grad = False
                logger.info("✓ Using lightweight U-Net segmentation model (frozen)")


    # Initialize EMA with current model state
    ema_update(model_ema, base_model, 0.)

    # Prepare for FSDP clip grad norm calculation
    if accelerator.distributed_type == DistributedType.FSDP:
        for m in accelerator._models:
            m.clip_grad_norm_ = types.MethodType(clip_grad_norm_, m)

    # Build dataloader - use ControlNet-specific dataset that includes masks
    set_data_root(config.data_root)
    dataset = build_dataset(
        config.data, 
        resolution=image_size, 
        aspect_ratio_type=config.aspect_ratio_type,
        real_prompt_ratio=config.real_prompt_ratio, 
        max_length=max_length, 
        config=config,
    )
    train_dataloader = build_dataloader(
        dataset, 
        num_workers=config.num_workers, 
        batch_size=config.train_batch_size, 
        shuffle=True
    )


    # Build optimizer - only optimize ControlNet parameters if specified
    lr_scale_ratio = 1
    if config.get('auto_lr', None):
        lr_scale_ratio = auto_scale_lr(
            config.train_batch_size * get_world_size() * config.gradient_accumulation_steps,
            config.optimizer, 
            **config.auto_lr
        )
    
    # Optimizer setup
    optimizer = build_optimizer(base_model, config.optimizer)
    lr_scheduler = build_lr_scheduler(config, optimizer, train_dataloader, lr_scale_ratio)

    timestamp = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())

    if accelerator.is_main_process:
        tracker_config = dict(vars(config))
        try:
            accelerator.init_trackers(args.tracker_project_name, tracker_config)
        except:
            accelerator.init_trackers(f"tb_{timestamp}")

    start_epoch = 0
    start_step = 0
    skip_step = args.skip_step or config.skip_step
    total_steps = len(train_dataloader) * config.num_epochs

    if config.resume_from is not None and config.resume_from['checkpoint'] is not None:
        resume_path = config.resume_from['checkpoint']
        path = os.path.basename(resume_path)
        start_epoch = int(path.replace('.pth', '').split("_")[1]) - 1
        start_step = int(path.replace('.pth', '').split("_")[3])
        _, missing, unexpected = load_checkpoint(
            **config.resume_from,
            model=base_model,
            model_ema=model_ema,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            max_length=max_length,
        )

        logger.warning(f'Missing keys: {missing}')
        logger.warning(f'Unexpected keys: {unexpected}')
        
    # Prepare everything
    base_model, model_ema = accelerator.prepare(base_model, model_ema)
    
    if discriminator is not None:
        discriminator = accelerator.prepare(discriminator)
        optimizer_d = accelerator.prepare(optimizer_d)
    
    optimizer, train_dataloader, lr_scheduler = accelerator.prepare(optimizer, train_dataloader, lr_scheduler)
    
    logger.info("Starting ControlNet training...")
    if discriminator is not None:
        logger.info("Adversarial training enabled with discriminator")
    
    train()
