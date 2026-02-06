import os
import argparse
import datetime
import time
import types
import warnings
from pathlib import Path
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import DistributedType
from diffusers.models import AutoencoderKL
from mmcv.runner import LogBuffer

from diffusion import IDDPM
from diffusion.data.builder import build_dataset, build_dataloader, set_data_root
from diffusion.model.builder import build_model
from diffusion.utils.checkpoint import save_checkpoint, load_checkpoint
from diffusion.utils.dist_utils import synchronize, get_world_size, clip_grad_norm_, flush
from diffusion.utils.logger import get_root_logger, rename_file_with_creation_time
from diffusion.utils.lr_scheduler import build_lr_scheduler
from diffusion.utils.misc import set_random_seed, read_config, init_random_seed, DebugUnderflowOverflow
from diffusion.utils.optimizer import build_optimizer, auto_scale_lr

warnings.filterwarnings("ignore")


def set_fsdp_env():
    """Set environment variables for FSDP training."""
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


def _find_checkpoint(resume_dir):
    """Find the latest checkpoint in a directory."""
    if os.path.isfile(resume_dir):
        return resume_dir
    
    checkpoints = [ckpt for ckpt in os.listdir(resume_dir) if ckpt.endswith('.pth')]
    if len(checkpoints) == 0:
        raise ValueError(f"No checkpoint found in {resume_dir}")
    
    checkpoints = sorted(
        checkpoints, 
        key=lambda x: int(x.split('_')[-1].replace('.pth', '')), 
        reverse=True
    )
    return os.path.join(resume_dir, checkpoints[0])


def _setup_accelerator(config, args):
    """Initialize accelerator with appropriate settings."""
    init_handler = InitProcessGroupKwargs()
    init_handler.timeout = datetime.timedelta(seconds=5400)
    
    fsdp_plugin = None
    if config.use_fsdp:
        from accelerate import FullyShardedDataParallelPlugin
        from torch.distributed.fsdp.fully_sharded_data_parallel import FullStateDictConfig
        set_fsdp_env()
        fsdp_plugin = FullyShardedDataParallelPlugin(
            state_dict_config=FullStateDictConfig(offload_to_cpu=False, rank0_only=False)
        )
    
    from accelerate import DataLoaderConfiguration
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
    
    return accelerator


def _resume_from_checkpoint(config, model, model_ema, optimizer, lr_scheduler, max_length, logger):
    """Resume training from a checkpoint."""
    resume_path = config.resume_from['checkpoint']
    path = os.path.basename(resume_path)
    start_epoch = int(path.replace('.pth', '').split("_")[1]) - 1
    start_step = int(path.replace('.pth', '').split("_")[3])
    
    _, missing, unexpected = load_checkpoint(
        **config.resume_from,
        model=model,
        model_ema=model_ema,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        max_length=max_length,
    )
    
    logger.warning(f'Missing keys: {missing}')
    logger.warning(f'Unexpected keys: {unexpected}')
    
    return start_epoch, start_step


def parse_args(args_list=None):
    """
    Parse command-line arguments.
    
    Args:
        args_list: List of arguments (for programmatic use) or None (for CLI use)
    """
    parser = argparse.ArgumentParser(description="Train ControlNet for PixCell-256")
    parser.add_argument("config", type=str, help="config")
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--resume-from', help='the dir to resume the training')
    parser.add_argument('--load-from', help='the checkpoint to load from')
    parser.add_argument('--local-rank', type=int, default=-1)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument("--report-to", type=str, default="tensorboard")
    parser.add_argument("--tracker-project-name", type=str, default="pixcell_controlnet")
    parser.add_argument("--slurm-time-limit", type=float, default=float('inf'))
    parser.add_argument("--loss-report-name", type=str, default="loss")
    parser.add_argument("--skip-step", type=int, default=0)
    
    return parser.parse_args(args_list)


def initialize_config_and_accelerator(args_list=None):
    """
    Parse arguments, read config, and initialize accelerator.
    
    Returns:
        dict with: config, accelerator, logger, args
    """
    args = parse_args(args_list)
    config = read_config(args.config)
    
    # Apply command-line overrides
    if args.work_dir is not None:
        config.work_dir = args.work_dir
    
    if args.resume_from is not None:
        resume_from = _find_checkpoint(args.resume_from)
        config.load_from = None
        config.resume_from = dict(
            checkpoint=resume_from,
            load_ema=True,
            resume_optimizer=True,
            resume_lr_scheduler=True
        )
    
    if args.debug:
        config.log_interval = 1
        config.train_batch_size = 4
    
    if args.batch_size is not None:
        config.train_batch_size = args.batch_size
    
    # Setup workspace
    os.umask(0o000)
    os.makedirs(config.work_dir, exist_ok=True)
    
    # Initialize accelerator
    accelerator = _setup_accelerator(config, args)
    
    # Setup logging
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
    
    return {
        'config': config,
        'accelerator': accelerator,
        'logger': logger,
        'args': args,
    }


def initialize_models(config, accelerator, logger):
    """
    Initialize all models: base model, EMA model, VAE, and diffusion.
    
    Returns:
        dict with: base_model, model_ema, vae, train_diffusion
    """
    image_size = config.image_size
    latent_size = int(image_size) // 8
    max_length = config.model_max_length
    
    # VAE setup
    vae = None
    if not config.data.load_vae_feat:
        vae = AutoencoderKL.from_pretrained(
            config.vae_pretrained, 
            torch_dtype=torch.float16
        ).to(accelerator.device)
        config.scale_factor = vae.config.scaling_factor
    
    logger.info(f"VAE scale factor: {config.scale_factor}")
    
    # Build model kwargs
    pred_sigma = getattr(config, 'pred_sigma', True)
    learn_sigma = getattr(config, 'learn_sigma', True) and pred_sigma
    kv_compress_config = config.kv_compress_config if config.kv_compress else None
    
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
    
    # Build diffusion
    train_diffusion = IDDPM(
        str(config.train_sampling_steps),
        learn_sigma=learn_sigma,
        pred_sigma=pred_sigma,
        snr=config.snr_loss
    )
    
    # Build base model
    base_model = build_model(
        config.model,
        config.grad_checkpointing,
        config.get('fp32_attention', False),
        input_size=latent_size,
        learn_sigma=learn_sigma,
        pred_sigma=pred_sigma,
        **model_kwargs
    ).train()
    
    logger.info(
        f"{base_model.__class__.__name__} Model Parameters: "
        f"{sum(p.numel() for p in base_model.parameters()):,}"
    )
    
    # Load checkpoint if specified
    if config.load_from is not None:
        missing, unexpected = load_checkpoint(
            config.load_from,
            base_model,
            load_ema=config.get('load_ema', False),
            max_length=max_length,
            ignore_keys=config.get('ignore_keys', [])
        )
        logger.warning(f'Missing keys: {missing}')
        logger.warning(f'Unexpected keys: {unexpected}')
    
    # Create EMA model
    model_ema = deepcopy(base_model).eval()
    ema_update(model_ema, base_model, 0.)
    
    return {
        'base_model': base_model,
        'model_ema': model_ema,
        'vae': vae,
        'train_diffusion': train_diffusion,
    }


def initialize_dataset_and_optimizer(config, accelerator, logger, base_model, discriminator=None):
    """
    Initialize dataset, dataloader, optimizer, and lr_scheduler.
    
    Returns:
        dict with: train_dataloader, optimizer, optimizer_d, lr_scheduler
    """
    image_size = config.image_size
    max_length = config.model_max_length
    
    # Build dataset
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
    
    # Auto-scale learning rate if needed
    lr_scale_ratio = 1
    if config.get('auto_lr', None):
        lr_scale_ratio = auto_scale_lr(
            config.train_batch_size * get_world_size() * config.gradient_accumulation_steps,
            config.optimizer,
            **config.auto_lr
        )
    
    # Build optimizer for base model
    optimizer = build_optimizer(base_model, config.optimizer)
    lr_scheduler = build_lr_scheduler(config, optimizer, train_dataloader, lr_scale_ratio)
    
    # Build discriminator optimizer if needed
    optimizer_d = None
    if discriminator is not None:
        disc_optimizer_config = config.get('discriminator_optimizer', config.optimizer)
        optimizer_d = build_optimizer(discriminator, disc_optimizer_config)
    
    return {
        'train_dataloader': train_dataloader,
        'optimizer': optimizer,
        'optimizer_d': optimizer_d,
        'lr_scheduler': lr_scheduler,
    }


def setup_training_state(config, accelerator, logger, args, train_dataloader,
                         base_model, model_ema, optimizer, lr_scheduler):
    """
    Setup tracking, resume from checkpoint if needed, prepare with accelerator.
    
    Returns:
        dict with: start_epoch, start_step, skip_step, total_steps
    """
    # Initialize tracking
    timestamp = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
    if accelerator.is_main_process:
        tracker_config = dict(vars(config))
        try:
            accelerator.init_trackers(args.tracker_project_name, tracker_config)
        except:
            accelerator.init_trackers(f"tb_{timestamp}")
    
    # Setup training state
    start_epoch = 0
    start_step = 0
    skip_step = args.skip_step or config.skip_step
    total_steps = len(train_dataloader) * config.num_epochs
    
    # Resume from checkpoint if specified
    if config.resume_from is not None and config.resume_from['checkpoint'] is not None:
        start_epoch, start_step = _resume_from_checkpoint(
            config, base_model, model_ema, optimizer, lr_scheduler,
            config.model_max_length, logger
        )
    
    # Prepare for FSDP clip grad norm
    if accelerator.distributed_type == DistributedType.FSDP:
        for m in accelerator._models:
            m.clip_grad_norm_ = types.MethodType(clip_grad_norm_, m)
    
    logger.info("Starting ControlNet training...")
    
    return {
        'start_epoch': start_epoch,
        'start_step': start_step,
        'skip_step': skip_step,
        'total_steps': total_steps,
    }


def train(models_dict):
    """
    Main training loop for PixCell ControlNet.
    
    Args:
        models_dict: Dictionary containing all models and training components
    """
    # Unpack models and components
    base_model = models_dict['base_model']
    model_ema = models_dict['model_ema']
    vae = models_dict['vae']
    train_diffusion = models_dict['train_diffusion']
    optimizer = models_dict['optimizer']
    optimizer_d = models_dict.get('optimizer_d', None)
    lr_scheduler = models_dict['lr_scheduler']
    train_dataloader = models_dict['train_dataloader']
    accelerator = models_dict['accelerator']
    config = models_dict['config']
    logger = models_dict['logger']
    args = models_dict['args']
    
    start_epoch = models_dict['start_epoch']
    start_step = models_dict['start_step']
    skip_step = models_dict['skip_step']
    total_steps = models_dict['total_steps']
    
    # Setup debugging if needed
    if config.get('debug_nan', False):
        DebugUnderflowOverflow(base_model)
        logger.info('NaN debugger registered. Start to detect overflow during training.')
    
    # Training state
    time_start, last_tic = time.time(), time.time()
    log_buffer = LogBuffer()
    ckpt_time_limit_saved = False
    global_step = start_step + 1
    
    load_vae_feat = getattr(train_dataloader.dataset, 'load_vae_feat', False)
    
    # Main training loop
    for epoch in range(start_epoch + 1, config.num_epochs + 1):
        data_time_start = time.time()
        data_time_all = 0
        
        for step, batch in enumerate(train_dataloader):
            # Skip steps if resuming
            if step < skip_step:
                if (step + 1) % 50 == 0 and accelerator.is_main_process:
                    info = f"Skipping Step/Epoch [{global_step}/{epoch}][{step + 1}/{len(train_dataloader)}]"
                    logger.info(info)
                continue
            
            # Encode images to latent space
            if load_vae_feat:
                z = batch[0]
            else:
                with torch.no_grad():
                    with torch.cuda.amp.autocast(
                        enabled=(config.mixed_precision in ['fp16', 'bf16'])
                    ):
                        posterior = vae.encode(batch[0]).latent_dist
                        if config.sample_posterior:
                            z = posterior.sample()
                        else:
                            z = posterior.mode()
            
            # Apply shift and scale
            if hasattr(config, 'shift_factor'):
                z = z - config.shift_factor
            clean_images = z * config.scale_factor
            
            # Unpack batch
            y = batch[1]
            control_input = batch[2]
            data_info = batch[3]
            
            
            # Sample timesteps
            bs = clean_images.shape[0]
            timesteps = torch.randint(
                0, config.train_sampling_steps, (bs,), device=clean_images.device
            ).long()
            
            grad_norm = None
            data_time_all += time.time() - data_time_start
            
            # Training step
            with accelerator.accumulate(base_model):
                optimizer.zero_grad()
                
                # Forward pass
                model_kwargs = dict(
                    y=y,
                    mask=None,
                    data_info=data_info,
                    control_input=control_input
                )
                loss_term = train_diffusion.training_losses(
                    base_model, clean_images, timesteps, model_kwargs=model_kwargs
                )
                loss = loss_term['loss'].mean()
                
                # Backward pass
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(
                        base_model.parameters(), config.gradient_clip
                    )
                
                optimizer.step()
                lr_scheduler.step()
                
                # Update EMA
                if accelerator.sync_gradients:
                    ema_update(model_ema, base_model, config.ema_rate)
            
            # Logging
            lr = lr_scheduler.get_last_lr()[0]
            logs = {args.loss_report_name: accelerator.gather(loss).mean().item()}
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
                
                info = (
                    f"Step/Epoch [{global_step}/{epoch}][{step + 1}/{len(train_dataloader)}]: "
                    f"total_eta: {eta}, epoch_eta:{eta_epoch}, time_all:{t:.3f}, "
                    f"time_data:{t_d:.3f}, lr:{lr:.3e}, "
                )
                
                if hasattr(base_model, 'module'):
                    info += f's:({base_model.module.h}, {base_model.module.w}), '
                else:
                    info += f's:({base_model.h}, {base_model.w}), '
                
                info += ', '.join([f"{k}:{v:.4f}" for k, v in log_buffer.output.items()])
                logger.info(info)
                
                last_tic = time.time()
                log_buffer.clear()
                data_time_all = 0
            
            logs.update(lr=lr)
            accelerator.log(logs, step=global_step)
            
            global_step += 1
            data_time_start = time.time()
            
            # Checkpoint saving
            def save_checkpoint_fn():
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    os.umask(0o000)
                    save_checkpoint(
                        os.path.join(config.work_dir, 'checkpoints'),
                        epoch=epoch,
                        step=global_step,
                        model=accelerator.unwrap_model(base_model),
                        model_ema=accelerator.unwrap_model(model_ema),
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler
                    )
            
            # Check time limit
            time_elapsed_minutes = (time.time() - time_start) / 60
            time_to_save = (time_elapsed_minutes >= args.slurm_time_limit) and (not ckpt_time_limit_saved)
            
            if (config.save_model_steps and global_step % config.save_model_steps == 0) or time_to_save:
                if not ckpt_time_limit_saved:
                    ckpt_time_limit_saved = True
                save_checkpoint_fn()
        
        # Epoch checkpoint
        if epoch % config.save_model_epochs == 0 or epoch == config.num_epochs:
            save_checkpoint_fn()
        
        accelerator.wait_for_everyone()


def main():
    """Main entry point for training."""
    # Initialize config and accelerator
    init_data = initialize_config_and_accelerator()
    config = init_data['config']
    accelerator = init_data['accelerator']
    logger = init_data['logger']
    args = init_data['args']
    
    # Initialize models
    model_data = initialize_models(config, accelerator, logger)
    base_model = model_data['base_model']
    model_ema = model_data['model_ema']
    vae = model_data['vae']
    train_diffusion = model_data['train_diffusion']
    
    # Initialize dataset and optimizer
    discriminator = None  # Add discriminator initialization if needed
    optim_data = initialize_dataset_and_optimizer(
        config, accelerator, logger, base_model, discriminator
    )
    train_dataloader = optim_data['train_dataloader']
    optimizer = optim_data['optimizer']
    optimizer_d = optim_data['optimizer_d']
    lr_scheduler = optim_data['lr_scheduler']
    
    # Prepare with accelerator (CRITICAL!)
    base_model = accelerator.prepare(base_model)
    model_ema = accelerator.prepare(model_ema)
    optimizer = accelerator.prepare(optimizer)
    train_dataloader = accelerator.prepare(train_dataloader)
    lr_scheduler = accelerator.prepare(lr_scheduler)
    
    if discriminator is not None:
        discriminator = accelerator.prepare(discriminator)
        optimizer_d = accelerator.prepare(optimizer_d)
    
    # Setup training state
    state_data = setup_training_state(
        config, accelerator, logger, args, train_dataloader,
        base_model, model_ema, optimizer, lr_scheduler
    )
    
    # Prepare models dict for training
    models = {
        'base_model': base_model,
        'model_ema': model_ema,
        'vae': vae,
        'train_diffusion': train_diffusion,
        'optimizer': optimizer,
        'optimizer_d': optimizer_d,
        'lr_scheduler': lr_scheduler,
        'train_dataloader': train_dataloader,
        'accelerator': accelerator,
        'config': config,
        'logger': logger,
        'args': args,
        **state_data
    }
    
    # Start training
    train(models)


if __name__ == "__main__":
    main()