# %%
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
from diffusion.utils.checkpoint import save_checkpoint, load_checkpoint, save_checkpoint_controlnet
from diffusion.utils.dist_utils import synchronize, get_world_size, clip_grad_norm_, flush
from diffusion.utils.logger import get_root_logger, rename_file_with_creation_time
from diffusion.utils.lr_scheduler import build_lr_scheduler
from diffusion.utils.misc import set_random_seed, read_config, init_random_seed, DebugUnderflowOverflow
from diffusion.utils.optimizer import build_optimizer, auto_scale_lr
import matplotlib.pyplot as plt
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


def _resume_from_checkpoint(config, model, controlnet, model_ema, optimizer, lr_scheduler, max_length, logger):
    """Resume training from a checkpoint."""
    resume_path = config.resume_from['checkpoint']
    path = os.path.basename(resume_path)
    start_epoch = int(path.replace('.pth', '').split("_")[2]) - 1
    start_step = int(path.replace('.pth', '').split("_")[4])
    
    _, missing, unexpected = load_checkpoint(
        **config.resume_from,
        model=model,
        controlnet=controlnet,
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
    Initialize all models for ControlNet training.
    
    Key changes from original:
    1. Loads FROZEN base transformer (pixcell_controlnet_transformer)
    2. Creates TRAINABLE ControlNet model
    3. Copies weights from base to ControlNet's transformer blocks
    
    Returns:
        dict with: 
            - base_model: FROZEN PixCellTransformer (for inference)
            - controlnet: TRAINABLE PixCellControlNet
            - model_ema: EMA version of ControlNet
            - vae: VAE for encoding
            - train_diffusion: Diffusion scheduler
    """
    image_size = config.image_size
    latent_size = int(image_size) // 8
    max_length = config.model_max_length
    
    # ===================================================================
    # 1. VAE setup (unchanged)
    # ===================================================================
    vae = AutoencoderKL.from_pretrained(
        config.vae_pretrained, 
        torch_dtype=torch.float16
    ).to(accelerator.device)
    config.scale_factor = vae.config.scaling_factor
    
    logger.info(f"VAE scale factor: {config.scale_factor}")
    
    # ===================================================================
    # 2. Build diffusion (unchanged)
    # ===================================================================
    pred_sigma = getattr(config, 'pred_sigma', True)
    learn_sigma = getattr(config, 'learn_sigma', True) and pred_sigma
    
    train_diffusion = IDDPM(
        str(config.train_sampling_steps),
        learn_sigma=learn_sigma,
        pred_sigma=pred_sigma,
        snr=config.snr_loss
    )
    
    # ===================================================================
    # 3. Load FROZEN base transformer model
    # ===================================================================
    logger.info("Loading frozen base transformer model...")
    
    # Base model kwargs
    kv_compress_config = config.kv_compress_config if config.kv_compress else None
    base_model_kwargs = {
        "pe_interpolation": config.pe_interpolation,
        "config": config,
        "model_max_length": max_length,
        "qk_norm": config.qk_norm,
        "kv_compress_config": kv_compress_config,
        "micro_condition": config.micro_condition,
        "add_pos_embed_to_cond": getattr(config, 'add_pos_embed_to_cond', False),
        **config.get('base_model_kwargs', {})
    }
    
    # Build base transformer (will be frozen)
    base_model = build_model(
        config.base_model,  # e.g., 'PixCell_Transformer_XL_2_UNI'
        False,  # No grad checkpointing for frozen model
        config.get('fp32_attention', False),
        input_size=latent_size,
        learn_sigma=learn_sigma,
        pred_sigma=pred_sigma,
        **base_model_kwargs
    )
    
    # Load pretrained weights for base model
    if config.base_model_path is None:
        raise ValueError("config.base_model_path must be specified for ControlNet training!")
    
    load_file = _find_model_file(config.base_model_path)
    missing, unexpected = load_checkpoint(
        load_file,
        model=base_model,
        load_ema=config.get('load_base_ema', False),
        max_length=max_length,
        ignore_keys=config.get('base_ignore_keys', [])
    )
    logger.warning(f'Base model - Missing keys: {missing}')
    logger.warning(f'Base model - Unexpected keys: {unexpected}')
    
    # FREEZE base model
    base_model.eval()
    for param in base_model.parameters():
        param.requires_grad = False
    
    logger.info(
        f"Frozen Base Model ({base_model.__class__.__name__}): "
        f"{sum(p.numel() for p in base_model.parameters()):,} parameters (all frozen)"
    )
    
    # ===================================================================
    # 4. Build TRAINABLE ControlNet
    # ===================================================================
    logger.info("Building trainable ControlNet model...")
    
    # ControlNet kwargs
    controlnet_model_kwargs = {
        "pe_interpolation": config.pe_interpolation,
        "config": config,
        "model_max_length": max_length,
        "qk_norm": config.qk_norm,
        "kv_compress_config": kv_compress_config,
        "conditioning_channels": config.controlnet_conditioning_channels,  # e.g., 16 for cell masks
        "n_controlnet_blocks": getattr(config, 'n_controlnet_blocks', None),
        **config.get('controlnet_model_kwargs', {})
    }
    
    # Build ControlNet (all parameters trainable)
    controlnet = build_model(
        config.controlnet_model,  # e.g., 'PixCell_ControlNet_XL_2_UNI'
        config.grad_checkpointing,
        config.get('fp32_attention', False),
        input_size=latent_size,
        learn_sigma=False,  # ControlNet doesn't predict sigma
        pred_sigma=False,
        **controlnet_model_kwargs
    ).train()
    
    logger.info(
        f"ControlNet Model ({controlnet.__class__.__name__}): "
        f"{sum(p.numel() for p in controlnet.parameters()):,} total parameters"
    )
    logger.info(
        f"  Trainable: {sum(p.numel() for p in controlnet.parameters() if p.requires_grad):,}"
    )
    
    # ===================================================================
    # 5. Copy weights from base model to ControlNet's transformer blocks
    # ===================================================================
    logger.info("Copying transformer block weights from base model to ControlNet...")
    
    # Get base model state dict
    base_state_dict = base_model.state_dict()
    
    # Copy matching weights
    copied_count = 0
    copied_params_count = 0
    for name, param in controlnet.named_parameters():
        # Try to find corresponding parameter in base model
        # Adjust these mappings based on your actual model structure
        possible_base_keys = [
            name.replace('blocks.', 'transformer_blocks.'),  # If ControlNet uses 'blocks'
            name,  # Direct match
        ]
        
        for base_key in possible_base_keys:
            if base_key in base_state_dict:
                if param.shape == base_state_dict[base_key].shape:
                    # Only copy to non-ControlNet-specific layers
                    if 'controlnet_blocks' not in name and 'cond_embedder' not in name:
                        param.data.copy_(base_state_dict[base_key])
                        copied_count += 1
                        copied_params_count += param.numel()
                        break
    
    logger.info(f"✓ Copied {copied_count} parameters from base model to ControlNet")
    logger.info(f"✓ Copied {copied_params_count} parameters from base model to ControlNet")


    # frozen controlnet except for controlnet_blocks
    #for name, param in controlnet.named_parameters():
    #    if 'controlnet_blocks' not in name:
    #        param.requires_grad = False
    # print trainable parameters
    logger.info(f"Trainable parameters: {sum(p.numel() for p in controlnet.parameters() if p.requires_grad):,}")
    # Optional: Load ControlNet checkpoint if resuming training
    if config.get('controlnet_load_from', None) is not None:
        logger.info(f"Loading ControlNet checkpoint from {config.controlnet_load_from}")
        load_file = _find_model_file(config.controlnet_load_from)
        missing, unexpected = load_checkpoint(
            load_file,
            controlnet=controlnet,
            load_ema=False,
            max_length=max_length,
            ignore_keys=config.get('controlnet_ignore_keys', [])
        )
        logger.warning(f'ControlNet - Missing keys: {missing}')
        logger.warning(f'ControlNet - Unexpected keys: {unexpected}')
    
    # ===================================================================
    # 6. Create EMA model for ControlNet
    # ===================================================================
    logger.info("Creating EMA model for ControlNet...")
    model_ema = deepcopy(controlnet).eval()
    ema_update(model_ema, controlnet, 0.)
    
    logger.info("="*80)
    logger.info("Model initialization complete!")
    logger.info(f"  Base Model (frozen): {sum(p.numel() for p in base_model.parameters()):,} params")
    logger.info(f"  ControlNet (trainable): {sum(p.numel() for p in controlnet.parameters() if p.requires_grad):,} params")
    logger.info("="*80)
    
    return {
        'base_model': base_model,  # Frozen transformer
        'controlnet': controlnet,  # Trainable ControlNet
        'model_ema': model_ema,    # EMA of ControlNet
        'vae': vae,
        'train_diffusion': train_diffusion,
    }

def _find_model_file(load_from):
    """Find safetensors file in directory or return path"""
    load_path = Path(load_from)
    if load_path.is_dir():
        st_files = list(load_path.glob("**/diffusion_pytorch_model.safetensors"))
        return str(st_files[0]) if st_files else str(load_from)
    return str(load_from)


def initialize_dataset_and_optimizer(config, accelerator, logger, model, discriminator=None):
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
        shuffle=True,
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
    optimizer = build_optimizer(model, config.optimizer)
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
                         base_model, controlnet, model_ema, optimizer, lr_scheduler):
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
            config, base_model, controlnet, model_ema, optimizer, lr_scheduler,
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


def train_controlnet(models_dict):
    """
    Main training loop for PixCell ControlNet.
    
    Key differences from standard training:
    1. base_model is FROZEN (only used for inference)
    2. controlnet is TRAINABLE (optimized)
    3. Forward pass: controlnet -> base_model (with controlnet outputs)
    
    Args:
        models_dict: Dictionary containing all models and training components
    """
    # Unpack models and components
    base_model = models_dict['base_model']  # FROZEN transformer
    controlnet = models_dict['controlnet']  # TRAINABLE ControlNet
    model_ema = models_dict['model_ema']    # EMA of ControlNet
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
        DebugUnderflowOverflow(controlnet)  # Debug ControlNet, not base_model
        logger.info('NaN debugger registered for ControlNet. Start to detect overflow during training.')
    
    # Ensure base model is frozen
    base_model.eval()
    for param in base_model.parameters():
        param.requires_grad = False
    
    # Ensure controlnet is trainable
    controlnet.train()
    for param in controlnet.parameters():
        param.requires_grad = True
    
    # Training state
    time_start, last_tic = time.time(), time.time()
    log_buffer = LogBuffer()
    ckpt_time_limit_saved = False
    global_step = start_step + 1
    
    load_vae_feat = getattr(train_dataloader.dataset, 'load_vae_feat', False)
    print(f"load_vae_feat: {load_vae_feat}")
    # Main training loop
    logger.info("="*80)
    logger.info("Starting ControlNet Training")
    # Print current training state
    logger.info(f"start_epoch: {start_epoch}, start_step: {start_step}, total_steps: {total_steps}")
    logger.info("="*80)

    for epoch in range(start_epoch + 1, config.num_epochs + 1):
        data_time_start = time.time()
        data_time_all = 0
        
        for step, batch in enumerate(train_dataloader):
            # Skip steps if resuming
            #print(f"Step: {step}")
            if step < skip_step:
                if (step + 1) % 50 == 0 and accelerator.is_main_process:
                    info = f"Skipping Step/Epoch [{global_step}/{epoch}][{step + 1}/{len(train_dataloader)}]"
                    logger.info(info)
                continue
            
            # ============================================================
            # 1. Encode images to latent space
            # ============================================================
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
            clean_images = (z.float() - config.shift_factor) * config.scale_factor
            # ============================================================
            # 2. Unpack batch (includes control_input for ControlNet)
            # ============================================================
            y = batch[1]                # UNI embeddings
            control_input = batch[2]    # Cell masks for ControlNet
            vae_mask = batch[3]    # VAE masks for ControlNet
            data_info = batch[4]
            import matplotlib.pyplot as plt
            if 0:
                if 1:
                    print(f"control_input.shape: {control_input.shape}")
                    print(f"control_input.min(): {control_input.min()}")
                    print(f"control_input.max(): {control_input.max()}")
                from PIL import Image
                control_inputs = []
                for i in range(control_input.shape[0]):
                    plt_image = (control_input[i][0].cpu().numpy()*255.).clip(0, 255).astype(np.uint8)
                    control_inputs.append(Image.fromarray(plt_image))
                from torchvision import transforms as T

                transform = T.Compose(
                    [
                        T.Lambda(lambda img: img.convert("RGB")),
                        T.Resize((256, 256)),
                        T.ToTensor(),
                        T.Normalize([0.5], [0.5]),
                    ]
                )

                controlnet_input_torch = []
                for i in range(len(control_inputs)):
                    controlnet_input_torch.append(transform(control_inputs[i]).unsqueeze(0).to(accelerator.device).to(vae.dtype))
                controlnet_input_torch = torch.cat(controlnet_input_torch, dim=0)
                print(f"controlnet_input_torch.shape: {controlnet_input_torch.shape}")
                print(f"controlnet_input_torch.min(): {controlnet_input_torch.min()}")
                print(f"controlnet_input_torch.max(): {controlnet_input_torch.max()}")
                print(f"controlnet_input_torch.mean(): {controlnet_input_torch.mean()}")
                print(f"controlnet_input_torch.std(): {controlnet_input_torch.std()}")

                vae_scale = config.scale_factor
                vae_shift = config.shift_factor
                controlnet_input_latent = vae.encode(controlnet_input_torch).latent_dist.mean
                controlnet_input_latent = (controlnet_input_latent-vae_shift)*vae_scale
                print(f"controlnet_input_latent.shape: {controlnet_input_latent.shape}")
                print(f"controlnet_input_latent.mean(): {controlnet_input_latent.mean()}")
                print(f"controlnet_input_latent.std(): {controlnet_input_latent.std()}")
                #asd()
                if 0:
                    from diffusers import DPMSolverMultistepScheduler
                    scheduler_folder = "../pretrained_models/pixcell-256/scheduler/"
                    scheduler = DPMSolverMultistepScheduler.from_pretrained(
                        scheduler_folder,
                    )
                    scheduler.set_timesteps(config.train_sampling_steps, device=accelerator.device)
                    timesteps = scheduler.timesteps
                    latent_shape = (1, 16, 32, 32)
                    latents = torch.randn(latent_shape, device=accelerator.device, dtype=torch.float32).to(accelerator.device)
                    latents = latents * config.train_sampling_steps
                    with torch.no_grad():
                        with torch.amp.autocast(device_type="cuda", enabled=(accelerator.device=='cuda')):
                            for t in timesteps:
                                # Expand for Classifier-Free Guidance (CFG) batching
                                # This runs cond and uncond in ONE pass
                                latent_model_input = torch.cat([latents] * 1)
                                latent_model_input = scheduler.scale_model_input(latent_model_input, t)
                                current_timestep = t.expand(latent_model_input.shape[0])
                                
                                noise_pred = base_model(
                                    x=latent_model_input,
                                    y=y[:1],
                                    timestep=current_timestep,
                                    controlnet_outputs=None,
                                    return_dict=False,
                                )[0]
                                # --- Step ---
                                print(f"noise_pred.shape: {noise_pred.shape}")
                                print(f"t: {t}")
                                print(f"latents.shape: {latents.shape}")
                                print(f"latent_input.shape: {latent_model_input.shape}")
                                print(f"current_timestep: {current_timestep}")
                                latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                    asd()
                if 0:
                    with torch.no_grad():
                        controlnet_input_latent = controlnet_input_latent/vae_scale+vae_shift
                        decoded_image = vae.decode(controlnet_input_latent).sample
                        decoded_image_mask = vae.decode(vae_mask).sample
                        #decoded_image = vae.decode(z).sample
                    decoded_image = (decoded_image / 2 + 0.5).clamp(0, 1)
                    decoded_image = decoded_image.cpu().permute(0, 2, 3, 1).detach().numpy()
                    decoded_image = (decoded_image * 255).round().astype(np.uint8)
                    import matplotlib.pyplot as plt
                    hist_image = batch[-2][0].cpu().numpy().transpose(1, 2, 0)
                    hist_image = hist_image.clip(0, 255).astype(np.uint8)
                    hist_image = batch[-1][0].cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
                    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
                    ax[0].imshow(hist_image)
                    ax[0].set_title("Controlnet Input")
                    ax[1].imshow(decoded_image[0])
                    ax[1].set_title("Decoded Image")
                    decoded_image_mask = (decoded_image_mask / 2 + 0.5).clamp(0, 1)
                    decoded_image_mask = decoded_image_mask.cpu().permute(0, 2, 3, 1).detach().numpy()
                    decoded_image_mask = (decoded_image_mask * 255).round().astype(np.uint8)
                    ax[2].imshow(decoded_image_mask[0])
                    ax[2].set_title("Decoded Image Mask")
                    plt.show()
                    asd()
                if 0:
                    latent_shape = (1, 16, 32, 32)
                    latents = torch.randn(latent_shape, device=accelerator.device, dtype=torch.float32).to(accelerator.device)
                    latents = latents * config.train_sampling_steps
                    
                if 0:
                    print(f"control_input_latent.shape: {controlnet_input_latent.shape}")
                    print(f"controlnet_input_latent.min(): {controlnet_input_latent.min()}")
                    print(f"controlnet_input_latent.max(): {controlnet_input_latent.max()}")
                    asd()
                if 0:
                    print(f"y.min(): {y.min()}")
                    print(f"y.max(): {y.max()}")
                    #print(f"clean_images.shape: {clean_images.shape}")
                    print(f"y.shape: {y.shape}")
                    asd()
            if control_input.shape[-1] != clean_images.shape[-1]:
                control_input = nn.functional.interpolate(
                    control_input.float(), 
                    size=(clean_images.shape[-2], clean_images.shape[-1]), 
                    mode='nearest'
                ).to(clean_images.device)
            if control_input.shape[1] != config.controlnet_conditioning_channels:
                control_input = control_input.repeat(1, config.controlnet_conditioning_channels, 1, 1)
            #print(f"control_input.shape: {control_input.shape}")
            #asd()
            # Sample timesteps
            bs = clean_images.shape[0]
            timesteps = torch.randint(
                0, config.train_sampling_steps, (bs,), device=clean_images.device
            ).long()
            vae_mask = (vae_mask-vae_shift)*vae_scale
            vae_mask = vae_mask #controlnet_input_latent
            grad_norm = None
            data_time_all += time.time() - data_time_start
            
            # ============================================================
            # 3. Training step with ControlNet
            # ============================================================
            with accelerator.accumulate(controlnet):  # Accumulate gradients for ControlNet
                optimizer.zero_grad()
                # Forward pass through ControlNet AND base model
                model_kwargs = dict(
                    y=y,
                    mask=None,
                    data_info=data_info,
                    control_input=vae_mask,
                    #conditioning_scale=0.0,
                )
                
                # Compute loss using custom training_losses_controlnet
                if 0:
                    print(f"clean_images.shape: {clean_images.shape}")
                    print(f"clean_images.mean(): {clean_images.mean()}")
                    print(f"clean_images.std(): {clean_images.std()}")
                    print('--------------------------------')
                    print(f"timesteps.shape: {timesteps.shape}")
                    print(f"y.shape: {y.shape}")
                    print(f"y.mean(): {y.mean()}")
                    print(f"y.std(): {y.std()}")
                    print('--------------------------------')
                    print(f"vae_mask.shape: {vae_mask.shape}")
                    print(f"vae_mask.mean(): {vae_mask.mean()}")
                    print(f"vae_mask.std(): {vae_mask.std()}")
                    print('--------------------------------')
                    asd()
                loss_term = training_losses_controlnet(
                    diffusion=train_diffusion,
                    controlnet=controlnet,      # Trainable
                    base_model=base_model,      # Frozen
                    x_start=clean_images,
                    timesteps=timesteps,
                    model_kwargs=model_kwargs,
                    config=config
                )
                loss = loss_term['loss']
                # Backward pass (only ControlNet gets gradients)
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    # --- DIAGNOSTIC START ---
                    #print(f"DEBUG: Optimizer Param Groups: {len(optimizer.param_groups)}")
                    #print(f"DEBUG: Current LR: {optimizer.param_groups[0]['lr']:.10f}")
                    
                    # Check if the optimizer is actually tracking the weights we care about
                    # We'll check if the object ID of the weight is in the optimizer's state
                    target_param = controlnet.controlnet_blocks[0].weight
                    is_tracked = any(target_param is p for group in optimizer.param_groups for p in group['params'])
                    #print(f"DEBUG: Is weight_0 tracked by optimizer? {is_tracked}")
                    # --- DIAGNOSTIC END ---
                    accelerator.clip_grad_norm_(controlnet.parameters(), config.gradient_clip)
                    optimizer.step()
                    lr_scheduler.step()
                if accelerator.is_main_process:
                    ema_update(model_ema, controlnet, config.ema_rate)
            if 0:#step % 50 == 0:
                print(f"Step {step}:")
                print(f"  Loss = {loss.item():.6f}")
                
                # Check gradient norms:
                total_norm = 0
                for p in controlnet.parameters():
                    if p.grad is not None:
                        total_norm += p.grad.data.norm(2).item() ** 2
                total_norm = total_norm ** 0.5
                print(f"  Gradient norm: {total_norm:.6f}")
                
                # Check weight magnitudes:
                weight_norm = controlnet.controlnet_blocks[0].weight.norm().item()
                print(f"  Weight norm: {weight_norm:.6f}")
            # ============================================================
            # 4. Logging and monitoring
            # ============================================================
            if accelerator.sync_gradients:
                global_step += 1
                
                # Log metrics
                log_buffer.update({
                    'loss': loss.item(),
                    'lr': optimizer.param_groups[0]['lr'],
                })
                
                if grad_norm is not None:
                    log_buffer.update({'grad_norm': grad_norm.item()})
                
                # Print training info
                if global_step % config.log_interval == 0:
                    time_cost = time.time() - last_tic
                    samples_per_sec = config.log_interval * config.train_batch_size / time_cost
                    
                    info = (
                        f"Epoch [{epoch}/{config.num_epochs}] "
                        f"Step [{global_step}/{total_steps}] "
                        f"Loss: {loss.item():.4f} "
                        f"LR: {optimizer.param_groups[0]['lr']:.2e} "
                        f"Samples/s: {samples_per_sec:.2f}"
                    )
                    
                    if grad_norm is not None:
                        info += f" GradNorm: {grad_norm.item():.4f}"
                    
                    logger.info(info)
                    last_tic = time.time()
                
                # Save checkpoint
                if global_step % config.save_model_steps == 0:
                    save_checkpoint_controlnet(
                        accelerator,
                        controlnet,
                        model_ema,
                        optimizer,
                        lr_scheduler,
                        global_step,
                        epoch,
                        config,
                        logger
                    )
            
            data_time_start = time.time()
            #if step == 10: break
            # Check if we've reached max steps
            if global_step >= total_steps:
                logger.info(f"Reached max steps ({total_steps}). Stopping training.")
                break
        
        if epoch % config.save_model_epochs == 0 or epoch == config.num_epochs:
            save_checkpoint_controlnet(
                accelerator,
                controlnet,
                model_ema,
                optimizer,
                lr_scheduler,
                global_step,
                epoch,
                config,
                logger
            )
        # End of epoch
        if global_step >= total_steps:
            break
    
    logger.info("="*80)
    logger.info("Training Complete!")
    logger.info("="*80)


def training_losses_controlnet(
    diffusion, 
    controlnet, 
    base_model, 
    x_start, 
    timesteps, 
    model_kwargs=None,
    config=None
):
    """
    Custom training loss function that uses both ControlNet and base model.
    
    Flow:
    1. ControlNet processes latents + conditioning → outputs residuals
    2. Base model uses residuals to predict noise
    3. Compute loss between predicted and actual noise
    
    Args:
        diffusion: Diffusion scheduler
        controlnet: Trainable ControlNet model
        base_model: Frozen base transformer model
        x_start: Clean latent images
        timesteps: Sampled timesteps
        model_kwargs: Dict with 'y', 'control_input', 'data_info', etc.
        config: Training config
    
    Returns:
        Dict with 'loss' and other metrics
    """
    if model_kwargs is None:
        model_kwargs = {}
    
    # Add noise to clean images
    noise = torch.randn_like(x_start).float()
    x_t = diffusion.q_sample(x_start, timesteps, noise=noise)
    # Extract control input
    control_input = model_kwargs.pop('control_input', None)
    
    if control_input is None:
        raise ValueError("control_input must be provided in model_kwargs for ControlNet training!")
    
    # Get conditioning scale (can be dynamic during training)
    conditioning_scale = getattr(config, 'controlnet_conditioning_scale', 1.0)
    
    pred_sigma = getattr(config, 'pred_sigma', True)
    learn_sigma = getattr(config, 'learn_sigma', True) and pred_sigma

    # ============================================================
    # Forward through ControlNet (TRAINABLE)
    # ============================================================
    controlnet_outputs = controlnet(
        hidden_states=x_t,
        conditioning=control_input,
        encoder_hidden_states=model_kwargs['y'],
        timestep=timesteps,
        conditioning_scale=conditioning_scale,
        mask=model_kwargs.get('mask', None),
        data_info=model_kwargs.get('data_info', None),
    )

    # Extract outputs (controlnet returns tuple)
    if isinstance(controlnet_outputs, tuple):
        controlnet_residuals = controlnet_outputs[0]
    else:
        controlnet_residuals = controlnet_outputs['controlnet_block_samples']
    if config.mixed_precision in ['fp16', 'bf16']:
        controlnet_residuals = [r.float() for r in controlnet_residuals]
    # ============================================================
    # Forward through base model (FROZEN)
    # ============================================================
    model_output = base_model(
        x=x_t,
        y=model_kwargs['y'],
        timestep=timesteps,
        controlnet_outputs=controlnet_residuals,  # Add ControlNet residuals
        attention_mask=model_kwargs.get('mask', None),
        return_dict=True,
    )
    
    # Get predicted noise
    if hasattr(model_output, 'sample'):
        model_pred = model_output.sample
    else:
        model_pred = model_output


    model_var_values = None
    if learn_sigma and model_pred.shape[1] == x_start.shape[1] * 2:
        # Model is predicting both noise and variance
        model_pred, model_var_values = model_pred.chunk(2, dim=1)
    # ============================================================
    # Compute loss
    # ============================================================
    # Standard MSE loss between predicted and actual noise
    loss = torch.nn.functional.mse_loss(model_pred, noise, reduction='none')
    loss = loss.mean(dim=list(range(1, len(loss.shape))))
    return {
        'loss': loss.mean(),
        'pred': model_pred,
        'target': noise,
        'var_values': model_var_values if learn_sigma else None,
    }
# %%
def main():
    # %%
    # Start training
    # train_controlnet(models)
    # %%
    """Main entry point for training."""
    # %%
    # Initialize config and accelerator
    init_data = initialize_config_and_accelerator([
        '../configs/pan_cancer/config_controlnet_gan.py',
    ])
    config = init_data['config']
    accelerator = init_data['accelerator']
    logger = init_data['logger']
    args = init_data['args']
    device = accelerator.device
    # %%
    # Initialize models
    model_data = initialize_models(config, accelerator, logger)
    base_model = model_data['base_model']
    controlnet = model_data['controlnet']
    model_ema = model_data['model_ema']
    vae = model_data['vae']
    train_diffusion = model_data['train_diffusion']
    # %%
    # Initialize dataset and optimizer
    discriminator = None  # Add discriminator initialization if needed
    optim_data = initialize_dataset_and_optimizer(
        config, accelerator, logger, model=controlnet, discriminator=discriminator
    )
    train_dataloader = optim_data['train_dataloader']
    optimizer = optim_data['optimizer']
    optimizer_d = optim_data['optimizer_d']
    lr_scheduler = optim_data['lr_scheduler']
    # %%
    # Prepare with accelerator
    (
        base_model, 
        controlnet, 
        model_ema, 
        optimizer, 
        train_dataloader, 
        lr_scheduler
    ) = accelerator.prepare(
        base_model, 
        controlnet, 
        model_ema, 
        optimizer, 
        train_dataloader, 
        lr_scheduler
    )
    
    if discriminator is not None:
        discriminator = accelerator.prepare(discriminator)
        optimizer_d = accelerator.prepare(optimizer_d)
    
    # Setup training state
    state_data = setup_training_state(
        config, accelerator, logger, args, train_dataloader,
        base_model, controlnet, model_ema, optimizer, lr_scheduler
    )

    # Prepare models dict for training
    models = {
        'base_model': base_model,
        'controlnet': controlnet,
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
    # %%
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt

    for step, batch in enumerate(train_dataloader):
        # Adjust indexing based on your return: 
        # batch[-2] is img, batch[-4] is cell_mask (binary)
        hist_image = batch[-2][0].cpu().numpy() # Assuming 'img' is last
        mask_image = batch[-1][0].cpu().numpy() # Assuming 'cell_mask' is here
        bg_img = hist_image.transpose(1, 2, 0)
        bg_img = bg_img.clip(0, 255).astype(np.uint8)
        overlap_img = bg_img.copy()

        # 2. Extract Contours
        # Ensure mask is 2D uint8 (H, W)
        binary_mask = (mask_image[0] > 0.5).astype(np.uint8)
        if binary_mask.ndim == 3:
            binary_mask = binary_mask.squeeze()
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 3. Draw Contours (Color is BGR, so (0, 255, 0) is Green)
        cv2.drawContours(overlap_img, contours, -1, (0, 255, 0), 1)

        # 4. Plot
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(bg_img)
        ax[0].set_title("Original Hist")
        
        ax[1].imshow(binary_mask, cmap='gray')
        ax[1].set_title("Binary Mask")
        
        ax[2].imshow(overlap_img)
        ax[2].set_title("Overlap (Contours)")
        plt.show()
        if step == 0:break
    # %% [Check ssl features]
    from diffusers import DPMSolverMultistepScheduler
    from diffusers.utils.torch_utils import randn_tensor
    for step, batch in enumerate(train_dataloader):
        y = batch[1][:1]
        break
    y = y.unsqueeze(0)
    scheduler = DPMSolverMultistepScheduler.from_pretrained(
        "StonyBrook-CVLab/PixCell-256",
        subfolder="scheduler"
    )
    latent_shape = (
        1,
        16,
        32,
        32,
    )
    latent_channels = base_model.in_channels
    generator = torch.Generator(device=device).manual_seed(42)
    num_inference_steps = 20
    scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = scheduler.timesteps
    latents = randn_tensor(latent_shape, generator=generator, device=device, dtype=torch.float16)
    latents = latents * scheduler.init_noise_sigma
    # ============================================
    # 8. Denoising Loop
    # ============================================
    with torch.no_grad(), torch.amp.autocast('cuda'):
        for i, t in enumerate(timesteps):
            # Expand latents for CFG
            latent_model_input = latents
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)

            # Prepare timestep
            timestep = t
            if not torch.is_tensor(timestep):
                timestep = torch.tensor([timestep], dtype=torch.float32, device=device)
            timestep = timestep.expand(latent_model_input.shape[0])
            # Predict noise
            noise_pred = base_model(
                x=latent_model_input,
                y=y,
                timestep=timestep,
                controlnet_outputs=None,
                return_dict=False,
            )
            if base_model.out_channels // 2 == latent_channels:
                noise_pred = noise_pred.chunk(2, dim=1)[0]
            # Denoise step
            latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            print(f"Step {i+1}/{num_inference_steps}")
    vae_scale = vae.config.scaling_factor
    vae_shift = getattr(vae.config, "shift_factor", 0)
    # Convert latents to float32 to match VAE's dtype
    latents_for_decode = latents.float()

    with torch.no_grad():
        latents_in = (latents_for_decode / vae_scale) + vae_shift
        latents_in = latents_in.to(dtype=vae.dtype, device=vae.device)  # match fp16/fp32 + device

        generated_image = vae.decode(latents_in, return_dict=False)[0]

    # Post-process to PIL image
    generated_image = (generated_image / 2 + 0.5).clamp(0, 1)
    generated_image = generated_image.cpu().permute(0, 2, 3, 1).numpy()
    generated_image = (generated_image * 255).round().astype(np.uint8)
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(batch[-2][0].cpu().numpy().transpose(1, 2, 0).astype(np.uint8))
    ax[0].set_title("Original Image")
    ax[1].imshow(generated_image[0])
    ax[1].set_title("Generated Image")
    plt.show()
    # %%
    # Start training
    train_controlnet(models)

# %%
if __name__ == "__main__":
    main()
# %%
