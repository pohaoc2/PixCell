# %%
import os 
import argparse
import datetime
import sys
import time
import types
import warnings
from pathlib import Path
from copy import deepcopy
import math
current_file_path = Path(__file__).resolve()
sys.path.insert(0, str(current_file_path.parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
torch.cuda.set_per_process_memory_fraction(0.995, 0)

print(f"GPU limit set to {torch.cuda.get_device_properties(0).total_memory * 0.995 / 1024**2:.0f} MB")

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

@torch.no_grad()
def verify_controlnet_initialization(model):
    """
    Verify that ControlNet weights were correctly copied from base model
    
    Args:
        model: PixArt_UNI_ControlNet instance
        
    Returns:
        bool: True if verification passes
    """
    logger = get_root_logger()
    
    print("\n" + "="*70)
    print("VERIFYING CONTROLNET INITIALIZATION")
    print("="*70)
    
    base_depth = len(model.blocks)
    control_depth = len(model.controlnet.control_blocks)
    
    print(f"\nBase model blocks: {base_depth}")
    print(f"ControlNet blocks: {control_depth}")
    
    all_passed = True
    
    # 1. Verify positional embeddings
    print("\n1. Checking positional embeddings...")
    if torch.allclose(model.controlnet.pos_embed, model.pos_embed, atol=1e-6):
        print("   ✅ Positional embeddings match")
    else:
        print("   ❌ Positional embeddings DON'T match")
        all_passed = False
    
    # 2. Verify transformer blocks
    print("\n2. Checking transformer block weights...")
    
    for control_idx in range(control_depth):
        # Calculate which base block should have been copied
        base_idx = int(control_idx * base_depth / control_depth)
        base_idx = min(base_idx, base_depth - 1)
        
        control_block = model.controlnet.control_blocks[control_idx]
        base_block = model.blocks[base_idx]
        
        checks = []
        
        # Check self-attention QKV
        qkv_match = torch.allclose(
            control_block.attn.qkv.weight, 
            base_block.attn.qkv.weight, 
            atol=1e-6
        )
        checks.append(("Self-attn QKV", qkv_match))
        
        # Check self-attention projection
        proj_match = torch.allclose(
            control_block.attn.proj.weight,
            base_block.attn.proj.weight,
            atol=1e-6
        )
        checks.append(("Self-attn Proj", proj_match))
        
        # Check cross-attention Q
        q_match = torch.allclose(
            control_block.cross_attn.q_linear.weight,
            base_block.cross_attn.q_linear.weight,
            atol=1e-6
        )
        checks.append(("Cross-attn Q", q_match))
        
        # Check cross-attention KV
        kv_match = torch.allclose(
            control_block.cross_attn.kv_linear.weight,
            base_block.cross_attn.kv_linear.weight,
            atol=1e-6
        )
        checks.append(("Cross-attn KV", kv_match))
        
        # Check MLP fc1
        fc1_match = torch.allclose(
            control_block.mlp.fc1.weight,
            base_block.mlp.fc1.weight,
            atol=1e-6
        )
        checks.append(("MLP FC1", fc1_match))
        
        # Check MLP fc2
        fc2_match = torch.allclose(
            control_block.mlp.fc2.weight,
            base_block.mlp.fc2.weight,
            atol=1e-6
        )
        checks.append(("MLP FC2", fc2_match))
        
        # Check scale_shift_table
        scale_match = torch.allclose(
            control_block.scale_shift_table,
            base_block.scale_shift_table,
            atol=1e-6
        )
        checks.append(("Scale-shift", scale_match))
        
        # Report for this block
        block_passed = all(match for _, match in checks)
        if block_passed:
            print(f"   ✅ Control block {control_idx} ← Base block {base_idx}: ALL weights match")
        else:
            print(f"   ❌ Control block {control_idx} ← Base block {base_idx}: MISMATCHES:")
            for name, match in checks:
                if not match:
                    print(f"      ✗ {name}")
            all_passed = False
    
    # 3. Verify zero convs are still zero
    print("\n3. Checking zero convolutions...")
    zero_convs_ok = True
    for i, zero_conv in enumerate(model.controlnet.zero_convs):
        is_zero = torch.allclose(zero_conv.weight, torch.zeros_like(zero_conv.weight), atol=1e-8)
        if is_zero:
            print(f"   ✅ Zero conv {i}: initialized to zero")
        else:
            print(f"   ❌ Zero conv {i}: NOT zero (mean: {zero_conv.weight.abs().mean():.6f})")
            zero_convs_ok = False
    
    if not zero_convs_ok:
        all_passed = False
    
    # 4. Verify control_x_embedder is different (not copied)
    print("\n4. Checking control_x_embedder (should be different)...")
    control_x_weight = model.controlnet.control_x_embedder.proj.weight
    base_x_weight = model.x_embedder.proj.weight
    
    # They should have different shapes (different input channels)
    if control_x_weight.shape != base_x_weight.shape:
        print(f"   ✅ Different shapes: control={control_x_weight.shape[1]} vs base={base_x_weight.shape[1]} channels")
    else:
        # Same shape - check if they're different values
        if not torch.allclose(control_x_weight, base_x_weight, atol=1e-6):
            print(f"   ✅ Different weights (as expected)")
        else:
            print(f"   ⚠️  Weights are identical (might be ok if both use same init)")
    
    # Summary
    print("\n" + "="*70)
    if all_passed:
        print("✅ ALL VERIFICATIONS PASSED - ControlNet correctly initialized!")
    else:
        print("❌ SOME VERIFICATIONS FAILED - Check logs above")
    print("="*70)
    
    return all_passed

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


    
def verify_pixcell_checkpoint(model, checkpoint_path=None, device='cuda'):
    """Load checkpoint directly and verify key values"""
    from safetensors.torch import load_file
    if checkpoint_path is None:
        checkpoint_path = "../pretrained_models/pixcell-256/transformer/diffusion_pytorch_model.safetensors"
    else:
        checkpoint_path = checkpoint_path
    state_dict = load_file(checkpoint_path)
    
    # Check a specific weight from the checkpoint
    original_timestep_weight = state_dict['adaln_single.emb.timestep_embedder.linear_1.weight']
    print(f"Original checkpoint timestep embedder weight (first 5 values):")
    print(original_timestep_weight[0, :5])
    
    # After loading into your model, check if it matches
    model_timestep_weight = model.t_embedder.mlp[0].weight
    print(f"\nModel timestep embedder weight (first 5 values):")
    print(model_timestep_weight[0, :5])
    
    # Check if they match
    if torch.allclose(original_timestep_weight.to(device), model_timestep_weight.to(device), atol=1e-6):
        print("\n✅ Weights match! PixCell loaded correctly.")
    else:
        print("\n❌ Weights don't match! Something went wrong.")

@torch.no_grad()
def generate_image_with_diffusion(model, vae, uni_feature_path, num_steps=50, device='cuda'):
    """
    Proper image generation using diffusion sampling
    """
    from diffusers import DDPMScheduler  # or DDIMScheduler for faster sampling
    
    model.eval()
    vae.eval()
    
    # Load UNI features
    y = torch.from_numpy(np.load(uni_feature_path))
    y = y.view(1, 1, 1, 1536).expand(1, 1, 120, 1536).to(device)
    
    # Initialize scheduler
    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_schedule="linear",
    )
    scheduler.set_timesteps(num_steps)
    
    # Start with random noise
    latent_shape = (1, 16, 32, 32)
    latents = torch.randn(latent_shape, device=device)
    
    print(f"Starting diffusion sampling with {num_steps} steps...")
    
    # Iterative denoising
    for i, t in enumerate(scheduler.timesteps):
        print(f"Step {i+1}/{num_steps}, timestep: {t}", end='\r')
        
        timestep = torch.tensor([t], device=device)
        
        # Predict noise
        noise_pred = model.forward_without_controlnet(latents, timestep, y)
        
        # Handle pred_sigma
        if model.pred_sigma:
            noise_pred = noise_pred.chunk(2, dim=1)[0]
        
        # Denoise one step
        latents = scheduler.step(noise_pred, t, latents).prev_sample
    
    print("\nSampling complete! Decoding...")
    
    # Decode final latents
    vae_dtype = next(vae.parameters()).dtype
    latents = latents.to(vae_dtype)
    image = vae.decode((latents / vae.config.scaling_factor) + vae.config.shift_factor, return_dict=False)[0]
    
    # Save
    from torchvision.utils import save_image
    save_image((image + 1) / 2, 'pixcell_generated.png')
    print("✅ Saved generated image to: pixcell_generated.png")
    
    return image


def parse_args(args_list=None):
    """
    Args:
        args_list: List of arguments (for programmatic use) or None (for CLI use)
    """
    parser = argparse.ArgumentParser(description="Train ControlNet for PixCell-256")
    parser.add_argument("config", type=str, help="config")
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--resume-from', help='the dir to resume the training')
    parser.add_argument('--local-rank', type=int, default=-1)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument("--report-to", type=str, default="tensorboard")
    parser.add_argument("--tracker-project-name", type=str, default="pixcell_controlnet")
    parser.add_argument("--slurm-time-limit", type=float, default=float('inf'))
    parser.add_argument("--loss-report-name", type=str, default="loss")
    parser.add_argument("--skip-step", type=int, default=0)
    
    # Parse from args_list if provided, otherwise from sys.argv
    return parser.parse_args(args_list)


def initialize_config_and_accelerator(args_list=None):
    """
    Parse arguments, read config, and initialize accelerator
    
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
            resume_lr_scheduler=True)
    
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
    Initialize base model, EMA, VAE, and optional discriminator/segmentation
    
    Returns:
        dict with: base_model, model_ema, vae, train_diffusion, 
                   discriminator, segmentation_checker
    """
    image_size = config.image_size
    latent_size = int(image_size) // 8
    pred_sigma = getattr(config, 'pred_sigma', True)
    learn_sigma = getattr(config, 'learn_sigma', True) and pred_sigma
    max_length = config.model_max_length
    kv_compress_config = config.kv_compress_config if config.kv_compress else None
    print("VAE" + "="*70)
    print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    # Load VAE
    vae = AutoencoderKL.from_pretrained(
        "../pretrained_models/sd-3.5-vae/vae",
        local_files_only=True,
        use_safetensors=True,  # Explicitly tell it to look for the safetensors file you have
        trust_remote_code=True # Sometimes required if the VAE uses custom scaling
    )
    vae.to('cpu')
    # Delete encoder components before moving to GPU
    if hasattr(vae, 'encoder') and vae.encoder is not None:
        vae.encoder.cpu()  # Ensure it's on CPU
        del vae.encoder
        
    if hasattr(vae, 'quant_conv') and vae.quant_conv is not None:
        vae.quant_conv.cpu()
        del vae.quant_conv
    # Set to None to break any remaining references
    vae.encoder = None
    vae.quant_conv = None

    # Force garbage collection
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    # Now move only the decoder to GPU
    vae.to(accelerator.device)
    print('-'*70)
    print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    
    config.scale_factor = vae.config.scaling_factor
    logger.info(f"vae scale factor: {config.scale_factor}")
    
    # Setup model kwargs
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
    
    # Build base model with ControlNet
    print("Base Model" + "="*70)
    print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    base_model = _build_and_load_base_model(config, model_kwargs, accelerator, logger, max_length)
    print('-'*70)
    print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    # Build EMA model
    print("EMA Model" + "="*70)
    print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    model_ema = _build_ema_model(base_model, config, model_kwargs, accelerator, logger)
    print('-'*70)
    print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    
    # Build optional discriminator
    discriminator = _build_discriminator(config, logger) if config.get('use_discriminator', False) else None
    
    # Build optional segmentation checker
    segmentation_checker = _build_segmentation_checker(config, accelerator, image_size, logger) \
        if config.get('use_segmentation_consistency', False) else None
    
    return {
        'base_model': base_model,
        'model_ema': model_ema,
        'vae': vae,
        'train_diffusion': train_diffusion,
        'discriminator': discriminator,
        'segmentation_checker': segmentation_checker,
    }


def initialize_dataset_and_optimizer(config, accelerator, logger, base_model, discriminator=None):
    """
    Initialize dataset, dataloader, optimizer, and lr_scheduler
    
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
    Setup tracking, resume from checkpoint if needed, prepare with accelerator
    
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


def initialize_all(args_list=None):
    """
    Main initialization function that orchestrates all setup steps
    
    Returns:
        Complete dict with all training components
    """
    # Step 1: Config and accelerator
    init_data = initialize_config_and_accelerator(args_list)
    config = init_data['config']
    accelerator = init_data['accelerator']
    logger = init_data['logger']
    args = init_data['args']
    
    # Step 2: Models
    model_data = initialize_models(config, accelerator, logger)
    
    # Step 3: Dataset and optimizers
    optim_data = initialize_dataset_and_optimizer(
        config, accelerator, logger,
        model_data['base_model'],
        model_data['discriminator']
    )
    
    # Step 4: Prepare everything with accelerator
    base_model = accelerator.prepare(model_data['base_model'])
    model_ema = accelerator.prepare(model_data['model_ema'])
    optimizer = accelerator.prepare(optim_data['optimizer'])
    train_dataloader = accelerator.prepare(optim_data['train_dataloader'])
    lr_scheduler = accelerator.prepare(optim_data['lr_scheduler'])
    
    discriminator = None
    optimizer_d = None
    if model_data['discriminator'] is not None:
        discriminator = accelerator.prepare(model_data['discriminator'])
        optimizer_d = accelerator.prepare(optim_data['optimizer_d'])
    
    # Step 5: Training state
    state_data = setup_training_state(
        config, accelerator, logger, args, train_dataloader,
        base_model, model_ema, optimizer, lr_scheduler
    )
    
    # Return everything
    return {
        # Core Models
        'base_model': base_model,
        'model_ema': model_ema,
        'vae': model_data['vae'],
        'train_diffusion': model_data['train_diffusion'],
        
        # Training Components
        'optimizer': optimizer,
        'optimizer_d': optimizer_d,
        'lr_scheduler': lr_scheduler,
        'train_dataloader': train_dataloader,
        
        # Optional Models
        'discriminator': discriminator,
        'segmentation_checker': model_data['segmentation_checker'],
        
        # Accelerator & Config
        'accelerator': accelerator,
        'config': config,
        'logger': logger,
        
        # Training State
        'start_epoch': state_data['start_epoch'],
        'start_step': state_data['start_step'],
        'skip_step': state_data['skip_step'],
        'total_steps': state_data['total_steps'],
        
        # Additional info
        'args': args,
    }


# Helper functions
def _find_checkpoint(resume_from):
    """Find latest checkpoint in directory or return file path"""
    if os.path.isdir(resume_from):
        checkpoints = [ckpt for ckpt in os.listdir(resume_from) if ckpt.endswith('.pth')]
        if len(checkpoints) == 0:
            raise ValueError(f"No checkpoint found in {resume_from}")
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split('_')[-1].replace('.pth', '')), reverse=True)
        return os.path.join(resume_from, checkpoints[0])
    return resume_from


def _setup_accelerator(config, args):
    """Setup accelerator with FSDP or DDP"""
    init_handler = InitProcessGroupKwargs()
    init_handler.timeout = datetime.timedelta(seconds=5400)
    
    if config.use_fsdp:
        from accelerate import FullyShardedDataParallelPlugin
        from torch.distributed.fsdp.fully_sharded_data_parallel import FullStateDictConfig
        set_fsdp_env()
        fsdp_plugin = FullyShardedDataParallelPlugin(
            state_dict_config=FullStateDictConfig(offload_to_cpu=False, rank0_only=False)
        )
    else:
        fsdp_plugin = None
    
    from accelerate import Accelerator, DataLoaderConfiguration
    dataloader_config = DataLoaderConfiguration(dispatch_batches=True)
    
    return Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with=args.report_to,
        project_dir=os.path.join(config.work_dir, "logs"),
        fsdp_plugin=fsdp_plugin,
        dataloader_config=dataloader_config,
        kwargs_handlers=[init_handler]
    )


def _build_and_load_base_model(config, model_kwargs, accelerator, logger, max_length):
    """Build base model and load pretrained weights"""
    from diffusion.model.builder import build_model
    
    logger.info("Building PixCell model architecture...")
    base_model = build_model(config.model, **model_kwargs).to(accelerator.device)
    
    if config.load_from is not None:
        load_file = _find_model_file(config.load_from)
        logger.info(f"Loading pretrained base model weights from {load_file}")
        
        missing, unexpect = load_checkpoint(
            load_file,
            base_model,
            load_ema=config.get('load_ema', False),
            max_length=max_length
        )
        
        logger.info(f"Missing keys: {len(missing)}")
        logger.info(f"Unexpected keys: {len(unexpect)}")
        
        if missing:
            logger.warning(f"Missing keys: {missing}")
        if unexpect:
            logger.warning(f"Unexpected keys (first 10): {unexpect[:10]}")
        
        logger.info("Initializing ControlNet from base model weights...")
        _initialize_controlnet_from_base(base_model)
        _print_trainable_parameters(base_model, logger)
    
    # Enable gradient checkpointing
    if hasattr(base_model, 'controlnet'):
        for block in base_model.controlnet.control_blocks:
            block.gradient_checkpointing = True
    
    logger.info(f"{base_model.__class__.__name__} Model Parameters: {sum(p.numel() for p in base_model.parameters()):,}")
    logger.info(f"Trainable Parameters: {sum(p.numel() for p in base_model.parameters() if p.requires_grad):,}")
    
    return base_model


def _find_model_file(load_from):
    """Find safetensors file in directory or return path"""
    load_path = Path(load_from)
    if load_path.is_dir():
        st_files = list(load_path.glob("**/diffusion_pytorch_model.safetensors"))
        return str(st_files[0]) if st_files else str(load_from)
    return str(load_from)


def _build_ema_model(base_model, config, model_kwargs, accelerator, logger):
    """Build EMA model and initialize from base model"""
    from diffusion.model.builder import build_model
    
    logger.info("Initializing EMA model architecture...")
    model_ema = build_model(config.model, **model_kwargs).to(accelerator.device)
    model_ema.load_state_dict(base_model.state_dict())
    model_ema.eval()
    
    for param in model_ema.parameters():
        param.requires_grad = False
    
    logger.info("✓ EMA model initialized via state_dict copy.")
    ema_update(model_ema, base_model, 0.)
    
    return model_ema


def _build_discriminator(config, logger):
    """Build discriminator for adversarial training"""
    logger.info("Building discriminator for adversarial training...")
    from discriminator import build_discriminator
    
    discriminator = build_discriminator(config.discriminator).train()
    logger.info(f"Discriminator Parameters: {sum(p.numel() for p in discriminator.parameters()):,}")
    
    return discriminator


def _build_segmentation_checker(config, accelerator, image_size, logger):
    """Build segmentation consistency checker"""
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
            for param in segmentation_checker.model.parameters():
                param.requires_grad = False
            logger.info("✓ Using lightweight U-Net segmentation model (frozen)")
    
    return segmentation_checker


def _resume_from_checkpoint(config, base_model, model_ema, optimizer, lr_scheduler, max_length, logger):
    """Resume training from checkpoint"""
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
    
    return start_epoch, start_step

def extract_uni_emb(model_path=None):
    import timm
    if model_path is None:
        model_path = "/home/ec2-user/PixCell/pretrained_models/uni-2h/"
    model_path = Path(model_path)
    weight_files = list(model_path.glob("*.pth")) + list(model_path.glob("pytorch_model.bin"))
    timm_kwargs = {
                'img_size': 224,
                'patch_size': 14,
                'depth': 24,
                'num_heads': 24,
                'init_values': 1e-5,
                'embed_dim': 1536,
                'mlp_ratio': 2.66667*2,
                'num_classes': 0,
                'no_embed_class': True,
                'mlp_layer': timm.layers.SwiGLUPacked,
                'act_layer': torch.nn.SiLU,
                'reg_tokens': 8,
                'dynamic_img_size': True
            }

    uni_model = timm.create_model("vit_huge_patch14_224", pretrained=False, **timm_kwargs)
    checkpoint = torch.load(weight_files[0], map_location='cpu')
    
    # Handle different checkpoint formats
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    uni_model.load_state_dict(state_dict, strict=False)
    uni_model.eval()
    uni_model.to('cpu')
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform
    uni_transforms = create_transform(**resolve_data_config(uni_model.pretrained_cfg, model=uni_model))
    image = Image.open("../test_image.png").convert("RGB")
    uni_inp = uni_transforms(image).unsqueeze(dim=0)
    with torch.inference_mode():
        uni_emb = uni_model(uni_inp.to('cpu'))
    # save the uni_emb
    np.save("uni_emb.npy", uni_emb.cpu().numpy())
# %%
if __name__ == "__main__":
    # %%
    init_data = initialize_config_and_accelerator([
        '../configs/pan_cancer/config_controlnet_gan.py',
    ])
    config = init_data['config']
    accelerator = init_data['accelerator']
    logger = init_data['logger']
    args = init_data['args']
    # %%
    model_data = initialize_models(config, accelerator, logger)
    base_model = model_data['base_model']
    model_ema = model_data['model_ema']
    vae = model_data['vae']
    train_diffusion = model_data['train_diffusion']
    discriminator = model_data['discriminator']
    segmentation_checker = model_data['segmentation_checker']
    # %%
    verify_controlnet_initialization(base_model)
    verify_pixcell_checkpoint(base_model, checkpoint_path="../pretrained_models/pixcell-256/transformer/diffusion_pytorch_model.safetensors", device=accelerator.device)
    # %%
    optim_data = initialize_dataset_and_optimizer(
        config, accelerator, logger,
        base_model,
        discriminator
    )
    train_dataloader = optim_data['train_dataloader']
    optimizer = optim_data['optimizer']
    optimizer_d = optim_data['optimizer_d']
    lr_scheduler = optim_data['lr_scheduler']
    optim_data = initialize_dataset_and_optimizer(
    config, accelerator, logger,
    base_model,
    discriminator
    )
    # %%
    train_dataloader = optim_data['train_dataloader']
    optimizer = optim_data['optimizer']
    optimizer_d = optim_data['optimizer_d']
    lr_scheduler = optim_data['lr_scheduler']

    # Step 4: Prepare with accelerator (CRITICAL!)
    base_model = accelerator.prepare(base_model)
    model_ema = accelerator.prepare(model_ema)
    optimizer = accelerator.prepare(optimizer)
    train_dataloader = accelerator.prepare(train_dataloader)
    lr_scheduler = accelerator.prepare(lr_scheduler)

    if discriminator is not None:
        discriminator = accelerator.prepare(discriminator)
        optimizer_d = accelerator.prepare(optimizer_d)

    # Step 5: Training state
    state_data = setup_training_state(
        config, accelerator, logger, args, train_dataloader,
        base_model, model_ema, optimizer, lr_scheduler
    )
    # %%
    models = {
        # Core Models
        'base_model': base_model,
        'model_ema': model_ema,
        'vae': vae,
        'train_diffusion': train_diffusion,
        
        # Training Components
        'optimizer': optimizer,
        'optimizer_d': optimizer_d,
        'lr_scheduler': lr_scheduler,
        'train_dataloader': train_dataloader,
        
        # Optional Models
        'discriminator': discriminator,
        'segmentation_checker': segmentation_checker,
        
        # Accelerator & Config
        'accelerator': accelerator,
        'config': config,
        'logger': logger,
        
        # Training State
        'start_epoch': state_data['start_epoch'],
        'start_step': state_data['start_step'],
        'skip_step': state_data['skip_step'],
        'total_steps': state_data['total_steps'],
        
        # Additional info
        'args': args,
    }
    # %%
    # reload the module
    import importlib
    import train_scripts.train_controlnet  # Import the module itself

    # 1. Reload the module to pick up code changes
    importlib.reload(train_scripts.train_controlnet)

    # 2. Access the function from the freshly reloaded module
    from train_scripts.train_controlnet import train

    # 3. Execute
    train(models)
    # %%
    #models = initialize_all([
    #    '../configs/pan_cancer/config_controlnet_gan.py',
    #])
    
    import importlib
    import inference
    importlib.reload(inference)
    from inference import load_models, generate_image, generate_image_independent_cfg
    from torchvision.utils import save_image
    from diffusers import AutoencoderKL, DPMSolverMultistepScheduler
    import matplotlib.pyplot as plt
    import torch
    import numpy as np
    import json
    from PIL import Image
    with open('../pretrained_models/pixcell-256/scheduler/scheduler_config.json', 'r', encoding='utf-8') as file:
        scheduler_config = json.load(file)
    scheduler = DPMSolverMultistepScheduler(**scheduler_config)
    file_name = "epoch_5_step_785.pth"
    if file_name is not None:
        model, vae, config = load_models(
            model_path=f"../{file_name}",
            vae_path="../pretrained_models/sd-3.5-vae/vae",
            config_path="../configs/pan_cancer/config_controlnet_gan.py",
            device='cuda'
        )
    else:
        model = base_model
        vae = model_data['vae']
        config = config
    # %%
    from huggingface_hub import hf_hub_download
    from PIL import Image
    from inference import load_models, generate_image, generate_image_independent_cfg
    import torch
    import json
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    from torchvision.utils import save_image
    from diffusers import AutoencoderKL, DPMSolverMultistepScheduler
    # This is an example image we provide
    torch.cuda.empty_cache()
    with open('../pretrained_models/pixcell-256/scheduler/scheduler_config.json', 'r', encoding='utf-8') as file:
        scheduler_config = json.load(file)
    scheduler = DPMSolverMultistepScheduler(**scheduler_config)
    # reshape UNI to (bs, 1, D)
    uni_emb = torch.from_numpy(np.load("uni_emb.npy")).unsqueeze(1).unsqueeze(1)
    uni_emb = uni_emb / 1536 ** 0.5
    print("Extracted UNI:", uni_emb.shape)
    device = 'cuda'
    y = torch.randn(1, 1, 1,1536).to(device)
    y /= 1536 ** 0.5
    uni_feature = torch.from_numpy(np.load(f"../features/sample_0_uni.npy"))
    uni_feature /= 1536 ** 0.5
    #uni_feature = y
    os.makedirs("../controlNet_gen", exist_ok=True)
    for idx in range(1):
        image = generate_image_independent_cfg(
            model=base_model,
            vae=vae,
            uni_feature=uni_feature,#f"features/sample_{idx}_uni.npy",#
            cell_mask=f"../masks/sample_{idx}_mask.png",
            config=config,
            num_inference_steps=20,
            seed=42,
            scheduler=scheduler,
            uni_guidance_scale=1,
            mask_guidance_scale=0, 
        )
        save_image(image, f"../controlNet_gen/generated_{idx}.png")
        print(f"✓ saved generated_{idx}.png")
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(Image.open(f"../tcga_subset_0.1k/sample_{idx}.png"))
        ax[0].imshow(Image.open(f"../test_image.png"))
        ax[1].imshow(Image.open(f"../masks/sample_{idx}_mask.png"))
        ax[2].imshow(Image.open(f"../controlNet_gen/generated_{idx}.png"))
    plt.imshow(Image.open(f"../controlNet_gen/generated_{idx}.png"))
    # %%
    #tmp_image = image.clone()
    plt.imshow(tmp_image.cpu().numpy().transpose(1, 2, 0))
    # %%
    min_diff = np.min(tmp_image.cpu().numpy().transpose(1, 2, 0) - image.cpu().numpy().transpose(1, 2, 0))
    max_diff = np.max(tmp_image.cpu().numpy().transpose(1, 2, 0) - image.cpu().numpy().transpose(1, 2, 0))
    print(f"Min diff: {min_diff}, Max diff: {max_diff}")
    plt.imshow((tmp_image.cpu().numpy().transpose(1, 2, 0) - image.cpu().numpy().transpose(1, 2, 0)) / (max_diff - min_diff), vmin=0, vmax=1)

# %%
