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
from initialize_controlnet import initialize_pixcell_controlnet_model, initialize_controlnet_model, load_pixcell_controlnet_model, load_controlnet_model, load_vae


warnings.filterwarnings("ignore")
torch.cuda.set_per_process_memory_fraction(0.995, 0)


def train():
    pass


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
    pixcell_controlnet_model = initialize_pixcell_controlnet_model(config.pixcell_controlnet_module_name, config.pixcell_controlnet_file_path, config.pixcell_controlnet_checkpoints_folder, accelerator.device)
    controlnet_model = initialize_controlnet_model(config.controlnet_module_name, config.controlnet_file_path, config.controlnet_checkpoints_folder, accelerator.device)
    vae = load_vae(config.vae_pretrained, accelerator.device)
    max_length = config.model_max_length
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
    pred_sigma = getattr(config, 'pred_sigma', True)
    learn_sigma = getattr(config, 'learn_sigma', True) and pred_sigma
    train_diffusion = IDDPM(
        str(config.train_sampling_steps),
        learn_sigma=learn_sigma,
        pred_sigma=pred_sigma,
        snr=config.snr_loss
    )
    model_ema = _build_ema_model(pixcell_controlnet_model, config, model_kwargs, accelerator, logger)
    return {
        'pixcell_controlnet_model': pixcell_controlnet_model,
        'controlnet_model': controlnet_model,
        'model_ema': model_ema,
        'vae': vae,
        'train_diffusion': train_diffusion,
    }
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
    models = initialize_models(config, accelerator, logger)