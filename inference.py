"""
PixCell ControlNet Inference Script

Usage:
    python inference.py \
        --model_path checkpoints/pixcell_controlnet_full/model_step_25000.pt \
        --config_path configs/config_controlnet_gan.py \
        --vae_path pretrained_models/sd-3.5-vae/vae \
        --uni_feature features/sample_0_uni.npy \
        --cell_mask masks/sample_0_mask.png \
        --output generated.png
"""

import argparse
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from diffusers import DDPMScheduler, DDIMScheduler, AutoencoderKL
from torchvision.utils import save_image
import torch.nn.functional as F
from tqdm import tqdm
import sys


def load_config(config_path):
    """Load config from Python file."""
    import importlib.util
    
    spec = importlib.util.spec_from_file_location("config", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    
    # Create a simple namespace with config values
    class Config:
        pass
    
    config = Config()
    for key in dir(config_module):
        if not key.startswith('_'):
            setattr(config, key, getattr(config_module, key))
    
    return config


def load_models(model_path, vae_path, config_path=None, device='cuda'):
    """Load PixCell ControlNet and VAE with debugging."""
    
    print("Loading models...")
    print("="*70)
    
    # Load config
    if config_path:
        print(f"Loading config from {config_path}")
        try:
            config = load_config(config_path)
            print(f"✓ Config loaded")
        except Exception as e:
            print(f"❌ Config loading failed: {e}")
            config = None
    else:
        config = None
    
    # Load VAE
    print(f"\nLoading VAE from {vae_path}")
    try:
        vae = AutoencoderKL.from_pretrained(vae_path, torch_dtype=torch.float16)
        # DELETE THE ENCODER - You only need the decoder for generation
        if hasattr(vae, 'encoder'):
            del vae.encoder
            # This is a bit of a hack for diffusers, but it prevents 
            # the model from trying to access the encoder later.
            vae.encoder = None 
            
        vae.to(device)
        vae.eval()
        torch.cuda.empty_cache() # Flush the freed memory
        print(f"✓ VAE loaded")
    except Exception as e:
        print(f"❌ VAE loading failed: {e}")
        raise
    
    # Load checkpoint
    print(f"\nLoading checkpoint from {model_path}")
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        print(f"✓ Checkpoint loaded")
    except Exception as e:
        print(f"❌ Checkpoint loading failed: {e}")
        raise
    
    # Extract state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    del checkpoint 
    import gc
    gc.collect()
    print(f"State dict has {len(state_dict)} keys")
    
    # Detect ControlNet depth
    print("\nDetecting model configuration...")
    controlnet_block_indices = set()
    for key in state_dict.keys():
        if 'controlnet.control_blocks.' in key:
            parts = key.split('.')
            try:
                idx = int(parts[2])
                controlnet_block_indices.add(idx)
            except (ValueError, IndexError):
                continue
    
    controlnet_depth = max(controlnet_block_indices) + 1 if controlnet_block_indices else None
    
    # Detect control channels
    control_x_key = 'controlnet.control_x_embedder.proj.weight'
    if control_x_key in state_dict:
        control_channels = state_dict[control_x_key].shape[1]
    else:
        control_channels = 1
    
    print(f"  ControlNet depth: {controlnet_depth}")
    print(f"  Control channels: {control_channels}")
    
    # Import model - THIS IS WHERE IT MIGHT HANG
    print("\nImporting model class...")
    try:
        from diffusion.model.nets.PixArtControlNet import PixArt_UNI_ControlNet
        print("✓ Model class imported")
    except Exception as e:
        print(f"❌ Model import failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # Get config values
    if config:
        model_max_length = getattr(config, 'model_max_length', 1)
        image_size = getattr(config, 'image_size', 256)
    else:
        model_max_length = 1
        image_size = 256
    
    input_size = image_size // 8
    
    print(f"\nInitializing model:")
    print(f"  Image size: {image_size}")
    print(f"  Latent size: {input_size}")
    print(f"  Model max length: {model_max_length}")
    print(f"  Control channels: {control_channels}")
    print(f"  ControlNet depth: {controlnet_depth}")
    
    # Initialize model - THIS IS WHERE IT'S HANGING
    print("\nCreating model instance...")
    sys.stdout.flush()  # Force print to show
    
    try:
        with torch.device(device):
            model = PixArt_UNI_ControlNet(
                input_size=input_size,
                patch_size=2,
                in_channels=16,
                control_channels=control_channels,
                hidden_size=1152,
                depth=28,
                controlnet_depth=controlnet_depth,
                num_heads=16,
                mlp_ratio=4.0,
                class_dropout_prob=0.1,
                caption_channels=1536,
                model_max_length=model_max_length,
                pred_sigma=True,
                qk_norm=False,
                drop_path=0.0,
                freeze_base=False,
            ).to(torch.float16)
            print("✓ Model instance created")
    except Exception as e:
        print(f"❌ Model initialization failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # Load weights
    print("\nLoading weights into model...")
    sys.stdout.flush()
    
    try:
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        print(f"✓ Weights loaded")
        print(f"  Missing keys: {len(missing_keys)}")
        print(f"  Unexpected keys: {len(unexpected_keys)}")
    except Exception as e:
        print(f"❌ Weight loading failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    print("\nMoving model to device...")
    sys.stdout.flush()
    
    model.to(device)
    model.eval()
    
    print(f"✓ Model ready on {device}")
    print("="*70)
    
    return model, vae, config

@torch.no_grad()
def generate_image(
    model,
    vae,
    uni_feature,
    cell_mask,
    config=None,
    num_inference_steps=50,
    seed=None,
    device='cuda',
    scheduler=None,
    verbose=True,
):
    """Generate image using PixCell ControlNet."""
    
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    model.eval()
    vae.eval()
    
    # Get scale/shift factors from config
    if config:
        scale_factor = getattr(config, 'scale_factor', 1.5305)
        shift_factor = getattr(config, 'shift_factor', 0.0609)
    else:
        scale_factor = vae.config.scaling_factor if hasattr(vae.config, 'scaling_factor') else 1.5305
        shift_factor = 0.0609
    
    # 1. Process UNI feature
    if isinstance(uni_feature, (str, Path)):
        uni_feature = torch.from_numpy(np.load(uni_feature))
    
    if uni_feature.dim() == 1:
        y = uni_feature.view(1, 1, 1, 1536).to(device)
    else:
        y = uni_feature.to(device)
    y /= np.sqrt(y.shape[-1])
    # 2. Process cell mask
    if isinstance(cell_mask, (str, Path)):
        mask_image = Image.open(cell_mask).convert('L')
        cell_mask = torch.from_numpy(np.array(mask_image)).float() / 255.0
        cell_mask = cell_mask.unsqueeze(0).unsqueeze(0)
    
    if cell_mask.dim() == 2:
        cell_mask = cell_mask.unsqueeze(0).unsqueeze(0)
    elif cell_mask.dim() == 3:
        cell_mask = cell_mask.unsqueeze(0)
    
    # Resize to 256x256 if needed
    if cell_mask.shape[-1] != 256:
        cell_mask = F.interpolate(cell_mask, size=(256, 256), mode='nearest')
    
    # Downsample to latent resolution (32x32)
    cell_mask_latent = F.max_pool2d(cell_mask, kernel_size=8, stride=8).to(device)
    
    # Match expected channels
    expected_channels = model.controlnet.control_x_embedder.proj.weight.shape[1]
    if cell_mask_latent.shape[1] != expected_channels:
        if expected_channels == 4:
            cell_mask_latent = cell_mask_latent.repeat(1, 4, 1, 1)
    scheduler.set_timesteps(num_inference_steps)
    
    # 4. Initialize latents
    latents = torch.randn(1, 16, 32, 32, device=device, dtype=model.dtype)
    latents = latents * scheduler.init_noise_sigma
    
    # 5. Denoising loop
    iterator = tqdm(scheduler.timesteps, desc="Denoising") if verbose else scheduler.timesteps
    for t in iterator:
        timestep = torch.tensor([t], device=device)
        noise_pred = model(
            latents,
            timestep,
            y=y,
            control_input=cell_mask_latent,
            mask=None,
            data_info={'mask_type': ['null']}
        )
        
        if model.pred_sigma:
            noise_pred, _ = noise_pred.chunk(2, dim=1)
        
        # Denoise
        latents = scheduler.step(noise_pred, t, latents).prev_sample
    
    # 6. Decode with proper scaling
    vae_dtype = next(vae.parameters()).dtype
    
    # Apply shift and scale
    latents_shifted = (latents / scale_factor) + shift_factor
    
    image = vae.decode(latents_shifted.to(vae_dtype)).sample
    
    # 7. Post-process
    image = (image + 1) / 2
    image = torch.clamp(image, 0, 1)
    
    return image.squeeze(0)


def parse_args(args_list=None):
    """
    Args:
        args_list: List of arguments (for programmatic use) or None (for CLI use)
    """
    parser = argparse.ArgumentParser(
        description='PixCell ControlNet Inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required paths
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--vae_path', type=str, required=True)
    parser.add_argument('--config_path', type=str, default=None)

    # Single image mode
    parser.add_argument('--uni_feature', type=str, default=None)
    parser.add_argument('--cell_mask', type=str, default=None)
    parser.add_argument('--output', type=str, default='generated.png')

    # Batch mode
    parser.add_argument('--batch', action='store_true')
    parser.add_argument('--uni_dir', type=str, default=None)
    parser.add_argument('--mask_dir', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--uni_suffix', type=str, default='_uni.npy')
    parser.add_argument('--mask_suffix', type=str, default='_mask.png')

    # Generation parameters
    parser.add_argument('--num_steps', type=int, default=50)
    parser.add_argument('--scheduler', type=str, default='ddim')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--device', type=str, default='cuda')

    return parser.parse_args(args_list)

def main(args_list=None):
    args = parse_args(args_list)
    
    # Validate
    if not args.batch:
        if args.uni_feature is None or args.cell_mask is None:
            parser.error("--uni_feature and --cell_mask required")
    else:
        if args.uni_dir is None or args.mask_dir is None:
            parser.error("--uni_dir and --mask_dir required for batch mode")
    
    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, using CPU")
        args.device = 'cpu'
    
    # Load models
    model, vae, config = load_models(
        args.model_path, 
        args.vae_path,
        args.config_path,
        args.device
    )
    
    # Single image generation
    if not args.batch:
        print("\n" + "="*70)
        print("GENERATING IMAGE")
        print("="*70)
        print(f"UNI feature: {args.uni_feature}")
        print(f"Cell mask:   {args.cell_mask}")
        print(f"Output:      {args.output}")
        print(f"Steps:       {args.num_steps}")
        print(f"Scheduler:   {args.scheduler}")
        print("="*70)
        
        image = generate_image(
            model=model,
            vae=vae,
            uni_feature=args.uni_feature,
            cell_mask=args.cell_mask,
            config=config,
            num_inference_steps=args.num_steps,
            seed=args.seed,
            device=args.device,
            scheduler=args.scheduler,
            verbose=True,
        )
        
        save_image(image, args.output)
        print(f"\n✓ Generated image saved to: {args.output}")
    
    # Batch generation
    else:
        print("\n" + "="*70)
        print("BATCH GENERATION")
        print("="*70)
        
        uni_dir = Path(args.uni_dir)
        mask_dir = Path(args.mask_dir)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        uni_files = sorted(uni_dir.glob(f"*{args.uni_suffix}"))
        
        print(f"UNI directory:  {uni_dir}")
        print(f"Mask directory: {mask_dir}")
        print(f"Output directory: {output_dir}")
        print(f"Found {len(uni_files)} UNI features")
        print("="*70)
        
        if len(uni_files) == 0:
            print(f"❌ No files found with suffix '{args.uni_suffix}'")
            return
        
        successful = 0
        failed = 0
        
        for uni_path in tqdm(uni_files, desc="Generating"):
            base_name = uni_path.stem.replace(args.uni_suffix.replace('.npy', ''), '')
            mask_name = base_name + args.mask_suffix
            mask_path = mask_dir / mask_name
            
            if not mask_path.exists():
                print(f"\n⚠️  Mask not found: {mask_path}")
                failed += 1
                continue
            
            output_path = output_dir / f"{base_name}_generated.png"
            
            try:
                image = generate_image(
                    model=model,
                    vae=vae,
                    uni_feature=str(uni_path),
                    cell_mask=str(mask_path),
                    config=config,
                    num_inference_steps=args.num_steps,
                    seed=args.seed,
                    device=args.device,
                    scheduler=args.scheduler,
                    verbose=False,
                )
                
                save_image(image, str(output_path))
                successful += 1
                
            except Exception as e:
                print(f"\n❌ Error: {base_name}: {e}")
                import traceback
                traceback.print_exc()
                failed += 1
        
        print("\n" + "="*70)
        print("BATCH COMPLETE")
        print("="*70)
        print(f"Successful: {successful}/{len(uni_files)}")
        print(f"Failed:     {failed}/{len(uni_files)}")
        print(f"Output:     {output_dir}")
        print("="*70)


if __name__ == "__main__":
    main()