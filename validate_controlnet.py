"""
Validation and Visualization Script for PixCell ControlNet

This script:
1. Loads trained ControlNet model
2. Generates images from test masks + UNI embeddings
3. Visualizes results with mask overlays
4. Computes quantitative metrics
5. Creates comparison grids

Usage:
    python validate_controlnet.py \
        --checkpoint ./checkpoints/pixcell_controlnet/epoch_25_step_5000.pth \
        --config config_controlnet_full.py \
        --output-dir ./validation_results \
        --num-samples 20
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm

# Add parent directory to path
current_file_path = Path(__file__).resolve()
sys.path.insert(0, str(current_file_path.parent.parent))

from diffusion import IDDPM, DPMS
from diffusion.model.builder import build_model
from diffusion.utils.checkpoint import load_checkpoint
from diffusion.utils.misc import read_config
from diffusers.models import AutoencoderKL


class ControlNetValidator:
    """Validator for PixCell ControlNet"""
    
    def __init__(self, checkpoint_path, config_path, device='cuda'):
        """
        Initialize validator
        
        Args:
            checkpoint_path: Path to trained model checkpoint
            config_path: Path to config file
            device: Device to run on
        """
        self.device = device
        self.config = read_config(config_path)
        
        # Load VAE
        print("Loading VAE...")
        self.vae = AutoencoderKL.from_pretrained(
            self.config.vae_pretrained, 
            torch_dtype=torch.float16
        ).to(device)
        self.vae.eval()
        
        self.scale_factor = self.config.scale_factor
        self.shift_factor = getattr(self.config, 'shift_factor', 0.0)
        
        # Build model
        print("Building model...")
        image_size = self.config.image_size
        latent_size = image_size // 8
        
        model_kwargs = {
            "pe_interpolation": self.config.pe_interpolation,
            "config": self.config,
            "model_max_length": self.config.model_max_length,
            "qk_norm": self.config.qk_norm,
            "micro_condition": self.config.micro_condition,
        }
        
        self.model = build_model(
            self.config.model,
            grad_checkpointing=False,
            fp32_attention=False,
            input_size=latent_size,
            **model_kwargs
        ).to(device)
        self.model.eval()
        
        # Load checkpoint
        print(f"Loading checkpoint: {checkpoint_path}")
        load_checkpoint(
            checkpoint_path,
            self.model,
            load_ema=True,  # Use EMA weights for best quality
            max_length=self.config.model_max_length
        )
        
        print("✓ Model loaded successfully")
    
    @torch.no_grad()
    def generate(self, mask, uni_embedding, num_inference_steps=14, guidance_scale=4.5):
        """
        Generate image from mask and UNI embedding
        
        Args:
            mask: Cell mask (1, 1, H, W) or (H, W)
            uni_embedding: UNI embedding (1024,) or (1, 1024)
            num_inference_steps: Number of denoising steps
            guidance_scale: CFG guidance scale
            
        Returns:
            Generated image (H, W, 3) as numpy array [0, 255]
        """
        # Prepare inputs
        if mask.ndim == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.ndim == 3:
            mask = mask.unsqueeze(0)
        mask = mask.to(self.device).float()
        
        if uni_embedding.ndim == 1:
            uni_embedding = uni_embedding.unsqueeze(0).unsqueeze(0)
        elif uni_embedding.ndim == 2:
            uni_embedding = uni_embedding.unsqueeze(1)
        uni_embedding = uni_embedding.to(self.device)
        
        # Prepare mask at latent resolution
        latent_size = self.config.image_size // 8
        if mask.shape[-1] != latent_size:
            mask_latent = F.max_pool2d(
                mask, 
                kernel_size=mask.shape[-1] // latent_size,
                stride=mask.shape[-1] // latent_size
            )
        else:
            mask_latent = mask
        
        # Initial noise
        z = torch.randn(1, 16, latent_size, latent_size, device=self.device)
        
        # Prepare model kwargs
        hw = torch.tensor([[self.config.image_size, self.config.image_size]], 
                         dtype=torch.float, device=self.device)
        ar = torch.tensor([[1.0]], device=self.device)
        
        model_kwargs = {
            'data_info': {'img_hw': hw, 'aspect_ratio': ar},
            'mask': None,
            'controlnet_cond': mask_latent
        }
        
        # Null conditioning for CFG
        null_y = torch.zeros_like(uni_embedding)
        
        # Sampling
        sampler = DPMS(
            self.model.forward_with_dpmsolver,
            condition=uni_embedding,
            uncondition=null_y,
            cfg_scale=guidance_scale,
            model_kwargs=model_kwargs
        )
        
        latent = sampler.sample(
            z,
            steps=num_inference_steps,
            order=2,
            skip_type="time_uniform",
            method="multistep",
        )
        
        # Decode
        latent = latent / self.scale_factor
        if self.shift_factor != 0:
            latent = latent + self.shift_factor
        
        image = self.vae.decode(latent.to(torch.float16)).sample
        image = torch.clamp(127.5 * image + 128.0, 0, 255)
        image = image.permute(0, 2, 3, 1).cpu().numpy()[0].astype(np.uint8)
        
        return image
    
    def overlay_mask(self, image, mask, alpha=0.3, color=(255, 0, 0)):
        """
        Overlay mask on image
        
        Args:
            image: RGB image (H, W, 3)
            mask: Binary mask (H, W) or (1, H, W)
            alpha: Overlay transparency
            color: Mask color (R, G, B)
            
        Returns:
            Image with mask overlay
        """
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        
        if mask.ndim == 3:
            mask = mask[0]
        
        # Resize mask to image size if needed
        if mask.shape != image.shape[:2]:
            from scipy.ndimage import zoom
            factors = (image.shape[0] / mask.shape[0], image.shape[1] / mask.shape[1])
            mask = zoom(mask, factors, order=0)
        
        # Create overlay
        overlay = image.copy()
        overlay[mask > 0.5] = (
            (1 - alpha) * image[mask > 0.5] + 
            alpha * np.array(color)
        ).astype(np.uint8)
        
        return overlay
    
    def create_comparison_grid(self, real_image, generated_image, mask, save_path=None):
        """
        Create comparison visualization
        
        Args:
            real_image: Ground truth image (H, W, 3)
            generated_image: Generated image (H, W, 3)
            mask: Input mask (H, W)
            save_path: Optional path to save figure
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Row 1: Real
        axes[0, 0].imshow(real_image)
        axes[0, 0].set_title('Real Image', fontsize=14)
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(self.overlay_mask(real_image, mask))
        axes[0, 1].set_title('Real + Mask Overlay', fontsize=14)
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(mask, cmap='gray')
        axes[0, 2].set_title('Input Mask', fontsize=14)
        axes[0, 2].axis('off')
        
        # Row 2: Generated
        axes[1, 0].imshow(generated_image)
        axes[1, 0].set_title('Generated Image', fontsize=14)
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(self.overlay_mask(generated_image, mask))
        axes[1, 1].set_title('Generated + Mask Overlay', fontsize=14)
        axes[1, 1].axis('off')
        
        # Difference map
        diff = np.abs(real_image.astype(float) - generated_image.astype(float)).mean(axis=2)
        im = axes[1, 2].imshow(diff, cmap='hot')
        axes[1, 2].set_title('Difference Map', fontsize=14)
        axes[1, 2].axis('off')
        plt.colorbar(im, ax=axes[1, 2])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved: {save_path}")
        
        return fig
    
    def compute_metrics(self, real_image, generated_image, mask):
        """
        Compute quantitative metrics
        
        Args:
            real_image: Ground truth (H, W, 3)
            generated_image: Generated (H, W, 3)
            mask: Input mask (H, W)
            
        Returns:
            Dictionary of metrics
        """
        from skimage.metrics import structural_similarity as ssim
        from skimage.metrics import peak_signal_noise_ratio as psnr
        
        # Convert to grayscale for some metrics
        real_gray = real_image.mean(axis=2)
        gen_gray = generated_image.mean(axis=2)
        
        metrics = {}
        
        # PSNR
        metrics['psnr'] = psnr(real_image, generated_image, data_range=255)
        
        # SSIM
        metrics['ssim'] = ssim(real_image, generated_image, 
                              channel_axis=2, data_range=255)
        
        # MSE
        metrics['mse'] = ((real_image - generated_image) ** 2).mean()
        
        # Mask coverage (how much of mask region looks realistic)
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        if mask.ndim == 3:
            mask = mask[0]
        
        # Resize mask if needed
        if mask.shape != real_image.shape[:2]:
            from scipy.ndimage import zoom
            factors = (real_image.shape[0] / mask.shape[0], real_image.shape[1] / mask.shape[1])
            mask = zoom(mask, factors, order=0)
        
        mask_bool = mask > 0.5
        if mask_bool.any():
            metrics['mse_in_mask'] = (
                (real_image[mask_bool] - generated_image[mask_bool]) ** 2
            ).mean()
        
        return metrics


def main():
    parser = argparse.ArgumentParser(description='Validate PixCell ControlNet')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('--test-data-dir', type=str, default='./patches',
                       help='Directory with test images')
    parser.add_argument('--features-dir', type=str, default='./features',
                       help='Directory with precomputed features')
    parser.add_argument('--masks-dir', type=str, default='./masks',
                       help='Directory with cell masks')
    parser.add_argument('--output-dir', type=str, default='./validation_results',
                       help='Output directory for results')
    parser.add_argument('--num-samples', type=int, default=20,
                       help='Number of samples to validate')
    parser.add_argument('--num-inference-steps', type=int, default=14,
                       help='Number of denoising steps')
    parser.add_argument('--guidance-scale', type=float, default=4.5,
                       help='CFG guidance scale')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("PixCell ControlNet Validation")
    print("="*70)
    
    # Initialize validator
    validator = ControlNetValidator(
        args.checkpoint,
        args.config,
        device=args.device
    )
    
    # Get test images
    test_images = list(Path(args.test_data_dir).glob('*.png'))[:args.num_samples]
    
    if len(test_images) == 0:
        print(f"❌ No images found in {args.test_data_dir}")
        return
    
    print(f"\nValidating on {len(test_images)} samples...")
    print("="*70)
    
    all_metrics = []
    
    for idx, img_path in enumerate(tqdm(test_images, desc="Generating")):
        base_name = img_path.stem
        
        # Load real image
        real_image = Image.open(img_path).convert('RGB')
        # Resize to match generated size (256x256)
        real_image = real_image.resize((256, 256), Image.LANCZOS)
        real_image = np.array(real_image)
        
        # Load mask
        mask_path = Path(args.masks_dir) / f"{base_name}_cellvit_mask.png"
        if not mask_path.exists():
            mask_path = Path(args.masks_dir) / f"{base_name}_mask.png"
        
        if not mask_path.exists():
            print(f"⚠ Mask not found for {base_name}, skipping")
            continue
        
        mask = np.array(Image.open(mask_path).convert('L')) > 127
        mask = torch.from_numpy(mask.astype(np.float32))
        
        # Load UNI embedding
        uni_path = Path(args.features_dir) / f"{base_name}_uni.npy"
        if not uni_path.exists():
            print(f"⚠ UNI embedding not found for {base_name}, skipping")
            continue
        
        uni_embedding = torch.from_numpy(np.load(uni_path))
        
        # Generate
        generated_image = validator.generate(
            mask,
            uni_embedding,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale
        )
        
        # Save individual results
        Image.fromarray(generated_image).save(
            output_dir / f"{base_name}_generated.png"
        )
        
        # Create comparison
        validator.create_comparison_grid(
            real_image,
            generated_image,
            mask,
            save_path=output_dir / f"{base_name}_comparison.png"
        )
        plt.close('all')
        
        # Compute metrics
        metrics = validator.compute_metrics(real_image, generated_image, mask)
        metrics['image_name'] = base_name
        all_metrics.append(metrics)
    
    # Aggregate metrics
    print("\n" + "="*70)
    print("Quantitative Results")
    print("="*70)
    
    avg_metrics = {}
    for key in ['psnr', 'ssim', 'mse', 'mse_in_mask']:
        values = [m[key] for m in all_metrics if key in m]
        if values:
            avg_metrics[key] = np.mean(values)
            print(f"{key.upper():15s}: {avg_metrics[key]:.4f} ± {np.std(values):.4f}")
    
    # Save metrics
    import json
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump({
            'individual': all_metrics,
            'average': avg_metrics
        }, f, indent=2)
    
    print(f"\n✓ Validation complete!")
    print(f"  Results saved to: {output_dir}")
    print(f"  - Individual images: *_generated.png")
    print(f"  - Comparisons: *_comparison.png")
    print(f"  - Metrics: metrics.json")


if __name__ == "__main__":
    main()
