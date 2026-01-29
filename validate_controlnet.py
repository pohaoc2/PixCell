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


class PixCellValidator:
    """Validator for PixCell (Vanilla, not ControlNet)"""
    
    def __init__(self, model_path="StonyBrook-CVLab/PixCell-256", config_path=None, device='cuda'):
        """
        Initialize validator
        
        Args:
            model_path: Path or HF repo for PixCell model
            config_path: Optional config file (not needed if loading from HF)
            device: Device to run on
        """
        from diffusers import DPMSolverMultistepScheduler

        # Add the transformer folder to the path
        model_dir = os.path.join(os.getcwd(), "pretrained_models/pixcell-256/transformer")
        if model_dir not in sys.path:
            sys.path.append(model_dir)

        # Now you can import the file name directly
        from pixcell_transformer_2d import PixCellTransformer2DModel
        self.device = device
        
        # Load VAE
        print("Loading VAE...")
        self.vae = AutoencoderKL.from_pretrained(
            "stabilityai/stable-diffusion-3.5-large",
            subfolder="vae",
            torch_dtype=torch.float32  # Use float32 for stability
        ).to(device)
        self.vae.eval()
        
        # Load PixCell Model
        print("Loading PixCell model...")
        self.model = PixCellTransformer2DModel.from_pretrained(
            model_path,
            subfolder="transformer"
        ).to(device)
        self.model.eval()
        
        # Load Scheduler
        print("Loading scheduler...")
        self.scheduler = DPMSolverMultistepScheduler.from_pretrained(
            model_path,
            subfolder="scheduler"
        )
        
        # Get config from model
        self.config = type('Config', (), {
            'image_size': 256,
            'scale_factor': self.vae.config.scaling_factor,
            'shift_factor': getattr(self.vae.config, 'shift_factor', 0.0)
        })()
        
        self.scale_factor = self.config.scale_factor
        self.shift_factor = self.config.shift_factor
        
        print("✓ Model loaded successfully")
    
    @torch.no_grad()
    def generate(self, uni_embedding, num_inference_steps=14, guidance_scale=4.5, mask=None):
        """
        Generate image from UNI embedding
        
        Args:
            uni_embedding: UNI embedding (1024,) or (1, 1024)
            num_inference_steps: Number of denoising steps
            guidance_scale: CFG guidance scale
            mask: Optional (ignored for vanilla PixCell)
            
        Returns:
            Generated image (H, W, 3) as numpy array [0, 255]
        """
        # Prepare UNI embedding
        if uni_embedding.ndim == 1:
            uni_embedding = uni_embedding.unsqueeze(0)
        
        # Check if we need to expand to match caption_num_tokens
        if hasattr(self.model.config, 'caption_num_tokens'):
            num_tokens = self.model.config.caption_num_tokens
            if uni_embedding.ndim == 2:
                # Expand (1, 1536) -> (1, num_tokens, 1536)
                uni_embedding = uni_embedding.unsqueeze(1).repeat(1, num_tokens, 1)
        
        uni_embedding = uni_embedding.to(self.device)
        
        # Initial noise - use model's in_channels (should be 16 for PixCell)
        latent_size = self.config.image_size // 8
        latent_channels = self.model.config.in_channels
        
        z = torch.randn(1, latent_channels, latent_size, latent_size, device=self.device)
        z = z * self.scheduler.init_noise_sigma
        
        # Get unconditional embedding for CFG
        if guidance_scale > 1.0:
            null_y = self.model.caption_projection.uncond_embedding.clone().unsqueeze(0)
            null_y = null_y.to(device=uni_embedding.device, dtype=uni_embedding.dtype)
        else:
            null_y = None
        
        # Setup scheduler
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps
        
        # Denoising loop
        latent = z
        for i, t in enumerate(timesteps):
            # Expand latent for CFG
            if guidance_scale > 1.0:
                latent_model_input = torch.cat([latent] * 2)
                encoder_hidden_states = torch.cat([null_y, uni_embedding], dim=0)
            else:
                latent_model_input = latent
                encoder_hidden_states = uni_embedding
            
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            
            # Prepare timestep
            timestep = t
            if not torch.is_tensor(timestep):
                timestep = torch.tensor([timestep], dtype=torch.float32, device=self.device)
            timestep = timestep.expand(latent_model_input.shape[0])
            
            # Predict noise
            noise_pred = self.model(
                latent_model_input,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                return_dict=False,
            )[0]
            
            # Perform CFG
            if guidance_scale > 1.0:
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            
            # Handle learned sigma (if model predicts variance)
            if self.model.config.out_channels // 2 == latent_channels:
                noise_pred = noise_pred.chunk(2, dim=1)[0]
            
            # Denoise step
            latent = self.scheduler.step(noise_pred, t, latent, return_dict=False)[0]
        
        # Decode with PixCell's 16-channel VAE
        latent = latent.float()
        latent = (latent / self.scale_factor) + self.shift_factor
        
        image = self.vae.decode(latent).sample
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


def main():
    parser = argparse.ArgumentParser(description='Validate PixCell (Vanilla)')
    parser.add_argument('--model-path', type=str, 
                       default='StonyBrook-CVLab/PixCell-256',
                       help='HuggingFace model path or local checkpoint')
    parser.add_argument('--test-data-dir', type=str, default='./patches',
                       help='Directory with test images')
    parser.add_argument('--features-dir', type=str, default='./features',
                       help='Directory with precomputed UNI features')
    parser.add_argument('--output-dir', type=str, default='./validation_results',
                       help='Output directory for results')
    parser.add_argument('--num-samples', type=int, default=20,
                       help='Number of samples to validate')
    parser.add_argument('--num-inference-steps', type=int, default=20,
                       help='Number of denoising steps')
    parser.add_argument('--guidance-scale', type=float, default=1.5,
                       help='CFG guidance scale (1.0 = no guidance)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("PixCell Validation (Vanilla - UNI-conditioned generation)")
    print("="*70)
    print(f"Model: {args.model_path}")
    print(f"Inference steps: {args.num_inference_steps}")
    print(f"Guidance scale: {args.guidance_scale}")
    print(f"Device: {args.device}")
    print("="*70)
    
    device = args.device if torch.cuda.is_available() else 'cpu'
    
    # Initialize validator
    print("\nInitializing model...")
    validator = PixCellValidator(
        model_path=args.model_path,
        device=device
    )
    
    # Get test images
    test_images = sorted(list(Path(args.test_data_dir).glob('*.png')))[:args.num_samples]
    
    if len(test_images) == 0:
        print(f"❌ No images found in {args.test_data_dir}")
        return
    
    print(f"\nValidating on {len(test_images)} samples...")
    print("="*70)
    
    all_metrics = []
    successful_generations = 0
    
    for idx, img_path in enumerate(tqdm(test_images, desc="Generating")):
        base_name = img_path.stem
        
        try:
            # Load real image
            real_image = Image.open(img_path).convert('RGB')
            # Resize to match generated size (256x256)
            real_image = real_image.resize((256, 256), Image.LANCZOS)
            real_image_np = np.array(real_image)
            
            # Load UNI embedding
            uni_path = Path(args.features_dir) / f"{base_name}_uni.npy"
            if not uni_path.exists():
                # Try alternative naming
                uni_path = Path(args.features_dir) / f"{base_name}.npy"
            
            if not uni_path.exists():
                print(f"⚠ UNI embedding not found for {base_name}, skipping")
                continue
            
            uni_embedding = torch.from_numpy(np.load(uni_path))
            
            # Generate image from UNI embedding only (no mask)
            generated_image = validator.generate(
                uni_embedding=uni_embedding,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
            )
            
            # Save individual results
            Image.fromarray(generated_image).save(
                output_dir / f"{base_name}_generated.png"
            )
            
            # Create comparison (without mask overlay for vanilla PixCell)
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            axes[0].imshow(real_image_np)
            axes[0].set_title('Real Image', fontsize=14)
            axes[0].axis('off')
            
            axes[1].imshow(generated_image)
            axes[1].set_title('Generated Image', fontsize=14)
            axes[1].axis('off')
            
            # Difference map
            diff = np.abs(real_image_np.astype(float) - generated_image.astype(float)).mean(axis=2)
            im = axes[2].imshow(diff, cmap='hot')
            axes[2].set_title('Difference Map', fontsize=14)
            axes[2].axis('off')
            plt.colorbar(im, ax=axes[2])
            
            plt.tight_layout()
            plt.savefig(output_dir / f"{base_name}_comparison.png", dpi=150, bbox_inches='tight')
            plt.close('all')
            
            successful_generations += 1
            
        except Exception as e:
            print(f"❌ Error processing {base_name}: {str(e)}")
            continue


if __name__ == "__main__":
    main()