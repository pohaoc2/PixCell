"""
Feature extraction script for PixCell ControlNet training
Extracts:
1. VAE features using Stable Diffusion 3.5 VAE
2. UNI-2h embeddings (histopathology foundation model)
"""
import argparse
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from types import SimpleNamespace
from tqdm import tqdm
import timm
from diffusers.models import AutoencoderKL
from torchvision import transforms
from timm.data.transforms_factory import create_transform
from timm.data import resolve_data_config
import os
from huggingface_hub import login
import cv2

from pipeline.patch_extractors import (
    extract_ctranspath_patches,
    extract_resnet50_patches,
    extract_uni_patches,
    extract_virchow_patches,
)


def _ensure_hf_auth(token=None):
    """Avoid redundant hub login when HF_TOKEN is already active in the environment."""
    if token:
        login(token=token)
        return token

    env_token = os.getenv("HF_TOKEN")
    if env_token:
        return env_token

    return None


class UNI2hExtractor:
    """
    UNI-2h feature extractor for histopathology images
    
    UNI-2h is a vision transformer trained on 100M+ histopathology images
    Output: 1024-dimensional feature vector
    """
    def __init__(self, model_path, device='cuda', token=None):
        """
        Initialize UNI-2h model
        
        Args:
            model_path: Path to UNI-2h model directory
            device: Device to run model on
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        print(f"Loading UNI-2h model from {model_path}...")
        token = _ensure_hf_auth(token)
        # Load UNI-2h model (ViT architecture)
        # The model is a Vision Transformer pretrained on histopathology
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
        #uni_model = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)
        self.model = timm.create_model("vit_giant_patch14_224", pretrained=False, **timm_kwargs)
        
        # Load pretrained weights
        model_path = Path(model_path)
        weight_files = list(model_path.glob("*.pth")) + list(model_path.glob("pytorch_model.bin"))
        
        if weight_files:
            checkpoint = torch.load(weight_files[0], map_location='cpu')
            
            # Handle different checkpoint formats
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Remove 'module.' prefix if present
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            
            self.model.load_state_dict(state_dict, strict=False)
            print(f"✓ Loaded weights from {weight_files[0].name}")
        else:
            print("⚠ No weight files found, using random initialization")
        
        self.model = self.model.to(device)
        self.model.eval()
        
        # UNI-2h preprocessing (ImageNet normalization, 224x224 input)

        self.transform = create_transform(
            **resolve_data_config(self.model.pretrained_cfg, model=self.model)
        )
        print(f"✓ UNI-2h model ready on {device}")
    
    @torch.no_grad()
    def extract(self, image):
        """
        Extract UNI-2h features from an image
        
        Args:
            image: PIL Image or numpy array (H, W, 3)
            
        Returns:
            numpy array of shape (1024,) - UNI-2h embedding
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Preprocess
        x = self.transform(image).unsqueeze(0).to(self.device)
        
        # Extract features
        features = self.model(x)
        
        return features.cpu().numpy()[0]
    
    @torch.no_grad()
    def extract_batch(self, images):
        """
        Extract UNI-2h features from a batch of images

        Args:
            images: List of PIL Images or numpy arrays

        Returns:
            numpy array of shape (N, 1024)
        """
        # Preprocess batch
        x = torch.stack([self.transform(img if isinstance(img, Image.Image) else Image.fromarray(img))
                        for img in images]).to(self.device)

        # Extract features
        features = self.model(x)

        return features.cpu().numpy()

    @torch.no_grad()
    def extract_patch_tokens(self, image):
        """Return per-patch tokens for one image as (P, D) float16.

        Drops CLS + register tokens; keeps only the patch-grid tokens. With
        img_size=224 and patch_size=14, P = (224/14)^2 = 256 and D = 1536.
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        x = self.transform(image).unsqueeze(0).to(self.device)
        return self._patch_tokens_from_batch(x)[0]

    @torch.no_grad()
    def extract_patch_tokens_batch(self, images):
        """Per-patch tokens for a batch as (N, P, D) float16."""
        x = torch.stack([
            self.transform(img if isinstance(img, Image.Image) else Image.fromarray(img))
            for img in images
        ]).to(self.device)
        return self._patch_tokens_from_batch(x)

    @torch.no_grad()
    def _patch_tokens_from_batch(self, x):
        """Run forward_features and strip CLS + register tokens."""
        tokens = self.model.forward_features(x)
        if tokens.ndim != 3:
            raise RuntimeError(f"expected (N,T,D) tokens, got shape {tuple(tokens.shape)}")
        n_prefix = int(getattr(self.model, "num_prefix_tokens", 1))
        patch_tokens = tokens[:, n_prefix:, :]
        # vit_giant_patch14_224 with img_size=224 -> 16x16=256 patches
        expected = (224 // 14) ** 2
        if patch_tokens.shape[1] != expected:
            raise RuntimeError(
                f"unexpected patch count {patch_tokens.shape[1]}; expected {expected}"
            )
        return patch_tokens.to(torch.float16).cpu().numpy()


class SD3VAEExtractor:
    """
    Stable Diffusion 3.5 VAE feature extractor
    
    Extracts latent representations (mean and std) for diffusion training
    Output: (2, 16, 32, 32) for 256x256 input
    """
    def __init__(self, model_path, device='cuda', dtype=torch.float16, size=256, token=None):
        """
        Initialize SD3.5 VAE model
        
        Args:
            model_path: Path to SD3.5 model directory
            device: Device to run model on
            dtype: Data type for model
        """
        self.device = device
        self.dtype = dtype
        self.size = size
        print(f"Loading SD3.5 VAE from {model_path}...")
        token = _ensure_hf_auth(token)
        # Load VAE
        model_path = Path(model_path)
        
        # Try different paths
        vae_paths = [
            model_path,  # Direct path
            model_path / "vae",  # Subfolder
        ]
        
        vae_loaded = False
        for vae_path in vae_paths:
            if vae_path.exists():
                try:
                    self.vae = AutoencoderKL.from_pretrained(
                        str(vae_path),
                        torch_dtype=dtype,
                    ).to(device)
                    print(f"✓ Loaded VAE from {vae_path}")
                    vae_loaded = True
                    break
                except Exception as e:
                    continue
        
        if not vae_loaded:
            raise ValueError(f"Could not load VAE from {model_path}")
        
        self.vae.eval()
        
        # Get scaling factor
        self.scale_factor = self.vae.config.scaling_factor
        print(f"✓ VAE scaling factor: {self.scale_factor}")
        from torchvision import transforms as T

        self.transform = T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB")),
            T.Resize((self.size, self.size)),
            T.ToTensor(),
            T.Normalize([0.5], [0.5]),
        ]
    )
        
        print(f"✓ SD3.5 VAE ready on {device}")
    
    @torch.no_grad()
    def extract(self, image, return_latent=False):
        """
        Extract VAE features (mean and std) from an image
        
        Args:
            image: PIL Image or numpy array (H, W, 3)
            return_latent: If True, return latent; if False, return mean+std
            
        Returns:
            numpy array of shape (2, 16, 32, 32) - [mean, std]
            or (16, 32, 32) if return_latent=True
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Preprocess
        x = self.transform(image).unsqueeze(0).to(self.device).to(self.dtype)
        
        # Encode
        posterior = self.vae.encode(x).latent_dist
        
        if return_latent:
            # Return single latent sample
            z = posterior.mode()
            return z.cpu().numpy()[0]
        else:
            # Return mean and std for sampling during training
            mean = posterior.mean
            std = posterior.std
            
            # Stack mean and std
            mean_std = torch.cat([mean, std], dim=1)  # (1, 2*C, H, W)
            
            # Reshape to (1, 2, C, H, W) -> (2, C, H, W)
            mean_std = mean_std.view(1, 2, -1, mean.shape[-2], mean.shape[-1])
            
            return mean_std.cpu().numpy()[0]
    
    @torch.no_grad()
    def extract_batch(self, images, return_latent=False):
        """
        Extract VAE features from a batch of images
        
        Args:
            images: List of PIL Images or numpy arrays
            return_latent: If True, return latent; if False, return mean+std
            
        Returns:
            numpy array of shape (N, 2, 16, 32, 32) or (N, 16, 32, 32)
        """
        # Preprocess batch
        x = torch.stack([self.transform(img if isinstance(img, Image.Image) else Image.fromarray(img)) 
                        for img in images]).to(self.device).to(self.dtype)
        
        # Encode
        posterior = self.vae.encode(x).latent_dist
        
        if return_latent:
            z = posterior.mode()
            return z.cpu().numpy()
        else:
            mean = posterior.mean
            std = posterior.std
            
            # Stack and reshape
            mean_std = torch.cat([mean, std], dim=1)
            mean_std = mean_std.view(mean.shape[0], 2, -1, mean.shape[-2], mean.shape[-1])
            
            return mean_std.cpu().numpy()


def extract_features_from_images(
    image_dir,
    output_dir,
    uni_model_path,
    vae_model_path,
    batch_size=8,
    device='cuda',
    vae_prefix="sd3_vae",
    uni_prefix="uni",
    skip_uni=False,
    skip_vae=False,
    save_uni_tokens=False,
    uni_tokens_prefix="uni_tokens",
):
    """
    Extract VAE and UNI features from a directory of images
    
    Args:
        image_dir: Directory containing images
        output_dir: Directory to save features
        uni_model_path: Path to UNI-2h model
        vae_model_path: Path to SD3.5 VAE model
        batch_size: Batch size for processing
        device: Device to use
        vae_prefix: Prefix for VAE feature files
        uni_prefix: Prefix for UNI feature files
    """
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all images
    image_extensions = ['.png', '.jpg', '.jpeg']
    images = []
    for ext in image_extensions:
        images.extend(image_dir.glob(f'*{ext}'))
    images = sorted(images)
    
    print(f"\nFound {len(images)} images in {image_dir}")
    
    if len(images) == 0:
        print("❌ No images found!")
        return
    
    # Initialize extractors
    print("\nInitializing feature extractors...")
    uni_extractor = None if skip_uni and not save_uni_tokens else UNI2hExtractor(uni_model_path, device=device)
    vae_extractor = None if skip_vae else SD3VAEExtractor(vae_model_path, device=device)
    
    # Process images
    print(f"\nExtracting features (batch_size={batch_size})...")
    print("="*70)
    for i in tqdm(range(0, len(images), batch_size), desc="Processing"):
        batch_images_paths = images[i:i+batch_size]
        batch_images = []
        
        # Load images
        for img_path in batch_images_paths:
            try:
                #img = Image.open(img_path).convert('RGB')
                img = cv2.imread(str(img_path))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                batch_images.append(img)
            except Exception as e:
                print(f"⚠ Error loading {img_path.name}: {e}")
                continue
        
        if len(batch_images) == 0:
            continue
        
        # Extract features
        uni_features = None
        uni_tokens = None
        vae_features = None
        if uni_extractor is not None and not skip_uni:
            uni_features = uni_extractor.extract_batch(batch_images)
        if uni_extractor is not None and save_uni_tokens:
            uni_tokens = uni_extractor.extract_patch_tokens_batch(batch_images)
        if vae_extractor is not None:
            vae_features = vae_extractor.extract_batch(batch_images)

        # Save features
        for j, img_path in enumerate(batch_images_paths[:len(batch_images)]):
            base_name = img_path.stem

            if vae_features is not None:
                np.save(output_dir / f"{base_name}_{vae_prefix}.npy", vae_features[j])
            if uni_features is not None:
                np.save(output_dir / f"{base_name}_{uni_prefix}.npy", uni_features[j])
            if uni_tokens is not None:
                np.save(output_dir / f"{base_name}_{uni_tokens_prefix}.npy", uni_tokens[j])
            
        #except Exception as e:
        #     print(f"\n⚠ Error processing batch: {e}")
        #    continue
    
    print("\n" + "="*70)
    print(f"✓ Feature extraction complete!")
    print(f"  Output directory: {output_dir}")
    print(f"  Files per image: {base_name}_{vae_prefix}.npy, {base_name}_{uni_prefix}.npy")


def _list_image_paths(image_dir):
    image_dir = Path(image_dir)
    image_extensions = [".png", ".jpg", ".jpeg"]
    paths = []
    for ext in image_extensions:
        paths.extend(image_dir.glob(f"*{ext}"))
    return sorted(paths)


def _build_patch_extractor(
    *,
    encoder,
    uni_model_path,
    encoder_model_path,
    device,
):
    resolved_device = device if torch.cuda.is_available() or not str(device).startswith("cuda") else "cpu"
    if encoder == "uni":
        extractor = UNI2hExtractor(uni_model_path, device=resolved_device)
        return lambda images: extract_uni_patches(extractor, images)

    if encoder == "virchow2":
        from src.a1_probe_encoders.main import _build_virchow_extractor

        if not encoder_model_path:
            raise ValueError("--encoder-model is required for --encoder virchow2")
        extractor = _build_virchow_extractor(encoder_model_path, device=resolved_device)
        return lambda images: extract_virchow_patches(extractor, images)

    if encoder == "ctranspath":
        from src.a1_probe_encoders.main import _build_ctranspath_extractor, _default_ctranspath_model_source

        extractor = _build_ctranspath_extractor(
            encoder_model_path or _default_ctranspath_model_source(),
            device=resolved_device,
        )
        return lambda images: extract_ctranspath_patches(extractor, images)

    if encoder == "resnet50":
        import torchvision.models as tv_models

        weights = tv_models.ResNet50_Weights.IMAGENET1K_V2
        model = tv_models.resnet50(weights=weights).to(resolved_device)
        model.eval()
        wrapper = SimpleNamespace(model=model, transform=weights.transforms())
        return lambda images: extract_resnet50_patches(wrapper, images)

    raise ValueError(f"unsupported encoder for patch extraction: {encoder!r}")


def cache_patch_features(
    image_dir,
    output_dir,
    *,
    encoder,
    uni_model_path,
    encoder_model_path=None,
    batch_size=8,
    device="cuda",
    patches_prefix="patches",
):
    """Cache per-tile patch features for the selected encoder."""
    image_paths = _list_image_paths(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nFound {len(image_paths)} images in {image_dir}")
    if len(image_paths) == 0:
        print("❌ No images found!")
        return

    extract_batch = _build_patch_extractor(
        encoder=encoder,
        uni_model_path=uni_model_path,
        encoder_model_path=encoder_model_path,
        device=device,
    )

    print(f"\nCaching {encoder} patch features (batch_size={batch_size})...")
    print("=" * 70)
    for start in tqdm(range(0, len(image_paths), batch_size), desc=f"{encoder} patches"):
        batch_paths = image_paths[start : start + batch_size]
        batch_images = []
        valid_paths = []
        for image_path in batch_paths:
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"⚠ Error loading {image_path.name}: cv2.imread returned None")
                continue
            batch_images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            valid_paths.append(image_path)
        if not batch_images:
            continue

        features = extract_batch(batch_images)
        for index, image_path in enumerate(valid_paths):
            np.save(output_dir / f"{image_path.stem}_{patches_prefix}.npy", features[index])

    print("\n" + "=" * 70)
    print(f"✓ {encoder} patch extraction complete!")
    print(f"  Output directory: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract VAE and UNI-2h features for PixCell ControlNet training"
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        required=True,
        help="Directory containing input images"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./features",
        help="Directory to save extracted features (default: ./features)"
    )
    parser.add_argument(
        "--uni-model",
        type=str,
        default="./pretrained_models/uni-2h",
        help="Path to UNI-2h model (default: ./pretrained_models/uni-2h)"
    )
    parser.add_argument(
        "--vae-model",
        type=str,
        default="./pretrained_models/sd-3.5-vae",
        help="Path to SD3.5 VAE model (default: ./pretrained_models/sd-3.5-vae)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for processing (default: 8)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else 'cpu',
        help="Device to use (default: cuda)"
    )
    parser.add_argument(
        "--vae-prefix",
        type=str,
        default="sd3_vae",
        help="Prefix for VAE feature files (default: sd3_vae)"
    )
    parser.add_argument(
        "--uni-prefix",
        type=str,
        default="uni",
        help="Prefix for UNI feature files (default: uni)"
    )
    parser.add_argument("--skip-uni", action="store_true", help="Skip UNI CLS embedding cache")
    parser.add_argument("--skip-vae", action="store_true", help="Skip SD3.5 VAE latent cache")
    parser.add_argument(
        "--save-uni-tokens",
        action="store_true",
        help="Also cache the (P, D) UNI patch-token grid as <stem>_<uni-tokens-prefix>.npy",
    )
    parser.add_argument(
        "--uni-tokens-prefix",
        type=str,
        default="uni_tokens",
        help="Prefix for UNI patch-token files (default: uni_tokens)",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        choices=("uni", "virchow2", "ctranspath", "resnet50"),
        default=None,
        help="Cache per-patch features for a specific encoder instead of the default UNI/VAE flow.",
    )
    parser.add_argument(
        "--encoder-model",
        type=str,
        default=None,
        help="Optional encoder weights/model source for --encoder virchow2 or ctranspath.",
    )
    parser.add_argument(
        "--save-patches",
        action="store_true",
        help="Compatibility flag for patch-cache runs; patch mode always saves encoder patches.",
    )
    parser.add_argument(
        "--patches-output-dir",
        type=str,
        default=None,
        help="Output directory for per-tile patch features. Defaults to --output-dir.",
    )
    parser.add_argument(
        "--patches-prefix",
        type=str,
        default="patches",
        help="Filename prefix for cached patch tensors (default: patches).",
    )

    args = parser.parse_args()

    if args.encoder is not None:
        cache_patch_features(
            image_dir=args.image_dir,
            output_dir=args.patches_output_dir or args.output_dir,
            encoder=args.encoder,
            uni_model_path=args.uni_model,
            encoder_model_path=args.encoder_model,
            batch_size=args.batch_size,
            device=args.device,
            patches_prefix=args.patches_prefix,
        )
        return

    extract_features_from_images(
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        uni_model_path=args.uni_model,
        vae_model_path=args.vae_model,
        batch_size=args.batch_size,
        device=args.device,
        vae_prefix=args.vae_prefix,
        uni_prefix=args.uni_prefix,
        skip_uni=args.skip_uni,
        skip_vae=args.skip_vae,
        save_uni_tokens=args.save_uni_tokens,
        uni_tokens_prefix=args.uni_tokens_prefix,
    )


if __name__ == "__main__":
    main()
