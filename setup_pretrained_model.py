"""
Setup script to download pretrained models from HuggingFace:
1. PixCell-256 base model
2. UNI-2h (foundation model for histopathology)
3. Stable Diffusion 3.5 VAE
4. CellViT-SAM-H (cell segmentation model)
"""
import os
from pathlib import Path
from huggingface_hub import snapshot_download


def download_pixcell_controlnet(save_dir="./pretrained_models", token=None):
    if token is None:
        token = os.getenv("HF_TOKEN")
    if token is None:
        raise ValueError("HF_TOKEN environment variable is not set")
    """
    Download PixCell-256 base model from HuggingFace
    
    Args:
        save_dir: Directory to save the model
        
    Returns:
        str: Path to downloaded model
    """
    save_path = Path(save_dir) / "pixcell-256-controlnet"
    
    if save_path.exists() and any(save_path.glob("*.pth")):
        print(f"✓ PixCell-256-controlnet already exists at: {save_path}")
        return str(save_path)
    
    save_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"Downloading PixCell-256-controlnet from HuggingFace...")
    print(f"{'='*70}")
    print(f"Repository: StonyBrook-CVLab/PixCell-256-Cell-ControlNet")
    print(f"Save location: {save_path}")
    
    try:
        model_path = snapshot_download(
            repo_id="StonyBrook-CVLab/PixCell-256-Cell-ControlNet",
            local_dir=save_path,
            local_dir_use_symlinks=False,
            token=token,
        )
        
        print(f"\n✓ PixCell-256-controlnet downloaded successfully")
        
        # List downloaded files
        model_files = list(save_path.glob("*.pth")) + list(save_path.glob("*.safetensors"))
        if model_files:
            print(f"  Model files:")
            for file in model_files:
                size_mb = file.stat().st_size / (1024**2)
                print(f"    - {file.name} ({size_mb:.1f} MB)")
        
        return str(save_path)
    except Exception as e:
        print(f"\n❌ Error downloading PixCell-256-controlnet: {e}")
        return None

def download_pixcell_256(save_dir="./pretrained_models", token=None):
    if token is None:
        token = os.getenv("HF_TOKEN")
    if token is None:
        raise ValueError("HF_TOKEN environment variable is not set")
    """
    Download PixCell-256 base model from HuggingFace
    
    Args:
        save_dir: Directory to save the model
        
    Returns:
        str: Path to downloaded model
    """
    save_path = Path(save_dir) / "pixcell-256"
    
    if save_path.exists() and any(save_path.glob("*.pth")):
        print(f"✓ PixCell-256 already exists at: {save_path}")
        return str(save_path)
    
    save_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"Downloading PixCell-256 from HuggingFace...")
    print(f"{'='*70}")
    print(f"Repository: StonyBrook-CVLab/PixCell-256")
    print(f"Save location: {save_path}")
    
    try:
        model_path = snapshot_download(
            repo_id="StonyBrook-CVLab/PixCell-256",
            local_dir=save_path,
            local_dir_use_symlinks=False,
            token=token,
        )
        
        print(f"\n✓ PixCell-256 downloaded successfully")
        
        # List downloaded files
        model_files = list(save_path.glob("*.pth")) + list(save_path.glob("*.safetensors"))
        if model_files:
            print(f"  Model files:")
            for file in model_files:
                size_mb = file.stat().st_size / (1024**2)
                print(f"    - {file.name} ({size_mb:.1f} MB)")
        
        return str(save_path)
    except Exception as e:
        print(f"\n❌ Error downloading PixCell-256: {e}")
        return None


def download_uni_2h(save_dir="./pretrained_models", token=None):
    if token is None:
        token = os.getenv("HF_TOKEN")
    if token is None:
        raise ValueError("HF_TOKEN environment variable is not set")
    
    """
    Download UNI-2h foundation model from HuggingFace
    
    Args:
        save_dir: Directory to save the model
        
    Returns:
        str: Path to downloaded model
    """
    save_path = Path(save_dir) / "uni-2h"
    
    if save_path.exists() and any(save_path.glob("*.pth")):
        print(f"✓ UNI-2h already exists at: {save_path}")
        return str(save_path)
    
    save_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"Downloading UNI-2h from HuggingFace...")
    print(f"{'='*70}")
    print(f"Repository: MahmoodLab/UNI2-h")
    print(f"Save location: {save_path}")
    print(f"Note: This is a large model (~2GB), download may take a while...")
    
    try:
        model_path = snapshot_download(
            repo_id="MahmoodLab/UNI2-h",
            local_dir=save_path,
            local_dir_use_symlinks=False,
            token=token,
        )
        
        print(f"\n✓ UNI-2h downloaded successfully")
        
        # List downloaded files
        model_files = list(save_path.glob("*.pth")) + list(save_path.glob("*.bin"))
        if model_files:
            print(f"  Model files:")
            for file in model_files:
                size_mb = file.stat().st_size / (1024**2)
                print(f"    - {file.name} ({size_mb:.1f} MB)")
        
        return str(save_path)
    except Exception as e:
        print(f"\n❌ Error downloading UNI-2h: {e}")
        return None


def download_sd3_vae(save_dir="./pretrained_models", token=None):
    if token is None:
        token = os.getenv("HF_TOKEN")
    if token is None:
        raise ValueError("HF_TOKEN environment variable is not set")
    """
    Download Stable Diffusion 3.5 VAE from HuggingFace
    
    Args:
        save_dir: Directory to save the model
        
    Returns:
        str: Path to downloaded model
    """
    save_path = Path(save_dir) / "sd-3.5-vae"
    
    if save_path.exists() and (save_path / "config.json").exists():
        print(f"✓ SD3.5 VAE already exists at: {save_path}")
        return str(save_path)
    
    save_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"Downloading Stable Diffusion 3.5 VAE from HuggingFace...")
    print(f"{'='*70}")
    print(f"Repository: stabilityai/stable-diffusion-3.5-large")
    print(f"Save location: {save_path}")
    print(f"Note: Only downloading VAE components (not the full SD3.5 model)...")
    
    try:
        # Only download VAE-related files to save space
        model_path = snapshot_download(
            repo_id="stabilityai/stable-diffusion-3.5-large",
            local_dir=save_path,
            local_dir_use_symlinks=False,
            allow_patterns=["vae/*", "*.json", "*.txt"],  # Only download VAE
            token=token,
        )
        
        print(f"\n✓ SD3.5 VAE downloaded successfully")
        
        # List downloaded files
        vae_files = list((save_path / "vae").glob("*")) if (save_path / "vae").exists() else []
        if vae_files:
            print(f"  VAE files:")
            for file in vae_files:
                if file.is_file():
                    size_mb = file.stat().st_size / (1024**2)
                    print(f"    - {file.name} ({size_mb:.1f} MB)")
        
        return str(save_path)
    except Exception as e:
        print(f"\n❌ Error downloading SD3.5 VAE: {e}")
        return None


def download_all_models(save_dir="./pretrained_models", token=None):
    if token is None:
        token = os.getenv("HF_TOKEN")
    if token is None:
        raise ValueError("HF_TOKEN environment variable is not set")
    """
    Download all required pretrained models
    
    Args:
        save_dir: Directory to save all models
    """
    print("\n" + "="*70)
    print(" "*15 + "Downloading Pretrained Models")
    print("="*70)
    print("\nThis script will download:")
    print("  1. PixCell-256 base model (~600MB)")
    print("  2. UNI-2h foundation model (~2GB)")
    print("  3. Stable Diffusion 3.5 VAE (~300MB)")
    print(f"\nAll models will be saved to: {save_dir}")
    print("\nNote: For cell segmentation consistency, install Cellpose separately:")
    print("  pip install git+https://www.github.com/mouseland/cellpose.git")
    print("="*70)
    
    results = {}
    
    # Download each model

    results['pixcell'] = download_pixcell_256(save_dir, token)
    results['pixcell-controlnet'] = download_pixcell_controlnet(save_dir, token)
    results['uni2h'] = download_uni_2h(save_dir, token)
    results['sd3_vae'] = download_sd3_vae(save_dir, token)
    
    # Summary
    print("\n" + "="*70)
    print(" "*20 + "Download Summary")
    print("="*70)
    
    success_count = sum(1 for v in results.values() if v is not None)
    
    for name, path in results.items():
        status = "✓" if path else "❌"
        print(f"{status} {name:12} {path if path else 'FAILED'}")
    
    print("="*70)
    
    if success_count == len(results):
        print("\n✓ All models downloaded successfully!")
        print("\nNext steps:")
        print("  1. Install Cellpose for segmentation consistency:")
        print("     pip install git+https://www.github.com/mouseland/cellpose.git")
        print("  2. Generate VAE features and UNI embeddings:")
        print("     python extract_features.py --image-dir ./patches")
        print("  3. Create metadata file:")
        print("     python create_metadata.py")
        print("  4. Start training:")
        print("     python train_controlnet.py config_controlnet_full.py")
    else:
        print(f"\n⚠ {len(results) - success_count} model(s) failed to download")
        print("Please check the error messages above and try again.")
    
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Download pretrained models for PixCell ControlNet training"
    )
    parser.add_argument(
        "--save-dir", 
        type=str, 
        default="./pretrained_models",
        help="Directory to save the pretrained models (default: ./pretrained_models)"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=['pixcell', 'pixcell-controlnet', 'uni2h', 'sd3_vae', 'all'],
        default='all',
        help="Which model to download (default: all)"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace token to use for downloading models"
    )
    args = parser.parse_args()
    
    if args.model == 'all':
        download_all_models(args.save_dir, args.token)
    elif args.model == 'pixcell':
        download_pixcell_256(args.save_dir, args.token)
    elif args.model == 'pixcell-controlnet':
        download_pixcell_controlnet(args.save_dir, args.token)
    elif args.model == 'uni2h':
        download_uni_2h(args.save_dir, args.token)
    elif args.model == 'sd3_vae':
        download_sd3_vae(args.save_dir, args.token)