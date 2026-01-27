"""
Verification script for PixCell ControlNet data organization
"""
import h5py
from pathlib import Path
import numpy as np
from PIL import Image
import argparse


def verify_data_organization(root_dir='.', config_file=None):
    """Verify all required files exist and have correct formats"""
    root = Path(root_dir)
    
    print("=" * 70)
    print(" " * 15 + "PixCell-256 ControlNet Data Verification")
    print("=" * 70)
    
    # Load config if provided
    vae_prefix = "sd3_vae"
    uni_prefix = "uni"
    mask_prefix = "cellvit_mask"
    
    if config_file:
        print(f"\nLoading config from: {config_file}")
        # Simple config parsing
        with open(config_file, 'r') as f:
            for line in f:
                if 'vae_prefix' in line and '=' in line:
                    vae_prefix = line.split('=')[1].strip().strip('"').strip("'").strip(',')
                elif 'ssl_prefix' in line and '=' in line:
                    uni_prefix = line.split('=')[1].strip().strip('"').strip("'").strip(',')
                elif 'mask_prefix' in line and '=' in line:
                    mask_prefix = line.split('=')[1].strip().strip('"').strip("'").strip(',')
    
    print(f"\nFile naming conventions:")
    print(f"  VAE prefix: {vae_prefix}")
    print(f"  UNI prefix: {uni_prefix}")
    print(f"  Mask prefix: {mask_prefix}")
    
    # 1. Check directory structure
    print("\n" + "─" * 70)
    print("DIRECTORY STRUCTURE")
    print("─" * 70)
    
    required_dirs = {
        'patches': root / "patches",
        'features': root / "features",
        'masks': root / "masks",
        'pretrained': root / "pretrained_models",
    }
    
    for name, path in required_dirs.items():
        if path.exists():
            count = len(list(path.glob("*"))) if path.is_dir() else 0
            print(f"✓ {name:<12} {path} ({count} files)")
        else:
            print(f"❌ {name:<12} {path} (NOT FOUND)")
    
    # 2. Check metadata file
    print("\n" + "─" * 70)
    print("METADATA FILE")
    print("─" * 70)
    
    metadata_file = root / "patches/metadata/patch_names_controlnet.hdf5"
    if not metadata_file.exists():
        print(f"❌ Metadata file not found: {metadata_file}")
        print(f"\n   Run: python create_metadata.py")
        return False
    
    image_names = []
    with h5py.File(metadata_file, 'r') as h5:
        keys = list(h5.keys())
        print(f"✓ Metadata file exists: {metadata_file}")
        print(f"  Dataset keys: {keys}")
        
        # Get images from all keys
        for key in keys:
            names = [name.decode('utf-8') if isinstance(name, bytes) else name for name in h5[key][:]]
            image_names.extend(names)
        
        print(f"  Total images: {len(image_names)}")
    
    if len(image_names) == 0:
        print("❌ No images found in metadata file!")
        return False
    
    # 3. Check sample files
    print("\n" + "─" * 70)
    print("FILE VERIFICATION (checking first 5 samples)")
    print("─" * 70)
    
    check_count = min(5, len(image_names))
    all_valid = True
    
    for idx, img_name in enumerate(image_names[:check_count], 1):
        print(f"\n[{idx}/{check_count}] {img_name}")
        print("─" * 50)
        
        base_name = img_name.replace('.png', '').replace('.jpg', '').replace('.jpeg', '')
        
        # Check image
        img_path = root / f"patches/{img_name}"
        if img_path.exists():
            try:
                img = Image.open(img_path)
                print(f"  ✓ Image:   {img.size[0]}×{img.size[1]} pixels")
            except Exception as e:
                print(f"  ❌ Image:   ERROR - {e}")
                all_valid = False
        else:
            print(f"  ❌ Image:   NOT FOUND")
            all_valid = False
        
        # Check VAE feature
        vae_path = root / f"features/{base_name}_{vae_prefix}.npy"
        if vae_path.exists():
            try:
                vae_feat = np.load(vae_path)
                expected_shape = (2, 16, 32, 32)
                if vae_feat.shape == expected_shape:
                    print(f"  ✓ VAE:     {vae_feat.shape} (mean+std, 16 channels, 32×32)")
                else:
                    print(f"  ⚠ VAE:     {vae_feat.shape} (expected {expected_shape})")
                    all_valid = False
            except Exception as e:
                print(f"  ❌ VAE:     ERROR - {e}")
                all_valid = False
        else:
            print(f"  ❌ VAE:     NOT FOUND at {vae_path}")
            all_valid = False
        
        # Check UNI embedding
        uni_path = root / f"features/{base_name}_{uni_prefix}.npy"
        if uni_path.exists():
            try:
                uni_feat = np.load(uni_path)
                print(f"  ✓ UNI:     {uni_feat.shape}")
            except Exception as e:
                print(f"  ❌ UNI:     ERROR - {e}")
                all_valid = False
        else:
            print(f"  ❌ UNI:     NOT FOUND at {uni_path}")
            all_valid = False
        
        # Check mask
        mask_path = root / f"masks/{base_name}_{mask_prefix}.png"
        if not mask_path.exists():
            mask_path = root / f"masks/{base_name}_{mask_prefix}.npy"
        
        if mask_path.exists():
            try:
                if mask_path.suffix == '.png':
                    mask = Image.open(mask_path).convert('L')
                    mask_array = np.array(mask)
                else:
                    mask_array = np.load(mask_path)
                
                if mask_array.ndim == 3:
                    mask_array = mask_array[0]  # Take first channel
                
                unique_vals = np.unique(mask_array)
                expected_shape = (256, 256)
                
                if mask_array.shape == expected_shape:
                    is_binary = len(unique_vals) <= 2
                    status = "binary" if is_binary else f"{len(unique_vals)} values"
                    print(f"  ✓ Mask:    {mask_array.shape} ({status})")
                else:
                    print(f"  ⚠ Mask:    {mask_array.shape} (expected {expected_shape})")
                    all_valid = False
            except Exception as e:
                print(f"  ❌ Mask:    ERROR - {e}")
                all_valid = False
        else:
            print(f"  ❌ Mask:    NOT FOUND at {mask_path.parent}/{base_name}_{mask_prefix}.[png/npy]")
            all_valid = False
    
    # 4. Check pretrained models
    print("\n" + "─" * 70)
    print("PRETRAINED MODELS")
    print("─" * 70)
    
    pixcell_path = root / "pretrained_models/pixcell-256"
    if pixcell_path.exists():
        model_files = list(pixcell_path.glob("*.pth")) + list(pixcell_path.glob("*.safetensors"))
        if model_files:
            print(f"✓ PixCell-256: {pixcell_path}")
            for mf in model_files:
                size_mb = mf.stat().st_size / (1024**2)
                print(f"    - {mf.name} ({size_mb:.1f} MB)")
        else:
            print(f"⚠ PixCell-256 directory exists but no model files found")
    else:
        print(f"❌ PixCell-256 not found: {pixcell_path}")
        print(f"   Download with: python setup_pretrained_model.py")
        all_valid = False
    
    vae_path = root / "pretrained_models/sd-3.5-vae"
    if vae_path.exists():
        print(f"✓ SD3.5 VAE: {vae_path}")
    else:
        print(f"⚠ SD3.5 VAE not found: {vae_path}")
        print(f"   (May not be needed if using pre-computed VAE features)")
    
    # Summary
    print("\n" + "=" * 70)
    if all_valid:
        print("✓ VERIFICATION PASSED - Ready to train!")
    else:
        print("⚠ VERIFICATION FAILED - Please fix the issues above")
    print("=" * 70)
    
    # Quick start command
    if all_valid:
        print("\nQuick Start:")
        print("  python train_controlnet.py config_controlnet_pretrained.py")
    
    return all_valid


def main():
    parser = argparse.ArgumentParser(
        description="Verify data organization for PixCell ControlNet training"
    )
    parser.add_argument(
        "--root-dir",
        type=str,
        default=".",
        help="Root directory of the project (default: current directory)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file to read naming conventions"
    )
    
    args = parser.parse_args()
    verify_data_organization(args.root_dir, args.config)


if __name__ == "__main__":
    main()
