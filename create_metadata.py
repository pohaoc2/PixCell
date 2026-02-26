"""
Helper script to create metadata HDF5 file for ControlNet training
"""
import h5py
from pathlib import Path
import numpy as np
import argparse


def create_metadata_file(image_dir, output_file, resolution=256, mask_dir=None, 
                        vae_prefix="sd3_vae", uni_prefix="uni", mask_prefix="cellvit_mask"):
    """
    Create HDF5 metadata file for ControlNet training.
    Only includes images that have all required files (VAE, UNI, mask).
    
    Args:
        image_dir: Directory containing images
        output_file: Output HDF5 file path
        resolution: Image resolution (256 for PixCell-256)
        mask_dir: Directory containing masks (if None, uses root/masks/)
        vae_prefix: Prefix for VAE feature files
        uni_prefix: Prefix for UNI embedding files
        mask_prefix: Prefix for mask files
    """
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    parent_dir = image_dir.parent
    # Get root directory
    features_dir = parent_dir / f'features_{image_dir.name}'
    features_mask_dir = parent_dir / f'features_{mask_dir.name}'


    print(f"Looking for images in: {image_dir}")
    print(f"Looking for features in: {features_dir}")
    print(f"Looking for features_mask in: {features_mask_dir}")
    print(f"Looking for masks in: {mask_dir}")
    # Find all images
    image_extensions = ['.png', '.jpg', '.jpeg']
    images = []
    for ext in image_extensions:
        images.extend(image_dir.glob(f'*{ext}'))
    images = sorted(images, key=lambda x: int(x.name.split('.')[0].split('_')[-1]))
    #images = images[:49]
    print(f"\nFound {len(images)} image files")
    # sort image by index
    
    # Filter images that have all required files
    valid_images = []
    missing_files = {'vae': 0, 'uni': 0, 'mask': 0, 'vae_mask': 0}
    
    for img_path in images:
        img_name = img_path.name
        base_name = img_name.replace('.png', '').replace('.jpg', '').replace('.jpeg', '')
        
        # Check for required files
        vae_path = features_dir / f"{base_name}_{vae_prefix}.npy"
        uni_path = features_dir / f"{base_name}_{uni_prefix}.npy"
        mask_path = mask_dir / f"{base_name}_{mask_prefix}.png"
        vae_mask_path = features_mask_dir / f"{base_name}_mask_{vae_prefix}.npy"
        # Also check .npy for masks
        if not mask_path.exists():
            mask_path = mask_dir / f"{base_name}_{mask_prefix}.npy"
        
        has_vae = vae_path.exists()
        has_uni = uni_path.exists()
        has_mask = mask_path.exists()
        has_vae_mask = vae_mask_path.exists()
        if has_vae and has_uni and has_mask and has_vae_mask:
            valid_images.append(img_name)
        else:
            if not has_vae:
                missing_files['vae'] += 1
            if not has_uni:
                missing_files['uni'] += 1
            if not has_mask:
                missing_files['mask'] += 1
            if not has_vae_mask:
                missing_files['vae_mask'] += 1
    print(f"\n✓ Valid images (with all files): {len(valid_images)}")
    if sum(missing_files.values()) > 0:
        print(f"\n⚠ Skipped images due to missing files:")
        print(f"  - Missing VAE features: {missing_files['vae']}")
        print(f"  - Missing UNI embeddings: {missing_files['uni']}")
        print(f"  - Missing masks: {missing_files['mask']}")
        print(f"  - Missing VAE masks: {missing_files['vae_mask']}")
    
    if len(valid_images) == 0:
        print("\n❌ No valid images found! Check your file naming conventions.")
        print(f"\nExpected file pattern for image 'example.png':")
        print(f"  - Image: {image_dir}/example.png")
        print(f"  - VAE: {features_dir}/example_{vae_prefix}.npy")
        print(f"  - UNI: {features_dir}/example_{uni_prefix}.npy")
        print(f"  - Mask: {mask_dir}/example_{mask_prefix}.png")
        print(f"  - VAE mask: {features_mask_dir}/example_mask_{vae_prefix}.npy")
        return
    
    # Create HDF5 file
    with h5py.File(output_file, 'w') as h5:
        # Create dataset key with resolution
        key = f"controlnet_{resolution}"
        
        # Store image names as bytes
        dt = h5py.string_dtype(encoding='utf-8')
        h5.create_dataset(
            key, 
            data=np.array(valid_images, dtype=object),
            dtype=dt
        )
        
        print(f"\n✓ Created HDF5 dataset '{key}' with {len(valid_images)} images")
        print(f"✓ Saved to: {output_file}")
    
    # Print sample entries
    print(f"\nSample entries (first 5):")
    for i, img_name in enumerate(valid_images[:5], 1):
        print(f"  {i}. {img_name}")
    
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="Create HDF5 metadata file for PixCell ControlNet training"
    )
    parser.add_argument(
        "--image-dir", 
        type=str, 
        default="./patches",
        help="Directory containing images"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="./patches/metadata/patch_names_controlnet.hdf5",
        help="Output HDF5 file path"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help="Image resolution (default: 256)"
    )
    parser.add_argument(
        "--mask-dir",
        type=str,
        default=None,
        help="Directory containing masks (default: auto-detect)"
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
        help="Prefix for UNI embedding files (default: uni)"
    )
    parser.add_argument(
        "--mask-prefix",
        type=str,
        default="mask",
        help="Prefix for mask files (default: cellvit_mask)"
    )
    
    args = parser.parse_args()
    
    create_metadata_file(
        image_dir=args.image_dir,
        output_file=args.output_file,
        resolution=args.resolution,
        mask_dir=args.mask_dir,
        vae_prefix=args.vae_prefix,
        uni_prefix=args.uni_prefix,
        mask_prefix=args.mask_prefix
    )


if __name__ == "__main__":
    main()
