"""
Utility functions for image format conversion.
"""
from pathlib import Path
from PIL import Image
from typing import Union, Optional


def tif_to_png(
    input_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    page: Optional[int] = None
) -> Path:
    """
    Convert a .tif image file to PNG format.
    
    Args:
        input_path: Path to the input .tif file
        output_path: Path to the output PNG file. If None, will be generated
                     by replacing the extension of input_path with .png
        page: For multi-page TIFF files, specify which page to convert.
              If None, converts the first page (page 0)
    
    Returns:
        Path to the output PNG file
    
    Raises:
        FileNotFoundError: If input_path does not exist
        ValueError: If input_path is not a .tif file
        IOError: If image cannot be opened or saved
    """
    input_path = Path(input_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    if not input_path.suffix.lower() in ['.tif', '.tiff']:
        raise ValueError(f"Input file must be a .tif or .tiff file, got: {input_path.suffix}")
    
    # Generate output path if not provided
    if output_path is None:
        output_path = input_path.with_suffix('.png')
    else:
        output_path = Path(output_path)
        # Ensure output has .png extension
        if output_path.suffix.lower() != '.png':
            output_path = output_path.with_suffix('.png')
    
    # Open and convert the image
    with Image.open(input_path) as img:
        # Handle multi-page TIFF files
        if hasattr(img, 'n_frames') and img.n_frames > 1:
            if page is not None:
                img.seek(page)
            else:
                img.seek(0)  # Use first page by default
        
        # Convert to RGB if necessary (PNG doesn't support all TIFF modes)
        if img.mode in ('RGBA', 'LA', 'P'):
            # Keep transparency if present
            if img.mode == 'P':
                img = img.convert('RGBA')
            # Convert to RGB if no transparency needed
            elif img.mode == 'LA':
                # Convert grayscale with alpha to RGBA
                rgb_img = Image.new('RGBA', img.size)
                rgb_img.paste(img, img)
                img = rgb_img
        elif img.mode not in ('RGB', 'RGBA', 'L'):
            # Convert other modes to RGB
            img = img.convert('RGB')
        
        # Save as PNG
        img.save(output_path, 'PNG')
    
    return output_path


def batch_tif_to_png(
    input_dir: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    pattern: str = "*.tif*"
) -> list[Path]:
    """
    Convert all .tif files in a directory to PNG format.
    
    Args:
        input_dir: Directory containing .tif files
        output_dir: Directory to save PNG files. If None, saves in input_dir
        pattern: Glob pattern to match TIFF files (default: "*.tif*")
    
    Returns:
        List of paths to converted PNG files
    """
    input_dir = Path(input_dir)
    if output_dir is None:
        output_dir = input_dir
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    tif_files = list(input_dir.glob(pattern))
    converted_files = []
    
    for tif_file in tif_files:
        if tif_file.suffix.lower() in ['.tif', '.tiff']:
            output_path = output_dir / tif_file.with_suffix('.png').name
            try:
                converted_path = tif_to_png(tif_file, output_path)
                converted_files.append(converted_path)
                print(f"Converted: {tif_file.name} -> {converted_path.name}")
            except Exception as e:
                print(f"Error converting {tif_file.name}: {e}")
    
    return converted_files


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python image_converter.py <input.tif> [output.png]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        result = tif_to_png(input_file, output_file)
        print(f"Successfully converted to: {result}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

