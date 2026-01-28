"""
Tiler module for Hurricane Damage Detection.
Cuts large satellite maps into small "chips" (e.g., 224x224) that can be 
fed into a neural network for classification.
"""

import os
from pathlib import Path
from typing import Tuple, List, Optional, Iterator, Dict

import numpy as np
from PIL import Image
from tqdm import tqdm


def calculate_grid_dimensions(
    image_size: Tuple[int, int],
    chip_size: int = 224,
    stride: Optional[int] = None,
    include_partial: bool = False
) -> Tuple[int, int, int]:
    """
    Calculate the grid dimensions for tiling.
    
    Args:
        image_size: (width, height) of the original image.
        chip_size: Size of each square chip.
        stride: Step size between chips. If None, equals chip_size (no overlap).
        include_partial: If True, includes partial chips at edges.
        
    Returns:
        Tuple of (num_rows, num_cols, stride).
    """
    width, height = image_size
    stride = stride or chip_size
    
    if include_partial:
        num_cols = (width + stride - 1) // stride
        num_rows = (height + stride - 1) // stride
    else:
        num_cols = max(1, (width - chip_size) // stride + 1)
        num_rows = max(1, (height - chip_size) // stride + 1)
    
    return num_rows, num_cols, stride


def get_chip_coordinates(
    row: int,
    col: int,
    chip_size: int,
    stride: int,
    image_size: Tuple[int, int]
) -> Tuple[int, int, int, int]:
    """
    Calculate the pixel coordinates for a chip at given grid position.
    
    Args:
        row: Row index in the grid.
        col: Column index in the grid.
        chip_size: Size of each square chip.
        stride: Step size between chips.
        image_size: (width, height) of the original image.
        
    Returns:
        Tuple of (x1, y1, x2, y2) bounding box coordinates.
    """
    width, height = image_size
    
    x1 = col * stride
    y1 = row * stride
    x2 = min(x1 + chip_size, width)
    y2 = min(y1 + chip_size, height)
    
    # Clamp coordinates
    x1 = max(0, x1)
    y1 = max(0, y1)
    
    return x1, y1, x2, y2


def extract_chip(
    image: Image.Image,
    row: int,
    col: int,
    chip_size: int = 224,
    stride: Optional[int] = None,
    pad_partial: bool = True,
    pad_color: Tuple[int, int, int] = (0, 0, 0)
) -> Image.Image:
    """
    Extract a single chip from the image at the given grid position.
    
    Args:
        image: Source PIL Image.
        row: Row index in the grid.
        col: Column index in the grid.
        chip_size: Size of each square chip.
        stride: Step size between chips. If None, equals chip_size.
        pad_partial: If True, pads partial chips to full chip_size.
        pad_color: RGB color for padding.
        
    Returns:
        Extracted chip as PIL Image.
    """
    stride = stride or chip_size
    x1, y1, x2, y2 = get_chip_coordinates(
        row, col, chip_size, stride, image.size
    )
    
    chip = image.crop((x1, y1, x2, y2))
    
    # Pad if chip is smaller than expected
    if pad_partial and (chip.size[0] < chip_size or chip.size[1] < chip_size):
        padded = Image.new(image.mode, (chip_size, chip_size), pad_color)
        padded.paste(chip, (0, 0))
        chip = padded
    
    return chip


def tile_image(
    image_path: str,
    output_dir: str,
    chip_size: int = 224,
    stride: Optional[int] = None,
    include_partial: bool = True,
    pad_partial: bool = True,
    output_format: str = "png",
    filename_pattern: str = "chip_{row:04d}_{col:04d}"
) -> List[Dict]:
    """
    Tile a large satellite image into smaller chips.
    
    Args:
        image_path: Path to the input satellite image.
        output_dir: Directory to save the chips.
        chip_size: Size of each square chip (default: 224 for neural networks).
        stride: Step size between chips. If None, equals chip_size (no overlap).
                Use stride < chip_size for overlapping tiles.
        include_partial: If True, includes partial chips at image edges.
        pad_partial: If True, pads partial chips to full chip_size.
        output_format: Image format for chips (png, jpg, etc.).
        filename_pattern: Pattern for chip filenames. Use {row} and {col} placeholders.
        
    Returns:
        List of dictionaries with chip metadata:
        [{"path": str, "row": int, "col": int, "x1": int, "y1": int, "x2": int, "y2": int}, ...]
    """
    # Load the image
    image = Image.open(image_path)
    image_size = image.size
    
    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Calculate grid dimensions
    num_rows, num_cols, stride = calculate_grid_dimensions(
        image_size, chip_size, stride, include_partial
    )
    
    print(f"Tiling image: {image_path}")
    print(f"  Image size: {image_size[0]}x{image_size[1]}")
    print(f"  Chip size: {chip_size}x{chip_size}")
    print(f"  Stride: {stride}")
    print(f"  Grid: {num_rows} rows x {num_cols} cols = {num_rows * num_cols} chips")
    
    chips_metadata = []
    
    for row in tqdm(range(num_rows), desc="Tiling rows"):
        for col in range(num_cols):
            # Extract the chip
            chip = extract_chip(
                image, row, col, chip_size, stride, pad_partial
            )
            
            # Get coordinates for metadata
            x1, y1, x2, y2 = get_chip_coordinates(
                row, col, chip_size, stride, image_size
            )
            
            # Save the chip
            filename = filename_pattern.format(row=row, col=col) + f".{output_format}"
            chip_path = output_path / filename
            chip.save(chip_path)
            
            chips_metadata.append({
                "path": str(chip_path),
                "filename": filename,
                "row": row,
                "col": col,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "actual_size": chip.size
            })
    
    print(f"  Saved {len(chips_metadata)} chips to {output_dir}")
    return chips_metadata


def tile_image_generator(
    image_path: str,
    chip_size: int = 224,
    stride: Optional[int] = None,
    include_partial: bool = True,
    pad_partial: bool = True
) -> Iterator[Tuple[Image.Image, int, int, Dict]]:
    """
    Generator that yields chips from a large image without saving to disk.
    
    Useful for inference pipelines where you want to process chips on-the-fly
    without the I/O overhead of saving to disk.
    
    Args:
        image_path: Path to the input satellite image.
        chip_size: Size of each square chip.
        stride: Step size between chips. If None, equals chip_size.
        include_partial: If True, includes partial chips at edges.
        pad_partial: If True, pads partial chips to full chip_size.
        
    Yields:
        Tuple of (chip_image, row, col, metadata_dict).
    """
    image = Image.open(image_path)
    image_size = image.size
    
    num_rows, num_cols, stride = calculate_grid_dimensions(
        image_size, chip_size, stride, include_partial
    )
    
    for row in range(num_rows):
        for col in range(num_cols):
            chip = extract_chip(
                image, row, col, chip_size, stride, pad_partial
            )
            
            x1, y1, x2, y2 = get_chip_coordinates(
                row, col, chip_size, stride, image_size
            )
            
            metadata = {
                "row": row,
                "col": col,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "image_size": image_size,
                "chip_size": chip_size,
                "stride": stride
            }
            
            yield chip, row, col, metadata


def tile_folder(
    input_dir: str,
    output_dir: str,
    chip_size: int = 224,
    stride: Optional[int] = None,
    extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tif", ".tiff"),
    **kwargs
) -> Dict[str, List[Dict]]:
    """
    Tile all images in a folder.
    
    Args:
        input_dir: Directory containing satellite images.
        output_dir: Directory to save all chips.
        chip_size: Size of each square chip.
        stride: Step size between chips.
        extensions: File extensions to process.
        **kwargs: Additional arguments passed to tile_image().
        
    Returns:
        Dictionary mapping source image paths to their chip metadata lists.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Find all images
    images = []
    for ext in extensions:
        images.extend(input_path.glob(f"*{ext}"))
        images.extend(input_path.glob(f"*{ext.upper()}"))
    
    images = sorted(set(images))
    print(f"Found {len(images)} images to tile")
    
    all_metadata = {}
    
    for image_path in images:
        # Create subdirectory for each source image
        image_output_dir = output_path / image_path.stem
        
        chips = tile_image(
            str(image_path),
            str(image_output_dir),
            chip_size=chip_size,
            stride=stride,
            **kwargs
        )
        
        all_metadata[str(image_path)] = chips
    
    return all_metadata


def get_reconstruction_info(
    image_path: str,
    chip_size: int = 224,
    stride: Optional[int] = None
) -> Dict:
    """
    Get information needed to reconstruct the original image from chips.
    
    This is useful for the stitcher module to know how to piece
    the chips back together.
    
    Args:
        image_path: Path to the original image.
        chip_size: Size of each chip.
        stride: Step size between chips.
        
    Returns:
        Dictionary with reconstruction metadata.
    """
    image = Image.open(image_path)
    image_size = image.size
    
    num_rows, num_cols, stride = calculate_grid_dimensions(
        image_size, chip_size, stride
    )
    
    return {
        "original_path": str(image_path),
        "image_size": image_size,
        "chip_size": chip_size,
        "stride": stride,
        "num_rows": num_rows,
        "num_cols": num_cols,
        "total_chips": num_rows * num_cols
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Tile large satellite images into smaller chips for neural network processing."
    )
    parser.add_argument(
        "input",
        help="Input image path or directory"
    )
    parser.add_argument(
        "output",
        help="Output directory for chips"
    )
    parser.add_argument(
        "--chip-size", "-s",
        type=int,
        default=224,
        help="Size of each chip (default: 224)"
    )
    parser.add_argument(
        "--stride", "-t",
        type=int,
        default=None,
        help="Stride between chips (default: same as chip size, no overlap)"
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=None,
        help="Overlap ratio (0.0-1.0). Alternative to --stride."
    )
    parser.add_argument(
        "--format", "-f",
        default="png",
        help="Output image format (default: png)"
    )
    parser.add_argument(
        "--no-partial",
        action="store_true",
        help="Exclude partial chips at image edges"
    )
    
    args = parser.parse_args()
    
    # Calculate stride from overlap if provided
    stride = args.stride
    if args.overlap is not None:
        stride = int(args.chip_size * (1 - args.overlap))
        print(f"Using stride {stride} for {args.overlap*100:.0f}% overlap")
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        tile_image(
            args.input,
            args.output,
            chip_size=args.chip_size,
            stride=stride,
            include_partial=not args.no_partial,
            output_format=args.format
        )
    elif input_path.is_dir():
        tile_folder(
            args.input,
            args.output,
            chip_size=args.chip_size,
            stride=stride,
            include_partial=not args.no_partial,
            output_format=args.format
        )
    else:
        print(f"Error: {args.input} is not a valid file or directory")
        exit(1)
