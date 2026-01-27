"""
Stitcher module for Hurricane Damage Detection.
Reconstructs the original large map with semi-transparent overlays 
on regions predicted as flood/damage by the classifier.
"""

import os
from pathlib import Path
from typing import Tuple, List, Optional, Dict

import numpy as np
from PIL import Image


def create_overlay_color(
    color: Tuple[int, int, int] = (255, 0, 0),
    alpha: int = 128
) -> Tuple[int, int, int, int]:
    """
    Create an RGBA color tuple for overlays.
    
    Args:
        color: RGB tuple for the overlay color (default: red).
        alpha: Transparency value (0=fully transparent, 255=fully opaque).
        
    Returns:
        RGBA tuple for the overlay.
    """
    return (*color, alpha)


def apply_overlay_to_chip(
    chip: Image.Image,
    color: Tuple[int, int, int] = (255, 0, 0),
    alpha: int = 128
) -> Image.Image:
    """
    Apply a semi-transparent color overlay to an image chip.
    
    Args:
        chip: The original image chip (PIL Image).
        color: RGB tuple for overlay color (default: red for flood).
        alpha: Overlay transparency (0-255, default 128 = 50%).
        
    Returns:
        Image with color overlay applied.
    """
    # Convert to RGBA if needed
    if chip.mode != 'RGBA':
        chip = chip.convert('RGBA')
    
    # Create a solid color overlay with transparency
    overlay = Image.new('RGBA', chip.size, (*color, alpha))
    
    # Composite the overlay onto the original chip
    result = Image.alpha_composite(chip, overlay)
    
    return result


def stitch_predictions(
    original_image_path: str,
    chip_predictions: Dict[Tuple[int, int], dict],
    chip_size: int = 224,
    stride: Optional[int] = None,
    flood_color: Tuple[int, int, int] = (255, 0, 0),
    damage_color: Tuple[int, int, int] = (255, 165, 0),
    overlay_alpha: int = 128,
    output_path: Optional[str] = None
) -> Image.Image:
    """
    Stitch prediction overlays onto the original large map.
    
    For each chip predicted as "flood" or "damage", applies a 
    semi-transparent colored overlay on that region of the map.
    
    Args:
        original_image_path: Path to the original large satellite image.
        chip_predictions: Dictionary mapping (row, col) grid positions to 
                         prediction results. Each result should have:
                         {'label': str, 'confidence': float, 'chip_path': str}
        chip_size: Size of each chip in pixels.
        stride: Step size between chips (default: same as chip_size, no overlap).
        flood_color: RGB color for flood overlay (default: red).
        damage_color: RGB color for damage overlay (default: orange).
        overlay_alpha: Transparency of overlay (0-255).
        output_path: Optional path to save the result.
        
    Returns:
        PIL Image with overlays applied to predicted regions.
    """
    if stride is None:
        stride = chip_size
    
    # Load the original image
    original = Image.open(original_image_path).convert('RGBA')
    width, height = original.size
    
    # Create a copy to draw overlays on
    result = original.copy()
    
    # Color mapping for different prediction labels
    color_map = {
        'flood': flood_color,
        'damage': flood_color,  # Use same color for damage
        'flooded': flood_color,
        'damaged': flood_color,
        '1': flood_color,  # Class index 1 often = damage
    }
    
    # Process each prediction
    for (row, col), prediction in chip_predictions.items():
        label = prediction.get('label', '').lower()
        confidence = prediction.get('confidence', 0.0)
        
        # Skip if not a flood/damage prediction
        if label not in color_map:
            continue
        
        # Calculate pixel coordinates for this chip
        x = col * stride
        y = row * stride
        
        # Ensure we don't exceed image boundaries
        chip_w = min(chip_size, width - x)
        chip_h = min(chip_size, height - y)
        
        if chip_w <= 0 or chip_h <= 0:
            continue
        
        # Extract the region from the original image
        region = result.crop((x, y, x + chip_w, y + chip_h))
        
        # Apply overlay with color based on prediction
        overlay_color = color_map[label]
        
        # Scale alpha by confidence if desired
        # scaled_alpha = int(overlay_alpha * confidence)
        
        overlaid = apply_overlay_to_chip(region, overlay_color, overlay_alpha)
        
        # Paste the overlaid region back
        result.paste(overlaid, (x, y))
    
    # Save if output path provided
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        # Convert to RGB for saving as JPEG/PNG without alpha issues
        result_rgb = Image.new('RGB', result.size, (255, 255, 255))
        result_rgb.paste(result, mask=result.split()[3])  # Use alpha as mask
        result_rgb.save(output_path)
        print(f"Saved overlay result to: {output_path}")
    
    return result


def stitch_from_chip_folder(
    original_image_path: str,
    chips_folder: str,
    predictions: Dict[str, dict],
    chip_size: int = 224,
    stride: Optional[int] = None,
    output_path: Optional[str] = None,
    **kwargs
) -> Image.Image:
    """
    Stitch predictions from a folder of chip images.
    
    This is a convenience function that infers grid positions from 
    chip filenames (expected format: chip_ROW_COL.ext).
    
    Args:
        original_image_path: Path to the original large image.
        chips_folder: Folder containing chip images.
        predictions: Dictionary mapping chip filename to prediction 
                    {'label': str, 'confidence': float}.
        chip_size: Size of each chip.
        stride: Step between chips.
        output_path: Path to save result.
        **kwargs: Additional arguments passed to stitch_predictions.
        
    Returns:
        PIL Image with overlays.
    """
    # Convert filename-based predictions to grid-position-based
    chip_predictions = {}
    
    for filename, pred in predictions.items():
        # Parse filename to extract row and col
        # Expected format: chip_0_0.png, chip_1_2.jpg, etc.
        name = Path(filename).stem
        parts = name.split('_')
        
        try:
            if len(parts) >= 3:
                row = int(parts[-2])
                col = int(parts[-1])
            else:
                # Alternative format: row-col or just numbers
                continue
                
            chip_predictions[(row, col)] = {
                'label': pred.get('label', ''),
                'confidence': pred.get('confidence', 0.0),
                'chip_path': os.path.join(chips_folder, filename)
            }
        except ValueError:
            print(f"Could not parse grid position from filename: {filename}")
            continue
    
    return stitch_predictions(
        original_image_path,
        chip_predictions,
        chip_size=chip_size,
        stride=stride,
        output_path=output_path,
        **kwargs
    )


def create_prediction_summary(
    chip_predictions: Dict[Tuple[int, int], dict],
    image_size: Tuple[int, int],
    chip_size: int = 224
) -> dict:
    """
    Create a summary of predictions over the image.
    
    Args:
        chip_predictions: Dictionary of (row, col) -> prediction.
        image_size: (width, height) of original image.
        chip_size: Size of each chip.
        
    Returns:
        Summary statistics dictionary.
    """
    total_chips = len(chip_predictions)
    flood_chips = sum(
        1 for p in chip_predictions.values() 
        if p.get('label', '').lower() in ['flood', 'damage', 'flooded', 'damaged', '1']
    )
    
    width, height = image_size
    total_possible = (width // chip_size) * (height // chip_size)
    
    return {
        'total_chips_analyzed': total_chips,
        'flood_damage_detected': flood_chips,
        'no_damage_detected': total_chips - flood_chips,
        'flood_percentage': (flood_chips / total_chips * 100) if total_chips > 0 else 0,
        'coverage': (total_chips / total_possible * 100) if total_possible > 0 else 0,
        'image_size': image_size,
        'chip_size': chip_size
    }


if __name__ == "__main__":
    # Example usage
    print("Stitcher module for hurricane damage overlay visualization")
    print("=" * 60)
    
    # Example: Create a test overlay
    test_size = (1000, 1000)
    test_image = Image.new('RGB', test_size, (100, 150, 100))  # Green background
    test_path = "test_input.png"
    test_image.save(test_path)
    
    # Simulate predictions (some chips are "flood")
    test_predictions = {
        (0, 0): {'label': 'no_damage', 'confidence': 0.95},
        (0, 1): {'label': 'flood', 'confidence': 0.87},
        (0, 2): {'label': 'no_damage', 'confidence': 0.92},
        (1, 0): {'label': 'flood', 'confidence': 0.78},
        (1, 1): {'label': 'flood', 'confidence': 0.91},
        (1, 2): {'label': 'no_damage', 'confidence': 0.88},
        (2, 0): {'label': 'no_damage', 'confidence': 0.96},
        (2, 1): {'label': 'flood', 'confidence': 0.82},
        (2, 2): {'label': 'no_damage', 'confidence': 0.90},
    }
    
    # Stitch with overlays
    result = stitch_predictions(
        test_path,
        test_predictions,
        chip_size=224,
        output_path="test_overlay_output.png",
        overlay_alpha=140
    )
    
    # Print summary
    summary = create_prediction_summary(test_predictions, test_size, chip_size=224)
    print(f"\nPrediction Summary:")
    print(f"  Total chips: {summary['total_chips_analyzed']}")
    print(f"  Flood/Damage: {summary['flood_damage_detected']}")
    print(f"  No damage: {summary['no_damage_detected']}")
    print(f"  Flood %: {summary['flood_percentage']:.1f}%")
    
    # Cleanup test files
    os.remove(test_path)
    print(f"\nTest complete! Check 'test_overlay_output.png'")
