"""
Example: Demonstrating stitcher.py using REAL images from Kaggle Hurricane Damage Dataset.

This script:
1. Loads actual satellite chip images from the Kaggle dataset
2. Creates a synthetic "large map" by stitching chips together 
3. Simulates flood predictions on each chip position
4. Uses stitcher.py to apply red overlays on flood regions
5. Saves the output images
"""

import os
import sys
from pathlib import Path
import random

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from PIL import Image
import kagglehub

from src.stitcher import stitch_predictions, create_prediction_summary


def get_kaggle_dataset_path() -> Path:
    """Get the path to the Kaggle hurricane damage dataset, downloading if needed."""
    print("Checking for Kaggle dataset...")
    path = kagglehub.dataset_download("kmader/satellite-images-of-hurricane-damage")
    print(f"Dataset location: {path}")
    return Path(path)


def get_images_by_class(dataset_path: Path) -> dict:
    """Find all images organized by damage/no_damage class."""
    images = {
        'damage': [],
        'no_damage': []
    }
    
    # Search for damage images
    for folder in ['train_another/damage', 'test_another/damage', 'validation_another/damage']:
        damage_dir = dataset_path / folder
        if damage_dir.exists():
            for img_path in damage_dir.glob('*.jpeg'):
                images['damage'].append(img_path)
            for img_path in damage_dir.glob('*.jpg'):
                images['damage'].append(img_path)
    
    # Search for no_damage images
    for folder in ['train_another/no_damage', 'test_another/no_damage', 'validation_another/no_damage']:
        no_damage_dir = dataset_path / folder
        if no_damage_dir.exists():
            for img_path in no_damage_dir.glob('*.jpeg'):
                images['no_damage'].append(img_path)
            for img_path in no_damage_dir.glob('*.jpg'):
                images['no_damage'].append(img_path)
    
    print(f"Found {len(images['damage'])} damage images")
    print(f"Found {len(images['no_damage'])} no_damage images")
    
    return images


def create_composite_map(
    images: dict,
    grid_size: tuple = (3, 3),
    chip_size: int = 224,
    flood_ratio: float = 0.4
) -> tuple:
    """
    Create a composite satellite map from real chips.
    
    Returns:
        Tuple of (composite_image, predictions_dict)
    """
    rows, cols = grid_size
    total_chips = rows * cols
    
    # Decide how many should be "flood" vs "no_damage"
    num_flood = int(total_chips * flood_ratio)
    num_no_damage = total_chips - num_flood
    
    # Randomly select images
    damage_samples = random.sample(images['damage'], min(num_flood, len(images['damage'])))
    no_damage_samples = random.sample(images['no_damage'], min(num_no_damage, len(images['no_damage'])))
    
    # Pad if we don't have enough
    while len(damage_samples) < num_flood:
        damage_samples.append(random.choice(images['damage']))
    while len(no_damage_samples) < num_no_damage:
        no_damage_samples.append(random.choice(images['no_damage']))
    
    # Combine and shuffle positions
    all_chips = [(img, 'flood') for img in damage_samples] + \
                [(img, 'no_damage') for img in no_damage_samples]
    random.shuffle(all_chips)
    
    # Create composite image
    composite_width = cols * chip_size
    composite_height = rows * chip_size
    composite = Image.new('RGB', (composite_width, composite_height))
    
    predictions = {}
    chip_idx = 0
    
    for row in range(rows):
        for col in range(cols):
            if chip_idx >= len(all_chips):
                break
                
            img_path, label = all_chips[chip_idx]
            
            # Load and resize chip
            try:
                chip = Image.open(img_path).convert('RGB')
                chip = chip.resize((chip_size, chip_size), Image.Resampling.LANCZOS)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                # Create placeholder
                chip = Image.new('RGB', (chip_size, chip_size), (100, 100, 100))
            
            # Paste into composite
            x = col * chip_size
            y = row * chip_size
            composite.paste(chip, (x, y))
            
            # Record prediction with simulated confidence
            confidence = random.uniform(0.75, 0.98)
            predictions[(row, col)] = {
                'label': label,
                'confidence': confidence,
                'source_image': str(img_path.name)
            }
            
            chip_idx += 1
    
    return composite, predictions


def main():
    random.seed(42)  # For reproducibility
    
    # Setup output directory
    output_dir = Path(__file__).parent.parent / "outputs" / "stitcher_demo_real"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Stitcher Demo: Using REAL Kaggle Hurricane Damage Images")
    print("=" * 70)
    
    # Get dataset
    dataset_path = get_kaggle_dataset_path()
    images = get_images_by_class(dataset_path)
    
    if not images['damage'] or not images['no_damage']:
        print("ERROR: Could not find images in the dataset!")
        print("Please ensure the dataset is properly downloaded.")
        return
    
    # Configuration
    chip_size = 224
    
    # Define 3 demo scenarios with different flood ratios
    scenarios = [
        {
            "name": "light_flooding",
            "description": "Area with minimal flood damage (20% affected)",
            "grid_size": (3, 3),
            "flood_ratio": 0.2
        },
        {
            "name": "moderate_flooding", 
            "description": "Area with moderate flood damage (50% affected)",
            "grid_size": (3, 3),
            "flood_ratio": 0.5
        },
        {
            "name": "severe_flooding",
            "description": "Area with severe flood damage (80% affected)",
            "grid_size": (3, 3),
            "flood_ratio": 0.8
        }
    ]
    
    # Process each scenario
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n[{i}/3] Creating: {scenario['name']}")
        print(f"      {scenario['description']}")
        
        # Create composite map from real images
        composite, predictions = create_composite_map(
            images,
            grid_size=scenario['grid_size'],
            chip_size=chip_size,
            flood_ratio=scenario['flood_ratio']
        )
        
        # Save input composite
        input_path = output_dir / f"{scenario['name']}_input.png"
        composite.save(input_path)
        print(f"      Created composite map: {input_path.name}")
        
        # Apply stitcher overlay
        output_path = output_dir / f"{scenario['name']}_overlay.png"
        result = stitch_predictions(
            str(input_path),
            predictions,
            chip_size=chip_size,
            overlay_alpha=140,  # 55% opacity red overlay
            output_path=str(output_path)
        )
        
        # Generate and print summary
        grid_rows, grid_cols = scenario['grid_size']
        summary = create_prediction_summary(
            predictions,
            (grid_cols * chip_size, grid_rows * chip_size),
            chip_size
        )
        
        flood_count = summary['flood_damage_detected']
        total_count = summary['total_chips_analyzed']
        
        print(f"      Flood regions: {flood_count}/{total_count} chips")
        print(f"      Actual flood %: {summary['flood_percentage']:.1f}%")
        print(f"      Output saved: {output_path.name}")
        
        # Print which source images were used
        print("      Source chips used:")
        for (row, col), pred in sorted(predictions.items()):
            status = "🔴 FLOOD" if pred['label'] == 'flood' else "🟢 OK"
            print(f"        ({row},{col}): {status} - {pred['source_image'][:30]}...")
    
    print("\n" + "=" * 70)
    print(f"Demo complete! Check outputs in:\n{output_dir}")
    print("=" * 70)
    
    # List generated files
    print("\nGenerated files:")
    for f in sorted(output_dir.glob("*.png")):
        size_kb = f.stat().st_size / 1024
        print(f"  - {f.name} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
