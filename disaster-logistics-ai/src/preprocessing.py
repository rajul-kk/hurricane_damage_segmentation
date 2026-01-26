"""
Preprocessing module for Hurricane Damage Satellite Images Dataset.
Downloads data from Kaggle and prepares it for training.
"""

import os
import shutil
from pathlib import Path
from typing import Tuple, Optional

import kagglehub
from PIL import Image
from tqdm import tqdm


def download_dataset(target_dir: Optional[str] = None) -> str:
    """
    Download the hurricane damage satellite images dataset from Kaggle.
    
    Args:
        target_dir: Optional directory to copy data to. If None, returns kaggle cache path.
        
    Returns:
        Path to the dataset files.
    """
    print("Downloading Hurricane Damage Satellite Images dataset...")
    path = kagglehub.dataset_download("kmader/satellite-images-of-hurricane-damage")
    print(f"Dataset downloaded to: {path}")
    
    if target_dir:
        target_path = Path(target_dir)
        target_path.mkdir(parents=True, exist_ok=True)
        
        # Copy files to target directory
        src_path = Path(path)
        for item in src_path.rglob("*"):
            if item.is_file():
                relative = item.relative_to(src_path)
                dest = target_path / relative
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, dest)
        
        print(f"Dataset copied to: {target_dir}")
        return target_dir
    
    return path


def explore_dataset(data_path: str) -> dict:
    """
    Explore the dataset structure and return statistics.
    
    Args:
        data_path: Path to the dataset.
        
    Returns:
        Dictionary with dataset statistics.
    """
    data_path = Path(data_path)
    stats = {
        "total_images": 0,
        "classes": {},
        "image_formats": set(),
        "sample_sizes": []
    }
    
    # Walk through the dataset directory
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                stats["total_images"] += 1
                stats["image_formats"].add(file.split('.')[-1].lower())
                
                # Get class from parent directory name
                class_name = Path(root).name
                if class_name not in stats["classes"]:
                    stats["classes"][class_name] = 0
                stats["classes"][class_name] += 1
                
                # Sample some image sizes
                if len(stats["sample_sizes"]) < 10:
                    try:
                        img_path = os.path.join(root, file)
                        with Image.open(img_path) as img:
                            stats["sample_sizes"].append(img.size)
                    except Exception as e:
                        print(f"Could not read {file}: {e}")
    
    stats["image_formats"] = list(stats["image_formats"])
    return stats


def prepare_data_splits(
    data_path: str,
    output_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[str, str, str]:
    """
    Split the dataset into train, validation, and test sets.
    
    Args:
        data_path: Path to the original dataset.
        output_dir: Directory to save the splits.
        train_ratio: Proportion of data for training.
        val_ratio: Proportion of data for validation.
        test_ratio: Proportion of data for testing.
        seed: Random seed for reproducibility.
        
    Returns:
        Tuple of paths (train_dir, val_dir, test_dir).
    """
    import random
    random.seed(seed)
    
    data_path = Path(data_path)
    output_path = Path(output_dir)
    
    train_dir = output_path / "train"
    val_dir = output_path / "val"
    test_dir = output_path / "test"
    
    # Find all image directories (classes)
    class_dirs = []
    for item in data_path.rglob("*"):
        if item.is_dir():
            # Check if directory contains images
            images = list(item.glob("*.jpeg")) + list(item.glob("*.jpg")) + \
                     list(item.glob("*.png")) + list(item.glob("*.tif"))
            if images:
                class_dirs.append(item)
    
    # If no subdirectories with images, check for labeled image names
    if not class_dirs:
        # Dataset might have flat structure with labels in filenames
        all_images = []
        for ext in ['*.jpeg', '*.jpg', '*.png', '*.tif', '*.tiff']:
            all_images.extend(data_path.rglob(ext))
        
        if all_images:
            print(f"Found {len(all_images)} images in flat structure")
            # Try to infer classes from filenames or parent directories
            classes = set()
            for img in all_images:
                # Check parent directory name
                parent = img.parent.name
                classes.add(parent)
            
            class_dirs = [data_path / c for c in classes if (data_path / c).exists()]
    
    print(f"Found {len(class_dirs)} class directories")
    
    for class_dir in tqdm(class_dirs, desc="Processing classes"):
        class_name = class_dir.name
        
        # Get all images in this class
        images = []
        for ext in ['*.jpeg', '*.jpg', '*.png', '*.tif', '*.tiff']:
            images.extend(class_dir.glob(ext))
        
        if not images:
            continue
            
        random.shuffle(images)
        
        # Calculate split indices
        n = len(images)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        splits = [
            (images[:train_end], train_dir / class_name),
            (images[train_end:val_end], val_dir / class_name),
            (images[val_end:], test_dir / class_name)
        ]
        
        for split_images, split_dir in splits:
            split_dir.mkdir(parents=True, exist_ok=True)
            for img_path in split_images:
                dest = split_dir / img_path.name
                shutil.copy2(img_path, dest)
    
    print(f"Data splits created:")
    print(f"  Train: {train_dir}")
    print(f"  Validation: {val_dir}")
    print(f"  Test: {test_dir}")
    
    return str(train_dir), str(val_dir), str(test_dir)


def get_image_transforms(image_size: int = 224, augment: bool = False):
    """
    Get image transformations for training/inference.
    
    Args:
        image_size: Target image size.
        augment: Whether to apply data augmentation.
        
    Returns:
        torchvision transforms composition.
    """
    from torchvision import transforms
    
    if augment:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


if __name__ == "__main__":
    # Download and explore the dataset
    dataset_path = download_dataset()
    
    print("\n--- Dataset Statistics ---")
    stats = explore_dataset(dataset_path)
    print(f"Total images: {stats['total_images']}")
    print(f"Image formats: {stats['image_formats']}")
    print(f"Classes: {stats['classes']}")
    print(f"Sample image sizes: {stats['sample_sizes']}")
