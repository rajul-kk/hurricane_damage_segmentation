"""
Training script for Hurricane Damage Classifier.
Run this script to train the model from the command line.
"""

import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from src.classifier import HurricaneDamageClassifier, get_model
from src.preprocessing import download_dataset, prepare_data_splits, get_image_transforms


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> tuple[float, float]:
    """
    Train the model for one epoch.
    
    This function iterates through all batches in the training dataloader,
    performs forward pass, computes loss, backpropagates gradients, and
    updates model weights.
    
    Args:
        model: The neural network model to train.
        dataloader: DataLoader providing training batches.
        criterion: Loss function (e.g., CrossEntropyLoss).
        optimizer: Optimizer for updating weights (e.g., AdamW).
        device: Device to run training on (cuda/cpu).
        
    Returns:
        Tuple of (average_loss, accuracy_percentage) for the epoch.
    """
    model.train()  # Set model to training mode (enables dropout, batch norm updates)
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    for images, labels in progress_bar:
        # 1. Move data to device (GPU/CPU)
        images, labels = images.to(device), labels.to(device)
        
        # 2. Zero the gradients from previous iteration
        optimizer.zero_grad()
        
        # 3. Forward pass: compute predictions
        outputs = model(images)
        
        # 4. Compute loss between predictions and ground truth
        loss = criterion(outputs, labels)
        
        # 5. Backward pass: compute gradients
        loss.backward()
        
        # 6. Update weights using computed gradients
        optimizer.step()
        
        # Track metrics
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)  # Get class with highest score
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100. * correct / total:.2f}%'
        })
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> tuple[float, float]:
    """
    Validate the model on a held-out dataset.
    
    This function evaluates model performance without updating weights.
    Uses torch.no_grad() to disable gradient computation for efficiency.
    
    Args:
        model: The neural network model to evaluate.
        dataloader: DataLoader providing validation/test batches.
        criterion: Loss function for computing validation loss.
        device: Device to run validation on (cuda/cpu).
        
    Returns:
        Tuple of (average_loss, accuracy_percentage) for the validation set.
    """
    model.eval()  # Set model to evaluation mode (disables dropout, freezes batch norm)
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Disable gradient computation for validation (saves memory and speeds up)
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validating", leave=False)
        for images, labels in progress_bar:
            # Move data to device
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass only (no backward pass needed)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Track metrics
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 20,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    save_dir: str = "./checkpoints",
    device: torch.device = None
) -> dict:
    """
    Complete training loop with validation and model checkpointing.
    
    This is the main training orchestrator that:
    1. Sets up optimizer and learning rate scheduler
    2. Runs training and validation for each epoch
    3. Saves the best model based on validation accuracy
    4. Tracks training history for visualization
    
    Args:
        model: The neural network model to train.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        num_epochs: Number of training epochs.
        learning_rate: Initial learning rate.
        weight_decay: L2 regularization strength.
        save_dir: Directory to save model checkpoints.
        device: Device to run training on.
        
    Returns:
        Dictionary containing training history (losses and accuracies).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    
    # Loss function: CrossEntropyLoss combines LogSoftmax + NLLLoss
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer: AdamW = Adam with decoupled weight decay (better regularization)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Scheduler: Reduce LR when validation loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',      # Minimize validation loss
        patience=3,      # Wait 3 epochs before reducing
        factor=0.5       # Multiply LR by 0.5 when reducing
    )
    
    # Create checkpoint directory
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    best_model_path = os.path.join(save_dir, "best_model.pth")
    
    # Training history for plotting
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    
    # ============ MAIN TRAINING LOOP ============
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 40)
        
        # ---------- Training Phase ----------
        # Model learns from training data, weights are updated
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # ---------- Validation Phase ----------
        # Evaluate on held-out data to monitor generalization
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # ---------- Learning Rate Adjustment ----------
        # Reduce LR if validation loss stops improving
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print epoch summary
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        print(f"LR: {current_lr:.2e}")
        
        # ---------- Model Checkpointing ----------
        # Save the model if it achieves best validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, best_model_path)
            print(f"✓ Saved best model (val acc: {val_acc:.2f}%)")
    
    print(f"\n{'='*40}")
    print(f"Training complete! Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Best model saved to: {best_model_path}")
    
    return history


def main():
    """Main entry point for command-line training."""
    parser = argparse.ArgumentParser(description="Train Hurricane Damage Classifier")
    parser.add_argument("--data-dir", type=str, default="./data",
                        help="Directory to download/store dataset")
    parser.add_argument("--splits-dir", type=str, default="./data_splits",
                        help="Directory for train/val/test splits")
    parser.add_argument("--backbone", type=str, default="resnet18",
                        choices=["resnet18", "resnet34", "resnet50", "efficientnet_b0", "mobilenet_v3"],
                        help="Backbone architecture")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--image-size", type=int, default=224,
                        help="Input image size")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of data loading workers")
    parser.add_argument("--save-dir", type=str, default="./checkpoints",
                        help="Directory to save model checkpoints")
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Download and prepare data
    print("\n1. Preparing dataset...")
    if not os.path.exists(args.splits_dir):
        dataset_path = download_dataset(args.data_dir)
        train_dir, val_dir, test_dir = prepare_data_splits(
            dataset_path, args.splits_dir
        )
    else:
        train_dir = os.path.join(args.splits_dir, "train")
        val_dir = os.path.join(args.splits_dir, "val")
        test_dir = os.path.join(args.splits_dir, "test")
        print(f"Using existing splits from {args.splits_dir}")
    
    # Create datasets and dataloaders
    print("\n2. Creating data loaders...")
    train_transforms = get_image_transforms(args.image_size, augment=True)
    val_transforms = get_image_transforms(args.image_size, augment=False)
    
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Classes: {train_dataset.classes}")
    
    # Create model
    print("\n3. Initializing model...")
    num_classes = len(train_dataset.classes)
    model = get_model(
        model_type="transfer",
        num_classes=num_classes,
        backbone=args.backbone,
        pretrained=True
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Backbone: {args.backbone}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Train
    print("\n4. Starting training...")
    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        save_dir=args.save_dir,
        device=device
    )
    
    # Final evaluation on test set
    print("\n5. Evaluating on test set...")
    test_dataset = datasets.ImageFolder(test_dir, transform=val_transforms)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # Load best model
    checkpoint = torch.load(os.path.join(args.save_dir, "best_model.pth"))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc = validate(model, test_loader, nn.CrossEntropyLoss(), device)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")


if __name__ == "__main__":
    main()
