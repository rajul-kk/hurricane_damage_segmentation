"""
CNN Classifier for Hurricane Damage Detection from Satellite Images.
Uses transfer learning with pre-trained models for better performance.
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional, Literal


class HurricaneDamageClassifier(nn.Module):
    """
    CNN classifier for detecting hurricane damage in satellite images.
    Uses transfer learning from pre-trained models.
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        backbone: Literal["resnet18", "resnet34", "resnet50", "efficientnet_b0", "mobilenet_v3"] = "resnet18",
        pretrained: bool = True,
        dropout_rate: float = 0.5,
        freeze_backbone: bool = False
    ):
        """
        Initialize the classifier.
        
        Args:
            num_classes: Number of output classes (default 2: damage/no_damage).
            backbone: Pre-trained model to use as feature extractor.
            pretrained: Whether to use pre-trained weights.
            dropout_rate: Dropout rate for regularization.
            freeze_backbone: Whether to freeze backbone weights.
        """
        super().__init__()
        
        self.backbone_name = backbone
        self.num_classes = num_classes
        
        # Load pre-trained backbone
        if backbone == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet18(weights=weights)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            
        elif backbone == "resnet34":
            weights = models.ResNet34_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet34(weights=weights)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            
        elif backbone == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet50(weights=weights)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            
        elif backbone == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            self.backbone = models.efficientnet_b0(weights=weights)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
            
        elif backbone == "mobilenet_v3":
            weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            self.backbone = models.mobilenet_v3_small(weights=weights)
            num_features = self.backbone.classifier[0].in_features
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(256, num_classes)
        )
        
        # Initialize classifier weights
        self._init_classifier_weights()
    
    def _init_classifier_weights(self):
        """Initialize the classifier head weights."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, 3, height, width).
            
        Returns:
            Output tensor of shape (batch_size, num_classes).
        """
        features = self.backbone(x)
        output = self.classifier(features)
        return output
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get predicted class labels.
        
        Args:
            x: Input tensor.
            
        Returns:
            Predicted class indices.
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.argmax(logits, dim=1)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get predicted class probabilities.
        
        Args:
            x: Input tensor.
            
        Returns:
            Class probabilities.
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.softmax(logits, dim=1)
    
    def unfreeze_backbone(self, num_layers: Optional[int] = None):
        """
        Unfreeze backbone layers for fine-tuning.
        
        Args:
            num_layers: Number of layers to unfreeze from the end.
                       If None, unfreeze all layers.
        """
        params = list(self.backbone.parameters())
        
        if num_layers is None:
            for param in params:
                param.requires_grad = True
        else:
            # Freeze all first
            for param in params:
                param.requires_grad = False
            # Unfreeze last n layers
            for param in params[-num_layers:]:
                param.requires_grad = True


class SimpleCNN(nn.Module):
    """
    Simple CNN for baseline comparison (no transfer learning).
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        input_size: int = 224,
        dropout_rate: float = 0.5
    ):
        super().__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 5
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


def get_model(
    model_type: str = "transfer",
    num_classes: int = 2,
    backbone: str = "resnet18",
    pretrained: bool = True,
    **kwargs
) -> nn.Module:
    """
    Factory function to create a model.
    
    Args:
        model_type: "transfer" for transfer learning, "simple" for basic CNN.
        num_classes: Number of output classes.
        backbone: Backbone architecture (for transfer learning).
        pretrained: Use pre-trained weights.
        **kwargs: Additional arguments passed to the model.
        
    Returns:
        Initialized model.
    """
    if model_type == "transfer":
        return HurricaneDamageClassifier(
            num_classes=num_classes,
            backbone=backbone,
            pretrained=pretrained,
            **kwargs
        )
    elif model_type == "simple":
        return SimpleCNN(num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test model creation
    model = HurricaneDamageClassifier(num_classes=2, backbone="resnet18")
    print(f"Model created: {model.backbone_name}")
    
    # Test forward pass
    x = torch.randn(4, 3, 224, 224)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
