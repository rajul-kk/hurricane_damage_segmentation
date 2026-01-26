# Hurricane Damage Classification from Satellite Images

A deep learning pipeline for classifying hurricane damage in satellite imagery using transfer learning with PyTorch.

## Dataset

This project uses the [Satellite Images of Hurricane Damage](https://www.kaggle.com/datasets/kmader/satellite-images-of-hurricane-damage) dataset from Kaggle. The dataset contains satellite images categorized into:
- **damage**: Areas with visible hurricane damage
- **no_damage**: Areas without visible damage

## Project Structure

```
disaster-logistics-ai/
├── src/
│   ├── classifier.py      # CNN model definitions (ResNet, EfficientNet, etc.)
│   ├── preprocessing.py   # Data download, exploration, and transforms
│   ├── tiler.py           # Image tiling utilities
│   └── stitcher.py        # Image stitching utilities
├── data/
│   ├── train/             # Training data
│   ├── test/              # Test data
│   └── inference_maps/    # Maps for inference
├── outputs/
│   └── heatmaps/          # Generated heatmaps
├── train_model.py         # Training script
├── assess_damage.py       # Inference script
├── requirements.txt       # Dependencies
└── README.md
```

## Installation

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Set up Kaggle API credentials:**
   - Go to https://www.kaggle.com/account
   - Click "Create New API Token"
   - Save the `kaggle.json` file to `~/.kaggle/` (Linux/Mac) or `C:\Users\<username>\.kaggle\` (Windows)

## Usage

### Quick Start - Download and Explore Dataset

```python
import kagglehub

# Download latest version
path = kagglehub.dataset_download("kmader/satellite-images-of-hurricane-damage")
print("Path to dataset files:", path)
```

Or run the preprocessing module:
```bash
python -m src.preprocessing
```

### Training

Train the model with default settings (ResNet-18 backbone):
```bash
python train_model.py
```

Train with custom settings:
```bash
python train_model.py \
    --backbone resnet50 \
    --epochs 50 \
    --batch-size 64 \
    --learning-rate 1e-4 \
    --freeze-backbone \
    --unfreeze-epoch 5
```

**Available backbones:**
- `resnet18` (default, fast and efficient)
- `resnet34`
- `resnet50` (better accuracy, more compute)
- `efficientnet_b0` (good balance of speed/accuracy)
- `mobilenet_v3` (lightweight, good for deployment)

**Key training arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--backbone` | resnet18 | Backbone architecture |
| `--epochs` | 30 | Number of training epochs |
| `--batch-size` | 32 | Batch size |
| `--learning-rate` | 1e-4 | Initial learning rate |
| `--freeze-backbone` | False | Freeze backbone for initial epochs |
| `--unfreeze-epoch` | 5 | Epoch to unfreeze backbone |
| `--dropout` | 0.5 | Dropout rate |
| `--patience` | 10 | Early stopping patience |

### Inference

Assess damage on a single image:
```bash
python assess_damage.py \
    --model-path outputs/best_model.pth \
    --image-path path/to/satellite_image.jpg
```

Batch process a directory of images:
```bash
python assess_damage.py \
    --model-path outputs/best_model.pth \
    --image-dir path/to/images/ \
    --output-dir outputs/inference
```

## Model Architecture

The classifier uses transfer learning with pre-trained ImageNet weights:

1. **Backbone**: Feature extraction (ResNet, EfficientNet, or MobileNet)
2. **Custom Head**:
   - Linear(features → 512) + ReLU + BatchNorm + Dropout
   - Linear(512 → 256) + ReLU + BatchNorm + Dropout
   - Linear(256 → num_classes)

## Training Pipeline

1. **Data Download**: Automatically downloads from Kaggle
2. **Data Splitting**: 70% train, 15% validation, 15% test
3. **Data Augmentation**: Random flips, rotations, color jitter
4. **Class Balancing**: Weighted loss for imbalanced classes
5. **Learning Rate**: Cosine annealing schedule
6. **Early Stopping**: Prevents overfitting

## Output Files

After training:
- `outputs/best_model.pth` - Best model checkpoint
- `outputs/final_model.pth` - Final model
- `outputs/training_history.png` - Loss/accuracy curves
- `outputs/confusion_matrix.png` - Test set confusion matrix

After inference:
- `outputs/inference/predictions_visualization.png` - Sample predictions
- `outputs/inference/assessment_report.txt` - Summary report
- `outputs/inference/assessment_report_detailed.csv` - Detailed results

## Performance Tips

1. **GPU Acceleration**: Ensure CUDA is available for faster training
2. **Batch Size**: Increase if you have more GPU memory
3. **Transfer Learning**: Start with frozen backbone, then fine-tune
4. **Data Augmentation**: Essential for satellite imagery
5. **Model Selection**: 
   - `resnet18` for quick experiments
   - `resnet50` or `efficientnet_b0` for production

## Example Results

Typical accuracy on the hurricane damage dataset:
- ResNet-18: ~90-93% test accuracy
- ResNet-50: ~92-95% test accuracy
- EfficientNet-B0: ~91-94% test accuracy

## License

This project is for educational and research purposes.
