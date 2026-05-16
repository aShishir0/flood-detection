### Data Access
Data is sourced from Kaggle datasets:
- Sen1Floods11 Essentials: https://www.kaggle.com/datasets/smabrarrajin/sen1floods11-essentials
- JRC Water Hand: https://www.kaggle.com/datasets/shishiradh/jrcwaterhand
- Checkpoint: https://www.kaggle.com/datasets/shishiradh/aquilla-checkpoint

- KaggleNotebook : https://www.kaggle.com/code/shishiradh/flood-detection-with-unetpp

Ensure paths are configured correctly in the notebook.



# Flood Detection with UNet++ and Sentinel-1 SAR

A deep learning system for automatic flood water detection using Sentinel-1 Synthetic Aperture Radar (SAR) imagery. This project implements **UNet++** with an **EfficientNet-B4** encoder to perform 3-class semantic segmentation on SAR data, distinguishing between background, permanent water, and flood water.

## Overview

### Problem Statement
Flood detection from satellite imagery is critical for disaster response and monitoring. SAR imagery is particularly valuable because it:
- Works in all weather conditions (no cloud obstruction)
- Can be acquired day/night
- Provides reliable water detection capability

This project automates flood water identification by separating it from permanent water bodies using hand-labeled training data and JRC Global Surface Water reference masks.

### Key Features
- **3-Class Segmentation**: Background, Permanent Water, Flood Water
- **Robust Architecture**: UNet++ with EfficientNet-B4 backbone
- **Advanced Loss Function**: Focal Loss + Dice Loss for class imbalance handling
- **Complete Pipeline**: Data loading → augmentation → training → validation → testing
- **Rich Visualization**: Themed plots for SAR, Sen1Floods11 label, JRC permanent water label,      ground truth, and prediction

## Dataset

### Source: Sen1Floods11
- **Sentinel-1 SAR**: VV and VH polarization (2 channels)
- **Spatial Resolution**: ~10m per pixel
- **Tile Size**: 512×512 pixels per scene
- **Splits**: Train, Validation, Test

### Label Sources
1. **LabelHand**: Hand-labeled water masks {-1 (ignore), 0 (background), 1 (water)}
2. **JRCWaterHand**: JRC Global Surface Water permanent water masks {0, 1}

### 3-Class Label Fusion
```
Class 0 (Background):      LabelHand==0
Class 1 (Permanent Water): LabelHand==1 AND JRCWaterHand==1
Class 2 (Flood Water):     LabelHand==1 AND JRCWaterHand==0
Class -1 (Ignore):         LabelHand==-1
```

## Model Architecture

### UNet++
- **Encoder**: EfficientNet-B4 (ImageNet pre-trained)
- **Decoder**: Dense skip connections with recursive refinement
- **Input Channels**: 3 (VV, VH, JRC)
- **Output Classes**: 3 (Background, Permanent, Flood)
- **Activation**: Softmax for multi-class segmentation

### Loss Function: FocalDiceLoss
Combines two complementary losses:
- **Focal Loss** (50% weight): Addresses extreme class imbalance by down-weighting easy examples
- **Dice Loss** (50% weight): Directly optimizes IoU-like metric

```python
FocalDiceLoss = 0.5 × FocalLoss + 0.5 × DiceLoss
```

## Data Processing

### Input Preprocessing
1. **Clipping & Scaling**: VV and VH channels clipped to dB ranges and normalized to [0, 1]
   - VV: [-18.6, -4.3] dB
   - VH: [-26.7, -10.7] dB
2. **Normalization**: Channel-wise standardization
   - Mean: [0.6851, 0.5235]
   - Std: [0.0820, 0.1102]

### Training Augmentations
- **Random Crops**: 256×256 patches
- **Flips**: Horizontal and vertical (50% probability each)
- **Rotation**: ±15° (50% probability)
- **Brightness/Contrast**: ±10% scaling (50% probability each)
- **Noise**: Gaussian noise σ=0.02 (50% probability)

### Test Processing
- **Tiling**: 512×512 scenes split into four 256×256 crops
- **No Augmentation**: Images processed as-is for evaluation

## Training

### Hyperparameters
```python
Learning Rate: 1e-4
Optimizer: AdamW
Scheduler: ReduceLROnPlateau (mode='max', factor=0.5, patience=5)
Batch Size: 8
Max Epochs: 1000
Checkpoint: Saved on validation IoU improvement
```

### Metrics
- **IoU** (Intersection over Union): Macro mean across Permanent + Flood classes (excluding Background)
- **F1 Score**: Macro mean across all valid classes
- **Pixel Accuracy**: Valid pixels correctly classified
- **Loss**: Focal + Dice combined

### Early Stopping
Model checkpoints saved when validation IoU exceeds previous best, with F1 score and epoch recorded.

## Usage

### Requirements
```bash
pip install torch torchvision
pip install segmentation-models-pytorch
pip install rasterio numpy pandas scikit-learn matplotlib
```

### Training
```python
# Load data
train_data = load_flood_train_data(input_root, label_root)
train_dataset = InMemoryDataset(train_data, processAndAugment)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, ...)

# Initialize model
net = smp.UnetPlusPlus(
    encoder_name="efficientnet-b4",
    encoder_weights="imagenet",
    in_channels=3,
    classes=3
)

# Train
train_validation_loop(
    net, optimizer, scheduler,
    train_loader, valid_loader,
    num_epochs=1000,
    device="cuda"
)
```

### Testing & Evaluation
```python
avg_loss, avg_iou, avg_f1, avg_acc = test_loop(
    test_loader,
    net,
    device="cuda",
    num_show=10  # Show 10 random sample predictions
)

print(f"Avg IoU: {avg_iou:.4f} | Avg F1: {avg_f1:.4f}")
```

## Visualization

### Sample Predictions (4-Panel Layout)
Each test sample displays:
- **A: SAR** - Gray-scale VV polarization for reference
- **B: Hand Label** - Ground truth water mask provided by sen1flood11 dataset
- **C: JRC** - Ground truth permanent water reference
- **D: Ground Truth** - Target label created by fusing Hand label(total water) and JRC(perm water)
- **E: Prediction** - Model prediction (Blue=Permanent, Red=Flood)

### Confusion Matrix
Per-class breakdown with percentages and pixel counts, styled with dark theme for clarity.


## Project Structure

```
flood-detection/
├── flood-detection-with-unetpp.ipynb  # Main notebook (complete pipeline)
├── README.md                           # This file
└── checkpoints/                        # Saved model weights (generated)
    └── Sen1Floods11_epoch{N}_iou{X}_f1{Y}.pt
```

## Key Results & Insights

### Class Imbalance
The training set exhibits significant imbalance:
- **Background**: ~97% of valid pixels
- **Permanent Water**: ~1-2% of valid pixels
- **Flood Water**: ~1-2% of valid pixels

Mitigation strategies:
- Focal Loss prioritizes minority classes
- Dice Loss provides balance-agnostic optimization
- Ignore index (-1) for uncertain regions

### Model Advantages
1. **EfficientNet-B4**: Superior feature extraction and generalization
2. **UNet++**: Dense skip connections enable multi-scale fusion
3. **Two-Stage Loss**: Complements Focal Loss with Dice for robust optimization

## Performance Evaluation

### Metrics Definition
- **IoU**: Jaccard similarity, computed per-class then macro-averaged (excluding background)
- **F1**: Harmonic mean of precision and recall, macro-averaged
- **Accuracy**: Pixel-level classification accuracy on valid pixels

### Test Loop Features
- **Random Sampling**: Selects `num_show` random samples instead of sequential
- **Metrics First**: Prints aggregated statistics before visualizations
- **Comprehensive Reporting**: Loss, IoU, F1, Accuracy

## Installation & Setup

### Clone & Install
```bash
cd flood-detection
pip install -r requirements.txt  # If available
# OR manually install key packages
pip install torch segmentation-models-pytorch rasterio scikit-learn
```

## References

- **UNet++**: Huang et al., "UNet++: A Nested U-Net Architecture for Medical Image Segmentation"
- **EfficientNet**: Tan & Le, "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks"
- **Focal Loss**: Lin et al., "Focal Loss for Dense Object Detection"
- **Sen1Floods11**: https://github.com/torchvision/models
- **segmentation-models-pytorch**: https://github.com/qubvel-org/segmentation_models.pytorch

## License

This project uses publicly available datasets. Adapt licensing as needed for your use case.

## Contact & Contributions

For questions or improvements, refer to the project documentation or reach out through standard channels. Contributions are welcome!

---

**Last Updated**: May 2026  
**Model Name**: Sen1Floods11 with UNet++  
**Framework**: PyTorch + segmentation-models-pytorch
