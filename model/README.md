# Object Detection Model for Industrial Safety Equipment

This directory contains a YOLOv8-based object detection model trained to detect three critical industrial safety equipment categories:

- **Fire Extinguisher** (Class 0)
- **Tool Box** (Class 1) 
- **Oxygen Tank** (Class 2)

The model is trained on synthetic data from Duality AI's digital twin simulator platform - Falcon.

## Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the Model**:
   ```bash
   python train_model.py
   ```

3. **Run Inference**:
   ```bash
   python inference.py --source path/to/image.jpg
   ```

4. **Evaluate Model**:
   ```bash
   python evaluate_model.py
   ```

## Project Structure

- `train_model.py` - Main training script
- `inference.py` - Inference script for predictions
- `evaluate_model.py` - Model evaluation and metrics
- `data_analysis.py` - Dataset analysis and visualization
- `utils.py` - Utility functions
- `config.yaml` - Model configuration
- `model_zoo.py` - Pre-trained model management
- `export_model.py` - Model export utilities
- `demo.py` - Interactive demo script

## Model Configuration

The model uses YOLOv8 architecture optimized for industrial safety equipment detection with:
- Input resolution: 640x640
- Batch size: 16
- Learning rate: 0.01
- Epochs: 100

## Dataset Information

- **Training images**: 846 images
- **Validation images**: Available in validation set
- **Test images**: Available in test set
- **Classes**: 3 (FireExtinguisher, ToolBox, OxygenTank)
- **Format**: YOLO format with normalized coordinates

## Performance Metrics

The model will be evaluated on:
- mAP@0.5 (mean Average Precision at IoU threshold 0.5)
- mAP@0.5:0.95 (mean Average Precision across multiple IoU thresholds)
- Precision and Recall per class
- Inference speed (FPS)
