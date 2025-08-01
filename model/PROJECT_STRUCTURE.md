# Industrial Safety Equipment Detection - Project Structure

This document describes the structure and organization of the Industrial Safety Equipment Detection project.

## Overview

This project implements a YOLOv8-based object detection system for identifying critical industrial safety equipment:
- Fire Extinguishers
- Tool Boxes  
- Oxygen Tanks

The model is trained on synthetic data from Duality AI's Falcon digital twin simulator platform.

## Project Structure

```
model/
├── README.md                 # Main project documentation
├── requirements.txt          # Python dependencies
├── config.yaml              # Model configuration file
├── setup.py                 # Package setup script
├── setup.bat                # Windows setup batch script
├── run_model.bat            # Windows run batch script
├── run_model.ps1            # PowerShell run script
├── PROJECT_STRUCTURE.md     # This file
│
├── Core Scripts:
├── train_model.py           # Main training script
├── inference.py             # Inference script for predictions
├── evaluate_model.py        # Model evaluation and metrics
├── data_analysis.py         # Dataset analysis and visualization
├── utils.py                 # Utility functions
├── model_zoo.py             # Pre-trained model management
├── export_model.py          # Model export utilities
├── demo.py                  # Interactive GUI demo
│
├── Generated Directories:
├── runs/                    # Training runs and results
│   ├── train/              # Training outputs
│   │   └── industrial_safety_detection/
│   │       ├── weights/    # Model weights (best.pt, last.pt)
│   │       ├── results.png # Training curves
│   │       └── ...         # Other training artifacts
│   └── val/                # Validation outputs
│
├── exports/                 # Exported models (ONNX, TensorRT, etc.)
├── inference_results/       # Inference output images
├── analysis_results/        # Dataset analysis outputs
├── evaluation_results/      # Model evaluation outputs
├── venv/                   # Python virtual environment (created by setup)
└── models/                 # Downloaded pre-trained models cache
```

## File Descriptions

### Core Scripts

**train_model.py**
- Main training script for YOLOv8 model
- Supports various command-line arguments for customization
- Handles dataset validation, training setup, and result analysis
- Generates training curves and model checkpoints

**inference.py**
- Run inference on images, directories, or videos
- Supports batch processing and various output formats
- Includes confidence thresholding and NMS
- Provides detailed statistics and visualization

**evaluate_model.py**
- Comprehensive model evaluation on test/validation sets
- Calculates mAP, precision, recall, F1-score
- Generates performance plots and comparison charts
- Supports multi-model comparison

**data_analysis.py**
- Analyzes dataset structure and properties
- Creates visualizations of class distributions
- Generates sample image grids with annotations
- Provides detailed statistics and recommendations

**utils.py**
- Common utility functions used across scripts
- Image processing, coordinate transformations
- Visualization helpers, metrics calculations
- Configuration management

**model_zoo.py**
- Manages different YOLOv8 model variants
- Downloads and caches pre-trained weights
- Provides model recommendations based on requirements
- Benchmarking and comparison tools

**export_model.py**
- Exports trained models to various formats
- Supports ONNX, TensorRT, TorchScript, OpenVINO, etc.
- Includes optimization options and benchmarking
- Facilitates deployment to different platforms

**demo.py**
- Interactive GUI application for model demonstration
- Load images, adjust confidence thresholds
- Real-time visualization of detections
- Save results and generate reports

### Configuration Files

**config.yaml**
- Central configuration for model training
- Dataset paths, model architecture settings
- Training hyperparameters, augmentation options
- Logging and export configurations

**requirements.txt**
- Python package dependencies
- Includes YOLOv8 (ultralytics), PyTorch, OpenCV
- Visualization libraries (matplotlib, seaborn)
- Additional utilities for data processing

### Setup Scripts

**setup.py**
- Standard Python package setup configuration
- Defines package metadata and dependencies
- Creates console entry points for main scripts
- Enables pip installation of the package

**setup.bat / run_model.bat**
- Windows batch scripts for easy setup and execution
- Handles virtual environment creation and activation
- Installs dependencies and checks system requirements
- Provides convenient one-click operation

**run_model.ps1**
- PowerShell version of the run script
- Better error handling and colored output
- Cross-platform PowerShell compatibility

## Usage Workflow

### 1. Initial Setup
```bash
# Run setup script to install dependencies
setup.bat  # Windows
# or
python -m pip install -r requirements.txt
```

### 2. Dataset Analysis
```bash
python data_analysis.py --data ../HackByte_Dataset/yolo_params.yaml
```

### 3. Model Training
```bash
python train_model.py --epochs 100 --batch-size 16
```

### 4. Model Evaluation
```bash
python evaluate_model.py --model runs/train/industrial_safety_detection/weights/best.pt
```

### 5. Inference
```bash
python inference.py --model runs/train/industrial_safety_detection/weights/best.pt --source test_image.jpg
```

### 6. Interactive Demo
```bash
python demo.py --model runs/train/industrial_safety_detection/weights/best.pt
```

### 7. Model Export
```bash
python export_model.py --model runs/train/industrial_safety_detection/weights/best.pt --formats onnx tensorrt
```

## Generated Outputs

### Training Outputs (`runs/train/`)
- `weights/best.pt` - Best model checkpoint
- `weights/last.pt` - Latest model checkpoint  
- `results.png` - Training/validation curves
- `confusion_matrix.png` - Confusion matrix
- `train_batch*.jpg` - Training batch samples
- `val_batch*.jpg` - Validation predictions

### Analysis Outputs (`analysis_results/`)
- `dataset_overview.png` - Dataset statistics visualization
- `detailed_analysis.png` - Detailed dataset analysis
- `sample_images.png` - Sample images with annotations
- `analysis_report.txt` - Comprehensive text report
- `class_distribution.png` - Class distribution chart

### Evaluation Outputs (`evaluation_results/`)
- `evaluation_metrics.txt` - Detailed metrics report
- `overall_metrics.png` - Overall performance chart
- `per_class_map.png` - Per-class mAP comparison
- `model_comparison.csv` - Multi-model comparison results

### Inference Outputs (`inference_results/`)
- Annotated images with detected objects
- Bounding boxes and confidence scores
- Results summary and statistics

## Dependencies

### Core Dependencies
- `ultralytics>=8.3.0` - YOLOv8 implementation
- `torch>=2.0.0` - PyTorch deep learning framework
- `torchvision>=0.15.0` - Computer vision utilities
- `opencv-python>=4.8.0` - Image processing

### Visualization & Analysis
- `matplotlib>=3.7.0` - Plotting and visualization
- `seaborn>=0.12.0` - Statistical visualization
- `pandas>=2.0.0` - Data manipulation
- `plotly>=5.15.0` - Interactive plots

### Utilities
- `numpy>=1.24.0` - Numerical computing
- `Pillow>=10.0.0` - Image processing
- `PyYAML>=6.0` - YAML configuration files
- `tqdm>=4.65.0` - Progress bars
- `scikit-learn>=1.3.0` - Machine learning utilities

## Hardware Requirements

### Minimum Requirements
- CPU: Multi-core processor (Intel i5/AMD Ryzen 5 or better)
- RAM: 8GB minimum, 16GB recommended
- Storage: 10GB free space for models and datasets
- Python: 3.8 or later

### Recommended for Training
- GPU: NVIDIA GPU with 6GB+ VRAM (GTX 1060/RTX 2060 or better)
- CUDA: Compatible CUDA installation for GPU acceleration
- RAM: 16GB+ for larger batch sizes
- Storage: SSD recommended for faster data loading

### For Inference Only
- CPU-only operation is supported
- GPU recommended for real-time applications
- Minimum 4GB RAM for inference

## Performance Expectations

### Training Time (approximate)
- YOLOv8n: 2-4 hours (100 epochs, batch size 16)
- YOLOv8s: 3-6 hours (100 epochs, batch size 16)  
- YOLOv8m: 4-8 hours (100 epochs, batch size 16)

### Inference Speed (approximate)
- YOLOv8n: 20-50 FPS (GPU), 5-10 FPS (CPU)
- YOLOv8s: 15-35 FPS (GPU), 3-7 FPS (CPU)
- YOLOv8m: 10-25 FPS (GPU), 2-5 FPS (CPU)

### Model Accuracy (expected)
- mAP@0.5: 0.7-0.9 (depending on model size and training data)
- Per-class performance varies based on data distribution
- Synthetic data may require domain adaptation for real-world use

## Customization Options

### Model Architecture
- Choose from YOLOv8n/s/m/l/x variants
- Adjust input resolution (320, 640, 1280)
- Modify backbone and neck architectures

### Training Configuration  
- Learning rate scheduling
- Data augmentation parameters
- Loss function weights
- Batch size and epochs

### Inference Settings
- Confidence and IoU thresholds
- Non-Maximum Suppression parameters
- Input preprocessing options
- Output post-processing

## Troubleshooting

### Common Issues
1. **CUDA out of memory** - Reduce batch size or use smaller model
2. **Slow training** - Enable GPU acceleration, increase workers
3. **Poor accuracy** - Increase epochs, adjust learning rate, check data quality
4. **Import errors** - Verify all dependencies installed correctly

### Performance Optimization
1. Use GPU acceleration when available
2. Optimize batch size for your hardware
3. Use mixed precision training (AMP)
4. Enable multi-GPU training for large datasets

### Data Issues
1. Verify dataset format and paths
2. Check class balance and distribution
3. Validate annotation quality
4. Consider data augmentation strategies
