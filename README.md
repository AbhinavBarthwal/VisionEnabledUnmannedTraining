# Industrial Safety Equipment Detection - YOLOv8 Model

## 🎯 Project Overview

This project implements a **YOLOv8-based object detection system** for identifying critical industrial safety equipment in workplace environments. The model is trained on synthetic data from **Duality AI's Falcon digital twin simulator platform** to detect three essential safety equipment categories:

- 🧯 **Fire Extinguisher** (Class 0)
- 🧰 **Tool Box** (Class 1)
- 🔥 **Oxygen Tank** (Class 2)


## 📊 Performance Metrics (YOLOv8n Final Model)

- **Precision:** 94.4%

- **Recall:** 91.4%

- **mAP@0.5:** 95.1%

- **Training Time:** ~4.7 hours (CPU)

## 📊 Dataset Information

### Dataset Statistics
- **Total Images**: 1,400
- **Training Set**: 846 images (60.4%)
- **Validation Set**: 154 images (11.0%)
- **Test Set**: 400 images (28.6%)
- **Classes**: 3 object categories
- **Format**: YOLO format with normalized bounding box coordinates

### Data Source
The dataset is synthetic data generated from **Duality AI's Falcon** digital twin simulator platform, providing high-quality, diverse training samples for industrial safety equipment detection.

## 🚀 Quick Start Guide

### 1. Setup Environment
```bash
# Navigate to model directory
cd model

# Install dependencies (Windows)
pip install -r requirements.txt

# Or use the automated setup script
setup.bat
```

### 2. Analyze Dataset
```bash
python data_analysis.py --data ../HackByte_Dataset/yolo_params.yaml
```

### 3. Train Model
```bash
python train_model.py --epochs 100 --batch-size 16
```

### 4. Run Inference
```bash
python inference.py --model runs/train/industrial_safety_detection/weights/best.pt --source test_image.jpg
```

### 5. Interactive Demo
```bash
python demo.py --model runs/train/industrial_safety_detection/weights/best.pt
```

## 🏗️ Project Structure

```
model/
├── 📋 Core Scripts
│   ├── train_model.py           # Main training script
│   ├── inference.py             # Inference and prediction
│   ├── evaluate_model.py        # Model evaluation and metrics
│   ├── data_analysis.py         # Dataset analysis and visualization
│   ├── demo.py                  # Interactive GUI demo
│   ├── export_model.py          # Model export utilities
│   ├── model_zoo.py             # Pre-trained model management
│   └── utils.py                 # Utility functions
│
├── ⚙️ Configuration
│   ├── config.yaml              # Model configuration
│   ├── requirements.txt         # Python dependencies
│   └── setup.py                 # Package setup
│
├── 🔧 Setup Scripts
│   ├── setup.bat                # Windows setup script
│   ├── run_model.bat            # Windows run script
│   └── run_model.ps1            # PowerShell run script
│
└── 📁 Generated Directories
    ├── runs/                    # Training outputs
    ├── exports/                 # Exported models
    ├── analysis_results/        # Dataset analysis
    └── inference_results/       # Inference outputs
```

## 📈 Model Performance

### Expected Metrics
- **mAP@0.5**: 0.7-0.9 (depending on model variant)
- **Inference Speed**: 
  - YOLOv8n: 20-50 FPS (GPU), 5-10 FPS (CPU)
  - YOLOv8s: 15-35 FPS (GPU), 3-7 FPS (CPU)
  - YOLOv8m: 10-25 FPS (GPU), 2-5 FPS (CPU)

### Model Variants Available
| Model | Parameters | Speed | Accuracy | Use Case |
|-------|------------|-------|----------|----------|
| YOLOv8n | 3.2M | ⚡ Fastest | ⭐ Good | Real-time applications |
| YOLOv8s | 11.2M | ⚡ Fast | ⭐⭐ Better | Balanced performance |
| YOLOv8m | 25.9M | ⚡ Moderate | ⭐⭐⭐ High | Higher accuracy needs |
| YOLOv8l | 43.7M | ⚡ Slower | ⭐⭐⭐⭐ Higher | Demanding applications |
| YOLOv8x | 68.2M | ⚡ Slowest | ⭐⭐⭐⭐⭐ Highest | Maximum accuracy |

## 🛠️ Key Features

### Training Features
- ✅ **Automated dataset validation**
- ✅ **Comprehensive data augmentation**
- ✅ **Real-time training monitoring**
- ✅ **Automatic best model saving**
- ✅ **Early stopping and learning rate scheduling**
- ✅ **Multi-GPU support**

### Inference Capabilities
- ✅ **Single image, batch, and video processing**
- ✅ **Confidence and IoU threshold adjustment**
- ✅ **Multiple output formats (images, JSON, txt)**
- ✅ **Real-time visualization**
- ✅ **Performance statistics**

### Analysis Tools
- ✅ **Dataset structure analysis**
- ✅ **Class distribution visualization**
- ✅ **Sample image inspection**
- ✅ **Model performance metrics**
- ✅ **Comprehensive reporting**

### Export Options
- ✅ **ONNX** - Cross-platform deployment
- ✅ **TensorRT** - NVIDIA GPU optimization
- ✅ **TorchScript** - PyTorch native format
- ✅ **OpenVINO** - Intel hardware optimization
- ✅ **CoreML** - iOS deployment
- ✅ **TensorFlow Lite** - Mobile/edge deployment

## 💻 Hardware Requirements

### Minimum Requirements
- **CPU**: Multi-core processor (Intel i5/AMD Ryzen 5 or better)
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 10GB free space
- **Python**: 3.8 or later

### Recommended for Training
- **GPU**: NVIDIA GPU with 6GB+ VRAM
- **CUDA**: Compatible CUDA installation
- **RAM**: 16GB+ for larger batch sizes
- **Storage**: SSD for faster data loading

## 🎮 Usage Examples

### Command Line Training
```bash
# Basic training
python train_model.py

# Custom configuration
python train_model.py --epochs 200 --batch-size 32 --lr 0.001

# Resume from checkpoint
python train_model.py --resume runs/train/last_experiment/weights/last.pt
```

### Batch Inference
```bash
# Process directory of images
python inference.py --model best.pt --source ./images/ --save

# Process video
python inference.py --model best.pt --source video.mp4 --save
```

### Model Evaluation
```bash
# Evaluate on test set
python evaluate_model.py --model best.pt --data ../HackByte_Dataset/yolo_params.yaml

# Compare multiple models
python evaluate_model.py --model best.pt --compare model1.pt model2.pt
```

## 📊 Analysis Results

The dataset analysis revealed:

### Class Distribution
- **Balanced dataset** with good representation of all three classes
- **High-quality synthetic data** from Falcon simulator
- **Consistent annotation quality** across all splits

### Image Properties
- **Consistent resolution** across dataset
- **Varied lighting and viewing angles**
- **Realistic industrial environments**

### Recommendations
- **Data augmentation** already configured for robust training
- **Balanced training approach** suitable for all classes
- **Standard YOLO training configuration** optimal for this dataset

## 🔧 Customization Options

### Model Architecture
```yaml
# In config.yaml
model:
  architecture: "yolov8s"  # Change to n, s, m, l, or x
  input_size: [640, 640]   # Adjust input resolution
```

### Training Parameters
```yaml
training:
  epochs: 100
  batch_size: 16
  learning_rate: 0.01
  # Extensive augmentation options available
```

### Inference Settings
```yaml
validation:
  conf_threshold: 0.25    # Confidence threshold
  iou_threshold: 0.45     # NMS IoU threshold
  max_det: 300           # Maximum detections per image
```

## 🚨 Safety Applications

This model is designed for **industrial safety compliance** and can be used in:

- **🏭 Manufacturing facilities** - Ensuring safety equipment availability
- **🔧 Construction sites** - Monitoring tool and safety equipment presence
- **⚗️ Chemical plants** - Tracking safety equipment in hazardous areas
- **🚛 Warehouses** - Automated safety equipment inventory
- **📹 Security systems** - Real-time safety compliance monitoring


##📊 Performance Metrics (YOLOv8n Final Model)

-**Precision:** 94.4%

-**Recall:** 91.4%

-**mAP@0.5:** 95.1%

-**mAP@0.5: **0.95: 67.0%

-**Training Time:** ~4.7 hours (CPU)
