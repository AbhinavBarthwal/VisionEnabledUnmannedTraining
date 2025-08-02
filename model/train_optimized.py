#!/usr/bin/env python3
"""
Optimized Training Script for Industrial Safety Equipment Detection
Focus: Achieving HIGH PRECISION and HIGH RECALL
"""

import os
import yaml
from ultralytics import YOLO
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def setup_training_environment():
    """Setup optimized training environment"""
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Configure PyTorch for best CPU performance
    torch.set_num_threads(os.cpu_count())
    
    print("🚀 Training Environment Setup Complete")
    print(f"   - PyTorch Version: {torch.__version__}")
    print(f"   - CPU Threads: {torch.get_num_threads()}")

def create_optimized_config():
    """Create optimized training configuration"""
    config = {
        # Basic training parameters - OPTIMIZED FOR HIGH PERFORMANCE
        'epochs': 100,          # Increased for better convergence
        'batch': 16,            # Optimal batch size for CPU
        'imgsz': 640,
        'device': 'cpu',
        'workers': 8,
        
        # Learning rate optimization for precision/recall
        'lr0': 0.003,           # Lower initial LR for fine-tuning
        'lrf': 0.0001,          # Much lower final LR
        'momentum': 0.95,       # Higher momentum for stability
        'weight_decay': 0.001,  # Regularization
        
        # Advanced training strategies
        'optimizer': 'AdamW',   # Better optimizer for precision
        'cos_lr': True,         # Cosine LR scheduler
        'warmup_epochs': 5,     # Gradual warmup
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        
        # Loss function weights - OPTIMIZED FOR PRECISION/RECALL
        'box': 8.0,             # Higher box regression weight
        'cls': 1.0,             # Higher classification weight
        'dfl': 1.5,             # Distribution focal loss weight
        
        # Augmentation - ENHANCED FOR ROBUSTNESS
        'hsv_h': 0.03,          # Color augmentation
        'hsv_s': 0.9,
        'hsv_v': 0.6,
        'degrees': 8.0,         # Rotation for robustness
        'translate': 0.2,       # Translation
        'scale': 0.7,           # Scale variation
        'shear': 3.0,           # Shear transformation
        'perspective': 0.0005,  # Perspective transformation
        'flipud': 0.0,          # No vertical flip (equipment orientation)
        'fliplr': 0.5,          # Horizontal flip
        'mosaic': 1.0,          # Mosaic augmentation
        'mixup': 0.2,           # Mixup for generalization
        'copy_paste': 0.4,      # Copy-paste augmentation
        
        # Validation and saving
        'val': True,
        'plots': True,
        'save': True,
        'save_period': 10,      # Save every 10 epochs
        'patience': 20,         # Early stopping patience
        
        # Model-specific optimizations
        'close_mosaic': 15,     # Disable mosaic in final epochs
        'amp': False,           # Disable AMP for CPU training
        'fraction': 1.0,        # Use entire dataset
    }
    
    return config

def train_optimized_model():
    """Train model with optimized configuration for high precision/recall"""
    
    print("🔧 Setting up optimized training environment...")
    setup_training_environment()
    
    # Initialize model
    print("📦 Loading YOLOv8 model...")
    model = YOLO('yolov8n.pt')  # Start with nano for faster iteration
    
    # Get optimized configuration
    config = create_optimized_config()
    
    # Data path
    data_path = r"C:\Users\user\OneDrive\Documents\pro\sambhav\HackByte_Dataset\yolo_params.yaml"
    
    print("🎯 Starting OPTIMIZED TRAINING for High Precision & Recall")
    print("=" * 60)
    print(f"📊 Training Configuration:")
    print(f"   - Epochs: {config['epochs']}")
    print(f"   - Batch Size: {config['batch']}")
    print(f"   - Learning Rate: {config['lr0']} → {config['lrf']}")
    print(f"   - Optimizer: {config['optimizer']}")
    print(f"   - Box Loss Weight: {config['box']}")
    print(f"   - Classification Loss Weight: {config['cls']}")
    print("=" * 60)
    
    try:
        # Start training with optimized parameters
        results = model.train(
            data=data_path,
            name='industrial_safety_optimized',
            **config
        )
        
        print("✅ TRAINING COMPLETED SUCCESSFULLY!")
        print(f"📈 Results saved to: {results.save_dir}")
        
        return results
        
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        return None

def evaluate_model_performance(model_path):
    """Evaluate trained model performance"""
    print("\n🔍 Evaluating Model Performance...")
    
    try:
        model = YOLO(model_path)
        
        # Validate model
        results = model.val(
            data=r"C:\Users\user\OneDrive\Documents\pro\sambhav\HackByte_Dataset\yolo_params.yaml",
            conf=0.25,  # Lower confidence for higher recall
            iou=0.45,   # Standard IoU threshold
            save_json=True,
            plots=True
        )
        
        print(f"📊 Model Performance Metrics:")
        print(f"   - Overall Precision: {results.box.mp:.1%}")
        print(f"   - Overall Recall: {results.box.mr:.1%}")
        print(f"   - mAP@0.5: {results.box.map50:.1%}")
        print(f"   - mAP@0.5-0.95: {results.box.map:.1%}")
        
        # Per-class metrics
        class_names = ["FireExtinguisher", "ToolBox", "OxygenTank"]
        for i, class_name in enumerate(class_names):
            if i < len(results.box.ap_class_index):
                print(f"   - {class_name}: P={results.box.ap[i]:.1%}, R={results.box.ar[i]:.1%}")
        
        return results
        
    except Exception as e:
        print(f"❌ Evaluation failed: {str(e)}")
        return None

def main():
    """Main training pipeline"""
    print("🏭 INDUSTRIAL SAFETY EQUIPMENT DETECTION - OPTIMIZED TRAINING")
    print("🎯 Goal: Achieve >90% Precision and >80% Recall")
    print("=" * 70)
    
    # Train the model
    results = train_optimized_model()
    
    if results:
        # Evaluate the best model
        best_model_path = results.save_dir / 'weights' / 'best.pt'
        if best_model_path.exists():
            evaluate_model_performance(best_model_path)
        else:
            print("⚠️ Best model not found, checking last model...")
            last_model_path = results.save_dir / 'weights' / 'last.pt'
            if last_model_path.exists():
                evaluate_model_performance(last_model_path)
    
    print("\n🎉 Training pipeline completed!")
    print("📁 Check 'runs/detect/industrial_safety_optimized' for results")

if __name__ == "__main__":
    main()
