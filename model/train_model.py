"""
Main training script for industrial safety equipment detection using YOLOv8.

This script trains a YOLOv8 model on the synthetic dataset from Duality AI's Falcon platform
to detect FireExtinguisher, ToolBox, and OxygenTank objects.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Optional

import torch
import yaml
from ultralytics import YOLO
import matplotlib.pyplot as plt

from utils import (
    setup_logging, 
    load_config, 
    create_directories, 
    get_device,
    print_dataset_summary,
    create_class_distribution_plot
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train YOLOv8 model for industrial safety equipment detection")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--data", type=str, default="../HackByte_Dataset/yolo_params.yaml", help="Path to dataset YAML")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Model architecture (yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--device", type=str, default="auto", help="Device to use (auto, cpu, 0, 1, etc.)")
    parser.add_argument("--workers", type=int, default=8, help="Number of worker threads")
    parser.add_argument("--project", type=str, default="runs/train", help="Project directory")
    parser.add_argument("--name", type=str, default="industrial_safety_detection", help="Experiment name")
    parser.add_argument("--resume", type=str, default=None, help="Resume training from checkpoint")
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained weights")
    parser.add_argument("--save-period", type=int, default=10, help="Save checkpoint every N epochs")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    return parser.parse_args()


def validate_dataset(data_path: str) -> bool:
    """Validate that the dataset exists and has the correct structure."""
    data_path = Path(data_path)
    
    if not data_path.exists():
        logging.error(f"Dataset YAML file not found: {data_path}")
        return False
    
    # Load and validate YAML
    try:
        with open(data_path, 'r') as f:
            data_config = yaml.safe_load(f)
        
        required_keys = ['train', 'val', 'nc', 'names']
        for key in required_keys:
            if key not in data_config:
                logging.error(f"Missing required key '{key}' in dataset YAML")
                return False
        
        # Check if paths exist (relative to YAML file location)
        base_path = data_path.parent
        train_path = base_path / data_config['train']
        val_path = base_path / data_config['val']
        
        if not train_path.exists():
            logging.error(f"Training images directory not found: {train_path}")
            return False
            
        if not val_path.exists():
            logging.error(f"Validation images directory not found: {val_path}")
            return False
        
        logging.info(f"Dataset validation successful!")
        logging.info(f"Classes: {data_config['names']}")
        logging.info(f"Number of classes: {data_config['nc']}")
        
        return True
        
    except Exception as e:
        logging.error(f"Error reading dataset YAML: {e}")
        return False


def setup_training_environment(config: Dict, args) -> Dict:
    """Set up the training environment and parameters."""
    
    # Create output directories
    create_directories([args.project, f"{args.project}/{args.name}"])
    
    # Setup device
    if args.device == "auto":
        device = get_device()
        device_str = str(device)
    else:
        device_str = args.device
    
    logging.info(f"Using device: {device_str}")
    
    # Prepare training parameters
    train_params = {
        'data': args.data,
        'epochs': args.epochs or config['training']['epochs'],
        'batch': args.batch_size or config['training']['batch_size'],
        'lr0': args.lr or config['training']['learning_rate'],
        'imgsz': args.imgsz,
        'device': device_str,
        'workers': args.workers,
        'project': args.project,
        'name': args.name,
        'save_period': args.save_period,
        'patience': config['training']['patience'],
        'optimizer': config['training']['optimizer'],
        'momentum': config['training']['momentum'],
        'weight_decay': config['training']['weight_decay'],
        'warmup_epochs': config['training']['warmup_epochs'],
        'verbose': args.verbose,
        'seed': 42,  # For reproducibility
        'deterministic': True,
        'single_cls': False,
        'rect': False,
        'cos_lr': True,
        'close_mosaic': 10,
        'resume': args.resume,
        'amp': True,  # Automatic Mixed Precision
        'fraction': 1.0,
        'profile': False,
        'freeze': None,
        'multi_scale': False,
        'overlap_mask': True,
        'mask_ratio': 4,
        'dropout': 0.0,
        'val': True,
        'plots': True,
        'save': True,
        'save_json': False,
        'save_hybrid': False,
        'conf': config['validation']['conf_threshold'],
        'iou': config['validation']['iou_threshold'],
        'max_det': config['validation']['max_det'],
        'half': False,
        'dnn': False,
        'exist_ok': True,
    }
    
    # Add augmentation parameters
    aug_config = config['training']['augmentation']
    train_params.update({
        'hsv_h': aug_config['hsv_h'],
        'hsv_s': aug_config['hsv_s'],
        'hsv_v': aug_config['hsv_v'],
        'degrees': aug_config['degrees'],
        'translate': aug_config['translate'],
        'scale': aug_config['scale'],
        'shear': aug_config['shear'],
        'perspective': aug_config['perspective'],
        'flipud': aug_config['flipud'],
        'fliplr': aug_config['fliplr'],
        'mosaic': aug_config['mosaic'],
        'mixup': aug_config['mixup'],
        'copy_paste': aug_config['copy_paste'],
    })
    
    return train_params


def train_model(model_path: str, train_params: Dict) -> YOLO:
    """Train the YOLO model."""
    
    # Load model
    logging.info(f"Loading model: {model_path}")
    model = YOLO(model_path)
    
    # Log model info
    logging.info(f"Model: {model_path}")
    logging.info(f"Parameters: {sum(p.numel() for p in model.model.parameters()):,}")
    
    # Start training
    logging.info("Starting training...")
    logging.info(f"Training parameters: {train_params}")
    
    try:
        results = model.train(**train_params)
        logging.info("Training completed successfully!")
        return model, results
        
    except Exception as e:
        logging.error(f"Training failed: {e}")
        raise


def analyze_training_results(results, save_dir: str):
    """Analyze and visualize training results."""
    save_dir = Path(save_dir)
    
    try:
        # Training curves are automatically saved by ultralytics
        # We can add custom analysis here if needed
        
        logging.info(f"Training results saved to: {save_dir}")
        logging.info("Check the results directory for:")
        logging.info("  - Training curves (train_batch*.jpg, results.png)")
        logging.info("  - Validation predictions (val_batch*.jpg)")
        logging.info("  - Confusion matrix (confusion_matrix.png)")
        logging.info("  - Model weights (best.pt, last.pt)")
        
    except Exception as e:
        logging.warning(f"Error analyzing results: {e}")


def main():
    """Main training function."""
    
    # Parse arguments
    args = parse_args()
    
    # Setup logging
    logger = setup_logging("INFO" if not args.verbose else "DEBUG")
    
    # Load configuration
    try:
        config = load_config(args.config)
        logging.info(f"Configuration loaded from: {args.config}")
    except Exception as e:
        logging.error(f"Error loading configuration: {e}")
        return
    
    # Validate dataset
    if not validate_dataset(args.data):
        logging.error("Dataset validation failed!")
        return
    
    # Print dataset summary
    class_names = ["FireExtinguisher", "ToolBox", "OxygenTank"]
    print_dataset_summary(args.data, class_names)
    
    # Create class distribution plot
    try:
        label_dir = Path(args.data).parent / "data" / "train" / "labels"
        if label_dir.exists():
            create_class_distribution_plot(
                str(label_dir), 
                class_names, 
                save_path="class_distribution.png"
            )
    except Exception as e:
        logging.warning(f"Could not create class distribution plot: {e}")
    
    # Setup training environment
    train_params = setup_training_environment(config, args)
    
    # Determine model path
    if args.pretrained or config['model']['pretrained']:
        model_path = f"{config['model']['architecture']}.pt"
    else:
        model_path = config['model']['architecture']
    
    # Train model
    try:
        model, results = train_model(model_path, train_params)
        
        # Analyze results
        results_dir = Path(args.project) / args.name
        analyze_training_results(results, results_dir)
        
        logging.info("="*60)
        logging.info("TRAINING COMPLETED SUCCESSFULLY!")
        logging.info("="*60)
        logging.info(f"Best model saved to: {results_dir / 'weights' / 'best.pt'}")
        logging.info(f"Last model saved to: {results_dir / 'weights' / 'last.pt'}")
        logging.info(f"Results directory: {results_dir}")
        logging.info("="*60)
        
        # Run validation on best model
        logging.info("Running final validation...")
        best_model = YOLO(results_dir / 'weights' / 'best.pt')
        val_results = best_model.val()
        
        logging.info(f"Final validation mAP@0.5: {val_results.box.map50:.4f}")
        logging.info(f"Final validation mAP@0.5:0.95: {val_results.box.map:.4f}")
        
    except KeyboardInterrupt:
        logging.info("Training interrupted by user")
    except Exception as e:
        logging.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
