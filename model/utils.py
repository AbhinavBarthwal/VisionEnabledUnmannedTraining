"""
Utility functions for the industrial safety equipment detection model.
"""

import os
import yaml
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import cv2
from PIL import Image
import torch


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('./training.log')
        ]
    )
    return logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def save_config(config: Dict, config_path: str = "config.yaml") -> None:
    """Save configuration to YAML file."""
    with open(config_path, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)


def create_directories(paths: List[str]) -> None:
    """Create directories if they don't exist."""
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def get_device() -> torch.device:
    """Get the best available device (CUDA, MPS, or CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def count_parameters(model) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def visualize_predictions(image_path: str, predictions: List, class_names: List[str], 
                         confidence_threshold: float = 0.5, save_path: Optional[str] = None):
    """Visualize predictions on an image."""
    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    
    # Create figure
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    plt.axis('off')
    
    # Draw bounding boxes
    for pred in predictions:
        if len(pred) >= 6:  # x1, y1, x2, y2, conf, class
            x1, y1, x2, y2, conf, cls = pred[:6]
            
            if conf >= confidence_threshold:
                # Convert normalized coordinates to pixel coordinates if needed
                if x1 <= 1.0 and y1 <= 1.0:
                    x1, x2 = x1 * w, x2 * w
                    y1, y2 = y1 * h, y2 * h
                
                # Draw bounding box
                rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                   fill=False, edgecolor='red', linewidth=2)
                plt.gca().add_patch(rect)
                
                # Add label
                label = f"{class_names[int(cls)]}: {conf:.2f}"
                plt.text(x1, y1-10, label, color='red', fontsize=12, 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.title(f"Detections (Confidence >= {confidence_threshold})")
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Visualization saved to {save_path}")
    
    plt.show()


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """Calculate Intersection over Union (IoU) between two bounding boxes."""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection area
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calculate union area
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def non_max_suppression(predictions: List, iou_threshold: float = 0.45, 
                       conf_threshold: float = 0.25) -> List:
    """Apply Non-Maximum Suppression to predictions."""
    if not predictions:
        return []
    
    # Filter by confidence threshold
    predictions = [pred for pred in predictions if pred[4] >= conf_threshold]
    
    if not predictions:
        return []
    
    # Sort by confidence score (descending)
    predictions = sorted(predictions, key=lambda x: x[4], reverse=True)
    
    # Apply NMS
    keep = []
    while predictions:
        current = predictions.pop(0)
        keep.append(current)
        
        # Remove boxes with high IoU
        predictions = [
            pred for pred in predictions 
            if calculate_iou(current[:4], pred[:4]) <= iou_threshold
        ]
    
    return keep


def resize_image_with_padding(image: np.ndarray, target_size: Tuple[int, int]) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """Resize image while maintaining aspect ratio with padding."""
    h, w = image.shape[:2]
    target_h, target_w = target_size
    
    # Calculate scaling factor
    scale = min(target_w / w, target_h / h)
    
    # Calculate new dimensions
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize image
    resized = cv2.resize(image, (new_w, new_h))
    
    # Create padded image
    padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    
    # Calculate padding offsets
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2
    
    # Place resized image in center
    padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return padded, scale, (x_offset, y_offset)


def denormalize_coordinates(boxes: np.ndarray, img_width: int, img_height: int) -> np.ndarray:
    """Convert normalized YOLO coordinates to pixel coordinates."""
    boxes = boxes.copy()
    boxes[:, [0, 2]] *= img_width   # x coordinates
    boxes[:, [1, 3]] *= img_height  # y coordinates
    return boxes


def normalize_coordinates(boxes: np.ndarray, img_width: int, img_height: int) -> np.ndarray:
    """Convert pixel coordinates to normalized YOLO coordinates."""
    boxes = boxes.copy()
    boxes[:, [0, 2]] /= img_width   # x coordinates
    boxes[:, [1, 3]] /= img_height  # y coordinates
    return boxes


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """Convert YOLO format (x_center, y_center, width, height) to (x1, y1, x2, y2)."""
    boxes_xyxy = boxes.copy()
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2
    return boxes_xyxy


def xyxy_to_xywh(boxes: np.ndarray) -> np.ndarray:
    """Convert (x1, y1, x2, y2) format to YOLO format (x_center, y_center, width, height)."""
    boxes_xywh = boxes.copy()
    boxes_xywh[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2  # x_center
    boxes_xywh[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2  # y_center
    boxes_xywh[:, 2] = boxes[:, 2] - boxes[:, 0]        # width
    boxes_xywh[:, 3] = boxes[:, 3] - boxes[:, 1]        # height
    return boxes_xywh


def plot_training_curves(metrics_file: str, save_path: Optional[str] = None):
    """Plot training and validation curves from metrics file."""
    # This would be implemented based on the specific metrics format
    # For now, providing a placeholder structure
    pass


def create_class_distribution_plot(label_dir: str, class_names: List[str], save_path: Optional[str] = None):
    """Create a plot showing the distribution of classes in the dataset."""
    class_counts = {class_name: 0 for class_name in class_names}
    
    # Count instances of each class
    for label_file in Path(label_dir).glob("*.txt"):
        with open(label_file, 'r') as f:
            for line in f:
                if line.strip():
                    class_id = int(line.split()[0])
                    if class_id < len(class_names):
                        class_counts[class_names[class_id]] += 1
    
    # Create plot
    plt.figure(figsize=(10, 6))
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    
    sns.barplot(x=classes, y=counts)
    plt.title("Class Distribution in Dataset")
    plt.xlabel("Class")
    plt.ylabel("Number of Instances")
    plt.xticks(rotation=45)
    
    # Add count labels on bars
    for i, count in enumerate(counts):
        plt.text(i, count + max(counts) * 0.01, str(count), ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Class distribution plot saved to {save_path}")
    
    plt.show()
    
    return class_counts


def print_dataset_summary(dataset_path: str, class_names: List[str]):
    """Print a summary of the dataset."""
    dataset_path = Path(dataset_path).parent
    
    print("=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    
    # Count images and labels
    splits = ['train', 'val', 'test']
    total_images = 0
    total_labels = 0
    
    for split in splits:
        img_dir = dataset_path / 'data' / split / 'images'
        label_dir = dataset_path / 'data' / split / 'labels'
        
        if img_dir.exists():
            img_count = len(list(img_dir.glob("*.png"))) + len(list(img_dir.glob("*.jpg")))
            label_count = len(list(label_dir.glob("*.txt"))) if label_dir.exists() else 0
            
            print(f"{split.upper()} SET:")
            print(f"  Images: {img_count}")
            print(f"  Labels: {label_count}")
            print()
            
            total_images += img_count
            total_labels += label_count
    
    print(f"TOTAL:")
    print(f"  Images: {total_images}")
    print(f"  Labels: {total_labels}")
    print(f"  Classes: {len(class_names)}")
    print(f"  Class names: {', '.join(class_names)}")
    print("=" * 60)


if __name__ == "__main__":
    # Test some utility functions
    config = load_config("config.yaml")
    print("Configuration loaded successfully!")
    
    device = get_device()
    print(f"Using device: {device}")
    
    class_names = ["FireExtinguisher", "ToolBox", "OxygenTank"]
    print_dataset_summary("../HackByte_Dataset/yolo_params.yaml", class_names)
