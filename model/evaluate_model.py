"""
Model evaluation script for industrial safety equipment detection.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

try:
    from ultralytics import YOLO
except ImportError:
    print("ultralytics not installed. Please run: pip install ultralytics")
    sys.exit(1)

from utils import setup_logging, load_config, calculate_iou


class ModelEvaluator:
    """Evaluate trained YOLOv8 model performance."""
    
    def __init__(self, model_path: str, data_path: str, device: str = "auto"):
        """
        Initialize the evaluator.
        
        Args:
            model_path: Path to trained model
            data_path: Path to dataset YAML
            device: Device to use for evaluation
        """
        self.model_path = model_path
        self.data_path = data_path
        self.device = device
        self.class_names = ["FireExtinguisher", "ToolBox", "OxygenTank"]
        
        # Load model
        self.model = YOLO(model_path)
        
        logging.info(f"Evaluator initialized with model: {model_path}")
        logging.info(f"Dataset: {data_path}")
        logging.info(f"Device: {device}")
    
    def evaluate_model(self, conf_threshold: float = 0.25, iou_threshold: float = 0.45,
                      save_dir: str = "evaluation_results") -> Dict:
        """
        Evaluate model performance on validation/test set.
        
        Args:
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold for NMS
            save_dir: Directory to save evaluation results
            
        Returns:
            Dictionary containing evaluation metrics
        """
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        logging.info("Running model evaluation...")
        
        # Run validation
        results = self.model.val(
            data=self.data_path,
            conf=conf_threshold,
            iou=iou_threshold,
            device=self.device,
            save_json=True,
            save_hybrid=False,
            plots=True,
            verbose=True
        )
        
        # Extract metrics
        metrics = {
            'mAP_50': float(results.box.map50),
            'mAP_50_95': float(results.box.map),
            'precision': float(results.box.mp),
            'recall': float(results.box.mr),
            'f1_score': 2 * (float(results.box.mp) * float(results.box.mr)) / (float(results.box.mp) + float(results.box.mr)) if (float(results.box.mp) + float(results.box.mr)) > 0 else 0,
        }
        
        # Per-class metrics
        if hasattr(results.box, 'map50') and results.box.map50 is not None:
            per_class_map50 = results.box.map50
            per_class_map = results.box.map
            
            for i, class_name in enumerate(self.class_names):
                if i < len(per_class_map50):
                    metrics[f'{class_name}_mAP_50'] = float(per_class_map50[i])
                    metrics[f'{class_name}_mAP_50_95'] = float(per_class_map[i])
        
        # Save metrics to file
        metrics_file = Path(save_dir) / "evaluation_metrics.txt"
        with open(metrics_file, 'w') as f:
            f.write("Model Evaluation Results\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Model: {self.model_path}\n")
            f.write(f"Dataset: {self.data_path}\n")
            f.write(f"Confidence Threshold: {conf_threshold}\n")
            f.write(f"IoU Threshold: {iou_threshold}\n\n")
            
            f.write("Overall Metrics:\n")
            f.write(f"  mAP@0.5: {metrics['mAP_50']:.4f}\n")
            f.write(f"  mAP@0.5:0.95: {metrics['mAP_50_95']:.4f}\n")
            f.write(f"  Precision: {metrics['precision']:.4f}\n")
            f.write(f"  Recall: {metrics['recall']:.4f}\n")
            f.write(f"  F1-Score: {metrics['f1_score']:.4f}\n\n")
            
            f.write("Per-Class mAP@0.5:\n")
            for class_name in self.class_names:
                if f'{class_name}_mAP_50' in metrics:
                    f.write(f"  {class_name}: {metrics[f'{class_name}_mAP_50']:.4f}\n")
            
            f.write("\nPer-Class mAP@0.5:0.95:\n")
            for class_name in self.class_names:
                if f'{class_name}_mAP_50_95' in metrics:
                    f.write(f"  {class_name}: {metrics[f'{class_name}_mAP_50_95']:.4f}\n")
        
        logging.info(f"Evaluation metrics saved to: {metrics_file}")
        
        return metrics
    
    def create_performance_plots(self, metrics: Dict, save_dir: str = "evaluation_results"):
        """Create performance visualization plots."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Overall metrics bar plot
        overall_metrics = {
            'mAP@0.5': metrics['mAP_50'],
            'mAP@0.5:0.95': metrics['mAP_50_95'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1-Score': metrics['f1_score']
        }
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(overall_metrics.keys(), overall_metrics.values())
        plt.title('Overall Model Performance Metrics')
        plt.ylabel('Score')
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, overall_metrics.values()):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(save_dir / 'overall_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Per-class mAP comparison
        class_map50 = []
        class_map_50_95 = []
        
        for class_name in self.class_names:
            map50_key = f'{class_name}_mAP_50'
            map_key = f'{class_name}_mAP_50_95'
            
            class_map50.append(metrics.get(map50_key, 0))
            class_map_50_95.append(metrics.get(map_key, 0))
        
        # Create grouped bar chart
        x = np.arange(len(self.class_names))
        width = 0.35
        
        plt.figure(figsize=(12, 7))
        bars1 = plt.bar(x - width/2, class_map50, width, label='mAP@0.5', alpha=0.8)
        bars2 = plt.bar(x + width/2, class_map_50_95, width, label='mAP@0.5:0.95', alpha=0.8)
        
        plt.xlabel('Classes')
        plt.ylabel('mAP Score')
        plt.title('Per-Class Mean Average Precision')
        plt.xticks(x, self.class_names)
        plt.legend()
        plt.ylim(0, 1)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(save_dir / 'per_class_map.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Performance plots saved to: {save_dir}")
    
    def analyze_inference_speed(self, test_images: List[str], batch_sizes: List[int] = [1, 4, 8, 16]) -> Dict:
        """Analyze model inference speed."""
        import time
        
        speed_results = {}
        
        for batch_size in batch_sizes:
            logging.info(f"Testing inference speed with batch size: {batch_size}")
            
            times = []
            num_batches = max(1, len(test_images) // batch_size)
            
            for i in range(0, min(len(test_images), num_batches * batch_size), batch_size):
                batch_images = test_images[i:i+batch_size]
                
                start_time = time.time()
                results = self.model(batch_images, verbose=False)
                end_time = time.time()
                
                batch_time = end_time - start_time
                times.append(batch_time / len(batch_images))  # Time per image
            
            avg_time = np.mean(times)
            fps = 1.0 / avg_time if avg_time > 0 else 0
            
            speed_results[batch_size] = {
                'avg_time_per_image': avg_time,
                'fps': fps,
                'total_batches': len(times)
            }
            
            logging.info(f"Batch size {batch_size}: {avg_time:.4f}s per image, {fps:.1f} FPS")
        
        return speed_results
    
    def compare_models(self, model_paths: List[str], model_names: List[str] = None) -> pd.DataFrame:
        """Compare multiple models."""
        if model_names is None:
            model_names = [f"Model_{i+1}" for i in range(len(model_paths))]
        
        comparison_results = []
        
        for model_path, model_name in zip(model_paths, model_names):
            try:
                # Load model
                model = YOLO(model_path)
                
                # Run evaluation
                results = model.val(
                    data=self.data_path,
                    verbose=False
                )
                
                # Extract metrics
                result_dict = {
                    'Model': model_name,
                    'mAP@0.5': float(results.box.map50),
                    'mAP@0.5:0.95': float(results.box.map),
                    'Precision': float(results.box.mp),
                    'Recall': float(results.box.mr),
                }
                
                # Add F1 score
                p, r = result_dict['Precision'], result_dict['Recall']
                result_dict['F1-Score'] = 2 * (p * r) / (p + r) if (p + r) > 0 else 0
                
                comparison_results.append(result_dict)
                logging.info(f"Evaluated {model_name}")
                
            except Exception as e:
                logging.error(f"Error evaluating {model_name}: {e}")
        
        # Create DataFrame
        df = pd.DataFrame(comparison_results)
        
        return df
    
    def create_comparison_plot(self, comparison_df: pd.DataFrame, save_path: str = "model_comparison.png"):
        """Create model comparison visualization."""
        metrics = ['mAP@0.5', 'mAP@0.5:0.95', 'Precision', 'Recall', 'F1-Score']
        
        fig, axes = plt.subplots(1, len(metrics), figsize=(20, 4))
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            bars = ax.bar(comparison_df['Model'], comparison_df[metric])
            ax.set_title(metric)
            ax.set_ylim(0, 1)
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, value in zip(bars, comparison_df[metric]):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Model comparison plot saved to: {save_path}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate YOLOv8 model performance")
    
    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    parser.add_argument("--data", type=str, default="../HackByte_Dataset/yolo_params.yaml", help="Path to dataset YAML")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold")
    parser.add_argument("--device", type=str, default="auto", help="Device to use")
    parser.add_argument("--output", type=str, default="evaluation_results", help="Output directory")
    parser.add_argument("--speed-test", action="store_true", help="Run inference speed test")
    parser.add_argument("--compare", nargs="+", help="Paths to models for comparison")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    return parser.parse_args()


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Setup logging
    setup_logging("INFO" if not args.verbose else "DEBUG")
    
    # Initialize evaluator
    try:
        evaluator = ModelEvaluator(
            model_path=args.model,
            data_path=args.data,
            device=args.device
        )
    except Exception as e:
        logging.error(f"Failed to initialize evaluator: {e}")
        return
    
    # Run evaluation
    try:
        logging.info("Starting model evaluation...")
        
        metrics = evaluator.evaluate_model(
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            save_dir=args.output
        )
        
        # Create performance plots
        evaluator.create_performance_plots(metrics, args.output)
        
        # Print results
        print("\n" + "="*60)
        print("MODEL EVALUATION RESULTS")
        print("="*60)
        print(f"Model: {args.model}")
        print(f"mAP@0.5: {metrics['mAP_50']:.4f}")
        print(f"mAP@0.5:0.95: {metrics['mAP_50_95']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-Score: {metrics['f1_score']:.4f}")
        print("\nPer-Class mAP@0.5:")
        for class_name in evaluator.class_names:
            key = f'{class_name}_mAP_50'
            if key in metrics:
                print(f"  {class_name}: {metrics[key]:.4f}")
        print("="*60)
        
        # Speed test if requested
        if args.speed_test:
            logging.info("Running inference speed test...")
            # For speed test, we'd need test images - this is a placeholder
            print("Speed test would be implemented with actual test images")
        
        # Model comparison if requested
        if args.compare:
            logging.info("Running model comparison...")
            comparison_df = evaluator.compare_models(args.compare)
            print("\nModel Comparison:")
            print(comparison_df.to_string(index=False))
            
            # Save comparison
            comparison_df.to_csv(Path(args.output) / "model_comparison.csv", index=False)
            evaluator.create_comparison_plot(comparison_df, Path(args.output) / "model_comparison.png")
        
        logging.info("Evaluation completed successfully!")
        
    except Exception as e:
        logging.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()
