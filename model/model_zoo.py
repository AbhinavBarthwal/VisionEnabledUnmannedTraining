"""
Model management utilities for different YOLO variants and pre-trained models.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import requests
import hashlib

try:
    from ultralytics import YOLO
except ImportError:
    print("ultralytics not installed. Please run: pip install ultralytics")
    import sys
    sys.exit(1)

from utils import setup_logging


class ModelZoo:
    """Manage different YOLO model variants and pre-trained weights."""
    
    YOLO_MODELS = {
        'yolov8n': {
            'name': 'YOLOv8 Nano',
            'size': 'nano',
            'parameters': '3.2M',
            'description': 'Fastest, smallest model for real-time applications',
            'url': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt'
        },
        'yolov8s': {
            'name': 'YOLOv8 Small',
            'size': 'small',
            'parameters': '11.2M',
            'description': 'Good balance between speed and accuracy',
            'url': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt'
        },
        'yolov8m': {
            'name': 'YOLOv8 Medium',
            'size': 'medium',
            'parameters': '25.9M',
            'description': 'Higher accuracy with moderate inference time',
            'url': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt'
        },
        'yolov8l': {
            'name': 'YOLOv8 Large',
            'size': 'large',
            'parameters': '43.7M',
            'description': 'High accuracy model for demanding applications',
            'url': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt'
        },
        'yolov8x': {
            'name': 'YOLOv8 Extra Large',
            'size': 'extra_large',
            'parameters': '68.2M',
            'description': 'Highest accuracy, slowest inference',
            'url': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt'
        }
    }
    
    def __init__(self, cache_dir: str = "models"):
        """
        Initialize the model zoo.
        
        Args:
            cache_dir: Directory to cache downloaded models
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logging.info(f"Model zoo initialized with cache directory: {cache_dir}")
    
    def list_available_models(self) -> Dict:
        """List all available pre-trained models."""
        return self.YOLO_MODELS.copy()
    
    def get_model_info(self, model_name: str) -> Optional[Dict]:
        """Get information about a specific model."""
        return self.YOLO_MODELS.get(model_name.lower())
    
    def download_model(self, model_name: str, force_download: bool = False) -> str:
        """
        Download a pre-trained model.
        
        Args:
            model_name: Name of the model to download
            force_download: Force download even if model exists
            
        Returns:
            Path to downloaded model
        """
        model_name = model_name.lower()
        
        if model_name not in self.YOLO_MODELS:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(self.YOLO_MODELS.keys())}")
        
        model_info = self.YOLO_MODELS[model_name]
        model_path = self.cache_dir / f"{model_name}.pt"
        
        # Check if model already exists
        if model_path.exists() and not force_download:
            logging.info(f"Model {model_name} already exists at: {model_path}")
            return str(model_path)
        
        # Download model
        logging.info(f"Downloading {model_info['name']} ({model_info['parameters']})...")
        
        try:
            response = requests.get(model_info['url'], stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded_size = 0
            
            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        
                        # Progress indication
                        if total_size > 0:
                            progress = (downloaded_size / total_size) * 100
                            print(f"\rProgress: {progress:.1f}%", end='', flush=True)
            
            print()  # New line after progress
            logging.info(f"Model downloaded successfully: {model_path}")
            return str(model_path)
            
        except Exception as e:
            if model_path.exists():
                model_path.unlink()  # Remove incomplete download
            raise Exception(f"Failed to download model {model_name}: {e}")
    
    def load_model(self, model_name_or_path: str, download_if_missing: bool = True) -> YOLO:
        """
        Load a YOLO model.
        
        Args:
            model_name_or_path: Model name or path to model file
            download_if_missing: Download model if it's a known model name and not found locally
            
        Returns:
            Loaded YOLO model
        """
        # Check if it's a file path
        if Path(model_name_or_path).exists():
            logging.info(f"Loading model from file: {model_name_or_path}")
            return YOLO(model_name_or_path)
        
        # Check if it's a known model name
        model_name = model_name_or_path.lower()
        if model_name in self.YOLO_MODELS:
            if download_if_missing:
                model_path = self.download_model(model_name)
                return YOLO(model_path)
            else:
                # Try to load from cache
                model_path = self.cache_dir / f"{model_name}.pt"
                if model_path.exists():
                    return YOLO(str(model_path))
                else:
                    raise FileNotFoundError(f"Model {model_name} not found in cache. Set download_if_missing=True to download.")
        
        # Try to load directly (might be a model identifier that ultralytics recognizes)
        try:
            return YOLO(model_name_or_path)
        except Exception as e:
            raise ValueError(f"Could not load model {model_name_or_path}: {e}")
    
    def compare_models(self, dataset_path: str, models: List[str] = None, 
                      epochs: int = 10, imgsz: int = 640) -> Dict:
        """
        Compare different model architectures on the same dataset.
        
        Args:
            dataset_path: Path to dataset YAML
            models: List of model names to compare (if None, compares nano, small, medium)
            epochs: Number of training epochs for comparison
            imgsz: Image size for training
            
        Returns:
            Comparison results dictionary
        """
        if models is None:
            models = ['yolov8n', 'yolov8s', 'yolov8m']
        
        results = {}
        
        for model_name in models:
            logging.info(f"Training {model_name} for comparison...")
            
            try:
                # Load model
                model = self.load_model(model_name)
                
                # Train model
                train_results = model.train(
                    data=dataset_path,
                    epochs=epochs,
                    imgsz=imgsz,
                    project=f"comparison_runs",
                    name=f"{model_name}_comparison",
                    verbose=False,
                    plots=False,
                    save=False,
                    val=True
                )
                
                # Extract key metrics
                results[model_name] = {
                    'model_info': self.get_model_info(model_name),
                    'final_mAP50': float(train_results.box.map50),
                    'final_mAP50_95': float(train_results.box.map),
                    'training_time': train_results.trainer.train_time_start,  # This might need adjustment
                    'model_size_mb': self._get_model_size(model_name)
                }
                
                logging.info(f"{model_name} comparison complete - mAP@0.5: {results[model_name]['final_mAP50']:.4f}")
                
            except Exception as e:
                logging.error(f"Failed to train {model_name}: {e}")
                results[model_name] = {'error': str(e)}
        
        return results
    
    def _get_model_size(self, model_name: str) -> float:
        """Get model file size in MB."""
        model_path = self.cache_dir / f"{model_name}.pt"
        if model_path.exists():
            return model_path.stat().st_size / (1024 * 1024)
        return 0.0
    
    def benchmark_inference_speed(self, models: List[str], test_images: List[str],
                                device: str = "cpu", warmup_runs: int = 10) -> Dict:
        """
        Benchmark inference speed of different models.
        
        Args:
            models: List of model names to benchmark
            test_images: List of test image paths
            device: Device to run inference on
            warmup_runs: Number of warmup runs before timing
            
        Returns:
            Benchmark results
        """
        import time
        import numpy as np
        
        results = {}
        
        for model_name in models:
            logging.info(f"Benchmarking {model_name}...")
            
            try:
                # Load model
                model = self.load_model(model_name)
                
                # Warmup runs
                for _ in range(warmup_runs):
                    _ = model(test_images[0], device=device, verbose=False)
                
                # Benchmark runs
                times = []
                for img_path in test_images:
                    start_time = time.time()
                    _ = model(img_path, device=device, verbose=False)
                    end_time = time.time()
                    times.append(end_time - start_time)
                
                # Calculate statistics
                avg_time = np.mean(times)
                std_time = np.std(times)
                fps = 1.0 / avg_time if avg_time > 0 else 0
                
                results[model_name] = {
                    'avg_inference_time': avg_time,
                    'std_inference_time': std_time,
                    'fps': fps,
                    'min_time': np.min(times),
                    'max_time': np.max(times),
                    'model_info': self.get_model_info(model_name),
                    'model_size_mb': self._get_model_size(model_name)
                }
                
                logging.info(f"{model_name}: {avg_time:.4f}s avg, {fps:.1f} FPS")
                
            except Exception as e:
                logging.error(f"Failed to benchmark {model_name}: {e}")
                results[model_name] = {'error': str(e)}
        
        return results
    
    def recommend_model(self, priority: str = "balanced", constraints: Dict = None) -> Tuple[str, Dict]:
        """
        Recommend a model based on priorities and constraints.
        
        Args:
            priority: Priority type - "speed", "accuracy", "balanced", "size"
            constraints: Dictionary of constraints (e.g., {"max_params": "25M", "min_fps": 30})
            
        Returns:
            Tuple of (recommended_model_name, model_info)
        """
        if constraints is None:
            constraints = {}
        
        # Define model rankings for different priorities
        rankings = {
            "speed": ['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x'],
            "accuracy": ['yolov8x', 'yolov8l', 'yolov8m', 'yolov8s', 'yolov8n'],
            "balanced": ['yolov8s', 'yolov8m', 'yolov8n', 'yolov8l', 'yolov8x'],
            "size": ['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x']
        }
        
        if priority not in rankings:
            logging.warning(f"Unknown priority '{priority}', using 'balanced'")
            priority = "balanced"
        
        # Get ranked models
        ranked_models = rankings[priority]
        
        # Apply constraints (simplified logic)
        for model_name in ranked_models:
            model_info = self.get_model_info(model_name)
            
            # Check parameter constraint
            if "max_params" in constraints:
                max_params = constraints["max_params"]
                # Simple string comparison for now - could be enhanced
                if max_params == "10M" and model_name in ['yolov8m', 'yolov8l', 'yolov8x']:
                    continue
                elif max_params == "30M" and model_name in ['yolov8l', 'yolov8x']:
                    continue
            
            # If all constraints satisfied, return this model
            return model_name, model_info
        
        # Fallback to nano if no model satisfies constraints
        return 'yolov8n', self.get_model_info('yolov8n')
    
    def print_model_comparison_table(self):
        """Print a comparison table of all available models."""
        print("\n" + "="*80)
        print("YOLO MODEL COMPARISON")
        print("="*80)
        print(f"{'Model':<12} {'Name':<20} {'Parameters':<12} {'Description':<30}")
        print("-"*80)
        
        for model_id, info in self.YOLO_MODELS.items():
            print(f"{model_id:<12} {info['name']:<20} {info['parameters']:<12} {info['description']:<30}")
        
        print("="*80)
        print("Recommendations:")
        print("  • Real-time applications: yolov8n")
        print("  • Balanced performance: yolov8s or yolov8m")
        print("  • High accuracy needs: yolov8l or yolov8x")
        print("  • Resource constrained: yolov8n")
        print("="*80)


def main():
    """Demo of model zoo functionality."""
    setup_logging()
    
    # Initialize model zoo
    model_zoo = ModelZoo()
    
    # Print available models
    model_zoo.print_model_comparison_table()
    
    # Get recommendation
    recommended_model, model_info = model_zoo.recommend_model("balanced")
    print(f"\nRecommended model for balanced performance: {recommended_model}")
    print(f"Details: {model_info['name']} ({model_info['parameters']})")
    
    # Download and load a model (this will actually download)
    try:
        print(f"\nDownloading {recommended_model}...")
        model_path = model_zoo.download_model(recommended_model)
        print(f"Model cached at: {model_path}")
        
        print("Loading model...")
        model = model_zoo.load_model(recommended_model)
        print("Model loaded successfully!")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
