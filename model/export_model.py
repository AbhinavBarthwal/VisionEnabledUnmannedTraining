"""
Model export utilities for different deployment formats.
"""

import argparse
import logging
from pathlib import Path
from typing import List, Optional

try:
    from ultralytics import YOLO
except ImportError:
    print("ultralytics not installed. Please run: pip install ultralytics")
    import sys
    sys.exit(1)

from utils import setup_logging, load_config


class ModelExporter:
    """Export trained YOLOv8 models to various formats."""
    
    def __init__(self, model_path: str):
        """
        Initialize the model exporter.
        
        Args:
            model_path: Path to trained YOLO model
        """
        self.model_path = model_path
        self.model = YOLO(model_path)
        
        logging.info(f"Model loaded: {model_path}")
    
    def export_onnx(self, output_dir: str = "exports", optimize: bool = True, 
                   dynamic: bool = False, simplify: bool = True) -> str:
        """
        Export model to ONNX format.
        
        Args:
            output_dir: Output directory
            optimize: Optimize for inference
            dynamic: Dynamic input shapes
            simplify: Simplify the model
            
        Returns:
            Path to exported ONNX model
        """
        logging.info("Exporting to ONNX format...")
        
        exported_model = self.model.export(
            format='onnx',
            optimize=optimize,
            dynamic=dynamic,
            simplify=simplify
        )
        
        # Move to output directory
        output_path = Path(output_dir) / f"{Path(self.model_path).stem}.onnx"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if Path(exported_model).exists():
            Path(exported_model).rename(output_path)
        
        logging.info(f"ONNX model exported to: {output_path}")
        return str(output_path)
    
    def export_tensorrt(self, output_dir: str = "exports", half: bool = True,
                       workspace: int = 4, verbose: bool = False) -> str:
        """
        Export model to TensorRT format.
        
        Args:
            output_dir: Output directory
            half: Use FP16 precision
            workspace: Workspace memory in GB
            verbose: Verbose output
            
        Returns:
            Path to exported TensorRT model
        """
        logging.info("Exporting to TensorRT format...")
        
        try:
            exported_model = self.model.export(
                format='engine',
                half=half,
                workspace=workspace,
                verbose=verbose
            )
            
            # Move to output directory
            output_path = Path(output_dir) / f"{Path(self.model_path).stem}.engine"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if Path(exported_model).exists():
                Path(exported_model).rename(output_path)
            
            logging.info(f"TensorRT model exported to: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logging.error(f"TensorRT export failed: {e}")
            logging.info("Make sure TensorRT is properly installed")
            raise
    
    def export_torchscript(self, output_dir: str = "exports", optimize: bool = True) -> str:
        """
        Export model to TorchScript format.
        
        Args:
            output_dir: Output directory
            optimize: Optimize for inference
            
        Returns:
            Path to exported TorchScript model
        """
        logging.info("Exporting to TorchScript format...")
        
        exported_model = self.model.export(
            format='torchscript',
            optimize=optimize
        )
        
        # Move to output directory
        output_path = Path(output_dir) / f"{Path(self.model_path).stem}.torchscript"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if Path(exported_model).exists():
            Path(exported_model).rename(output_path)
        
        logging.info(f"TorchScript model exported to: {output_path}")
        return str(output_path)
    
    def export_openvino(self, output_dir: str = "exports", half: bool = False) -> str:
        """
        Export model to OpenVINO format.
        
        Args:
            output_dir: Output directory
            half: Use FP16 precision
            
        Returns:
            Path to exported OpenVINO model
        """
        logging.info("Exporting to OpenVINO format...")
        
        try:
            exported_model = self.model.export(
                format='openvino',
                half=half
            )
            
            # Move to output directory
            output_path = Path(output_dir) / f"{Path(self.model_path).stem}_openvino_model"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if Path(exported_model).exists():
                if Path(exported_model).is_dir():
                    import shutil
                    if output_path.exists():
                        shutil.rmtree(output_path)
                    shutil.move(exported_model, output_path)
                else:
                    Path(exported_model).rename(output_path)
            
            logging.info(f"OpenVINO model exported to: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logging.error(f"OpenVINO export failed: {e}")
            logging.info("Make sure OpenVINO is properly installed")
            raise
    
    def export_coreml(self, output_dir: str = "exports", int8: bool = False,
                     half: bool = False, nms: bool = False) -> str:
        """
        Export model to CoreML format (for iOS deployment).
        
        Args:
            output_dir: Output directory
            int8: Use INT8 quantization
            half: Use FP16 precision
            nms: Include NMS in the model
            
        Returns:
            Path to exported CoreML model
        """
        logging.info("Exporting to CoreML format...")
        
        try:
            exported_model = self.model.export(
                format='coreml',
                int8=int8,
                half=half,
                nms=nms
            )
            
            # Move to output directory
            output_path = Path(output_dir) / f"{Path(self.model_path).stem}.mlmodel"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if Path(exported_model).exists():
                Path(exported_model).rename(output_path)
            
            logging.info(f"CoreML model exported to: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logging.error(f"CoreML export failed: {e}")
            logging.info("CoreML export requires macOS or coremltools installation")
            raise
    
    def export_tflite(self, output_dir: str = "exports", int8: bool = False,
                     half: bool = False) -> str:
        """
        Export model to TensorFlow Lite format.
        
        Args:
            output_dir: Output directory
            int8: Use INT8 quantization
            half: Use FP16 precision
            
        Returns:
            Path to exported TFLite model
        """
        logging.info("Exporting to TensorFlow Lite format...")
        
        try:
            exported_model = self.model.export(
                format='tflite',
                int8=int8,
                half=half
            )
            
            # Move to output directory
            output_path = Path(output_dir) / f"{Path(self.model_path).stem}.tflite"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if Path(exported_model).exists():
                Path(exported_model).rename(output_path)
            
            logging.info(f"TensorFlow Lite model exported to: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logging.error(f"TensorFlow Lite export failed: {e}")
            logging.info("TensorFlow Lite export requires tensorflow installation")
            raise
    
    def export_all_formats(self, output_dir: str = "exports", 
                          formats: Optional[List[str]] = None) -> dict:
        """
        Export model to multiple formats.
        
        Args:
            output_dir: Output directory
            formats: List of formats to export. If None, exports common formats.
            
        Returns:
            Dictionary mapping format names to export paths
        """
        if formats is None:
            formats = ['onnx', 'torchscript']  # Safe formats that work everywhere
        
        results = {}
        
        for fmt in formats:
            try:
                if fmt.lower() == 'onnx':
                    results['onnx'] = self.export_onnx(output_dir)
                elif fmt.lower() == 'tensorrt':
                    results['tensorrt'] = self.export_tensorrt(output_dir)
                elif fmt.lower() == 'torchscript':
                    results['torchscript'] = self.export_torchscript(output_dir)
                elif fmt.lower() == 'openvino':
                    results['openvino'] = self.export_openvino(output_dir)
                elif fmt.lower() == 'coreml':
                    results['coreml'] = self.export_coreml(output_dir)
                elif fmt.lower() == 'tflite':
                    results['tflite'] = self.export_tflite(output_dir)
                else:
                    logging.warning(f"Unknown format: {fmt}")
                    
            except Exception as e:
                logging.error(f"Failed to export {fmt}: {e}")
                results[fmt] = None
        
        return results
    
    def benchmark_formats(self, exported_models: dict, test_images: List[str],
                         output_file: str = "benchmark_results.txt"):
        """
        Benchmark different exported model formats.
        
        Args:
            exported_models: Dictionary of format -> model_path
            test_images: List of test image paths
            output_file: Output file for benchmark results
        """
        import time
        import numpy as np
        
        logging.info("Benchmarking exported models...")
        
        results = []
        
        for format_name, model_path in exported_models.items():
            if model_path is None:
                continue
                
            try:
                # Load model based on format
                if format_name == 'onnx':
                    model = YOLO(model_path)
                elif format_name == 'torchscript':
                    model = YOLO(model_path)
                else:
                    logging.info(f"Benchmarking for {format_name} not implemented yet")
                    continue
                
                # Run inference on test images
                times = []
                for img_path in test_images[:10]:  # Use first 10 images
                    start_time = time.time()
                    _ = model(img_path, verbose=False)
                    end_time = time.time()
                    times.append(end_time - start_time)
                
                avg_time = np.mean(times)
                fps = 1.0 / avg_time if avg_time > 0 else 0
                
                results.append({
                    'format': format_name,
                    'avg_time': avg_time,
                    'fps': fps,
                    'model_path': model_path
                })
                
                logging.info(f"{format_name}: {avg_time:.4f}s per image, {fps:.1f} FPS")
                
            except Exception as e:
                logging.error(f"Benchmarking {format_name} failed: {e}")
        
        # Save results
        with open(output_file, 'w') as f:
            f.write("MODEL FORMAT BENCHMARK RESULTS\n")
            f.write("=" * 50 + "\n\n")
            
            for result in results:
                f.write(f"Format: {result['format']}\n")
                f.write(f"  Average time per image: {result['avg_time']:.4f}s\n")
                f.write(f"  FPS: {result['fps']:.1f}\n")
                f.write(f"  Model path: {result['model_path']}\n\n")
        
        logging.info(f"Benchmark results saved to: {output_file}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Export YOLOv8 model to various formats")
    
    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    parser.add_argument("--output", type=str, default="exports", help="Output directory")
    parser.add_argument("--formats", nargs="+", default=['onnx', 'torchscript'],
                       choices=['onnx', 'tensorrt', 'torchscript', 'openvino', 'coreml', 'tflite'],
                       help="Export formats")
    parser.add_argument("--optimize", action="store_true", help="Optimize for inference")
    parser.add_argument("--half", action="store_true", help="Use FP16 precision")
    parser.add_argument("--int8", action="store_true", help="Use INT8 quantization")
    parser.add_argument("--benchmark", action="store_true", help="Benchmark exported models")
    parser.add_argument("--test-images", type=str, help="Directory with test images for benchmarking")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    return parser.parse_args()


def main():
    """Main export function."""
    args = parse_args()
    
    # Setup logging
    setup_logging("INFO" if not args.verbose else "DEBUG")
    
    # Check if model exists
    if not Path(args.model).exists():
        logging.error(f"Model not found: {args.model}")
        return
    
    try:
        # Initialize exporter
        exporter = ModelExporter(args.model)
        
        # Export to specified formats
        logging.info(f"Exporting model to formats: {args.formats}")
        
        exported_models = exporter.export_all_formats(
            output_dir=args.output,
            formats=args.formats
        )
        
        # Print results
        print("\n" + "="*50)
        print("EXPORT RESULTS")
        print("="*50)
        
        for format_name, model_path in exported_models.items():
            if model_path:
                print(f"✓ {format_name.upper()}: {model_path}")
            else:
                print(f"✗ {format_name.upper()}: Failed")
        
        print("="*50)
        
        # Benchmark if requested
        if args.benchmark and args.test_images:
            test_img_dir = Path(args.test_images)
            if test_img_dir.exists():
                test_images = []
                for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                    test_images.extend(list(test_img_dir.glob(f"*{ext}")))
                    test_images.extend(list(test_img_dir.glob(f"*{ext.upper()}")))
                
                if test_images:
                    exporter.benchmark_formats(
                        exported_models,
                        [str(p) for p in test_images[:10]],
                        Path(args.output) / "benchmark_results.txt"
                    )
                else:
                    logging.warning("No test images found for benchmarking")
            else:
                logging.warning(f"Test images directory not found: {args.test_images}")
        
        logging.info("Export completed successfully!")
        
    except Exception as e:
        logging.error(f"Export failed: {e}")
        raise


if __name__ == "__main__":
    main()
