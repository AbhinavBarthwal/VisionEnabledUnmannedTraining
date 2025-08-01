"""
Inference script for industrial safety equipment detection using trained YOLOv8 model.
"""

import os
import argparse
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Union

import cv2
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

try:
    from ultralytics import YOLO
except ImportError:
    print("ultralytics not installed. Please run: pip install ultralytics")
    sys.exit(1)

from utils import (
    setup_logging,
    load_config,
    visualize_predictions,
    non_max_suppression,
    resize_image_with_padding
)


class IndustrialSafetyDetector:
    """Industrial Safety Equipment Detector using YOLOv8."""
    
    def __init__(self, model_path: str, conf_threshold: float = 0.25, 
                 iou_threshold: float = 0.45, device: str = "auto"):
        """
        Initialize the detector.
        
        Args:
            model_path: Path to the trained YOLO model
            conf_threshold: Confidence threshold for predictions
            iou_threshold: IoU threshold for NMS
            device: Device to run inference on
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.class_names = ["FireExtinguisher", "ToolBox", "OxygenTank"]
        
        # Load model
        self.model = self._load_model()
        
        logging.info(f"Detector initialized with model: {model_path}")
        logging.info(f"Device: {self.device}")
        logging.info(f"Confidence threshold: {conf_threshold}")
        logging.info(f"IoU threshold: {iou_threshold}")
    
    def _load_model(self) -> YOLO:
        """Load the YOLO model."""
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        model = YOLO(self.model_path)
        
        # Set device
        if self.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        
        return model
    
    def predict_image(self, image_path: str, save_results: bool = True, 
                     output_dir: str = "inference_results") -> List:
        """
        Run inference on a single image.
        
        Args:
            image_path: Path to the input image
            save_results: Whether to save visualization results
            output_dir: Directory to save results
            
        Returns:
            List of detections with format [x1, y1, x2, y2, confidence, class_id]
        """
        # Check if image exists
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Run inference
        results = self.model(
            image_path,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False
        )
        
        # Extract detections
        detections = []
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy()
                
                for box, conf, cls_id in zip(boxes, confidences, class_ids):
                    detection = [*box, conf, cls_id]
                    detections.append(detection)
        
        # Save visualization if requested
        if save_results:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            output_path = Path(output_dir) / f"result_{Path(image_path).stem}.jpg"
            
            visualize_predictions(
                image_path=image_path,
                predictions=detections,
                class_names=self.class_names,
                confidence_threshold=self.conf_threshold,
                save_path=str(output_path)
            )
        
        return detections
    
    def predict_batch(self, image_paths: List[str], save_results: bool = True,
                     output_dir: str = "inference_results") -> List[List]:
        """
        Run inference on multiple images.
        
        Args:
            image_paths: List of paths to input images
            save_results: Whether to save visualization results
            output_dir: Directory to save results
            
        Returns:
            List of detection lists for each image
        """
        all_detections = []
        
        for image_path in image_paths:
            try:
                detections = self.predict_image(
                    image_path=image_path,
                    save_results=save_results,
                    output_dir=output_dir
                )
                all_detections.append(detections)
                logging.info(f"Processed: {image_path} - Found {len(detections)} objects")
                
            except Exception as e:
                logging.error(f"Error processing {image_path}: {e}")
                all_detections.append([])
        
        return all_detections
    
    def predict_directory(self, input_dir: str, output_dir: str = "inference_results",
                         image_extensions: Tuple[str] = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')) -> List[List]:
        """
        Run inference on all images in a directory.
        
        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save results
            image_extensions: Tuple of supported image extensions
            
        Returns:
            List of detection lists for each image
        """
        input_path = Path(input_dir)
        if not input_path.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        # Find all images
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(list(input_path.glob(f"*{ext}")))
            image_paths.extend(list(input_path.glob(f"*{ext.upper()}")))
        
        if not image_paths:
            logging.warning(f"No images found in {input_dir}")
            return []
        
        logging.info(f"Found {len(image_paths)} images in {input_dir}")
        
        # Process all images
        return self.predict_batch(
            image_paths=[str(p) for p in image_paths],
            save_results=True,
            output_dir=output_dir
        )
    
    def predict_video(self, video_path: str, output_path: str = "output_video.mp4",
                     skip_frames: int = 1) -> None:
        """
        Run inference on a video file.
        
        Args:
            video_path: Path to input video
            output_path: Path to save output video
            skip_frames: Process every Nth frame to speed up processing
        """
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        processed_frames = 0
        
        logging.info(f"Processing video: {video_path}")
        logging.info(f"Total frames: {total_frames}, FPS: {fps}")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process every skip_frames frame
                if frame_count % skip_frames == 0:
                    # Save frame temporarily
                    temp_image = "temp_frame.jpg"
                    cv2.imwrite(temp_image, frame)
                    
                    # Run inference
                    detections = self.predict_image(
                        image_path=temp_image,
                        save_results=False
                    )
                    
                    # Draw detections on frame
                    frame = self._draw_detections(frame, detections)
                    processed_frames += 1
                    
                    # Clean up
                    if os.path.exists(temp_image):
                        os.remove(temp_image)
                
                # Write frame to output
                out.write(frame)
                frame_count += 1
                
                # Progress update
                if frame_count % 100 == 0:
                    progress = (frame_count / total_frames) * 100
                    logging.info(f"Progress: {progress:.1f}% ({frame_count}/{total_frames})")
        
        finally:
            cap.release()
            out.release()
            cv2.destroyAllWindows()
        
        logging.info(f"Video processing complete. Output saved to: {output_path}")
        logging.info(f"Processed {processed_frames} frames with detections")
    
    def _draw_detections(self, image: np.ndarray, detections: List) -> np.ndarray:
        """Draw detections on image."""
        for detection in detections:
            if len(detection) >= 6:
                x1, y1, x2, y2, conf, cls_id = detection[:6]
                
                if conf >= self.conf_threshold:
                    # Draw bounding box
                    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    
                    # Draw label
                    label = f"{self.class_names[int(cls_id)]}: {conf:.2f}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(image, (int(x1), int(y1) - label_size[1] - 10), 
                                (int(x1) + label_size[0], int(y1)), (0, 255, 0), -1)
                    cv2.putText(image, label, (int(x1), int(y1) - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        return image
    
    def get_statistics(self, detections_list: List[List]) -> dict:
        """Get statistics from detection results."""
        stats = {
            'total_detections': 0,
            'detections_per_class': {name: 0 for name in self.class_names},
            'avg_confidence': 0.0,
            'confidence_per_class': {name: [] for name in self.class_names}
        }
        
        all_confidences = []
        
        for detections in detections_list:
            for detection in detections:
                if len(detection) >= 6:
                    conf, cls_id = detection[4], int(detection[5])
                    
                    if conf >= self.conf_threshold and cls_id < len(self.class_names):
                        stats['total_detections'] += 1
                        stats['detections_per_class'][self.class_names[cls_id]] += 1
                        stats['confidence_per_class'][self.class_names[cls_id]].append(conf)
                        all_confidences.append(conf)
        
        if all_confidences:
            stats['avg_confidence'] = np.mean(all_confidences)
        
        return stats


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run inference with trained YOLOv8 model")
    
    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    parser.add_argument("--source", type=str, required=True, help="Path to image, directory, or video")
    parser.add_argument("--output", type=str, default="inference_results", help="Output directory")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold for NMS")
    parser.add_argument("--device", type=str, default="auto", help="Device to use")
    parser.add_argument("--save", action="store_true", help="Save visualization results")
    parser.add_argument("--video", action="store_true", help="Process as video")
    parser.add_argument("--skip-frames", type=int, default=1, help="Skip frames for video processing")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    return parser.parse_args()


def main():
    """Main inference function."""
    args = parse_args()
    
    # Setup logging
    setup_logging("INFO" if not args.verbose else "DEBUG")
    
    # Initialize detector
    try:
        detector = IndustrialSafetyDetector(
            model_path=args.model,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            device=args.device
        )
    except Exception as e:
        logging.error(f"Failed to initialize detector: {e}")
        return
    
    # Run inference based on source type
    try:
        source_path = Path(args.source)
        
        if args.video or source_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
            # Video processing
            output_video = Path(args.output) / f"result_{source_path.stem}.mp4"
            detector.predict_video(
                video_path=args.source,
                output_path=str(output_video),
                skip_frames=args.skip_frames
            )
            
        elif source_path.is_dir():
            # Directory processing
            detections_list = detector.predict_directory(
                input_dir=args.source,
                output_dir=args.output
            )
            
            # Print statistics
            stats = detector.get_statistics(detections_list)
            print("\n" + "="*50)
            print("INFERENCE STATISTICS")
            print("="*50)
            print(f"Total detections: {stats['total_detections']}")
            print(f"Average confidence: {stats['avg_confidence']:.3f}")
            print("\nDetections per class:")
            for class_name, count in stats['detections_per_class'].items():
                avg_conf = np.mean(stats['confidence_per_class'][class_name]) if stats['confidence_per_class'][class_name] else 0
                print(f"  {class_name}: {count} (avg conf: {avg_conf:.3f})")
            print("="*50)
            
        elif source_path.is_file():
            # Single image processing
            detections = detector.predict_image(
                image_path=args.source,
                save_results=args.save,
                output_dir=args.output
            )
            
            # Print results
            print(f"\nDetected {len(detections)} objects:")
            for i, detection in enumerate(detections):
                x1, y1, x2, y2, conf, cls_id = detection[:6]
                class_name = detector.class_names[int(cls_id)]
                print(f"  {i+1}. {class_name}: {conf:.3f} at [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")
        
        else:
            logging.error(f"Source not found or unsupported: {args.source}")
            return
        
        logging.info("Inference completed successfully!")
        
    except Exception as e:
        logging.error(f"Inference failed: {e}")
        raise


if __name__ == "__main__":
    main()
