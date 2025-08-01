"""
Interactive demo script for industrial safety equipment detection.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    print("ultralytics not installed. Please run: pip install ultralytics")
    sys.exit(1)

from utils import setup_logging


class IndustrialSafetyDemo:
    """Interactive GUI demo for industrial safety equipment detection."""
    
    def __init__(self, model_path: str, conf_threshold: float = 0.25):
        """
        Initialize the demo application.
        
        Args:
            model_path: Path to trained YOLO model
            conf_threshold: Confidence threshold for detections
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.class_names = ["FireExtinguisher", "ToolBox", "OxygenTank"]
        self.class_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # RGB colors
        
        # Load model
        try:
            self.model = YOLO(model_path)
            logging.info(f"Model loaded: {model_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {e}")
            sys.exit(1)
        
        # Initialize GUI
        self.setup_gui()
        
        # Current image data
        self.current_image = None
        self.current_image_path = None
        self.detections = []
    
    def setup_gui(self):
        """Set up the GUI interface."""
        self.root = tk.Tk()
        self.root.title("Industrial Safety Equipment Detection Demo")
        self.root.geometry("1200x800")
        
        # Create main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Control panel
        self.setup_control_panel(main_frame)
        
        # Image display area
        self.setup_image_display(main_frame)
        
        # Results panel
        self.setup_results_panel(main_frame)
        
        # Status bar
        self.setup_status_bar(main_frame)
    
    def setup_control_panel(self, parent):
        """Set up the control panel."""
        control_frame = ttk.LabelFrame(parent, text="Controls", padding="10")
        control_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Load image button
        ttk.Button(control_frame, text="Load Image", 
                  command=self.load_image).grid(row=0, column=0, padx=(0, 10))
        
        # Load from camera button
        ttk.Button(control_frame, text="Camera", 
                  command=self.start_camera).grid(row=0, column=1, padx=(0, 10))
        
        # Confidence threshold
        ttk.Label(control_frame, text="Confidence:").grid(row=0, column=2, padx=(20, 5))
        self.conf_var = tk.DoubleVar(value=self.conf_threshold)
        conf_scale = ttk.Scale(control_frame, from_=0.1, to=1.0, 
                              variable=self.conf_var, orient=tk.HORIZONTAL, length=200)
        conf_scale.grid(row=0, column=3, padx=(0, 10))
        conf_scale.configure(command=self.on_confidence_change)
        
        # Confidence value label
        self.conf_label = ttk.Label(control_frame, text=f"{self.conf_threshold:.2f}")
        self.conf_label.grid(row=0, column=4, padx=(0, 20))
        
        # Run detection button
        self.detect_button = ttk.Button(control_frame, text="Run Detection", 
                                       command=self.run_detection, state=tk.DISABLED)
        self.detect_button.grid(row=0, column=5, padx=(0, 10))
        
        # Save results button
        self.save_button = ttk.Button(control_frame, text="Save Results", 
                                     command=self.save_results, state=tk.DISABLED)
        self.save_button.grid(row=0, column=6)
    
    def setup_image_display(self, parent):
        """Set up the image display area."""
        # Image frame
        img_frame = ttk.LabelFrame(parent, text="Image", padding="10")
        img_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        img_frame.columnconfigure(0, weight=1)
        img_frame.rowconfigure(0, weight=1)
        
        # Image canvas with scrollbars
        canvas_frame = ttk.Frame(img_frame)
        canvas_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.rowconfigure(0, weight=1)
        
        self.image_canvas = tk.Canvas(canvas_frame, bg='white', width=800, height=600)
        self.image_canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Scrollbars
        h_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=self.image_canvas.xview)
        h_scrollbar.grid(row=1, column=0, sticky=(tk.W, tk.E))
        self.image_canvas.configure(xscrollcommand=h_scrollbar.set)
        
        v_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.image_canvas.yview)
        v_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.image_canvas.configure(yscrollcommand=v_scrollbar.set)
    
    def setup_results_panel(self, parent):
        """Set up the results panel."""
        results_frame = ttk.LabelFrame(parent, text="Detection Results", padding="10")
        results_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        results_frame.columnconfigure(0, weight=1)
        
        # Results text area
        text_frame = ttk.Frame(results_frame)
        text_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))
        text_frame.columnconfigure(0, weight=1)
        
        self.results_text = tk.Text(text_frame, height=8, width=80)
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        # Scrollbar for text
        text_scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.results_text.yview)
        text_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.results_text.configure(yscrollcommand=text_scrollbar.set)
    
    def setup_status_bar(self, parent):
        """Set up the status bar."""
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(parent, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E))
    
    def load_image(self):
        """Load an image file."""
        file_types = [
            ('Image files', '*.jpg *.jpeg *.png *.bmp *.tiff'),
            ('JPEG files', '*.jpg *.jpeg'),
            ('PNG files', '*.png'),
            ('All files', '*.*')
        ]
        
        file_path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=file_types
        )
        
        if file_path:
            try:
                # Load and display image
                self.current_image_path = file_path
                self.current_image = cv2.imread(file_path)
                self.current_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
                
                self.display_image(self.current_image)
                
                # Enable detection button
                self.detect_button.configure(state=tk.NORMAL)
                self.status_var.set(f"Loaded: {Path(file_path).name}")
                
                # Clear previous results
                self.results_text.delete(1.0, tk.END)
                self.detections = []
                self.save_button.configure(state=tk.DISABLED)
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {e}")
    
    def display_image(self, image_array, detections=None):
        """Display image on canvas with optional detections."""
        # Create a copy for drawing
        display_image = image_array.copy()
        
        # Draw detections if provided
        if detections:
            display_image = self.draw_detections(display_image, detections)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(display_image)
        
        # Resize if too large
        max_size = 800
        if pil_image.width > max_size or pil_image.height > max_size:
            pil_image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        # Convert to PhotoImage
        self.photo = ImageTk.PhotoImage(pil_image)
        
        # Clear canvas and display image
        self.image_canvas.delete("all")
        self.image_canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        
        # Update scroll region
        self.image_canvas.configure(scrollregion=self.image_canvas.bbox("all"))
    
    def draw_detections(self, image, detections):
        """Draw detection boxes on image."""
        image = image.copy()
        h, w = image.shape[:2]
        
        for detection in detections:
            if len(detection) >= 6:
                x1, y1, x2, y2, conf, cls_id = detection[:6]
                
                if conf >= self.conf_threshold and int(cls_id) < len(self.class_names):
                    # Convert coordinates to integers
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                    
                    # Get class info
                    class_name = self.class_names[int(cls_id)]
                    color = self.class_colors[int(cls_id) % len(self.class_colors)]
                    
                    # Draw bounding box
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw label
                    label = f"{class_name}: {conf:.2f}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    
                    # Label background
                    cv2.rectangle(image, (x1, y1 - label_size[1] - 10), 
                                (x1 + label_size[0], y1), color, -1)
                    
                    # Label text
                    cv2.putText(image, label, (x1, y1 - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return image
    
    def run_detection(self):
        """Run object detection on current image."""
        if self.current_image is None:
            messagebox.showwarning("Warning", "Please load an image first")
            return
        
        try:
            self.status_var.set("Running detection...")
            self.root.update()
            
            # Run inference
            results = self.model(
                self.current_image_path,
                conf=self.conf_threshold,
                verbose=False
            )
            
            # Extract detections
            self.detections = []
            if results and len(results) > 0:
                result = results[0]
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy()
                    
                    for box, conf, cls_id in zip(boxes, confidences, class_ids):
                        detection = [*box, conf, cls_id]
                        self.detections.append(detection)
            
            # Display results
            self.display_image(self.current_image, self.detections)
            self.update_results_text()
            
            # Enable save button
            self.save_button.configure(state=tk.NORMAL)
            
            self.status_var.set(f"Detection complete - Found {len(self.detections)} objects")
            
        except Exception as e:
            messagebox.showerror("Error", f"Detection failed: {e}")
            self.status_var.set("Detection failed")
    
    def update_results_text(self):
        """Update the results text area."""
        self.results_text.delete(1.0, tk.END)
        
        if not self.detections:
            self.results_text.insert(tk.END, "No objects detected.")
            return
        
        # Filter detections by confidence
        filtered_detections = [
            d for d in self.detections 
            if len(d) >= 6 and d[4] >= self.conf_threshold
        ]
        
        self.results_text.insert(tk.END, f"Detected {len(filtered_detections)} objects:\\n\\n")
        
        # Group by class
        class_counts = {name: 0 for name in self.class_names}
        
        for i, detection in enumerate(filtered_detections):
            x1, y1, x2, y2, conf, cls_id = detection[:6]
            class_name = self.class_names[int(cls_id)]
            class_counts[class_name] += 1
            
            self.results_text.insert(tk.END, 
                f"{i+1}. {class_name}\\n"
                f"   Confidence: {conf:.3f}\\n"
                f"   Location: [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]\\n"
                f"   Size: {x2-x1:.0f} x {y2-y1:.0f} pixels\\n\\n"
            )
        
        # Summary
        self.results_text.insert(tk.END, "Summary:\\n")
        for class_name, count in class_counts.items():
            if count > 0:
                self.results_text.insert(tk.END, f"  {class_name}: {count}\\n")
    
    def on_confidence_change(self, value):
        """Handle confidence threshold change."""
        self.conf_threshold = float(value)
        self.conf_label.configure(text=f"{self.conf_threshold:.2f}")
        
        # Update display if we have detections
        if self.detections and self.current_image is not None:
            self.display_image(self.current_image, self.detections)
            self.update_results_text()
    
    def save_results(self):
        """Save detection results."""
        if not self.detections or self.current_image is None:
            messagebox.showwarning("Warning", "No results to save")
            return
        
        # Ask for save location
        save_path = filedialog.asksaveasfilename(
            title="Save results",
            defaultextension=".jpg",
            filetypes=[
                ('JPEG files', '*.jpg'),
                ('PNG files', '*.png'),
                ('All files', '*.*')
            ]
        )
        
        if save_path:
            try:
                # Create image with detections
                result_image = self.draw_detections(self.current_image, self.detections)
                result_image = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
                
                # Save image
                cv2.imwrite(save_path, result_image)
                
                # Save text results
                text_path = Path(save_path).with_suffix('.txt')
                with open(text_path, 'w') as f:
                    f.write("Industrial Safety Equipment Detection Results\\n")
                    f.write("=" * 50 + "\\n\\n")
                    f.write(f"Source image: {self.current_image_path}\\n")
                    f.write(f"Model: {self.model_path}\\n")
                    f.write(f"Confidence threshold: {self.conf_threshold}\\n\\n")
                    
                    filtered_detections = [
                        d for d in self.detections 
                        if len(d) >= 6 and d[4] >= self.conf_threshold
                    ]
                    
                    f.write(f"Detected objects: {len(filtered_detections)}\\n\\n")
                    
                    for i, detection in enumerate(filtered_detections):
                        x1, y1, x2, y2, conf, cls_id = detection[:6]
                        class_name = self.class_names[int(cls_id)]
                        
                        f.write(f"{i+1}. {class_name}\\n")
                        f.write(f"   Confidence: {conf:.3f}\\n")
                        f.write(f"   Bounding box: [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]\\n\\n")
                
                messagebox.showinfo("Success", f"Results saved to:\\n{save_path}\\n{text_path}")
                self.status_var.set("Results saved")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save results: {e}")
    
    def start_camera(self):
        """Start camera capture (placeholder for now)."""
        messagebox.showinfo("Info", "Camera functionality not implemented in this demo.\\n"
                                  "You can extend this by adding webcam capture using cv2.VideoCapture(0)")
    
    def run(self):
        """Run the demo application."""
        self.root.mainloop()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Interactive demo for industrial safety equipment detection")
    
    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    parser.add_argument("--conf", type=float, default=0.25, help="Initial confidence threshold")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    return parser.parse_args()


def main():
    """Main demo function."""
    args = parse_args()
    
    # Setup logging
    setup_logging("INFO" if not args.verbose else "DEBUG")
    
    # Check if model exists
    if not Path(args.model).exists():
        print(f"Error: Model not found: {args.model}")
        return
    
    try:
        # Create and run demo
        demo = IndustrialSafetyDemo(
            model_path=args.model,
            conf_threshold=args.conf
        )
        
        print("Starting interactive demo...")
        print("Controls:")
        print("- Load Image: Select an image file")
        print("- Confidence: Adjust detection threshold")
        print("- Run Detection: Perform object detection")
        print("- Save Results: Save annotated image and results")
        
        demo.run()
        
    except Exception as e:
        print(f"Demo failed: {e}")
        logging.error(f"Demo failed: {e}")


if __name__ == "__main__":
    main()
