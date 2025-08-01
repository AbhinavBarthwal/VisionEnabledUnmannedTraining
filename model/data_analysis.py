"""
Dataset analysis and visualization script for industrial safety equipment detection.
"""

import os
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import json

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from PIL import Image, ImageDraw
import cv2

from utils import setup_logging, create_class_distribution_plot, print_dataset_summary


def analyze_dataset_structure(dataset_yaml_path: str) -> Dict:
    """Analyze the structure of the dataset."""
    import yaml
    
    dataset_path = Path(dataset_yaml_path)
    base_path = dataset_path.parent
    
    with open(dataset_path, 'r') as f:
        config = yaml.safe_load(f)
    
    analysis = {
        'dataset_config': config,
        'base_path': str(base_path),
        'splits': {}
    }
    
    # Analyze each split
    splits = ['train', 'val', 'test']
    
    for split in splits:
        split_info = {'images': 0, 'labels': 0, 'image_files': [], 'label_files': []}
        
        # Check if split exists in config
        if split in config:
            img_path = base_path / config[split]
            if 'images' not in str(img_path):
                # If path doesn't contain 'images', assume it's the base path
                img_path = img_path / 'images' if (img_path / 'images').exists() else img_path
        else:
            # Default path structure
            img_path = base_path / 'data' / split / 'images'
        
        label_path = img_path.parent / 'labels' if img_path.parent.name == 'images' else img_path.parent / 'labels'
        
        # Count images
        if img_path.exists():
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
            for ext in image_extensions:
                split_info['image_files'].extend(list(img_path.glob(f"*{ext}")))
                split_info['image_files'].extend(list(img_path.glob(f"*{ext.upper()}")))
            split_info['images'] = len(split_info['image_files'])
        
        # Count labels
        if label_path.exists():
            split_info['label_files'] = list(label_path.glob("*.txt"))
            split_info['labels'] = len(split_info['label_files'])
        
        split_info['image_path'] = str(img_path)
        split_info['label_path'] = str(label_path)
        analysis['splits'][split] = split_info
    
    return analysis


def analyze_image_properties(image_files: List[Path]) -> Dict:
    """Analyze properties of images in the dataset."""
    widths, heights, ratios = [], [], []
    channels = []
    file_sizes = []
    
    sample_size = min(100, len(image_files))  # Sample for faster analysis
    sample_files = np.random.choice(image_files, sample_size, replace=False)
    
    for img_path in sample_files:
        try:
            with Image.open(img_path) as img:
                w, h = img.size
                widths.append(w)
                heights.append(h)
                ratios.append(w / h)
                channels.append(len(img.getbands()))
                file_sizes.append(img_path.stat().st_size / 1024)  # KB
        except Exception as e:
            logging.warning(f"Could not analyze image {img_path}: {e}")
    
    return {
        'count': len(image_files),
        'sample_size': len(sample_files),
        'width': {'min': min(widths), 'max': max(widths), 'mean': np.mean(widths), 'std': np.std(widths)},
        'height': {'min': min(heights), 'max': max(heights), 'mean': np.mean(heights), 'std': np.std(heights)},
        'aspect_ratio': {'min': min(ratios), 'max': max(ratios), 'mean': np.mean(ratios), 'std': np.std(ratios)},
        'channels': {'mode': max(set(channels), key=channels.count), 'unique': list(set(channels))},
        'file_size_kb': {'min': min(file_sizes), 'max': max(file_sizes), 'mean': np.mean(file_sizes), 'std': np.std(file_sizes)}
    }


def analyze_annotations(label_files: List[Path], class_names: List[str]) -> Dict:
    """Analyze annotation properties."""
    class_counts = {i: 0 for i in range(len(class_names))}
    bbox_areas = []
    bbox_widths = []
    bbox_heights = []
    objects_per_image = []
    class_areas = {i: [] for i in range(len(class_names))}
    
    for label_file in label_files:
        try:
            objects_in_image = 0
            with open(label_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            x_center, y_center, width, height = map(float, parts[1:5])
                            
                            if class_id < len(class_names):
                                class_counts[class_id] += 1
                                objects_in_image += 1
                                
                                # Calculate area (normalized)
                                area = width * height
                                bbox_areas.append(area)
                                bbox_widths.append(width)
                                bbox_heights.append(height)
                                class_areas[class_id].append(area)
            
            objects_per_image.append(objects_in_image)
            
        except Exception as e:
            logging.warning(f"Could not analyze label {label_file}: {e}")
    
    # Calculate statistics
    total_objects = sum(class_counts.values())
    
    return {
        'total_objects': total_objects,
        'objects_per_image': {
            'min': min(objects_per_image) if objects_per_image else 0,
            'max': max(objects_per_image) if objects_per_image else 0,
            'mean': np.mean(objects_per_image) if objects_per_image else 0,
            'std': np.std(objects_per_image) if objects_per_image else 0
        },
        'class_distribution': {class_names[i]: count for i, count in class_counts.items()},
        'class_counts': class_counts,
        'bbox_area': {
            'min': min(bbox_areas) if bbox_areas else 0,
            'max': max(bbox_areas) if bbox_areas else 0,
            'mean': np.mean(bbox_areas) if bbox_areas else 0,
            'std': np.std(bbox_areas) if bbox_areas else 0
        },
        'bbox_width': {
            'min': min(bbox_widths) if bbox_widths else 0,
            'max': max(bbox_widths) if bbox_widths else 0,
            'mean': np.mean(bbox_widths) if bbox_widths else 0,
            'std': np.std(bbox_widths) if bbox_widths else 0
        },
        'bbox_height': {
            'min': min(bbox_heights) if bbox_heights else 0,
            'max': max(bbox_heights) if bbox_heights else 0,
            'mean': np.mean(bbox_heights) if bbox_heights else 0,
            'std': np.std(bbox_heights) if bbox_heights else 0
        },
        'class_areas': class_areas
    }


def create_visualizations(analysis: Dict, class_names: List[str], save_dir: str = "analysis_results"):
    """Create comprehensive visualizations of the dataset analysis."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Dataset split distribution
    splits = ['train', 'val', 'test']
    split_counts = [analysis['splits'][split]['images'] for split in splits if split in analysis['splits']]
    split_labels = [split.upper() for split in splits if split in analysis['splits']]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Split distribution pie chart
    colors = sns.color_palette("husl", len(split_labels))
    wedges, texts, autotexts = ax1.pie(split_counts, labels=split_labels, autopct='%1.1f%%', colors=colors)
    ax1.set_title('Dataset Split Distribution')
    
    # Class distribution bar plot
    train_analysis = None
    for split in ['train', 'val']:  # Use train or val for class distribution
        if split in analysis['splits'] and analysis['splits'][split]['labels'] > 0:
            label_files = analysis['splits'][split]['label_files']
            train_analysis = analyze_annotations(label_files, class_names)
            break
    
    if train_analysis:
        class_counts = [train_analysis['class_counts'][i] for i in range(len(class_names))]
        bars = ax2.bar(class_names, class_counts, color=sns.color_palette("husl", len(class_names)))
        ax2.set_title('Class Distribution')
        ax2.set_ylabel('Number of Instances')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add count labels on bars
        for bar, count in zip(bars, class_counts):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(class_counts) * 0.01,
                    str(count), ha='center', va='bottom')
    
    # Image resolution distribution (if available)
    if 'train' in analysis['splits'] and analysis['splits']['train']['images'] > 0:
        img_files = analysis['splits']['train']['image_files'][:50]  # Sample
        img_analysis = analyze_image_properties(img_files)
        
        # Scatter plot of image dimensions
        sample_files = img_files[:20]  # Smaller sample for detailed analysis
        widths, heights = [], []
        for img_path in sample_files:
            try:
                with Image.open(img_path) as img:
                    w, h = img.size
                    widths.append(w)
                    heights.append(h)
            except:
                continue
        
        if widths and heights:
            ax3.scatter(widths, heights, alpha=0.6, color='coral')
            ax3.set_xlabel('Width (pixels)')
            ax3.set_ylabel('Height (pixels)')
            ax3.set_title('Image Resolution Distribution (Sample)')
            ax3.grid(True, alpha=0.3)
    
    # Bounding box size distribution
    if train_analysis and train_analysis['total_objects'] > 0:
        all_areas = []
        all_labels = []
        for class_id, areas in train_analysis['class_areas'].items():
            all_areas.extend(areas)
            all_labels.extend([class_names[class_id]] * len(areas))
        
        if all_areas:
            # Box plot of bbox areas by class
            df = pd.DataFrame({'Area': all_areas, 'Class': all_labels})
            sns.boxplot(data=df, x='Class', y='Area', ax=ax4)
            ax4.set_title('Bounding Box Area Distribution by Class')
            ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'dataset_overview.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Detailed statistics plots
    if train_analysis:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Objects per image histogram
        objects_per_img = []
        for label_file in analysis['splits']['train']['label_files'][:100]:  # Sample
            try:
                with open(label_file, 'r') as f:
                    count = sum(1 for line in f if line.strip())
                objects_per_img.append(count)
            except:
                continue
        
        if objects_per_img:
            ax1.hist(objects_per_img, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax1.set_xlabel('Objects per Image')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Distribution of Objects per Image')
            ax1.grid(True, alpha=0.3)
        
        # Class proportion pie chart
        class_counts = [train_analysis['class_counts'][i] for i in range(len(class_names))]
        if sum(class_counts) > 0:
            ax2.pie(class_counts, labels=class_names, autopct='%1.1f%%', 
                   colors=sns.color_palette("husl", len(class_names)))
            ax2.set_title('Class Proportion')
        
        # Bounding box width vs height scatter
        widths = []
        heights = []
        for label_file in analysis['splits']['train']['label_files'][:50]:  # Sample
            try:
                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            width, height = float(parts[3]), float(parts[4])
                            widths.append(width)
                            heights.append(height)
            except:
                continue
        
        if widths and heights:
            ax3.scatter(widths, heights, alpha=0.6, color='lightgreen')
            ax3.set_xlabel('Normalized Width')
            ax3.set_ylabel('Normalized Height')
            ax3.set_title('Bounding Box Dimensions')
            ax3.grid(True, alpha=0.3)
        
        # Class distribution comparison across splits
        split_class_data = []
        for split in ['train', 'val', 'test']:
            if split in analysis['splits'] and analysis['splits'][split]['labels'] > 0:
                label_files = analysis['splits'][split]['label_files']
                split_analysis = analyze_annotations(label_files, class_names)
                for i, class_name in enumerate(class_names):
                    split_class_data.append({
                        'Split': split.upper(),
                        'Class': class_name,
                        'Count': split_analysis['class_counts'][i]
                    })
        
        if split_class_data:
            df = pd.DataFrame(split_class_data)
            sns.barplot(data=df, x='Class', y='Count', hue='Split', ax=ax4)
            ax4.set_title('Class Distribution Across Splits')
            ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'detailed_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    logging.info(f"Visualizations saved to: {save_dir}")


def visualize_sample_images(analysis: Dict, class_names: List[str], num_samples: int = 9, 
                          save_dir: str = "analysis_results"):
    """Visualize sample images with annotations."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Get sample images from training set
    if 'train' not in analysis['splits']:
        logging.warning("No training split found for sample visualization")
        return
    
    train_split = analysis['splits']['train']
    image_files = train_split['image_files']
    label_path = Path(train_split['label_path'])
    
    if len(image_files) == 0:
        logging.warning("No images found for sample visualization")
        return
    
    # Select random samples
    sample_size = min(num_samples, len(image_files))
    sample_images = np.random.choice(image_files, sample_size, replace=False)
    
    # Create subplot grid
    rows = int(np.ceil(np.sqrt(sample_size)))
    cols = int(np.ceil(sample_size / rows))
    
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'brown']
    
    for idx, img_path in enumerate(sample_images):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        
        try:
            # Load image
            image = Image.open(img_path)
            ax.imshow(image)
            ax.axis('off')
            ax.set_title(f'{img_path.stem}', fontsize=10)
            
            # Load corresponding label
            label_file = label_path / f"{img_path.stem}.txt"
            if label_file.exists():
                img_width, img_height = image.size
                
                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            x_center, y_center, width, height = map(float, parts[1:5])
                            
                            # Convert normalized coordinates to pixel coordinates
                            x_center *= img_width
                            y_center *= img_height
                            width *= img_width
                            height *= img_height
                            
                            # Calculate bounding box corners
                            x1 = x_center - width / 2
                            y1 = y_center - height / 2
                            x2 = x_center + width / 2
                            y2 = y_center + height / 2
                            
                            # Draw bounding box
                            color = colors[class_id % len(colors)]
                            rect = plt.Rectangle((x1, y1), width, height, 
                                               fill=False, edgecolor=color, linewidth=2)
                            ax.add_patch(rect)
                            
                            # Add label
                            if class_id < len(class_names):
                                ax.text(x1, y1-5, class_names[class_id], 
                                       color=color, fontsize=8, fontweight='bold',
                                       bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
        
        except Exception as e:
            ax.text(0.5, 0.5, f'Error loading\n{img_path.name}', 
                   ha='center', va='center', transform=ax.transAxes)
            logging.warning(f"Could not visualize {img_path}: {e}")
    
    # Hide empty subplots
    for idx in range(sample_size, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'Sample Images with Annotations (n={sample_size})', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_dir / 'sample_images.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Sample images visualization saved to: {save_dir}/sample_images.png")


def save_analysis_report(analysis: Dict, class_names: List[str], save_path: str = "analysis_report.txt"):
    """Save detailed analysis report to text file."""
    with open(save_path, 'w') as f:
        f.write("DATASET ANALYSIS REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        # Dataset overview
        f.write("DATASET OVERVIEW\n")
        f.write("-" * 20 + "\n")
        f.write(f"Classes: {len(class_names)}\n")
        f.write(f"Class names: {', '.join(class_names)}\n\n")
        
        # Split information
        f.write("SPLIT INFORMATION\n")
        f.write("-" * 20 + "\n")
        total_images = 0
        total_labels = 0
        
        for split, info in analysis['splits'].items():
            f.write(f"{split.upper()}:\n")
            f.write(f"  Images: {info['images']}\n")
            f.write(f"  Labels: {info['labels']}\n")
            f.write(f"  Image path: {info['image_path']}\n")
            f.write(f"  Label path: {info['label_path']}\n\n")
            
            total_images += info['images']
            total_labels += info['labels']
        
        f.write(f"TOTAL:\n")
        f.write(f"  Images: {total_images}\n")
        f.write(f"  Labels: {total_labels}\n\n")
        
        # Annotation analysis for training set
        if 'train' in analysis['splits'] and analysis['splits']['train']['labels'] > 0:
            label_files = analysis['splits']['train']['label_files']
            ann_analysis = analyze_annotations(label_files, class_names)
            
            f.write("ANNOTATION ANALYSIS (Training Set)\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total objects: {ann_analysis['total_objects']}\n")
            f.write(f"Objects per image: {ann_analysis['objects_per_image']['mean']:.2f} ± {ann_analysis['objects_per_image']['std']:.2f}\n")
            f.write(f"  Min: {ann_analysis['objects_per_image']['min']}\n")
            f.write(f"  Max: {ann_analysis['objects_per_image']['max']}\n\n")
            
            f.write("Class distribution:\n")
            for class_name, count in ann_analysis['class_distribution'].items():
                percentage = (count / ann_analysis['total_objects']) * 100 if ann_analysis['total_objects'] > 0 else 0
                f.write(f"  {class_name}: {count} ({percentage:.1f}%)\n")
            f.write("\n")
            
            f.write("Bounding box statistics (normalized):\n")
            f.write(f"  Area: {ann_analysis['bbox_area']['mean']:.4f} ± {ann_analysis['bbox_area']['std']:.4f}\n")
            f.write(f"  Width: {ann_analysis['bbox_width']['mean']:.4f} ± {ann_analysis['bbox_width']['std']:.4f}\n")
            f.write(f"  Height: {ann_analysis['bbox_height']['mean']:.4f} ± {ann_analysis['bbox_height']['std']:.4f}\n\n")
        
        # Image analysis for training set
        if 'train' in analysis['splits'] and analysis['splits']['train']['images'] > 0:
            img_files = analysis['splits']['train']['image_files']
            img_analysis = analyze_image_properties(img_files)
            
            f.write("IMAGE ANALYSIS (Training Set Sample)\n")
            f.write("-" * 40 + "\n")
            f.write(f"Sample size: {img_analysis['sample_size']} / {img_analysis['count']}\n")
            f.write(f"Resolution (W x H):\n")
            f.write(f"  Width: {img_analysis['width']['mean']:.0f} ± {img_analysis['width']['std']:.0f} pixels\n")
            f.write(f"    Range: {img_analysis['width']['min']:.0f} - {img_analysis['width']['max']:.0f}\n")
            f.write(f"  Height: {img_analysis['height']['mean']:.0f} ± {img_analysis['height']['std']:.0f} pixels\n")
            f.write(f"    Range: {img_analysis['height']['min']:.0f} - {img_analysis['height']['max']:.0f}\n")
            f.write(f"  Aspect ratio: {img_analysis['aspect_ratio']['mean']:.2f} ± {img_analysis['aspect_ratio']['std']:.2f}\n")
            f.write(f"  Channels: {img_analysis['channels']['mode']} (typical)\n")
            f.write(f"  File size: {img_analysis['file_size_kb']['mean']:.1f} ± {img_analysis['file_size_kb']['std']:.1f} KB\n\n")
        
        f.write("RECOMMENDATIONS\n")
        f.write("-" * 20 + "\n")
        f.write("Based on the analysis:\n")
        f.write("1. Consider data augmentation if class distribution is imbalanced\n")
        f.write("2. Monitor for overfitting given the dataset size\n")
        f.write("3. Use appropriate input resolution based on image sizes\n")
        f.write("4. Consider different training strategies for rare classes\n")
    
    logging.info(f"Analysis report saved to: {save_path}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Analyze dataset for industrial safety equipment detection")
    
    parser.add_argument("--data", type=str, default="../HackByte_Dataset/yolo_params.yaml", 
                       help="Path to dataset YAML file")
    parser.add_argument("--output", type=str, default="analysis_results", 
                       help="Output directory for analysis results")
    parser.add_argument("--samples", type=int, default=9, 
                       help="Number of sample images to visualize")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    return parser.parse_args()


def main():
    """Main analysis function."""
    args = parse_args()
    
    # Setup logging
    setup_logging("INFO" if not args.verbose else "DEBUG")
    
    # Class names
    class_names = ["FireExtinguisher", "ToolBox", "OxygenTank"]
    
    logging.info("Starting dataset analysis...")
    
    try:
        # Analyze dataset structure
        analysis = analyze_dataset_structure(args.data)
        
        # Create output directory
        Path(args.output).mkdir(parents=True, exist_ok=True)
        
        # Print basic summary
        print_dataset_summary(args.data, class_names)
        
        # Create visualizations
        create_visualizations(analysis, class_names, args.output)
        
        # Visualize sample images
        visualize_sample_images(analysis, class_names, args.samples, args.output)
        
        # Save detailed report
        save_analysis_report(analysis, class_names, Path(args.output) / "analysis_report.txt")
        
        # Save analysis data as JSON
        # Convert Path objects to strings for JSON serialization
        analysis_json = analysis.copy()
        for split, info in analysis_json['splits'].items():
            info['image_files'] = [str(p) for p in info['image_files']]
            info['label_files'] = [str(p) for p in info['label_files']]
        
        with open(Path(args.output) / "analysis_data.json", 'w') as f:
            json.dump(analysis_json, f, indent=2)
        
        logging.info("Dataset analysis completed successfully!")
        logging.info(f"Results saved to: {args.output}")
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        print(f"Results directory: {args.output}")
        print("Generated files:")
        print("  - dataset_overview.png")
        print("  - detailed_analysis.png") 
        print("  - sample_images.png")
        print("  - analysis_report.txt")
        print("  - analysis_data.json")
        print("="*60)
        
    except Exception as e:
        logging.error(f"Analysis failed: {e}")
        raise


if __name__ == "__main__":
    main()
