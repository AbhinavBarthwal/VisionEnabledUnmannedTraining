#!/usr/bin/env python3
"""
Real-time Training Monitor for Industrial Safety Detection
"""

import time
import os
import glob
from pathlib import Path

def monitor_training():
    """Monitor training progress and results"""
    
    print("📊 TRAINING PROGRESS MONITOR")
    print("=" * 50)
    
    # Look for latest results directory
    results_dirs = glob.glob("runs/detect/industrial_safety_optimized*")
    if not results_dirs:
        print("⏳ Waiting for training to start...")
        return
    
    latest_dir = max(results_dirs, key=os.path.getctime)
    results_file = Path(latest_dir) / "results.csv"
    
    print(f"📁 Monitoring: {latest_dir}")
    
    if results_file.exists():
        try:
            with open(results_file, 'r') as f:
                lines = f.readlines()
                
            if len(lines) > 1:  # Header + at least one data line
                header = lines[0].strip().split(',')
                latest_data = lines[-1].strip().split(',')
                
                # Find indices for key metrics
                epoch_idx = header.index('epoch') if 'epoch' in header else 0
                precision_idx = header.index('metrics/precision(B)') if 'metrics/precision(B)' in header else -1
                recall_idx = header.index('metrics/recall(B)') if 'metrics/recall(B)' in header else -1
                map50_idx = header.index('metrics/mAP50(B)') if 'metrics/mAP50(B)' in header else -1
                map50_95_idx = header.index('metrics/mAP50-95(B)') if 'metrics/mAP50-95(B)' in header else -1
                
                epoch = latest_data[epoch_idx] if epoch_idx < len(latest_data) else "N/A"
                precision = latest_data[precision_idx] if precision_idx != -1 and precision_idx < len(latest_data) else "N/A"
                recall = latest_data[recall_idx] if recall_idx != -1 and recall_idx < len(latest_data) else "N/A"
                map50 = latest_data[map50_idx] if map50_idx != -1 and map50_idx < len(latest_data) else "N/A"
                map50_95 = latest_data[map50_95_idx] if map50_95_idx != -1 and map50_95_idx < len(latest_data) else "N/A"
                
                print(f"🔄 Current Epoch: {epoch}")
                print(f"🎯 Precision: {precision}")
                print(f"🎯 Recall: {recall}")
                print(f"📈 mAP@0.5: {map50}")
                print(f"📈 mAP@0.5-0.95: {map50_95}")
                
                # Performance assessment
                try:
                    p_val = float(precision) if precision != "N/A" else 0
                    r_val = float(recall) if recall != "N/A" else 0
                    
                    print("\n🏆 Performance Assessment:")
                    if p_val > 0.9:
                        print("✅ Precision: EXCELLENT (>90%)")
                    elif p_val > 0.8:
                        print("🟡 Precision: GOOD (>80%)")
                    else:
                        print("🔴 Precision: NEEDS IMPROVEMENT")
                        
                    if r_val > 0.8:
                        print("✅ Recall: EXCELLENT (>80%)")
                    elif r_val > 0.7:
                        print("🟡 Recall: GOOD (>70%)")
                    else:
                        print("🔴 Recall: NEEDS IMPROVEMENT")
                        
                except ValueError:
                    print("⏳ Metrics being calculated...")
                    
            else:
                print("⏳ Training just started, waiting for first epoch...")
        
        except Exception as e:
            print(f"⚠️ Error reading results: {e}")
    
    else:
        print("⏳ Results file not yet created...")

if __name__ == "__main__":
    monitor_training()
