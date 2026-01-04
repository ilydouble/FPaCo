import argparse
import json
import os
import glob
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

def analyze_detections(dataset_dir):
    dataset_path = Path(dataset_dir)
    json_files = list(dataset_path.rglob('*.json'))
    
    print(f"Found {len(json_files)} JSON files in {dataset_dir}")
    
    all_areas = []
    all_confidences = []
    all_ratios = []
    all_centroids = []
    all_counts_per_img = []
    
    out_of_bounds_count = 0
    empty_detections_count = 0
    count_large_50 = 0
    count_large_80 = 0
    total_images = len(json_files)
    
    for json_file in tqdm(json_files, desc="Analyzing"):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # 尝试获取对应图片尺寸
            img_path_str = data.get('image_path')
            if not img_path_str:
                # 尝试推断
                img_path = json_file.with_suffix('.png') 
                if not img_path.exists(): img_path = json_file.with_suffix('.jpg')
            else:
                img_path = Path(img_path_str)
                if not img_path.is_absolute():
                     # 尝试相对于 json location
                     img_path = json_file.parent / img_path 

            if img_path and img_path.exists():
                with Image.open(img_path) as img:
                    width, height = img.size
            else:
                width, height = None, None
                
            detections = data.get('detections', [])
            if not detections:
                empty_detections_count += 1
                continue
                
            for det in detections:
                box = det['bbox'] # [x1, y1, x2, y2]
                score = det.get('confidence', 1.0)
                
                x1, y1, x2, y2 = box
                area = (x2 - x1) * (y2 - y1)
                
                # Check Bounds if size is known
                if width and height:
                    # Allow small margin for floating point
                    if x1 < -1 or y1 < -1 or x2 > width + 1 or y2 > height + 1:
                        out_of_bounds_count += 1
                        # print(f"OOB: {json_file.name} Box: {box} Size: {width}x{height}")

                # Relative Area if size is known
                if width and height:
                    rel_area = area / (width * height)
                    all_areas.append(rel_area)
                    
                    # Aspect Ratio
                    w_box = x2 - x1
                    h_box = y2 - y1
                    if h_box > 0:
                        all_ratios.append(w_box / h_box)
                    
                    # Centroid (Normalized)
                    cx = (x1 + x2) / 2.0 / width
                    cy = (y1 + y2) / 2.0 / height
                    all_centroids.append((cx, cy))
                    
                    if rel_area > 0.5: count_large_50 += 1
                    if rel_area > 0.8: count_large_80 += 1
                else:
                    pass # Absolute area not as useful without scale, but could log
                    
                all_confidences.append(score)
            
            all_counts_per_img.append(len(detections))
                
        except Exception as e:
            print(f"Error reading {json_file}: {e}")

    # Report
    print(f"\nAnalysis Report for {dataset_dir}")
    print(f"Total Images Checked: {total_images}")
    print(f"Images with NO detections: {empty_detections_count} ({empty_detections_count/total_images*100:.2f}%)")
    print(f"Detections Out of Bounds: {out_of_bounds_count}")

    if all_areas:
        all_areas = np.array(all_areas)
        print(f"\nBox Area (Relative to Image):")
        print(f"  Mean: {all_areas.mean():.4f}")
        print(f"  Min:  {all_areas.min():.4f}")
        print(f"  Max:  {all_areas.max():.4f}")
        print(f"  > 50% Image: {count_large_50} ({count_large_50/len(all_areas)*100:.1f}%)")
        print(f"  > 80% Image: {count_large_80} ({count_large_80/len(all_areas)*100:.1f}%)")

        print(f"\nDetections per Image:")
        print(f"  Mean: {np.mean(all_counts_per_img):.2f}")
        print(f"  Max:  {np.max(all_counts_per_img)}")

        # Plot Histogram
        plt.figure(figsize=(15, 10))
        
        # 1. Area Distribution
        plt.subplot(2, 3, 1)
        plt.hist(all_areas, bins=50, color='blue', alpha=0.7)
        plt.title('Relative Box Area')
        plt.xlabel('Area / ImageArea')
        
        # 2. Confidence Distribution
        plt.subplot(2, 3, 2)
        plt.hist(all_confidences, bins=50, color='green', alpha=0.7)
        plt.title('Confidence Score')
        plt.xlabel('Score')
        
        # 3. Aspect Ratio
        plt.subplot(2, 3, 3)
        plt.hist(all_ratios, bins=50, color='orange', range=(0, 3), alpha=0.7)
        plt.title('Aspect Ratio (W/H)')
        plt.xlabel('Ratio')
        
        # 4. Count Per Image
        plt.subplot(2, 3, 4)
        plt.hist(all_counts_per_img, bins=range(min(all_counts_per_img), max(all_counts_per_img)+2), color='purple', alpha=0.7, align='left')
        plt.title('Detections Per Image')
        plt.xlabel('Count')
        
        # 5. Spatial Distribution
        plt.subplot(2, 3, 5)
        if all_centroids:
            cx, cy = zip(*all_centroids)
            plt.scatter(cx, cy, s=10, alpha=0.3, color='red')
            plt.xlim(0, 1)
            plt.ylim(1, 0) # Invert Y to match image coordinates
            plt.title('Box Centroids (Norm)')
            plt.xlabel('X')
            plt.ylabel('Y')
            
        plt.tight_layout()
        save_path = dataset_path / 'detection_stats_advanced.png'
        plt.savefig(save_path)
        print(f"Saved advanced distribution plot to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help="Path to dataset root containing JSONs")
    args = parser.parse_args()
    
    analyze_detections(args.dataset)
