import os
import random
import json
import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob

# Inline the heatmap generation function to avoid import/tensor issues
def generate_gaussian_heatmap(h, w, boxes, scores=None, sigma=15):
    heatmap = np.zeros((h, w), dtype=np.float32)
    if len(boxes) == 0:
        return heatmap

    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        
        if scores is not None:
             # Handle both tensor and float
            score = scores[i]
            if hasattr(score, 'item'):
                score = score.item()
        else:
            score = 1.0
        
        # Distance squared to center
        dist_sq = (xx - cx)**2 + (yy - cy)**2
        # Gaussian blob
        blob = np.exp(-dist_sq / (2 * sigma**2))
        
        # Aggregate using maximum
        heatmap = np.maximum(heatmap, blob * score)
        
    return heatmap

def resolve_image_path(json_path, image_path_in_json):
    # Case 1: As is (relative to current working dir)
    if os.path.exists(image_path_in_json):
        return image_path_in_json
    
    # Case 2: Strip leading ../
    clean_path = image_path_in_json
    while clean_path.startswith('../'):
        clean_path = clean_path[3:]
    
    if os.path.exists(clean_path):
        return clean_path
        
    # Case 3: Relative to the JSON file
    json_dir = os.path.dirname(json_path)
    rel_path = os.path.join(json_dir, image_path_in_json)
    if os.path.exists(rel_path):
        return rel_path
        
    return None

def visualize_samples():
    dataset_dirs = {
        'fingerA': 'datasets/fingerA',
        'fingerB': 'datasets/fingerB',
        'fingerC': 'datasets/fingerC',
        'mias': 'datasets/mias_classification_dataset',
        'aptos': 'datasets/aptos_classification_dataset',
        'octa': 'datasets/octa_classification_dataset'
    }
    
    # Create a figure
    num_datasets = len(dataset_dirs)
    fig, axes = plt.subplots(num_datasets, 3, figsize=(15, 4 * num_datasets))
    plt.subplots_adjust(hspace=0.4)
    
    row_idx = 0
    for name, path in dataset_dirs.items():
        print(f"Processing {name}...")
        
        # Find all JSONs
        # Use simple recursive glob
        json_files = glob.glob(os.path.join(path, '**', '*.json'), recursive=True)
        
        if not json_files:
            print(f"  No JSON files found in {path}")
            axes[row_idx, 0].text(0.5, 0.5, 'No JSON found', ha='center')
            row_idx += 1
            continue
            
        # Pick random sample
        sample_json = random.choice(json_files)
        print(f"  Selected: {sample_json}")
        
        with open(sample_json, 'r') as f:
            data = json.load(f)
            
        img_path = resolve_image_path(sample_json, data['image_path'])
        
        if not img_path:
            print(f"  Image not found: {data['image_path']}")
            axes[row_idx, 0].text(0.5, 0.5, 'Image not found', ha='center')
            row_idx += 1
            continue
            
        # Load Image
        # cv2 loads BGR, convert to RGB
        img = cv2.imread(img_path)
        if img is None:
             print(f"  Failed to load image: {img_path}")
             axes[row_idx, 0].text(0.5, 0.5, 'Load failed', ha='center')
             row_idx += 1
             continue
             
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        
        # Process Detections
        boxes = []
        scores = []
        for det in data.get('detections', []):
            boxes.append(det['bbox'])
            scores.append(det.get('confidence', 1.0))
            
        # Generate Heatmap
        heatmap = generate_gaussian_heatmap(h, w, boxes, scores, sigma=50) # Larger sigma for visibility
        
        # Draw Boxes on a copy of image
        img_boxes = img_rgb.copy()
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img_boxes, (x1, y1), (x2, y2), (255, 0, 0), 2)
            # Add label?
            # cv2.putText(img_boxes, "{:.2f}".format(scores[i]), (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # Plot Original
        axes[row_idx, 0].imshow(img_rgb)
        axes[row_idx, 0].set_title(f"{name} - Original")
        axes[row_idx, 0].axis('off')
        
        # Plot Boxes
        axes[row_idx, 1].imshow(img_boxes)
        axes[row_idx, 1].set_title(f"{name} - BBoxes ({len(boxes)})")
        axes[row_idx, 1].axis('off')
        
        # Plot Heatmap
        im_h = axes[row_idx, 2].imshow(heatmap, cmap='jet')
        axes[row_idx, 2].set_title(f"{name} - Heatmap")
        axes[row_idx, 2].axis('off')
        # plt.colorbar(im_h, ax=axes[row_idx, 2])
        
        row_idx += 1
        
    output_file = 'heatmap_visualization.png'
    plt.savefig(output_file, bbox_inches='tight')
    print(f"Saved visualization to {output_file}")

if __name__ == "__main__":
    visualize_samples()
