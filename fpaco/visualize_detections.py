import os
import json
import argparse
import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

def visualize(dataset_dir, limit=5, output_dir="vis_results"):
    dataset_path = Path(dataset_dir)
    json_files = list(dataset_path.rglob("*.json"))
    
    if not json_files:
        print(f"No JSON files found in {dataset_dir}")
        return

    if limit is not None and limit > 0:
        random.shuffle(json_files)
        selected_files = json_files[:limit]
        print(f"Visualizing random {limit} images.")
    else:
        selected_files = json_files
        print(f"Visualizing ALL {len(json_files)} images.")
    
    os.makedirs(output_dir, exist_ok=True)
    
    for json_file in selected_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # 1. Try path from JSON (if absolute or correct relative)
            img_path_str = data.get('image_path')
            if img_path_str:
                img_path = Path(img_path_str)
                if not img_path.exists():
                     # Try treating it as relative to dataset_dir if it looks absolute but fails
                     # or relative to the JSON file
                     pass 

            # 2. Try same name as JSON in same folder (Most common for FPaCo)
            if not img_path_str or not img_path.exists():
                for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']:
                    probe = json_file.with_suffix(ext)
                    if probe.exists():
                        img_path = probe
                        break
            
            if not img_path.exists():
                print(f"[WARN] Image not found for {json_file.name}")
                continue
                
            img = Image.open(img_path).convert("RGB")
            draw = ImageDraw.Draw(img)
            
            detections = data.get('detections', [])
            for det in detections:
                bbox = det.get('bbox', [])
                if not bbox: continue
                
                label = det.get('label', 'unknown')
                conf = det.get('confidence', 1.0)
                
                # Draw Box
                draw.rectangle(bbox, outline="red", width=3)
                
                # Draw Text
                text = f"{label} ({conf:.2f})"
                # Default font
                draw.text((bbox[0], bbox[1]), text, fill="red")
            
            save_name = f"{dataset_path.name}_{img_path.name}"
            img.save(os.path.join(output_dir, save_name))
            print(f"Saved visualization to {os.path.join(output_dir, save_name)}")
            
        except Exception as e:
            print(f"Error visualizing {json_file}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--limit', type=int, default=5)
    parser.add_argument('--output-dir', type=str, default="vis_results")
    args = parser.parse_args()
    
    visualize(args.dataset, args.limit, args.output_dir)
