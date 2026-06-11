import os
import argparse
import json
import torch
import traceback
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from florence2_agent import Florence2Agent

def generate_detections(dataset_dir, output_dir=None, model_id='microsoft/Florence-2-large-ft', device='cuda'):
    """
    Iterate over dataset, detect using Florence-2, and save JSONs.
    """
    
    # Initialize Agent
    print(f"Initializing Florence-2 Agent ({model_id})...")
    agent = Florence2Agent(model_id=model_id, device=device)
    
    dataset_path = Path(dataset_dir)
    
    # Prompt Map (Simple mapping for now, can be extended to use LLM guidance)
    # Customize for your datasets
    PROMPT_MAP = {
        'octa': 'lesion, vessel anomaly, choroidal neovascularization, capillary non-perfusion, drusen, retinal vein occlusion',
        'fingerprint': 'fingerprint core, fingerprint delta, fingerprint whorl, fingerprint loop, fingerprint arch',
        'mias': 'breast mass, benign mass, malignant mass, calcification, architectural distortion, breast asymmetry, spiculated mass',
        'oral_cancer': 'oral lesion, ulcer, tumor region, malignant tissue, squamous cell carcinoma, atypical cells',
        'aptos': 'hemorrhage, hard exudate, soft exudate, microaneurysm, neovascularization, venous beading, diabetic retinopathy lesion'
    }
    
    # Determine prompt based on dataset name
    dataset_name = dataset_path.name.lower()
    prompt = "salient object" # default
    for key, val in PROMPT_MAP.items():
        if key in dataset_name:
            prompt = val
            break
            
    print(f"Using prompt: '{prompt}' for dataset {dataset_name}")
    
    # Iterate
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp'}
    
    # Walker
    files = []
    for root, _, filenames in os.walk(dataset_path):
        for f in filenames:
            if Path(f).suffix.lower() in image_extensions:
                files.append(Path(root) / f)
                
    print(f"Found {len(files)} images.")
    
    count = 0
    for img_path in tqdm(files):
        try:
            # Check if JSON already exists
            json_path = img_path.with_suffix('.json')
            if json_path.exists():
                # Optional: Skip if exists
                # continue
                pass

            # Detect
            # Note: Doing this one-by-one. Batching is harder with various image sizes.
            boxes, _, labels = agent.detect(img_path, prompt)
            
            # Save format
            # {
            #   "image_path": "...",
            #   "detections": [
            #       {"bbox": [x1, y1, x2, y2], "label": "...", "confidence": 1.0}
            #   ]
            # }
            
            detections_list = []
            for i in range(len(boxes)):
                box = boxes[i].tolist() # [x1,y1,x2,y2]
                label = labels[i] if i < len(labels) else prompt
                
                detections_list.append({
                    "bbox": box,
                    "label": label,
                    "class_name": "center_point" if "core" in label else "delta_point" if "delta" in label else "lesion", # Simple logic mapping? Or just keep raw.
                    # For compatibility with heatmap_utils logic (center/delta), we might need logic.
                    # But for generic datasets, let's keep it generic.
                    "confidence": 1.0
                })
            
            output_data = {
                "image_path": str(img_path),
                "prompt": prompt,
                "detections": detections_list
            }
            
            with open(json_path, 'w') as f:
                json.dump(output_data, f, indent=2)
                
            count += 1
            
        except Exception as e:
            print(f"Error processing {img_path}:")
            traceback.print_exc()
            
            
    print(f"Processed {count} images. Saved JSONs alongside images.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help="Path to dataset root")
    parser.add_argument('--model', type=str, default='microsoft/Florence-2-large-ft')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    generate_detections(args.dataset, model_id=args.model, device=args.device)
