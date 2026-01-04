import os
import sys
import argparse
import json
import torch
import traceback
import importlib.metadata
from pathlib import Path
from PIL import Image
from tqdm import tqdm
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

try:
    from florence2_agent import Florence2Agent
except ImportError:
    # Handle the case where the agent might not be importable due to missing deps
    print("Error: Could not import Florence2Agent. Check dependencies.")
    traceback.print_exc()
    sys.exit(1)

def check_dependencies():
    """Checks for known incompatible library versions."""
    try:
        urllib3_version = importlib.metadata.version('urllib3')
        # Check if version starts with '2.'
        if urllib3_version.startswith('2.'):
            print(f"Warning: urllib3 version {urllib3_version} detected.")
    except importlib.metadata.PackageNotFoundError:
        pass

def generate_detections(dataset_dir, output_dir=None, model_id='microsoft/Florence-2-large-ft', device='cuda', overwrite=False):
    """
    Iterate over dataset, detect using Florence-2, and save JSONs.
    """
    check_dependencies()
    
    # Initialize Agent
    try:
        agent = Florence2Agent(model_id=model_id, device=device)
    except Exception as e:
        print(f"CRITICAL ERROR initializing Florence2Agent: {e}")
        traceback.print_exc()
        return

    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        print(f"Error: Dataset path does not exist: {dataset_path}")
        return

    # Prompt Map
    PROMPT_MAP = {
        'octa': 'locate dark void zones (non-perfusion), tangled bright vessels (CNV), and distorted vessel loops',
        'finger':'fingerprint core, fingerprint delta, fingerprint whorl, fingerprint loop, fingerprint arch',
        'mias': 'mass, benign mass, malignant mass, calcification, architectural distortion, spiculated mass, white spot, density',
        'aptos': 'locate small red dots of microaneurysm, red hemorrhage, and yellow hard exudates',
        'oral': 'oral squamous cell carcinoma, cancer cells, atypical cells, keratin pearls'
    }
    
    # Determine prompt based on dataset name
    dataset_name = dataset_path.name.lower()
    prompt = "salient object" # default
    current_dataset_key = "unknown"
    for key, val in PROMPT_MAP.items():
        if key in dataset_name:
            prompt = val
            current_dataset_key = key
            break
            
    print(f"Using prompt: '{prompt}' for dataset {dataset_name} (Key: {current_dataset_key})")
    
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
    skipped = 0
    errors = 0
    skipped_huge = 0
    
    for img_path in tqdm(files, desc="Processing Images"):
        try:
            # Determine Output JSON Path
            if output_dir:
                # Maintain relative structure if output_dir is specified
                rel_path = img_path.relative_to(dataset_path)
                json_path = Path(output_dir) / rel_path.with_suffix('.json')
                json_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                # Save alongside image
                json_path = img_path.with_suffix('.json')

            # Check if exists
            if json_path.exists() and not overwrite:
                skipped += 1
                continue

            # Detect
            try:
                boxes, scores, labels = agent.detect(img_path, prompt)
            except Exception as e_det:
                 print(f"[DEBUG] agent.detect FAILED for {img_path.name}: {e_det}")
                 traceback.print_exc()
                 raise e_det
            
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
                
                # Expand invalid/degenerate boxes slightly
                x1, y1, x2, y2 = box
                if x2 == x1: x1 -= 0.1; x2 += 0.1
                if y2 == y1: y1 -= 0.1; y2 += 0.1
                box = [x1, y1, x2, y2]
                
                label = labels[i] if i < len(labels) else prompt
                confidence = float(scores[i]) if i < len(scores) else 1.0
                
                # Intelligent class_name assignment for validation_visuals compatibility
                class_name = "lesion" # Default
                if current_dataset_key == "fingerprint":
                    if "core" in label:
                        class_name = "center_point"
                    elif "delta" in label:
                        class_name = "delta_point"
                    else:
                        class_name = "other" # Or specific fingerprint type
                else:
                    class_name = label

                # Filter: Ignore huge boxes (likely entire organ/background)
                # MIAS specific: > 30% area is almost certainly noise/organ segmentation
                box_w = x2 - x1
                box_h = y2 - y1
                img_w, img_h = 1024, 1024 # Approximate for MIAS if not known, or calculate relative
                # Actually x1,y1,x2,y2 are normalized (0-1)? 
                # Wait, Florence-2 output from agent.detect is usually absolute pixel coords?
                # Let's check florence2_agent.py. 
                # "parsed_answer = self.processor.post_process_generation(..., image_size=(width, height))"
                # Yes, it returns absolute coordinates.
                # However, we don't strictly know image size here unless we opened it or passed it.
                # But boxes are relative to image size.
                # Let's check if we can get image size. Using PIL.Image.open(img_path) overhead is small.
                
                with Image.open(img_path) as tmp_img:
                    im_w, im_h = tmp_img.size
                
                # Check normalized area
                norm_area = (box_w * box_h) / (im_w * im_h)
                
                if norm_area > 0.6:
                    # print(f"[DEBUG] Ignored huge box: {norm_area:.2f} for {img_path.name}")
                    continue

                detections_list.append({
                    "bbox": box,
                    "label": label,
                    "class_name": class_name,
                    "confidence": confidence
                })
            
            output_data = {
                "image_path": str(img_path),
                "prompt": prompt,
                "detections": detections_list
            }
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2)
                
            count += 1
            
        except BaseException as e:
            errors += 1
            print(f"\n[CRITICAL ERROR] Error processing {img_path}: {e}")
            traceback.print_exc() 
            
    print(f"\nProcessing Complete.")
    print(f"  Processed: {count}")
    print(f"  Skipped (Existing): {skipped}")
    print(f"  Skipped (Huge > 0.3): {skipped_huge}")
    print(f"  Errors:    {errors}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help="Path to dataset root")
    parser.add_argument('--output-dir', type=str, default=None, help="Optional separate output directory for JSONs")
    parser.add_argument('--model', type=str, default='microsoft/Florence-2-large-ft')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--overwrite', action='store_true', help="Overwrite existing JSON files")
    
    args = parser.parse_args()
    
    generate_detections(
        args.dataset, 
        output_dir=args.output_dir, 
        model_id=args.model, 
        device=args.device,
        overwrite=args.overwrite
    )
