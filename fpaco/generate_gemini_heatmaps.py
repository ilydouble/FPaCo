import os
import sys
import argparse
import json
import traceback
import base64
import time
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from openai import OpenAI

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def generate_detections(dataset_dir, output_dir=None, model_id='gemini-3-flash-preview', api_key=None, base_url=None, overwrite=False, limit=None):
    """
    Iterate over dataset, detect using Gemini (via Yunwu), and save JSONs.
    """
    
    # Initialize Client
    if not api_key:
        api_key = os.environ.get("YUNWU_API_KEY")
    if not base_url:
        base_url = "https://api.yunwu.ai/v1" # Hypothetical default, usually passed or env

    if not api_key:
        print("Error: YUNWU_API_KEY not found in environment or arguments.")
        return

    try:
        client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
    except Exception as e:
        print(f"CRITICAL ERROR initializing OpenAI/Yunwu Client: {e}")
        return

    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        print(f"Error: Dataset path does not exist: {dataset_path}")
        return

    #Prompt Map (Optimized)
    # PROMPT_MAP = {
    #     'octa': 'The central black void or circular gap where there are no white lines. Focus strictly on the dark hole in the geometric center of the image.',
    #     'finger': 'The fingerprint Core point (top of the innermost loop) and the Delta point (triangular ridge intersection).',
    #     'mias': 'The single brightest and densest white patch that stands out against the surrounding gray tissue. Ignore the background muscle.',
    #     'aptos': 'Small red spots or scattered red dots (hemorrhages) on the retina. Do NOT detect the large bright circular optic disc. Focus only on the small red anomalies.',
    #     'oral': 'Pathological regions showing keratin pearls or invading nests of squamous epithelial cells.'
    # }
    PROMPT_MAP = {
    # OCTA: 增加“血管网包围”的上下文描述，防止它把边缘的黑色背景当成 FAZ
    'octa': 'A black gap or circular area in the center of the image without white lines. Please focus your attention on the black holes not at the edges of the image.',

    # Finger: 明确区分 Core 和 Delta 的形态差异，引导它进行分类
    'finger': 'The fingerprint Core point (top of the innermost loop) and the Delta point (triangular ridge intersection).',

    # MIAS: 这是最难的。增加了对“背景噪声”和“肌肉”的详细视觉区分
    'mias': (
        "Target: Breast Mass or Microcalcifications. "
        "Distinctive Features: "
        "1. Mass: A high-density (bright white) region that disrupts the natural tissue texture. Malignant masses often have 'spiculated' (star-like) or fuzzy edges, distinct from the smooth edges of benign cysts. "
        "2. Calcifications: Clusters of tiny, grain-like, high-intensity white specks. "
        "Constraint: Distinguish these from the large triangular Pectoral Muscle at the top corner."
    ),

    # APTOS: 强调“点状”特征，并给出视盘的详细描述以供排除
    'aptos': (
        "Target: Diabetic Retinopathy Lesions. "
        "Distinctive Features: "
        "1. Red Lesions: Look for 'Microaneurysms' (tiny, sharp red dots) or 'Hemorrhages' (larger, irregular red blotches). "
        "2. Bright Lesions: Look for 'Hard Exudates' (sharp, bright yellow deposits) or 'Cotton Wool Spots' (fuzzy white patches). "
        "Negative Constraint: Ignore the Optic Disc (large bright vertical oval) and main vessel branches."
    ),

    # Oral: 将病理术语翻译成几何形状（同心圆、巢状）
    'oral': (
        "Target: Squamous Cell Carcinoma (OSCC). "
        "Distinctive Features: "
        "1. Keratin Pearls: Distinctive concentric, whorled structures that appear bright pink/red (eosinophilic), resembling sliced onions. "
        "2. Tumor Nests: Irregular islands of atypical epithelial cells invading the connective tissue stroma. "
        "3. Atypia: Enlarged, hyperchromatic (dark purple) nuclei compared to normal cells."
    )
}
    
    # Determine prompt based on dataset name
    dataset_name = dataset_path.name.lower()
    domain_prompt = "salient object" # default
    current_dataset_key = "unknown"
    for key, val in PROMPT_MAP.items():
        if key in dataset_name:
            domain_prompt = val
            current_dataset_key = key
            break
            
    print(f"Using domain prompt: '{domain_prompt}' for dataset {dataset_name} (Key: {current_dataset_key})")
    
    # System prompt definition moved inside loop for dynamic usage

    # Iterate
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp'}
    
    # Walker
    files = []
    for root, _, filenames in os.walk(dataset_path):
        for f in filenames:
            if Path(f).suffix.lower() in image_extensions:
                files.append(Path(root) / f)
                
    print(f"Found {len(files)} images.")
    
    if limit:
        files = files[:limit]
        print(f"Limiting to first {limit} images.")

    count = 0
    skipped = 0
    errors = 0
    
    for img_path in tqdm(files, desc="Processing Images"):
        try:
            # Determine Output JSON Path
            if output_dir:
                rel_path = img_path.relative_to(dataset_path)
                json_path = Path(output_dir) / rel_path.with_suffix('.json')
                json_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                json_path = img_path.with_suffix('.json')

            # Check if exists
            if json_path.exists() and not overwrite:
                skipped += 1
                continue
            
            # Load Image for Dimensions (needed if model returns normalized, but we asked for absolute)
            # We'll just read it for base64 anyway
            try:
                base64_image = encode_image(img_path)
            except Exception as e:
                print(f"Error encoding image {img_path}: {e}")
                errors += 1
                continue
            
            # --- DYNAMIC PROMPT LOGIC ---
            current_prompt_text = domain_prompt
            should_run_inference = True
            
            if current_dataset_key == 'mias':
                class_dir = img_path.parent.name
                
                # Special handling for MIAS Classes
                if class_dir == 'class_1':
                    # NORMAL CLASS: Force empty detection to avoid hallucinations
                    should_run_inference = False
                    detections = []
                elif class_dir == 'class_0':
                    current_prompt_text = "Target: Circumscribed Mass. Look for a well-defined, round or oval high-density region. Ignore normal glands."
                elif class_dir == 'class_2':
                    current_prompt_text = "Target: Breast Abnormalities. Look for significant texture disruptions or unclassified masses."
                elif class_dir == 'class_3':
                    current_prompt_text = "Target: Asymmetry. Look for focal density that stands out from the surrounding pattern."
                elif class_dir == 'class_4':
                    current_prompt_text = "Target: Architectural Distortion. Look for spiculations or radiating lines without a central mass, or focal retraction."
                elif class_dir == 'class_5':
                    current_prompt_text = "Target: Spiculated Mass. Look for a central density with radiating star-like lines (malignant feature)."
                elif class_dir == 'class_6':
                    current_prompt_text = "Target: Microcalcifications. Look for clusters of tiny, bright white specks (grains of salt)."

            elif current_dataset_key == 'octa':
                class_dir = img_path.parent.name
                
                # Special handling for OCTA Classes
                # 0:AMD, 1:CNV, 2:CSC, 3:DR, 4:NORMAL, 5:OTHERS, 6:RVO
                if class_dir == 'class_4':
                    # NORMAL CLASS: Force empty detection
                    should_run_inference = False
                    detections = []
                elif class_dir == 'class_0': # AMD
                    current_prompt_text = "Target: AMD. Look for Choroidal Neovascularization (CNV) networks or Drusen. Focus on abnormal vascular loops or dark gaps in the foveal avascular zone."
                elif class_dir == 'class_1': # CNV
                    current_prompt_text = "Target: CNV. Look for a hyper-reflective vascular membrane or tangled vessel network disrupting the normal capillary layout."
                elif class_dir == 'class_2': # CSC
                    current_prompt_text = "Target: CSC. Look for subretinal fluid accumulation (dark voids) or pigment epithelial detachment."
                elif class_dir == 'class_3': # DR
                    current_prompt_text = "Target: Diabetic Retinopathy. Look for Microaneurysms (tiny dilation), capillary non-perfusion areas (dark gaps), or enlarged foveal avascular zone."
                elif class_dir == 'class_5': # OTHERS
                    current_prompt_text = "Target: Retinal Anomalies. Look for any vascular disruption, non-perfusion, or abnormal vessel growth not typical of healthy retina."
                elif class_dir == 'class_6': # RVO
                    current_prompt_text = "Target: RVO. Look for venous dilation, tortuosity (twisted vessels), and capillary dropout sectors (dark wedges)."

            if not should_run_inference:
                 # Skip API call for Normal cases
                 detections = []
            else:
                # Construct System Prompt dynamically
                system_prompt = f"""
    You are an expert medical image analyst. Your task is to detect specific features in the provided image.
    
    Features to detect: {current_prompt_text}
    
    Return the result ONLY as a JSON object with the following structure:
    {{
      "detections": [
        {{
          "bbox": [x1, y1, x2, y2], 
          "label": "string",
          "confidence": float
        }}
      ]
    }}
    
    - bbox should be [x1, y1, x2, y2] in ABSOLUTE PIXEL COORDINATES (0 to ImageWidth/Height).
    - If no features are found, return an empty list for "detections".
    - Do not encompass the entire image. Be specific.
    - IMPORTANT: Output RAW JSON only. Do NOT use markdown code blocks (e.g., ```json).
    """
    
                # Call API
                response = client.chat.completions.create(
                    model=model_id,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": system_prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}"
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=8192*2,
                    response_format={"type": "json_object"}
                )
    
                content = response.choices[0].message.content
            
            if should_run_inference:
                try:
                    # Robust JSON Extraction
                    import re
                    import ast
                    
                    json_str = content
                    # Try to find JSON object structure
                    match = re.search(r'(\{.*\})', content, re.DOTALL)
                    if match:
                        json_str = match.group(1)
                    
                    # Attempt 1: Standard JSON
                    try:
                        result = json.loads(json_str)
                    except:
                        # Attempt 2: Python Literal Eval (Handles single quotes, trailing commas)
                        try:
                            result = ast.literal_eval(json_str)
                        except:
                            # Attempt 3: Relaxed Regex for just the list
                            # sometimes model returns just the list [ ... ]
                            match_list = re.search(r'(\[.*\])', content, re.DOTALL)
                            if match_list:
                                try:
                                    result = {"detections": ast.literal_eval(match_list.group(1))}
                                except:
                                    raise ValueError("Could not parse JSON or List")
                            else:
                                 raise ValueError("No JSON object or list found")

                    if isinstance(result, list):
                         detections = result
                    else:
                         detections = result.get("detections", [])

                except Exception as e:
                    print(f"[WARN] Failed to parse response for {img_path.name}: {e}")
                    print(f"RAW RESP: {content}") 
                    detections = []

            # Post-process Detections (Add class_name logic for compatibility)
            processed_detections = []
            for det in detections:
                label = det.get("label", "unknown")
                class_name = "lesion"
                if current_dataset_key == "fingerprint":
                    if "core" in label.lower():
                        class_name = "center_point"
                    elif "delta" in label.lower():
                        class_name = "delta_point"
                    else:
                        class_name = "other"
                else:
                    class_name = label # simple pass-through

                processed_detections.append({
                    "bbox": det.get("bbox", [0,0,0,0]),
                    "label": label,
                    "class_name": class_name,
                    "confidence": det.get("confidence", 1.0)
                })

            output_data = {
                "image_path": str(img_path),
                "prompt": domain_prompt,
                "detections": processed_detections,
                "model": model_id
            }
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2)
                
            count += 1
            # Rate limit protection (simple)
            time.sleep(0.5) 
            
        except Exception as e:
            errors += 1
            print(f"\n[ERROR] Processing {img_path}: {e}")
            # traceback.print_exc()
            
    print(f"\nProcessing Complete.")
    print(f"  Processed: {count}")
    print(f"  Skipped: {skipped}")
    print(f"  Errors: {errors}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help="Path to dataset root")
    parser.add_argument('--output-dir', type=str, default=None, help="Optional separate output directory for JSONs")
    parser.add_argument('--model', type=str, default='gemini-3-flash-preview', help="Gemini Model ID")
    parser.add_argument('--api-key', type=str, default=None, help="Yunwu/OpenAI API Key (or set YUNWU_API_KEY env)")
    parser.add_argument('--base-url', type=str, default="https://api.yunwu.ai/v1", help="Base URL for Yunwu API")
    parser.add_argument('--overwrite', action='store_true', help="Overwrite existing JSON files")
    parser.add_argument('--limit', type=int, default=None, help="Limit number of images for testing")
    
    args = parser.parse_args()
    
    generate_detections(
        args.dataset, 
        output_dir=args.output_dir, 
        model_id=args.model, 
        api_key=args.api_key,
        base_url=args.base_url,
        overwrite=args.overwrite,
        limit=args.limit
    )
