import os
import argparse
import json
import torch
from PIL import Image
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

try:
    import open_clip
except ImportError:
    print("Error: open_clip not found. Please install it using `pip install open_clip_torch`")
    exit(1)

# Dataset Labels Mapping
# Note: For MIAS and OCTA, these are best-guess standard labels. 
# User verify if these match the numeric class IDs in the dataset generation logic.

DATASET_CONFIGS = {
    'oral_cancer': {
        'classes': ['Normal', 'Oral Cancer']
    },
    'aptos': {
        'classes': ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
    },
    'finger': {
        'classes': [
            'Stone', 'Gold', 'Dream', 'Electricity', 'Wind', 'Electricity with Wind', 
            'Drill', 'Light', 'Water', 'Fire', 'Wood', 'Earth', 'Ground', 
            'Mountain', 'Rock', 'Fire Light', 'Fire Wood', 'Fire Earth', 'Fire Drill'
        ]
    },
    'mias': {
        # Assuming standard MIAS 7-class: CALC, CIRC, SPIC, MISC, ARCH, ASYM, NORM
        # Ordering is unknown, assuming alphabetical or standard numeric.
        # This is high risk of being wrong order.
        # Let's use a generic 7-class prompt and hope for alignment or just test.
        'classes': ['CALC', 'CIRC', 'SPIC', 'MISC', 'ARCH', 'ASYM', 'NORM']
    },
    'octa': {
        # Assuming Common OCTA 7 classes: AMD, CNV, DR, ERM, Normal, OHT, RVO
        'classes': ['AMD', 'CNV', 'DR', 'ERM', 'Normal', 'OHT', 'RVO']
    }
}

def load_prompts_from_json(json_path, dataset_name):
    """Load prompts from the unified JSON file."""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Prompts file not found: {json_path}")
        
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    if dataset_name not in data:
        raise ValueError(f"Dataset {dataset_name} not found in {json_path}")
        
    class_prompts_map = data[dataset_name]
    # Keys are strings "0", "1", etc. Sort them numerically to ensure order
    sorted_keys = sorted(class_prompts_map.keys(), key=lambda x: int(x))
    
    prompts_list = []
    for k in sorted_keys:
        # Return the full list of prompts for ensemble
        p_list = class_prompts_map[k]
        if isinstance(p_list, list):
            prompts_list.append(p_list) 
        else:
             prompts_list.append([str(p_list)])
             
    return prompts_list

def load_biomedclip(device, model_name, cache_dir=None):
    print(f"Loading BioMedCLIP model: {model_name}...")
    # open_clip.create_model_and_transforms automatically handles downloading from HF Hub if model_name starts with 'hf-hub:'
    # The cache_dir arg allows specifying where to save/load the model.
    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        model_name, 
        cache_dir=cache_dir
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    model.to(device)
    model.eval()
    return model, preprocess_val, tokenizer

def run_dataset(model, preprocess, tokenizer, dataset_name, dataset_path, device, prompts_file):
    print(f"\nProcessing {dataset_name} at {dataset_path}...")
    
    config = DATASET_CONFIGS.get(dataset_name)
    if not config:
        print(f"Unknown dataset config for {dataset_name}")
        return
    
    # Load prompts from JSON
    try:
        class_prompts = load_prompts_from_json(prompts_file, dataset_name)
        print(f"Loaded prompts for {len(class_prompts)} classes from {prompts_file}")
    except Exception as e:
        print(f"Error loading prompts: {e}")
        return

    # Encode prompts (Ensemble)
    print("Encoding prompts (Ensemble)...")
    class_embeddings = []
    with torch.no_grad():
        for prompts in class_prompts:
            text_tokens = tokenizer(prompts).to(device)
            embeddings = model.encode_text(text_tokens)
            embeddings /= embeddings.norm(dim=-1, keepdim=True)
            # Mean pooling for ensemble
            mean_embedding = embeddings.mean(dim=0)
            mean_embedding /= mean_embedding.norm()
            class_embeddings.append(mean_embedding)
            
    text_features = torch.stack(class_embeddings, dim=0).to(device) # [C, D]

    # Load images
    image_paths = []
    labels = []
    
    # Walk through test set (or val if test empty)
    split_dir = os.path.join(dataset_path, 'test')
    if not os.path.exists(split_dir) or len(os.listdir(split_dir)) == 0:
        split_dir = os.path.join(dataset_path, 'val')
    
    class_dirs = sorted([d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))])
    class_to_idx = {d: i for i, d in enumerate(class_dirs)}
    
    print(f"Found {len(class_dirs)} classes in {split_dir}")
    if len(class_dirs) != len(config['classes']):
        print(f"Warning: Number of classes in dataset ({len(class_dirs)}) does not match config ({len(config['classes'])})!")
        # Proceeding might allow mismatch if mapping is by index
    
    for cls_name in class_dirs:
        cls_dir = os.path.join(split_dir, cls_name)
        for img_name in os.listdir(cls_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(cls_dir, img_name))
                labels.append(class_names_to_idx(cls_name)) # Helper to extract class ID from folder name 'class_0'

    print(f"Evaluating on {len(image_paths)} images...")
    
    preds = []
    probs = []
    true_labels = []
    
    batch_size = 32
    for i in tqdm(range(0, len(image_paths), batch_size)):
        batch_paths = image_paths[i:i+batch_size]
        batch_labels = labels[i:i+batch_size]
        
        images = []
        for p in batch_paths:
            img = Image.open(p).convert('RGB')
            img = preprocess(img)
            images.append(img)
        
        image_input = torch.stack(images).to(device)
        
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
            # Similarity
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            
            p, predicted = similarity.max(dim=1)
            preds.extend(predicted.cpu().numpy())
            probs.extend(similarity.cpu().numpy())
            true_labels.extend(batch_labels)

    # Metrics
    acc = accuracy_score(true_labels, preds)
    macro_f1 = f1_score(true_labels, preds, average='macro')
    
    try:
        if len(config['classes']) == 2:
            auc = roc_auc_score(true_labels, np.array(probs)[:, 1])
        else:
            auc = roc_auc_score(true_labels, probs, multi_class='ovr', average='macro')
    except:
        auc = 0.0

    print(f"Results for {dataset_name}:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"AUC: {auc:.4f}")
    
    return {'dataset': dataset_name, 'acc': acc, 'f1': macro_f1, 'auc': auc}

def class_names_to_idx(folder_name):
    # e.g., 'class_0' -> 0
    try:
        return int(folder_name.split('_')[-1])
    except:
        return 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', default='../', help='Root workspace path')
    parser.add_argument('--dataset', default='all', choices=['all', 'oral_cancer', 'aptos', 'finger', 'mias', 'octa'], help='Specific dataset to run')
    parser.add_argument('--model', default='hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224', help='BioMedCLIP model name (e.g. hf-hub:ZiyueWang/med-clip for 448 resolution)')
    parser.add_argument('--cache-dir', default=None, help='Directory to cache the downloaded model')
    parser.add_argument('--prompts-file', default='prompts/unified_prompts.json', help='Path to unified prompts json file relative to workspace')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, preprocess, tokenizer = load_biomedclip(device, args.model, args.cache_dir)
    
    datasets_map = {
        'oral_cancer': 'oral_cancer_classification_dataset',
        'aptos': 'aptos_classification_dataset',
        'finger': 'fingerA',
        'mias': 'mias_classification_dataset',
        'octa': 'octa_classification_dataset'
    }
    
    results = []
    
    # Filter datasets if specific one requested
    if args.dataset != 'all':
        if args.dataset in datasets_map:
            target_datasets = {args.dataset: datasets_map[args.dataset]}
        else:
            print(f"Dataset {args.dataset} not supported.")
            return
    else:
        target_datasets = datasets_map
    
    for name, folder in target_datasets.items():
        dataset_path = os.path.join(args.workspace, 'datasets', folder)
        if not os.path.exists(dataset_path):
            print(f"Skipping {name}: path {dataset_path} not found")
            continue
            
        # Resolve prompts file path
        # Assume args.prompts_file is relative to workspace or absolute
        prompts_path = os.path.join(args.workspace, args.prompts_file)
        if not os.path.exists(prompts_path):
             # Try relative to script
             prompts_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.prompts_file)
        
        # Fixed path fallback for robustness if workspace arg is just '../'
        if not os.path.exists(prompts_path):
             prompts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../prompts/unified_prompts.json'))

        res = run_dataset(model, preprocess, tokenizer, name, dataset_path, device, prompts_path)
        if res:
            results.append(res)
            
    # Determine output directory (same as script directory)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    if args.dataset != 'all':
        output_filename = f'results_{args.dataset}.json'
    else:
        output_filename = 'results_all.json'
        
    output_file = os.path.join(script_dir, output_filename)
        
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nResults saved to {output_file}")
    
    print("\nFinal Summary:")
    print(f"{'Dataset':<15} {'Acc':<10} {'F1':<10} {'AUC':<10}")
    print("-" * 45)
    for r in results:
        print(f"{r['dataset']:<15} {r['acc']:<10.4f} {r['f1']:<10.4f} {r['auc']:<10.4f}")

if __name__ == "__main__":
    main()
