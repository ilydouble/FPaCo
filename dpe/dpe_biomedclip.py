import os
import argparse
import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import open_clip
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from collections import defaultdict
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# =========================================================
# Utils & DPE Classes
# =========================================================

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--data-root', type=str, default='../datasets')
    parser.add_argument('--model', type=str, default='hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    parser.add_argument('--output-dir', type=str, default='results')
    parser.add_argument('--shots', type=int, default=16)
    parser.add_argument('--load-cache', action='store_true') # For future use if caching features
    
    # DPE Hyperparams
    parser.add_argument('--lr-text', type=float, default=0.001)
    parser.add_argument('--lr-image', type=float, default=0.001)
    parser.add_argument('--align-loss-weight', type=float, default=10.0)
    
    parser.add_argument('--pos-alpha', type=float, default=1.0)
    parser.add_argument('--pos-beta', type=float, default=0.5)
    parser.add_argument('--shot-capacity', type=int, default=10) # Max items per class in Positive Cache
    
    parser.add_argument('--seed', type=int, default=1)
    
    parser.add_argument('--prompts-file', type=str, default='gpt3_prompts/prompts_simple.json', help='Path to prompts json file')
    parser.add_argument('--entropy-threshold', type=float, default=None, help='Entropy threshold for cache update')
    parser.add_argument('--use-train-cache', action='store_true', help='Use training data to initialize positive cache')
    
    args = parser.parse_args()
    return args

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

class TextResidue(nn.Module):
    def __init__(self, clip_weights):
        super(TextResidue, self).__init__()
        self.feat_dim, self.cate_num = clip_weights.shape
        self.residual = nn.Parameter(torch.zeros([self.feat_dim, self.cate_num]).to(clip_weights.device), requires_grad=True)
        
    def forward(self, x):
        new_clip_weights = x.clone() + self.residual
        new_clip_weights = F.normalize(new_clip_weights, dim=0)
        return new_clip_weights

class PositiveCacheResidue(nn.Module):
    def __init__(self, pos_cache_keys):
        super(PositiveCacheResidue, self).__init__()
        self.feat_dim, self.cache_size = pos_cache_keys.shape
        self.residual = nn.Parameter(torch.zeros([self.feat_dim, self.cache_size]).to(pos_cache_keys.device), requires_grad=True)
        
    def forward(self, x):
        new_pos_cache_keys = x.clone() + self.residual
        new_pos_cache_keys = F.normalize(new_pos_cache_keys, dim=0)
        return new_pos_cache_keys

def info_nce_loss(features, prototypes, temperature=0.07):
    # features: [N, D], prototypes: [D, K] or [N, D] depending on usage
    # Simple implementation: A matches B if they separate same classes? 
    # DPE implementation: InfoNCELoss(new_pos_cache_keys.T, new_clip_weights[:, all_classes].T)
    # A: [Cache_Size, Dim] (keys), B: [Cache_Size, Dim] (text_prototypes corresponding to keys)
    
    A = F.normalize(features, dim=1)
    B = F.normalize(prototypes, dim=1)
    
    # Cosine similarity
    logits = torch.mm(A, B.t()) / temperature
    labels = torch.arange(A.shape[0]).to(A.device)
    
    loss = F.cross_entropy(logits, labels)
    return loss

def entropy_loss(logits):
    p = F.softmax(logits, dim=1)
    log_p = F.log_softmax(logits, dim=1)
    return -(p * log_p).sum(dim=1).mean()

def compute_cache_logits(image_features, cache_keys, cache_values, alpha, beta):
    # image_features: [1, D]
    # cache_keys: [D, N_cache]
    # cache_values: [N_cache, N_classes]
    
    affinity = image_features @ cache_keys
    cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
    return alpha * cache_logits

def update_cache(cache, pred_label, feature_item, shot_capacity, entropy_threshold=None):
    """
    feature_item: [feature_vector, entropy_score]
    cache: dict {label: list of [feature, entropy]}
    """
    # Filter by entropy if threshold provided
    if entropy_threshold is not None and feature_item[1] > entropy_threshold:
        return

    with torch.no_grad():
        if pred_label in cache:
            if len(cache[pred_label]) < shot_capacity:
                cache[pred_label].append(feature_item)
            else:
                # If full, replace if new entropy is lower (better confidence)
                # Sort by entropy (asc) -> last item has highest entropy (worst)
                cache[pred_label].sort(key=lambda x: x[1])
                if feature_item[1] < cache[pred_label][-1][1]:
                    cache[pred_label][-1] = feature_item
            
            # Keep sorted
            cache[pred_label].sort(key=lambda x: x[1])
        else:
            cache[pred_label] = [feature_item]

def build_cache_tensors(cache, feat_dim, num_classes, device):
    cache_keys = []
    cache_values_indices = []
    
    # Collect all items
    for label in sorted(cache.keys()):
        for item in cache[label]:
            cache_keys.append(item[0]) # feature
            cache_values_indices.append(label)
    
    if not cache_keys:
        return None, None, None
        
    cache_keys = torch.stack(cache_keys, dim=0).t() # [D, N_cache]
    
    # cache_values: one-hot
    cache_values_indices = torch.tensor(cache_values_indices).to(device)
    cache_values = F.one_hot(cache_values_indices, num_classes=num_classes).float() # [N_cache, C]
    
    return cache_keys, cache_values, cache_values_indices

# =========================================================
# Main Logic
# =========================================================

def load_prompts(json_path, dataset_name):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    if dataset_name not in data:
        raise ValueError(f"Dataset {dataset_name} not found in prompts file.")
    
    # Convert "0": [...], "1": [...] to list of lists ordered by int key
    class_prompts_map = data[dataset_name]
    sorted_keys = sorted(class_prompts_map.keys(), key=lambda x: int(x))
    
    prompts_list = []
    for k in sorted_keys:
        prompts_list.append(class_prompts_map[k])
        
    return prompts_list

def clip_classifier(class_prompts, model, tokenizer, device):
    """
    class_prompts: List of List of strings. dim 0 is class, dim 1 is sentences.
    """
    weights = []
    with torch.no_grad():
        for prompts in class_prompts:
            # Tokenize
            texts = tokenizer(prompts).to(device) # [N_sentences, 77]
            class_embeddings = model.encode_text(texts) # [N_sentences, D]
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            
            # Mean embedding for the class
            mean_embedding = class_embeddings.mean(dim=0)
            mean_embedding /= mean_embedding.norm()
            weights.append(mean_embedding)
            
    weights = torch.stack(weights, dim=1).to(device) # [D, C]
    return weights

def get_clip_logits(images, model, clip_weights):
    with torch.no_grad():
        image_features = model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        
    logits = 100. * image_features @ clip_weights
    return image_features, logits

def run_dpe(args, model, test_loader, clip_weights_init, device, train_loader=None):
    # DPE State
    pos_cache = {}
    
    # Store initial weights globally
    clip_weights = clip_weights_init.clone()
    
    predictions = []
    targets = []
    probabilities = []
    
    num_classes = clip_weights.shape[1]
    
    # Initialize Cache with Training Data if provided
    if train_loader is not None:
        print(f"Initializing cache with training data (Capacity: {args.shot_capacity})...")
        for images, target in tqdm(train_loader, desc="Init Cache"):
            images = images.to(device)
            target = target.to(device)
            
            with torch.no_grad():
                img_feats = model.encode_image(images)
                img_feats /= img_feats.norm(dim=-1, keepdim=True)
                
                # Assume GT label is correct, and entropy is 0 (or -inf to prioritize)
                # We use a very small entropy so test data won't easily replace it
                # unless we want test data to refine it?
                # Usually support set is trusted.
                entropy_val = -1e9 
                
                for idx in range(images.size(0)):
                    p = target[idx].item() # Use GT label
                    f = img_feats[idx].detach()
                    # Add to cache
                    update_cache(pos_cache, p, [f, entropy_val], args.shot_capacity, args.entropy_threshold)
                    
    print(f"Starting TTA on {len(test_loader)} batches...")
    
    for i, (images, target) in enumerate(tqdm(test_loader)):
        images = images.to(device)
        target = target.to(device)
        
        # 1. Initial Forward Pass (Zero-shot / current weights)
        # We need this to get initial prediction and entropy for cache update
        # Use current global weights for initial check
        
        # For DPE, we create a FRESH TextResidue for each batch/image? 
        # Or persistent? Original main_dpe.py creates it inside loop:
        # clip_weights_local = clip_weights_global.clone().detach()
        # text_residue = TextResidue(clip_weights_local)
        # It's episodic adaptation per batch/image for test time?
        # Yes, "Test-time adaptation" usually implies adapting to the current input.
        
        clip_weights_local = clip_weights.clone().detach()
        text_residue = TextResidue(clip_weights_local)
        
        with torch.no_grad():
            img_feats = model.encode_image(images)
            img_feats /= img_feats.norm(dim=-1, keepdim=True)
            
            # Initial logits
            logits0 = 100. * img_feats @ clip_weights_local
            probs0 = F.softmax(logits0, dim=1)
            entropy0 = -(probs0 * torch.log(probs0 + 1e-6)).sum(dim=1)
            
            pred0 = logits0.argmax(dim=1)
        
        # 2. Update Positive Cache
        # For each image in batch, add to cache
        for idx in range(images.size(0)):
            p = pred0[idx].item()
            e = entropy0[idx].item()
            f = img_feats[idx].detach()
            update_cache(pos_cache, p, [f, e], args.shot_capacity, args.entropy_threshold)
            
        # 3. Adaptation Steps
        # Build Cache Tensors
        pos_cache_keys, pos_cache_values, pos_cache_labels_idx = build_cache_tensors(pos_cache, img_feats.shape[1], num_classes, device)
        
        pos_cache_residue = None
        if pos_cache_keys is not None:
            pos_cache_residue = PositiveCacheResidue(pos_cache_keys)
            
        # Optimization
        # Optimize residues for 1 step (as per DPE default)
        
        params = [{'params': text_residue.parameters(), 'lr': args.lr_text}]
        if pos_cache_residue is not None:
             params.append({'params': pos_cache_residue.parameters(), 'lr': args.lr_image})
             
        optimizer = torch.optim.AdamW(params, weight_decay=1e-1) # weight decay from DPE
        
        steps = 1
        for step in range(steps):
            optimizer.zero_grad()
            
            # Forward with residues
            new_clip_weights = text_residue(clip_weights_local)
            
            # Re-compute image features? No, assume fixed image features from model
            # DPE uses `image_features_x` which is fixed
            
            curr_logits = 100. * img_feats @ new_clip_weights
            
            if pos_cache_keys is not None:
                new_pos_cache_keys = pos_cache_residue(pos_cache_keys)
                cache_logits = compute_cache_logits(img_feats, new_pos_cache_keys, pos_cache_values, args.pos_alpha, args.pos_beta)
                curr_logits = curr_logits + cache_logits
                
                # Alignment Loss
                # Match adapted cache keys with adapted text prototypes
                # new_pos_cache_keys: [D, N_cache]
                # new_clip_weights: [D, C]
                # We want keys of class k to match text prototype of class k
                
                # Extract text prototypes corresponding to cache keys
                # pos_cache_labels_idx: [N_cache]
                target_text_protos = new_clip_weights[:, pos_cache_labels_idx] # [D, N_cache]
                
                align_loss = info_nce_loss(new_pos_cache_keys.t(), target_text_protos.t())
                
                total_loss = entropy_loss(curr_logits) + args.align_loss_weight * align_loss
            else:
                total_loss = entropy_loss(curr_logits)
                
            total_loss.backward()
            optimizer.step()
        
        # 4. Final Inference
        with torch.no_grad():
            new_clip_weights = text_residue(clip_weights_local)
            final_logits = 100. * img_feats @ new_clip_weights
            
            if pos_cache_keys is not None:
                new_pos_cache_keys = pos_cache_residue(pos_cache_keys)
                cache_logits = compute_cache_logits(img_feats, new_pos_cache_keys, pos_cache_values, args.pos_alpha, args.pos_beta)
                final_logits += cache_logits
            
            probs = F.softmax(final_logits, dim=1)
            preds = final_logits.argmax(dim=1)
            
            predictions.extend(preds.cpu().numpy())
            targets.extend(target.cpu().numpy())
            probabilities.extend(probs.cpu().numpy())
            
        # Optional: Global update of prototypes? (DPE has logic for this, let's skip for simplicity or check main_dpe.py)
        # main_dpe.py does global update if entropy < 0.1
        # "clip_weights_global = clip_weights_global * (num_avg / (num_avg + 1)) + new_clip_weights * (1 / (num_avg + 1))"
        # Let's keep it simple first (local adaptation only), or enabling it might help stability.
        # Let's Implement Global Text Prototype Update (Cumulative Avg)
        with torch.no_grad():
             avg_ent = entropy0.mean().item()
             if avg_ent < 0.1: # Only update if confident
                 # Use moving average or cumulative
                 # Simple EMA
                 clip_weights = clip_weights * 0.9 + new_clip_weights.detach() * 0.1
                 clip_weights = F.normalize(clip_weights, dim=0)

    return predictions, targets, probabilities

def main():
    args = get_arguments()
    set_seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 1. Load Model
    print(f"Loading BioMedCLIP: {args.model}")
    model, _, preprocess = open_clip.create_model_and_transforms(args.model)
    tokenizer = open_clip.get_tokenizer(args.model)
    model.to(device)
    model.eval()
    
    # 2. Config & Prompts
    prompts_file = args.prompts_file
    print(f"Loading prompts from {prompts_file}")
    class_prompts = load_prompts(prompts_file, args.dataset)
    num_classes = len(class_prompts)
    print(f"Dataset {args.dataset} has {num_classes} classes.")
    
    # 3. Text Features
    print("Encoding text prompts...")
    clip_weights = clip_classifier(class_prompts, model, tokenizer, device)
    
    # 4. Check Dataset Path
    dataset_folder_map = {
        'oral_cancer': 'oral_cancer_classification_dataset',
        'aptos': 'aptos_classification_dataset',
        'finger': 'fingerprint_classification_dataset',
        'mias': 'mias_classification_dataset',
        'octa': 'octa_classification_dataset'
    }
    
    folder_name = dataset_folder_map.get(args.dataset, args.dataset)
    # Handle absolute or relative paths
    if os.path.exists(os.path.join(args.data_root, folder_name)):
        dataset_dir = os.path.join(args.data_root, folder_name)
    elif os.path.exists(os.path.join(args.data_root, 'datasets', folder_name)): # Try nested
        dataset_dir = os.path.join(args.data_root, 'datasets', folder_name)
    elif os.path.exists(folder_name): # Try direct path
        dataset_dir = folder_name
    else:
        # Fallback to direct concatenation just in case user passed full path in data_root
        dataset_dir = os.path.join(args.data_root, folder_name)
        
    print(f"Dataset directory: {dataset_dir}")
    
    # 5. Dataset Loader
    # Use test set for TTA
    test_dir = os.path.join(dataset_dir, 'test')
    if not os.path.exists(test_dir):
        print("Test dir not found, checking for val...")
        test_dir = os.path.join(dataset_dir, 'val')
        
    if not os.path.exists(test_dir):
        raise ValueError(f"No test/val directory found in {dataset_dir}")
        
    print(f"Loading test data from {test_dir}")
    test_dataset = datasets.ImageFolder(test_dir, transform=preprocess)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=32,
        shuffle=True, # Shuffle to simulate random stream
        num_workers=4
    )
    
    train_loader = None
    if args.use_train_cache:
        train_dir = os.path.join(dataset_dir, 'train')
        if os.path.exists(train_dir):
             print(f"Loading train data from {train_dir} for cache init")
             train_dataset = datasets.ImageFolder(train_dir, transform=preprocess)
             # Shuffle to get random samples for cache
             train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
        else:
             print(f"Warning: Train dir {train_dir} not found. Skipping train cache.")
    
    # 6. Zero-Shot Evaluation (Baseline)
    print("Running Zero-Shot Evaluation...")
    zs_preds = []
    zs_targets = []
    zs_probs = []
    with torch.no_grad():
        for images, target in tqdm(test_loader, desc="Zero-Shot"):
            images = images.to(device)
            target = target.to(device)
            _, logits = get_clip_logits(images, model, clip_weights)
            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            zs_preds.extend(preds.cpu().numpy())
            zs_targets.extend(target.cpu().numpy())
            zs_probs.extend(probs.cpu().numpy())
            
    zs_acc = accuracy_score(zs_targets, zs_preds)
    zs_f1 = f1_score(zs_targets, zs_preds, average='macro')
    try:
        zs_probs_array = np.array(zs_probs)
        if num_classes == 2:
            zs_auc = roc_auc_score(zs_targets, zs_probs_array[:, 1])
        else:
            zs_auc = roc_auc_score(zs_targets, zs_probs_array, multi_class='ovr', average='macro')
    except:
        zs_auc = 0.0
        
    print(f"Zero-Shot Accuracy: {zs_acc:.4f}")
    print(f"Zero-Shot Macro F1: {zs_f1:.4f}")
    print(f"Zero-Shot AUC: {zs_auc:.4f}\n")

    # 7. Run DPE
    print("Running DPE Adaptation...")
    preds, targets, probs = run_dpe(args, model, test_loader, clip_weights, device, train_loader=train_loader)
    
    # 8. Metrics
    acc = accuracy_score(targets, preds)
    f1 = f1_score(targets, preds, average='macro')
    
    try:
        if num_classes == 2:
            auc = roc_auc_score(targets, np.array(probs)[:, 1])
        else:
            auc = roc_auc_score(targets, probs, multi_class='ovr', average='macro')
    except:
        auc = 0.0
        
    print(f"\nResults for {args.dataset}:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {f1:.4f}")
    print(f"AUC: {auc:.4f}")
    
    # 9. Save
    os.makedirs(args.output_dir, exist_ok=True)
    res = {
        'dataset': args.dataset,
        'acc': acc,
        'f1': f1,
        'auc': auc,
        'zs_acc': zs_acc,
        'zs_f1': zs_f1,
        'zs_auc': zs_auc
    }
    with open(os.path.join(args.output_dir, f'results_{args.dataset}.json'), 'w') as f:
        json.dump(res, f, indent=4)

if __name__ == '__main__':
    main()
