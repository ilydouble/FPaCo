import os
import random
import argparse
from tqdm import tqdm
from pathlib import Path
import json
import operator

import torch
import torch.nn.functional as F
import open_clip
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
import torchvision.datasets as datasets


# =========================
# Arguments
# =========================

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument(
        '--dataset',
        default='all',
        choices=['all', 'oral_cancer', 'aptos', 'finger', 'mias', 'octa'],
        help='Run one dataset or all'
    )
    parser.add_argument('--data-root', required=True)
    parser.add_argument(
        '--model',
        type=str,
        default='hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224',
        help='BioMedCLIP model name'
    )
    parser.add_argument('--prompts-file', type=str, default='prompts/unified_prompts.json',
                        help='Path to prompts json file')
    parser.add_argument('--use-train-cache', action='store_true', help='Initialize cache with training data')
    parser.add_argument('--evaluate-zeroshot', action='store_true', default=True, help='Run zero-shot evaluation before TDA')
    return parser.parse_args()


class CustomImageFolder(datasets.ImageFolder):
    def find_classes(self, directory):
        """
        Override find_classes to sort folders numerically if they look like 'class_X'.
        Standard ImageFolder sorts alphabetically (class_10 before class_2).
        We want class_2 before class_10.
        """
        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
        
        # Check if classes follow 'class_X' pattern
        can_sort_numerically = True
        try:
            # Sort by the integer suffix
            classes.sort(key=lambda x: int(x.split('_')[-1]))
        except (ValueError, IndexError):
            can_sort_numerically = False
            
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")
            
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        
        print(f"DEBUG: CustomImageFolder found {len(classes)} classes.")
        print(f"DEBUG: First 5 classes: {classes[:5]}")
        print(f"DEBUG: Last 5 classes: {classes[-5:]}")
        print(f"DEBUG: class_to_idx sample: {list(class_to_idx.items())[:5]}")
        
        return classes, class_to_idx
# =========================

def get_config_file(config_path, dataset_name):
    import yaml
    config_file = os.path.join(config_path, f"{dataset_name}.yaml")
    
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file not found: {config_file}")
    
    with open(config_file, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    
    return cfg


def load_prompts(json_path, dataset_name):
    """Load prompts from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    if dataset_name not in data:
        raise ValueError(f"Dataset {dataset_name} not found in prompts file.")
    
    class_prompts_map = data[dataset_name]
    sorted_keys = sorted(class_prompts_map.keys(), key=lambda x: int(x))
    
    prompts_list = []
    for k in sorted_keys:
        prompts_list.append(class_prompts_map[k])
    
    return prompts_list


def clip_classifier(class_prompts, model, tokenizer, device):
    """Build classifier weights from prompts."""
    weights = []
    with torch.no_grad():
        for prompts in class_prompts:
            texts = tokenizer(prompts).to(device)
            class_embeddings = model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            
            mean_embedding = class_embeddings.mean(dim=0)
            mean_embedding /= mean_embedding.norm()
            weights.append(mean_embedding)
    
    weights = torch.stack(weights, dim=1).to(device)
    return weights


def softmax_entropy(x):
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def get_entropy(loss, clip_weights):
    import math
    max_entropy = math.log2(clip_weights.size(1))
    return float(loss / max_entropy)


def get_clip_logits(images, model, clip_weights, device):
    """Get CLIP logits using BioMedCLIP model."""
    with torch.no_grad():
        images = images.to(device)
        
        image_features = model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        
        clip_logits = 100. * image_features @ clip_weights
        
        loss = softmax_entropy(clip_logits)
        prob_map = clip_logits.softmax(1)
        pred = int(clip_logits.topk(1, 1, True, True)[1].t()[0])
        
        return image_features, clip_logits, loss, prob_map, pred


# =========================
# Cache helpers
# =========================

def update_cache(cache, pred, features_loss, shot_capacity, include_prob_map=False):
    """Update cache with new features and loss."""
    with torch.no_grad():
        item = features_loss if not include_prob_map else features_loss[:2] + [features_loss[2]]
        if pred in cache:
            if len(cache[pred]) < shot_capacity:
                cache[pred].append(item)
            else:
                # Find the item with the highest entropy (last one in sorted list) to potentially replace
                # Items are sorted by entropy (item[1]), so [-1] is the worst
                if features_loss[1] < cache[pred][-1][1]:
                    cache[pred][-1] = item
            
            # Sort by entropy (ascending): lower entropy (better confidence) first
            cache[pred] = sorted(cache[pred], key=operator.itemgetter(1))
        else:
            cache[pred] = [item]


def compute_cache_logits(image_features, cache, alpha, beta, clip_weights, neg_mask_thresholds=None):
    """Compute logits using positive/negative cache."""
    device = clip_weights.device

    cache_keys = []
    cache_values = []

    for class_index in sorted(cache.keys()):
        for item in cache[class_index]:
            cache_keys.append(item[0])
            if neg_mask_thresholds:
                cache_values.append(item[2])
            else:
                cache_values.append(class_index)

    cache_keys = torch.cat(cache_keys, dim=0).permute(1, 0)

    if neg_mask_thresholds:
        cache_values = torch.cat(cache_values, dim=0)
        cache_values = (
            ((cache_values > neg_mask_thresholds[0]) & (cache_values < neg_mask_thresholds[1]))
            .type(torch.int8)
        ).to(device).float()
    else:
        cache_values = F.one_hot(
            torch.tensor(cache_values, dtype=torch.int64),
            num_classes=clip_weights.size(1)
        ).to(device).float()

    affinity = (image_features @ cache_keys).float()
    cache_logits = torch.exp(-beta + beta * affinity) @ cache_values

    return alpha * cache_logits


# =========================
# Additional Evaluation Helpers
# =========================

def run_zeroshot(loader, clip_model, clip_weights, device):
    y_true, y_pred, y_score = [], [], []
    
    with torch.no_grad():
        for images, target in tqdm(loader, desc='Zero-Shot Eval'):
            images = images.to(device)
            target = target.to(device)
            
            image_features = clip_model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            clip_logits = 100. * image_features @ clip_weights
            
            probs = clip_logits.softmax(dim=1)
            
            y_true.extend(target.cpu().numpy().tolist())
            y_pred.extend(probs.argmax(dim=1).cpu().numpy().tolist())
            y_score.extend(probs.cpu().numpy().tolist())
            
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_score = np.array(y_score)

    acc = (y_true == y_pred).mean() * 100
    f1 = f1_score(y_true, y_pred, average='macro')
    try:
        if len(np.unique(y_true)) > 2:
             auc = roc_auc_score(y_true, y_score, multi_class='ovr')
        else:
             # Handle binary case or single class present
             if y_score.shape[1] == 2:
                  auc = roc_auc_score(y_true, y_score[:, 1])
             else:
                  auc = 0.0
    except ValueError:
        auc = None
        
    print(f"Zero-Shot -> Acc: {acc:.2f} | F1: {f1:.4f} | AUC: {auc}")
    return acc, f1, auc


def init_cache_with_train(cache, loader, clip_model, shot_capacity, device):
    """
    Initialize cache with training data. 
    We assign a very low entropy (-1e9) so these trusted samples are rarely evicted.
    """
    print(f"Initializing positive cache with training data (Capacity: {shot_capacity})...")
    with torch.no_grad():
        for images, target in tqdm(loader, desc="Train Init"):
            images = images.to(device)
            # We don't need gradients here, but we use the model to get features
            image_features = clip_model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
            # Simulated low entropy for ground truth
            entropy_val = -1e9
            
            for i in range(images.size(0)):
                label = target[i].item()
                # Dummy loss/prob_map structure to match TDA requirements: [features, loss (entropy)]
                # Note: TDA update_cache expects [features, loss] for pos and [features, loss, prob] for neg.
                # Here we only init positive cache.
                feat = image_features[i].unsqueeze(0) # [1, D]
                
                # We update cache directly
                update_cache(cache, label, [feat, entropy_val], shot_capacity)


# =========================
# TDA evaluation
# =========================

def run_test_tda(pos_cfg, neg_cfg, loader, clip_model, clip_weights, device, initial_pos_cache=None):
    pos_cache, neg_cache = {}, {}
    
    # Copy initial cache if provided
    if initial_pos_cache:
        import copy
        pos_cache = copy.deepcopy(initial_pos_cache)

    y_true, y_pred, y_score = [], [], []

    pos_enabled, neg_enabled = pos_cfg['enabled'], neg_cfg['enabled']
    if pos_enabled:
        pos_params = {k: pos_cfg[k] for k in ['shot_capacity', 'alpha', 'beta']}
    if neg_enabled:
        neg_params = {k: neg_cfg[k] for k in ['shot_capacity', 'alpha', 'beta', 'entropy_threshold', 'mask_threshold']}

    for images, target in tqdm(loader, desc='Processed test images: '):
        image_features, clip_logits, loss, prob_map, pred = get_clip_logits(
            images, clip_model, clip_weights, device
        )

        target = target.to(device)
        prop_entropy = get_entropy(loss, clip_weights)

        if pos_enabled:
            update_cache(pos_cache, pred, [image_features, loss], pos_params['shot_capacity'])

        if neg_enabled and neg_params['entropy_threshold']['lower'] < prop_entropy < neg_params['entropy_threshold']['upper']:
            update_cache(neg_cache, pred, [image_features, loss, prob_map], neg_params['shot_capacity'], include_prob_map=True)

        final_logits = clip_logits.clone()
        
        if pos_enabled and pos_cache:
            final_logits += compute_cache_logits(
                image_features,
                pos_cache,
                pos_params['alpha'],
                pos_params['beta'],
                clip_weights
            )
        
        if neg_enabled and neg_cache:
            final_logits -= compute_cache_logits(
                image_features,
                neg_cache,
                neg_params['alpha'],
                neg_params['beta'],
                clip_weights,
                (neg_params['mask_threshold']['lower'], neg_params['mask_threshold']['upper'])
            )

        probs = final_logits.softmax(dim=1)

        y_true.append(target.item())
        y_pred.append(probs.argmax(dim=1).item())
        y_score.append(probs.squeeze().cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_score = np.array(y_score)

    acc = (y_true == y_pred).mean() * 100
    f1 = f1_score(y_true, y_pred, average='macro')

    try:
        auc = roc_auc_score(y_true, y_score, multi_class='ovr')
    except ValueError:
        auc = None

    print(f"---- Acc: {acc:.2f} | F1: {f1:.4f} | AUC: {auc} ----")

    return acc, f1, auc


# =========================
# Entry point
# =========================

def main():
    args = get_arguments()

    # Load BioMedCLIP model
    print(f"Loading model: {args.model}")
    clip_model, _, preprocess = open_clip.create_model_and_transforms(args.model)
    tokenizer = open_clip.get_tokenizer(args.model)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_model = clip_model.to(device)
    clip_model.eval()

    random.seed(1)
    torch.manual_seed(1)

    # Dataset folder mapping (same as dpe_biomedclip.py)
    datasets_map = {
        'oral_cancer': 'oral_cancer_classification_dataset',
        'aptos': 'aptos_classification_dataset',
        'finger': 'fingerprint_classification_dataset',
        'mias': 'mias_classification_dataset',
        'octa': 'octa_classification_dataset'
    }

    if args.dataset != 'all':
        target_datasets = [args.dataset]
    else:
        target_datasets = list(datasets_map.keys())

    results = []
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)

    for dataset_name in target_datasets:
        folder_name = datasets_map[dataset_name]
        dataset_dir = os.path.join(args.data_root, folder_name)
        
        if not os.path.exists(dataset_dir):
            print(f"Skipping {dataset_name}: path not found at {dataset_dir}")
            continue

        print(f"\nProcessing {dataset_name} dataset.")

        # Load config
        cfg = get_config_file(args.config, dataset_name)
        print("Running dataset configurations:")
        print(cfg)

        # Load prompts
        prompts_file = os.path.join(Path(__file__).parent, args.prompts_file)
        class_prompts = load_prompts(prompts_file, dataset_name)
        num_classes = len(class_prompts)
        print(f"Dataset has {num_classes} classes.")

        # Build text classifier
        clip_weights = clip_classifier(class_prompts, clip_model, tokenizer, device)

        # Load test data
        test_dir = os.path.join(dataset_dir, 'test')
        if not os.path.exists(test_dir):
            test_dir = os.path.join(dataset_dir, 'val')
        
        if not os.path.exists(test_dir):
            print(f"No test/val directory found in {dataset_dir}")
            continue
        
        print(f"Loading test data from {test_dir}")
        test_dataset = CustomImageFolder(test_dir, transform=preprocess)
        # For TDA, batch_size=1 is typical as it processes stream, but larger batches can work if logic supports it.
        # Original code used batch_size=1
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1, 
            shuffle=True,
            num_workers=4
        )

        # --- Zero-Shot Evaluation ---
        if args.evaluate_zeroshot:
            print(f"\nRunning Zero-Shot Evaluation for {dataset_name}...")
            zs_acc, zs_f1, zs_auc = run_zeroshot(test_loader, clip_model, clip_weights, device)
        else:
            zs_acc, zs_f1, zs_auc = 0.0, 0.0, 0.0

        # --- Train Cache Initialization ---
        pos_cache_init = {}
        if args.use_train_cache:
            train_dir = os.path.join(dataset_dir, 'train')
            train_dir = os.path.join(dataset_dir, 'train')
            if os.path.exists(train_dir):
                print(f"Loading train data from {train_dir} for cache initialization...")
                train_dataset = CustomImageFolder(train_dir, transform=preprocess)
                init_loader = torch.utils.data.DataLoader(
                    train_dataset, batch_size=32, shuffle=True, num_workers=4
                )
                init_cache_with_train(
                    pos_cache_init, 
                    init_loader, 
                    clip_model, 
                    cfg['positive']['shot_capacity'], 
                    device
                )
            else:
                print(f"Warning: Train directory not found at {train_dir}. Skipping train cache init.")

        # --- Run TDA ---
        print(f"\nRunning TDA for {dataset_name}...")
        acc, f1, auc = run_test_tda(
            cfg['positive'],
            cfg['negative'],
            test_loader,
            clip_model,
            clip_weights,
            device,
            initial_pos_cache=pos_cache_init
        )

        result = {
            "dataset": dataset_name,
            "accuracy": acc,
            "f1": f1,
            "auc": auc,
            "zs_accuracy": zs_acc,
            "zs_f1": zs_f1,
            "zs_auc": zs_auc
        }

        with open(output_dir / f"{dataset_name}.json", "w") as f:
            json.dump(result, f, indent=4)

        results.append(result)

    print("\nFinal Summary:")
    print(f"{'Dataset':<15} {'ZS Acc':<10} {'TDA Acc':<10} {'ZS F1':<10} {'TDA F1':<10}")
    print("-" * 65)

    for r in results:
        print(
            f"{r['dataset']:<15} "
            f"{r['zs_accuracy']:<10.2f} "
            f"{r['accuracy']:<10.2f} "
            f"{r['zs_f1']:<10.4f} "
            f"{r['f1']:<10.4f} "
        )


if __name__ == "__main__":
    main()
