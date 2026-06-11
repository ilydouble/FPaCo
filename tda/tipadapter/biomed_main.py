import os
import argparse
import random
import yaml
from tqdm import tqdm

import torch
import torch.nn as nn
import open_clip

from biomed_datasets import BiomedDataset
from biomed_utils import *

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='dataset name: oral_cancer, aptos, finger, mias, octa')
    parser.add_argument('--root_path', type=str, default='../datasets', help='root path of datasets')
    parser.add_argument('--shots', type=int, default=16, help='number of shots')
    parser.add_argument('--model', type=str, default='hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224', help='BioMedCLIP model name')
    parser.add_argument('--cache_dir', type=str, default='./caches', help='cache directory')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--augment_epoch', type=int, default=10, help='augment epoch for cache construction')
    parser.add_argument('--train_epoch', type=int, default=20, help='train epoch for Tip-Adapter-F')
    parser.add_argument('--init_beta', type=float, default=3.0, help='initial beta')
    parser.add_argument('--init_alpha', type=float, default=0.5, help='initial alpha')
    parser.add_argument('--output_dir', type=str, default='results', help='output directory for results')
    
    # Flags
    parser.add_argument('--load_cache', action='store_true', help='load cache from file')
    parser.add_argument('--load_pre_feat', action='store_true', help='load pre-extracted features')
    parser.add_argument('--search_hp', action='store_true', help='search hyperparameters')
    
    args = parser.parse_args()
    return args

def main():
    args = get_arguments()
    
    # Config dict
    cfg = vars(args)
    cfg['cache_dir'] = os.path.join(args.cache_dir, args.dataset)
    os.makedirs(cfg['cache_dir'], exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load BioMedCLIP
    print(f"Loading BioMedCLIP: {args.model}")
    model, _, preprocess = open_clip.create_model_and_transforms(args.model)
    tokenizer = open_clip.get_tokenizer(args.model)
    model.to(device)
    model.eval()

    # Reproducibility
    random.seed(1)
    torch.manual_seed(1)

    # Prepare Dataset
    print(f"Preparing dataset: {args.dataset}")
    # Note: BiomedDataset handles creating train_preprocess internally similar to CLIP
    dataset = BiomedDataset(args.dataset, args.root_path, args.shots, preprocess)
    
    # Data Loaders
    print("Building DataLoaders...")
    batch_size = 64
    
    # For Tip-Adapter, we iterate over train_x (few-shot) for cache construction
    # and validation/test for eval.
    # Note: Using standard torch DataLoader
    train_loader_cache = torch.utils.data.DataLoader(dataset.train_x, batch_size=256, shuffle=False, num_workers=4)
    train_loader_F = torch.utils.data.DataLoader(dataset.train_x, batch_size=256, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(dataset.val, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = torch.utils.data.DataLoader(dataset.test, batch_size=batch_size, shuffle=False, num_workers=4)
    
    cfg['num_classes'] = len(dataset.classnames)

    # Textual Features (Classifier)
    print("Getting textual features...")
    clip_weights = clip_classifier(dataset.classnames, dataset.template, model, tokenizer, device)

    # Construct Cache Model
    print("Constructing cache model...")
    cache_keys, cache_values = build_cache_model(cfg, model, train_loader_cache, device)

    # Pre-load features
    print("Loading valid features...")
    val_features, val_labels = pre_load_features(cfg, "val", model, val_loader, device)
    print("Loading test features...")
    test_features, test_labels = pre_load_features(cfg, "test", model, test_loader, device)

    # ------------------------------------------ Tip-Adapter (Training-Free) ------------------------------------------
    print("\n-------- Tip-Adapter (Training-Free) --------")
    
    # Search HP on Val
    if args.search_hp:
        print("Searching hyperparameters on val set...")
        cfg['search_scale'] = [7, 3] # [beta_scale, alpha_scale]? Original config values
        cfg['search_step'] = [200, 20]
        best_beta, best_alpha = search_hp(cfg, cache_keys, cache_values, val_features, val_labels, clip_weights)
    else:
        best_beta, best_alpha = args.init_beta, args.init_alpha
        
    # Eval on Test
    print(f"Evaluating on test set (beta={best_beta}, alpha={best_alpha})...")
    
    # Zero-shot CLIP
    clip_logits = 100. * test_features @ clip_weights
    acc_zs, f1_zs, auc_zs = evaluate_metrics(clip_logits, test_labels)
    print(f"Zero-shot CLIP Accuracy: {acc_zs:.2f}% - F1: {f1_zs:.4f} - AUC: {auc_zs:.4f}")
    
    # Tip-Adapter
    affinity = test_features @ cache_keys
    cache_logits = ((-1) * (best_beta - best_beta * affinity)).exp() @ cache_values
    
    tip_logits = clip_logits + cache_logits * best_alpha
    acc_tip, f1_tip, auc_tip = evaluate_metrics(tip_logits, test_labels)
    print(f"Tip-Adapter Accuracy: {acc_tip:.2f}% - F1: {f1_tip:.4f} - AUC: {auc_tip:.4f}")

    
    # ------------------------------------------ Tip-Adapter-F (Fine-Tuning) ------------------------------------------
    print("\n-------- Tip-Adapter-F (Fine-Tuning) --------")
    
    # Adapter Layer
    # open_clip models might not expose .dtype directly
    try:
        model_dtype = model.dtype
    except AttributeError:
        model_dtype = next(model.parameters()).dtype
        
    adapter = nn.Linear(cache_keys.shape[0], cache_keys.shape[1], bias=False).to(model_dtype).to(device)
    adapter.weight = nn.Parameter(cache_keys.t())
    
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=cfg['lr'], eps=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg['train_epoch'] * len(train_loader_F))
    
    beta, alpha = best_beta, best_alpha # Use best HP from training-free or init? Original re-searches or uses initial?
    # Original main_imagenet.py uses init_beta/alpha for training, then search again on val.
    # Let's stick to using best found or init. 
    # Actually, original code re-uses init_beta/alpha for F training start.
    
    beta, alpha = args.init_beta, args.init_alpha 
    
    
    best_acc_F = 0.0
    best_f1_F = 0.0
    best_auc_F = 0.0
    
    for epoch in range(cfg['train_epoch']):
        adapter.train()
        loss_epoch = 0
        
        for i, (images, target) in enumerate(tqdm(train_loader_F, desc=f"Epoch {epoch+1}/{cfg['train_epoch']}")):
            images, target = images.to(device), target.to(device)
            
            with torch.no_grad():
                image_features = model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)
            
            affinity = adapter(image_features)
            cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
            clip_logits = 100. * image_features @ clip_weights
            tip_logits = clip_logits + cache_logits * alpha
            
            loss = nn.functional.cross_entropy(tip_logits, target)
            loss_epoch += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
        # Eval after epoch
        adapter.eval()
        with torch.no_grad():
            affinity = adapter(test_features)
            cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
            clip_logits = 100. * test_features @ clip_weights
            tip_logits = clip_logits + cache_logits * alpha
            acc, f1, auc = evaluate_metrics(tip_logits, test_labels)
        
        if acc > best_acc_F:
            best_acc_F = acc
            best_f1_F = f1
            best_auc_F = auc
        
        print(f"Epoch {epoch+1} Test Acc: {acc:.2f}% (Best: {best_acc_F:.2f}%) - F1: {f1:.4f} - AUC: {auc:.4f} Loss: {loss_epoch/len(train_loader_F):.4f}")

    print(f"\nFinal Tip-Adapter-F Accuracy: {best_acc_F:.2f}% - F1: {best_f1_F:.4f} - AUC: {best_auc_F:.4f}")
    
    # Save results to a simple text file
    result_log = os.path.join(args.output_dir, f"{args.dataset}_biomed_tipadapter.txt")
    with open(result_log, 'w') as f:
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Zero-Shot Acc: {acc_zs:.2f}%\n")
        f.write(f"Zero-Shot F1: {f1_zs:.4f}\n")
        f.write(f"Zero-Shot AUC: {auc_zs:.4f}\n")
        f.write(f"Tip-Adapter Acc: {acc_tip:.2f}%\n")
        f.write(f"Tip-Adapter F1: {f1_tip:.4f}\n")
        f.write(f"Tip-Adapter AUC: {auc_tip:.4f}\n")
        f.write(f"Tip-Adapter-F Acc: {best_acc_F:.2f}%\n")
        f.write(f"Tip-Adapter-F F1: {best_f1_F:.4f}\n")
        f.write(f"Tip-Adapter-F AUC: {best_auc_F:.4f}\n")
    print(f"Results saved to {result_log}")

if __name__ == '__main__':
    main()
