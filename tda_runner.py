import random
import argparse
from tqdm import tqdm
from pathlib import Path
import json
import operator

import torch
import torch.nn.functional as F
import clip
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

from utils import *


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
    parser.add_argument('--backbone', choices=['RN50', 'ViT-B/16'], required=True)
    return parser.parse_args()


# =========================
# Cache helpers
# =========================

def update_cache(cache, pred, features_loss, shot_capacity):
    with torch.no_grad():
        if pred in cache:
            if len(cache[pred]) < shot_capacity:
                cache[pred].append(features_loss)
            elif features_loss[1] < cache[pred][-1][1]:
                cache[pred][-1] = features_loss
            cache[pred] = sorted(cache[pred], key=operator.itemgetter(1))
        else:
            cache[pred] = [features_loss]


def compute_cache_logits(image_features, cache, alpha, beta, clip_weights):
    device = clip_weights.device

    cache_keys = []
    cache_values = []

    for class_index in sorted(cache.keys()):
        for item in cache[class_index]:
            cache_keys.append(item[0])
            cache_values.append(class_index)

    cache_keys = torch.cat(cache_keys, dim=0).permute(1, 0)

    cache_values = F.one_hot(
        torch.tensor(cache_values, dtype=torch.int64),
        num_classes=clip_weights.size(1)
    ).to(device).float()

    affinity = (image_features @ cache_keys).float()
    cache_logits = torch.exp(-beta + beta * affinity) @ cache_values

    return alpha * cache_logits


# =========================
# TDA evaluation
# =========================

def run_test_tda(pos_cfg, loader, clip_model, clip_weights):
    device = next(clip_model.parameters()).device
    pos_cache = {}

    y_true, y_pred, y_score = [], [], []

    pos_enabled = pos_cfg['enabled']
    pos_params = {k: pos_cfg[k] for k in ['shot_capacity', 'alpha', 'beta']}

    for images, target in tqdm(loader, desc='Processed test images: '):
        image_features, clip_logits, loss, _, pred = get_clip_logits(
            images, clip_model, clip_weights
        )

        target = target.to(device)

        if pos_enabled:
            update_cache(pos_cache, pred, [image_features, loss], pos_params['shot_capacity'])

        final_logits = clip_logits.clone()
        if pos_cache:
            final_logits += compute_cache_logits(
                image_features,
                pos_cache,
                pos_params['alpha'],
                pos_params['beta'],
                clip_weights
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

    clip_model, preprocess = clip.load(args.backbone)
    clip_model.eval()

    random.seed(1)
    torch.manual_seed(1)

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
        dataset_path = Path(args.data_root) / datasets_map[dataset_name]
        if not dataset_path.exists():
            print(f"Skipping {dataset_name}: path not found")
            continue

        print(f"\nProcessing {dataset_name} dataset.")

        cfg = get_config_file(args.config, dataset_name)
        print("Running dataset configurations:")
        print(cfg)

        test_loader, classnames, template = build_test_data_loader(
            dataset_name, args.data_root, preprocess
        )

        clip_weights = clip_classifier(classnames, template, clip_model)

        acc, f1, auc = run_test_tda(
            cfg['positive'],
            test_loader,
            clip_model,
            clip_weights
        )

        result = {
            "dataset": dataset_name,
            "accuracy": acc,
            "f1": f1,
            "auc": auc
        }

        with open(output_dir / f"{dataset_name}.json", "w") as f:
            json.dump(result, f, indent=4)

        results.append(result)

    print("\nFinal Summary:")
    print(f"{'Dataset':<15} {'Acc':<10} {'F1':<10} {'AUC':<10}")
    print("-" * 50)

    for r in results:
        auc_str = f"{r['auc']:.4f}" if r['auc'] is not None else "N/A"
        print(
            f"{r['dataset']:<15} "
            f"{r['accuracy']:<10.2f} "
            f"{r['f1']:<10.4f} "
            f"{auc_str:<10}"
        )


if __name__ == "__main__":
    main()
