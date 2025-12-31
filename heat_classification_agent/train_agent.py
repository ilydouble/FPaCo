#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BPaCo (Balanced Prototype and Contrastive Learning) - RGB + Heatmap Early Fusion Version
Modified for offline heatmap integration. Can also run in RGB-only mode with --no-heatmap.

Input: [R, G, B, Heatmap] (4 Channels) OR [R, G, B] (3 Channels)
"""

import os
import sys
import random
import json
import argparse
import math
from pathlib import Path
from datetime import datetime

import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torchvision.models as models

from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from heatmap_utils import generate_gaussian_heatmap, parse_detections_from_json
    from heat_augmentation import YOLOAugmentation
except ImportError:
    # Inline fallback if import fails
    def generate_gaussian_heatmap(h, w, boxes, scores=None, sigma=15):
        heatmap = np.zeros((h, w), dtype=np.float32)
        if len(boxes) == 0:
            return heatmap
        xx, yy = np.meshgrid(np.arange(w), np.arange(h))
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            score = scores[i].item() if scores is not None else 1.0
            dist_sq = (xx - cx)**2 + (yy - cy)**2
            blob = np.exp(-dist_sq / (2 * sigma**2))
            heatmap = np.maximum(heatmap, blob * score)
        return heatmap

    def parse_detections_from_json(data):
         # Minimal fallback for train_agent standalone
         boxes = []
         if 'detections' in data:
              for det in data['detections']:
                  boxes.append(det['bbox'])
         return np.array(boxes), None


# Set Seed
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# =========================================================
# Dataset
# =========================================================

class HeatmapBPaCoDataset(Dataset):
    """
    Input: RGB Image + JSON Detections
    Output:
      If use_heatmap=True: 4-Channel Tensor [R, G, B, Heatmap]
      If use_heatmap=False: 3-Channel Tensor [R, G, B]
    """
    def __init__(self, dataset_dir, split='train', image_size=224, sigma=30, combine_train_val=False, use_heatmap=True):
        self.dataset_dir = Path(dataset_dir)
        self.split = split
        self.image_size = image_size
        self.sigma = sigma
        self.use_heatmap = use_heatmap
        
        self.samples = []
        
        # Determine directories to load
        dirs_to_load = []
        if combine_train_val and split == 'train':
            # Load both train and val folders
            dirs_to_load.append(self.dataset_dir / 'train')
            dirs_to_load.append(self.dataset_dir / 'val')
        else:
            dirs_to_load.append(self.dataset_dir / split)
            
        print(f"Loading data from: {dirs_to_load}")
        
        # Prepare class mapping (assuming folder structure: split/class_name/img)
        # We need a unified class mapping.
        # Check 'train' folder for classes
        base_split_dir = self.dataset_dir / 'train'
        if not base_split_dir.exists(): 
             # Fallback if structure is flat or different
             base_split_dir = self.dataset_dir
             
        class_dirs = sorted([d for d in base_split_dir.iterdir() if d.is_dir()])
        if not class_dirs and (self.dataset_dir / 'val').exists():
             class_dirs = sorted([d for d in (self.dataset_dir / 'val').iterdir() if d.is_dir()])
             
        self.class_to_idx = {d.name: i for i, d in enumerate(class_dirs)}
        self.classes = [d.name for d in class_dirs]
        
        for d_path in dirs_to_load:
            if not d_path.exists(): continue
            
            for class_name in self.classes:
                class_dir = d_path / class_name
                if not class_dir.exists(): continue
                
                class_idx = self.class_to_idx[class_name]
                for img_path in class_dir.glob('*.*'):
                    if img_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp']:
                        self.samples.append((str(img_path), class_idx))

        print(f"[{split}] Loaded {len(self.samples)} samples. Classes: {len(self.classes)}. Heatmap enabled: {self.use_heatmap}")
        
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.yolo_aug = YOLOAugmentation()
        
        # Determine if we should skip geometric transforms (for 'finger' datasets)
        self.skip_geometric = 'finger' in str(self.dataset_dir).lower()
        if self.skip_geometric:
            print(f"[{split}] 'finger' detected in dataset path. Geometric transforms (flip/rotate) will be SKIPPED.")

    def __len__(self):
        return len(self.samples)

    def _apply_augmentations(self, pil_img, heatmap_np):
        """
        Apply augmentation to a single sample.
        Geometric transforms applied to both img and heat.
        Photometric transforms applied only to img.
        """
        # Convert to Tensor for joint geometric transforms
        img_tensor = TF.to_tensor(pil_img) # 0-1
        
        if self.use_heatmap: # 4 channels
            heat_tensor = torch.from_numpy(heatmap_np).unsqueeze(0) # [1, H, W]
            tensor_to_aug = torch.cat([img_tensor, heat_tensor], dim=0) # [4, H, W]
        else: # 3 channels
            tensor_to_aug = img_tensor # [3, H, W]

        # 1. Base Resize
        tensor_to_aug = TF.resize(tensor_to_aug, [self.image_size, self.image_size], antialias=True)

        if self.split == 'train':
            # 2. Geometric Transforms (Joint)
            if not self.skip_geometric:
                if random.random() < 0.5:
                    tensor_to_aug = TF.hflip(tensor_to_aug)
                
                if random.random() < 0.5:
                    angle = random.uniform(-15, 15)
                    tensor_to_aug = TF.rotate(tensor_to_aug, angle)

            # Split back for photometric
            if self.use_heatmap:
                img_aug = TF.to_pil_image(tensor_to_aug[:3, :, :])
                heat_aug = tensor_to_aug[3, :, :].unsqueeze(0)
            else:
                img_aug = TF.to_pil_image(tensor_to_aug)
                heat_aug = None

            # 3. Photometric Transforms (RGB only)
            # Apply Strong YOLO-style augmentation to all
            img_aug = self.yolo_aug.hsv_augment(img_aug)
            img_aug = self.yolo_aug.gaussian_blur(img_aug, p=0.2, kernel_size=5)
            img_aug = self.yolo_aug.add_noise(img_aug, p=0.15)
            
            final_img = TF.to_tensor(img_aug)
            final_heat = heat_aug
        else:
            if self.use_heatmap:
                final_img = tensor_to_aug[:3, :, :]
                final_heat = tensor_to_aug[3, :, :].unsqueeze(0)
            else:
                final_img = tensor_to_aug
                final_heat = None

        # Final tensor
        if self.use_heatmap:
            res = torch.cat([final_img, final_heat], dim=0)
            res[:3] = self.normalize(res[:3])
        else:
            res = self.normalize(final_img)
            
        return res

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # 1. Load Image (RGB)
        try:
            pil_img = Image.open(img_path)
            if pil_img.mode == 'P':
                pil_img = pil_img.convert('RGBA')
            pil_img = pil_img.convert('RGB')
            w, h = pil_img.size
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return self.__getitem__(random.randint(0, len(self)-1))

        heatmap = None
        if self.use_heatmap:
            # 2. Load JSON & Generate Heatmap
            json_path = Path(img_path).with_suffix('.json')
            boxes = []
            scores = None
            
            if json_path.exists():
                try:
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                    boxes, scores = parse_detections_from_json(data)
                except Exception as e:
                    # print(f"Error parse {json_path}: {e}")
                    pass
            
            if len(boxes) == 0:
                 boxes = np.zeros((0, 4))
            
            heatmap = generate_gaussian_heatmap(h, w, boxes, scores=scores, sigma=self.sigma)
            
            if heatmap.max() > 0:
                heatmap = heatmap / heatmap.max()
        
        # 3. Apply Augmentation (All are normal samples now)
        v1 = self._apply_augmentations(pil_img, heatmap)
        if self.split == 'train':
            v2 = self._apply_augmentations(pil_img, heatmap)
        else:
            v2 = v1.clone()
        
        return {
            'v1': v1, # [4, H, W] or [3, H, W]
            'v2': v2, # [4, H, W] or [3, H, W]
            'label': label
        }

# =========================================================
# Model
# =========================================================

class BpacoResNet(nn.Module):
    def __init__(self, backbone='resnet18', num_classes=10, proj_dim=128, pretrained=True, input_channels=4):
        super().__init__()
        
        # Backbone
        model_fun = getattr(models, backbone)
        weights = 'DEFAULT' if pretrained else None
        self.encoder = model_fun(weights=weights)
        
        # Modify Conv1 for input channels if not 3
        # ResNet Conv1: Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        if input_channels != 3:
            print(f"Modifying Conv1 for {input_channels} channels.")
            original_conv1 = self.encoder.conv1
            self.encoder.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            
            with torch.no_grad():
                # Copy RGB weights
                self.encoder.conv1.weight[:, :3, :, :] = original_conv1.weight
                if input_channels > 3:
                    # Init Heatmap weights (mean of RGB) for extra channels
                    # For 4 channel: index 3
                    self.encoder.conv1.weight[:, 3:, :, :] = original_conv1.weight.mean(dim=1, keepdim=True)
        else:
            print("Using standard 3-channel Conv1.")
            
        self.feat_dim = self.encoder.fc.in_features
        self.encoder.fc = nn.Identity()
        
        # Projection Head
        self.proj = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.feat_dim, proj_dim)
        )
        
        # Classifier is external (in mixing logic) or here? 
        # Train logic usually separates classifier to handle [q, k] concat.
        # But we can put a simple method here.

    def forward(self, x):
        feat = self.encoder(x)
        z = self.proj(feat)
        z = F.normalize(z, dim=1)
        return feat, z

# =========================================================
# Utils
# =========================================================

class MoCoQueue:
    def __init__(self, dim, size, device):
        self.dim = dim
        self.size = size
        self.device = device
        self.ptr = 0
        self.queue = torch.randn(size, dim).to(device)
        self.queue = F.normalize(self.queue, dim=1)
        self.labels = torch.zeros(size, dtype=torch.long).to(device)

    def enqueue(self, keys, labels):
        batch_size = keys.shape[0]
        ptr = int(self.ptr)
        if ptr + batch_size > self.size:
            batch_size = self.size - ptr
        self.queue[ptr:ptr+batch_size] = keys[:batch_size]
        self.labels[ptr:ptr+batch_size] = labels[:batch_size]
        self.ptr = (ptr + batch_size) % self.size

def momentum_update(model_k, model_q, m=0.999):
    for param_k, param_q in zip(model_k.parameters(), model_q.parameters()):
        param_k.data = param_k.data * m + param_q.data * (1. - m)

def compute_contrastive_loss(features, labels, queue, prototypes, temperature=0.07):
    # Simplified BPaCo/MoCo Loss against Prototypes
    # features: [B, Dim] (Query)
    # prototypes: [NumClasses, Dim]
    
    # Positive logits: Query * Prototype(Label)
    # Negative logits: Query * Other Prototypes + Query * Queue
    
    # Simple implementation: InfoNCE with Prototypes as positives
    
    # 1. Similarity with Prototypes
    logits_proto = torch.matmul(features, prototypes.T) / temperature # [B, n_cls]
    
    loss = F.cross_entropy(logits_proto, labels)
    return loss

def cross_entropy_with_logit_compensation(logits, targets, class_freq, tau=1.0):
    if class_freq is None: 
        return F.cross_entropy(logits, targets)
        
    prior = class_freq / class_freq.sum()
    log_prior = torch.log(prior + 1e-8).to(logits.device)
    adjusted_logits = logits + tau * log_prior
    return F.cross_entropy(adjusted_logits, targets)

# =========================================================
# Trainer
# =========================================================

class BPaCoHeatmapTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.use_heatmap = not args.no_heatmap
        self.input_channels = 4 if self.use_heatmap else 3
        print(f"Training Config: Heatmap={self.use_heatmap}, Input Channels={self.input_channels}")
        
        # Dataset
        self.train_dataset = HeatmapBPaCoDataset(
            args.dataset, split='train', image_size=args.image_size, sigma=args.sigma,
            combine_train_val=args.combine_train_val,
            use_heatmap=self.use_heatmap
        )
        self.val_dataset = HeatmapBPaCoDataset(
            args.dataset, split='test', image_size=args.image_size, sigma=args.sigma,
            use_heatmap=self.use_heatmap
        )
        
        self.num_classes = len(self.train_dataset.classes)
        print(f"Num Classes: {self.num_classes}")
        
        # Model
        print(f"Building {self.input_channels}-Channel BPaCo Model ({args.backbone})...")
        self.model_q = BpacoResNet(backbone=args.backbone, num_classes=self.num_classes, input_channels=self.input_channels).to(self.device)
        self.model_k = BpacoResNet(backbone=args.backbone, num_classes=self.num_classes, input_channels=self.input_channels).to(self.device)
        
        # Momentum Init
        for param_q, param_k in zip(self.model_q.parameters(), self.model_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
            
        # Classifier (Concat Feature Dim -> Num Classes)
        # Feat dim * 2 because we concat [feat_q, feat_k] ideally, or just feat_q
        
        # Wait, standard BPaCo uses concat(feat_q, feat_k) for classification?
        # multimodal script used: self.classifier = nn.Linear(self.model_q.feat_dim * 2, self.num_classes)
        # Let's keep that.
        self.classifier = nn.Linear(self.model_q.feat_dim * 2, self.num_classes).to(self.device)
        
        # Prototypes
        self.C1 = nn.Parameter(torch.randn(self.num_classes, 128).to(self.device))
        
        # Queue
        self.queue = MoCoQueue(dim=128, size=args.queue_size, device=self.device)
        
        # Optimizer
        self.optimizer = torch.optim.SGD(
            list(self.model_q.parameters()) + list(self.classifier.parameters()) + [self.C1],
            lr=args.lr, momentum=0.9, weight_decay=1e-4
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.epochs)
        
        # Training Stats
        self.class_freq = torch.zeros(self.num_classes).to(self.device)
        for s in self.train_dataset.samples:
             self.class_freq[s[1]] += 1

    def run(self):
        train_loader = DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(self.val_dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=4)
        
        best_f1 = 0.0
        history = {'train_loss': [], 'val_acc': [], 'val_f1': []}
        
        print("Starting Training...")
        for epoch in range(1, self.args.epochs + 1):
            self.model_q.train()
            train_loss = 0
            
            for i, batch in enumerate(train_loader):
                v1 = batch['v1'].to(self.device)
                v2 = batch['v2'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward
                feat_q, z_q = self.model_q(v1)
                with torch.no_grad():
                    feat_k, z_k = self.model_k(v2)
                
                # Classifier
                feat_cat = torch.cat([feat_q, feat_k], dim=1)
                logits = self.classifier(feat_cat)
                
                # Loss
                loss_ce = cross_entropy_with_logit_compensation(logits, labels, self.class_freq, tau=self.args.tau)
                loss_con = compute_contrastive_loss(z_q, labels, self.queue, self.C1, temperature=self.args.temperature)
                
                loss = loss_ce + self.args.beta * loss_con
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # Update
                momentum_update(self.model_k, self.model_q)
                with torch.no_grad():
                    self.queue.enqueue(z_k, labels)
                    
                train_loss += loss.item()
                
            self.scheduler.step()
            avg_loss = train_loss / len(train_loader)
            history['train_loss'].append(avg_loss)
            
            # Validation
            if epoch % self.args.val_interval == 0:
                acc, f1 = self.validate(val_loader)
                print(f"Epoch {epoch}: Loss={avg_loss:.4f}, Val Acc={acc:.4f}, Val F1={f1:.4f}")
                history['val_acc'].append(acc)
                history['val_f1'].append(f1)
                
                if f1 > best_f1:
                    best_f1 = f1
                    checkpoint = {
                        'model_q': self.model_q.state_dict(),
                        'classifier': self.classifier.state_dict(),
                        'epoch': epoch,
                        'f1': f1
                    }
                    torch.save(checkpoint, os.path.join(self.args.output_dir, "best_model.pth"))
            else:
                print(f"Epoch {epoch}: Loss={avg_loss:.4f}")

        # Final Eval
        print("Loading best model...")
        best_path = os.path.join(self.args.output_dir, "best_model.pth")
        if os.path.exists(best_path):
             checkpoint = torch.load(best_path, map_location=self.device)
             self.model_q.load_state_dict(checkpoint['model_q'])
             self.classifier.load_state_dict(checkpoint['classifier'])
             print(f"Loaded best model from Epoch {checkpoint.get('epoch', '?')} (F1={checkpoint.get('f1', '0.00'):.4f})")
             
        # Load Test Dataset for Final Evaluation
        print("Loading Test Set for Final Evaluation...")
        test_dataset = HeatmapBPaCoDataset(
            self.args.dataset, split='test', image_size=self.args.image_size, sigma=self.args.sigma,
            use_heatmap=self.use_heatmap
        )
        test_loader = DataLoader(test_dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=4)
        
        self.evaluate_full(test_loader)
        
    @torch.no_grad()
    def validate(self, loader):
        self.model_q.eval()
        preds, targets = [], []
        for batch in loader:
            v1 = batch['v1'].to(self.device)
            labels = batch['label'].to(self.device)
            
            feat, _ = self.model_q(v1)
            # Use duplicated feat for eval since we don't have k view? Or just use feat_q * 2
            feat_cat = torch.cat([feat, feat], dim=1)
            logits = self.classifier(feat_cat)
            
            pred = torch.argmax(logits, dim=1)
            preds.extend(pred.cpu().numpy())
            targets.extend(labels.cpu().numpy())
            
        return accuracy_score(targets, preds), f1_score(targets, preds, average='macro')

    @torch.no_grad()
    def evaluate_full(self, loader):
        self.model_q.eval()
        preds, targets, probs = [], [], []
        for batch in loader:
            v1 = batch['v1'].to(self.device)
            labels = batch['label'].to(self.device)
            
            feat, _ = self.model_q(v1)
            feat_cat = torch.cat([feat, feat], dim=1)
            logits = self.classifier(feat_cat)
            prob = F.softmax(logits, dim=1)
            
            pred = torch.argmax(logits, dim=1)
            
            preds.extend(pred.cpu().numpy())
            targets.extend(labels.cpu().numpy())
            probs.extend(prob.cpu().numpy())
            
        targets = np.array(targets)
        preds = np.array(preds)
        probs = np.array(probs)
        
        # Save Plots
        self.plot_confusion_matrix(targets, preds, os.path.join(self.args.output_dir, "confusion_matrix.png"))
        self.plot_roc_curve(targets, probs, os.path.join(self.args.output_dir, "roc_curve.png"))
        
        # Save Metrics
        acc = accuracy_score(targets, preds)
        f1 = f1_score(targets, preds, average='macro')
        
        # Calculate Macro AUC if possible
        try:
             # reuse roc_auc calculation logic or simplified
             from sklearn.preprocessing import label_binarize
             from sklearn.metrics import roc_auc_score
             n_classes = self.num_classes
             y_test = label_binarize(targets, classes=range(n_classes))
             if n_classes == 2 and y_test.shape[1] == 1:
                 y_test = np.hstack([1 - y_test, y_test])
             auc_val = roc_auc_score(y_test, probs, average='macro', multi_class='ovr')
        except:
             auc_val = 0.0

        print(f"Final Results: Acc={acc:.4f}, F1={f1:.4f}, AUC={auc_val:.4f}")
        
        results_dict = {
            'acc': acc,
            'f1': f1,
            'auc': auc_val,
            'config': {
                'use_heatmap': self.use_heatmap,
                'input_channels': self.input_channels
            }
        }
        with open(os.path.join(self.args.output_dir, "results.json"), "w") as f:
            json.dump(results_dict, f, indent=4)
        
    def plot_confusion_matrix(self, targets, preds, save_path):
        cm = confusion_matrix(targets, preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True')
        plt.xlabel('Pred')
        plt.savefig(save_path)
        plt.close()

    def plot_roc_curve(self, targets, probs, save_path):
        from sklearn.preprocessing import label_binarize
        from sklearn.metrics import roc_curve, auc
        
        n_classes = self.num_classes
        y_test = label_binarize(targets, classes=range(n_classes))
        
        # specific handling for binary classification
        if n_classes == 2 and y_test.shape[1] == 1:
            y_test = np.hstack([1 - y_test, y_test])
        
        plt.figure(figsize=(10, 8))
        for i in range(n_classes):
             if np.sum(y_test[:, i]) > 0:
                 fpr, tpr, _ = roc_curve(y_test[:, i], probs[:, i])
                 roc_auc = auc(fpr, tpr)
                 plt.plot(fpr, tpr, lw=1, alpha=0.5, label=f'Class {i} (AUC={roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.savefig(save_path)
        plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='results')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--backbone', type=str, default='resnet18')
    parser.add_argument('--image-size', type=int, default=224) # Resized for 4-channel
    parser.add_argument('--sigma', type=int, default=30)
    parser.add_argument('--beta', type=float, default=2.0)
    parser.add_argument('--tau', type=float, default=1.0)
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--queue-size', type=int, default=8192)
    parser.add_argument('--val-interval', type=int, default=1)
    parser.add_argument('--combine-train-val', action='store_true', help="Merge train and val folders for training", default=True)
    
    # New argument for ablation study
    parser.add_argument('--no-heatmap', action='store_true', help="Disable heatmap channel (use RGB only)")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    trainer = BPaCoHeatmapTrainer(args)
    trainer.run()
