#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Baseline Focal Loss Training Script
Supports: APTOS2019, MIAS, OralCancer, Fingerprint, OCTA
"""

import os
import sys
import random
import json
import argparse
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import torchvision.models as models
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle
import matplotlib.pyplot as plt
import seaborn as sns

# =========================================================
# Focal Loss Definition
# =========================================================

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        # Alpha is handled as weight in CE if provided, or manual scaling
        self.alpha = alpha 

    def forward(self, inputs, targets):
        # inputs: [N, C], logits
        # targets: [N], class indices
        
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# =========================================================
# Morphology Transform for Fingerprint (Copied)
# =========================================================

class FingerprintMorphologyTransform:
    def __init__(self, kernel_tophat=15, kernel_blackhat=25, kernel_open=3, block_size=21, C=8):
        self.kernel_tophat = kernel_tophat
        self.kernel_blackhat = kernel_blackhat
        self.kernel_open = kernel_open
        self.block_size = block_size
        self.C = C

    def __call__(self, x):
        if isinstance(x, Image.Image):
            x = T.ToTensor()(x)
        
        if x.ndim == 4:
            x = x.squeeze(0)
            
        img = x.cpu().numpy()
        
        if img.ndim == 3 and img.shape[0] == 3:
             img = img[0, :, :]
        elif img.ndim == 3 and img.shape[0] == 1:
             img = img.squeeze(0)
             
        # Ensure contiguous array for OpenCV
        img = np.ascontiguousarray(img)
        
        img_u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)

        k_top = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.kernel_tophat, self.kernel_tophat))
        tophat = cv2.morphologyEx(img_u8, cv2.MORPH_TOPHAT, k_top)
        tophat_enh = cv2.normalize(tophat, None, 0, 255, cv2.NORM_MINMAX) if tophat.max() > 0 else tophat

        binary = cv2.adaptiveThreshold(tophat_enh, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, self.block_size, self.C)

        k_bh = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.kernel_blackhat, self.kernel_blackhat))
        blackhat = cv2.morphologyEx(img_u8, cv2.MORPH_BLACKHAT, k_bh)
        if blackhat.max() > 0:
            blackhat = cv2.normalize(blackhat, None, 0, 255, cv2.NORM_MINMAX)
        blackhat_bin = cv2.adaptiveThreshold(blackhat, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, self.block_size, self.C)

        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.kernel_open, self.kernel_open))
        blackhat_open = cv2.morphologyEx(blackhat_bin, cv2.MORPH_OPEN, k_open)

        inv_blackhat = 255 - blackhat_open
        result = cv2.bitwise_and(binary, inv_blackhat)
        result_t = torch.from_numpy(result.astype(np.float32) / 255.0).unsqueeze(0)
        return result_t

# =========================================================
# Dataset
# =========================================================

class BaselineDataset(Dataset):
    def __init__(self, dataset_dir, split='train', image_size=224, is_finger=False):
        self.dataset_dir = Path(dataset_dir)
        self.split = split
        self.image_size = image_size
        self.is_finger = is_finger

        self.samples = []
        class_dirs = sorted([d for d in (self.dataset_dir / split).iterdir() if d.is_dir() and d.name.startswith('class_')])
        self.num_classes = len(class_dirs)
        
        for d in class_dirs:
            class_id = int(d.name.split('_')[1])
            for img_path in d.glob('*.*'):
                if img_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.pgm']:
                    self.samples.append({'path': str(img_path), 'label': class_id})
        
        print(f"Loaded {len(self.samples)} samples for {split} split ({self.num_classes} classes).")

        norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if is_finger:
            norm = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        
        # Training Transforms
        if is_finger:
            morph = FingerprintMorphologyTransform()
            self.train_transform = T.Compose([
                T.Resize((image_size, image_size)),
                T.RandomHorizontalFlip(),
                T.RandomApply([T.GaussianBlur(3)], p=0.2),
                T.ToTensor(),
                T.RandomApply([morph], p=0.7),
                T.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x),
                norm
            ])
        else:
            self.train_transform = T.Compose([
                T.RandomResizedCrop(image_size, scale=(0.6, 1.0)),
                T.RandomHorizontalFlip(),
                T.AutoAugment(T.AutoAugmentPolicy.IMAGENET),
                T.ToTensor(),
                norm
            ])

        # Validation Transforms
        self.val_transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x),
            norm
        ])

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = Image.open(s['path'])
        if img.mode == 'P' and 'transparency' in img.info:
            img = img.convert('RGBA')
        img = img.convert('RGB')
        label = s['label']
        
        if self.split == 'train':
            return self.train_transform(img), label
        else:
            return self.val_transform(img), label

# =========================================================
# Trainer
# =========================================================

class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Datasets
        self.train_ds = BaselineDataset(args.dataset, 'train', args.image_size, args.is_finger)
        self.val_ds = BaselineDataset(args.dataset, 'val', args.image_size, args.is_finger)
        
        # Optional: Merge Train + Val
        if args.merge_train_val:
            print("Merging validation set into training set...")
            self.train_ds.samples.extend(self.val_ds.samples)
            print(f"New training set size: {len(self.train_ds)}")
            
            print("Switching validation dataset to TEST set for model selection...")
            self.val_ds = BaselineDataset(args.dataset, 'test', args.image_size, args.is_finger)
        
        self.num_classes = self.train_ds.num_classes
        
        # Model (ResNet18)
        self.model = models.resnet18(weights='DEFAULT')
        # Modify last FC layer
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.num_classes)
        )
        self.model.to(self.device)

        # Loss (Focal)
        self.criterion = FocalLoss(gamma=args.focal_gamma)
        
        # Optimizer
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, args.epochs)
        
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(self, epoch):
        self.model.train()
        train_loader = DataLoader(self.train_ds, batch_size=self.args.batch_size, shuffle=True, num_workers=4)
        total_loss = 0
        
        for img, label in train_loader:
            img, label = img.to(self.device), label.to(self.device)
            
            logits = self.model(img)
            loss = self.criterion(logits, label)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(train_loader)

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        val_loader = DataLoader(self.val_ds, batch_size=self.args.batch_size, shuffle=False)
        all_preds, all_labels, all_probs = [], [], []
        
        for img, label in val_loader:
            img, label = img.to(self.device), label.to(self.device)
            logits = self.model(img)
            
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
        all_probs = np.array(all_probs)
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')
        return acc, f1, all_labels, all_preds, all_probs

    def plot_roc_curve(self, labels, probs, num_classes, save_path):
        # Binarize labels
        y_test = label_binarize(labels, classes=range(num_classes))
        n_classes = y_test.shape[1]
        
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        if n_classes == 1 and num_classes == 2:
             fpr[1], tpr[1], _ = roc_curve(y_test, probs[:, 1])
             roc_auc[1] = auc(fpr[1], tpr[1])
             plt.figure()
             plt.plot(fpr[1], tpr[1], label=f'ROC curve (area = {roc_auc[1]:.2f})')
        else:
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_test[:, i], probs[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            
            fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), probs.ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
            
            plt.figure(figsize=(10, 8))
            plt.plot(fpr["micro"], tpr["micro"],
                     label=f'micro-average ROC curve (area = {roc_auc["micro"]:.2f})',
                     color='deeppink', linestyle=':', linewidth=4)
    
            colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple', 'brown'])
            for i, color in zip(range(n_classes), colors):
                plt.plot(fpr[i], tpr[i], color=color, lw=2,
                         label=f'ROC curve of class {i} (area = {roc_auc[i]:.2f})')

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        plt.savefig(save_path)
        plt.close()

    def run(self):
        best_f1 = 0
        history = {'loss': [], 'acc': [], 'f1': [], 'auc': []}
        
        for epoch in range(1, self.args.epochs + 1):
            loss = self.train_epoch(epoch)
            acc, f1, labels, preds, probs = self.evaluate()
            self.scheduler.step()
            
            # AUC & Other F1s
            try:
                if self.num_classes == 2:
                     auc_score = roc_auc_score(labels, probs[:, 1])
                     f1_binary = f1_score(labels, preds, average='binary')
                     metric_str = f"Loss: {loss:.4f} - Acc: {acc:.4f} - Macro-F1: {f1:.4f} - Binary-F1: {f1_binary:.4f} - AUC: {auc_score:.4f}"
                else:
                    auc_score = roc_auc_score(labels, probs, multi_class='ovr', average='macro')
                    auc_weighted = roc_auc_score(labels, probs, multi_class='ovr', average='weighted')
                    f1_weighted = f1_score(labels, preds, average='weighted')
                    f1_micro = f1_score(labels, preds, average='micro')
                    metric_str = f"Loss: {loss:.4f} - Acc: {acc:.4f} - Macro-F1: {f1:.4f} - Micro-F1: {f1_micro:.4f} - Weighted-F1: {f1_weighted:.4f} - AUC: {auc_score:.4f} - Weighted-AUC: {auc_weighted:.4f}"
            except:
                auc_score = 0.0
                metric_str = f"Loss: {loss:.4f} - Acc: {acc:.4f} - Macro-F1: {f1:.4f} - AUC: Error"
            
            history['loss'].append(loss)
            history['acc'].append(acc)
            history['f1'].append(f1)
            history['auc'].append(auc_score)
            
            print(f"Epoch {epoch}/{self.args.epochs} - {metric_str}")
            
            if f1 > best_f1:
                best_f1 = f1
                torch.save(self.model.state_dict(), self.output_dir / 'best_model.pth')
                
                # CM
                cm = confusion_matrix(labels, preds)
                plt.figure(figsize=(10, 8))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.savefig(self.output_dir / 'best_cm.png')
                plt.close()
                
                # ROC
                try:
                    self.plot_roc_curve(labels, probs, self.num_classes, self.output_dir / 'best_roc_curve.png')
                except Exception as e:
                     print(f"Warning: ROC plotting failed: {e}")
        
        with open(self.output_dir / 'history.json', 'w') as f:
            json.dump(history, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--is-finger', action='store_true')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--wd', type=float, default=1e-4)
    parser.add_argument('--image-size', type=int, default=448)
    parser.add_argument('--merge-train-val', action='store_true', help='Merge train and val sets for training')
    
    # Focal Loss params
    parser.add_argument('--focal-gamma', type=float, default=2.0, help='Gamma for Focal Loss')
    
    args = parser.parse_args()
    
    Trainer(args).run()
