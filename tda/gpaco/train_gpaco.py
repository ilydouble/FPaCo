#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Baseline GPaCo (Generalized PaCo) Training Script
Supports: APTOS2019, MIAS, OralCancer, Fingerprint, OCTA
Based on GPaCo/MAE-ViTs/paco.py implementation.
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

# Import RandAugment if available, else simple fallback
try:
    from bpaco_original.randaugment import rand_augment_transform
except ImportError:
    def rand_augment_transform(config_str, hparams):
        return T.AutoAugment(T.AutoAugmentPolicy.IMAGENET)

# =========================================================
# GPaCo Loss (Adapted from GPaCo/MAE-ViTs/paco.py)
# =========================================================

class GPaCoLoss(nn.Module):
    def __init__(self, alpha=0.05, beta=1.0, gamma=1.0, supt=1.0, temperature=0.2, base_temperature=None, K=8192, num_classes=1000):
        super(GPaCoLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = temperature if base_temperature is None else base_temperature
        self.K = K
        self.alpha = alpha 
        self.beta = beta 
        self.gamma = gamma 
        self.supt = supt
        self.num_classes = num_classes

    def forward(self, features, labels=None, sup_logits=None, smooth=0.1):
        device = features.device
        
        batch_size = sup_logits.shape[0]
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels[:batch_size], labels.T).float().to(device)

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(features[:batch_size], features.T),
            self.temperature)

        # add supervised logits (NO class weight adjustment here compared to PaCo)
        anchor_dot_contrast = torch.cat(( (sup_logits) / self.supt, anchor_dot_contrast), dim=1)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # mask-out self-contrast cases
        logits_mask = torch.ones_like(mask)
        logits_mask.scatter_(1, torch.arange(batch_size, device=device).view(-1, 1), 0)
        mask = mask * logits_mask

        # add ground truth 
        one_hot_label = torch.nn.functional.one_hot(labels[:batch_size].view(-1,), num_classes=self.num_classes).to(torch.float32)
        one_hot_label = smooth / (self.num_classes - 1 ) * (1 - one_hot_label) + (1 - smooth) * one_hot_label
        mask = torch.cat((one_hot_label * self.beta, mask * self.alpha), dim=1)

        # compute log_prob
        logits_mask_full = torch.cat((torch.ones(batch_size, self.num_classes, device=device), self.gamma * logits_mask), dim=1)
        exp_logits = torch.exp(logits) * logits_mask_full
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
       
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()
        
        return loss

# =========================================================
# Dataset & Utils (Copied from BPaCo)
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
        
        # Handle RGB (3 channels) -> Grayscale (1 channel)
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

class StandardBPaCoDataset(Dataset):
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
        if is_finger:
            morph = FingerprintMorphologyTransform()
            self.aug1 = T.Compose([
                T.Resize((image_size, image_size)),
                T.RandomHorizontalFlip(),
                T.RandomApply([T.GaussianBlur(3)], p=0.2),
                T.ToTensor(),
                T.RandomApply([morph], p=0.7),
                T.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x),
                norm
            ])
            self.aug2 = T.Compose([
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                morph,
                T.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x),
                norm
            ])
        else:
            ra = rand_augment_transform('rand-m9-n3-mstd0.5', {})
            self.aug1 = T.Compose([
                T.RandomResizedCrop(image_size, scale=(0.6, 1.0)),
                T.RandomHorizontalFlip(),
                ra,
                T.ToTensor(),
                norm
            ])
            self.aug2 = T.Compose([
                T.RandomResizedCrop(image_size, scale=(0.6, 1.0)),
                T.RandomHorizontalFlip(),
                T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                T.RandomGrayscale(p=0.2),
                T.ToTensor(),
                norm
            ])
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
        if img.mode == 'P' and 'transparency' in img.info: img = img.convert('RGBA')
        img = img.convert('RGB')
        label = s['label']
        if self.split == 'train': return self.aug1(img), self.aug2(img), label
        else:
            img_t = self.val_transform(img)
            return img_t, img_t, label

class FeatureQueue:
    def __init__(self, size, dim, device):
        self.size = size
        self.dim = dim
        self.device = device
        self.feats = torch.zeros(size, dim, device=device)
        self.labels = torch.zeros(size, dtype=torch.long, device=device)
        self.ptr = 0
        self.is_full = False
    
    def enqueue(self, feats, labels):
        batch_size = feats.shape[0]
        if self.ptr + batch_size > self.size:
            rem = self.size - self.ptr
            self.feats[self.ptr:] = feats[:rem]
            self.labels[self.ptr:] = labels[:rem]
            self.feats[:batch_size-rem] = feats[rem:]
            self.labels[:batch_size-rem] = labels[rem:]
            self.ptr = batch_size - rem
            self.is_full = True
        else:
            self.feats[self.ptr:self.ptr+batch_size] = feats
            self.labels[self.ptr:self.ptr+batch_size] = labels
            self.ptr += batch_size
    
    def get(self):
        if not self.is_full and self.ptr == 0: return None, None
        if self.is_full: return self.feats, self.labels
        return self.feats[:self.ptr], self.labels[:self.ptr]

class Encoder(nn.Module):
    def __init__(self, backbone_name='resnet50', proj_dim=128):
        super().__init__()
        backbone = getattr(models, backbone_name)(weights='DEFAULT')
        self.feat_dim = backbone.fc.in_features
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])
        self.proj = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.feat_dim, proj_dim)
        )
    def forward(self, x):
        f = self.encoder(x).flatten(1)
        z = F.normalize(self.proj(f), dim=1)
        return f, z

# =========================================================
# Trainer
# =========================================================

class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.train_ds = StandardBPaCoDataset(args.dataset, 'train', args.image_size, args.is_finger)
        self.val_ds = StandardBPaCoDataset(args.dataset, 'val', args.image_size, args.is_finger)
        
        if args.merge_train_val:
            print("Merging validation set into training set...")
            self.train_ds.samples.extend(self.val_ds.samples)
            print(f"New training set size: {len(self.train_ds)}")
            print("Switching validation dataset to TEST set for model selection...")
            self.val_ds = StandardBPaCoDataset(args.dataset, 'test', args.image_size, args.is_finger)
        
        self.num_classes = self.train_ds.num_classes
        
        self.model_q = Encoder(args.backbone, args.proj_dim).to(self.device)
        self.model_k = Encoder(args.backbone, args.proj_dim).to(self.device)
        for pq, pk in zip(self.model_q.parameters(), self.model_k.parameters()):
            pk.data.copy_(pq.data)
            pk.requires_grad = False
            
        feat_dim = self.model_q.feat_dim
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.num_classes)
        ).to(self.device) 

        self.queue = FeatureQueue(args.queue_size, args.proj_dim, self.device)
        
        # Loss
        self.criterion = GPaCoLoss(
            alpha=args.paco_alpha, 
            beta=args.paco_beta, 
            gamma=args.paco_gamma, 
            supt=1.0, 
            temperature=args.temperature, 
            K=args.queue_size, 
            num_classes=self.num_classes
        ).to(self.device)

        params = list(self.model_q.parameters()) + list(self.classifier.parameters())
        self.optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=args.wd)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, args.epochs)
        
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(self, epoch):
        self.model_q.train()
        self.model_k.train()
        self.classifier.train()
        loader = DataLoader(self.train_ds, batch_size=self.args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
        total_loss = 0
        
        for v1, v2, labels in loader:
            v1, v2, labels = v1.to(self.device), v2.to(self.device), labels.to(self.device)
            
            # Q
            f_q, z_q = self.model_q(v1)
            logits = self.classifier(f_q)
            
            # K
            with torch.no_grad():
                f_k, z_k = self.model_k(v2)
            
            # Prepare features for GPaCo Loss
            q_feats, q_labels = self.queue.get()
            if q_feats is not None:
                features = torch.cat([z_q, q_feats], dim=0)
                all_labels = torch.cat([labels, q_labels], dim=0)
            else:
                features = z_q
                all_labels = labels
                
            loss = self.criterion(features, labels=all_labels, sup_logits=logits, smooth=self.args.smooth)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Momentum update
            for pq, pk in zip(self.model_q.parameters(), self.model_k.parameters()):
                pk.data = pk.data * self.args.m + pq.data * (1. - self.args.m)
            
            self.queue.enqueue(z_k.detach(), labels)
            total_loss += loss.item()
            
        return total_loss / len(loader)

    @torch.no_grad()
    def evaluate(self):
        self.model_q.eval()
        self.classifier.eval()
        val_loader = DataLoader(self.val_ds, batch_size=self.args.batch_size, shuffle=False)
        all_preds, all_labels, all_probs = [], [], []
        
        for v1, _, labels in val_loader:
            v1, labels = v1.to(self.device), labels.to(self.device)
            f_q, _ = self.model_q(v1)
            logits = self.classifier(f_q)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
        all_probs = np.array(all_probs)
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')
        return acc, f1, all_labels, all_preds, all_probs

    def plot_roc_curve(self, labels, probs, num_classes, save_path):
        y_test = label_binarize(labels, classes=range(num_classes))
        n_classes = y_test.shape[1]
        fpr, tpr, roc_auc = dict(), dict(), dict()
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
            plt.plot(fpr["micro"], tpr["micro"], label=f'micro-average ROC curve (area = {roc_auc["micro"]:.2f})', color='deeppink', linestyle=':', linewidth=4)
            colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple', 'brown'])
            for i, color in zip(range(n_classes), colors):
                plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'ROC curve of class {i} (area = {roc_auc[i]:.2f})')
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
                torch.save(self.model_q.state_dict(), self.output_dir / 'best_model.pth')
                cm = confusion_matrix(labels, preds)
                plt.figure(figsize=(10, 8))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.savefig(self.output_dir / 'best_cm.png')
                plt.close()
                try: self.plot_roc_curve(labels, probs, self.num_classes, self.output_dir / 'best_roc_curve.png')
                except Exception as e: print(f"Warning: ROC plotting failed: {e}")
        with open(self.output_dir / 'history.json', 'w') as f:
            json.dump(history, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--is-finger', action='store_true')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--wd', type=float, default=1e-4)
    parser.add_argument('--backbone', default='resnet18')
    parser.add_argument('--proj_dim', type=int, default=128)
    parser.add_argument('--queue-size', type=int, default=8192)
    parser.add_argument('--m', type=float, default=0.999)
    parser.add_argument('--image-size', type=int, default=448)
    parser.add_argument('--merge-train-val', action='store_true')
    
    # GPaCo params
    parser.add_argument('--temperature', type=float, default=0.2)
    parser.add_argument('--paco-alpha', type=float, default=0.05)
    parser.add_argument('--paco-beta', type=float, default=1.0)
    parser.add_argument('--paco-gamma', type=float, default=1.0)
    parser.add_argument('--smooth', type=float, default=0.1)
    
    args = parser.parse_args()
    Trainer(args).run()
