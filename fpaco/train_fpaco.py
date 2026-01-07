#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FPaCo (Fine-Grained PaCo) - Advanced Heatmap Guidance Version
Includes:
1. Attention Alignment (L_guide)
2. Feature Disentanglement (Foreground/Background Contrastive)
3. Uncertainty-Gated Fusion (Optional, but implicitly handled via Disentanglement mechanisms)

Input: [R, G, B, Heatmap] (4 Channels) - Heatmap is used for both Input and Guidance.
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
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torchvision.models as models

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from heatmap_utils import generate_gaussian_heatmap, parse_detections_from_json
    from heat_augmentation import YOLOAugmentation
except ImportError:
    # Fallback definitions
    def generate_gaussian_heatmap(h, w, boxes, scores=None, sigma=15):
        heatmap = np.zeros((h, w), dtype=np.float32)
        if len(boxes) == 0: return heatmap
        xx, yy = np.meshgrid(np.arange(w), np.arange(h))
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            w_box = max(x2 - x1, 1.0)
            h_box = max(y2 - y1, 1.0)
            score = scores[i].item() if scores is not None else 1.0
            
            sigma_x = max(w_box / 2.0, 2.0)
            sigma_y = max(h_box / 2.0, 2.0)

            exponent = -((xx - cx)**2 / (2 * sigma_x**2) + (yy - cy)**2 / (2 * sigma_y**2))
            blob = np.exp(exponent)
            heatmap = np.maximum(heatmap, blob * score)
        return heatmap

    def parse_detections_from_json(data):
         boxes = []
         if 'detections' in data:
              for det in data['detections']:
                  boxes.append(det['bbox'])
         return np.array(boxes), None

    class YOLOAugmentation:
        @staticmethod
        def hsv_augment(image, **kwargs): return image
        @staticmethod
        def gaussian_blur(image, **kwargs): return image
        @staticmethod
        def add_noise(image, **kwargs): return image

# Set Seed
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# =========================================================
# Morphology Transform for Fingerprint (Copied from offline version)
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


set_seed(42)

# =========================================================
# Dataset
# =========================================================

class HeatmapBPaCoDataset(Dataset):
    def __init__(self, dataset_dir, split='train', image_size=224, sigma=30, combine_train_val=False, use_heatmap=True):
        self.dataset_dir = Path(dataset_dir)
        self.split = split
        self.image_size = image_size
        self.sigma = sigma
        self.use_heatmap = use_heatmap
        self.samples = []
        
        # Determine directories
        dirs_to_load = []
        if combine_train_val and split == 'train':
            dirs_to_load.append(self.dataset_dir / 'train')
            dirs_to_load.append(self.dataset_dir / 'val')
        else:
            dirs_to_load.append(self.dataset_dir / split)
            
        print(f"Loading data from: {dirs_to_load}")
        
        # Mapping
        base_split_dir = self.dataset_dir / 'train'
        if not base_split_dir.exists(): base_split_dir = self.dataset_dir
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
        if 'finger' in str(self.dataset_dir).lower():
            self.normalize = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        self.is_finger = 'finger' in str(self.dataset_dir).lower()
        if self.is_finger:
            self.morph = FingerprintMorphologyTransform()


    def __len__(self): return len(self.samples)

    def _apply_augmentations(self, pil_img, heatmap_np):
        """
        Apply augmentation to a single sample using logic aligned with Focal/CE/GPaCo baselines.
        (Ported from fpaco-noheat for consistency)
        """
        # --- 1. Fingerprint Strategy (Morphology) ---
        if self.is_finger:
            img_tensor = TF.to_tensor(pil_img)
            
            # Apply Morphology Logic (Prob 0.7)
            if self.split == 'train' and random.random() < 0.7:
                 # Morph expects tensor [C, H, W]
                 img_tensor = self.morph(img_tensor)
                 # Reconstruct 3 channels if it became 1 channel
                 if img_tensor.shape[0] == 1:
                     img_tensor = img_tensor.repeat(3, 1, 1)

            # Apply Geometric Transforms (Resize, Flip, Blur)
            img_tensor = TF.resize(img_tensor, [self.image_size, self.image_size], antialias=True)
            
            if self.use_heatmap:
                heat_tensor = torch.from_numpy(heatmap_np).unsqueeze(0)
                heat_tensor = TF.resize(heat_tensor, [self.image_size, self.image_size], interpolation=T.InterpolationMode.NEAREST)
            else:
                 heat_tensor = None
            
            if self.split == 'train':
                # Gaussian Blur (p=0.4)
                if random.random() < 0.4:
                    img_tensor = T.GaussianBlur(3)(img_tensor)

        # --- 2. Generic Medical Strategy (AutoAugment + RandomResizedCrop) ---
        else:
            if self.split == 'train':
                # Random Resized Crop parameters
                i, j, h, w = T.RandomResizedCrop.get_params(pil_img, scale=(0.6, 1.0), ratio=(3./4., 4./3.))
                img_pil = TF.resized_crop(pil_img, i, j, h, w, size=(self.image_size, self.image_size))
                
                if self.use_heatmap:
                    # Heatmap is float numpy array. Convert to Tensor/Image for cropping
                    heat_tensor = torch.from_numpy(heatmap_np).unsqueeze(0)
                    heat_tensor = TF.resized_crop(heat_tensor, i, j, h, w, size=(self.image_size, self.image_size), interpolation=T.InterpolationMode.NEAREST)
                else:
                    heat_tensor = None
                
                # Random Horizontal Flip
                if random.random() < 0.5:
                    img_pil = TF.hflip(img_pil)
                    if self.use_heatmap: heat_tensor = TF.hflip(heat_tensor)
                
                # --- Strategy Branching based on Domain Knowledge ---
                is_color_sensitive = 'aptos' in str(self.dataset_dir).lower() or 'oral' in str(self.dataset_dir).lower()
                
                if is_color_sensitive:
                    # Mild photometric noise for Retina/Oral
                    if random.random() < 0.8:
                        img_pil = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.01)(img_pil)
                else:
                    # AutoAugment for Structure (MIAS, OCTA)
                    try:
                        policy = T.AutoAugmentPolicy.IMAGENET
                        img_pil = T.AutoAugment(policy)(img_pil)
                    except:
                        pass 
                    
                img_tensor = TF.to_tensor(img_pil)

            else:
                # Validation: Resize only
                img_tensor = TF.to_tensor(pil_img)
                img_tensor = TF.resize(img_tensor, [self.image_size, self.image_size], antialias=True)
                
                if self.use_heatmap:
                    heat_tensor = torch.from_numpy(heatmap_np).unsqueeze(0)
                    heat_tensor = TF.resize(heat_tensor, [self.image_size, self.image_size], interpolation=T.InterpolationMode.NEAREST)
                else:
                    heat_tensor = None

        # Final concatenation
        if self.use_heatmap:
            res = torch.cat([img_tensor, heat_tensor], dim=0)
            res[:3] = self.normalize(res[:3])
        else:
            res = self.normalize(img_tensor)
        return res


    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            pil_img = Image.open(img_path).convert('RGB')
            w, h = pil_img.size
        except:
            return self.__getitem__(random.randint(0, len(self)-1))

        heatmap = np.zeros((h, w), dtype=np.float32)
        if self.use_heatmap:
            json_path = Path(img_path).with_suffix('.json')
            boxes = []
            scores = None
            if json_path.exists():
                try:
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                    boxes, scores = parse_detections_from_json(data)
                except: pass
            
            heatmap = generate_gaussian_heatmap(h, w, boxes, scores=scores, sigma=self.sigma)
            
            # [MODIFIED] Do NOT normalize to 1.0 (heatmap / max).
            # Instead, reduce confidence globally to combat overconfidence.
            # e.g., multiply by 0.8
            # [MODIFIED] Do NOT normalize to 1.0 (heatmap / max).
            # heatmap = heatmap * 0.2 -> Removed scaling to use full 0.0-1.0 range
            # This allows the adaptive alpha logic to work correctly (trusting high confidence heatmaps)
            
            # Clip just in case, though it shouldn't exceed 1.0 if scores are <=1.0
            heatmap = np.clip(heatmap, 0.0, 1.0)
        
        v1 = self._apply_augmentations(pil_img, heatmap)
        if self.split == 'train':
            v2 = self._apply_augmentations(pil_img, heatmap)
        else:
            v2 = v1.clone()
        
        return {'v1': v1, 'v2': v2, 'label': label}

# =========================================================
# Model
# =========================================================

class FPaCoResNet(nn.Module):
    def __init__(self, backbone='resnet18', num_classes=10, proj_dim=128, pretrained=True, input_channels=4):
        super().__init__()
        
        model_fun = getattr(models, backbone)
        self.encoder = model_fun(weights='DEFAULT' if pretrained else None)
        
        # Modify Conv1
        if input_channels != 3:
            original_conv1 = self.encoder.conv1
            self.encoder.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            with torch.no_grad():
                self.encoder.conv1.weight[:, :3, :, :] = original_conv1.weight
                if input_channels > 3:
                    self.encoder.conv1.weight[:, 3:, :, :] = original_conv1.weight.mean(dim=1, keepdim=True)
            
        self.feat_dim = self.encoder.fc.in_features
        # Remove FC, keeping the pooling
        self.encoder.fc = nn.Identity()
        
        # Projection Head
        self.proj = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.feat_dim, proj_dim)
        )

    def forward(self, x, return_feature_map=False):
        # We need intermediate feature maps for Attention Alignment
        # ResNet structure: (conv1->maxpool) -> layer1 -> layer2 -> layer3 -> layer4 -> avgpool -> fc
        
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.maxpool(x)

        x = self.encoder.layer1(x)
        x = self.encoder.layer2(x)
        x = self.encoder.layer3(x)
        feature_map = self.encoder.layer4(x) # [B, 2048/512, 7, 7]
        
        # Standard Global Pooling
        feat_global = self.encoder.avgpool(feature_map) # [B, Dim, 1, 1]
        feat_global = torch.flatten(feat_global, 1) # [B, Dim]
        
        z = self.proj(feat_global)
        z = F.normalize(z, dim=1)
        
        if return_feature_map:
            return feat_global, z, feature_map
        return feat_global, z

# =========================================================
# Losses
# =========================================================

class AttentionAlignmentLoss(nn.Module):
    """
    Innovation 1: VLM-Guided Attention Alignment with Teacher Correction
    Now supports Dynamic Hybrid Target:
    Target = (1 - alpha) * VLM_Heatmap + alpha * Teacher_Attention
    """
    def __init__(self):
        super().__init__()

    def forward(self, feature_map_student, feature_map_teacher, gt_heatmap, alpha=0.0):
        """
        feature_map_student: [B, C, H, W] from Model Q
        feature_map_teacher: [B, C, H, W] from Model K (Teacher)
        gt_heatmap: [B, 1, H_img, W_img] from VLM (Noisy)
        alpha: scalar 0.0 -> 1.0 (Trust Teacher more as alpha increases)
        """
        # 1. Generate Student Attention
        attn_student = torch.mean(feature_map_student, dim=1, keepdim=True) # [B, 1, H, W]
        attn_student = F.relu(attn_student)
        B, _, H, W = attn_student.shape
        
        # Normalize Student (for comparison)
        attn_s_flat = attn_student.view(B, -1)
        # Avoid div by zero
        s_min, _ = attn_s_flat.min(dim=1, keepdim=True)
        s_max, _ = attn_s_flat.max(dim=1, keepdim=True)
        attn_student_norm = (attn_s_flat - s_min) / (s_max - s_min + 1e-8)
        attn_student_norm = attn_student_norm.view(B, 1, H, W)

        # 2. Build Target
        # Part A: VLM Heatmap (Downsampled)
        if gt_heatmap is not None:
            gt_small = F.interpolate(gt_heatmap, size=(H, W), mode='bilinear', align_corners=False)
        else:
            gt_small = torch.zeros_like(attn_student) # Should not happen if use_heatmap is True

        # Part B: Teacher Attention
        # Teacher is Momentum Encoder, supposedly more stable/robust after some epochs
        attn_teacher = torch.mean(feature_map_teacher, dim=1, keepdim=True)
        attn_teacher = F.relu(attn_teacher)
        
        # Normalize Teacher
        attn_t_flat = attn_teacher.view(B, -1)
        t_min, _ = attn_t_flat.min(dim=1, keepdim=True)
        t_max, _ = attn_t_flat.max(dim=1, keepdim=True)
        attn_teacher_norm = (attn_t_flat - t_min) / (t_max - t_min + 1e-8)
        attn_teacher_norm = attn_teacher_norm.view(B, 1, H, W)
        
        # 3. Hybrid Target
        # detach() because we don't backprop into Teacher or GT
        target_hybrid = (1.0 - alpha) * gt_small + alpha * attn_teacher_norm
        target_hybrid = target_hybrid.detach()
        
        # 4. Asymmetric Loss: ReLU(Target - Student)
        # Penalize if Student is LOWER than Target (Under-attention)
        diff = target_hybrid - attn_student_norm
        loss = torch.mean(F.relu(diff) ** 2)
        
        return loss



class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
             # alpha weighting
             if isinstance(self.alpha, (float, int)):
                 alpha_t = self.alpha
             else:
                 alpha_t = self.alpha[targets]
             focal_loss = alpha_t * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


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
    # # 1. Similarity with Prototypes (Positives + Class Negatives) => [B, Num_Classes]
    # logits_proto = torch.matmul(features, prototypes.T)
    
    # # 2. Similarity with Queue (Negatives) => [B, Queue_Size]
    # logits_queue = torch.matmul(features, queue.queue.clone().detach().T)
    
    # # 3. Combine: Denominator = exp(Proto_Pos) + sum(exp(Proto_Neg)) + sum(exp(Queue_Neg))
    # # Concatenate along class dimension
    # logits = torch.cat([logits_proto, logits_queue], dim=1) / temperature
    
    # # F.cross_entropy will handle the log-softmax. 
    # # Since labels are in [0, Num_Classes-1], they naturally match the 'logits_proto' part.
    # # The 'logits_queue' part merely adds more negatives to the denominator.
    # loss = F.cross_entropy(logits, labels)
    logits_proto = torch.matmul(features, prototypes.T) / temperature
    loss = F.cross_entropy(logits_proto, labels)
    return loss

def cross_entropy_with_logit_compensation(logits, targets, class_freq, tau=1.0):
    if class_freq is None: return F.cross_entropy(logits, targets)
    prior = class_freq / class_freq.sum()
    log_prior = torch.log(prior + 1e-8).to(logits.device)
    adjusted_logits = logits - tau * log_prior
    return F.cross_entropy(adjusted_logits, targets)

# =========================================================
# Trainer
# =========================================================

class FPaCoTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.use_heatmap = not args.no_heatmap
        self.input_channels = 4 if self.use_heatmap else 3
        print(f"FPaCo Config: Heatmap={self.use_heatmap}, Input Channels={self.input_channels}")
        
        self.train_dataset = HeatmapBPaCoDataset(
            args.dataset, split='train', image_size=args.image_size, sigma=args.sigma,
            combine_train_val=args.combine_train_val, use_heatmap=self.use_heatmap
        )
        self.val_dataset = HeatmapBPaCoDataset(
            args.dataset, split='test', image_size=args.image_size, sigma=args.sigma,
            use_heatmap=self.use_heatmap
        )
        
        self.num_classes = len(self.train_dataset.classes)
        print(f"Num Classes: {self.num_classes}")
        
        # Model
        self.model_q = FPaCoResNet(backbone=args.backbone, num_classes=self.num_classes, input_channels=self.input_channels).to(self.device)
        self.model_k = FPaCoResNet(backbone=args.backbone, num_classes=self.num_classes, input_channels=self.input_channels).to(self.device)
        
        for param_q, param_k in zip(self.model_q.parameters(), self.model_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
            
        self.classifier = nn.Linear(self.model_q.feat_dim * 2, self.num_classes).to(self.device)
        self.C1 = nn.Parameter(torch.randn(self.num_classes, 128).to(self.device))
        self.queue = MoCoQueue(dim=128, size=args.queue_size, device=self.device)
        
        params = list(self.model_q.parameters()) + list(self.classifier.parameters()) + [self.C1]
        self.optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.epochs)
        
        self.class_freq = torch.zeros(self.num_classes).to(self.device)
        for s in self.train_dataset.samples: self.class_freq[s[1]] += 1
        
        # New Losses
        self.criterion_align = AttentionAlignmentLoss()
        # self.criterion_disen removed
        
        # Main Classification Criterion
        if args.focal_gamma > 0.0:
            print(f"Using Focal Loss with gamma={args.focal_gamma}")
            self.criterion_ce = FocalLoss(gamma=args.focal_gamma)
        else:
            print(f"Using Standard Cross Entropy (Logit Compensation: tau={args.tau})")
            self.criterion_ce = None # formatting helper? No, we'll handle in loop


    def run(self):
        train_loader = DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(self.val_dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=4)
        
        best_f1 = 0.0
        
        print("Starting FPaCo Training...")
        for epoch in range(1, self.args.epochs + 1):
            self.model_q.train()
            train_loss = 0
            
            for i, batch in enumerate(train_loader):
                v1 = batch['v1'].to(self.device).float()
                v2 = batch['v2'].to(self.device).float()
                labels = batch['label'].to(self.device)

                # --- Fallback Strategy (Pre-Forward): If VLM gives no attention, use WHOLE IMAGE ---
                # Check per sample: if sum(heatmap) is effectively 0
                # if self.use_heatmap:
                #     heat_channel = v1[:, 3:4, :, :] 
                #     heat_sums = heat_channel.view(v1.size(0), -1).sum(dim=1)
                #     empty_mask = (heat_sums < 1e-6).view(-1, 1, 1, 1) # [B, 1, 1, 1]
                    
                #     if empty_mask.any():
                #         # Modify v1 IN-PLACE before Forward Pass
                #         # This generates a new version of the tensor but before it's used in computation graph
                #         v1[:, 3:4, :, :] = torch.where(empty_mask, torch.ones_like(heat_channel), heat_channel)
                
                # Input v1 is 4-ch. Extract heatmap (Channel 3) - AFTER Fallback modification
                if self.use_heatmap:
                     gt_heatmap = v1[:, 3:4, :, :] 
                else:
                     gt_heatmap = None
                
                # 1. Forward
                # Get features AND feature map for alignment
                feat_q, z_q, map_q = self.model_q(v1, return_feature_map=True)
                with torch.no_grad():
                    # Teacher needs to return Feature Map now for correction
                    feat_k, z_k, map_k = self.model_k(v2, return_feature_map=True)

                
                feat_cat = torch.cat([feat_q, feat_k], dim=1)
                logits = self.classifier(feat_cat)
                
                
                
                # 2. Base Losses
                # Logit Compensation Pre-processing
                if self.args.tau > 0 and self.class_freq is not None:
                      prior = self.class_freq / self.class_freq.sum()
                      log_prior = torch.log(prior + 1e-8).to(logits.device)
                      # Apply adjustment
                      final_logits = logits - self.args.tau * log_prior
                else:
                      final_logits = logits


                if self.criterion_ce is not None:
                     loss_ce = self.criterion_ce(final_logits, labels)
                else:
                     # Standard path (helper does adjustment internally)
                     loss_ce = cross_entropy_with_logit_compensation(logits, labels, self.class_freq, tau=self.args.tau)

                     
                loss_con = compute_contrastive_loss(z_q, labels, self.queue, self.C1, temperature=self.args.temperature)
                
                # 3. Advanced Losses (Only if Heatmap is being used)
                loss_guide = torch.tensor(0.0).to(self.device)
                
                if self.use_heatmap:
                    
                    # --- Innovation 1: Attention Alignment (Dynamic & Confidence-Adaptive) ---
                    # 1. Base Schedule (Global trust in Teacher increases over time)
                    max_alpha = self.args.max_alpha # e.g. 0.8
                    alpha_base = (epoch / self.args.epochs) * max_alpha
                    alpha_base = min(max_alpha, alpha_base)
                    
                    # 2. Confidence-Adaptive Adjustment (Per Sample)
                    # Measure VLM Confidence: Max value in heatmap (0.0 to 1.0)
                    # [B, 1, H, W] -> [B]
                    if gt_heatmap is not None:
                         heatmap_max = gt_heatmap.view(gt_heatmap.size(0), -1).max(dim=1).values
                    else:
                         heatmap_max = torch.zeros(v1.size(0)).to(self.device)
                         
                    # Adaptive Logic:
                    # If heatmap_max is HIGH (1.0) -> Trust VLM (Use alpha_base)
                    # If heatmap_max is LOW (0.0)  -> Trust Teacher (Boost alpha towards 1.0)
                    # Formula: alpha = alpha_base + (1 - confidence) * (1 - alpha_base)
                    # Explanation:
                    #   Conf=1.0 => alpha = alpha_base (Normal schedule)
                    #   Conf=0.0 => alpha = alpha_base + (1 - alpha_base) = 1.0 (Full Teacher)
                    
                    alpha_adaptive = alpha_base + (1.0 - heatmap_max) * (1.0 - alpha_base)
                    
                    # Reshape for broadcasting [B, 1, 1, 1]
                    alpha_adaptive = alpha_adaptive.view(-1, 1, 1, 1)

                    loss_guide = self.criterion_align(map_q, map_k, gt_heatmap, alpha=alpha_adaptive)
                
                # Total Loss
                loss = (loss_ce 
                        + self.args.beta * loss_con 
                        + 0.1 * loss_guide)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                momentum_update(self.model_k, self.model_q)
                with torch.no_grad():
                    self.queue.enqueue(z_k, labels)
                    
                train_loss += loss.item()
                
            self.scheduler.step()
            avg_loss = train_loss / len(train_loader)
            
            # Validation
            if epoch % self.args.val_interval == 0:
                acc, f1 = self.validate(val_loader)
                print(f"Epoch {epoch}: Loss={avg_loss:.4f}, Guide={loss_guide:.4f}, MeanAlpha={alpha_adaptive.mean().item():.2f}, Val Acc={acc:.4f}, Val F1={f1:.4f}")
                
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

    @torch.no_grad()
    def validate(self, loader):
        self.model_q.eval()
        preds, targets = [], []
        for batch in loader:
            v1 = batch['v1'].to(self.device).float()
            labels = batch['label'].to(self.device)
            
            feat, z = self.model_q(v1)
            feat_cat = torch.cat([feat, feat], dim=1)
            logits = self.classifier(feat_cat)
            
            pred = torch.argmax(logits, dim=1)
            preds.extend(pred.cpu().numpy())
            targets.extend(labels.cpu().numpy())
            
        return accuracy_score(targets, preds), f1_score(targets, preds, average='macro')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='results')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--backbone', type=str, default='resnet18')
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--sigma', type=int, default=30)
    parser.add_argument('--beta', type=float, default=2.0)
    parser.add_argument('--tau', type=float, default=1.0)
    parser.add_argument('--focal-gamma', type=float, default=2.0, help="Gamma for Focal Loss. If > 0, replaces Logit Compensation CE.")
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--queue-size', type=int, default=8192)
    parser.add_argument('--val-interval', type=int, default=1)
    parser.add_argument('--max-alpha', type=float, default=0.5, help="Max teacher attention trust alpha")
    parser.add_argument('--combine-train-val', action='store_true', default=True)
    parser.add_argument('--no-heatmap', action='store_true', help="Disable heatmap channel (use RGB only)")
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    trainer = FPaCoTrainer(args)
    trainer.run()
