#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BPaCo (Balanced Prototype and Contrastive Learning) - Heatmap Early Fusion Version
核心思想:
1.  前融合 (Early Fusion): 将关键点检测结果转换为"热力图" (Heatmap) 通道。
2.  多通道输入: 输入 = [灰度图, 灰度图, 热力图] (3通道)，完美适配 Pretrained ResNet。
3.  利用 CNN 归纳偏置: 复用 CNN 强大的空间特征提取能力，无需 Transformer。
"""

import os
import sys
import random
import json
import csv
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

from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

# 设置随机种子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# =========================================================
# 模块 1: 形态学增强 (保持不变)
# =========================================================
class FingerprintMorphologyTransform:
    def __init__(self, kernel_size=3):
        self.kernel_size = kernel_size

    def __call__(self, img_tensor):
        # img_tensor: [C, H, W], range [0, 1] or result of ToTensor
        # 转为 numpy uint8 [H, W]
        img_np = (img_tensor.squeeze().numpy() * 255).astype(np.uint8)
        
        # 随机选择操作: 腐蚀, 膨胀, 开运算, 闭运算
        op = random.choice([cv2.MORPH_OPEN, cv2.MORPH_CLOSE, cv2.MORPH_ERODE, cv2.MORPH_DILATE])
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.kernel_size, self.kernel_size))
        
        # 针对不同操作设置随机迭代次数
        iterations = random.randint(1, 2)
        
        if op == cv2.MORPH_ERODE or op == cv2.MORPH_DILATE:
            result = cv2.morphologyEx(img_np, op, kernel, iterations=iterations)
        else:
            result = cv2.morphologyEx(img_np, op, kernel)
            
        # 随机二值化
        if random.random() < 0.5:
             _, result = cv2.threshold(result, 127, 255, cv2.THRESH_BINARY)

        # 随机 Top-Hat / Black-Hat (提取纹理细节)
        if random.random() < 0.3:
             op2 = random.choice([cv2.MORPH_TOPHAT, cv2.MORPH_BLACKHAT])
             result2 = cv2.morphologyEx(img_np, op2, kernel)
             # 融合
             result = cv2.addWeighted(result, 0.7, result2, 0.3, 0)
             
        return torch.from_numpy(result.astype(np.float32) / 255.0).unsqueeze(0)


# =========================================================
# 模块 2: 热力图数据集 (Heatmap Early Fusion)
# =========================================================

class HeatmapBPaCoDataset(Dataset):
    """
    输入: 图像 + JSON检测 + CSV统计特征
    输出: 3通道张量 [Gray, Gray, Heatmap], stat_features [8]
    """
    def __init__(self, dataset_dir, keypoint_features_file, split='train', image_size=384, sigma_center=15, sigma_delta=15):
        self.dataset_dir = Path(dataset_dir)
        self.split = split
        self.image_size = image_size
        self.sigma_center = sigma_center
        self.sigma_delta = sigma_delta
        
        # 加载关键点特征CSV
        self.keypoint_features = {}
        if os.path.exists(keypoint_features_file):
            with open(keypoint_features_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row['split'] == split:
                        self.keypoint_features[row['image_path']] = {
                            'num_kp1': int(row['num_kp1']),
                            'num_kp2': int(row['num_kp2']),
                            'is_left_hand': int(row['is_left_hand']),
                            'is_right_hand': int(row['is_right_hand']),
                            'is_hard_sample': int(row['is_hard_sample']),
                            'kp1_between_kp2': int(row['kp1_between_kp2']),
                            'kp1_left_of_kp2': int(row['kp1_left_of_kp2']),
                            'kp1_right_of_kp2': int(row['kp1_right_of_kp2']),
                        }
        else:
            print(f"Warning: CSV file {keypoint_features_file} not found. Stat features will be zeros.")
        
        # 加载样本列表
        self.samples = []
        split_dir = self.dataset_dir / split
        
        # 假设 19 类
        for class_id in range(19):
            class_dir = split_dir / f'class_{class_id}'
            if not class_dir.exists():
                continue
            
            # 遍历图像
            for ext in ['*.png', '*.jpg', '*.jpeg']:
                for img_path in class_dir.glob(ext):
                    # 获取相对路径以匹配CSV
                    try:
                        rel_path = str(img_path.relative_to(self.dataset_dir))
                    except ValueError:
                        # Fallback if path structure is different
                        rel_path = img_path.name 

                    if rel_path in self.keypoint_features:
                         self.samples.append({
                            'image_path': str(img_path),
                            'class_id': class_id,
                            'rel_path': rel_path
                        })
                    else:
                        # Fallback for images not in CSV (should verify if this is desired)
                        # For now, include them but use dummy stats
                         self.samples.append({
                            'image_path': str(img_path),
                            'class_id': class_id,
                            'rel_path': None
                        })

        print(f"[{split}] 数据集加载完毕: {len(self.samples)} 张样本")
        
        # 基础变换
        self.morph = FingerprintMorphologyTransform()
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet Norm

    def _generate_heatmap(self, h, w, detections, target_class=None, base_sigma=15):
        """生成高斯热力图
        Args:
            target_class: 'center_point' or 'delta_point'. If None, include all.
        """
        heatmap = np.zeros((h, w), dtype=np.float32)
        
        if not detections:
            return heatmap
            
        # 生成网格
        xx, yy = np.meshgrid(np.arange(w), np.arange(h))
        
        for det in detections:
            # Filter by class
            if target_class is not None:
                if det.get('class_name') != target_class:
                    continue
                    
            bbox = det['bbox'] # [x1, y1, x2, y2]
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0
            conf = det.get('confidence', 1.0)
            
            # 策略2: 置信度自适应 Sigma
            sigma = base_sigma * (1.0 + 1.0 * (1.0 - conf))
            
            # 距离平方
            dist_sq = (xx - cx)**2 + (yy - cy)**2
            # Gaussian: exp(-dist / (2*sigma^2))
            blob = np.exp(-dist_sq / (2 * sigma**2))
            
            # 乘以置信度叠加
            heatmap = np.maximum(heatmap, blob * conf)
            
        return heatmap

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = sample['image_path']
        label = sample['class_id']
        
        # 0. Load Stat Features
        if sample['rel_path'] and sample['rel_path'] in self.keypoint_features:
            kp_data = self.keypoint_features[sample['rel_path']].copy()
            stat_features = np.array([
                kp_data['num_kp1'],
                kp_data['num_kp2'],
                kp_data['is_left_hand'],
                kp_data['is_right_hand'],
                kp_data['is_hard_sample'],
                kp_data['kp1_between_kp2'],
                kp_data['kp1_left_of_kp2'],
                kp_data['kp1_right_of_kp2'],
            ], dtype=np.float32)
        else:
             stat_features = np.zeros(8, dtype=np.float32)

        # 1. 加载图像 (Gray)
        try:
            pil_img = Image.open(img_path).convert('L')
            original_w, original_h = pil_img.size
            img_np = np.array(pil_img)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return dummy
            return self.__getitem__(random.randint(0, len(self)-1))

        # 2. 加载 JSON -> 生成 Heatmap
        json_path = Path(img_path).with_suffix('.json')
        detections = []
        if json_path.exists():
            try:
                with open(json_path, 'r') as f:
                    detections = json.load(f).get('detections', [])
            except:
                pass
        
        # 策略1: 语义通道分离
        # Channel 1: Center Points
        heatmap_center = self._generate_heatmap(original_h, original_w, detections, target_class='center_point', base_sigma=self.sigma_center)
        # Channel 2: Delta Points
        heatmap_delta = self._generate_heatmap(original_h, original_w, detections, target_class='delta_point', base_sigma=self.sigma_delta)
        
        # 3. 组合成 3 通道 (H, W, 3) -> PIL
        # Ch0: Gray (Texture)
        # Ch1: Center Heatmap (Geometry 1)
        # Ch2: Delta Heatmap (Geometry 2)
        
        heatmap_c_uint8 = (heatmap_center * 255).astype(np.uint8)
        heatmap_d_uint8 = (heatmap_delta * 255).astype(np.uint8)
        
        combined_np = np.stack([img_np, heatmap_c_uint8, heatmap_d_uint8], axis=-1) # [H, W, 3]
        combined_pil = Image.fromarray(combined_np) # Let Pillow infer RGB
        
        # 4. 数据增强
        if self.split == 'train':
            # Manual Flip Logic
            if random.random() < 0.5:
                combined_pil = TF.hflip(combined_pil)
                # Swap Left/Right features
                stat_features[[2, 3]] = stat_features[[3, 2]]
                stat_features[[6, 7]] = stat_features[[7, 6]]
            
            # Rotation (stat features invariant to small rotation)
            if random.random() < 0.5:
                 angle = random.uniform(-15, 15)
                 combined_pil = TF.rotate(combined_pil, angle)

            transform = T.Compose([
                T.Resize((self.image_size, self.image_size)),
                # No Flip here!
                T.ToTensor(), 
            ])
            
            # 视图1
            v1 = transform(combined_pil)
            # 视图2 (Same geometry to match stats? Yes. BPaCo usually augments twice differently.
            # But if we did different flip for v2, we'd need different stat_features for v2.
            # For simplicity, we share the flip state for both views, but maybe add noise to v2?)
            # BPaCo standard: Two distinct views. 
            # If v2 has DIFFERENT flip, v2 needs DIFFERENT stat vector.
            # To handle this: Generate two random flip states?
            # Complexity increases.
            # Decision: Use SAME geometry (Flip) for both views, but different photometric/morph/noise.
            # This is "Weak/Strong" augmentation paradigm which is fine.
            v2 = transform(combined_pil)
            
            # 对 v1, v2 进行 Normalize
            v1 = self.normalize(v1)
            v2 = self.normalize(v2)
            
            stat_features = torch.from_numpy(stat_features)
            
            return {
                'v1': v1,
                'v2': v2,
                'stat_features': stat_features,
                'label': label
            }
        
        else:
            # Val transform
            transform = T.Compose([
                T.Resize((self.image_size, self.image_size)),
                T.ToTensor(),
                self.normalize
            ])
            img_t = transform(combined_pil)
            stat_features = torch.from_numpy(stat_features)
            
            return {
                'v1': img_t,
                'v2': img_t,
                'stat_features': stat_features,
                'label': label
            }

# =========================================================
# 模块 3: 模型 (Standard ResNet Wrapper)
# =========================================================

# =========================================================
# 模块 3: 模型 (Standard ResNet Wrapper + Stat Fusion)
# =========================================================

class BpacoResNet(nn.Module):
    def __init__(self, backbone='resnet18', num_classes=19, proj_dim=128, pretrained=True):
        super().__init__()
        
        # 标准 ResNet
        model_fun = getattr(models, backbone)
        if pretrained:
            weights = 'DEFAULT'
        else:
            weights = None
        self.encoder = model_fun(weights=weights)
        
        # 获取 FC 输入维度
        fc_in_features = self.encoder.fc.in_features
        self.img_feat_dim = fc_in_features
        
        # 移除原始 FC
        self.encoder.fc = nn.Identity()
        
        # Stat Feature Projection
        self.stat_input_dim = 8
        self.stat_proj_dim = 32
        self.stat_proj = nn.Linear(self.stat_input_dim, self.stat_proj_dim)
        
        # Combined Feature Dim
        self.feat_dim = self.img_feat_dim + self.stat_proj_dim
        
        # BPaCo Projection Head (Processing Combined Features)
        self.proj = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.feat_dim, proj_dim)
        )

    def forward(self, x, stat_features=None):
        # x: [B, 3, H, W]
        # stat_features: [B, 8]
        
        f_img = self.encoder(x) # [B, 512]
        
        if stat_features is not None:
            f_stat = self.stat_proj(stat_features) # [B, 32]
        else:
            f_stat = torch.zeros(f_img.size(0), self.stat_proj_dim, device=f_img.device)
            
        feat = torch.cat([f_img, f_stat], dim=1) # [B, 544]
        
        # Projection
        z = self.proj(feat)
        z = F.normalize(z, dim=1)
        
        return feat, z

# =========================================================
# 模块 4: 辅助函数 (Loss, Queue, Etc) -> 复用原代码逻辑
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
        
        # 如果溢出，替换队首
        if ptr + batch_size > self.size:
            batch_size = self.size - ptr # 只填剩下的
            
        self.queue[ptr:ptr+batch_size] = keys[:batch_size]
        self.labels[ptr:ptr+batch_size] = labels[:batch_size]
        
        self.ptr = (ptr + batch_size) % self.size

def momentum_update(model_k, model_q, m=0.999):
    for param_k, param_q in zip(model_k.parameters(), model_q.parameters()):
        param_k.data = param_k.data * m + param_q.data * (1. - m)

def compute_LBPaCo(features, labels, queue, C1, classifier_weight, mapper, temperature=0.07, device='cuda'):
    # 取样 (Batch)
    batch_size = features.shape[0]
    
    # 1. 这种实现需要比较复杂的 Mask 逻辑，为简化，我们使用 BPaCo 的核心 Loss:
    # L_contrast = -log ( exp(q * k+) / Sum(exp(q * k)) )
    # Positive pairs: 原型 (C1) or Queue中同类.
    # 这里我们只实现基础的 Supervised Contrastive Loss
    
    # 简化版 BPaCo Loss: 
    # Query 与 同类原型 C1 靠近，与异类 C1 远离
    
    # C1: [NumClasses, Dim]
    # features: [B, Dim]
    
    # Cosine Sim
    logits = torch.matmul(features, C1.T) / temperature # [B, 19]
    
    # Log softmax
    loss = F.cross_entropy(logits, labels)
    return loss


def cross_entropy_with_logit_compensation(logits, targets, class_freq, tau=1.0):
    # Logit Adjustment
    # logit = logit + tau * log(P(y))
    # P(y) = freq / sum(freq)
    
    prior = class_freq / class_freq.sum()
    log_prior = torch.log(prior + 1e-8).to(logits.device)
    
    adjusted_logits = logits + tau * log_prior
    
    return F.cross_entropy(adjusted_logits, targets)

# =========================================================
# 模块 5: 训练主流程
# =========================================================

class BPaCoHeatmapTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 数据集
        self.train_dataset = HeatmapBPaCoDataset(
            args.dataset, args.keypoint_features, split='train', image_size=args.image_size, 
            sigma_center=args.sigma_center, sigma_delta=args.sigma_delta
        )
        self.val_dataset = HeatmapBPaCoDataset(
            args.dataset, args.keypoint_features, split='val', image_size=args.image_size,
            sigma_center=args.sigma_center, sigma_delta=args.sigma_delta
        ) 
        
        self.num_classes = 19
        
        # 模型
        print(f"Building Heatmap Model (Backbone: {args.backbone})...")
        self.model_q = BpacoResNet(backbone=args.backbone, num_classes=self.num_classes).to(self.device)
        self.model_k = BpacoResNet(backbone=args.backbone, num_classes=self.num_classes).to(self.device)
        
        # 初始化 momentum model
        for param_q, param_k in zip(self.model_q.parameters(), self.model_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
            
        # 分类器
        self.classifier = nn.Linear(self.model_q.feat_dim * 2, self.num_classes).to(self.device) # Concat(q, k)
        
        # 原型 C1 (Class Centers)
        self.C1 = nn.Parameter(torch.randn(self.num_classes, 128).to(self.device)) # Proj dim
        
        # 队列
        self.queue = MoCoQueue(dim=128, size=args.queue_size, device=self.device)
        
        # 优化器
        self.optimizer = torch.optim.SGD(
            list(self.model_q.parameters()) + list(self.classifier.parameters()) + [self.C1],
            lr=args.lr, momentum=0.9, weight_decay=1e-4
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.epochs)
        
        # Mapper (C2) - Simplified as Identity for now or reuse C1
        self.mapper = None 

    def run(self):
        # 类别频率
        print("Calculating class frequencies...")
        class_freq = torch.zeros(self.num_classes).to(self.device)
        # 简单估算
        for sample in self.train_dataset.samples:
            class_freq[sample['class_id']] += 1
        
        train_loader = DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(self.val_dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=4)
        
        print("Start Training...")
        best_f1 = 0
        
        # Training history
        history = {'train_loss': [], 'val_acc': [], 'val_f1': []}
        
        for epoch in range(1, self.args.epochs + 1):
            self.model_q.train()
            train_loss = []
            
            for batch in train_loader:
                v1 = batch['v1'].to(self.device)
                v2 = batch['v2'].to(self.device)
                stat = batch['stat_features'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward
                feat_q, z_q = self.model_q(v1, stat)
                
                with torch.no_grad():
                     feat_k, z_k = self.model_k(v2, stat)
                     
                # Classifier (Concat)
                feat_cat = torch.cat([feat_q, feat_k], dim=1)
                logits = self.classifier(feat_cat)
                
                # Loss 1: CE
                loss_ce = cross_entropy_with_logit_compensation(logits, labels, class_freq, tau=self.args.tau)
                
                # Loss 2: Contrastive
                loss_con = compute_LBPaCo(z_q, labels, self.queue, self.C1, None, None, temperature=self.args.temperature, device=self.device)
                
                loss = loss_ce + self.args.beta * loss_con
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # Momentum Update
                momentum_update(self.model_k, self.model_q)
                
                # Queue Update
                with torch.no_grad():
                    self.queue.enqueue(z_k, labels)
                    
                train_loss.append(loss.item())
                
            self.scheduler.step()
            avg_loss = sum(train_loss)/len(train_loss)
            history['train_loss'].append(avg_loss)
            
            # Validation
            if epoch % self.args.val_interval == 0:
                acc, f1 = self.validate(val_loader)
                print(f"Epoch {epoch}: Loss={avg_loss:.4f}, Val Acc={acc:.4f}, Val F1={f1:.4f}")
                
                history['val_acc'].append(acc)
                history['val_f1'].append(f1)
                
                if f1 > best_f1:
                    best_f1 = f1
                    torch.save(self.model_q.state_dict(), os.path.join(self.args.output_dir, "best_heatmap_model.pth"))
            else:
                 print(f"Epoch {epoch}: Loss={avg_loss:.4f}")
                 
        # Save history
        with open(os.path.join(self.args.output_dir, "training_history.json"), "w") as f:
            json.dump(history, f, indent=2)

        # Final Evaluation and Plotting
        print("Loading best model for final evaluation...")
        best_model_path = os.path.join(self.args.output_dir, "best_heatmap_model.pth")
        if os.path.exists(best_model_path):
            self.model_q.load_state_dict(torch.load(best_model_path))
        
        targets, preds, probs = self.get_predictions(val_loader)
        
        print("Generating plots...")
        self.plot_confusion_matrix(targets, preds, os.path.join(self.args.output_dir, "confusion_matrix.png"))
        self.plot_roc_curve(targets, probs, os.path.join(self.args.output_dir, "roc_curve.png"))
        print(f"Results saved to {self.args.output_dir}")

    @torch.no_grad()
    def get_predictions(self, loader):
        self.model_q.eval()
        preds, targets, probs = [], [], []
        for batch in loader:
            img = batch['v1'].to(self.device)
            stat = batch['stat_features'].to(self.device)
            labels = batch['label'].to(self.device)
            
            feat, _ = self.model_q(img, stat)
            feat_cat = torch.cat([feat, feat], dim=1)
            logits = self.classifier(feat_cat)
            prob = F.softmax(logits, dim=1)
            
            pred = torch.argmax(logits, dim=1)
            
            preds.extend(pred.cpu().numpy())
            targets.extend(labels.cpu().numpy())
            probs.extend(prob.cpu().numpy())
            
        return np.array(targets), np.array(preds), np.array(probs)

    def plot_confusion_matrix(self, targets, preds, save_path):
        cm = confusion_matrix(targets, preds)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def plot_roc_curve(self, targets, probs, save_path):
        # One-vs-Rest ROC
        from sklearn.metrics import roc_curve, auc
        from sklearn.preprocessing import label_binarize
        
        n_classes = self.num_classes
        y_test = label_binarize(targets, classes=range(n_classes))
        
        plt.figure(figsize=(10, 8))
        
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            if np.sum(y_test[:, i]) > 0: # Avoid error if class not present
                fpr[i], tpr[i], _ = roc_curve(y_test[:, i], probs[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
                plt.plot(fpr[i], tpr[i], lw=1, alpha=0.3) # label=f'Class {i} (area = {roc_auc[i]:.2f})'
        
        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), probs.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        
        plt.plot(fpr["micro"], tpr["micro"],
                 label='micro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["micro"]),
                 color='deeppink', linestyle=':', linewidth=4)
                 
        # Compute macro-average ROC curve and ROC area
        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes) if i in fpr]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        valid_classes = 0
        for i in range(n_classes):
             if i in fpr:
                mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
                valid_classes += 1
        
        if valid_classes > 0:
            mean_tpr /= valid_classes
            fpr["macro"] = all_fpr
            tpr["macro"] = mean_tpr
            roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
            
            plt.plot(fpr["macro"], tpr["macro"],
                     label='macro-average ROC curve (area = {0:0.2f})'
                           ''.format(roc_auc["macro"]),
                     color='navy', linestyle=':', linewidth=4)

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (Multi-class)')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    @torch.no_grad()
    def validate(self, loader):
        self.model_q.eval()
        preds, targets = [], []
        for batch in loader:
            img = batch['v1'].to(self.device)
            stat = batch['stat_features'].to(self.device)
            labels = batch['label'].to(self.device) # Don't forget label
            
            feat, _ = self.model_q(img, stat)
            # Eval use feat cat feat
            feat_cat = torch.cat([feat, feat], dim=1)
            logits = self.classifier(feat_cat)
            
            pred = torch.argmax(logits, dim=1)
            preds.extend(pred.cpu().numpy())
            targets.extend(labels.cpu().numpy())
            
        return accuracy_score(targets, preds), f1_score(targets, preds, average='macro')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--keypoint-features', type=str, default='keypoint_features.csv')
    parser.add_argument('--backbone', type=str, default='resnet18')
    parser.add_argument('--image-size', type=int, default=384)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--beta', type=float, default=2.0)
    parser.add_argument('--tau', type=float, default=1.0)
    parser.add_argument('--val-interval', type=int, default=5)
    parser.add_argument('--output-dir', type=str, default='results/bpaco_heatmap')
    parser.add_argument('--sigma-center', type=int, default=10, help='Base sigma for center points')
    parser.add_argument('--sigma-delta', type=int, default=10, help='Base sigma for delta points')
    parser.add_argument('--queue-size', type=int, default=8192)
    parser.add_argument('--temperature', type=float, default=0.10)
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    trainer = BPaCoHeatmapTrainer(args)
    trainer.run()

if __name__ == '__main__':
    main()
