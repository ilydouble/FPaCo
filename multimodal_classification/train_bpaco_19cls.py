#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BPaCo (Balanced Prototype and Contrastive Learning) 训练脚本 - 19类版本

核心思想:
1. 对比学习 + 分类学习联合训练
2. 三种原型: 批内均值、可学习中心C1、分类器权重映射C2
3. Logit Compensation处理长尾分布
4. 双视图增强 (v1: 随机增强, v2: 形态学增强)
"""

import os
import sys
import random
import json
import argparse
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
import torchvision.models as models

from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

# 设置随机种子
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# 设置HuggingFace镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


# =========================================================
# 模块 1: 形态学增强 (从原始代码复用)
# =========================================================

class FingerprintMorphologyTransform:
    """
    指纹形态学增强: 顶帽 -> 二值化 -> 黑帽 -> 开运算 -> 取反
    """
    def __init__(
        self,
        kernel_tophat: int = 15,
        kernel_blackhat: int = 25,
        kernel_open: int = 3,
        block_size: int = 21,
        C: int = 8,
    ):
        self.kernel_tophat = kernel_tophat
        self.kernel_blackhat = kernel_blackhat
        self.kernel_open = kernel_open
        self.block_size = block_size
        self.C = C

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3 or x.size(0) != 1:
            raise ValueError(f"期望输入 (1,H,W), 得到 {tuple(x.shape)}")

        img = x.squeeze(0).cpu().numpy()
        img_u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)

        # 1) 顶帽
        k_top = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.kernel_tophat, self.kernel_tophat)
        )
        tophat = cv2.morphologyEx(img_u8, cv2.MORPH_TOPHAT, k_top)
        if tophat.max() > 0:
            tophat_enh = cv2.normalize(tophat, None, 0, 255, cv2.NORM_MINMAX)
        else:
            tophat_enh = tophat

        # 2) 自适应阈值
        binary = cv2.adaptiveThreshold(
            tophat_enh, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            self.block_size, self.C
        )

        # 3) 黑帽
        k_bh = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.kernel_blackhat, self.kernel_blackhat)
        )
        blackhat = cv2.morphologyEx(img_u8, cv2.MORPH_BLACKHAT, k_bh)
        if blackhat.max() > 0:
            blackhat = cv2.normalize(blackhat, None, 0, 255, cv2.NORM_MINMAX)

        blackhat_bin = cv2.adaptiveThreshold(
            blackhat, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            self.block_size, self.C
        )

        # 4) 开运算
        k_open = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.kernel_open, self.kernel_open)
        )
        blackhat_open = cv2.morphologyEx(blackhat_bin, cv2.MORPH_OPEN, k_open)

        # 5) 取反 × 二值图
        inv_blackhat = 255 - blackhat_open
        result = cv2.bitwise_and(binary, inv_blackhat)

        # 转回 [0,1] 张量
        result_f = result.astype(np.float32) / 255.0
        result_t = torch.from_numpy(result_f).unsqueeze(0)

        return result_t


# =========================================================
# 模块 2: 数据集 (适配19类分类数据集)
# =========================================================

class MultiModalBPaCoDataset(Dataset):
    """
    19类手势分类数据集 + 双视图增强

    数据集结构:
    classification_dataset/
        train/
            class_0/
                xxx.png
            class_1/
                xxx.png
            ...
        val/
            ...
    """
    def __init__(self, dataset_dir, split='train', image_size=224):
        self.dataset_dir = Path(dataset_dir)
        self.split = split
        self.image_size = image_size

        # 加载样本
        self.samples = []
        split_dir = self.dataset_dir / split

        for class_id in range(19):
            class_dir = split_dir / f'class_{class_id}'
            if not class_dir.exists():
                continue

            for img_path in class_dir.glob('*.png'):
                self.samples.append({
                    'image_path': str(img_path),
                    'class_id': class_id
                })
            for img_path in class_dir.glob('*.jpg'):
                self.samples.append({
                    'image_path': str(img_path),
                    'class_id': class_id
                })

        print(f"{split} 数据集: {len(self.samples)} 张图片")

        # 归一化
        self.normalize = T.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )

        # 形态学变换
        self.morph = FingerprintMorphologyTransform()

        # 视图1: 随机增强 + 随机形态学
        self.aug1 = T.Compose([
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip(),
            T.RandomApply([T.GaussianBlur(3)], p=0.2),
            T.ToTensor(),
            T.RandomApply([self.morph], p=0.7),
            T.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x),
            self.normalize,
        ])

        # 视图2: 始终使用形态学增强
        self.aug2 = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            self.morph,
            T.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x),
            self.normalize,
        ])

        # 验证集: 不增强
        self.val_transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x),
            self.normalize,
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = sample['image_path']
        label = sample['class_id']

        # 加载图像
        img = Image.open(img_path).convert('L')  # 灰度图

        if self.split == 'train':
            # 训练: 返回两个增强视图
            v1 = self.aug1(img)
            v2 = self.aug2(img)
            return v1, v2, label
        else:
            # 验证: 返回单个视图
            img_t = self.val_transform(img)
            return img_t, img_t, label



# =========================================================
# 模块 3: BPaCo编码器
# =========================================================

class BpacoEncoder(nn.Module):
    """
    BPaCo编码器: backbone + projection head

    输出:
        feat: 用于分类的特征
        z: 投影头输出 (归一化后用于对比学习)
    """
    def __init__(self, backbone='resnet50', proj_dim=128, pretrained=True):
        super().__init__()

        # 选择backbone
        if backbone == 'resnet18':
            self.backbone = models.resnet18(
                weights=models.ResNet18_Weights.DEFAULT if pretrained else None
            )
            self.feat_dim = self.backbone.fc.in_features
            modules = list(self.backbone.children())[:-1]
            self.encoder = nn.Sequential(*modules)
        elif backbone == 'resnet34':
            self.backbone = models.resnet34(
                weights=models.ResNet34_Weights.DEFAULT if pretrained else None
            )
            self.feat_dim = self.backbone.fc.in_features
            modules = list(self.backbone.children())[:-1]
            self.encoder = nn.Sequential(*modules)
        elif backbone == 'resnet50':
            self.backbone = models.resnet50(
                weights=models.ResNet50_Weights.DEFAULT if pretrained else None
            )
            self.feat_dim = self.backbone.fc.in_features
            modules = list(self.backbone.children())[:-1]
            self.encoder = nn.Sequential(*modules)
        else:
            raise ValueError(f"不支持的backbone: {backbone}")

        # Projection head
        self.proj = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.feat_dim, proj_dim),
        )

    def forward(self, x):
        # Backbone输出 (B, feat_dim, 1, 1)
        f = self.encoder(x).reshape(x.size(0), -1)

        # Projection head
        z = self.proj(f)
        z = F.normalize(z, dim=1)  # 归一化

        return f, z


# =========================================================
# 模块 4: 分类器权重到原型的映射器 (C2)
# =========================================================

class ClassifierToProtoMapper(nn.Module):
    """
    将分类器最后一层权重映射为原型向量
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, W):
        """
        Args:
            W: 分类器最后一层权重 (num_classes, in_dim)
        Returns:
            C2: 原型向量 (num_classes, out_dim)
        """
        C2 = self.fc(W)
        C2 = F.normalize(C2, dim=1)
        return C2


# =========================================================
# 模块 5: 特征队列 (用于跨批次对比学习)
# =========================================================

class FeatureQueue:
    """
    FIFO队列存储历史特征
    """
    def __init__(self, feat_dim, queue_size, device):
        self.feat_dim = feat_dim
        self.queue_size = queue_size
        self.device = device

        self.feats = torch.zeros(queue_size, feat_dim, device=device)
        self.labels = torch.zeros(queue_size, dtype=torch.long, device=device)
        self.ptr = 0
        self.full = False

    def enqueue(self, feats, labels):
        """入队"""
        B = feats.size(0)

        if self.ptr + B <= self.queue_size:
            self.feats[self.ptr:self.ptr + B] = feats
            self.labels[self.ptr:self.ptr + B] = labels
            self.ptr += B
        else:
            # 队列满，循环覆盖
            remain = self.queue_size - self.ptr
            self.feats[self.ptr:] = feats[:remain]
            self.labels[self.ptr:] = labels[:remain]

            overflow = B - remain
            self.feats[:overflow] = feats[remain:]
            self.labels[:overflow] = labels[remain:]
            self.ptr = overflow
            self.full = True

    def get(self):
        """获取队列内容"""
        if (not self.full) and (self.ptr == 0):
            return None, None

        if self.full:
            return self.feats, self.labels
        else:
            return self.feats[:self.ptr], self.labels[:self.ptr]


@torch.no_grad()
def momentum_update(model_k, model_q, momentum):
    """
    EMA动量更新: θ_k ← m * θ_k + (1 - m) * θ_q
    """
    for param_k, param_q in zip(model_k.parameters(), model_q.parameters()):
        param_k.data = param_k.data * momentum + param_q.data * (1.0 - momentum)

# =========================================================
# 模块 6: BPaCo损失函数
# =========================================================

def compute_LBPaCo(
    z_batch,
    labels_batch,
    queue: FeatureQueue,
    C1_param: nn.Parameter,
    classifier,
    mapper: ClassifierToProtoMapper,
    temperature,
    device,
):
    """
    Balanced Prototype & Contrastive Loss

    使用三类原型:
        1. 批内类别均值
        2. 可学习中心 C1
        3. 分类器权重映射 C2
    """
    B, d = z_batch.shape
    K = C1_param.size(0)  # 类别数

    # 获取队列特征
    q_feats, q_labels = queue.get()

    if q_feats is None:
        A_feats = z_batch
        A_labels = labels_batch
    else:
        q_feats = q_feats.to(device)
        q_labels = q_labels.to(device)
        A_feats = torch.cat([z_batch, q_feats], dim=0)
        A_labels = torch.cat([labels_batch, q_labels], dim=0)

    # 计算每个类别的均值原型
    class_means = torch.zeros(K, d, device=device)
    valid_mask = torch.zeros(K, dtype=torch.bool, device=device)

    for j in range(K):
        mask = (A_labels == j)
        if mask.sum() > 0:
            class_means[j] = A_feats[mask].mean(dim=0)
            valid_mask[j] = True

    # 获取分类器权重 -> C2
    W = classifier[-1].weight  # (num_classes, feat_dim)
    C2 = mapper(W)

    # 计算与每个类别原型的相似度
    sims_per_class = torch.zeros(B, K, device=device)

    for j in range(K):
        reps = []
        if valid_mask[j]:
            reps.append(F.normalize(class_means[j].unsqueeze(0), dim=1))
        reps.append(C1_param[j].unsqueeze(0))
        reps.append(C2[j].unsqueeze(0))
        reps_cat = torch.cat(reps, dim=0)  # (m_j, d)

        sims = torch.matmul(z_batch, reps_cat.t()) / temperature
        sims_exp = torch.exp(sims)
        sims_avg = sims_exp.mean(dim=1)
        sims_per_class[:, j] = sims_avg

    # 计算损失
    numerators = sims_per_class[torch.arange(B, device=device), labels_batch] + 1e-12
    denominators = sims_per_class.sum(dim=1) + 1e-12
    loss = -torch.log(numerators / denominators).mean()

    return loss


def cross_entropy_with_logit_compensation(logits, labels, class_freq_tensor, tau):
    """
    Logit Compensation: 缓解长尾分布下多数类的优势

    logits_adj = logits - tau * log(class_freq + 1)
    """
    eps = 1e-12
    adjustment = tau * torch.log(
        class_freq_tensor.float().to(logits.device) + 1.0 + eps
    )
    logits_adj = logits - adjustment.unsqueeze(0)
    return F.cross_entropy(logits_adj, labels)


# =========================================================
# 模块 7: BPaCo训练器
# =========================================================

class BPaCoTrainer:
    """
    BPaCo训练器 - 19类版本
    """
    def __init__(
        self,
        dataset_dir,
        output_dir,
        num_classes=19,
        backbone='resnet50',
        proj_dim=128,
        queue_size=4096,
        momentum=0.999,
        beta=1.5,
        tau=1.2,
        temperature=0.1,
        device=None,
    ):
        self.dataset_dir = dataset_dir
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.num_classes = num_classes
        self.backbone = backbone
        self.proj_dim = proj_dim
        self.queue_size = queue_size
        self.momentum = momentum
        self.beta = beta
        self.tau = tau
        self.temperature = temperature

        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")

        # 初始化模型
        self._build_model()

    def _build_model(self):
        """构建模型"""
        # Query/Key编码器
        self.model_q = BpacoEncoder(
            self.backbone, proj_dim=self.proj_dim, pretrained=True
        ).to(self.device)

        self.model_k = BpacoEncoder(
            self.backbone, proj_dim=self.proj_dim, pretrained=True
        ).to(self.device)

        # 初始化key编码器
        for param_q, param_k in zip(self.model_q.parameters(), self.model_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # 分类器
        feat_dim = self.model_q.feat_dim
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, self.num_classes),
        ).to(self.device)

        # 可学习类别中心 C1
        self.C1 = nn.Parameter(
            torch.randn(self.num_classes, self.proj_dim, device=self.device)
        )

        # 分类器权重到原型的mapper (C2)
        self.mapper = ClassifierToProtoMapper(
            in_dim=self.classifier[-1].in_features,
            out_dim=self.proj_dim
        ).to(self.device)

        # 特征队列
        self.queue = FeatureQueue(
            feat_dim=self.proj_dim,
            queue_size=self.queue_size,
            device=self.device
        )

        # 优化器
        params = (
            list(self.model_q.parameters()) +
            list(self.classifier.parameters()) +
            [self.C1] +
            list(self.mapper.parameters())
        )
        self.optimizer = torch.optim.SGD(
            params, lr=0.001, momentum=0.9, weight_decay=1e-4
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=50, gamma=0.1
        )

    def train_one_epoch(self, dataloader, class_freq_tensor):
        """训练一个epoch"""
        self.model_q.train()
        self.classifier.train()
        self.mapper.train()

        losses = []

        for v1, v2, labels in dataloader:
            v1 = v1.to(self.device)
            v2 = v2.to(self.device)
            labels = labels.to(self.device)

            # Query编码器
            feat_q, z_q = self.model_q(v1)

            # Key编码器 (no grad)
            with torch.no_grad():
                feat_k, z_k = self.model_k(v2)

            # 分类
            feat_for_cls = torch.cat([feat_q, feat_k], dim=1)
            logits = self.classifier(feat_for_cls)

            # 损失
            self.optimizer.zero_grad()

            ce_loss = cross_entropy_with_logit_compensation(
                logits, labels, class_freq_tensor, tau=self.tau
            )

            lbpaco_loss = compute_LBPaCo(
                z_q, labels, self.queue, self.C1,
                self.classifier, self.mapper,
                temperature=self.temperature,
                device=self.device
            )

            loss = ce_loss + self.beta * lbpaco_loss

            loss.backward()
            self.optimizer.step()

            # 动量更新key编码器
            momentum_update(self.model_k, self.model_q, self.momentum)

            # 更新队列
            with torch.no_grad():
                self.queue.enqueue(z_k, labels)

            losses.append(loss.item())

        self.scheduler.step()
        return np.mean(losses)

    @torch.no_grad()
    def validate(self, dataloader):
        """验证"""
        self.model_q.eval()
        self.model_k.eval()
        self.classifier.eval()

        y_true = []
        y_pred = []

        for v1, v2, labels in dataloader:
            v1 = v1.to(self.device)
            v2 = v2.to(self.device)
            labels = labels.to(self.device)

            feat_q, _ = self.model_q(v1)
            feat_k, _ = self.model_k(v2)

            feat_for_cls = torch.cat([feat_q, feat_k], dim=1)
            logits = self.classifier(feat_for_cls)

            preds = logits.argmax(dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')

        return {
            'accuracy': float(acc),
            'f1': float(f1),
            'y_true': y_true.tolist(),
            'y_pred': y_pred.tolist()
        }

    def plot_confusion_matrix(self, y_true, y_pred, epoch):
        """绘制混淆矩阵"""
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - Epoch {epoch}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

        save_path = self.output_dir / f'confusion_matrix_epoch_{epoch}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"混淆矩阵已保存: {save_path}")

    def run_training(self, batch_size=32, epochs=100, image_size=224, val_interval=10):
        """完整训练流程"""
        print("\n" + "=" * 60)
        print("开始训练 BPaCo - 19类")
        print("=" * 60)

        # 创建数据集
        train_dataset = MultiModalBPaCoDataset(
            self.dataset_dir, split='train', image_size=image_size
        )
        val_dataset = MultiModalBPaCoDataset(
            self.dataset_dir, split='val', image_size=image_size
        )

        # 数据加载器
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size,
            shuffle=True, num_workers=4, pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size,
            shuffle=False, num_workers=4, pin_memory=True
        )

        # 统计类别频率
        class_freq = torch.zeros(self.num_classes, dtype=torch.long)
        for _, _, label in DataLoader(train_dataset, batch_size=1, shuffle=False):
            class_freq[label.item()] += 1

        print(f"\n类别频率: {class_freq.tolist()}")
        print(f"最多/最少: {class_freq.max()}/{class_freq.min()} = {class_freq.max()/class_freq.min():.1f}倍")

        # 训练
        best_f1 = 0.0
        history = {'train_loss': [], 'val_acc': [], 'val_f1': []}

        for epoch in range(1, epochs + 1):
            t0 = datetime.now()

            avg_loss = self.train_one_epoch(train_loader, class_freq)
            history['train_loss'].append(avg_loss)

            t1 = datetime.now()
            print(f"\n[Epoch {epoch}/{epochs}] 用时 {(t1-t0).total_seconds():.1f}s, Loss={avg_loss:.4f}")

            # 验证
            if epoch % val_interval == 0 or epoch == epochs:
                val_metrics = self.validate(val_loader)
                history['val_acc'].append(val_metrics['accuracy'])
                history['val_f1'].append(val_metrics['f1'])

                print(f"验证 - Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}")

                # 保存最佳模型
                if val_metrics['f1'] > best_f1:
                    best_f1 = val_metrics['f1']
                    self.save_checkpoint(epoch, val_metrics, is_best=True)
                    print(f"✅ 最佳模型已保存 (F1={best_f1:.4f})")

                # 绘制混淆矩阵
                self.plot_confusion_matrix(
                    val_metrics['y_true'],
                    val_metrics['y_pred'],
                    epoch
                )

        # 保存训练历史
        history_path = self.output_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)

        print("\n" + "=" * 60)
        print(f"训练完成！最佳F1: {best_f1:.4f}")
        print(f"结果保存在: {self.output_dir}")
        print("=" * 60)

        return history

    def save_checkpoint(self, epoch, metrics, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_q_state_dict': self.model_q.state_dict(),
            'model_k_state_dict': self.model_k.state_dict(),
            'classifier_state_dict': self.classifier.state_dict(),
            'C1': self.C1.data,
            'mapper_state_dict': self.mapper.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
        }

        if is_best:
            save_path = self.output_dir / 'best_model.pth'
        else:
            save_path = self.output_dir / f'checkpoint_epoch_{epoch}.pth'

        torch.save(checkpoint, save_path)


# =========================================================
# 模块 8: 主函数
# =========================================================

def main():
    parser = argparse.ArgumentParser(description='BPaCo训练 - 19类手势分类')
    parser.add_argument('--dataset', type=str, default='../classification_dataset',
                       help='数据集目录')
    parser.add_argument('--output', type=str, default='./results/bpaco_19cls',
                       help='输出目录')
    parser.add_argument('--backbone', type=str, default='resnet50',
                       choices=['resnet18', 'resnet34', 'resnet50'],
                       help='Backbone模型')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--epochs', type=int, default=100,
                       help='训练轮数')
    parser.add_argument('--image-size', type=int, default=224,
                       help='图像大小')
    parser.add_argument('--proj-dim', type=int, default=128,
                       help='投影维度')
    parser.add_argument('--queue-size', type=int, default=4096,
                       help='队列大小')
    parser.add_argument('--momentum', type=float, default=0.999,
                       help='EMA动量')
    parser.add_argument('--beta', type=float, default=1.5,
                       help='BPaCo损失权重 (长尾严重时增大)')
    parser.add_argument('--tau', type=float, default=1.2,
                       help='Logit补偿参数')
    parser.add_argument('--temperature', type=float, default=0.1,
                       help='对比学习温度')
    parser.add_argument('--val-interval', type=int, default=10,
                       help='验证间隔')

    args = parser.parse_args()

    # 创建训练器
    trainer = BPaCoTrainer(
        dataset_dir=args.dataset,
        output_dir=args.output,
        num_classes=19,
        backbone=args.backbone,
        proj_dim=args.proj_dim,
        queue_size=args.queue_size,
        momentum=args.momentum,
        beta=args.beta,
        tau=args.tau,
        temperature=args.temperature,
    )

    # 开始训练
    trainer.run_training(
        batch_size=args.batch_size,
        epochs=args.epochs,
        image_size=args.image_size,
        val_interval=args.val_interval,
    )


if __name__ == '__main__':
    main()

