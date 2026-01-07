#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Focal Loss - 处理长尾分布和类别不平衡
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FocalLoss(nn.Module):
    """
    Focal Loss: 降低简单样本的权重，聚焦于难样本
    
    论文: Focal Loss for Dense Object Detection
    https://arxiv.org/abs/1708.02002
    
    适用于长尾分布数据集
    """
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha: 类别权重，shape=[num_classes] 或 None
                   可以设置为类别频率的倒数来平衡长尾分布
            gamma: 聚焦参数，gamma越大，对简单样本的抑制越强
                   gamma=0 时退化为交叉熵损失
                   推荐值: 2.0
            reduction: 'mean', 'sum', 'none'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: [batch_size, num_classes] 模型输出logits
            targets: [batch_size] 类别标签
        Returns:
            loss: scalar
        """
        # 计算交叉熵
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        # 计算概率
        p = torch.exp(-ce_loss)

        # Focal Loss = (1-p)^gamma * CE
        focal_loss = (1 - p) ** self.gamma * ce_loss

        # 应用类别权重
        if self.alpha is not None:
            if isinstance(self.alpha, (list, np.ndarray)):
                alpha = torch.tensor(self.alpha, dtype=torch.float32, device=inputs.device)
            else:
                alpha = self.alpha
                # 确保alpha在正确的设备上
                if alpha.device != inputs.device:
                    alpha = alpha.to(inputs.device)

            alpha_t = alpha[targets]
            focal_loss = alpha_t * focal_loss

        # 聚合
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class ClassBalancedLoss(nn.Module):
    """
    Class-Balanced Loss: 基于有效样本数的类别平衡损失
    
    论文: Class-Balanced Loss Based on Effective Number of Samples
    https://arxiv.org/abs/1901.05555
    
    特别适合极度长尾分布
    """
    
    def __init__(self, samples_per_class, beta=0.9999, gamma=2.0):
        """
        Args:
            samples_per_class: 每个类别的样本数，list或array
            beta: 平衡参数，越接近1，对少样本类别的权重越大
                  推荐值: 0.9999 (极度长尾), 0.999 (中等长尾)
            gamma: Focal Loss的gamma参数
        """
        super().__init__()

        # 计算有效样本数
        effective_num = 1.0 - np.power(beta, samples_per_class)
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / weights.sum() * len(weights)

        # 保存为buffer，自动跟随模型设备
        self.register_buffer('weights', torch.tensor(weights, dtype=torch.float32))
        self.focal_loss = FocalLoss(alpha=None, gamma=gamma)  # alpha在forward中动态设置
        self.gamma = gamma

    def forward(self, inputs, targets):
        """
        Args:
            inputs: [batch_size, num_classes]
            targets: [batch_size]
        """
        # 手动应用权重
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p = torch.exp(-ce_loss)
        focal_loss = (1 - p) ** self.gamma * ce_loss

        # 应用类别权重（self.weights会自动在正确的设备上，因为是buffer）
        alpha_t = self.weights[targets]
        focal_loss = alpha_t * focal_loss

        return focal_loss.mean()


def get_loss_function(loss_type='focal', samples_per_class=None, **kwargs):
    """
    获取损失函数
    
    Args:
        loss_type: 'ce', 'focal', 'balanced'
        samples_per_class: 每个类别的样本数（用于balanced loss）
        **kwargs: 其他参数
    
    Returns:
        loss_fn
    """
    if loss_type == 'ce':
        return nn.CrossEntropyLoss()
    
    elif loss_type == 'focal':
        gamma = kwargs.get('gamma', 2.0)
        return FocalLoss(gamma=gamma)
    
    elif loss_type == 'balanced':
        if samples_per_class is None:
            raise ValueError("samples_per_class is required for balanced loss")
        beta = kwargs.get('beta', 0.9999)
        gamma = kwargs.get('gamma', 2.0)
        return ClassBalancedLoss(samples_per_class, beta=beta, gamma=gamma)
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")