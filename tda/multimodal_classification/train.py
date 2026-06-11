#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多模态分类模型训练脚本
"""

import os
# 设置HuggingFace镜像（必须在导入timm之前）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from pathlib import Path
import numpy as np
from tqdm import tqdm
import json
import argparse

from multimodal_model import MultiModalClassifier
from dataset import get_dataloaders
from focal_loss import get_loss_function
from collections import Counter


def train_epoch(model, train_loader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()

    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc='训练')
    first_batch = True
    for batch in pbar:
        images = batch['image'].to(device)
        keypoint_features = batch['keypoint_features'].to(device)
        labels = batch['label'].to(device)

        # 调试第一个batch
        if first_batch and hasattr(criterion, 'weights'):
            print(f"\n[DEBUG] 第一个batch:")
            print(f"  logits设备: {device}")
            print(f"  labels设备: {labels.device}")
            print(f"  criterion.weights设备: {criterion.weights.device}")
            first_batch = False

        # 前向传播
        optimizer.zero_grad()
        logits = model(images, keypoint_features)
        loss = criterion(logits, labels)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 统计
        total_loss += loss.item()
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # 更新进度条
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def validate(model, val_loader, criterion, device):
    """验证"""
    model.eval()
    
    total_loss = 0
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='验证'):
            images = batch['image'].to(device)
            keypoint_features = batch['keypoint_features'].to(device)
            labels = batch['label'].to(device)
            
            # 前向传播
            logits = model(images, keypoint_features)
            loss = criterion(logits, labels)
            
            # 统计
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(val_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy, all_preds, all_labels


def train(
    model,
    train_loader,
    val_loader,
    criterion,
    num_epochs,
    learning_rate,
    device,
    save_dir,
    patience=10
):
    """完整训练流程"""
    
    # 优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4
    )
    
    # 学习率调度器
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # 保存目录
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # 训练历史
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0
    patience_counter = 0
    
    print("\n开始训练...")
    print("=" * 60)
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 60)
        
        # 训练
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # 验证
        val_loss, val_acc, _, _ = validate(
            model, val_loader, criterion, device
        )
        
        # 更新学习率
        scheduler.step()
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # 打印结果
        print(f"\n训练 - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"验证 - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
        print(f"学习率: {optimizer.param_groups[0]['lr']:.6f}")

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, save_path / 'best_model.pth')

            print(f"✓ 保存最佳模型 (验证准确率: {val_acc:.2f}%)")
        else:
            patience_counter += 1
            print(f"早停计数: {patience_counter}/{patience}")

        # 早停
        if patience_counter >= patience:
            print(f"\n早停触发！验证准确率 {patience} 个epoch未提升")
            break

        # 保存检查点
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, save_path / f'checkpoint_epoch_{epoch+1}.pth')

    # 保存训练历史
    with open(save_path / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 60)
    print("训练完成！")
    print(f"最佳验证准确率: {best_val_acc:.2f}%")
    print(f"模型保存在: {save_path}")

    return history


def main():
    parser = argparse.ArgumentParser(description='训练多模态分类模型')
    parser.add_argument('--dataset', type=str, default='../classification_dataset',
                       help='数据集目录')
    parser.add_argument('--keypoint-features', type=str, default='keypoint_features.csv',
                       help='关键点特征CSV文件')
    parser.add_argument('--model', type=str, default='resnet50',
                       help='图像backbone模型')
    parser.add_argument('--pretrained', action='store_true',
                       help='是否使用预训练模型（需要网络访问HuggingFace）')
    parser.add_argument('--freeze-backbone', action='store_true',
                       help='是否冻结图像backbone（只训练映射层和交叉注意力）')
    parser.add_argument('--loss-type', type=str, default='balanced',
                       choices=['ce', 'focal', 'balanced'],
                       help='损失函数类型: ce(交叉熵), focal(Focal Loss), balanced(类别平衡损失)')
    parser.add_argument('--use-balanced-sampling', action='store_true', default=False,
                       help='是否使用类别平衡采样')
    parser.add_argument('--loss-beta', type=float, default=0.999,
                       help='类别平衡损失的beta参数 (0.9999=极度长尾, 0.999=中等长尾, 0.99=轻度长尾)')
    parser.add_argument('--loss-gamma', type=float, default=2.0,
                       help='Focal Loss的gamma参数 (越大越关注难样本, 推荐2.0-5.0)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='学习率')
    parser.add_argument('--image-size', type=int, default=640,
                       help='图像大小')
    parser.add_argument('--fusion-dim', type=int, default=512,
                       help='融合特征维度')
    parser.add_argument('--num-heads', type=int, default=8,
                       help='注意力头数')
    parser.add_argument('--save-dir', type=str, default='checkpoints',
                       help='模型保存目录')
    parser.add_argument('--patience', type=int, default=15,
                       help='早停耐心值')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='数据加载线程数')

    args = parser.parse_args()

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 数据加载
    print("\n加载数据...")
    train_loader, val_loader, test_loader = get_dataloaders(
        dataset_dir=args.dataset,
        keypoint_features_file=args.keypoint_features,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        use_balanced_sampling=args.use_balanced_sampling
    )

    # 创建模型
    print("\n创建模型...")
    model = MultiModalClassifier(
        num_classes=19,
        image_model_name=args.model,
        pretrained=args.pretrained,
        keypoint_input_dim=8,
        fusion_dim=args.fusion_dim,
        num_heads=args.num_heads,
        freeze_backbone=args.freeze_backbone
    ).to(device)

    # 损失函数（处理长尾分布）
    print(f"\n损失函数: {args.loss_type}")
    if args.loss_type == 'balanced':
        # 统计每个类别的样本数
        from pathlib import Path
        dataset_path = Path(args.dataset)
        samples_per_class = []
        for class_id in range(19):
            class_dir = dataset_path / 'train' / f'class_{class_id}'
            if class_dir.exists():
                count = len(list(class_dir.glob('*.png')) + list(class_dir.glob('*.jpg')))
                samples_per_class.append(count)
            else:
                samples_per_class.append(1)
        print(f"每类样本数: {samples_per_class}")
        print(f"损失函数参数: beta={args.loss_beta}, gamma={args.loss_gamma}")
        criterion = get_loss_function('balanced', samples_per_class=samples_per_class,
                                     beta=args.loss_beta, gamma=args.loss_gamma)
    elif args.loss_type == 'focal':
        print(f"损失函数参数: gamma={args.loss_gamma}")
        criterion = get_loss_function('focal', gamma=args.loss_gamma)
    else:
        criterion = get_loss_function(args.loss_type)

    # 将损失函数移到设备上
    criterion = criterion.to(device)

    # 调试：检查criterion的设备
    if hasattr(criterion, 'weights'):
        print(f"Criterion weights设备: {criterion.weights.device}")

    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")

    # 训练
    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        device=device,
        save_dir=args.save_dir,
        patience=args.patience
    )


if __name__ == "__main__":
    main()

