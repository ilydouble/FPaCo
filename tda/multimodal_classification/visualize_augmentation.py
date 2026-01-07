#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化数据增强效果
"""

import matplotlib.pyplot as plt
from dataset import MultiModalDataset
import torch
import numpy as np
from pathlib import Path
import argparse


def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """反归一化"""
    tensor = tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return torch.clamp(tensor, 0, 1)


def visualize_augmentations(dataset_dir, keypoint_features_file, num_samples=5, num_augments=6):
    """
    可视化数据增强效果
    
    Args:
        dataset_dir: 数据集目录
        keypoint_features_file: 关键点特征文件
        num_samples: 显示多少个样本
        num_augments: 每个样本显示多少个增强版本
    """
    # 创建数据集
    dataset = MultiModalDataset(
        dataset_dir=dataset_dir,
        keypoint_features_file=keypoint_features_file,
        split='train',
        image_size=224,
        augment=True
    )
    
    # 随机选择样本
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    # 创建图表
    fig, axes = plt.subplots(num_samples, num_augments + 1, figsize=(3*(num_augments+1), 3*num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for row, idx in enumerate(indices):
        sample = dataset.samples[idx]
        
        # 原始图像
        from PIL import Image
        original_img = Image.open(sample['image_path']).convert('RGB')
        axes[row, 0].imshow(original_img)
        axes[row, 0].set_title(f"Original\nClass {sample['class_id']}", fontsize=10)
        axes[row, 0].axis('off')
        
        # 增强版本
        for col in range(1, num_augments + 1):
            # 获取增强后的样本
            data = dataset[idx]
            img_tensor = data['image']
            kp_features = data['keypoint_features']
            
            # 反归一化
            img_tensor = denormalize(img_tensor)
            img_np = img_tensor.permute(1, 2, 0).numpy()
            
            axes[row, col].imshow(img_np)
            
            # 显示关键点特征
            is_hard = int(kp_features[4].item())
            is_left = int(kp_features[2].item())
            is_right = int(kp_features[3].item())
            
            hand_type = "Left" if is_left else ("Right" if is_right else "Unknown")
            difficulty = "Hard" if is_hard else "Easy"
            
            axes[row, col].set_title(f"Aug {col}\n{hand_type}, {difficulty}", fontsize=8)
            axes[row, col].axis('off')
    
    plt.tight_layout()
    
    # 保存
    output_dir = Path('augmentation_visualization')
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'augmentation_samples.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ 可视化结果保存到: {output_path}")
    
    plt.show()


def show_augmentation_types():
    """展示不同类型的增强"""
    from PIL import Image
    from dataset import YOLOAugmentation
    import torchvision.transforms as transforms
    
    # 加载一张示例图片
    dataset_dir = Path('../classification_dataset/train')
    sample_img = None
    for class_dir in dataset_dir.iterdir():
        if class_dir.is_dir():
            imgs = list(class_dir.glob('*.png')) + list(class_dir.glob('*.jpg'))
            if imgs:
                sample_img = Image.open(imgs[0]).convert('RGB')
                break
    
    if sample_img is None:
        print("未找到示例图片")
        return
    
    yolo_aug = YOLOAugmentation()
    
    # 不同的增强方法
    augmentations = {
        'Original': lambda x: x,
        'HSV Augment': lambda x: yolo_aug.hsv_augment(x),
        'Gaussian Blur': lambda x: yolo_aug.gaussian_blur(x, p=1.0),
        'Motion Blur': lambda x: yolo_aug.motion_blur(x, p=1.0),
        'Random Erasing': lambda x: yolo_aug.random_erasing(x, p=1.0, scale=(0.1, 0.2)),
        'Gaussian Noise': lambda x: yolo_aug.add_noise(x, p=1.0, noise_type='gaussian'),
        'Salt&Pepper': lambda x: yolo_aug.add_noise(x, p=1.0, noise_type='salt_pepper'),
    }
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for idx, (name, aug_fn) in enumerate(augmentations.items()):
        if idx < len(axes):
            aug_img = aug_fn(sample_img.copy())
            axes[idx].imshow(aug_img)
            axes[idx].set_title(name, fontsize=12, fontweight='bold')
            axes[idx].axis('off')
    
    # 隐藏多余的子图
    for idx in range(len(augmentations), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    output_dir = Path('augmentation_visualization')
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'augmentation_types.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ 增强类型展示保存到: {output_path}")
    
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='可视化数据增强')
    parser.add_argument('--dataset', type=str, default='../classification_dataset',
                       help='数据集目录')
    parser.add_argument('--keypoint-features', type=str, default='keypoint_features.csv',
                       help='关键点特征文件')
    parser.add_argument('--mode', type=str, default='samples', choices=['samples', 'types'],
                       help='可视化模式: samples(样本增强), types(增强类型)')
    
    args = parser.parse_args()
    
    if args.mode == 'samples':
        visualize_augmentations(args.dataset, args.keypoint_features, num_samples=5, num_augments=6)
    else:
        show_augmentation_types()

