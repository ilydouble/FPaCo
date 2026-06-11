#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多模态数据集加载器
"""

import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from pathlib import Path
from PIL import Image, ImageFilter, ImageEnhance
import json
import numpy as np
from torchvision import transforms
import torchvision.transforms.functional as TF
from collections import Counter
import random
import cv2


class YOLOAugmentation:
    """YOLO风格的数据增强"""

    @staticmethod
    def mosaic(images, size=224):
        """Mosaic增强：将4张图片拼接成1张"""
        # 简化版：随机裁剪并拼接
        h, w = size, size
        result = Image.new('RGB', (w, h))

        # 随机分割点
        cx = random.randint(w // 4, 3 * w // 4)
        cy = random.randint(h // 4, 3 * h // 4)

        # 4个区域
        regions = [
            (0, 0, cx, cy),           # 左上
            (cx, 0, w, cy),           # 右上
            (0, cy, cx, h),           # 左下
            (cx, cy, w, h)            # 右下
        ]

        for i, (x1, y1, x2, y2) in enumerate(regions):
            if i < len(images):
                img = images[i].resize((x2-x1, y2-y1))
                result.paste(img, (x1, y1))

        return result

    @staticmethod
    def mixup(img1, img2, alpha=0.5):
        """MixUp增强：混合两张图片"""
        # 转换为numpy
        arr1 = np.array(img1, dtype=np.float32)
        arr2 = np.array(img2, dtype=np.float32)

        # 混合
        mixed = (alpha * arr1 + (1 - alpha) * arr2).astype(np.uint8)

        return Image.fromarray(mixed)

    @staticmethod
    def hsv_augment(image, hgain=0.015, sgain=0.7, vgain=0.4):
        """HSV色彩空间增强（YOLO默认）"""
        # 转换为numpy
        img_np = np.array(image)

        # RGB to HSV
        img_hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV).astype(np.float32)

        # 随机增益
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1

        # 应用增益
        h, s, v = cv2.split(img_hsv)
        h = (h * r[0]) % 180
        s = np.clip(s * r[1], 0, 255)
        v = np.clip(v * r[2], 0, 255)

        img_hsv = cv2.merge([h, s, v]).astype(np.uint8)

        # HSV to RGB
        img_rgb = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)

        return Image.fromarray(img_rgb)

    @staticmethod
    def random_erasing(image, p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3)):
        """随机擦除（Random Erasing）"""
        if random.random() > p:
            return image

        img_np = np.array(image)
        h, w, c = img_np.shape

        # 随机擦除区域
        area = h * w
        target_area = random.uniform(scale[0], scale[1]) * area
        aspect_ratio = random.uniform(ratio[0], ratio[1])

        rh = int(round(np.sqrt(target_area * aspect_ratio)))
        rw = int(round(np.sqrt(target_area / aspect_ratio)))

        if rw < w and rh < h:
            x1 = random.randint(0, w - rw)
            y1 = random.randint(0, h - rh)

            # 随机填充颜色
            img_np[y1:y1+rh, x1:x1+rw, :] = np.random.randint(0, 255, (rh, rw, c))

        return Image.fromarray(img_np)

    @staticmethod
    def gaussian_blur(image, p=0.5, kernel_size=5):
        """高斯模糊"""
        if random.random() > p:
            return image
        return image.filter(ImageFilter.GaussianBlur(radius=kernel_size//2))

    @staticmethod
    def motion_blur(image, p=0.5, kernel_size=5):
        """运动模糊"""
        if random.random() > p:
            return image

        img_np = np.array(image)
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[kernel_size//2, :] = 1.0 / kernel_size

        # 随机旋转kernel
        angle = random.uniform(0, 360)
        center = (kernel_size // 2, kernel_size // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        kernel = cv2.warpAffine(kernel, M, (kernel_size, kernel_size))

        blurred = cv2.filter2D(img_np, -1, kernel)
        return Image.fromarray(blurred)

    @staticmethod
    def add_noise(image, p=0.5, noise_type='gaussian'):
        """添加噪声"""
        if random.random() > p:
            return image

        img_np = np.array(image, dtype=np.float32)

        if noise_type == 'gaussian':
            # 高斯噪声
            noise = np.random.normal(0, 10, img_np.shape)
            img_np = np.clip(img_np + noise, 0, 255)
        elif noise_type == 'salt_pepper':
            # 椒盐噪声
            prob = 0.01
            mask = np.random.random(img_np.shape[:2])
            img_np[mask < prob/2] = 0
            img_np[mask > 1 - prob/2] = 255

        return Image.fromarray(img_np.astype(np.uint8))


class MultiModalDataset(Dataset):
    """多模态数据集：图像 + 关键点特征"""

    def __init__(
        self,
        dataset_dir,
        keypoint_features_file,
        split='train',
        image_size=224,
        augment=True
    ):
        """
        Args:
            dataset_dir: 数据集根目录
            keypoint_features_file: 关键点特征CSV文件
            split: 'train', 'val', 'test'
            image_size: 图像大小
            augment: 是否使用数据增强
        """
        self.dataset_dir = Path(dataset_dir)
        self.split = split
        self.image_size = image_size

        # 加载关键点特征（CSV格式）
        import csv
        self.keypoint_features = {}
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
        
        # 收集所有样本
        self.samples = []
        split_dir = self.dataset_dir / split
        
        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            
            class_name = class_dir.name
            class_id = int(class_name.split('_')[1])  # class_0 -> 0
            
            # 收集图片
            for img_path in list(class_dir.glob('*.png')) + list(class_dir.glob('*.jpg')):
                rel_path = str(img_path.relative_to(self.dataset_dir))
                
                # 检查是否有关键点特征
                if rel_path in self.keypoint_features:
                    self.samples.append({
                        'image_path': img_path,
                        'class_id': class_id,
                        'rel_path': rel_path
                    })
        
        # 图像变换
        self.augment = augment
        self.split = split
        self.yolo_aug = YOLOAugmentation()

        # 基础变换（所有样本）
        self.base_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # 轻度增强（难样本）- 保留细节
        self.light_augment_list = [
            transforms.Resize((image_size, image_size)),
            transforms.RandomRotation(5),  # 小角度旋转
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
        ]

        # 强增强（简单样本）- YOLO风格
        self.strong_augment_list = [
            transforms.Resize((image_size, image_size)),
            transforms.RandomRotation(15),  # 大角度旋转
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        ]
        
        print(f"{split} 集: {len(self.samples)} 个样本")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 加载图像
        image = Image.open(sample['image_path']).convert('RGB')

        # 加载关键点特征（CSV，用于BPaCo的对比学习队列或其他用途）
        kp_features_dict = self.keypoint_features[sample['rel_path']]
        
        # --- 加载 JSON 检测结果 (序列特征) ---
        json_path = Path(sample['image_path']).with_suffix('.json')
        detections = []
        if json_path.exists():
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    detections = data.get('detections', [])
            except Exception as e:
                print(f"Error loading JSON {json_path}: {e}")
        
        # 处理检测序列 -> Tensor (MaxLen, 6)
        # Feature: [cx_norm, cy_norm, w_norm, h_norm, conf, class_id]
        MAX_LEN = 10
        seq_features = np.zeros((MAX_LEN, 6), dtype=np.float32)
        
        # 按照置信度排序
        detections.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        
        for i, det in enumerate(detections[:MAX_LEN]):
            bbox = det['bbox'] # xyxy
            img_w, img_h = data['image_size'] if 'image_size' in data else (224, 224) 
            # 简单的归一化 (如果JSON里没存image_size，可能有点问题，但通常有)
            # 实际上 JSON里的normalized_center已经有了，我们可以直接用详细信息
            
            # 使用 box 来计算，确保对应
            x1, y1, x2, y2 = bbox
            w_box = x2 - x1
            h_box = y2 - y1
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            
            # 归一化
            cx_norm = cx / img_w
            cy_norm = cy / img_h
            w_norm = w_box / img_w
            h_norm = h_box / img_h
            conf = det['confidence']
            cls_id = float(det['class_id'])
            
            seq_features[i] = [cx_norm, cy_norm, w_norm, h_norm, conf, cls_id]
            
        seq_features = torch.from_numpy(seq_features)
        # ----------------------------------------

        # 根据样本特性选择增强策略
        if self.augment and self.split == 'train':
            is_hard = kp_features_dict['is_hard_sample']
            is_left = kp_features_dict['is_left_hand']
            is_right = kp_features_dict['is_right_hand']

            # 难样本：轻度增强（保留细节）
            if is_hard == 1:
                # 应用轻度增强
                for t in self.light_augment_list:
                    image = t(image)

                # 可选：轻微模糊（10%概率）
                image = self.yolo_aug.gaussian_blur(image, p=0.1, kernel_size=3)

                # 转换为tensor
                image = transforms.ToTensor()(image)
                image = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])(image)

            # 简单样本：强增强（YOLO风格）
            else:
                # 1. HSV色彩增强（YOLO默认）
                image = self.yolo_aug.hsv_augment(image, hgain=0.015, sgain=0.7, vgain=0.4)

                # 2. 几何变换
                for t in self.strong_augment_list:
                    image = t(image)

                # 3. 随机擦除（30%概率）
                image = self.yolo_aug.random_erasing(image, p=0.3, scale=(0.02, 0.1))

                # 4. 模糊增强（20%概率）
                if random.random() < 0.2:
                    if random.random() < 0.5:
                        image = self.yolo_aug.gaussian_blur(image, p=1.0, kernel_size=5)
                    else:
                        image = self.yolo_aug.motion_blur(image, p=1.0, kernel_size=5)

                # 5. 添加噪声（15%概率）
                image = self.yolo_aug.add_noise(image, p=0.15, noise_type='gaussian')

                # 转换为tensor
                image = transforms.ToTensor()(image)
                image = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])(image)

                # 6. 左右手样本：可以水平翻转（同时翻转标签）
                if (is_left == 1 or is_right == 1) and np.random.rand() < 0.5:
                    # 翻转图像
                    image_pil = transforms.ToPILImage()(image)
                    image_pil = transforms.functional.hflip(image_pil)
                    image = transforms.ToTensor()(image_pil)
                    image = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])(image)

                    # 交换左右手标签
                    kp_features_dict['is_left_hand'], kp_features_dict['is_right_hand'] = \
                        kp_features_dict['is_right_hand'], kp_features_dict['is_left_hand']

                    # 交换左右位置关系
                    kp_features_dict['kp1_left_of_kp2'], kp_features_dict['kp1_right_of_kp2'] = \
                        kp_features_dict['kp1_right_of_kp2'], kp_features_dict['kp1_left_of_kp2']
        else:
            image = self.base_transform(image)

        # 提取统计特征（8维）- 依然保留作为补充或对比
        feature_vector = np.array([
            kp_features_dict['num_kp1'],
            kp_features_dict['num_kp2'],
            kp_features_dict['is_left_hand'],
            kp_features_dict['is_right_hand'],
            kp_features_dict['is_hard_sample'],
            kp_features_dict['kp1_between_kp2'],
            kp_features_dict['kp1_left_of_kp2'],
            kp_features_dict['kp1_right_of_kp2'],
        ], dtype=np.float32)

        feature_vector = torch.from_numpy(feature_vector)

        # 类别标签
        label = sample['class_id']

        return {
            'image': image,
            'stat_features': feature_vector, # 原来的 keypoint_features
            'seq_features': seq_features,    # 新的 JSON 序列特征
            'label': label,
            'image_path': str(sample['image_path'])
        }


def get_dataloaders(
    dataset_dir,
    keypoint_features_file,
    batch_size=32,
    num_workers=4,
    image_size=224,
    use_balanced_sampling=True
):
    """
    创建数据加载器

    Args:
        use_balanced_sampling: 是否使用类别平衡采样（处理长尾分布）

    Returns:
        train_loader, val_loader, test_loader
    """
    train_dataset = MultiModalDataset(
        dataset_dir=dataset_dir,
        keypoint_features_file=keypoint_features_file,
        split='train',
        image_size=image_size,
        augment=True
    )

    val_dataset = MultiModalDataset(
        dataset_dir=dataset_dir,
        keypoint_features_file=keypoint_features_file,
        split='val',
        image_size=image_size,
        augment=False
    )

    test_dataset = MultiModalDataset(
        dataset_dir=dataset_dir,
        keypoint_features_file=keypoint_features_file,
        split='test',
        image_size=image_size,
        augment=False
    )

    # 类别平衡采样（处理长尾分布）
    if use_balanced_sampling:
        # 统计每个类别的样本数
        class_counts = Counter([s['class_id'] for s in train_dataset.samples])
        print(f"\n类别分布: {dict(sorted(class_counts.items()))}")

        # 计算每个样本的权重（类别越少，权重越大）
        class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
        sample_weights = [class_weights[s['class_id']] for s in train_dataset.samples]

        # 创建加权采样器
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,  # 使用采样器，不能同时shuffle
            num_workers=num_workers,
            pin_memory=True
        )
        print("✅ 使用类别平衡采样")
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # 测试数据加载
    train_loader, val_loader, test_loader = get_dataloaders(
        dataset_dir='../classification_dataset',
        keypoint_features_file='keypoint_features.csv',
        batch_size=4,
        num_workers=0
    )

    print("\n测试数据加载...")
    for batch in train_loader:
        print(f"图像: {batch['image'].shape}")
        print(f"关键点特征: {batch['keypoint_features'].shape}")
        print(f"标签: {batch['label'].shape}")
        print(f"关键点特征示例: {batch['keypoint_features'][0]}")
        break

