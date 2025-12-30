#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从检测模型中提取关键点特征，并保存为CSV格式
同时保存详细的检测结果（JSON格式）
"""

import os
# 设置HuggingFace镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import json
import csv
from tqdm import tqdm
import cv2


class KeypointFeatureExtractor:
    """关键点特征提取器"""
    
    def __init__(self, detection_model_path, conf_threshold=0.25):
        """
        Args:
            detection_model_path: 检测模型权重路径
            conf_threshold: 置信度阈值
        """
        self.model = YOLO(detection_model_path)
        self.conf_threshold = conf_threshold
        # 类别名称映射
        self.class_names = {0: 'center_point', 1: 'delta_point'} # 假设 0:中心点, 1:三角点

    def extract_features(self, image_path, original_path=None):
        """
        从图像中提取关键点特征

        Args:
            image_path: 当前图像路径
            original_path: 原始图像路径（用于判断左右手和难样本）

        Returns:
            dict: 包含关键点特征的字典
            list: 原始检测结果列表
        """
        # 预测
        results = self.model.predict(
            source=str(image_path),
            save=False,
            conf=self.conf_threshold,
            verbose=False
        )

        detections = []
        
        if len(results) == 0 or results[0].boxes is None:
            return self._empty_features(original_path), detections

        result = results[0]
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()

        # 读取图像尺寸
        # img = cv2.imread(str(image_path)) # 不需要重新读取整个图片，YOLO结果里有orig_shape
        h, w = result.orig_shape

        # 分类关键点
        kp1_points = []  # 中心点 (class 0)
        kp2_points = []  # 三角点 (class 1)

        for box, cls, conf in zip(boxes, classes, confs):
            cls = int(cls)
            x1, y1, x2, y2 = map(float, box)

            # 计算中心点和归一化坐标
            cx = (x1 + x2) / 2 / w
            cy = (y1 + y2) / 2 / h
            box_w = (x2 - x1) / w
            box_h = (y2 - y1) / h
            
            # 关键点信息
            point_info = {
                'x': float(cx),
                'y': float(cy),
                'box_w': float(box_w),
                'box_h': float(box_h),
                'conf': float(conf)
            }
            
            # 详细检测结果 (用于JSON保存)
            detection_info = {
                'class_id': cls,
                'class_name': self.class_names.get(cls, str(cls)),
                'confidence': float(conf),
                'bbox': [x1, y1, x2, y2], # xyxy 像素坐标
                'center_point': [(x1 + x2) / 2, (y1 + y2) / 2], # xy 像素坐标
                'normalized_center': [float(cx), float(cy)] # 归一化中心坐标
            }
            detections.append(detection_info)

            if cls == 0:
                kp1_points.append(point_info)
            elif cls == 1:
                kp2_points.append(point_info)

        # 构建特征
        features = self._build_features(kp1_points, kp2_points, w, h, original_path)

        return features, detections
    
    def _build_features(self, kp1_points, kp2_points, img_w, img_h, original_path=None):
        """构建关键点特征向量"""
        features = {}

        # 基础统计特征
        features['num_kp1'] = len(kp1_points)  # 中心点数量
        features['num_kp2'] = len(kp2_points)  # 三角点数量

        # 判断左右手（从原始文件名）
        features['is_left_hand'] = 0
        features['is_right_hand'] = 0
        if original_path:
            orig_name = str(original_path).upper()
            if 'L' in orig_name or '左' in orig_name:
                features['is_left_hand'] = 1
            elif 'R' in orig_name or '右' in orig_name:
                features['is_right_hand'] = 1

        # 判断是否难样本（从原始路径）
        features['is_hard_sample'] = 0
        if original_path:
            orig_path_str = str(original_path)
            if '难度2' in orig_path_str or '难样本' in orig_path_str:
                features['is_hard_sample'] = 1

        # 位置关系特征
        features['kp1_between_kp2'] = 0  # 中心点是否在两个三角点中间
        features['kp1_left_of_kp2'] = 0  # 中心点是否在三角点左侧
        features['kp1_right_of_kp2'] = 0  # 中心点是否在三角点右侧

        if len(kp1_points) == 1 and len(kp2_points) >= 1:
            kp1_x = kp1_points[0]['x']
            kp2_xs = [p['x'] for p in kp2_points]

            if len(kp2_points) >= 2:
                # 有至少2个三角点，判断中心点是否在中间
                min_kp2_x = min(kp2_xs)
                max_kp2_x = max(kp2_xs)

                if min_kp2_x < kp1_x < max_kp2_x:
                    features['kp1_between_kp2'] = 1
                elif kp1_x < min_kp2_x:
                    features['kp1_left_of_kp2'] = 1
                elif kp1_x > max_kp2_x:
                    features['kp1_right_of_kp2'] = 1
            else:
                # 只有1个三角点
                kp2_x = kp2_xs[0]
                if kp1_x < kp2_x:
                    features['kp1_left_of_kp2'] = 1
                else:
                    features['kp1_right_of_kp2'] = 1

        # 保存原始关键点信息（用于后续处理）
        features['kp1_points'] = kp1_points
        features['kp2_points'] = kp2_points

        return features
    
    def _empty_features(self, original_path=None):
        """返回空特征"""
        features = {
            'num_kp1': 0,
            'num_kp2': 0,
            'is_left_hand': 0,
            'is_right_hand': 0,
            'is_hard_sample': 0,
            'kp1_between_kp2': 0,
            'kp1_left_of_kp2': 0,
            'kp1_right_of_kp2': 0,
            'kp1_points': [],
            'kp2_points': [],
        }

        # 判断左右手和难样本
        if original_path:
            orig_name = str(original_path).upper()
            if 'L' in orig_name or '左' in orig_name:
                features['is_left_hand'] = 1
            elif 'R' in orig_name or '右' in orig_name:
                features['is_right_hand'] = 1

            orig_path_str = str(original_path)
            if '难度2' in orig_path_str or '难样本' in orig_path_str:
                features['is_hard_sample'] = 1

        return features


def extract_dataset_features(detection_model_path, dataset_dir, output_csv, source_mapping_file):
    """
    提取整个数据集的关键点特征，保存为CSV
    同时保存每个样本的详细检测结果为JSON

    Args:
        detection_model_path: 检测模型路径
        dataset_dir: 数据集目录
        output_csv: 输出CSV文件路径
        source_mapping_file: 原始路径映射文件（build_classification_dataset.py生成）
    """
    extractor = KeypointFeatureExtractor(detection_model_path)
    dataset_path = Path(dataset_dir)

    # 加载原始路径映射
    source_mapping = {}
    if Path(source_mapping_file).exists():
        with open(source_mapping_file, 'r', encoding='utf-8') as f:
            source_mapping = json.load(f)

    all_rows = []

    for split in ['train', 'val', 'test']:
        split_dir = dataset_path / split
        if not split_dir.exists():
            continue

        print(f"\n处理 {split} 集...")

        # 遍历所有类别
        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue

            class_name = class_dir.name
            class_id = int(class_name.split('_')[1])
            print(f"  类别: {class_name}")

            # 遍历所有图片
            image_files = list(class_dir.glob('*.png')) + list(class_dir.glob('*.jpg'))

            for img_path in tqdm(image_files, desc=f"    提取特征"):
                rel_path = str(img_path.relative_to(dataset_path))

                # 获取原始路径
                original_path = source_mapping.get(rel_path, None)

                # 提取特征
                features, detections = extractor.extract_features(img_path, original_path)
                
                # --- 保存详细检测结果为 JSON ---
                json_path = img_path.with_suffix('.json')
                
                # 获取图像尺寸
                img = cv2.imread(str(img_path))
                h, w = img.shape[:2] if img is not None else (0, 0)
                
                json_data = {
                    'image_path': rel_path,
                    'image_size': [w, h],
                    'detections': detections
                }
                
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(json_data, f, ensure_ascii=False, indent=2)
                # -----------------------------

                # 构建CSV行
                row = {
                    'image_path': rel_path,
                    'split': split,
                    'class_id': class_id,
                    'class_name': class_name,
                    'num_kp1': features['num_kp1'],
                    'num_kp2': features['num_kp2'],
                    'is_left_hand': features['is_left_hand'],
                    'is_right_hand': features['is_right_hand'],
                    'is_hard_sample': features['is_hard_sample'],
                    'kp1_between_kp2': features['kp1_between_kp2'],
                    'kp1_left_of_kp2': features['kp1_left_of_kp2'],
                    'kp1_right_of_kp2': features['kp1_right_of_kp2'],
                }

                all_rows.append(row)

    # 保存到CSV
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        'image_path', 'split', 'class_id', 'class_name',
        'num_kp1', 'num_kp2',
        'is_left_hand', 'is_right_hand', 'is_hard_sample',
        'kp1_between_kp2', 'kp1_left_of_kp2', 'kp1_right_of_kp2'
    ]

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\n特征已保存到: {output_path}")
    print(f"总共 {len(all_rows)} 个样本")
    print(f"详细JSON结果已保存在各图片同级目录下")

    # 统计
    for split in ['train', 'val', 'test']:
        count = sum(1 for row in all_rows if row['split'] == split)
        print(f"{split}: {count} 张图片")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='提取关键点特征')
    
    # 默认使用改进后的模型路径
    default_model = '../runs/detect/keypoint_detection_improved/weights/best.pt'
    
    parser.add_argument('--model', type=str, default=default_model,
                       help=f'检测模型权重路径 (默认: {default_model})')
    parser.add_argument('--dataset', type=str, default='../classification_dataset',
                       help='数据集目录')
    parser.add_argument('--output', type=str, default='keypoint_features.csv',
                       help='输出CSV文件')
    parser.add_argument('--mapping', type=str, default='../source_mapping.json',
                       help='原始路径映射文件')

    args = parser.parse_args()

    # 检查模型是否存在，不存在则提示
    if not os.path.exists(args.model):
        print(f"警告: 模型文件不存在: {args.model}")
        print("请检查路径或先运行训练脚本 train_yolo_detection_improved.py")
        if not os.path.isabs(args.model) and not args.model.startswith('..'):
             # 尝试在当前目录查找
             if os.path.exists(os.path.join('.', args.model)):
                 args.model = os.path.join('.', args.model)
                 print(f"找到模型文件: {args.model}")

    extract_dataset_features(args.model, args.dataset, args.output, args.mapping)

