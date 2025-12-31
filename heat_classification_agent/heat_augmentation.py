#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Includes YOLO-style augmentation adapted for 4rd-channel heatmap integration.
"""

import random
import numpy as np
import cv2
from PIL import Image, ImageFilter
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF

class YOLOAugmentation:
    """YOLO风格的数据增强 (adapted for heat classification)"""

    @staticmethod
    def hsv_augment(image, hgain=0.015, sgain=0.7, vgain=0.4):
        """HSV色彩空间增强"""
        # image 预期为 PIL Image RGB
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
            noise = np.random.normal(0, 10, img_np.shape)
            img_np = np.clip(img_np + noise, 0, 255)
        elif noise_type == 'salt_pepper':
            prob = 0.01
            mask = np.random.random(img_np.shape[:2])
            img_np[mask < prob/2] = 0
            img_np[mask > 1 - prob/2] = 255

        return Image.fromarray(img_np.astype(np.uint8))
