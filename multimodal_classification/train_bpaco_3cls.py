import os
import random
import json
from pathlib import Path
from typing import Tuple, List

import numpy as np
import cv2
from PIL import Image
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import torchvision.models as models

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
import matplotlib.pyplot as plt

# =========================================================
# 形态学增强变换（开运算取反 × 二值化）
# =========================================================

class FingerprintMorphologyTransform:
    """
    将单通道灰度指纹张量做形态学增强，复现：
        顶帽 -> 局部阈值二值化 -> 黑帽 -> 黑帽开运算 -> 取反 × 二值图

    输入:  x  shape (1, H, W), 值域 [0,1]
    输出:  同 shape, 增强后的 0~1 浮点张量
    """

    def __init__(
        self,
        kernel_tophat: int = 15,
        kernel_blackhat: int = 25,
        kernel_open: int = 3,
        block_size: int = 21,
        C: int = 8,
    ) -> None:
        self.kernel_tophat = kernel_tophat
        self.kernel_blackhat = kernel_blackhat
        self.kernel_open = kernel_open
        self.block_size = block_size
        self.C = C

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # 期望输入是 (1, H, W) 的灰度图，值域 [0,1]
        if x.dim() != 3 or x.size(0) != 1:
            raise ValueError(
                f"FingerprintMorphologyTransform 期望输入为 (1,H,W)，得到 {tuple(x.shape)}"
            )

        # 转为 uint8 图像
        img = x.squeeze(0).cpu().numpy()
        img_u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)

        # 1) 顶帽去背景
        k_top = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.kernel_tophat, self.kernel_tophat)
        )
        tophat = cv2.morphologyEx(img_u8, cv2.MORPH_TOPHAT, k_top)
        if tophat.max() > 0:
            tophat_enh = cv2.normalize(tophat, None, 0, 255, cv2.NORM_MINMAX)
        else:
            tophat_enh = tophat

        # 2) 自适应阈值 -> 二值指纹
        binary = cv2.adaptiveThreshold(
            tophat_enh,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            self.block_size,
            self.C,
        )

        # 3) 黑帽提取暗 ridge 间隙
        k_bh = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.kernel_blackhat, self.kernel_blackhat)
        )
        blackhat = cv2.morphologyEx(img_u8, cv2.MORPH_BLACKHAT, k_bh)
        if blackhat.max() > 0:
            blackhat = cv2.normalize(blackhat, None, 0, 255, cv2.NORM_MINMAX)

        blackhat_bin = cv2.adaptiveThreshold(
            blackhat,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            self.block_size,
            self.C,
        )

        # 4) 黑帽结果做开运算，去除孤立噪声
        k_open = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.kernel_open, self.kernel_open)
        )
        blackhat_open = cv2.morphologyEx(
            blackhat_bin, cv2.MORPH_OPEN, k_open, iterations=1
        )

        # 5) 开运算结果取反，与原二值图相乘
        opening_inv = cv2.bitwise_not(blackhat_open)
        result = cv2.bitwise_and(opening_inv, binary)

        # 回到 (1,H,W) 的 float tensor，范围 0~1
        result_f = torch.from_numpy(result.astype(np.float32) / 255.0)
        if result_f.dim() == 2:
            result_f = result_f.unsqueeze(0)
        return result_f


# =========================================================
# 设置随机种子，保证训练可复现
# =========================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# 自动检测 GPU 或 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FingerprintDataset(Dataset):
    """
    每次返回：
        - v1：第一份增强视图（由 aug1 生成）
        - v2：第二份增强视图（由 aug2 生成）
        - label：类别标签

    特点：
    1. 支持形态学增强（黑帽开运算取反 × 二值化）
    2. 支持自动划分 train/val
    3. 输出经过标准化的 3 通道图像（适配 ImageNet 预训练 backbone）
    """

    def __init__(self, root: str, split: str = "train", image_size: int = 224):
        super().__init__()

        root = Path(root)
        assert root.exists(), f"❌ 数据目录不存在: {root}"

        # 类别顺序固定（粗粒度 3 类）
        classes = [d.name for d in root.iterdir() if d.is_dir()]
        classes.sort()
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
        self.num_classes = len(classes)  # 添加num_classes属性

        # 读取全部样本路径 - 支持子文件夹结构
        self.samples: List[Tuple[str, int]] = []
        for cls_name in classes:
            cls_path = root / cls_name
            if not cls_path.exists():
                continue

            # 遍历类别下的子文件夹
            for subfolder in cls_path.iterdir():
                if not subfolder.is_dir():
                    # 如果不是目录，检查是否是图像文件（直接放在类别文件夹下的情况）
                    if subfolder.suffix.lower() in [ ".jpg", ".jpeg", ".png", ]:
                        self.samples.append(
                            (str(subfolder), self.class_to_idx[cls_name])
                        )
                    continue

                # 遍历子文件夹下的图像文件
                for img_file in subfolder.iterdir():
                    if img_file.suffix.lower() in [ ".jpg", ".jpeg", ".png", ]:
                        self.samples.append(
                            (str(img_file), self.class_to_idx[cls_name])
                        )

        # 进行 train/val 划分
        train_files, val_files = train_test_split(
            self.samples,
            test_size=0.15,
            random_state=SEED,
            stratify=[lbl for _, lbl in self.samples],
        )

        self.samples = train_files if split == "train" else val_files

        # 归一化（这里用简单 0.5/0.5，指纹是黑白图，太讲究 ImageNet 均值反而不好）
        self.normalize = T.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        )

        # 形态学滤波变换实例（开运算取反 × 二值化）
        self.morph = FingerprintMorphologyTransform()

        # 两个视图的增强策略
        # v1：带随机模糊/翻转 + 随机形态学增强，用于学习鲁棒特征
        self.aug1 = T.Compose(
            [
                T.Resize((image_size, image_size)),
                T.RandomHorizontalFlip(),
                T.RandomApply([T.GaussianBlur(3)], p=0.2),
                T.ToTensor(),  # -> (1, H, W) 灰度
                T.RandomApply([self.morph], p=0.7),
                T.Lambda(lambda x: x.repeat(3, 1, 1)),  # 灰度复制为 3 通道
                self.normalize,
            ]
        )

        # v2：始终使用形态学增强，得到“干净”的结构视图
        self.aug2 = T.Compose(
            [
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                self.morph,
                T.Lambda(lambda x: x.repeat(3, 1, 1)),
                self.normalize,
            ]
        )

    # 返回样本数量
    def __len__(self):
        return len(self.samples)

    # 返回增强后的 v1、v2 和 标签
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # 打开图像（灰度指纹图，按灰度读取）
        img = Image.open(img_path).convert("L")

        # 两份不同增强视图
        v1 = self.aug1(img)
        v2 = self.aug2(img)

        return v1, v2, torch.tensor(label, dtype=torch.long)


class ALWAnnotationsDataset(Dataset):
    """
    使用 annotations/*.json 中的 f_code 首字母 (A/L/W) 作为三分类标签，
    图像路径来自同一标注的 image_filename，图像位于 images/ 目录。
    """

    def __init__(self, images_dir: str = "images", annotations_dir: str = "annotations", image_size: int = 224):
        super().__init__()
        self.images_dir = Path(images_dir)
        self.annotations_dir = Path(annotations_dir)
        assert self.images_dir.exists(), f"原始图像目录不存在: {self.images_dir}"
        assert self.annotations_dir.exists(), f"标注目录不存在: {self.annotations_dir}"

        label_map = {"A": 0, "L": 1, "W": 2}
        self.samples: List[Tuple[str, int]] = []

        for ann_file in sorted(self.annotations_dir.glob("*.json")):
            try:
                with open(ann_file, "r", encoding="utf-8") as f:
                    ann_data = json.load(f)
            except Exception:
                continue

            f_code = ann_data.get("f_code", "")
            if not f_code:
                continue
            family = f_code[0].upper()
            if family not in label_map:
                continue

            image_filename = ann_data.get("image_filename")
            if not image_filename:
                continue
            image_path = self.images_dir / image_filename
            if not image_path.exists():
                continue

            self.samples.append((str(image_path), label_map[family]))

        self.num_classes = len(label_map)
        if len(self.samples) == 0:
            raise ValueError("未找到可用的 A/L/W 样本，请检查 images/ 与 annotations/ 是否匹配。")

        self.normalize = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.morph = FingerprintMorphologyTransform()

        self.aug1 = T.Compose(
            [
                T.Resize((image_size, image_size)),
                T.RandomHorizontalFlip(),
                T.RandomApply([T.GaussianBlur(3)], p=0.2),
                T.ToTensor(),
                T.RandomApply([self.morph], p=0.7),
                T.Lambda(lambda x: x.repeat(3, 1, 1)),
                self.normalize,
            ]
        )
        self.aug2 = T.Compose(
            [
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                self.morph,
                T.Lambda(lambda x: x.repeat(3, 1, 1)),
                self.normalize,
            ]
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("L")
        v1 = self.aug1(img)
        v2 = self.aug2(img)
        return v1, v2, torch.tensor(label, dtype=torch.long)


# =========================================================
# 模块 2：编码器 BpacoEncoder + 分类器权重映射器 (C2 Mapper)
# =========================================================


class BpacoEncoder(nn.Module):
    """
    BpacoEncoder（编码器 backbone）

    输出：
        - feat：用于分类的特征（backbone 最后一层池化结果）
        - z   ：投影头输出的特征（归一化后用于对比学习）
    """

    def __init__(self, backbone="resnet34", proj_dim=128, pretrained=True):
        super().__init__()

        # ---------------------
        # 选择 backbone
        # ---------------------
        if backbone == "resnet18":
            self.backbone = models.resnet18(
                weights=models.ResNet18_Weights.DEFAULT if pretrained else None
            )
            self.feat_dim = self.backbone.fc.in_features
            modules = list(self.backbone.children())[:-1]
            self.encoder = nn.Sequential(
                *modules
            )  # 输出尺寸 (B, feat_dim, 1, 1)
        elif backbone == "resnet34":
            self.backbone = models.resnet34(
                weights=models.ResNet34_Weights.DEFAULT if pretrained else None
            )
            self.feat_dim = self.backbone.fc.in_features
            modules = list(self.backbone.children())[:-1]
            self.encoder = nn.Sequential(
                *modules
            )  # 输出尺寸 (B, feat_dim, 1, 1)
        elif backbone == "resnet50":
            self.backbone = models.resnet50(
                weights=models.ResNet50_Weights.DEFAULT if pretrained else None
            )
            self.feat_dim = self.backbone.fc.in_features
            modules = list(self.backbone.children())[:-1]
            self.encoder = nn.Sequential(
                *modules
            )  # 输出尺寸 (B, feat_dim, 1, 1)
        elif backbone == "efficientnet_b0":
            self.backbone = models.efficientnet_b0(
                weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            )
            self.feat_dim = self.backbone.classifier[1].in_features

            # 移除 EfficientNet 最后的分类器层，保留特征提取部分
            modules = list(self.backbone.children())[:-1]  # 移除最后的分类器层
            self.encoder = nn.Sequential(*modules)
        elif backbone == "efficientnet_b1":
            self.backbone = models.efficientnet_b1(
                weights=models.EfficientNet_B1_Weights.DEFAULT if pretrained else None
            )
            self.feat_dim = self.backbone.classifier[1].in_features

            # 移除 EfficientNet 最后的分类器层，保留特征提取部分
            modules = list(self.backbone.children())[:-1]  # 移除最后的分类器层
            self.encoder = nn.Sequential(*modules)
        elif backbone == "efficientnet_b2":
            self.backbone = models.efficientnet_b2(
                weights=models.EfficientNet_B2_Weights.DEFAULT if pretrained else None
            )
            self.feat_dim = self.backbone.classifier[1].in_features

            # 移除 EfficientNet 最后的分类器层，保留特征提取部分
            modules = list(self.backbone.children())[:-1]  # 移除最后的分类器层
            self.encoder = nn.Sequential(*modules)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # ---------------------
        # Projection Head（投影头）
        # ---------------------
        self.proj = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.feat_dim, proj_dim),
        )

    def forward(self, x):
        """
        前向传播：
        """

        # Backbone输出是 (B, feat_dim, 1, 1)，需 reshape
        f = self.encoder(x).reshape(x.size(0), -1)  # -> (B, feat_dim)

        # Projection head 输出 z（未归一化）
        z = self.proj(f)

        # 归一化，适配对比学习
        z = F.normalize(z, dim=1)

        return f, z


# 分类器权重映射器：ClassifierToProtoMapper
class ClassifierToProtoMapper(nn.Module):
    """
    将分类器权重矩阵 W（K × D）
    映射到投影空间 out_dim（与 z 维度一致），得到原型 C2
    """

    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.map = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, W):
        return F.normalize(self.map(W), dim=1)


# =========================================================
# 模块 3：特征队列 FeatureQueue + 动量更新函数 momentum_update
# =========================================================


class FeatureQueue:
    """
    BPaCo / MoCo 风格的特征队列
    """

    def __init__(self, feat_dim, queue_size, device="cuda"):
        self.queue_size = queue_size
        self.device = device

        self.feats = torch.zeros(queue_size, feat_dim, device=device)
        self.labels = -1 * torch.ones(
            queue_size, dtype=torch.long, device=device
        )  # -1 表示无效

        self.ptr = 0
        self.full = False

    @torch.no_grad()
    def enqueue(self, feats, labels):
        B = feats.shape[0]

        if B >= self.queue_size:
            feats = feats[-self.queue_size :]
            labels = labels[-self.queue_size :]
            B = feats.shape[0]

        idx = (self.ptr + torch.arange(B, device=feats.device)) % self.queue_size

        self.feats[idx] = feats.detach()
        self.labels[idx] = labels.detach()

        self.ptr = (self.ptr + B) % self.queue_size
        if self.ptr == 0:
            self.full = True

    def get(self):
        if (not self.full) and (self.ptr == 0):
            return None, None

        if self.full:
            return self.feats, self.labels
        else:
            return self.feats[: self.ptr], self.labels[: self.ptr]


@torch.no_grad()
def momentum_update(model_k, model_q, momentum):
    """
    EMA 动量更新：θ_k ← m * θ_k + (1 - m) * θ_q
    """
    for param_k, param_q in zip(model_k.parameters(), model_q.parameters()):
        param_k.data = param_k.data * momentum + param_q.data * (1.0 - momentum)


# =========================================================
# 模块 4：BPaCo 核心损失（LBPaCo） + Logit Compensation
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
    Balanced Prototype & Contrastive Loss (简化实现)
    使用三类原型：
        - 跨批 class mean
        - 可学习中心 C1
        - 分类器权重映射得到的 C2
    """

    B, d = z_batch.shape
    q_feats, q_labels = queue.get()

    if q_feats is None:
        A_feats = z_batch
        A_labels = labels_batch
    else:
        q_feats = q_feats.to(device)
        q_labels = q_labels.to(device)
        A_feats = torch.cat([z_batch, q_feats], dim=0)
        A_labels = torch.cat([labels_batch, q_labels], dim=0)

    K = C1_param.shape[0]
    device = z_batch.device

    # 2）跨批类别均值
    class_sums = torch.zeros(K, d, device=device)
    class_counts = torch.zeros(K, device=device)

    for k in range(K):
        mask = A_labels == k
        if mask.any():
            class_sums[k] = A_feats[mask].sum(dim=0)
            class_counts[k] = mask.sum()

    valid_mask = class_counts > 0
    class_means = torch.zeros_like(class_sums)
    if valid_mask.any():
        class_means[valid_mask] = class_sums[valid_mask] / class_counts[
            valid_mask
        ].unsqueeze(1)

    # 3）C1 归一化
    C1 = F.normalize(C1_param, dim=1)

    # 4）从分类器提取权重，映射为 C2
    last_linear = None
    for m in classifier.modules():
        if isinstance(m, nn.Linear):
            last_linear = m
    if last_linear is None:
        W = torch.zeros(K, d, device=device)
    else:
        W = last_linear.weight.detach()
        if W.device != device:
            W = W.to(device)

    try:
        C2 = mapper(W)
    except Exception:
        print(f"Warning: Mapper 失败, using random initialization: {e}")

    # 5）计算与每一类原型的相似度
    sims_per_class = torch.zeros(B, K, device=device)
    for j in range(K):
        reps = []
        if valid_mask[j]:
            reps.append(F.normalize(class_means[j].unsqueeze(0), dim=1))
        reps.append(C1[j].unsqueeze(0))
        reps.append(C2[j].unsqueeze(0))
        reps_cat = torch.cat(reps, dim=0)  # (m_j, d)

        sims = torch.matmul(z_batch, reps_cat.t()) / temperature
        sims_exp = torch.exp(sims)
        sims_avg = sims_exp.mean(dim=1)
        sims_per_class[:, j] = sims_avg

    numerators = sims_per_class[
        torch.arange(B, device=device), labels_batch
    ] + 1e-12
    denominators = sims_per_class.sum(dim=1) + 1e-12
    loss = -torch.log(numerators / denominators).mean()
    return loss


def cross_entropy_with_logit_compensation(logits, labels, class_freq_tensor, tau):
    """
    Logit Compensation：
        logits_adj = logits - tau * log(class_freq + 1)
    用来缓解长尾分布下多数类的优势
    """
    eps = 1e-12
    adjustment = tau * torch.log(
        class_freq_tensor.float().to(logits.device) + 1.0 + eps
    )
    logits_adj = logits - adjustment.unsqueeze(0)
    return F.cross_entropy(logits_adj, labels)


# =========================================================
# 模块 5：ContrastiveClassifier（训练器）
# =========================================================


class ContrastiveClassifier:
    """
    训练器封装类：包含模型、队列、优化器、训练/验证流程
    """

    def __init__(
        self,
        images_dir: str = "images",
        annotations_dir: str = "annotations",
        output_dir: str = "./results/fingerprint_classifier_results",
        backbone: str = "resnet50",
        out_dim=128,
        queue_size=4096,
        momentum=0.999,
        beta=2.0,
        tau=1.2,
        temperature=0.1,
        num_classes=3
    ):
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.device = device
        self.run_dir = output_dir
        self.backbone = backbone
        self.out_dim = out_dim
        self.queue_size = queue_size
        self.momentum = momentum
        self.beta = beta
        self.tau = tau
        self.temperature = temperature

        self.base_output_dir = Path(output_dir)

        temp_ds = ALWAnnotationsDataset(images_dir, annotations_dir, image_size=224)
        self.num_classes = temp_ds.num_classes # 动态获取

        # 1) 初始化 query / key 编码器
        self.model_q = BpacoEncoder(backbone, proj_dim=out_dim, pretrained=True).to(
            self.device
        )
        self.model_k = BpacoEncoder(backbone, proj_dim=out_dim, pretrained=True).to(
            self.device
        )

        for param_q, param_k in zip(
            self.model_q.parameters(), self.model_k.parameters()
        ):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # 2) 分类器：feat_for_cls = concat(feat_q, feat_k)
        feat_dim = self.model_q.feat_dim
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, self.num_classes),
        ).to(self.device)

        # 3) 可学习类别中心 C1
        self.C1 = nn.Parameter(
            torch.randn(self.num_classes, out_dim, device=self.device)
        )

        # 4) 分类器权重到原型的 mapper（C2）
        self.mapper = ClassifierToProtoMapper(
            in_dim=self.classifier[-1].in_features, out_dim=out_dim
        ).to(self.device)

        # 5) 特征队列
        self.queue = FeatureQueue(
            feat_dim=out_dim, queue_size=queue_size, device=self.device
        )

        # 6) 优化器
        params = (
            list(self.model_q.parameters())
            + list(self.classifier.parameters())
            + [self.C1]
            + list(self.mapper.parameters())
        )
        self.optimizer = torch.optim.SGD(
            params, lr=0.001, momentum=0.999, weight_decay=1e-4
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=50, gamma=0.1
        )

    def create_descriptive_output_dir(
        self,
        backbone,
        epochs=50,
        batch_size=128,
        queue_size=4096,
        momentum=0.999,
        beta=0.25,
        tau=1.2,
        temperature=0.1,
        proj_dim=128,
    ):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = (
            f"{backbone}-E{epochs}-B{batch_size}-queue{queue_size}-"
            f"momentum{momentum}-beta{beta}-tau{tau}-temp{temperature}-"
            f"proj{proj_dim}-{timestamp}"
        )
        output_dir = self.base_output_dir / folder_name
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def run_full_training(
        self, batch_size=16, epochs=300, image_size=224, val_interval=10
    ):
        # 1) Dataset
        base_ds = ALWAnnotationsDataset(
            images_dir=self.images_dir,
            annotations_dir=self.annotations_dir,
            image_size=image_size,
        )
        train_size = int(0.8 * len(base_ds))
        val_size = len(base_ds) - train_size
        generator = torch.Generator().manual_seed(SEED)
        train_ds, val_ds = torch.utils.data.random_split(base_ds, [train_size, val_size], generator=generator)

        # 2) Dataloader
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        # 3) 输出目录
        self.run_dir = self.create_descriptive_output_dir(
            backbone=self.backbone,
            epochs=epochs,
            batch_size=batch_size,
            queue_size=self.queue_size,
            momentum=self.momentum,
            beta=self.beta,
            tau=self.tau,
            temperature=self.temperature,
            proj_dim=self.out_dim,
        )
        print(f"\n输出目录已创建: {self.run_dir}")

        # 4) 类别频率统计（长尾补偿用）
        freq = torch.zeros(self.num_classes, dtype=torch.long)
        for _, _, lbl in DataLoader(train_ds, batch_size=1, shuffle=False):
            freq[lbl.item()] += 1
        print("类别频率统计 class freq:", freq.tolist())

        best_f1 = 0.0

        for epoch in range(1, epochs + 1):
            t0 = datetime.now()
            avg_loss = self.train_one_epoch(
                train_loader, class_freq_tensor=freq
            )
            t1 = datetime.now()
            print(
                f"\n[Epoch {epoch}] 用时 {(t1 - t0).total_seconds():.1f}s, 平均loss={avg_loss:.4f}"
            )

            if epoch % val_interval == 0:
                results = self.evaluate(val_loader)

                if results["f1"] > best_f1:
                    best_f1 = results["f1"]
                    save_dir = Path(self.run_dir)
                    save_dir.mkdir(parents=True, exist_ok=True)
                    model_save_path = save_dir / "best_bpaco_checkpoint.pth"
                    torch.save(
                        {
                            "model_q": self.model_q.state_dict(),
                            "model_k": self.model_k.state_dict(),
                            "classifier": self.classifier.state_dict(),
                            "C1": self.C1.detach().cpu(),
                            "mapper": self.mapper.state_dict(),
                        },
                        model_save_path,
                    )
                    print(
                        f"💾 已保存最佳模型（F1={best_f1:.4f}）至: {model_save_path}"
                    )

        final_results = self.evaluate(val_loader)

        print("\n================ 最终评估结果 ================")
        print(f"最终准确率 Accuracy : {final_results['accuracy']:.4f}")
        print(f"最终F1-score       : {final_results['f1']:.4f}")
        print(f"最终AUC            : {final_results['auc']:.4f}")
        print(f"最终 Precision     : {final_results['precision']:.4f}")
        print(f"最终 Recall        : {final_results['recall']:.4f}")

        with open(self.run_dir / "final_results.txt", "w", encoding="utf-8") as f:
            f.write("最终评估结果:\n")
            f.write(f"准确率 Accuracy : {final_results['accuracy']:.4f}\n")
            f.write(f"F1-score       : {final_results['f1']:.4f}\n")
            f.write(f"AUC            : {final_results['auc']:.4f}\n")
            f.write(f"Precision      : {final_results['precision']:.4f}\n")
            f.write(f"Recall         : {final_results['recall']:.4f}\n")
        print(f"评估结果已保存至: {self.run_dir / 'final_results.txt'}")
        print("================================================\n")

        return final_results

    def train_one_epoch(self, dataloader, class_freq_tensor):
        self.model_q.train()
        self.classifier.train()

        losses = []

        for step, (v1, v2, labels) in enumerate(dataloader):
            v1 = v1.to(self.device)
            v2 = v2.to(self.device)
            labels = labels.to(self.device)

            feat_q, z_q = self.model_q(v1)
            with torch.no_grad():
                feat_k, z_k = self.model_k(v2)

            feat_for_cls = torch.cat([feat_q, feat_k], dim=1)
            logits = self.classifier(feat_for_cls)

            self.optimizer.zero_grad()

            ce_loss = cross_entropy_with_logit_compensation(
                logits, labels, class_freq_tensor, tau=self.tau
            )
            lbpaco_loss = compute_LBPaCo(
                z_q,
                labels,
                self.queue,
                self.C1,
                self.classifier,
                self.mapper,
                temperature=self.temperature,
                device=self.device,
            )
            loss = ce_loss + self.beta * lbpaco_loss

            loss.backward()
            self.optimizer.step()

            momentum_update(self.model_k, self.model_q, self.momentum)

            with torch.no_grad():
                self.queue.enqueue(z_k, labels)

            losses.append(loss.item())

        self.scheduler.step()
        return np.mean(losses)

    def plot_confusion_matrix(self, y_true, y_pred):
        from sklearn.metrics import confusion_matrix, classification_report
        import seaborn as sns

        cm = confusion_matrix(y_true, y_pred)
        num_classes = cm.shape[0]
        
        # 动态生成类别名称，支持3类和19类
        if num_classes == 3:
            class_names = ["whorl", "loop", "arch"]
            title = "Confusion Matrix - Fingerprint 3-class"
        elif num_classes > 3:
            # 对于19类或更多类别，使用数字标识
            class_names = [f"Class {i}" for i in range(num_classes)]
            title = f"Confusion Matrix - Fingerprint {num_classes}-class"
        else:
            class_names = [f"Class {i}" for i in range(num_classes)]
            title = "Confusion Matrix - Fingerprint"
        
        # 对于多类别，调整图形大小
        if num_classes > 10:
            figsize = (15, 12)
        else:
            figsize = (8, 6)
        
        plt.figure(figsize=figsize)
        sns.heatmap(
            cm,
            annot=num_classes <= 10,  # 只在类别数量少时显示数值
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={"label": "Count"},
        )
        plt.title(title, fontsize=14, fontweight="bold")
        plt.ylabel("True Label", fontsize=12)
        plt.xlabel("Predicted Label", fontsize=12)
        
        # 对于多类别，调整标签旋转角度
        if num_classes > 10:
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)

        accuracy = np.trace(cm) / np.sum(cm)
        plt.figtext(0.15, 0.02, f"Overall Accuracy: {accuracy:.4f}", fontsize=10)

        plt.tight_layout()
        save_path = os.path.join(str(self.run_dir), "confusion_matrix.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

        print("\n=== 分类报告 ===")
        # 限制类别数量，避免输出过长
        if num_classes > 20:
            print(f"类别数量过多 ({num_classes})，仅显示部分评估指标")
            # 只使用前几个和后几个类别名称
            report_class_names = class_names[:5] + ["..."] + class_names[-5:]
            # 但实际报告还是包含所有类别
            report = classification_report(
                y_true, y_pred, digits=4
            )
        else:
            report = classification_report(
                y_true, y_pred, target_names=class_names, digits=4
            )
        print(report)

        report_path = os.path.join(str(self.run_dir), "classification_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"分类报告已保存至: {report_path}")

    def plot_roc_curve(self, y_true, y_scores):
        """
        绘制 ROC 曲线
        参数：
            y_true  : 形状 (N,) 的真实标签（int，0 ~ C-1）
            y_scores: 形状 (N, num_classes) 的类别概率矩阵
        说明：
            - 二分类：正常 ROC 曲线
            - 多分类（3 类 / 19 类）：画 micro-average ROC
        """
        from sklearn.metrics import roc_curve, roc_auc_score
        from sklearn.preprocessing import label_binarize

        y_true = np.array(y_true)
        y_scores = np.array(y_scores)
        num_classes = self.num_classes

        plt.figure(figsize=(8, 6))

        if num_classes == 2:
            # 二分类：直接用正类概率
            fpr, tpr, _ = roc_curve(y_true, y_scores[:, 1])
            auc_score = roc_auc_score(y_true, y_scores[:, 1])
            plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc_score:.3f})")
        else:
            # 多分类：micro-average ROC
            y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
            fpr, tpr, _ = roc_curve(y_true_bin.ravel(), y_scores.ravel())
            auc_score = roc_auc_score(
                y_true_bin,
                y_scores,
                multi_class="ovr",
                average="micro",
            )
            plt.plot(
                fpr,
                tpr,
                label=f"Micro-average ROC (AUC = {auc_score:.3f})",
            )

        # 对角线（随机分类器）
        plt.plot([0, 1], [0, 1], "k--", label="Random")

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)

        roc_path = os.path.join(str(self.run_dir), "roc_curve.png")
        plt.tight_layout()
        plt.savefig(roc_path, dpi=300, bbox_inches="tight")
        print(f"ROC 曲线已保存至: {roc_path}")
        plt.close()



    @torch.no_grad()
    def evaluate(self, dataloader):
        """
        在验证集或测试集上评估当前模型性能
        - 对任意类别数 self.num_classes 均适用（3 类 / 19 类）
        - AUC 使用多类别设置：
            * 二分类时：直接用正类概率
            * 多分类时：使用 one-vs-rest 的 micro-average AUC
        """
        self.model_q.eval()
        self.classifier.eval()

        y_true = []
        y_pred = []
        prob_list = []   # 每个样本的完整类别概率向量

        for v1, v2, labels in dataloader:
            v1 = v1.to(self.device)
            v2 = v2.to(self.device)
            labels = labels.to(self.device)

            # 前向（query + key）
            feat_q, _ = self.model_q(v1)
            feat_k, _ = self.model_k(v2)
            feat_for_cls = torch.cat([feat_q, feat_k], dim=1)
            logits = self.classifier(feat_for_cls)

            probs = torch.softmax(logits, dim=1)   # (B, num_classes)
            preds = probs.argmax(dim=1)

            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
            prob_list.append(probs.cpu().numpy())

        # 合并所有 batch 的概率
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_scores = np.concatenate(prob_list, axis=0)   # (N, num_classes)

        # --------- 各种指标 ----------
        num_classes = self.num_classes

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="macro")
        prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_true, y_pred, average="macro", zero_division=0)

        # AUC：
        try:
            if num_classes == 2:
                # 二分类：取正类（1）概率
                auc = roc_auc_score(y_true, y_scores[:, 1])
            else:
                # 多分类：使用 one-vs-rest 的 micro-average AUC
                from sklearn.preprocessing import label_binarize

                y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
                auc = roc_auc_score(
                    y_true_bin,
                    y_scores,
                    multi_class="ovr",
                    average="micro",
                )
        except Exception:
            print("AUC 计算失败")
            auc = 0.0

        # 混淆矩阵 & ROC 曲线
        self.plot_confusion_matrix(y_true.tolist(), y_pred.tolist())
        self.plot_roc_curve(y_true, y_scores)

        return {
            "accuracy": float(acc),
            "f1": float(f1),
            "auc": float(auc),
            "precision": float(prec),
            "recall": float(rec),
            "y_true": y_true.tolist(),
            "y_pred": y_pred.tolist(),
        }



# =========================================================
# 模块 6：主程序 main()
# =========================================================


def main():
    output_dir = "./results/fingerprint_bpaco_3cls_results"

    model = ContrastiveClassifier(
        images_dir="images",
        annotations_dir="annotations",
        output_dir=output_dir,
        backbone="resnet50",
        out_dim=128,
        queue_size=4096,
        momentum=0.999,
        beta=1.5, # BPaCo 损失的权重系数,beta 大 → 更重视对比学习;beta 小 → 更重视分类器 CE。 长尾效应严重就增大(0.5 ～ 2.0)
        tau=1.2,
        temperature=0.1,
    )
    model.run_full_training(
        batch_size=128,
        epochs=100,
        image_size=224,
        val_interval=10,
    )


if __name__ == "__main__":
    main()
