# 🧠 Multimodal B-PACO Classification

本目录包含用于指纹/手势细粒度分类的高级算法实现。我们核心的策略是将 **YOLO检测到的关键点信息** 与 **原始图像** 相结合，并利用 **B-PACO** (Balanced Prototype and Contrastive Learning) 算法来处理长尾分布问题。

## 🏆 推荐方法: Heatmap Early Fusion (热力图前融合)

这是目前的**最佳实践**方案。

### 核心思想
我们将关键点检测结果转换为**空间热力图 (Gaussian Heatmap)**，并将其作为图像的**第三个通道**叠加到灰度图上，形成 `[Gray, Gray, Heatmap]` 的 3 通道输入。这使得我们可以直接利用 ImageNet 预训练的 CNN (如 ResNet) 强大的空间特征提取能力，而无需引入复杂的 Transformer 结构。

同时，我们还在全连接层前**Late Fusion (后融合)** 了显式的统计特征 (如 `is_left_hand`, `num_keypoints`)，进一步增强分类线索。

### 🚀 快速开始

#### 1. 准备特征文件
虽然 Heatmap 是实时生成的，但统计特征需要预先提取为 CSV 文件。
```bash
python multimodal_classification/extract_keypoint_features.py \
    --detections-dir classification_dataset \
    --output keypoint_features.csv
```

#### 2. 开始训练 (Heatmap B-PACO)
```bash
python multimodal_classification/train_bpaco_heatmap.py \
    --dataset classification_dataset \
    --keypoint-features keypoint_features.csv \
    --output-dir results/bpaco_heatmap \
    --backbone resnet18 \
    --heatmap-sigma 15 \
    --epochs 100
```

#### 3. 自动调参 (Auto Tuning)
使用提供的 Shell 脚本自动搜索最佳超参数 (Sigma, LR, Beta):
```bash
bash multimodal_classification/run_heatmap_tuning.sh
```

---

## 🥈 备选方法: Multimodal Late Fusion (多模态后融合)

这是早期的尝试方案，使用 Transformer 或 MLP 将图像特征 (ResNet) 与关键点特征 (CSV) 在深层进行拼接。

**适用场景**: 当你想研究纯向量特征融合的效果，或者做对比实验时。

```bash
python multimodal_classification/train_bpaco_multimodal.py \
    --dataset classification_dataset \
    --keypoint-features keypoint_features.csv \
    --backbone resnet50 \
    --epochs 100
```

---

## 📂 文件说明

| 文件名 | 类型 | 说明 |
|--------|------|------|
| **`train_bpaco_heatmap.py`** | 🐍 脚本 | **[核心]** 基于热力图前融合的主训练脚本。 |
| **`run_heatmap_tuning.sh`** | 🐚 脚本 | **[核心]** 用于搜索最佳 Heatmap Sigma 和 LR 的自动化脚本。 |
| `train_bpaco_multimodal.py` | 🐍 脚本 | [备选] 基于向量拼接的旧版多模态训练脚本。 |
| `extract_keypoint_features.py` | 🐍 脚本 | [工具] 遍历数据集 JSON，提取统计特征生成 CSV。 |
| `dataset.py` | 🐍 模块 | 基础数据集定义 (被部分旧脚本使用)。 |
| `multimodal_model.py` | 🐍 模块 | 定义了旧版 Feature Fusion 的模型结构。 |
| `focal_loss.py` | 🐍 模块 | 损失函数实现 (Focal Loss, Balanced Softmax)。 |

## 📊 方法对比

| 特性 | Heatmap Fusion (新) | Multimodal Fusion (旧) |
|------|--------------------|-----------------------|
| **输入格式** | Image (3ch) + Stat (Vec) | Image (Vec) + Keypoint (Vec) |
| **空间感知** | **强** (CNN直接处理热力图) | 弱 (仅依赖坐标数值) |
| **模型结构** | 标准 ResNet (易于训练) | Custom Fusion Module (难收敛) |
| **数据增强** | **完美同步** (图与热力图一起变换) | 困难 (需手动对齐坐标变换) |
| **推荐指数** | ⭐⭐⭐⭐⭐ | ⭐⭐ |

## 🔧 常见问题

**Q: 为什么生成的图片是 3 通道的？**
A: 原始指纹是灰度的 (1通道)。我们将它复制为前两个通道，第三个通道放入生成的关键点热力图。这样刚好符合 ImageNet 预训练模型 (ResNet) 对 3 通道输入的预期。

**Q: `heatmap-sigma` 参数有什么用？**
A: 它控制热力图上高斯光斑的大小。
*   `sigma=10`: 光斑很小，位置要求非常精确。
*   `sigma=20`: 光斑很大，容忍检测误差，提供模糊的空间提示。
建议使用 `run_heatmap_tuning.sh` 自动测试哪个效果最好。
