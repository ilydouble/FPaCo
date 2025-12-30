# 🤚 YOLO11 Pose 手势关键点检测项目

基于YOLO11的手势关键点检测数据集构建和训练工具。

## 📋 项目概述

本项目提供了一套完整的工具链，用于将LabelMe格式的手势关键点标注数据转换为YOLO11 Pose格式，并进行模型训练和预测。

### 特性

- ✅ 自动转换LabelMe标注为YOLO格式
- ✅ 智能数据集划分（训练/验证/测试）
- ✅ 完整的训练脚本和配置
- ✅ 预测和可视化工具
- ✅ 数据集验证工具
- ✅ 支持多种模型导出格式

## 📊 数据集统计

- **总样本数**: 2722
- **训练集**: 1902 (70%)
- **验证集**: 540 (20%)
- **测试集**: 280 (10%)
- **关键点数**: 3个/手势
- **类别数**: 9个手势类别

## 🚀 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install ultralytics pyyaml opencv-python numpy

# 或使用requirements.txt
pip install -r requirements.txt
```

### 2. 构建数据集

```bash
python build_yolo_dataset.py
```

这将创建 `yolo_hand_pose_dataset/` 目录，包含YOLO格式的训练数据。

### 3. 验证数据集

```bash
python verify_dataset.py
```

### 4. 训练模型

```bash
```bash
# 标准训练 (YOLO11s)
python train_yolo_detection.py

# 改进版训练 (提高三角点召回率)
# 使用YOLO11m + 优化增强参数
python train_yolo_detection_improved.py
```

### 5. 预测

```bash
# 单张图片
python predict_and_visualize.py --model runs/pose/hand_pose_yolo11/weights/best.pt --source image.jpg

# 批量预测
python predict_and_visualize.py --model runs/pose/hand_pose_yolo11/weights/best.pt --source images/ --batch
```

## 📁 项目结构

```
.
├── build_yolo_dataset.py          # 数据集构建脚本
├── train_yolo_detection.py        # 训练脚本 (YOLO11s)
├── train_yolo_detection_improved.py # 训练脚本 (改进版 - 提高召回率)
├── predict_and_visualize.py       # 预测和可视化
├── verify_dataset.py              # 数据集验证
├── README.md                      # 本文件
├── README_dataset.md              # 详细文档
├── QUICKSTART.md                  # 快速开始指南
├── 项目总结.md                    # 项目总结
├── 25923打标文件/                 # 原始标注数据
│   ├── 水/
│   ├── 金/
│   ├── 地/
│   └── ...
├── yolo_hand_pose_dataset/        # YOLO格式数据集
│   ├── data.yaml
│   ├── train/
│   ├── val/
│   └── test/
└── runs/                          # 训练输出
    └── pose/
        └── hand_pose_yolo11/
            └── weights/
                ├── best.pt
                └── last.pt
```

## 🎯 使用示例

### Python API

```python
from ultralytics import YOLO

# 训练
model = YOLO('yolo11n-pose.pt')
model.train(data='yolo_hand_pose_dataset/data.yaml', epochs=100)

# 预测
results = model.predict('image.jpg')

# 获取关键点
for result in results:
    keypoints = result.keypoints.xy  # 关键点坐标
    boxes = result.boxes.xyxy        # 边界框
    confs = result.boxes.conf        # 置信度
```

### 命令行

```bash
# 训练
yolo pose train data=yolo_hand_pose_dataset/data.yaml model=yolo11n-pose.pt epochs=100

# 预测
yolo pose predict model=runs/pose/hand_pose_yolo11/weights/best.pt source=image.jpg

# 验证
yolo pose val model=runs/pose/hand_pose_yolo11/weights/best.pt data=yolo_hand_pose_dataset/data.yaml
```

## 📚 文档

- [README_dataset.md](README_dataset.md) - 详细的数据集说明和使用指南
- [QUICKSTART.md](QUICKSTART.md) - 快速开始指南
- [项目总结.md](项目总结.md) - 项目总结和技术细节

## 🔧 配置参数

### 数据集构建

编辑 `build_yolo_dataset.py`:

```python
TRAIN_RATIO = 0.7           # 训练集比例
VAL_RATIO = 0.2             # 验证集比例
TEST_RATIO = 0.1            # 测试集比例
SAMPLES_PER_CATEGORY = None # 每类样本数限制
```

### 训练参数

编辑 `train_yolo_detection.py` 或直接传参:

```python
model.train(
    data='yolo_hand_pose_dataset/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    lr0=0.01,
    device=0,  # GPU ID
)
```

## 🎨 可视化

训练过程会自动生成:
- 训练曲线 (`results.png`)
- 混淆矩阵 (`confusion_matrix.png`)
- 验证集预测示例 (`val_batch*.jpg`)

## 📦 模型导出

```python
model = YOLO('runs/pose/hand_pose_yolo11/weights/best.pt')

# ONNX
model.export(format='onnx')

# TorchScript
model.export(format='torchscript')

# CoreML (iOS)
model.export(format='coreml')

# TFLite (Android)
model.export(format='tflite')
```

## 🔍 数据集验证结果

```
✓ 目录结构完整
✓ 配置文件正确
✓ 图片和标注数量匹配
✓ 标注格式正确
✓ 图片可读
✓ 数值范围有效
```

## 💡 提示

1. **GPU训练**: 确保安装了CUDA和对应版本的PyTorch
2. **内存不足**: 减小 `batch` 参数或使用更小的模型
3. **数据不平衡**: 使用 `SAMPLES_PER_CATEGORY` 限制每类样本数
4. **提高准确率**: 增加训练轮数、使用更大的模型、调整数据增强

## 📈 性能基准

| 模型 | 大小 | mAP50 | 速度 (ms) | 推荐场景 |
|------|------|-------|-----------|----------|
| yolo11n-pose | 3.3M | - | ~2 | 实时应用 |
| yolo11s-pose | 11.6M | - | ~3 | 平衡 |
| yolo11m-pose | 26.4M | - | ~5 | 高准确率 |
| yolo11l-pose | 58.9M | - | ~8 | 离线处理 |
| yolo11x-pose | 78.9M | - | ~12 | 最高准确率 |

*注: 实际性能需要在你的数据集上训练后测试*

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📄 许可证

MIT License

## 🙏 致谢

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [LabelMe](https://github.com/wkentaro/labelme)

---

## 🧠 多模态分类 (B-PACO Heatmap Early Fusion)

本项目还包含了一个高级的多模态分类模型，结合了关键点检测信息和原始图像，使用 B-PACO (Balanced Prototype and Contrastive Learning) 算法进行训练。

### 核心思想
- **Early Fusion (前融合)**: 将 YOLO 检测到的关键点转换为高斯热力图 (Heatmap)，作为图像的第三个通道 (Gray, Gray, Heatmap)。
- **ResNet Backbone**: 直接利用 ImageNet 预训练的 ResNet 提取空间特征，无需复杂的 Transformer。
- **统计特征融合**: 将显式的统计特征 (如 `is_left_hand`, `num_keypoints`) 在全连接层前融合。
- **B-PACO Loss**: 结合对比学习损失 (Contrastive Loss) 和 交叉熵损失，解决长尾分布和类内差异大问题。

### 训练

```bash
# 单次训练
python multimodal_classification/train_bpaco_heatmap.py \
    --dataset classification_dataset \
    --keypoint-features keypoint_features.csv \
    --backbone resnet18 \
    --epochs 100 \
    --output-dir results/bpaco_heatmap
```

### 自动调参 (Auto Tuning)

使用提供的脚本自动搜索最佳超参数 (Sigma, LR, Beta):

```bash
bash multimodal_classification/run_heatmap_tuning.sh
```

结果将保存在 `results/bpaco_tuning/` 目录下。

### 关键文件
- `multimodal_classification/train_bpaco_heatmap.py`: 主训练脚本
- `multimodal_classification/run_heatmap_tuning.sh`: 自动调参脚本
- `keypoint_features.csv`: 预提取的关键点统计特征

---

**开始训练你的手势检测模型吧！** 🚀

