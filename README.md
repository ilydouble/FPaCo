# FPaCo Project Codebase

本项目包含了 FPaCo 论文相关的实验代码，涵盖了多种基于视觉-语言模型（VLM）的适配方法、长尾分布学习方法以及多模态融合算法。

## 📂 目录结构与方法介绍

### 1. 视觉-语言模型适配 (VLM Adaptation)
利用预训练的 BioMedCLIP 模型进行零样本推理或微调适配。

- **`biomedclip/`**: **Zero-Shot Baseline (零样本基准)**
- **`tipadapter/`**: **Tip-Adapter (Training-free Adaptation)**
- **`coop/`**: **CoOp (Context Optimization)**
- **`tda/`**: **Test-Time Adaptation (测试时适配)**
- **`dpe/`**: **Decomposed Prompt Ensemble (分解提示词集成)**

### 2. 长尾分布与监督学习基准 (Long-Tail & Supervised Baselines)
- **`paco/`**: **PaCo (Parametric Contrastive Learning)**
- **`gpaco/`**: **GPaCo (Generalized PaCo)**
- **`bpaco_original/`**: **B-PaCo (Balanced PaCo)**
- **`ce_loss/`**: **Cross Entropy Baseline**
- **`focal_loss/`**: **Focal Loss Baseline**

### 3. 多模态与高级方法 (Multimodal & Advanced)
- **`multimodal_classification/`**: **Multimodal B-PaCo (多模态 B-PaCo)**

---

## 🚀 实验运行方法 (Run Experiments)

请进入对应的方法文件夹，执行相应的 bash 脚本即可开始实验。

### 1. BioMedCLIP Zero-Shot
```bash
cd biomedclip
bash run_biomedclip_experiments.sh
```

### 2. Tip-Adapter
```bash
cd tipadapter
bash run_biomed_tipadapter.sh
```

### 3. CoOp
```bash
cd coop
bash run_coop_biomed.sh
```

### 4. Test-Time Adaptation (TDA)
```bash
cd tda
bash run_tda.sh
```

### 5. Decomposed Prompt Ensemble (DPE)
```bash
cd dpe
bash run_dpe_biomed.sh
```

### 6. PaCo (Parametric Contrastive Learning)
```bash
cd paco
bash run_paco_experiments.sh
```

### 7. GPaCo (Generalized PaCo)
```bash
cd gpaco
bash run_gpaco_experiments.sh
```

### 8. B-PaCo (Balanced PaCo - Original)
```bash
cd bpaco_original
bash run_bpaco_reproduced.sh
```

### 9. Multimodal B-PaCo Heatmap
```bash
cd multimodal_classification
# 自动调参运行
bash run_heatmap_tuning.sh
# 或单次训练
python train_bpaco_heatmap.py --dataset ...
```

---

## 🔧 提示词与数据集 (Prompts & Data)

- **`prompts/unified_prompts.json`**: 包含所有数据集的统一、高质量提示词（CuPL Style）。
- **`datasets/`**: 存放数据集文件。

请确保环境已安装必要的依赖库，并且数据集路径配置正确（默认为 `../datasets`）。
