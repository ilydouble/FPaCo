# FPaCo Project Codebase

This repository contains the experimental code for FPaCo and related baselines. It includes vision-language model adaptation methods, long-tailed supervised learning baselines, multimodal classification methods, and the proposed FPaCo training pipeline.

## Repository Structure and Methods

### 1. Vision-Language Model Adaptation
These methods use pretrained BioMedCLIP-style vision-language models for zero-shot inference or lightweight adaptation.

- **`biomedclip/`**: **Zero-Shot Baseline**
- **`tipadapter/`**: **Tip-Adapter (Training-free Adaptation)**
- **`coop/`**: **CoOp (Context Optimization)**
- **`tda/`**: **Test-Time Adaptation**
- **`dpe/`**: **Decomposed Prompt Ensemble**

### 2. Long-Tailed and Supervised Baselines
- **`paco/`**: **PaCo (Parametric Contrastive Learning)**
- **`gpaco/`**: **GPaCo (Generalized PaCo)**
- **`bpaco_original/`**: **B-PaCo (Balanced PaCo)**
- **`ce_loss/`**: **Cross Entropy Baseline**
- **`focal_loss/`**: **Focal Loss Baseline**

### 3. Multimodal and Advanced Methods
- **`multimodal_classification/`**: **Multimodal B-PaCo**

---

## Run Experiments

Enter the directory for the method you want to run, then execute the corresponding bash script.

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

### 9. FPaCo-NoHeat (B-PaCo Reduced)
```bash
cd fpaco_noheat
bash run_fpaco_noheat.sh
```

### 10. FPaCo
FPaCo uses RGB images plus a detection-guided heatmap channel. Before training, each image should have an optional same-name `.json` file containing detection boxes used to build the heatmap.

#### Reproduction Input Package

The processed datasets and trained FPaCo checkpoints/results required for reproducing the experiments are provided as a single input package:

[Download the FPaCo input package](https://drive.google.com/file/d/1Jkg9v61xhfo8Y6EKzCScRsqXmd_T0svQ/view?usp=sharing)

After downloading and extracting the package, place the extracted contents under the repository root so that the dataset folders are available under `datasets/` and the provided FPaCo checkpoints/results are available under `fpaco/results/`.

The package already contains the processed datasets and same-name `.json` detection files used to generate the FPaCo heatmap inputs. Therefore, if you use this package, you can skip the optional `.json` generation step below and run the training or evaluation scripts directly.

Expected dataset structure:
```text
datasets/<dataset_name>/
  train/
    class_0/
      image_001.png
      image_001.json
    class_1/
      image_002.png
      image_002.json
  val/
    class_0/
    class_1/
  test/
    class_0/
    class_1/
```

Each `.json` file should contain a `detections` list with bounding boxes in absolute pixel coordinates:
```json
{
  "image_path": "datasets/<dataset_name>/train/class_0/image_001.png",
  "prompt": "medical visual targets to locate",
  "detections": [
    {
      "bbox": [x1, y1, x2, y2],
      "label": "lesion",
      "class_name": "lesion",
      "confidence": 0.95
    }
  ]
}
```

If a same-name `.json` file is missing, training still runs, but the heatmap for that image will be all zeros.

#### Optional: Regenerate Detection `.json` Files

This step is optional. The following commands are only needed if you want to regenerate the `.json` files yourself, replace the provided detections, or apply FPaCo to a new dataset.

Gemini/Yunwu-based generation uses a remote vision-language API. It requires `YUNWU_API_KEY` and sends images to the API for detection:
```bash
cd fpaco
export YUNWU_API_KEY=<your_api_key>
python generate_gemini_heatmaps.py --dataset ../datasets/octa_classification_dataset --model gemini-3-flash-preview --overwrite
```

Florence-2 based generation runs a local/offline vision-language model. It does not require the Yunwu API key, but it requires the Florence-2 model dependencies and enough local compute:
```bash
cd fpaco
python generate_offline_detections.py --dataset ../datasets/fingerA --overwrite
```

The helper script `run_gen_heatmaps.sh` can be used for batch generation when regenerating detections, but check the script first because some dataset commands may be commented out:
```bash
bash run_gen_heatmaps.sh
```

After the `.json` files are prepared, run the full FPaCo experiments:
```bash
cd fpaco
bash run_fpaco.sh
```

The training script reads images and same-name `.json` files from:
```bash
datasets/aptos_classification_dataset
datasets/fingerA
datasets/fingerB
datasets/fingerC
datasets/mias_classification_dataset
datasets/octa_classification_dataset
```

Results are saved by default to:
```bash
fpaco/results/best_runs
```

To specify a GPU, set `CUDA_VISIBLE_DEVICES` before running the script:
```bash
CUDA_VISIBLE_DEVICES=0 bash run_fpaco.sh
```

### 11. FPaCo Ablation
Run the FPaCo ablation experiments after preparing the same-name `.json` detection files:
```bash
cd fpaco
bash run_fpaco_ablation.sh
```

This script runs the following ablation variants on APTOS, fingerA, MIAS, and OCTA:
- `1_naive_vlm`
- `2_mse_distill_adaptive`
- `3_static_guide`
- `4_full_adaptive`

Results are saved by default to:
```bash
fpaco/results/ablation
```

---

## Prompts and Data

- **`prompts/unified_prompts.json`**: Contains unified high-quality prompts for all datasets, following a CuPL-style prompt design.
- **`datasets/`**: Stores the processed dataset folders used by the experiments.

Make sure the required dependencies are installed and the dataset paths are configured correctly. The default dataset location used by most scripts is `../datasets` relative to each method directory.

---

# Dataset Statistics
Generated by datasets/count_stats.py

## aptos_classification_dataset

- **Total Samples**: Train: 2933, Val: 364, Test: 365

- **Imbalance Ratio (Train)**: 15.37 (Max/Min)

- **Class Distribution**:

| Class | Train | Val | Test |
| :--- | :--- | :--- | :--- |
| class_0 | 1552 | 180 | 73 |
| class_1 | 260 | 37 | 73 |
| class_2 | 827 | 99 | 73 |
| class_3 | 101 | 19 | 73 |
| class_4 | 193 | 29 | 73 |

## fingerprint_classification_dataset

- **Total Samples**: Train: 1290, Val: 270, Test: 296

- **Imbalance Ratio (Train)**: 42.60 (Max/Min)

- **Class Distribution**:

| Class | Train | Val | Test |
| :--- | :--- | :--- | :--- |
| class_0 | 147 | 31 | 32 |
| class_1 | 213 | 45 | 48 |
| class_2 | 159 | 32 | 36 |
| class_3 | 115 | 24 | 26 |
| class_4 | 38 | 8 | 9 |
| class_5 | 46 | 10 | 11 |
| class_6 | 45 | 9 | 11 |
| class_7 | 98 | 22 | 21 |
| class_8 | 70 | 15 | 15 |
| class_9 | 103 | 22 | 24 |
| class_10 | 23 | 4 | 6 |
| class_11 | 95 | 21 | 21 |
| class_12 | 25 | 5 | 7 |
| class_13 | 21 | 5 | 6 |
| class_14 | 7 | 1 | 2 |
| class_15 | 23 | 4 | 6 |
| class_16 | 5 | 1 | 2 |
| class_17 | 23 | 4 | 5 |
| class_18 | 34 | 7 | 8 |

## mias_classification_dataset

- **Total Samples**: Train: 266, Val: 28, Test: 28

- **Imbalance Ratio (Train)**: 20.33 (Max/Min)

- **Class Distribution**:

| Class | Train | Val | Test |
| :--- | :--- | :--- | :--- |
| class_0 | 183 | 20 | 4 |
| class_1 | 19 | 2 | 4 |
| class_2 | 17 | 2 | 4 |
| class_3 | 14 | 1 | 4 |
| class_4 | 9 | 1 | 4 |
| class_5 | 14 | 1 | 4 |
| class_6 | 10 | 1 | 4 |

## oral_cancer_classification_dataset

- **Total Samples**: Train: 107, Val: 12, Test: 12

- **Imbalance Ratio (Train)**: 2.34 (Max/Min)

- **Class Distribution**:

| Class | Train | Val | Test |
| :--- | :--- | :--- | :--- |
| class_0 | 32 | 6 | 6 |
| class_1 | 75 | 6 | 6 |

## octa_classification_dataset

- **Total Samples**: Train: 404, Val: 47, Test: 49

- **Imbalance Ratio (Train)**: 109.50 (Max/Min)

- **Class Distribution**:

| Class | Train | Val | Test |
| :--- | :--- | :--- | :--- |
| class_0 | 219 | 25 | 7 |
| class_1 | 8 | 1 | 7 |
| class_2 | 51 | 6 | 7 |
| class_3 | 38 | 4 | 7 |
| class_4 | 80 | 9 | 7 |
| class_5 | 2 | 1 | 7 |
| class_6 | 6 | 1 | 7 |

## fingerA

- **Total Samples**: Train: 3337, Val: 390, Test: 468

- **Imbalance Ratio (Train)**: 30.00 (Max/Min)

- **Class Distribution**:

| Class | Train | Val | Test |
| :--- | :--- | :--- | :--- |
| class_0 | 8 | 0 | 26 |
| class_1 | 210 | 26 | 26 |
| class_2 | 210 | 26 | 26 |
| class_3 | 210 | 26 | 26 |
| class_4 | 210 | 26 | 26 |
| class_5 | 172 | 0 | 26 |
| class_6 | 210 | 26 | 26 |
| class_7 | 7 | 0 | 26 |
| class_8 | 210 | 26 | 26 |
| class_9 | 210 | 26 | 26 |
| class_10 | 210 | 26 | 26 |
| class_11 | 210 | 26 | 26 |
| class_12 | 210 | 26 | 26 |
| class_13 | 210 | 26 | 26 |
| class_14 | 210 | 26 | 26 |
| class_15 | 210 | 26 | 26 |
| class_16 | 210 | 26 | 26 |
| class_17 | 210 | 26 | 26 |

## fingerB

- **Total Samples**: Train: 2402, Val: 206, Test: 468

- **Imbalance Ratio (Train)**: 73.43 (Max/Min)

- **Class Distribution**:

| Class | Train | Val | Test |
| :--- | :--- | :--- | :--- |
| class_0 | 8 | 0 | 26 |
| class_1 | 258 | 0 | 26 |
| class_2 | 514 | 64 | 26 |
| class_3 | 268 | 0 | 26 |
| class_4 | 332 | 41 | 26 |
| class_5 | 172 | 0 | 26 |
| class_6 | 215 | 26 | 26 |
| class_7 | 7 | 0 | 26 |
| class_8 | 139 | 17 | 26 |
| class_9 | 112 | 14 | 26 |
| class_10 | 90 | 11 | 26 |
| class_11 | 72 | 9 | 26 |
| class_12 | 58 | 7 | 26 |
| class_13 | 47 | 5 | 26 |
| class_14 | 37 | 4 | 26 |
| class_15 | 30 | 3 | 26 |
| class_16 | 24 | 3 | 26 |
| class_17 | 19 | 2 | 26 |

## fingerC

- **Total Samples**: Train: 2256, Val: 187, Test: 468

- **Imbalance Ratio (Train)**: 77.86 (Max/Min)

- **Class Distribution**:

| Class | Train | Val | Test |
| :--- | :--- | :--- | :--- |
| class_0 | 8 | 0 | 26 |
| class_1 | 258 | 0 | 26 |
| class_2 | 545 | 68 | 26 |
| class_3 | 268 | 0 | 26 |
| class_4 | 327 | 40 | 26 |
| class_5 | 172 | 0 | 26 |
| class_6 | 196 | 24 | 26 |
| class_7 | 7 | 0 | 26 |
| class_8 | 117 | 14 | 26 |
| class_9 | 90 | 11 | 26 |
| class_10 | 70 | 8 | 26 |
| class_11 | 54 | 6 | 26 |
| class_12 | 42 | 5 | 26 |
| class_13 | 32 | 4 | 26 |
| class_14 | 25 | 3 | 26 |
| class_15 | 19 | 2 | 26 |
| class_16 | 15 | 1 | 26 |
| class_17 | 11 | 1 | 26 |
