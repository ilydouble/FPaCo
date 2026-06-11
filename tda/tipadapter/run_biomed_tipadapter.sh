#!/bin/bash

# Directories
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
WORKSPACE="$(dirname "$SCRIPT_DIR")"
# Datasets are at WORKSPACE/datasets
DATASET_ROOT="$WORKSPACE/datasets"

export CUDA_VISIBLE_DEVICES=0

echo "=========================================================="
echo "Starting BioMedCLIP Tip-Adapter Experiments (16-shot)"
echo "=========================================================="

# 1. MIAS
echo "[1/5] Running on MIAS..."
python "$SCRIPT_DIR/biomed_main.py" \
    --dataset mias \
    --root_path "$DATASET_ROOT" \
    --shots 16 \
    --augment_epoch 10 \
    --train_epoch 20 \
    --lr 0.001

# 2. OralCancer
echo "[2/5] Running on OralCancer..."
python "$SCRIPT_DIR/biomed_main.py" \
    --dataset oral_cancer \
    --root_path "$DATASET_ROOT" \
    --shots 16 \
    --augment_epoch 10 \
    --train_epoch 20 \
    --lr 0.001

# 3. APTOS
echo "[3/5] Running on APTOS..."
python "$SCRIPT_DIR/biomed_main.py" \
    --dataset aptos \
    --root_path "$DATASET_ROOT" \
    --shots 16 \
    --augment_epoch 10 \
    --train_epoch 20 \
    --lr 0.001

# 4. Fingerprint
echo "[4/5] Running on Fingerprint..."
python "$SCRIPT_DIR/biomed_main.py" \
    --dataset finger \
    --root_path "$DATASET_ROOT" \
    --shots 16 \
    --augment_epoch 10 \
    --train_epoch 20 \
    --lr 0.001

# 5. OCTA
echo "[5/5] Running on OCTA..."
python "$SCRIPT_DIR/biomed_main.py" \
    --dataset octa \
    --root_path "$DATASET_ROOT" \
    --shots 16 \
    --augment_epoch 10 \
    --train_epoch 20 \
    --lr 0.001

echo "=========================================================="
echo "All Tip-Adapter experiments completed."
