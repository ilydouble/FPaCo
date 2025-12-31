#!/bin/bash

# Directories
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
WORKSPACE="$(dirname "$SCRIPT_DIR")"
RESULTS_ROOT="$WORKSPACE/ce_loss/results"
mkdir -p "$RESULTS_ROOT"

# GPU
export CUDA_VISIBLE_DEVICES=0

# Hyperparams
EPOCHS=100
BATCH_SIZE=32
LR=0.001

echo "=========================================================="
echo "Starting CE Baseline Experiments (ResNet18 + CE Loss)"
echo "=========================================================="

# 1. MIAS
echo "[1/5] Training on MIAS..."
python3 train_ce_standard.py \
    --dataset "$WORKSPACE/datasets/mias_classification_dataset" \
    --output-dir "$RESULTS_ROOT/mias" \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --lr $LR \
    --merge-train-val

# 2. OralCancer
echo "[2/5] Training on OralCancer..."
python3 train_ce_standard.py \
    --dataset "$WORKSPACE/datasets/oral_cancer_classification_dataset" \
    --output-dir "$RESULTS_ROOT/oral_cancer" \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --lr $LR \
    --merge-train-val

# 3. APTOS
echo "[3/5] Training on APTOS..."
python3 train_ce_standard.py \
    --dataset "$WORKSPACE/datasets/aptos_classification_dataset" \
    --output-dir "$RESULTS_ROOT/aptos" \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --lr $LR \
    --merge-train-val

# 4. Fingerprint
echo "[4/5] Training on Fingerprint..."
python3 train_ce_standard.py \
    --dataset "$WORKSPACE/datasets/fingerA" \
    --output-dir "$RESULTS_ROOT/finger" \
    --is-finger \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --lr $LR \
    --merge-train-val

# 4. Fingerprint
echo "[4/5] Training on Fingerprint..."
python3 train_ce_standard.py \
    --dataset "$WORKSPACE/datasets/fingerB" \
    --output-dir "$RESULTS_ROOT/fingerB" \
    --is-finger \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --lr $LR \
    --merge-train-val

# 4. Fingerprint
echo "[4/5] Training on Fingerprint..."
python3 train_ce_standard.py \
    --dataset "$WORKSPACE/datasets/fingerC" \
    --output-dir "$RESULTS_ROOT/fingerC" \
    --is-finger \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --lr $LR \
    --merge-train-val

# 5. OCTA
echo "[5/5] Training on OCTA..."
python3 train_ce_standard.py \
    --dataset "$WORKSPACE/datasets/octa_classification_dataset" \
    --output-dir "$RESULTS_ROOT/octa" \
    --epochs $EPOCHS \
    --batch-size 16 \
    --lr $LR \
    --merge-train-val

echo "=========================================================="
echo "All baseline experiments completed."
