#!/bin/bash

# Directories
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
WORKSPACE="$(dirname "$SCRIPT_DIR")"
RESULTS_ROOT="$WORKSPACE/heat_classification_agent/results_florence"
mkdir -p "$RESULTS_ROOT"

# GPU
export CUDA_VISIBLE_DEVICES=0

# Hyperparams (Default as per train_agent.py)
EPOCHS=50
LR=0.005
LAMBDA_START=0.1
LAMBDA_MAX=0.8
BETA=2.0
GAMMA=0.5
BATCH_SIZE=32

echo "=========================================================="
echo "Starting Heat Classification Agent (Florence-2) Experiments"
echo "=========================================================="

# 1. MIAS
echo "[1/5] Training on MIAS..."
python3 train_agent.py \
    --dataset "$WORKSPACE/datasets/mias_classification_dataset" \
    --output-dir "$RESULTS_ROOT/mias" \
    --epochs $EPOCHS \
    --lr $LR \
    --batch-size $BATCH_SIZE

# 2. OralCancer
echo "[2/5] Training on OralCancer..."
python3 train_agent.py \
    --dataset "$WORKSPACE/datasets/oral_cancer_classification_dataset" \
    --output-dir "$RESULTS_ROOT/oral_cancer" \
    --epochs $EPOCHS \
    --lr $LR \
    --batch-size $BATCH_SIZE

# 3. APTOS
echo "[3/5] Training on APTOS..."
python3 train_agent.py \
    --dataset "$WORKSPACE/datasets/aptos_classification_dataset" \
    --output-dir "$RESULTS_ROOT/aptos" \
    --epochs $EPOCHS \
    --lr $LR \
    --batch-size $BATCH_SIZE

# 4. Fingerprint
echo "[4/5] Training on Fingerprint..."
python3 train_agent.py \
    --dataset "$WORKSPACE/datasets/fingerprint_classification_dataset" \
    --output-dir "$RESULTS_ROOT/finger" \
    --epochs $EPOCHS \
    --lr $LR \
    --batch-size $BATCH_SIZE

# 5. OCTA
echo "[5/5] Training on OCTA..."
python3 train_agent.py \
    --dataset "$WORKSPACE/datasets/octa_classification_dataset" \
    --output-dir "$RESULTS_ROOT/octa" \
    --epochs $EPOCHS \
    --lr $LR \
    --batch-size 16 # OCTA usually requires smaller batch size

echo "=========================================================="
echo "All Heatmap experiments completed."
