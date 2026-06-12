#!/bin/bash

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
WORKSPACE="$(dirname "$SCRIPT_DIR")"
RESULTS_ROOT="$SCRIPT_DIR/results/best_runs"

mkdir -p "$RESULTS_ROOT"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

TRAIN_SCRIPT="$SCRIPT_DIR/train_fpaco.py"

echo "=========================================================="
echo "Starting FPaCo best experiments"
echo "Train script: $TRAIN_SCRIPT"
echo "Results root: $RESULTS_ROOT"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "=========================================================="

echo "[1/6] Training on APTOS..."
python "$TRAIN_SCRIPT" \
    --dataset "$WORKSPACE/datasets/aptos_classification_dataset" \
    --output-dir "$RESULTS_ROOT/APTOS" \
    --epochs 100 \
    --batch-size 16 \
    --lr 0.001 \
    --backbone resnet18 \
    --image-size 448 \
    --sigma 15 \
    --beta 2.5 \
    --tau 0.5 \
    --focal-gamma 2.0 \
    --temperature 0.1 \
    --queue-size 8192 \
    --val-interval 1 \
    --max-alpha 0.5 \
    --warmup-epochs 10

echo "[2/6] Training on fingerA..."
python "$TRAIN_SCRIPT" \
    --dataset "$WORKSPACE/datasets/fingerA" \
    --output-dir "$RESULTS_ROOT/fingerA" \
    --epochs 100 \
    --batch-size 16 \
    --lr 0.001 \
    --backbone resnet18 \
    --image-size 448 \
    --sigma 15 \
    --beta 2.5 \
    --tau 0.5 \
    --focal-gamma 2.0 \
    --temperature 0.1 \
    --queue-size 8192 \
    --val-interval 1 \
    --max-alpha 0.5 \
    --warmup-epochs 10

echo "[3/6] Training on fingerB..."
python "$TRAIN_SCRIPT" \
    --dataset "$WORKSPACE/datasets/fingerB" \
    --output-dir "$RESULTS_ROOT/fingerB" \
    --epochs 100 \
    --batch-size 16 \
    --lr 0.001 \
    --backbone resnet18 \
    --image-size 448 \
    --sigma 15 \
    --beta 2.5 \
    --tau 0.5 \
    --focal-gamma 2.0 \
    --temperature 0.1 \
    --queue-size 8192 \
    --val-interval 1 \
    --max-alpha 0.5 \
    --warmup-epochs 10

echo "[4/6] Training on fingerC..."
python "$TRAIN_SCRIPT" \
    --dataset "$WORKSPACE/datasets/fingerC" \
    --output-dir "$RESULTS_ROOT/fingerC" \
    --epochs 100 \
    --batch-size 16 \
    --lr 0.001 \
    --backbone resnet18 \
    --image-size 448 \
    --sigma 15 \
    --beta 2.5 \
    --tau 0.5 \
    --focal-gamma 2.0 \
    --temperature 0.1 \
    --queue-size 8192 \
    --val-interval 1 \
    --max-alpha 0.5 \
    --warmup-epochs 10

echo "[5/6] Training on MIAS..."
python "$TRAIN_SCRIPT" \
    --dataset "$WORKSPACE/datasets/mias_classification_dataset" \
    --output-dir "$RESULTS_ROOT/MIAS" \
    --epochs 100 \
    --batch-size 16 \
    --lr 0.005 \
    --backbone resnet18 \
    --image-size 224 \
    --sigma 30 \
    --beta 2.0 \
    --tau 2.0 \
    --focal-gamma 1.5 \
    --temperature 0.1 \
    --queue-size 8192 \
    --val-interval 1 \
    --max-alpha 0.8 \
    --warmup-epochs 20

echo "[6/6] Training on OCTA..."
python "$TRAIN_SCRIPT" \
    --dataset "$WORKSPACE/datasets/octa_classification_dataset" \
    --output-dir "$RESULTS_ROOT/OCTA" \
    --epochs 100 \
    --batch-size 16 \
    --lr 0.005 \
    --backbone resnet18 \
    --image-size 224 \
    --sigma 30 \
    --beta 2.0 \
    --tau 1.0 \
    --focal-gamma 2.5 \
    --temperature 0.1 \
    --queue-size 8192 \
    --val-interval 1 \
    --max-alpha 0.8 \
    --warmup-epochs 0

echo "=========================================================="
echo "All FPaCo best experiments completed."
echo "=========================================================="
