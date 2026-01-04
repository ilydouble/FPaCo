#!/bin/bash

# Directories
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
WORKSPACE="$(dirname "$SCRIPT_DIR")"
RESULTS_ROOT="$WORKSPACE/fpaco_noheat/results"
mkdir -p "$RESULTS_ROOT"

# GPU
export CUDA_VISIBLE_DEVICES=0

# Hyperparams
EPOCHS=100
LR=0.001
BATCH_SIZE=32
IMAGE_SIZE=448
BACKBONE="resnet18"
BETA=2.0
TAU=0.1
SIGMA=30
QUEUE_SIZE=8192
FOCAL_GAMMA=2.0

# Optional: Set to "--combine-train-val" to merge datasets for final training
#COMBINE_FLAG="" 
COMBINE_FLAG="--combine-train-val --no-heatmap"

echo "=========================================================="
echo "Starting FPaCo NoHeat (BPaCo 4-Channel Reduced) Experiments"
echo "=========================================================="

# # 1. MIAS
# echo "[1/5] Training on MIAS..."
# python train_fpaco_noheat.py \
#     --dataset "$WORKSPACE/datasets/mias_classification_dataset" \
#     --output-dir "$RESULTS_ROOT/mias" \
#     --epochs $EPOCHS \
#     --lr $LR \
#     --batch-size $BATCH_SIZE \
#     --image-size $IMAGE_SIZE \
#     --backbone $BACKBONE \
#     --beta $BETA \
#     --tau $TAU \
#     --sigma $SIGMA \
#     --queue-size $QUEUE_SIZE \
#     --focal-gamma $FOCAL_GAMMA \
#     $COMBINE_FLAG

# # 2. OralCancer
# echo "[2/5] Training on OralCancer..."
# python train_fpaco_noheat.py \
#     --dataset "$WORKSPACE/datasets/oral_cancer_classification_dataset" \
#     --output-dir "$RESULTS_ROOT/oral_cancer" \
#     --epochs $EPOCHS \
#     --lr $LR \
#     --batch-size $BATCH_SIZE \
#     --image-size $IMAGE_SIZE \
#     --backbone $BACKBONE \
#     --beta $BETA \
#     --tau $TAU \
#     --sigma $SIGMA \
#     --queue-size $QUEUE_SIZE \
#     --focal-gamma $FOCAL_GAMMA \
#     $COMBINE_FLAG

# 3. APTOS
echo "[3/5] Training on APTOS..."
python train_fpaco_noheat.py \
    --dataset "$WORKSPACE/datasets/aptos_classification_dataset" \
    --output-dir "$RESULTS_ROOT/aptos" \
    --epochs $EPOCHS \
    --lr $LR \
    --batch-size $BATCH_SIZE \
    --image-size $IMAGE_SIZE \
    --backbone $BACKBONE \
    --beta $BETA \
    --tau $TAU \
    --sigma $SIGMA \
    --queue-size $QUEUE_SIZE \
    --focal-gamma $FOCAL_GAMMA \
    $COMBINE_FLAG

# 4. Fingerprint
echo "[4/5] Training on Fingerprint..."
python train_fpaco_noheat.py \
    --dataset "$WORKSPACE/datasets/fingerA" \
    --output-dir "$RESULTS_ROOT/finger" \
    --epochs $EPOCHS \
    --lr $LR \
    --batch-size $BATCH_SIZE \
    --image-size $IMAGE_SIZE \
    --backbone $BACKBONE \
    --beta $BETA \
    --tau $TAU \
    --sigma $SIGMA \
    --queue-size $QUEUE_SIZE \
    --focal-gamma $FOCAL_GAMMA \
    $COMBINE_FLAG

# # 4. Fingerprint
# echo "[4/5] Training on Fingerprint..."
# python train_fpaco_noheat.py \
#     --dataset "$WORKSPACE/datasets/fingerB" \
#     --output-dir "$RESULTS_ROOT/fingerB" \
#     --epochs $EPOCHS \
#     --lr $LR \
#     --batch-size $BATCH_SIZE \
#     --image-size $IMAGE_SIZE \
#     --backbone $BACKBONE \
#     --beta $BETA \
#     --tau $TAU \
#     --sigma $SIGMA \
#     --queue-size $QUEUE_SIZE \
#     --focal-gamma $FOCAL_GAMMA \
#     $COMBINE_FLAG

# # 4. Fingerprint
# echo "[4/5] Training on Fingerprint..."
# python train_fpaco_noheat.py \
#     --dataset "$WORKSPACE/datasets/fingerC" \
#     --output-dir "$RESULTS_ROOT/fingerC" \
#     --epochs $EPOCHS \
#     --lr $LR \
#     --batch-size $BATCH_SIZE \
#     --image-size $IMAGE_SIZE \
#     --backbone $BACKBONE \
#     --beta $BETA \
#     --tau $TAU \
#     --sigma $SIGMA \
#     --queue-size $QUEUE_SIZE \
#     --focal-gamma $FOCAL_GAMMA \
#     $COMBINE_FLAG
# 5. OCTA
# echo "[5/5] Training on OCTA..."
# python train_fpaco_noheat.py \
#     --dataset "$WORKSPACE/datasets/octa_classification_dataset" \
#     --output-dir "$RESULTS_ROOT/octa" \
#     --epochs $EPOCHS \
#     --lr $LR \
#     --batch-size 16 \
#     --image-size $IMAGE_SIZE \
#     --backbone $BACKBONE \
#     --beta $BETA \
#     --tau $TAU \
#     --sigma $SIGMA \
#     --queue-size $QUEUE_SIZE \
#     --focal-gamma $FOCAL_GAMMA \
#     $COMBINE_FLAG

echo "=========================================================="
echo "All Heatmap experiments completed."
