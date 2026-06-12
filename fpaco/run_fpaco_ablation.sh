#!/bin/bash

set -e

# Ablation experiments for FPaCo on four datasets:
#   - APTOS
#   - fingerA
#   - MIAS
#   - OCTA
#
# For each dataset, use its best hyperparameters, then run:
#   1. naive_vlm       : CE + Contrastive + Heatmap Input, no guide loss
#   2. mse_distill     : Focal + Contrastive + MSE Alignment, adaptive alpha
#   3. static_guide    : Focal + Contrastive + ReLU Alignment, static alpha
#   4. full_adaptive   : Focal + Contrastive + ReLU Alignment, adaptive alpha

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
WORKSPACE="$(dirname "$SCRIPT_DIR")"
TRAIN_SCRIPT="$SCRIPT_DIR/train_fpaco.py"
RESULTS_ROOT="$SCRIPT_DIR/results/ablation"

mkdir -p "$RESULTS_ROOT"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo "Cannot find train_fpaco.py in $SCRIPT_DIR"
    exit 1
fi

run_one() {
    local dataset_name="$1"
    local variant="$2"
    local dataset_path="$3"
    local epochs="$4"
    local batch_size="$5"
    local lr="$6"
    local image_size="$7"
    local sigma="$8"
    local beta="$9"
    local tau="${10}"
    local focal_gamma="${11}"
    local max_alpha="${12}"
    local warmup_epochs="${13}"
    shift 13

    local output_dir="$RESULTS_ROOT/$dataset_name/$variant"

    echo "--------------------------------------------------------"
    echo "Dataset : $dataset_name"
    echo "Variant : $variant"
    echo "Output  : $output_dir"
    echo "--------------------------------------------------------"

    python "$TRAIN_SCRIPT" \
        --dataset "$dataset_path" \
        --output-dir "$output_dir" \
        --epochs "$epochs" \
        --batch-size "$batch_size" \
        --lr "$lr" \
        --backbone resnet18 \
        --image-size "$image_size" \
        --sigma "$sigma" \
        --beta "$beta" \
        --tau "$tau" \
        --focal-gamma "$focal_gamma" \
        --temperature 0.1 \
        --queue-size 8192 \
        --val-interval 1 \
        --max-alpha "$max_alpha" \
        --warmup-epochs "$warmup_epochs" \
        "$@"
}

run_dataset() {
    local dataset_name="$1"
    local dataset_path="$2"
    local epochs="$3"
    local batch_size="$4"
    local lr="$5"
    local image_size="$6"
    local sigma="$7"
    local beta="$8"
    local tau="$9"
    local best_focal_gamma="${10}"
    local max_alpha="${11}"
    local warmup_epochs="${12}"

    echo "========================================================"
    echo "Running ablations for $dataset_name"
    echo "========================================================"

    run_one "$dataset_name" "1_naive_vlm" \
        "$dataset_path" "$epochs" "$batch_size" "$lr" "$image_size" "$sigma" \
        "$beta" "$tau" 0.0 "$max_alpha" "$warmup_epochs" \
        --guide-weight 0.0

    run_one "$dataset_name" "2_mse_distill_adaptive" \
        "$dataset_path" "$epochs" "$batch_size" "$lr" "$image_size" "$sigma" \
        "$beta" "$tau" "$best_focal_gamma" "$max_alpha" "$warmup_epochs" \
        --align-loss-type mse

    run_one "$dataset_name" "3_static_guide" \
        "$dataset_path" "$epochs" "$batch_size" "$lr" "$image_size" "$sigma" \
        "$beta" "$tau" "$best_focal_gamma" "$max_alpha" "$warmup_epochs" \
        --no-adaptive-alpha

    run_one "$dataset_name" "4_full_adaptive" \
        "$dataset_path" "$epochs" "$batch_size" "$lr" "$image_size" "$sigma" \
        "$beta" "$tau" "$best_focal_gamma" "$max_alpha" "$warmup_epochs"
}

echo "========================================================"
echo "Starting FPaCo ablation experiments"
echo "Train script: $TRAIN_SCRIPT"
echo "Results root: $RESULTS_ROOT"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "========================================================"

run_dataset "APTOS" \
    "$WORKSPACE/datasets/aptos_classification_dataset" \
    100 16 0.001 448 15 2.5 0.5 2.0 0.5 10

run_dataset "fingerA" \
    "$WORKSPACE/datasets/fingerA" \
    100 16 0.001 448 15 2.5 0.5 2.0 0.5 10

run_dataset "MIAS" \
    "$WORKSPACE/datasets/mias_classification_dataset" \
    100 16 0.005 224 30 2.0 2.0 1.5 0.8 20

run_dataset "OCTA" \
    "$WORKSPACE/datasets/octa_classification_dataset" \
    100 16 0.005 224 30 2.0 1.0 2.5 0.8 0

echo "========================================================"
echo "All FPaCo ablation experiments completed."
echo "========================================================"
