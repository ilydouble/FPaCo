#!/bin/bash

# Hyperparameter tuning runner for heat_classification_agent
# Adjust the arrays below to define your search space.

set -euo pipefail

# Directories
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
WORKSPACE="$(dirname "$SCRIPT_DIR")"
RESULTS_ROOT="$WORKSPACE/heat_classification_agent/results_hyper"
mkdir -p "$RESULTS_ROOT"

# GPU
export CUDA_VISIBLE_DEVICES=0

# Toggle to merge train/val for final training
# COMBINE_FLAG=""
COMBINE_FLAG="--combine-train-val"

# Datasets (name:dataset_path:output_subdir:batch_size_override)
DATASETS=(
  "mias:$WORKSPACE/datasets/mias_classification_dataset:mias:"
  "oral_cancer:$WORKSPACE/datasets/oral_cancer_classification_dataset:oral_cancer:"
  "aptos:$WORKSPACE/datasets/aptos_classification_dataset:aptos:"
  "finger:$WORKSPACE/datasets/fingerprint_classification_dataset:finger:"
  "octa:$WORKSPACE/datasets/octa_classification_dataset:octa:16"
)

# Search space (edit these lists)
EPOCHS_LIST=(100)
LR_LIST=(0.001 0.005)
BATCH_SIZE_LIST=(32)
IMAGE_SIZE_LIST=(448)
BACKBONE_LIST=("resnet18")
BETA_LIST=(0.5 1.0 1.5 2.0 2.5) # 对比损失权重
TAU_LIST=(0.1 0.5 1.0 1.5) # 对比学习温度参数
SIGMA_LIST=(10 20 30 40 50) # 高斯核参数
QUEUE_SIZE_LIST=(8192)

run_one () {
  local dataset_name="$1"
  local dataset_path="$2"
  local output_subdir="$3"
  local batch_override="$4"
  local epochs="$5"
  local lr="$6"
  local batch_size="$7"
  local image_size="$8"
  local backbone="$9"
  local beta="${10}"
  local tau="${11}"
  local sigma="${12}"
  local queue_size="${13}"

  if [[ -n "$batch_override" ]]; then
    batch_size="$batch_override"
  fi

  local run_tag="ep${epochs}_lr${lr}_bs${batch_size}_img${image_size}_bb${backbone}_b${beta}_t${tau}_s${sigma}_q${queue_size}"
  local out_dir="$RESULTS_ROOT/$output_subdir/$run_tag"
  mkdir -p "$out_dir"

  echo "=========================================================="
  echo "Dataset: $dataset_name"
  echo "Run: $run_tag"
  echo "Output: $out_dir"
  echo "=========================================================="

  python3 "$SCRIPT_DIR/train_agent.py" \
    --dataset "$dataset_path" \
    --output-dir "$out_dir" \
    --epochs "$epochs" \
    --lr "$lr" \
    --batch-size "$batch_size" \
    --image-size "$image_size" \
    --backbone "$backbone" \
    --beta "$beta" \
    --tau "$tau" \
    --sigma "$sigma" \
    --queue-size "$queue_size" \
    $COMBINE_FLAG
}

for ds in "${DATASETS[@]}"; do
  IFS=":" read -r name path outdir batch_override <<< "$ds"
  for epochs in "${EPOCHS_LIST[@]}"; do
    for lr in "${LR_LIST[@]}"; do
      for bs in "${BATCH_SIZE_LIST[@]}"; do
        for img in "${IMAGE_SIZE_LIST[@]}"; do
          for bb in "${BACKBONE_LIST[@]}"; do
            for beta in "${BETA_LIST[@]}"; do
              for tau in "${TAU_LIST[@]}"; do
                for sigma in "${SIGMA_LIST[@]}"; do
                  for q in "${QUEUE_SIZE_LIST[@]}"; do
                    run_one "$name" "$path" "$outdir" "$batch_override" \
                      "$epochs" "$lr" "$bs" "$img" "$bb" "$beta" "$tau" "$sigma" "$q"
                  done
                done
              done
            done
          done
        done
      done
    done
  done
done

echo "=========================================================="
echo "All hyperparameter tuning runs completed."