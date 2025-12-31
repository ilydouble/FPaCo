#!/bin/bash

# Define datasets
DATASETS=("oral_cancer" "aptos" "finger" "mias" "octa")

# Common settings
MODEL="hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
WORKSPACE="../"
OUTPUT_DIR="results"
# CAPACITY=128 # Increase capacity to hold more training samples (Few-Shot/Many-Shot)
CAPACITY=5 # 大 cache 并无意义
mkdir -p $OUTPUT_DIR

echo "Starting DPE BioMedCLIP SUPERVISED experiments (Train Cache, Capacity $CAPACITY)..."

for dataset in "${DATASETS[@]}"; do
    echo "========================================================"
    echo "Running Supervised DPE on $dataset..."
    echo "========================================================"
    
    python dpe_biomedclip.py \
    --dataset $dataset \
    --data-root $WORKSPACE \
    --model $MODEL \
    --output-dir $OUTPUT_DIR \
    --lr-text 0.001 \
    --lr-image 0.001 \
    --pos-alpha 1.0 \
    --pos-beta 0.5 \
    --shot-capacity $CAPACITY \
    --seed 1 \
    --prompts-file ../prompts/unified_prompts.json \
    --use-train-cache
    
    echo "Finished $dataset"
    echo ""
done

echo "Supervised experiments completed. Results saved in $OUTPUT_DIR"
