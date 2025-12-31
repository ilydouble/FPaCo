#!/bin/bash

# Define datasets
DATASETS=("oral_cancer" "aptos" "finger" "mias" "octa")

# Common settings
MODEL="hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
WORKSPACE="../datasets"
OUTPUT_DIR="results"
SHOTS=16
EPOCHS=50

mkdir -p $OUTPUT_DIR

echo "Starting CoOp BioMedCLIP experiments (Shots: $SHOTS, Epochs: $EPOCHS)..."

for dataset in "${DATASETS[@]}"; do
    echo "========================================================"
    echo "Running CoOp on $dataset..."
    echo "========================================================"
    
    python coop_biomedclip.py \
    --dataset $dataset \
    --data-root $WORKSPACE \
    --model $MODEL \
    --output-dir $OUTPUT_DIR \
    --lr 0.002 \
    --epochs $EPOCHS \
    --shots $SHOTS \
    --n-ctx 16 \
    --seed 1
    
    echo "Finished $dataset"
    echo ""
done

echo "CoOp experiments completed. Results saved in $OUTPUT_DIR"
