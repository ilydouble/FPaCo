#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

CONFIG_DIR="$SCRIPT_DIR/configs/"
DATA_ROOT="$PROJECT_ROOT/datasets"

# BioMedCLIP model (default: microsoft/BiomedCLIP-PubMedBERT_256-vit_large_patch16_224)
MODEL="hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_large_patch16_224"

echo "========================================================="
echo "Starting TDA experiments with BioMedCLIP (ALL datasets)"
echo "Model: $MODEL"
echo "========================================================="

python "$SCRIPT_DIR/tda_runner.py" \
    --config "$CONFIG_DIR" \
    --dataset all \
    --data-root "$DATA_ROOT" \
    --model "$MODEL" \
    --prompts-file "prompts/prompts_simple.json"

echo "========================================================="
echo "TDA experiments completed"
echo "Results saved in tda/results/"
echo "========================================================="
