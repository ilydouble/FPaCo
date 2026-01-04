#!/bin/bash

# Script to generate Florence-2 Heatmaps for all datasets
# Usage: bash fpaco/run_gen_heatmaps.sh

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
WORKSPACE="$(dirname "$SCRIPT_DIR")"
GEN_SCRIPT="$SCRIPT_DIR/generate_offline_detections.py"

echo "=========================================================="
echo "Starting Florence-2 Heatmap Generation"
echo "=========================================================="
echo "Workspace: $WORKSPACE"
echo "Script: $GEN_SCRIPT"

# Ensure dependencies (transformers, etc.) are installed.
# Using default microsoft/Florence-2-large-ft

# 1. MIAS
echo "[1/7] Processing MIAS..."
python "$GEN_SCRIPT" --dataset "$WORKSPACE/datasets/mias_classification_dataset" --overwrite

# 2. Oral Cancer
echo "[2/7] Processing Oral Cancer..."
python "$GEN_SCRIPT" --dataset "$WORKSPACE/datasets/oral_cancer_classification_dataset" --overwrite

# 3. APTOS
echo "[3/7] Processing APTOS..."
python "$GEN_SCRIPT" --dataset "$WORKSPACE/datasets/aptos_classification_dataset" --overwrite

# 4. OCTA
echo "[4/7] Processing OCTA..."
python "$GEN_SCRIPT" --dataset "$WORKSPACE/datasets/octa_classification_dataset" --overwrite

# 5. Fingerprint A
echo "[5/7] Processing Fingerprint A..."
python "$GEN_SCRIPT" --dataset "$WORKSPACE/datasets/fingerA" --overwrite

# 6. Fingerprint B
echo "[6/7] Processing Fingerprint B..."
python "$GEN_SCRIPT" --dataset "$WORKSPACE/datasets/fingerB" --overwrite

# 7. Fingerprint C
echo "[7/7] Processing Fingerprint C..."
python "$GEN_SCRIPT" --dataset "$WORKSPACE/datasets/fingerC" --overwrite

echo "=========================================================="
echo "Heatmap Generation Complete!"
echo "JSON files have been saved alongside original images."
echo "You can now run 'run_fpaco_noheat.sh'."
