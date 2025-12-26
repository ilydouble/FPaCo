#!/bin/bash

# Directories
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
WORKSPACE="$(dirname "$SCRIPT_DIR")"

# GPU
export CUDA_VISIBLE_DEVICES=0

echo "=========================================================="
echo "Starting BioMedCLIP Zero-Shot classification"
echo "=========================================================="

# 1. MIAS
echo "[1/5] Evaluating on MIAS..."
python "$SCRIPT_DIR/zero_shot_classification.py" \
    --workspace "$WORKSPACE" \
    --dataset mias

# 2. OralCancer
echo "[2/5] Evaluating on OralCancer..."
python "$SCRIPT_DIR/zero_shot_classification.py" \
    --workspace "$WORKSPACE" \
    --dataset oral_cancer

# 3. APTOS
echo "[3/5] Evaluating on APTOS..."
python "$SCRIPT_DIR/zero_shot_classification.py" \
    --workspace "$WORKSPACE" \
    --dataset aptos

# 4. Fingerprint
echo "[4/5] Evaluating on Fingerprint..."
python "$SCRIPT_DIR/zero_shot_classification.py" \
    --workspace "$WORKSPACE" \
    --dataset finger

# 5. OCTA
echo "[5/5] Evaluating on OCTA..."
python "$SCRIPT_DIR/zero_shot_classification.py" \
    --workspace "$WORKSPACE" \
    --dataset octa

echo "=========================================================="
echo "All BioMedCLIP experiments completed."
