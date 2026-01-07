#!/bin/bash

# Script to generate Florence-2 Heatmaps for all datasets
# Usage: bash fpaco/run_gen_heatmaps.sh

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
WORKSPACE="$(dirname "$SCRIPT_DIR")"
FLORENCE_SCRIPT="$SCRIPT_DIR/generate_offline_detections.py"
GEMINI_SCRIPT="$SCRIPT_DIR/generate_gemini_heatmaps.py"

echo "=========================================================="
echo "Starting Hybrid Heatmap Generation"
echo "  - MIAS, APTOS, OCTA: Gemini 1.5/2.0 (via $GEMINI_SCRIPT)"
echo "  - Oral, Fingerprint: Florence-2 (via $FLORENCE_SCRIPT)"
echo "=========================================================="
echo "Workspace: $WORKSPACE"

# Check for API Key if running Gemini
if [ -z "$YUNWU_API_KEY" ]; then
    echo "WARNING: YUNWU_API_KEY is not set. Gemini generation might fail."
fi

# # 1. MIAS (Gemini)
# echo "[1/7] Processing MIAS (Gemini)..."
# python "$GEMINI_SCRIPT" --dataset "$WORKSPACE/datasets/mias_classification_dataset" --model gemini-3-flash-preview --overwrite

# # 2. Oral Cancer (Florence-2)
# echo "[2/7] Processing Oral Cancer (Florence-2)..."
# python "$FLORENCE_SCRIPT" --dataset "$WORKSPACE/datasets/oral_cancer_classification_dataset" --overwrite

# 3. APTOS (Gemini)
# echo "[3/7] Processing APTOS (Gemini)..."
# python "$GEMINI_SCRIPT" --dataset "$WORKSPACE/datasets/aptos_classification_dataset" --model gemini-3-flash-preview --overwrite

# 4. OCTA (Gemini)
echo "[4/7] Processing OCTA (Gemini)..."
python "$GEMINI_SCRIPT" --dataset "$WORKSPACE/datasets/octa_classification_dataset" --model gemini-3-flash-preview --overwrite

# # 5. Fingerprint A (Florence-2)
# echo "[5/7] Processing Fingerprint A (Florence-2)..."
# python "$FLORENCE_SCRIPT" --dataset "$WORKSPACE/datasets/fingerA" --overwrite

# # 6. Fingerprint B (Florence-2)
# echo "[6/7] Processing Fingerprint B (Florence-2)..."
# python "$FLORENCE_SCRIPT" --dataset "$WORKSPACE/datasets/fingerB" --overwrite

# # 7. Fingerprint C (Florence-2)
# echo "[7/7] Processing Fingerprint C (Florence-2)..."
# python "$FLORENCE_SCRIPT" --dataset "$WORKSPACE/datasets/fingerC" --overwrite

echo "=========================================================="
echo "Heatmap Generation Complete!"
echo "You can now run 'run_fpaco_noheat.sh'."
