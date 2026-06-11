#!/bin/bash

# BPaCo Heatmap Refined Hyperparameter Tuning Script
# Based on optimal results:
# - Best LR: 0.005
# - Best Temperature: 0.10
# - Best Queue Size: 8192
# - Best Sigma: ~10-15
# - Best Beta: ~2.0

DATASET_DIR="../classification_dataset"
KEYPOINT_FEATURES="../keypoint_features.csv"
BASE_OUTPUT_DIR="results/bpaco_refined_tuning"
EPOCHS=100

echo "=========================================================="
echo "Starting Refined BPaCo Hyperparameter Tuning"
echo "=========================================================="

# Fixed Parameters (Optimized in previous runs)
LR=0.005
QUEUE=8192
TEMP=0.10

# Grid Search Parameters (Refining around the optimal zone)
SIGMAS=(10 12 15)           # Refined range for Heatmap Smoothing
BETAS=(1.5 2.0 2.5 3.0)      # Refined range for Loss Balancing

for SIGMA in "${SIGMAS[@]}"
do
    for BETA in "${BETAS[@]}"
    do
        # Create a descriptive experiment name
        EXP_NAME="sigma${SIGMA}_beta${BETA}_lr${LR}"
        OUTPUT_DIR="${BASE_OUTPUT_DIR}/${EXP_NAME}"
        
        echo ""
        echo "----------------------------------------------------------"
        echo "Running Experiment: ${EXP_NAME}"
        echo "  Sigma: ${SIGMA}"
        echo "  Beta:  ${BETA}"
        echo "  LR:    ${LR} (Fixed)"
        echo "  Queue: ${QUEUE} (Fixed)"
        echo "  Temp:  ${TEMP} (Fixed)"
        echo "----------------------------------------------------------"
        
        if [ -d "$OUTPUT_DIR" ]; then
            echo "Skipping ${EXP_NAME}, already exists."
            continue
        fi

        python train_bpaco_heatmap.py \
            --dataset "${DATASET_DIR}" \
            --keypoint-features "${KEYPOINT_FEATURES}" \
            --backbone resnet18 \
            --epochs "${EPOCHS}" \
            --sigma-center "${SIGMA}" \
            --sigma-delta "${SIGMA}" \
            --lr "${LR}" \
            --beta "${BETA}" \
            --queue-size "${QUEUE}" \
            --temperature "${TEMP}" \
            --output-dir "${OUTPUT_DIR}"
            
        echo "Finished: ${EXP_NAME}"
    done
done

echo ""
echo "=========================================================="
echo "Refined Tuning Completed!"
echo "Check results in: ${BASE_OUTPUT_DIR}"
echo "=========================================================="
