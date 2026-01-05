#!/bin/bash

# Configuration
DATASET="../datasets" # Adjust this if your dataset path is different
EPOCHS=50
dataset_name="oral_cancer_classification_dataset" # Default dataset, change as needed

echo "========================================================"
echo "Experiment 1: Pure Focal Loss (Gamma=2.0, Tau=0)"
echo "========================================================"
python3 train_fpaco_noheat.py \
    --dataset "${DATASET}/${dataset_name}" \
    --output-dir "results_comparison/${dataset_name}/focal_only" \
    --epochs ${EPOCHS} \
    --focal-gamma 2.0 \
    --tau 0.0 \
    --no-heatmap

echo "========================================================"
echo "Experiment 2: CE + Logit Compensation (Gamma=0, Tau=1.0)"
echo "========================================================"
python3 train_fpaco_noheat.py \
    --dataset "${DATASET}/${dataset_name}" \
    --output-dir "results_comparison/${dataset_name}/ce_comp" \
    --epochs ${EPOCHS} \
    --focal-gamma 0.0 \
    --tau 1.0 \
    --no-heatmap

echo "========================================================"
echo "Comparison Completed."
echo "Results saved in results_comparison/${dataset_name}/"
