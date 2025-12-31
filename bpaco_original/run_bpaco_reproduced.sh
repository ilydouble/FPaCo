#!/bin/bash

# Directories
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
WORKSPACE="$(dirname "$SCRIPT_DIR")"
RESULTS_ROOT="$WORKSPACE/bpaco_original/results"
mkdir -p "$RESULTS_ROOT"

# GPU
export CUDA_VISIBLE_DEVICES=0

# Hyperparams (Paper Defaults)
EPOCHS=100
BATCH_SIZE=32
LR=0.01 # Paper uses 0.02

# BPaCo Params
ALPHA=1.0      # contrast weight among samples (Paper default for long-tail? Check bpaco_isic.py default is 1.0, but maybe user wants standard BPaCo settings)
               # In bpaco_isic.py: parser.add_argument('--alpha', default=1.0...)
               # Let's stick to defaults from the python script if not specified, but expose them here.
BETA=1.0       # contrast weight between centers and samples
GAMMA=1.0      # paco loss gamma
WEIGHT=1.0    # BPaCo loss weight in total loss (Paper uses 0.25)
TEMP=0.2       # MoCo temperature

echo "=========================================================="
echo "Starting BPaCo Reproduced (Original Logic) Experiments"
echo "=========================================================="

# 1. MIAS
echo "[1/5] Training on MIAS..."
python3 train_bpaco_reproduced.py \
    --dataset "$WORKSPACE/datasets/mias_classification_dataset" \
    --output-dir "$RESULTS_ROOT/mias" \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --lr $LR \
    --alpha $ALPHA \
    --beta $BETA \
    --bpaco-gamma $GAMMA \
    --bpaco-weight $WEIGHT \
    --moco-t $TEMP \
    --merge-train-val

# 2. OralCancer
echo "[2/5] Training on OralCancer..."
python3 train_bpaco_reproduced.py \
    --dataset "$WORKSPACE/datasets/oral_cancer_classification_dataset" \
    --output-dir "$RESULTS_ROOT/oral_cancer" \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --lr $LR \
    --alpha $ALPHA \
    --beta $BETA \
    --bpaco-gamma $GAMMA \
    --bpaco-weight $WEIGHT \
    --moco-t $TEMP \
    --merge-train-val

# 3. APTOS
echo "[3/5] Training on APTOS..."
python3 train_bpaco_reproduced.py \
    --dataset "$WORKSPACE/datasets/aptos_classification_dataset" \
    --output-dir "$RESULTS_ROOT/aptos" \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --lr $LR \
    --alpha $ALPHA \
    --beta $BETA \
    --bpaco-gamma $GAMMA \
    --bpaco-weight $WEIGHT \
    --moco-t $TEMP \
    --merge-train-val

# 4. Fingerprint
echo "[4/5] Training on Fingerprint..."
python3 train_bpaco_reproduced.py \
    --dataset "$WORKSPACE/datasets/fingerA" \
    --output-dir "$RESULTS_ROOT/finger" \
    --is-finger \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --lr $LR \
    --alpha $ALPHA \
    --beta $BETA \
    --bpaco-gamma $GAMMA \
    --bpaco-weight $WEIGHT \
    --moco-t $TEMP \
    --merge-train-val

# 4. Fingerprint
echo "[4/5] Training on Fingerprint..."
python3 train_bpaco_reproduced.py \
    --dataset "$WORKSPACE/datasets/fingerB" \
    --output-dir "$RESULTS_ROOT/fingerB" \
    --is-finger \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --lr $LR \
    --alpha $ALPHA \
    --beta $BETA \
    --bpaco-gamma $GAMMA \
    --bpaco-weight $WEIGHT \
    --moco-t $TEMP \
    --merge-train-val

# 4. Fingerprint
echo "[4/5] Training on Fingerprint..."
python3 train_bpaco_reproduced.py \
    --dataset "$WORKSPACE/datasets/fingerC" \
    --output-dir "$RESULTS_ROOT/fingerC" \
    --is-finger \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --lr $LR \
    --alpha $ALPHA \
    --beta $BETA \
    --bpaco-gamma $GAMMA \
    --bpaco-weight $WEIGHT \
    --moco-t $TEMP \
    --merge-train-val
# 5. OCTA
echo "[5/5] Training on OCTA..."
python3 train_bpaco_reproduced.py \
    --dataset "$WORKSPACE/datasets/octa_classification_dataset" \
    --output-dir "$RESULTS_ROOT/octa" \
    --epochs $EPOCHS \
    --batch-size 16 \
    --lr $LR \
    --alpha $ALPHA \
    --beta $BETA \
    --bpaco-gamma $GAMMA \
    --bpaco-weight $WEIGHT \
    --moco-t $TEMP \
    --merge-train-val

echo "=========================================================="
echo "All BPaCo reproduced experiments completed."
