#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

CONFIG_DIR="$SCRIPT_DIR/configs/"
DATA_ROOT="$PROJECT_ROOT/datasets"
BACKBONE="ViT-B/16"

echo "========================================================="
echo "Starting TDA experiments (ALL datasets)"
echo "========================================================="

python "$SCRIPT_DIR/tda_runner.py" \
    --config "$CONFIG_DIR" \
    --dataset all \
    --data-root "$DATA_ROOT" \
    --backbone "$BACKBONE"

echo "========================================================="
echo "TDA experiments completed"
echo "Results saved in tda/results/"
echo "========================================================="
