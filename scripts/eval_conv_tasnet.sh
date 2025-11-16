#!/bin/bash
source $(conda info --base)/etc/profile.d/conda.sh
conda activate mlsp-project

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Get the project root (parent of script directory)
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Set PYTHONPATH to include project root
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# Change to project root directory
cd "$PROJECT_ROOT"

# Configuration
DATA_ROOT="data/Libri2Mix"
DATA_CONFIG="config/libri2mix_16k_2src.yaml"
OUTPUT_DIR="output/results/conv_tasnet_small_multi_scale"

# Model checkpoint to evaluate
MODEL_PATH="output/models/conv_tasnet/conv_tasnet_small_multi_scale/conv_tasnet_small_multi_scale_20251105_044ce1/best_model.pth"

# Dataset split to evaluate on
SPLIT="dev"  # or "test"

# Run evaluation
CUDA_VISIBLE_DEVICES=0 python -m src.evaluate.eval_conv_tasnet \
    --model-path $MODEL_PATH \
    --root-dir-data $DATA_ROOT \
    --config-data $DATA_CONFIG \
    --split $SPLIT \
    --output-dir $OUTPUT_DIR

