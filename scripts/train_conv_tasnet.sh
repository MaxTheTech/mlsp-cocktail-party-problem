#!/bin/bash
source $(conda info --base)/etc/profile.d/conda.sh
conda activate mlsp-project

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Get the project root (parent of script directory)
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Set PYTHONPATH to include project root
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# Fix for NFS temporary file issues with DataLoader multiprocessing
export TMPDIR=/tmp
export TEMP=/tmp
export TMP=/tmp

# Change to project root directory
cd "$PROJECT_ROOT"

# Configuration
DATA_ROOT="data/Libri2Mix"
DATA_CONFIG="config/libri2mix_16k_2src.yaml"
MODEL_TYPE="multi_scale" # standard or multi_scale
MODEL_CONFIG="config/conv_tasnet_small_${MODEL_TYPE}.yaml"
SAVE_DIR="output/models/conv_tasnet/conv_tasnet_small_${MODEL_TYPE}"
LOG_FILE="output/logs/conv_tasnet_small_${MODEL_TYPE}_training.log"

mkdir -p $SAVE_DIR
mkdir -p logs


CUDA_VISIBLE_DEVICES=0 python -m src.train.conv_tasnet_train \
    --model-type $MODEL_TYPE \
    --root-dir-data $DATA_ROOT \
    --config-data $DATA_CONFIG \
    --config-model $MODEL_CONFIG \
    --save-dir $SAVE_DIR \
    --log-file $LOG_FILE \
    --save-checkpoints