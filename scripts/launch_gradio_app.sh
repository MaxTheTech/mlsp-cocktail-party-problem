#!/bin/bash
# Launch Gradio app for Conv-TasNet audio source separation

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

# Configuration (change this to your model path)
MODEL_PATH="output/models/dprnn/dprnn_20251113_d02712/best_model.pth"
DEVICE="cuda"  # cuda, mps, or cpu
PORT=7860
SHARE=true  # Set to true to create a public share link

echo "============================================"
echo "Conv-TasNet Gradio App"
echo "============================================"
echo "Model:  $MODEL_PATH"
echo "Device: $DEVICE"
echo "Port:   $PORT"
echo "============================================"

# Launch Gradio app
if [ "$SHARE" = true ]; then
    python -m src.gradio_app \
        --model-path "$MODEL_PATH" \
        --device "$DEVICE" \
        --port $PORT \
        --share
else
    python -m src.gradio_app \
        --model-path "$MODEL_PATH" \
        --device "$DEVICE" \
        --port $PORT
fi

