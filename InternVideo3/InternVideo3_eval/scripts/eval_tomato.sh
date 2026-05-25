#!/bin/bash
# Usage: Set MODEL_PATH and OUTPUT_DIR before running this script.
#   export MODEL_PATH=/path/to/your/model
#   export OUTPUT_DIR=./logs/tomato_results
#   bash eval_tomato.sh
set -x

# ============ User Configuration ============
MODEL_PATH="${MODEL_PATH:?Please set MODEL_PATH environment variable}"
OUTPUT_DIR="${OUTPUT_DIR:-./logs/tomato_results}"
# ============================================

SCRIPT_DIR=$(dirname "$0")

pip install transformers==4.57.3 > /dev/null 2>&1
pip install qwen-vl-utils > /dev/null 2>&1

torchrun --nproc_per_node=8 ${SCRIPT_DIR}/eval_tomato.py \
  --model_path $MODEL_PATH \
  --output_dir $OUTPUT_DIR
