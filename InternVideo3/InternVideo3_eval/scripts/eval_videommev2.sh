#!/bin/bash
# Usage: Set MODEL_PATH and OUTPUT_DIR before running this script.
#   export MODEL_PATH=/path/to/your/model
#   export OUTPUT_DIR=./logs/videommev2_results
#   bash eval_videommev2.sh
set -x

# ============ User Configuration ============
MODEL_PATH="${MODEL_PATH:?Please set MODEL_PATH environment variable}"
OUTPUT_DIR="${OUTPUT_DIR:-./logs/videommev2_results}"
# ============================================

SCRIPT_DIR=$(dirname "$0")

pip install transformers==4.57.3 > /dev/null 2>&1
pip install qwen_vl_utils > /dev/null 2>&1

torchrun --nproc_per_node=8 ${SCRIPT_DIR}/eval_videommev2.py \
  --model_path $MODEL_PATH \
  --output_dir $OUTPUT_DIR \
  --with_subtitle
