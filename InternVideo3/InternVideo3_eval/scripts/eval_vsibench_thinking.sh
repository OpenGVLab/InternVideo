#!/bin/bash
# Usage: Set MODEL_PATH and OUTPUT_DIR before running this script.
#   export MODEL_PATH=/path/to/your/model
#   export OUTPUT_DIR=./logs
#   bash this_script.sh
set -x

# ============ User Configuration ============
# MODEL_PATH: Path to the pretrained model checkpoint
export model_path="${MODEL_PATH:?Please set MODEL_PATH environment variable}"
# OUTPUT_DIR: Directory to save evaluation results
export output_path="${OUTPUT_DIR:-./logs}"
# ============================================

MAX_PIXELS=$((256*4 * 32 * 32))
MIN_PIXELS=$((256*4 * 32 * 32))
export LMMS_EVAL_LAUNCHER="accelerate"
export model_family='internvideo3'
export model_args="pretrained=${model_path},min_pixels=${MIN_PIXELS},max_pixels=${MAX_PIXELS},fps=4,max_num_frames=256,attn_implementation=flash_attention_2,enable_thinking=true"
export model='internvideo3'
export benchmark='vsibench'
export HF_DATASETS_OFFLINE=1

cd $(dirname "$0")/../lmms-eval
pip install -e . > /dev/null 2>&1
pip install transformers==4.57.3 > /dev/null 2>&1

accelerate launch --num_processes=8 \
        -m lmms_eval \
        --model $model_family \
        --model_args $model_args \
        --tasks $benchmark \
        --batch_size 1 \
        --log_samples \
        --log_samples_suffix $model \
        --output_path $output_path/$benchmark/$model
