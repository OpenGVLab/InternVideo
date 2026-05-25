#!/bin/bash
# Usage: Set MODEL_PATH before running this script.
#   export MODEL_PATH=/path/to/your/model
#   bash eval_all.sh
set -e

export MODEL_PATH="${MODEL_PATH:?Please set MODEL_PATH environment variable}"
export OUTPUT_DIR="${OUTPUT_DIR:-./logs}"

SCRIPT_DIR=$(dirname "$0")

declare -A tasks=(
    ["eval_mvbench"]="${SCRIPT_DIR}/eval_mvbench.sh"
    ["eval_lvbench"]="${SCRIPT_DIR}/eval_lvbench.sh"
    ["eval_videomme"]="${SCRIPT_DIR}/eval_videomme.sh"
    ["eval_videommmu"]="${SCRIPT_DIR}/eval_videommmu.sh"
    ["eval_vsibench"]="${SCRIPT_DIR}/eval_vsibench.sh"
    ["eval_mlvu"]="${SCRIPT_DIR}/eval_mlvu.sh"
    ["eval_longvideobench"]="${SCRIPT_DIR}/eval_longvideobench.sh"
)

for name in "${!tasks[@]}"; do
    sh_file="${tasks[$name]}"
    echo "Running: $name ($sh_file)"
    bash "$sh_file"
done
