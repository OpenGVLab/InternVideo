#!/bin/bash
# InternVideo3 SFT Training Entrypoint
# Usage: bash scripts/train.sh <config_file> [num_gpus]
#
# Examples:
#   bash scripts/train.sh configs/internvideo3_sft.py 8
#   bash scripts/train.sh configs/internvideo3_sft_debug.py 1
set -ex

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

# Configuration
config_file=${1:-"${ROOT_DIR}/configs/internvideo3_sft.py"}
PROC_PER_NODE=${2:-8}

# Environment
export XTUNER_PACK_WORKERS=${XTUNER_PACK_WORKERS:-8}
export XTUNER_TOKENIZE_WORKERS=${XTUNER_TOKENIZE_WORKERS:-16}
export XTUNER_USE_FA3=${XTUNER_USE_FA3:-1}
export XTUNER_GC_ENABLE=${XTUNER_GC_ENABLE:-1}
export XTUNER_SKIP_EMPTY_THINK=${XTUNER_SKIP_EMPTY_THINK:-1}
export AV_LOG_FORCE_NOCOLOR=1
export AV_LOG_LEVEL=16
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH}"
export HF_HOME="${ROOT_DIR}/"

# Distributed settings (auto-detect from cluster or use defaults)
NODE_COUNT=${NODE_COUNT:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-$((12000 + RANDOM % 20000))}

echo "========================================="
echo "InternVideo3 SFT Training"
echo "Config: ${config_file}"
echo "Nodes: ${NODE_COUNT}, Rank: ${NODE_RANK}"
echo "GPUs per node: ${PROC_PER_NODE}"
echo "Master: ${MASTER_ADDR}:${MASTER_PORT}"
echo "========================================="

torchrun \
    --nnodes=${NODE_COUNT} \
    --node_rank=${NODE_RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    --nproc_per_node=${PROC_PER_NODE} \
    "${ROOT_DIR}/xtuner/v1/train/cli/sft.py" --config "${config_file}"
