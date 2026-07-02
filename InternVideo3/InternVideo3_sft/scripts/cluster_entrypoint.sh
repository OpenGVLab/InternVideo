#!/bin/bash
# Cluster entrypoint: install dependencies then launch training
# Called by rjob_submit.sh on each node
set -ex

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG=${1:-"${ROOT_DIR}/configs/internvideo3_sft.py"}

# Install
bash "${SCRIPT_DIR}/install.sh"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH}"

# Launch training (use all 8 GPUs per node)
bash "${SCRIPT_DIR}/train.sh" "${CONFIG}" 8
