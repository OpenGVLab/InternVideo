#!/bin/bash
# Quick debug run on single GPU
set -ex

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

bash "${SCRIPT_DIR}/train.sh" "${ROOT_DIR}/configs/internvideo3_sft_debug.py" 1
