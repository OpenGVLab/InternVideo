#!/bin/bash

# =================================
# my device          RTX4090 * 4
# python             3.10.18
# flash-attn         2.5.7
# torch              2.4.1+cu124
# torchaudio         2.4.1+cu124
# torchvision        0.19.1+cu124
# =================================

export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1

echo "PYTHONPATH: ${PYTHONPATH}"
which_python=$(which python)
echo "Using Python at: ${which_python}"
export PYTHONPATH=${PYTHONPATH}:${which_python}
export PYTHONPATH=${PYTHONPATH}:.
echo "Updated PYTHONPATH: ${PYTHONPATH}"

JOB_NAME=$(basename $0)_$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="$(dirname $0)/$JOB_NAME"
LOG_DIR="$(dirname $0)/logs/${JOB_NAME}"

NNODE=1
NUM_GPUS=4
# export CUDA_VISIBLE_DEVICES="2,3"

# When not using srun, it is necessary to annotate the line about SLURM in tasks/pretrain.py
torchrun \
    --nnodes=${NNODE} \
    --nproc_per_node=${NUM_GPUS} \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:${MASTER_PORT} \
    tasks/pretrain.py \
    $(dirname $0)/config_didemo.py \
    output_dir ${OUTPUT_DIR} \
    evaluate True \
    pretrained_path 'your_model_path/1B_stage2_pt.pth'