#!/bin/bash
# Submit InternVideo3 SFT training job via rjob
# Usage: bash scripts/rjob_submit.sh [num_gpus] [config_file]
set -ex

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

GPU=${1:-64}
CONFIG=${2:-"${ROOT_DIR}/configs/internvideo3_sft.py"}
NNODES=$(( (GPU + 7) / 8 ))

job_name="internvideo3-sft-g${GPU}-$(date +%Y%m%d-%H%M%S)"

rjob submit \
    --name=$job_name \
    --gpu=8 --memory=1638400 --charged-group=ptdata_gpu --cpu=128 \
    --private-machine=group \
    -P $NNODES \
    --image registry.h.pjlab.org.cn/ailab-ptdata/yujiashuo-workspace:iv3train402w_libfuse2 \
    --mount=gpfs://gpfs1/heyinan:/mnt/shared-storage-user/heyinan \
    --mount=gpfs://gpfs1/yanziang:/mnt/shared-storage-user/yanziang \
    --mount=gpfs://gpfs1/yujiashuo:/mnt/shared-storage-user/yujiashuo \
    --mount=gpfs://gpfs1/wangyi:/mnt/shared-storage-user/wangyi \
    --mount=gpfs://gpfs1/video-shared:/mnt/shared-storage-user/video-shared \
    --mount=gpfs://gpfs1/puyullmgpu-shared:/mnt/shared-storage-user/puyullmgpu-shared \
    --mount=gpfs://gpfs1/puyudelivery:/mnt/shared-storage-user/puyudelivery \
    --mount=gpfs://gpfs2/sfteval:/mnt/shared-storage-user/sfteval \
    --mount=gpfs://gpfs1/large-model-center-share-weights:/mnt/shared-storage-user/large-model-center-share-weights \
    --host-network=true \
    --gang-start=true \
    --custom-resources rdma/mlnx_shared=8 \
    --custom-resources mellanox.com/mlnx_rdma=1 \
    --custom-resources brainpp.cn/fuse=1 \
    --store-host-nvme \
    --termination-grace-period-seconds 600 \
    -e DISTRIBUTED_JOB=true \
    -- bash -c "bash ${SCRIPT_DIR}/cluster_entrypoint.sh ${CONFIG}"
