unset http_proxy; unset https_proxy; unset HTTP_PROXY; unset HTTPS_PROXY
JOB_NAME='data-annotate_check'
OUTPUT_DIR="$(dirname $0)/$JOB_NAME"
LOG_DIR="$(dirname $0)/logs/${JOB_NAME}"
PARTITION='Video-aigc-general'
NNODE=1
NUM_GPUS=1
NUM_CPU=16

srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    -n${NNODE} \
    --gres=gpu:${NUM_GPUS} \
    --ntasks-per-node=1 \
    --cpus-per-task=${NUM_CPU} \
    jupyter lab --ip=0.0.0.0