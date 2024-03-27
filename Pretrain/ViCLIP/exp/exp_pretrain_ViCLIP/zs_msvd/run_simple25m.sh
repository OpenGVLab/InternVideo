export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1
echo "PYTHONPATH: ${PYTHONPATH}"
which_python=$(which python)
echo "which python: ${which_python}"
export PYTHONPATH=${PYTHONPATH}:${which_python}
export PYTHONPATH=${PYTHONPATH}:.
echo "PYTHONPATH: ${PYTHONPATH}"

JOB_NAME='simple25m'
# JOB_NAME='debug'
OUTPUT_DIR="$(dirname $0)/$JOB_NAME"
LOG_DIR="$(dirname $0)/logs/${JOB_NAME}"
PARTITION='video3'
NNODE=1
NUM_GPUS=2
NUM_CPU=16

srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    -n${NNODE} \
    --gres=gpu:${NUM_GPUS} \
    --ntasks-per-node=1 \
    --cpus-per-task=${NUM_CPU} \
    torchrun.sh \
    --nnodes=${NNODE} \
    --nproc_per_node=${NUM_GPUS} \
    --rdzv_backend=c10d \
    tasks/retrieval.py \
    $(dirname $0)/config.py \
    wandb.enable False \
    output_dir ${OUTPUT_DIR} \
    train_corpus webvid_dummy \
    evaluate True \
    pretrained_path phdd:s3://liyizhuo/projects/vindlu/exp/exp_pretrain_videoclip/simple_25m/simple_25m/ckpt_09.pth
