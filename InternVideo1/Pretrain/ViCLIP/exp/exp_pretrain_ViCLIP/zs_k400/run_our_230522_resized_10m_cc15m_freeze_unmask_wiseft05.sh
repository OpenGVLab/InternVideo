export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1
echo "PYTHONPATH: ${PYTHONPATH}"
which_python=$(which python)
echo "which python: ${which_python}"
export PYTHONPATH=${PYTHONPATH}:${which_python}
export PYTHONPATH=${PYTHONPATH}:.
echo "PYTHONPATH: ${PYTHONPATH}"

JOB_NAME='run_our_230522_resized_10m_cc15m_freeze_unmask_wiseft05'
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
    tasks/pretrain.py \
    $(dirname $0)/config.py \
    wandb.enable False \
    output_dir ${OUTPUT_DIR} \
    train_corpus webvid_dummy \
    evaluate True \
    pretrained_path exp/exp_pretrain_videoclip/our_230522_10m_cc15m/our_230522_resized_10m_cc15m_freeze_unmask/ckpt_00.pth \
    wiseft.enable True 
