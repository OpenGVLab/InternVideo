export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1
echo "PYTHONPATH: ${PYTHONPATH}"
which_python=$(which python)
echo "which python: ${which_python}"
export PYTHONPATH=${PYTHONPATH}:${which_python}
export PYTHONPATH=${PYTHONPATH}:.
echo "PYTHONPATH: ${PYTHONPATH}"

JOB_NAME='our_230522_resized_unmask'
# JOB_NAME='debug'
OUTPUT_DIR="$(dirname $0)/$JOB_NAME"
LOG_DIR="$(dirname $0)/logs/${JOB_NAME}"
PARTITION='Gvlab-P1'
NNODE=8
NUM_GPUS=8
NUM_CPU=128

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
    model.freeze_text False \
    wandb.enable False \
    output_dir ${OUTPUT_DIR} \
    pretrained_path exp/exp_pretrain_videoclip/our_230522/our_230522_resized/ckpt_09.pth \
    optimizer.lr 4e-6 \
    scheduler.epochs 0.5 \
    scheduler.warmup_epochs 0.1 \
    model.vision_encoder.masking_prob 0.0 \
    batch_size 32 \
    batch_size_test 4
