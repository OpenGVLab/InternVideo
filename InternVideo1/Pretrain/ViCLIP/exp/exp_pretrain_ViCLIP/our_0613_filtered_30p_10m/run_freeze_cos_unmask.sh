export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1
echo "PYTHONPATH: ${PYTHONPATH}"
which_python=$(which python)
echo "which python: ${which_python}"
export PYTHONPATH=${PYTHONPATH}:${which_python}
export PYTHONPATH=${PYTHONPATH}:.
echo "PYTHONPATH: ${PYTHONPATH}"

JOB_NAME='our_0613_filtered_30p_10m_freeze_cos_unmask'
# JOB_NAME='debug'
OUTPUT_DIR="$(dirname $0)/$JOB_NAME"
LOG_DIR="$(dirname $0)/logs/${JOB_NAME}"
PARTITION='video'
NNODE=1
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
    wandb.enable False \
    model.vision_encoder.pretrained 'CLIP-ViT-L/14' \
    model.text_encoder.pretrained 'CLIP-ViT-L/14' \
    criterion.loss_weight.vtc 0.0 \
    criterion.loss_weight.cos 1.0 \
    pretrained_path exp/exp_pretrain_videoclip/our_0613_filtered_30p_10m/our_0613_filtered_30p_10m_freeze_cos/ckpt_best.pth \
    output_dir ${OUTPUT_DIR} \
    optimizer.lr 4e-6 \
    scheduler.epochs 0.5 \
    scheduler.warmup_epochs 0.1 \
    model.vision_encoder.masking_prob 0.0 \
    batch_size 32 \
    batch_size_test 4

