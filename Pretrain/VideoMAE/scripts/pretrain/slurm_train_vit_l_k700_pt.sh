export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1

OUTPUT_DIR='/mnt/petrelfs/huangbingkun/VideoMAE-clean/work_dir/vit_l_k700_pretrain'
DATA_PATH='/mnt/petrelfs/share_data/huangbingkun/data/k700/k700_sp_train.csv'
MODEL_PATH='/mnt/petrelfs/huangbingkun/VideoMAE-clean/work_dir/vit_l_hybrid_1m_pretrain/checkpoint-800.pth'

JOB_NAME=$1
PARTITION=${PARTITION:-"video"}
# 8 for 1 node, 16 for 2 node, etc.
GPUS=${GPUS:-32}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
CPUS_PER_TASK=${CPUS_PER_TASK:-14}
SRUN_ARGS=${SRUN_ARGS:-""}
PY_ARGS=${@:2}

# batch_size can be adjusted according to the graphics card
srun -p $PARTITION \
        --job-name=${JOB_NAME} \
        --gres=gpu:${GPUS_PER_NODE} \
        --ntasks=${GPUS} \
        --ntasks-per-node=${GPUS_PER_NODE} \
        --cpus-per-task=${CPUS_PER_TASK} \
        --kill-on-bad-exit=1 \
        --quotatype=auto \
        --async \
        ${SRUN_ARGS} \
        python -u run_mae_pretraining.py \
        --data_path ${DATA_PATH} \
        --finetune ${MODEL_PATH} \
        --mask_type t_consist  \
        --mask_ratio 0.9 \
        --model pretrain_mae_large_patch16_224 \
        --decoder_depth 12 \
        --batch_size 6 \
        --num_frames 16 \
        --sampling_rate 4 \
        --num_sample 4 \
        --num_workers 14 \
        --opt adamw \
        --opt_betas 0.9 0.95 \
        --warmup_epochs 10 \
        --save_ckpt_freq 10 \
        --epochs 201 \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        ${PY_ARGS}
