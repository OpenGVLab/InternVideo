export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1

OUTPUT_DIR='/mnt/petrelfs/huangbingkun/VideoMAE-clean/pfs_work_dir/vit_b_ssv2_pt'
DATA_PATH='/mnt/petrelfs/share_data/huangbingkun/data/sthv2/ssv2_train.csv'

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
        --data_root '/mnt/petrelfs/videointern' \
        --data_path ${DATA_PATH} \
        --mask_type t_consist  \
        --mask_ratio 0.9 \
        --model pretrain_mae_base_patch16_224 \
        --decoder_depth 4 \
        --batch_size 64 \
        --num_frames 16 \
        --sampling_rate 2 \
        --num_workers 8 \
        --opt adamw \
        --opt_betas 0.9 0.95 \
        --warmup_epochs 40 \
        --save_ckpt_freq 20 \
        --epochs 801 \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        ${PY_ARGS}
