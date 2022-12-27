export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1

OUTPUT_DIR='/mnt/petrelfs/huangbingkun/VideoMAE-clean/work_dir/vit_l_hybrid_pt_ssv2_ft_tsn'
DATA_PATH='/mnt/petrelfs/share_data/huangbingkun/data/sthv2'
MODEL_PATH='/mnt/petrelfs/huangbingkun/VideoMAE-clean/work_dir/vit_l_hybrid_1m_pretrain/checkpoint-800.pth'

JOB_NAME=$1
PARTITION=${PARTITION:-"video"}
# 8 for 1 node, 16 for 2 node, etc.
GPUS=${GPUS:-32}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
CPUS_PER_TASK=${CPUS_PER_TASK:-8}
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
        --exclude=SH-IDC1-10-140-0-165 \
        ${SRUN_ARGS} \
        python -u run_class_finetuning.py \
        --model vit_large_patch16_224 \
        --data_set SSV2 \
        --nb_classes 174 \
        --data_path ${DATA_PATH} \
        --finetune ${MODEL_PATH} \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --batch_size 12 \
        --num_sample 1 \
        --input_size 224 \
        --short_side_size 224 \
        --save_ckpt_freq 10 \
        --num_frames 16 \
        --opt adamw \
        --lr 5e-4 \
        --num_workers 8 \
        --opt_betas 0.9 0.999 \
        --weight_decay 0.05 \
        --drop_path 0.2 \
        --epochs 30 \
        --dist_eval \
        --test_num_segment 2 \
        --test_num_crop 3 \
        --enable_deepspeed \
        ${PY_ARGS}
