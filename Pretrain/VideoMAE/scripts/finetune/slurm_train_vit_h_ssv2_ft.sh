export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1

OUTPUT_DIR='/mnt/petrelfs/huangbingkun/VideoMAE-clean/pfs_work_dir/vit_h_hybridv2_pt_ssv2_ft_tsn'
DATA_PATH='/mnt/petrelfs/share_data/huangbingkun/data/sthv2'
MODEL_PATH='/mnt/cache/share_data/huangbingkun/model/vit_h_hybridv2_pt_1200e.pth'

JOB_NAME=$1
PARTITION=${PARTITION:-"video"}
# 8 for 1 node, 16 for 2 node, etc.
GPUS=${GPUS:-32}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
CPUS_PER_TASK=${CPUS_PER_TASK:-10}
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
        python -u run_class_finetuning.py \
        --model vit_huge_patch16_224 \
        --data_set SSV2 \
        --nb_classes 174 \
        --data_path ${DATA_PATH} \
        --finetune ${MODEL_PATH} \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --batch_size 8 \
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
        --head_drop_rate 0.3 \
        --layer_decay 0.8 \
        --epochs 30 \
        --test_num_segment 2 \
        --test_num_crop 3 \
        --dist_eval --enable_deepspeed \
        ${PY_ARGS}
