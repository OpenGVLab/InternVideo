export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1

OUTPUT_DIR='/mnt/petrelfs/huangbingkun/VideoMAE-clean/work_dir/vit_b_ssv2_pt_dm05_ssv2_ft'
DATA_PATH='/mnt/petrelfs/share_data/huangbingkun/data/sthv2'
MODEL_PATH='/mnt/petrelfs/huangbingkun/work_dir_VideoMAE/vit_b_ssv2_pt_dm05/checkpoint-800.pth'

JOB_NAME=$1
PARTITION=${PARTITION:-"video"}
# 8 for 1 node, 16 for 2 node, etc.
GPUS=${GPUS:-16}
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
        python -u run_class_finetuning.py \
        --model vit_base_patch16_224 \
        --data_set SSV2 \
        --nb_classes 174 \
        --data_path ${DATA_PATH} \
        --finetune ${MODEL_PATH} \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --batch_size 16 \
        --num_sample 2 \
        --input_size 224 \
        --short_side_size 224 \
        --save_ckpt_freq 10 \
        --num_frames 16 \
        --opt adamw \
        --lr 2e-3 \
        --num_workers 8 \
        --layer_decay 0.65 \
        --opt_betas 0.9 0.999 \
        --weight_decay 0.05 \
        --epochs 25 \
        --dist_eval \
        --test_num_segment 2 \
        --test_num_crop 3 \
        --enable_deepspeed \
        ${PY_ARGS}
