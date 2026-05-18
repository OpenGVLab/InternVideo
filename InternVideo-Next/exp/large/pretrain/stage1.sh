export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

JOB_NAME='internvideo_next_vit_large_s1'
OUTPUT_DIR="$(dirname $0)/$JOB_NAME"
LOG_DIR="./logs/${JOB_NAME}"
DATA_PATH='{your_data_here}'

PARTITION='video'
GPUS=128
GPUS_PER_NODE=8
CPUS_PER_TASK=16

srun -p $PARTITION \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    python -u main_stage1.py \
    --data_path ${DATA_PATH} \
    --num_sample 1 \
    --flip \
    --mask_type 'attention' \
    --mask_ratio 0.75 \
    --reconstruction_ratio 0.0 \
    --model 'internvideo_next_stage1_base' \
    --clip_teacher 'teacher_siglip2_1b_once4all_mm_umt_res256' \
    --clip_input_resolution 256 \
    --clip_input_frame 16 \
    --input_size 224 \
    --num_segments 16 \
    --num_frames 16 \
    --clip_teacher_embed_dim 1536 \
    --clip_teacher_final_dim 1536 \
    --clip_loss_ratio 1 1 1 \
    --clip_norm_type 'l2' \
    --clip_return_attn \
    --clip_return_layer 6 \
    --clip_teacher_return_interval 1.67 \
    --clip_student_return_interval 1 \
    --tubelet_size 1 \
    --lr 1e-3 \
    --drop_path 0.2 \
    --layer_scale_init_value 1e-5 \
    --batch_size 16 \
    --sampling_rate 1 \
    --num_workers 14 \
    --opt adamw \
    --opt_eps 1e-6 \
    --opt_betas 0.9 0.98 \
    --clip_grad 5.0 \
    --weight_decay 0.05 \
    --warmup_epochs 7 \
    --save_ckpt_freq 20 \
    --epochs 62 \
    --enable_deepspeed \
    --bf16 \
    --zero_stage 1 \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR}