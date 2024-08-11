export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1

JOB_NAME='L14_dist_1B_stage2'
OUTPUT_DIR="$(dirname $0)/$JOB_NAME"
LOG_DIR="./logs/${JOB_NAME}"
DATA_PATH='train_1.1M.csv'

PARTITION='video'
GPUS=32
GPUS_PER_NODE=8
CPUS_PER_TASK=16

srun -p $PARTITION \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    python -u run_distill.py \
    --data_path ${DATA_PATH} \
    --num_sample 1 \
    --flip \
    --mask_type 'attention'  \
    --mask_ratio 0.8 \
    --model 'distill_internvideo2_small_patch14_224' \
    --clip_teacher 'teacher_internvideo2_stage2_1B' \
    --clip_input_resolution 224 \
    --clip_teacher_embed_dim 1408 \
    --clip_teacher_final_dim 768 \
    --clip_loss_ratio 1 1 \
    --clip_norm_type 'l2' \
    --clip_return_attn \
    --clip_return_layer 6 \
    --clip_teacher_return_interval 3.34 \
    --clip_student_return_interval 1 \
    --clip_student_decoder 'MLP_Decoder' \
    --tubelet_size 1 \
    --lr 1e-3 \
    --drop_path 0.05 \
    --use_checkpoint \
    --checkpoint_num 0 \
    --layer_scale_init_value 1e-5 \
    --batch_size 64 \
    --num_segments 8 \
    --num_frames 8 \
    --sampling_rate 1 \
    --num_workers 12 \
    --opt adamw \
    --opt_eps 1e-6 \
    --opt_betas 0.9 0.98 \
    --clip_grad 5.0 \
    --weight_decay 0.05 \
    --warmup_epochs 20 \
    --save_ckpt_freq 1000 \
    --epochs 101 \
    --enable_deepspeed \
    --bf16 \
    --zero_stage 1 \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR}