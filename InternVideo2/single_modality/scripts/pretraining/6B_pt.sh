export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1

JOB_NAME='6B_pt'
OUTPUT_DIR="$(dirname $0)/$JOB_NAME"
LOG_DIR="./logs/${JOB_NAME}"
DATA_PATH='train_2M.csv'

PARTITION='video'
GPUS=256
GPUS_PER_NODE=8
CPUS_PER_TASK=14

srun -p $PARTITION \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    python -u run_pretraining.py \
    --data_path ${DATA_PATH} \
    --num_sample 1 \
    --flip \
    --mask_type 'attention'  \
    --mask_ratio 0.8 \
    --model 'pretrain_internvideo2_6B_patch14_224' \
    --clip_teacher 'internvl_clip_6b' \
    --clip_input_resolution 224 \
    --clip_teacher_embed_dim 3200 \
    --clip_teacher_final_dim 768 \
    --clip_loss_ratio 1 1 \
    --clip_norm_type 'l2' \
    --clip_return_attn \
    --clip_return_layer 6 \
    --clip_teacher_return_interval 1 \
    --clip_student_return_interval 1 \
    --mae_teacher 'mae_g14_hybrid' \
    --mae_tubelet_size 2 \
    --mae_loss_ratio 1 \
    --mae_norm_type 'l2' \
    --mae_teacher_embed_dim 1408 \
    --mae_return_layer 4 \
    --mae_teacher_return_interval 1 \
    --mae_student_return_interval 1 \
    --tubelet_size 1 \
    --lr 1.5e-4 \
    --drop_path 0.3 \
    --use_checkpoint \
    --checkpoint_num 48 \
    --layer_scale_init_value 1e-5 \
    --batch_size 8 \
    --num_segments 16 \
    --num_frames 16 \
    --sampling_rate 1 \
    --num_workers 12 \
    --opt adamw \
    --opt_eps 1e-6 \
    --opt_betas 0.9 0.98 \
    --clip_grad 3.0 \
    --weight_decay 0.05 \
    --warmup_epochs 40 \
    --save_ckpt_freq 50 \
    --epochs 301 \
    --enable_deepspeed \
    --bf16 \
    --zero_stage 1 \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR}