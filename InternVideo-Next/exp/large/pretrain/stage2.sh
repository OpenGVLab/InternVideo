export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

JOB_NAME='internvideo_next_vit_large_s2'
OUTPUT_DIR="$(dirname $0)/$JOB_NAME"
LOG_DIR="./logs/${JOB_NAME}"
DATA_PATH='{your_data_here}'

PARTITION='videop1'
GPUS=128
GPUS_PER_NODE=8
CPUS_PER_TASK=16

srun -p $PARTITION \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    python -u main_stage2.py \
    --mom 1.0 \
    --stage1_checkpoint '{your_stage1_ckpt}' \
    --data_path ${DATA_PATH} \
    --num_sample 1 \
    --flip \
    --model 'internvideo_next_stage2_large' \
    --clip_input_resolution 224 \
    --clip_input_frame 32 \
    --input_size 224 \
    --num_segments 32 \
    --num_frames 32 \
    --tubelet_size 1 \
    --lr 1e-4 \
    --drop_path 0.2 \
    --layer_scale_init_value 1e-5 \
    --batch_size 8 \
    --sampling_rate 1 \
    --num_workers 14 \
    --opt adamw \
    --opt_eps 1e-6 \
    --opt_betas 0.9 0.98 \
    --clip_grad 5.0 \
    --weight_decay 0.05 \
    --warmup_epochs 0 \
    --save_ckpt_freq 80 \
    --epochs 101 \
    --enable_deepspeed \
    --bf16 \
    --zero_stage 1 \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR}