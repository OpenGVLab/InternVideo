export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1

JOB_NAME='B14_ft_k710_f8'
OUTPUT_DIR="$(dirname $0)/$JOB_NAME"
LOG_DIR="./logs/${JOB_NAME}"
DATA_PATH='your_data_path/k710'
MODEL_PATH='your_model_path/B14_dist_1B_stage2.pth'

PARTITION='video'
GPUS=16
GPUS_PER_NODE=8
CPUS_PER_TASK=16

srun -p $PARTITION \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    python run_finetuning.py \
    --model internvideo2_base_patch14_224 \
    --data_path ${DATA_PATH} \
    --data_set 'Kinetics_sparse' \
    --split ',' \
    --nb_classes 710 \
    --finetune ${MODEL_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --steps_per_print 10 \
    --batch_size 16 \
    --num_sample 2 \
    --input_size 224 \
    --short_side_size 224 \
    --save_ckpt_freq 100 \
    --num_frames 8 \
    --num_workers 12 \
    --warmup_epochs 4 \
    --tubelet_size 1 \
    --epochs 20 \
    --lr 2e-4 \
    --drop_path 0.1 \
    --head_drop_path 0.1 \
    --fc_drop_rate 0.0 \
    --layer_decay 0.75 \
    --layer_scale_init_value 1e-5 \
    --opt adamw \
    --opt_betas 0.9 0.999 \
    --weight_decay 0.05 \
    --test_num_segment 4 \
    --test_num_crop 3 \
    --dist_eval \
    --enable_deepspeed \
    --bf16 \
    --zero_stage 1 \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --test_best
    
