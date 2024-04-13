export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1

JOB_NAME='1B_ft_ssv1_f8'
OUTPUT_DIR="$(dirname $0)/$JOB_NAME"
LOG_DIR="./logs/${JOB_NAME}"
PREFIX='your_data_path/ssv1_frame'
DATA_PATH='your_data_path/ssv1_frame'
MODEL_PATH='your_model_path/1B_pt.pth'

PARTITION='video'
GPUS=32
GPUS_PER_NODE=8
CPUS_PER_TASK=16

srun -p $PARTITION \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    python run_finetuning.py \
    --model internvideo2_1B_patch14_224 \
    --data_path ${DATA_PATH} \
    --prefix ${PREFIX} \
    --data_set 'SSV2' \
    --filename_tmpl '{:05}.jpg' \
    --no_use_decord \
    --nb_classes 174 \
    --finetune ${MODEL_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 8 \
    --num_sample 2 \
    --input_size 224 \
    --short_side_size 224 \
    --save_ckpt_freq 100 \
    --num_frames 8 \
    --num_workers 12 \
    --warmup_epochs 3 \
    --tubelet_size 1 \
    --epochs 9 \
    --lr 1e-4 \
    --drop_path 0.3 \
    --layer_decay 0.915 \
    --use_checkpoint \
    --checkpoint_num 6 \
    --layer_scale_init_value 1e-5 \
    --opt adamw \
    --opt_betas 0.9 0.999 \
    --weight_decay 0.05 \
    --test_num_segment 2 \
    --test_num_crop 3 \
    --dist_eval \
    --enable_deepspeed \
    --bf16 \
    --zero_stage 1 \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --test_best \
    2>&1 | tee "$(dirname $0)/log_$JOB_NAME.txt"
    
