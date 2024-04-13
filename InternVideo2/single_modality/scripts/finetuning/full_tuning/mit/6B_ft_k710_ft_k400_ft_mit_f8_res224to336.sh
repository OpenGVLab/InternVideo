export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1

JOB_NAME='6B_ft_k710_ft_k400_ft_mit_f8_res224to336'
OUTPUT_DIR="$(dirname $0)/$JOB_NAME"
LOG_DIR="./logs/${JOB_NAME}"
PREFIX='your_data_path/mit'
DATA_PATH='your_data_path/mit'
MODEL_PATH='6B_ft_k710_ft_k400_ft_mit_f8'

PARTITION='video'
GPUS=64
GPUS_PER_NODE=8
CPUS_PER_TASK=16

srun -p $PARTITION \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    python run_finetuning.py \
    --model internvideo2_6B_patch14_224 \
    --data_path ${DATA_PATH} \
    --prefix ${PREFIX} \
    --data_set 'mitv1_sparse' \
    --nb_classes 339 \
    --finetune ${MODEL_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 4 \
    --num_sample 1 \
    --input_size 336 \
    --short_side_size 336 \
    --save_ckpt_freq 100 \
    --num_frames 8 \
    --sampling_rate 8 \
    --num_workers 12 \
    --warmup_epochs 0 \
    --tubelet_size 1 \
    --epochs 2 \
    --lr 1e-5 \
    --min_lr 0 \
    --drop_path 0.4 \
    --head_drop_path 0.4 \
    --layer_decay 0.915 \
    --use_checkpoint \
    --checkpoint_num 40 \
    --layer_scale_init_value 1e-5 \
    --opt adamw \
    --opt_betas 0.9 0.999 \
    --weight_decay 0.1 \
    --test_num_segment 4 \
    --test_num_crop 3 \
    --dist_eval \
    --enable_deepspeed \
    --bf16 \
    --zero_stage 1 \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --test_best 
    
