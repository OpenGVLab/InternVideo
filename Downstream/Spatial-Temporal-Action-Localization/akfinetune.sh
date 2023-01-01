#!/bin/sh
export MASTER_PORT=$((12000 + $RANDOM % 20000))
#module load anaconda/2021.05
#conda activate mae_stad
export NCCL_IB_HCA=mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TIMEOUT=23
export NCCL_IB_RETRY_CNT=7

module load nccl/2.11.4-1_cuda11.1.1
# Set the path to save checkpoints
OUTPUT_DIR='./workdir/vit_h_ak_good'
# path to Kinetics set (train.csv/val.csv/test.csv)
DATA_PATH='YOUR_PATH/list_kinetics-400'  # no use
# path to pretrain model
MODEL_PATH='/data/home/scw6003/chenguo/vit_ckpt/vit_h_hybridv2_pt_1200e_k700_ft.pth'

# We add repeated_aug (--num_sample = 2) on Kinetics-400 here, 
# which could better performance while need more time for fine-tuning
NUM_NODES=${NUM_NODES:-4}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
CPUS_PER_TASK=${CPUS_PER_TASK:-12}
# batch_size can be adjusted according to number of GPUs
# this script is for 64 GPUs (8 nodes x 8 GPUs)
srun -N 4 \
     --gres=gpu:8 \
     --ntasks-per-node=8 \
     --cpus-per-task=14 \
     --qos=gpugpu \
     python -u run_class_finetuning.py \
      --model vit_huge_patch16_224 \
      --data_path ${DATA_PATH} \
      --finetune ${MODEL_PATH} \
      --log_dir ${OUTPUT_DIR} \
      --output_dir ${OUTPUT_DIR} \
      --batch_size 4 \
      --update_freq 1 \
      --num_sample 1 \
      --input_size 224 \
      --save_ckpt_freq 1 \
      --num_frames 16 \
      --sampling_rate 4 \
      --opt adamw \
      --lr 0.00025 \
      --opt_betas 0.9 0.999 \
      --weight_decay 0.05 \
      --epochs 30 \
      --data_set "ava" \
      --val_freq 30 \
      --drop_path 0.2 \
      --enable_deepspeed \

