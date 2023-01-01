export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1
export PATH=/mnt/lustre/share/gcc/gcc-5.3.0/bin/:$PATH
export LD_LIBRARY_PATH=/mnt/lustre/share/gcc/gmp-4.3.2/lib:/mnt/lustre/share/gcc/mpfr-2.4.2/lib:/mnt/lustre/share/gcc/mpc-0.8.1/lib:$LD_LIBRARY_PATH
export PATH=/mnt/cache/share/cuda-11.3/bin:$PATH
export PATH=/mnt/cache/xingsen/.local/bin:$PATH
export LD_LIBRARY_PATH=/mnt/cache/share/cuda-11.3/lib64:$LD_LIBRARY_PATH
DATA_PATH='YOUR_PATH/list_kinetics-400'
MODEL_PATH='/mnt/petrelfs/share_data/huangbingkun/model/vit_g_ps14_hybridv3_pt_1200e_k710_ft_v2_8022.pth'
OUTPUT_DIR='/mnt/petrelfs/xingsen/videoMAE_ckp/vitg_hbk'
PARTITION=${PARTITION:-"video"}
GPUS=${GPUS:-32}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
CPUS_PER_TASK=${CPUS_PER_TASK:-12}
SRUN_ARGS=${SRUN_ARGS:-""}
PY_ARGS=${@:2}
srun -p video \
     --gres=gpu:${GPUS_PER_NODE} \
     --ntasks=${GPUS} \
     --ntasks-per-node=${GPUS_PER_NODE} \
     --cpus-per-task=${CPUS_PER_TASK} \
     --quotatype=reserved \
     -x SH-IDC1-10-140-0-163 \
     --async \
     ${SRUN_ARGS} \
     python -u run_class_finetuning.py \
      --model vit_giant_patch14_224 \
      --data_path ${DATA_PATH} \
      --finetune ${MODEL_PATH} \
      --log_dir ${OUTPUT_DIR} \
      --output_dir ${OUTPUT_DIR} \
      --batch_size 4 \
      --num_sample 1 \
      --input_size 224 \
      --save_ckpt_freq 1 \
      --num_frames 16 \
      --sampling_rate 4 \
      --opt adamw \
      --lr 0.0005 \
      --opt_betas 0.9 0.999 \
      --weight_decay 0.05 \
      --layer_decay 0.9 \
      --epochs 20 \
      --data_set "ava" \
      --enable_deepspeed \
      --val_freq 20 \
      --drop_path 0.2 \
      ${PY_ARGS}
