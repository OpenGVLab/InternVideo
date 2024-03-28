#!/bin/bash

export CUDA_HOME='/usr/local/cuda'

pwd_dir=$pwd
cd ../../

# DEVICE=$1
MODEL=$1
BATCHSIZE=$2
TRAIN_DATA='data/ucf101/ucf101_train_split_1_videos.txt'
RESULT_DIR='experiments/mae/results'
GPUS=$3
PORT=${PORT:-29500}

case ${MODEL} in
   dnn)
   #  get the BALD threshold for i3d model trained on UCF-101
   python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT experiments/get_threshold_dist.py \
      --config configs/recognition/mae/inference_vae_dnn.py \
      --checkpoint work_dirs/vae/finetune_ucf101_vae_dnn/latest.pth \
      --train_data ${TRAIN_DATA} \
      --batch_size ${BATCHSIZE} \
      --uncertainty BALD \
      --launcher pytorch \
      --result_prefix ${RESULT_DIR}/MAE_DNN_BALD
   ;;
   bnn)
   #  get the BALD threshold for I3D_BNN model trained on UCF-101
   CUDA_VISIBLE_DEVICES=${DEVICE} python experiments/get_threshold.py \
      --config configs/recognition/i3d/inference_i3d_bnn.py \
      --checkpoint work_dirs/i3d/finetune_ucf101_i3d_bnn/latest.pth \
      --train_data ${TRAIN_DATA} \
      --batch_size ${BATCHSIZE} \
      --uncertainty BALD \
      --result_prefix ${RESULT_DIR}/I3D_BNN_BALD
   ;;
   edlnokl)
   #  get the EDL threshold for I3D_EDL model trained on UCF-101
   python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT experiments/get_threshold_dist.py \
      --config configs/recognition/mae/inference_mae_enn.py \
      --checkpoint work_dirs/mae/ky/latest.pth \
      --train_data ${TRAIN_DATA} \
      --batch_size ${BATCHSIZE} \
      --uncertainty EDL \
      --result_prefix ${RESULT_DIR}/MAE_EDLNoKL_EDL \
      --launcher pytorch
   ;;
   edlnokl_avuc_debias)
   #  get the EDL threshold for I3D_EDL_AvUC_Debias model trained on UCF-101
   CUDA_VISIBLE_DEVICES=${DEVICE} python experiments/get_threshold.py \
      --config configs/recognition/i3d/inference_i3d_enn.py \
      --checkpoint work_dirs/i3d/finetune_ucf101_i3d_edlnokl_avuc_debias/latest.pth \
      --train_data ${TRAIN_DATA} \
      --batch_size ${BATCHSIZE} \
      --uncertainty EDL \
      --result_prefix ${RESULT_DIR}/I3D_EDLNoKLAvUCDebias_EDL
   ;;
   *)
    echo "Invalid model: "${MODEL}
    exit
    ;;
esac


cd $pwd_dir
echo "Experiments finished!"
