#!/bin/bash

export CUDA_HOME='/usr/local/cuda'

pwd_dir=$pwd
cd ../../

source activate mmaction

DEVICE=$1
MODEL=$2
BATCHSIZE=$3
TRAIN_DATA='data/ucf101/ucf101_train_split_1_videos.txt'
RESULT_DIR='experiments/i3d/results'

case ${MODEL} in
   dnn)
   #  get the BALD threshold for i3d model trained on UCF-101
   CUDA_VISIBLE_DEVICES=${DEVICE} python experiments/get_threshold.py \
      --config configs/recognition/i3d/inference_i3d_dnn.py \
      --checkpoint work_dirs/i3d/finetune_ucf101_i3d_dnn/latest.pth \
      --train_data ${TRAIN_DATA} \
      --batch_size ${BATCHSIZE} \
      --uncertainty BALD \
      --result_prefix ${RESULT_DIR}/I3D_DNN_BALD
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
   CUDA_VISIBLE_DEVICES=${DEVICE} python experiments/get_threshold.py \
      --config configs/recognition/i3d/inference_i3d_enn.py \
      --checkpoint work_dirs/i3d/finetune_ucf101_i3d_edlnokl/latest.pth \
      --train_data ${TRAIN_DATA} \
      --batch_size ${BATCHSIZE} \
      --uncertainty EDL \
      --result_prefix ${RESULT_DIR}/I3D_EDLNoKL_EDL
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