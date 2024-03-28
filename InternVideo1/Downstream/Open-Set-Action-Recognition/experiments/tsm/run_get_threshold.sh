#!/bin/bash

export CUDA_HOME='/usr/local/cuda'

pwd_dir=$pwd
cd ../../

source activate mmaction

DEVICE=$1
MODEL=$2
BATCHSIZE=$3
TRAIN_DATA='data/ucf101/ucf101_train_split_1_videos.txt'
RESULT_DIR='experiments/tsm/results'

case ${MODEL} in
   dnn)
   #  get the BALD threshold for tsm model trained on UCF-101
   CUDA_VISIBLE_DEVICES=${DEVICE} python experiments/get_threshold.py \
      --config configs/recognition/tsm/inference_tsm_dnn.py \
      --checkpoint work_dirs/tsm/finetune_ucf101_tsm_dnn/latest.pth \
      --train_data ${TRAIN_DATA} \
      --batch_size ${BATCHSIZE} \
      --uncertainty BALD \
      --result_prefix ${RESULT_DIR}/TSM_DNN_BALD
   ;;
   bnn)
   #  get the BALD threshold for tsm model trained on UCF-101
   CUDA_VISIBLE_DEVICES=${DEVICE} python experiments/get_threshold.py \
      --config configs/recognition/tsm/inference_tsm_bnn.py \
      --checkpoint work_dirs/tsm/finetune_ucf101_tsm_bnn/latest.pth \
      --train_data ${TRAIN_DATA} \
      --batch_size ${BATCHSIZE} \
      --uncertainty BALD \
      --result_prefix ${RESULT_DIR}/TSM_BNN_BALD
   ;;
   edlnokl_avuc_debias)
   #  get the EDL threshold for TSM_EDL_AvUC_Debias model trained on UCF-101
   CUDA_VISIBLE_DEVICES=${DEVICE} python experiments/get_threshold.py \
      --config configs/recognition/tsm/inference_tsm_enn.py \
      --checkpoint work_dirs/tsm/finetune_ucf101_tsm_edlnokl_avuc_debias/latest.pth \
      --train_data ${TRAIN_DATA} \
      --batch_size ${BATCHSIZE} \
      --uncertainty EDL \
      --result_prefix ${RESULT_DIR}/TSM_EDLNoKLAvUCDebias_EDL
   ;;
   *)
    echo "Invalid model: "${MODEL}
    exit
    ;;
esac


cd $pwd_dir
echo "Experiments finished!"