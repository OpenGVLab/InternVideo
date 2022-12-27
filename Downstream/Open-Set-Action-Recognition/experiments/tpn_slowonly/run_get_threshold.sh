#!/bin/bash

export CUDA_HOME='/usr/local/cuda'

pwd_dir=$pwd
cd ../../

source activate mmaction

DEVICE=$1
MODEL=$2
BATCHSIZE=$3
TRAIN_DATA='data/ucf101/ucf101_train_split_1_videos.txt'
RESULT_DIR='experiments/tpn_slowonly/results'

case ${MODEL} in
   dropout)
   #  get the BALD threshold for TPN_SlowOnly_Dropout model trained on UCF-101
   CUDA_VISIBLE_DEVICES=${DEVICE} python experiments/get_threshold.py \
      --config configs/recognition/tpn/inference_tpn_slowonly_dnn.py \
      --checkpoint work_dirs/tpn_slowonly/finetune_ucf101_tpn_slowonly_celoss/latest.pth \
      --train_data ${TRAIN_DATA} \
      --batch_size ${BATCHSIZE} \
      --uncertainty BALD \
      --result_prefix ${RESULT_DIR}/TPN_SlowOnly_Dropout_BALD
   ;;
   bnn)
   #  get the BALD threshold for TPN_SlowOnly_BNN model trained on UCF-101
   CUDA_VISIBLE_DEVICES=${DEVICE} python experiments/get_threshold.py \
      --config configs/recognition/tpn/inference_tpn_slowonly_bnn.py \
      --checkpoint work_dirs/tpn_slowonly/finetune_ucf101_tpn_slowonly_bnn/latest.pth \
      --train_data ${TRAIN_DATA} \
      --batch_size ${BATCHSIZE} \
      --uncertainty BALD \
      --result_prefix ${RESULT_DIR}/TPN_SlowOnly_BNN_BALD
   ;;
   edl)
   #  get the EDL threshold for TPN_SlowOnly_EDL model trained on UCF-101
   CUDA_VISIBLE_DEVICES=${DEVICE} python experiments/get_threshold.py \
      --config configs/recognition/tpn/inference_tpn_slowonly_dnn.py \
      --checkpoint work_dirs/tpn_slowonly/finetune_ucf101_tpn_slowonly_edlloss/latest.pth \
      --train_data ${TRAIN_DATA} \
      --batch_size ${BATCHSIZE} \
      --uncertainty EDL \
      --result_prefix ${RESULT_DIR}/TPN_SlowOnly_EDLlog_EDL
   ;;
   edl_nokl)
   #  get the EDL threshold for TPN_SlowOnly_EDL model trained on UCF-101
   CUDA_VISIBLE_DEVICES=${DEVICE} python experiments/get_threshold.py \
      --config configs/recognition/tpn/inference_tpn_slowonly_dnn.py \
      --checkpoint work_dirs/tpn_slowonly/finetune_ucf101_tpn_slowonly_edlloss_nokl/latest.pth \
      --train_data ${TRAIN_DATA} \
      --batch_size ${BATCHSIZE} \
      --uncertainty EDL \
      --result_prefix ${RESULT_DIR}/TPN_SlowOnly_EDLlogNoKL_EDL
   ;;
   edl_avuc)
   #  get the EDL threshold for TPN_SlowOnly_EDL_AvUC model trained on UCF-101
   CUDA_VISIBLE_DEVICES=${DEVICE} python experiments/get_threshold.py \
      --config configs/recognition/tpn/inference_tpn_slowonly_dnn.py \
      --checkpoint work_dirs/tpn_slowonly/finetune_ucf101_tpn_slowonly_edlloss_avuc/latest.pth \
      --train_data ${TRAIN_DATA} \
      --batch_size ${BATCHSIZE} \
      --uncertainty EDL \
      --result_prefix ${RESULT_DIR}/TPN_SlowOnly_EDLlogAvUC_EDL
   ;;
   edl_nokl_avuc)
   #  get the EDL threshold for TPN_SlowOnly_EDL_AvUC model trained on UCF-101
   CUDA_VISIBLE_DEVICES=${DEVICE} python experiments/get_threshold.py \
      --config configs/recognition/tpn/inference_tpn_slowonly_dnn.py \
      --checkpoint work_dirs/tpn_slowonly/finetune_ucf101_tpn_slowonly_edlloss_nokl_avuc/latest.pth \
      --train_data ${TRAIN_DATA} \
      --batch_size ${BATCHSIZE} \
      --uncertainty EDL \
      --result_prefix ${RESULT_DIR}/TPN_SlowOnly_EDLlogNoKLAvUC_EDL
   ;;
   edl_nokl_avuc_debias)
   #  get the EDL threshold for TPN_SlowOnly_EDL_AvUC model trained on UCF-101
   CUDA_VISIBLE_DEVICES=${DEVICE} python experiments/get_threshold.py \
      --config configs/recognition/tpn/inference_tpn_slowonly_enn.py \
      --checkpoint work_dirs/tpn_slowonly/finetune_ucf101_tpn_slowonly_edlloss_nokl_avuc_debias/latest.pth \
      --train_data ${TRAIN_DATA} \
      --batch_size ${BATCHSIZE} \
      --uncertainty EDL \
      --result_prefix ${RESULT_DIR}/TPN_SlowOnly_EDLlogNoKLAvUCDebias_EDL
   ;;
   *)
    echo "Invalid model: "${MODEL}
    exit
    ;;
esac


cd $pwd_dir
echo "Experiments finished!"