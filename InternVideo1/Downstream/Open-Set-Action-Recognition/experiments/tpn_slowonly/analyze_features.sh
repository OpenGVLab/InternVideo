#!/bin/bash

export CUDA_HOME='/usr/local/cuda'

pwd_dir=$pwd
cd ../../

source activate mmaction

DEVICE=$1
IND_DATA="data/ucf101/ucf101_val_split_1_videos.txt"
OOD_DATA="data/hmdb51/hmdb51_val_split_1_videos.txt"
RESULT_PATH="experiments/tpn_slowonly/results_tSNE"


CUDA_VISIBLE_DEVICES=${DEVICE} python experiments/analyze_features.py \
    --config configs/recognition/tpn/inference_tpn_slowonly_bnn.py \
    --checkpoint work_dirs/tpn_slowonly/finetune_ucf101_tpn_slowonly_bnn/latest.pth \
    --known_split ${IND_DATA} \
    --unknown_split ${OOD_DATA} \
    --result_file ${RESULT_PATH}/tSNE_bnn_HMDB.png

CUDA_VISIBLE_DEVICES=${DEVICE} python experiments/analyze_features.py \
    --config configs/recognition/tpn/inference_tpn_slowonly_dnn.py \
    --checkpoint work_dirs/tpn_slowonly/finetune_ucf101_tpn_slowonly_celoss/latest.pth \
    --known_split ${IND_DATA} \
    --unknown_split ${OOD_DATA} \
    --result_file ${RESULT_PATH}/tSNE_dnn_HMDB.png

CUDA_VISIBLE_DEVICES=${DEVICE} python experiments/analyze_features.py \
    --config configs/recognition/tpn/inference_tpn_slowonly_enn.py \
    --checkpoint work_dirs/tpn_slowonly/finetune_ucf101_tpn_slowonly_edlloss_nokl/latest.pth \
    --known_split ${IND_DATA} \
    --unknown_split ${OOD_DATA} \
    --result_file ${RESULT_PATH}/tSNE_enn_HMDB.png

CUDA_VISIBLE_DEVICES=${DEVICE} python experiments/analyze_features.py \
    --config configs/recognition/tpn/inference_tpn_slowonly_enn.py \
    --checkpoint work_dirs/tpn_slowonly/finetune_ucf101_tpn_slowonly_edlloss_nokl_avuc/latest.pth \
    --known_split ${IND_DATA} \
    --unknown_split ${OOD_DATA} \
    --result_file ${RESULT_PATH}/tSNE_enn_avuc_HMDB.png

CUDA_VISIBLE_DEVICES=${DEVICE} python experiments/analyze_features.py \
    --config configs/recognition/tpn/inference_tpn_slowonly_enn.py \
    --checkpoint work_dirs/tpn_slowonly/finetune_ucf101_tpn_slowonly_edlloss_nokl_avuc_debias/latest.pth \
    --known_split ${IND_DATA} \
    --unknown_split ${OOD_DATA} \
    --result_file ${RESULT_PATH}/tSNE_enn_avuc_debias_HMDB.png

cd $pwd_dir
echo "Experiments finished!"