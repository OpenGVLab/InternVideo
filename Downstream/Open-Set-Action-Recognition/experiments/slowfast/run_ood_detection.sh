#!/bin/bash

export CUDA_HOME='/usr/local/cuda'

pwd_dir=$pwd
cd ../../
# DEVICE=$1
OOD_DATASET=$1
MODEL=$2
GPUS=$3
PORT=${PORT:-29500}
IND_DATA='data/ucf101/ucf101_val_split_1_videos.txt'

case ${OOD_DATASET} in
  HMDB)
    # run ood detection on hmdb-51 validation set
    OOD_DATA='data/hmdb51/hmdb51_val_split_1_videos.txt'
    ;;
  MiT)
    # run ood detection on hmdb-51 validation set
    OOD_DATA='data/mit/mit_val_list_videos.txt'
    ;;
  *)
    echo "Dataset not supported: "${OOD_DATASET}
    exit
    ;;
esac
RESULT_DIR='experiments/slowfast/results'

case ${MODEL} in
    dnn)
    # DNN with Dropout model
    CUDA_VISIBLE_DEVICES=${DEVICE} python experiments/ood_detection.py \
        --config configs/recognition/slowfast/inference_slowfast_dnn.py \
        --checkpoint work_dirs/slowfast/finetune_ucf101_slowfast_dnn/latest.pth \
        --ind_data ${IND_DATA} \
        --ood_data ${OOD_DATA} \
        --uncertainty BALD \
        --result_prefix ${RESULT_DIR}/SlowFast_DNN_BALD_${OOD_DATASET}
    ;;
    bnn)
    # DNN with Dropout model
    CUDA_VISIBLE_DEVICES=${DEVICE} python experiments/ood_detection.py \
        --config configs/recognition/slowfast/inference_slowfast_bnn.py \
        --checkpoint work_dirs/slowfast/finetune_ucf101_slowfast_bnn/latest.pth \
        --ind_data ${IND_DATA} \
        --ood_data ${OOD_DATA} \
        --uncertainty BALD \
        --result_prefix ${RESULT_DIR}/SlowFast_BNN_BALD_${OOD_DATASET}
    ;;
    edlnokl_avuc_debias)
    # Evidential Deep Learning (without KL divergence loss term) with AvU Calibration and Debiasing
    # CUDA_VISIBLE_DEVICES=${DEVICE} python experiments/ood_detection.py \
    python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT experiments/ood_detection_dist.py \
        --config configs/recognition/slowfast/inference_slowfast_enn.py \
        --checkpoint work_dirs/slowfast/finetune_ucf101_slowfast_edlnokl_avuc_debias/latest.pth \
        --ind_data ${IND_DATA} \
        --ood_data ${OOD_DATA} \
        --uncertainty EDL \
        --result_prefix ${RESULT_DIR}/SlowFast_EDLNoKLAvUCDebias_EDL_${OOD_DATASET} \
        --launcher pytorch
    ;;
    *)
    echo "Invalid model: "${MODEL}
    exit
    ;;
esac


cd $pwd_dir
echo "Experiments finished!"