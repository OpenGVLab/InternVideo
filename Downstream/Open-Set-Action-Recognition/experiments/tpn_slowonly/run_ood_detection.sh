#!/bin/bash

export CUDA_HOME='/usr/local/cuda'

pwd_dir=$pwd
cd ../../

source activate mmaction

DEVICE=$1
OOD_DATASET=$2
MODEL=$3
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
RESULT_DIR='experiments/tpn_slowonly/results'

case ${MODEL} in
    dropout)
    # DNN with Dropout model
    CUDA_VISIBLE_DEVICES=${DEVICE} python experiments/ood_detection.py \
        --config configs/recognition/tpn/inference_tpn_slowonly_dnn.py \
        --checkpoint work_dirs/tpn_slowonly/finetune_ucf101_tpn_slowonly_celoss/latest.pth \
        --ind_data ${IND_DATA} \
        --ood_data ${OOD_DATA} \
        --uncertainty BALD \
        --result_prefix ${RESULT_DIR}/TPN_SlowOnly_Dropout_BALD_${OOD_DATASET}
    ;;
    bnn)
    # Bayesian Neural Network
    CUDA_VISIBLE_DEVICES=${DEVICE} python experiments/ood_detection.py \
        --config configs/recognition/tpn/inference_tpn_slowonly_bnn.py \
        --checkpoint work_dirs/tpn_slowonly/finetune_ucf101_tpn_slowonly_bnn/latest.pth \
        --ind_data ${IND_DATA} \
        --ood_data ${OOD_DATA} \
        --uncertainty BALD \
        --result_prefix ${RESULT_DIR}/TPN_SlowOnly_BNN_BALD_${OOD_DATASET}
    ;;
    edl)
    # Evidential Deep Learning
    CUDA_VISIBLE_DEVICES=${DEVICE} python experiments/ood_detection.py \
        --config configs/recognition/tpn/inference_tpn_slowonly_enn.py \
        --checkpoint work_dirs/tpn_slowonly/finetune_ucf101_tpn_slowonly_edlloss/latest.pth \
        --ind_data ${IND_DATA} \
        --ood_data ${OOD_DATA} \
        --uncertainty EDL \
        --result_prefix ${RESULT_DIR}/TPN_SlowOnly_EDLlog_EDL_${OOD_DATASET}
    ;;
    edl_nokl)
    # Evidential Deep Learning
    CUDA_VISIBLE_DEVICES=${DEVICE} python experiments/ood_detection.py \
        --config configs/recognition/tpn/inference_tpn_slowonly_enn.py \
        --checkpoint work_dirs/tpn_slowonly/finetune_ucf101_tpn_slowonly_edlloss_nokl/latest.pth \
        --ind_data ${IND_DATA} \
        --ood_data ${OOD_DATA} \
        --uncertainty EDL \
        --result_prefix ${RESULT_DIR}/TPN_SlowOnly_EDLlogNoKL_EDL_${OOD_DATASET}
    ;;
    edl_avuc)
    # Evidential Deep Learning with AvU Calibration
    CUDA_VISIBLE_DEVICES=${DEVICE} python experiments/ood_detection.py \
        --config configs/recognition/tpn/inference_tpn_slowonly_enn.py \
        --checkpoint work_dirs/tpn_slowonly/finetune_ucf101_tpn_slowonly_edlloss_avuc/latest.pth \
        --ind_data ${IND_DATA} \
        --ood_data ${OOD_DATA} \
        --uncertainty EDL \
        --result_prefix ${RESULT_DIR}/TPN_SlowOnly_EDLlogAvUC_EDL_${OOD_DATASET}
    ;;
    edl_nokl_avuc)
    # Evidential Deep Learning (without KL divergence loss term) with AvU Calibration
    CUDA_VISIBLE_DEVICES=${DEVICE} python experiments/ood_detection.py \
        --config configs/recognition/tpn/inference_tpn_slowonly_enn.py \
        --checkpoint work_dirs/tpn_slowonly/finetune_ucf101_tpn_slowonly_edlloss_nokl_avuc/latest.pth \
        --ind_data ${IND_DATA} \
        --ood_data ${OOD_DATA} \
        --uncertainty EDL \
        --result_prefix ${RESULT_DIR}/TPN_SlowOnly_EDLlogNoKLAvUC_EDL_${OOD_DATASET}
    ;;
    edl_nokl_avuc_debias)
    # Evidential Deep Learning (without KL divergence loss term) with AvU Calibration
    CUDA_VISIBLE_DEVICES=${DEVICE} python experiments/ood_detection.py \
        --config configs/recognition/tpn/inference_tpn_slowonly_enn.py \
        --checkpoint work_dirs/tpn_slowonly/finetune_ucf101_tpn_slowonly_edlloss_nokl_avuc_debias/latest.pth \
        --ind_data ${IND_DATA} \
        --ood_data ${OOD_DATA} \
        --uncertainty EDL \
        --result_prefix ${RESULT_DIR}/TPN_SlowOnly_EDLlogNoKLAvUCDebias_EDL_${OOD_DATASET}
    ;;
    *)
    echo "Invalid model: "${MODEL}
    exit
    ;;
esac


cd $pwd_dir
echo "Experiments finished!"