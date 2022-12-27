#!/bin/bash

export CUDA_HOME='/usr/local/cuda'

pwd_dir=$pwd
cd ../../

source activate mmaction

OOD_DATASET=$1
MODEL=$2
RESULT_DIR='experiments/tpn_slowonly/results'

case ${MODEL} in
    dnn)
    # DNN with Dropout model
    CUDA_VISIBLE_DEVICES=${DEVICE} python experiments/evaluate_calibration.py \
        --ood_result ${RESULT_DIR}/TPN_SlowOnly_Dropout_BALD_${OOD_DATASET}_result.npz \
        --threshold 0.000096 \
        --save_prefix ${RESULT_DIR}/../results_reliability/TPN_SlowOnly_Dropout_BALD_${OOD_DATASET}_reliability
    ;;
    bnn)
    # BNN model
    CUDA_VISIBLE_DEVICES=${DEVICE} python experiments/evaluate_calibration.py \
        --ood_result ${RESULT_DIR}/TPN_SlowOnly_BNN_BALD_${OOD_DATASET}_result.npz \
        --threshold 0.000007 \
        --save_prefix ${RESULT_DIR}/../results_reliability/TPN_SlowOnly_BNN_BALD_${OOD_DATASET}_reliability
    ;;
    edlnokl)
    # Evidential Deep Learning (without KL divergence loss term)
    CUDA_VISIBLE_DEVICES=${DEVICE} python experiments/evaluate_calibration.py \
        --ood_result ${RESULT_DIR}/TPN_SlowOnly_EDLlogNoKL_EDL_${OOD_DATASET}_result.npz \
        --threshold 0.495806 \
        --save_prefix ${RESULT_DIR}/../results_reliability/TPN_SlowOnly_EDLlogNoKL_EDL_${OOD_DATASET}_reliability
    ;;
    edlnokl_avuc)
    # Evidential Deep Learning (without KL divergence loss term) with AvU Calibration and Debiasing
    CUDA_VISIBLE_DEVICES=${DEVICE} python experiments/evaluate_calibration.py \
        --ood_result ${RESULT_DIR}/TPN_SlowOnly_EDLlogNoKLAvUC_EDL_${OOD_DATASET}_result.npz \
        --threshold 0.495800 \
        --save_prefix ${RESULT_DIR}/../results_reliability/TPN_SlowOnly_EDLlogNoKLAvUC_EDL_${OOD_DATASET}_reliability
    ;;
    *)
    echo "Invalid model: "${MODEL}
    exit
    ;;
esac


cd $pwd_dir
echo "Experiments finished!"