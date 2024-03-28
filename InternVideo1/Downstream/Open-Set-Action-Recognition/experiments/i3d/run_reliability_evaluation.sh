#!/bin/bash

export CUDA_HOME='/usr/local/cuda'

pwd_dir=$pwd
cd ../../

source activate mmaction

OOD_DATASET=$1
MODEL=$2
RESULT_DIR='experiments/i3d/results'

case ${MODEL} in
    dnn)
    # DNN with Dropout model
    CUDA_VISIBLE_DEVICES=${DEVICE} python experiments/evaluate_calibration.py \
        --ood_result ${RESULT_DIR}/I3D_DNN_BALD_${OOD_DATASET}_result.npz \
        --save_prefix ${RESULT_DIR}/../results_reliability/I3D_DNN_BALD_${OOD_DATASET}_reliability
    ;;
    bnn)
    # Evidential Deep Learning (without KL divergence loss term)
    CUDA_VISIBLE_DEVICES=${DEVICE} python experiments/evaluate_calibration.py \
        --ood_result ${RESULT_DIR}/I3D_BNN_BALD_${OOD_DATASET}_result.npz \
        --save_prefix ${RESULT_DIR}/../results_reliability/I3D_BNN_BALD_${OOD_DATASET}_reliability
    ;;
    edlnokl)
    # Evidential Deep Learning (without KL divergence loss term)
    CUDA_VISIBLE_DEVICES=${DEVICE} python experiments/evaluate_calibration.py \
        --ood_result ${RESULT_DIR}/I3D_EDLNoKL_EDL_${OOD_DATASET}_result.npz \
        --save_prefix ${RESULT_DIR}/../results_reliability/I3D_EDLNoKL_EDL_${OOD_DATASET}_reliability
    ;;
    edlnokl_avuc_debias)
    # Evidential Deep Learning (without KL divergence loss term) with AvU Calibration and Debiasing
    CUDA_VISIBLE_DEVICES=${DEVICE} python experiments/evaluate_calibration.py \
        --ood_result ${RESULT_DIR}/I3D_EDLNoKLAvUCCED_EDL_${OOD_DATASET}_result.npz \
        --save_prefix ${RESULT_DIR}/../results_reliability/I3D_EDLNoKLAvUCCED_EDL_${OOD_DATASET}_reliability
    ;;
    *)
    echo "Invalid model: "${MODEL}
    exit
    ;;
esac


cd $pwd_dir
echo "Experiments finished!"