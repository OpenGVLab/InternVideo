#!/bin/bash

export CUDA_HOME='/usr/local/cuda'

pwd_dir=$pwd
cd ../../

source activate mmaction

OOD_DATA=$1  # HMDB or MiT
RESULT_PATH="experiments/tpn_slowonly/results"

# Confusion Matrix comparison
python experiments/draw_confusion_matrix.py \
    --ood_result ${RESULT_PATH}/TPN_SlowOnly_EDLlogNoKLAvUCDebias_EDL_${OOD_DATA}_result.npz \
    --uncertain_thresh 0.004555 \
    --top_part \
    --save_file ${RESULT_PATH}/../results_confmat/TPN_DEAR_${OOD_DATA}_ConfMat.png

# python experiments/draw_confusion_matrix.py \
#     --ood_result ${RESULT_PATH}/TPN_SlowOnly_EDLlogNoKLAvUC_EDL_${OOD_DATA}_result.npz \
#     --uncertain_thresh 0.495800 \
#     --save_file ${RESULT_PATH}/../results_confmat/confmat_EDLlogNoKLAvUC_EDL_${OOD_DATA}.png

# python experiments/draw_confusion_matrix.py \
#     --ood_result ${RESULT_PATH}/TPN_SlowOnly_EDLlogNoKL_EDL_${OOD_DATA}_result.npz \
#     --uncertain_thresh 0.495806 \
#     --save_file ${RESULT_PATH}/../results_confmat/confmat_EDLlogNoKL_EDL_${OOD_DATA}.png

# python experiments/draw_confusion_matrix.py \
#     --ood_result ${RESULT_PATH}/TPN_SlowOnly_Dropout_BALD_${OOD_DATA}_result.npz \
#     --uncertain_thresh 0.000096 \
#     --save_file ${RESULT_PATH}/../results_confmat/confmat_Dropout_BALD_${OOD_DATA}.png

# python experiments/draw_confusion_matrix.py \
#     --ood_result ${RESULT_PATH}/TPN_SlowOnly_BNN_BALD_${OOD_DATA}_result.npz \
#     --uncertain_thresh 0.000007 \
#     --save_file ${RESULT_PATH}/../results_confmat/confmat_BNN_BALD_${OOD_DATA}.png

cd $pwd_dir
echo "Experiments finished!"