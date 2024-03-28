#!/bin/bash

export CUDA_HOME='/usr/local/cuda'

pwd_dir=$pwd
cd ../../

source activate mmaction

OOD_DATA=$1  # HMDB or MiT
RESULT_PATH="experiments/i3d/results"

# Confusion Matrix comparison
python experiments/draw_confusion_matrix.py \
    --ood_result ${RESULT_PATH}/I3D_EDLNoKLAvUCDebias_EDL_${OOD_DATA}_result.npz \
    --uncertain_thresh 0.004550 \
    --save_file ${RESULT_PATH}/../results_confmat/I3D_DEAR_${OOD_DATA}_ConfMat.png

cd $pwd_dir
echo "Experiments finished!"