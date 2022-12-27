#!/bin/bash

export CUDA_HOME='/usr/local/cuda'

pwd_dir=$pwd
cd ../../

source activate mmaction

OOD_DATA=$1  # HMDB or MiT
RESULT_FILES=("results_baselines/openmax/I3D_OpenMax_${OOD_DATA}_result.npz" \  # openmax
              "results/I3D_DNN_BALD_${OOD_DATA}_result.npz" \                   # mc dropout
              "results/I3D_BNN_BALD_${OOD_DATA}_result.npz" \                   # bnn svi
              "results_baselines/openmax/I3D_OpenMax_${OOD_DATA}_result.npz" \  # softmax
              "results_baselines/rpl/I3D_RPL_${OOD_DATA}_result.npz" \          # rpl
              "results/I3D_EDLNoKLAvUCDebias_EDL_${OOD_DATA}_result.npz")       # dear (ours)
THRESHOLDS=(-1 \
            0.000433 \
            0.000004 \
            0.996825 \
            0.995178 \
            0.004550)

# OOD Detection comparison
# The folders `results/` and `results_baselines` are in the `experiments/i3d/` folder.
python experiments/compare_openness_new.py \
    --base_model i3d \
    --ood_data ${OOD_DATA} \
    --baseline_results ${RESULT_FILES[@]}

# OOD Detection comparison by using thresholds
echo 'Results by using thresholds:'
python experiments/compare_openness_new.py \
    --base_model i3d \
    --ood_data ${OOD_DATA} \
    --thresholds ${THRESHOLDS[@]} \
    --baseline_results ${RESULT_FILES[@]}

cd $pwd_dir
echo "Experiments finished!"