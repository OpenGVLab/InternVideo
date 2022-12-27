#!/bin/bash

export CUDA_HOME='/usr/local/cuda'

pwd_dir=$pwd
cd ../../

source activate mmaction

OOD_DATA=$1  # HMDB or MiT
RESULT_FILES=("results_baselines/openmax/TPN_OpenMax_${OOD_DATA}_result.npz" \          # openmax
              "results/TPN_SlowOnly_Dropout_BALD_${OOD_DATA}_result.npz" \              # mc dropout
              "results/TPN_SlowOnly_BNN_BALD_${OOD_DATA}_result.npz" \                  # bnn svi
              "results_baselines/openmax/TPN_OpenMax_${OOD_DATA}_result.npz" \          # softmax
              "results_baselines/rpl/TPN_RPL_${OOD_DATA}_result.npz" \                  # rpl
              "results/TPN_SlowOnly_EDLlogNoKLAvUCDebias_EDL_${OOD_DATA}_result.npz")   # dear (ours)
THRESHOLDS=(-1 \
            0.000096 \
            0.000007 \
            0.997623 \
            0.996931 \
            0.004555)

# OOD Detection comparison
# The folders `results/` and `results_baselines` are in the `experiments/tpn_slowonly/` folder.
python experiments/compare_openness_new.py \
    --base_model tpn_slowonly \
    --ood_data ${OOD_DATA} \
    --baseline_results ${RESULT_FILES[@]}

# OOD Detection comparison by using thresholds
echo 'Results by using thresholds:'
python experiments/compare_openness_new.py \
    --base_model tpn_slowonly \
    --ood_data ${OOD_DATA} \
    --thresholds ${THRESHOLDS[@]} \
    --baseline_results ${RESULT_FILES[@]}

cd $pwd_dir
echo "Experiments finished!"