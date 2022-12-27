#!/bin/bash

export CUDA_HOME='/usr/local/cuda'

pwd_dir=$pwd
cd ../../

source activate mmaction

OOD_DATA=$1  # HMDB or MiT
RESULT_FILES=("results_baselines/openmax/SlowFast_OpenMax_${OOD_DATA}_result.npz" \  # openmax
              "results/SlowFast_DNN_BALD_${OOD_DATA}_result.npz" \                   # mc dropout
              "results/SlowFast_BNN_BALD_${OOD_DATA}_result.npz" \                   # bnn svi
              "results_baselines/openmax/SlowFast_OpenMax_${OOD_DATA}_result.npz" \  # softmax
              "results_baselines/rpl/SlowFast_RPL_${OOD_DATA}_result.npz" \          # rpl
              "results/SlowFast_EDLNoKLAvUCDebias_EDL_${OOD_DATA}_result.npz")       # dear (ours)
THRESHOLDS=(-1 \
            0.000065 \
            0.000010 \
            0.997915 \
            0.997780 \
            0.004552)

# OOD Detection comparison
# The folders `results/` and `results_baselines` are in the `experiments/slowfast/` folder.
python experiments/compare_openness_new.py \
    --base_model slowfast \
    --ood_data ${OOD_DATA} \
    --baseline_results ${RESULT_FILES[@]}

# OOD Detection comparison by using thresholds
echo 'Results by using thresholds:'
python experiments/compare_openness_new.py \
    --base_model slowfast \
    --ood_data ${OOD_DATA} \
    --thresholds ${THRESHOLDS[@]} \
    --baseline_results ${RESULT_FILES[@]}

cd $pwd_dir
echo "Experiments finished!"