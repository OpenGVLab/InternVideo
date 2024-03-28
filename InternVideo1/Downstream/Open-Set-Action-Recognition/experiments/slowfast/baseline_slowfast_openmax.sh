#!/bin/bash

pwd_dir=$pwd
cd ../../

source activate mmaction

DEVICE=$1
OOD_DATASET=$2
IND_DATA='data/ucf101/ucf101_val_split_1_videos.txt'

case ${OOD_DATASET} in
  HMDB)
    # run ood detection on hmdb-51 validation set
    OOD_DATA='data/hmdb51/hmdb51_val_split_1_videos.txt'
    NUM_CLASSES=51
    ;;
  MiT)
    # run ood detection on mit-v2 validation set
    OOD_DATA='data/mit/mit_val_list_videos.txt'
    NUM_CLASSES=305
    ;;
  *)
    echo "Dataset not supported: "${OOD_DATASET}
    exit
    ;;
esac
RESULT_DIR='experiments/slowfast/results'

CUDA_VISIBLE_DEVICES=${DEVICE} python experiments/baseline_openmax.py \
    --config configs/recognition/slowfast/inference_slowfast_dnn.py \
    --checkpoint work_dirs/slowfast/finetune_ucf101_slowfast_dnn/latest.pth \
    --cache_mav_dist experiments/slowfast/results_baselines/openmax/ucf101_mav_dist \
    --ind_data ${IND_DATA} \
    --ood_data ${OOD_DATA} \
    --ood_ncls ${NUM_CLASSES} \
    --result_prefix experiments/slowfast/results_baselines/openmax/SlowFast_OpenMax_${OOD_DATASET}

cd $pwd_dir
echo "Experiments finished!"