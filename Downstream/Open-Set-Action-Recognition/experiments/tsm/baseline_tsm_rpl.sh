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

CUDA_VISIBLE_DEVICES=${DEVICE} python experiments/baseline_rpl.py \
    --config configs/recognition/tsm/inference_tsm_rpl.py \
    --checkpoint work_dirs/tsm/finetune_ucf101_tsm_rpl/latest.pth \
    --train_data data/ucf101/ucf101_train_split_1_videos.txt \
    --ind_data ${IND_DATA} \
    --ood_data ${OOD_DATA} \
    --ood_ncls ${NUM_CLASSES} \
    --ood_dataname ${OOD_DATASET} \
    --result_prefix experiments/tsm/results_baselines/rpl/RPL

cd $pwd_dir
echo "Experiments finished!"