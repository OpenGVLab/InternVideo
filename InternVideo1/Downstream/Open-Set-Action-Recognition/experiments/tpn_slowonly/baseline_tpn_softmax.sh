#!/bin/bash

pwd_dir=$pwd
cd ../../

source activate mmaction

DEVICE=$1
OOD_DATASET=$2

case ${OOD_DATASET} in
  HMDB)
    NUM_CLASSES=51
    ;;
  MiT)
    NUM_CLASSES=305
    ;;
  *)
    echo "Dataset not supported: "${OOD_DATASET}
    exit
    ;;
esac
RESULT_DIR='experiments/tpn_slowonly/results'

CUDA_VISIBLE_DEVICES=${DEVICE} python experiments/baseline_softmax.py \
    --config configs/recognition/tpn/inference_tpn_slowonly_dnn.py \
    --checkpoint work_dirs/tpn_slowonly/finetune_ucf101_tpn_slowonly_celoss/latest.pth \
    --batch_size 1 \
    --ood_ncls ${NUM_CLASSES} \
    --result_prefix experiments/tpn_slowonly/results_baselines/softmax/TPN_SoftMax_${OOD_DATASET}

cd $pwd_dir
echo "Experiments finished!"