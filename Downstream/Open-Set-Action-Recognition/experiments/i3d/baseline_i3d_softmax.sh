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
RESULT_DIR='experiments/i3d/results'

CUDA_VISIBLE_DEVICES=${DEVICE} python experiments/baseline_softmax.py \
    --config configs/recognition/i3d/inference_i3d_dnn.py \
    --checkpoint work_dirs/i3d/finetune_ucf101_i3d_dnn/latest.pth \
    --batch_size 4 \
    --ood_ncls ${NUM_CLASSES} \
    --result_prefix experiments/i3d/results_baselines/softmax/I3D_SoftMax_${OOD_DATASET}

cd $pwd_dir
echo "Experiments finished!"