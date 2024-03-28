#!/bin/bash

export CUDA_HOME='/usr/local/cuda'

pwd_dir=$pwd
cd ../../

source activate mmaction

DEVICE=$1

CUDA_VISIBLE_DEVICES=${DEVICE} python demo/demo_gradcam.py configs/recognition/tpn/inference_tpn_slowonly_dnn.py \
    work_dirs/tpn_slowonly/finetune_ucf101_tpn_slowonly_edlloss_nokl_avuc/latest.pth demo/demo.mp4 \
    --target-layer-name backbone/layer4/1/relu --fps 10 \
    --out-filename experiments/tpn_slowonly/results/demo_gradcam.gif



cd $pwd_dir
echo "Experiments finished!"