#!/bin/bash

pwd_dir=$pwd
cd ../../

source activate mmaction

CUDA_VISIBLE_DEVICES=$1 python tools/train.py configs/recognition/i3d/finetune_ucf101_i3d_dnn.py \
	--work-dir work_dirs/i3d/finetune_ucf101_i3d_dnn \
	--validate \
	--seed 0 \
	--deterministic \
	--gpu-ids 0

cd $pwd_dir
echo "Experiments finished!"
