#!/bin/bash

pwd_dir=$pwd
cd ../../

source activate mmaction

CUDA_VISIBLE_DEVICES=$1 python tools/train.py configs/recognition/slowfast/finetune_ucf101_slowfast_bnn.py \
	--work-dir work_dirs/slowfast/finetune_ucf101_slowfast_bnn \
	--validate \
	--seed 0 \
	--deterministic \
	--gpu-ids 0

cd $pwd_dir
echo "Experiments finished!"
