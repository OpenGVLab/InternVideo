#!/bin/bash

pwd_dir=$pwd
cd ../../

source activate mmaction

# --validate
CUDA_VISIBLE_DEVICES=$1 python tools/train.py configs/recognition/tsm/finetune_ucf101_tsm_dnn.py \
	--work-dir work_dirs/tsm/finetune_ucf101_tsm_dnn \
	--seed 0 \
	--deterministic \
	--gpu-ids 0 \
	--validate

cd $pwd_dir
echo "Experiments finished!"
