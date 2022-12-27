#!/bin/bash

pwd_dir=$pwd
cd ../../

source activate mmaction

CUDA_VISIBLE_DEVICES=$1 python tools/train.py configs/recognition/tsm/finetune_ucf101_tsm_bnn.py \
	--work-dir work_dirs/tsm/finetune_ucf101_tsm_bnn \
	--validate \
	--seed 0 \
	--deterministic \
	--gpu-ids 0

cd $pwd_dir
echo "Experiments finished!"
