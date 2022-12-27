#!/bin/bash

pwd_dir=$pwd
cd ../../

source activate mmaction

# --validate
CUDA_VISIBLE_DEVICES=$1 python tools/train.py configs/recognition/slowfast/finetune_ucf101_slowfast_edlnokl_avuc_debias.py \
	--work-dir work_dirs/slowfast/finetune_ucf101_slowfast_edlnokl_avuc_debias \
	--seed 0 \
	--deterministic \
	--gpu-ids 0 \
	--validate

cd $pwd_dir
echo "Experiments finished!"
