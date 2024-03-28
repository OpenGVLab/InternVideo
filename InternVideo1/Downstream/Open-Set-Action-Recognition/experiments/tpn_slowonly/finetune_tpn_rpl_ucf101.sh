#!/bin/bash

pwd_dir=$pwd
cd ../../

source activate mmaction

# --validate
CUDA_VISIBLE_DEVICES=$1 python tools/train.py configs/recognition/tpn/finetune_ucf101_tpn_slowonly_rpl.py \
	--work-dir work_dirs/tpn_slowonly/finetune_ucf101_tpn_slowonly_rpl \
	--seed 0 \
	--deterministic \
	--gpu-ids 0 \
	--validate

cd $pwd_dir
echo "Experiments finished!"
