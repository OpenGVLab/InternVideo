#!/bin/bash

pwd_dir=$pwd
cd ../../

source activate mmaction

# --validate
CUDA_VISIBLE_DEVICES=$1 python tools/train.py configs/recognition/tpn/tpn_slowonly_celoss_r50_8x8x1_150e_kinetics_rgb.py \
	--work-dir work_dirs/tpn_slowonly/finetune_ucf101_tpn_slowonly_celoss \
	--seed 0 \
	--deterministic \
	--gpu-ids 0 \
	--validate

cd $pwd_dir
echo "Experiments finished!"
