#!/bin/bash

pwd_dir=$pwd
cd ../../

source activate mmaction

# --validate
CUDA_VISIBLE_DEVICES=$1 python tools/train.py configs/recognition/tsm/train_kinetics10_tsm_DEAR_noDebias.py \
	--work-dir work_dirs/tsm/train_kinetics10_tsm_DEAR_noDebias \
	--validate \
	--seed 0 \
	--deterministic \
	--gpu-ids 0

cd $pwd_dir
echo "Experiments finished!"
