#!/bin/bash

pwd_dir=$pwd
cd ../../

source activate mmaction

CUDA_VISIBLE_DEVICES=$1 python tools/test.py configs/recognition/tsm/finetune_ucf101_tsm_bnn.py \
	work_dirs/tsm/finetune_ucf101_tsm_bnn/latest.pth \
	--out work_dirs/tsm/test_ucf101_tsm_bnn.pkl \
	--eval top_k_accuracy mean_class_accuracy

cd $pwd_dir
echo "Experiments finished!"
