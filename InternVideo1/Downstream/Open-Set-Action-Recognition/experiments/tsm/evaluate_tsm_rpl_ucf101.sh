#!/bin/bash

pwd_dir=$pwd
cd ../../

source activate mmaction

CUDA_VISIBLE_DEVICES=$1 python tools/test.py configs/recognition/tsm/finetune_ucf101_tsm_rpl.py \
	work_dirs/tsm/finetune_ucf101_tsm_rpl/latest.pth \
	--out work_dirs/tsm/test_ucf101_tsm_rpl.pkl \
	--eval top_k_accuracy mean_class_accuracy

cd $pwd_dir
echo "Experiments finished!"
