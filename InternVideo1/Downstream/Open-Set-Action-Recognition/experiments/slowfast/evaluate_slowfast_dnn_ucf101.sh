#!/bin/bash

pwd_dir=$pwd
cd ../../

source activate mmaction

CUDA_VISIBLE_DEVICES=$1 python tools/test.py configs/recognition/slowfast/finetune_ucf101_slowfast_dnn.py \
	work_dirs/slowfast/finetune_ucf101_slowfast_dnn/latest.pth \
	--out work_dirs/slowfast/test_ucf101_slowfast_dnn.pkl \
	--eval top_k_accuracy mean_class_accuracy

cd $pwd_dir
echo "Experiments finished!"
