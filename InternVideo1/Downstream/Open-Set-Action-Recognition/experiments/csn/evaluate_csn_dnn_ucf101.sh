#!/bin/bash

pwd_dir=$pwd
cd ../../

source activate mmaction

CUDA_VISIBLE_DEVICES=$1 python tools/test.py configs/recognition/csn/finetune_ucf101_csn_dnn.py \
	work_dirs/csn/finetune_ucf101_csn_dnn/latest.pth \
	--videos_per_gpu 1 \
	--out work_dirs/csn/test_ucf101_csn_dnn.pkl \
	--eval top_k_accuracy mean_class_accuracy

cd $pwd_dir
echo "Experiments finished!"
