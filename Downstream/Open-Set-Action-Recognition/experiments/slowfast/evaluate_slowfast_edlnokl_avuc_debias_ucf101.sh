#!/bin/bash

pwd_dir=$pwd
cd ../../

source activate mmaction

CUDA_VISIBLE_DEVICES=$1 python tools/test.py configs/recognition/slowfast/finetune_ucf101_slowfast_edlnokl_avuc_debias.py \
	work_dirs/slowfast/finetune_ucf101_slowfast_edlnokl_avuc_debias/latest.pth \
	--out work_dirs/slowfast/test_ucf101_slowfast_edlnokl_avuc_debias.pkl \
	--eval top_k_accuracy mean_class_accuracy

cd $pwd_dir
echo "Experiments finished!"
