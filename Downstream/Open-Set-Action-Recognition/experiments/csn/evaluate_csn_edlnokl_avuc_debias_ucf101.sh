#!/bin/bash

pwd_dir=$pwd
cd ../../

source activate mmaction

CUDA_VISIBLE_DEVICES=$1 python tools/test.py configs/recognition/csn/finetune_ucf101_csn_edlnokl_avuc_debias.py \
	work_dirs/csn/finetune_ucf101_csn_edlnokl_avuc_debias/latest.pth \
	--videos_per_gpu 1 \
	--out work_dirs/csn/test_ucf101_csn_edlnokl_avuc_debias.pkl \
	--eval top_k_accuracy mean_class_accuracy

cd $pwd_dir
echo "Experiments finished!"
