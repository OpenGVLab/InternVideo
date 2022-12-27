#!/bin/bash

pwd_dir=$pwd
cd ../../

source activate mmaction

CUDA_VISIBLE_DEVICES=$1 python tools/test.py configs/recognition/i3d/finetune_ucf101_i3d_edlnokl_avuc_debias.py \
	work_dirs/i3d/finetune_ucf101_i3d_edlnokl_avuc_debias/latest.pth \
	--out work_dirs/i3d/test_ucf101_i3d_edlnokl_avuc_debias.pkl \
	--eval top_k_accuracy mean_class_accuracy

cd $pwd_dir
echo "Experiments finished!"
