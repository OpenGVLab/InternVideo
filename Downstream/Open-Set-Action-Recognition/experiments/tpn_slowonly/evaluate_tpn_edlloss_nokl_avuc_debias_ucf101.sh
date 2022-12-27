#!/bin/bash

pwd_dir=$pwd
cd ../../

source activate mmaction

CUDA_VISIBLE_DEVICES=$1 python tools/test.py configs/recognition/tpn/tpn_slowonly_edlloss_nokl_avuc_debias_r50_8x8x1_150e_kinetics_rgb.py \
	work_dirs/tpn_slowonly/finetune_ucf101_tpn_slowonly_edlloss_nokl_avuc_debias/latest.pth \
	--out work_dirs/tpn_slowonly/test_ucf101_tpn_slowonly_edlloss_nokl_avuc_debias.pkl \
	--eval top_k_accuracy mean_class_accuracy

cd $pwd_dir
echo "Experiments finished!"
