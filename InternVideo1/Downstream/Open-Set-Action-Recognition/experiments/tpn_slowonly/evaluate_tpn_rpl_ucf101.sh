#!/bin/bash

pwd_dir=$pwd
cd ../../

source activate mmaction

CUDA_VISIBLE_DEVICES=$1 python tools/test.py configs/recognition/tpn/finetune_ucf101_tpn_slowonly_rpl.py \
	work_dirs/tpn_slowonly/finetune_ucf101_tpn_slowonly_rpl/latest.pth \
	--out work_dirs/tpn_slowonly/test_ucf101_tpn_slowonly_rpl.pkl \
	--eval top_k_accuracy mean_class_accuracy

cd $pwd_dir
echo "Experiments finished!"
